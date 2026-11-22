# -*- coding: utf-8 -*-
"""Tests for the further JEPA versions of TSJEPA."""

import contextlib
import io
import math
import os
import pickle
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.utils.data import generate_ts_data


def _torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False


NEW = ('vjepa', 'vjepa21', 'cjepa', 'sjepa', 'salt', 'vicreg', 'kerjepa',
       'lpjepa', 'lambdajepa', 'lenepa', 'lewm', 'cfjepa')
EMA = ('vjepa', 'vjepa21', 'cjepa', 'sjepa', 'cfjepa')
MASK_FREE = ('lenepa', 'lewm', 'cfjepa')
# ema_momentum_ after fit with ema_momentum=None
MOMENTUM = {'vjepa': 0.998, 'vjepa21': 0.99925, 'cjepa': 0.996,
            'sjepa': 0.996, 'cfjepa': 0.983}


@unittest.skipUnless(_torch_available(), "torch not installed")
class TestJEPARegularisers(unittest.TestCase):
    """The anti-collapse terms of the versions that do not use SIGReg."""

    def setUp(self):
        import torch
        self.torch = torch
        self.gen = torch.Generator().manual_seed(0)
        self.z = torch.randn(4, 2000, 16, generator=self.gen)

    def test_variance_loss(self):
        from pyod.models._jepa_variants import variance_loss
        assert variance_loss(self.z).item() < 0.03
        constant = self.torch.ones(50, 8)
        assert abs(variance_loss(constant).item() - (1 - 1e-2)) < 1e-5
        assert abs(variance_loss(0.5 * self.z).item() - 0.5) < 0.03
        # a 2-D input is one group
        assert abs(variance_loss(self.z[0]).item()
                   - variance_loss(self.z[:1]).item()) < 1e-6

    def test_covariance_loss(self):
        from pyod.models._jepa_variants import covariance_loss
        assert covariance_loss(self.z).item() < 0.02
        # one direction repeated in every dimension: all covariances are 1
        duplicated = self.z[..., :1].expand(-1, -1, 16)
        assert abs(covariance_loss(duplicated).item() - 15) < 0.5
        assert abs(covariance_loss(duplicated, pairwise=True).item()
                   - covariance_loss(duplicated).item() / 15) < 1e-5

    def test_cw_mmd_is_zero_for_the_prior_only(self):
        from pyod.models._jepa_variants import cw_mmd
        z = self.torch.randn(2, 1500, 32, generator=self.gen)
        assert abs(cw_mmd(z).item()) < 5e-3
        assert cw_mmd(z + 1.5).item() > 0.05
        assert cw_mmd(0.1 * z).item() > 0.05
        assert cw_mmd(3.0 * z).item() > 0.05
        assert abs(cw_mmd(2.0 * z, sigma=2.0).item()) < 5e-3
        assert cw_mmd(z[:, :1]).item() == 0.0  # one sample: no estimate

    def test_cw_mmd_matches_sliced_gaussian_mmd(self):
        from pyod.models._jepa_variants import cw_mmd
        torch = self.torch
        # brute force: the exact 1-D Gaussian-kernel MMD to N(0, 1),
        # averaged over many random directions
        n_samples, n_dims, gamma = 300, 48, 0.7
        z = 0.6 * torch.randn(n_samples, n_dims, generator=self.gen) + 0.3
        directions = torch.randn(n_dims, 4000, generator=self.gen)
        directions = directions / directions.norm(dim=0, keepdim=True)
        p = z @ directions  # (N, M)
        k = torch.exp(-gamma * (p.unsqueeze(0) - p.unsqueeze(1)).pow(2))
        within = (k.sum((0, 1)) - n_samples) / (n_samples * (n_samples - 1))
        s = 1 + 2 * gamma
        cross = torch.exp(-gamma * p.pow(2) / s).mean(0) / math.sqrt(s)
        expected = (within - 2 * cross + 1 / math.sqrt(1 + 4 * gamma)).mean()
        assert abs(cw_mmd(z, gamma=gamma).item() - expected.item()) < 0.01

    def test_rectified_target(self):
        from pyod.models._jepa_variants import rectified_gn_sample
        y = rectified_gn_sample((200000,), 1.0, 0.0, self.gen)
        assert y.min().item() >= 0.0
        assert abs((y == 0).float().mean().item() - 0.5) < 0.01
        # E relu(Laplace(0, b)) = b / 2 with b = 1 / sqrt(2)
        assert abs(y.mean().item() - 0.5 / math.sqrt(2)) < 0.01
        g = rectified_gn_sample((200000,), 2.0, 0.0, self.gen)
        assert abs(g.mean().item() - 1 / math.sqrt(2 * math.pi)) < 0.01
        shifted = rectified_gn_sample((50000,), 1.0, 5.0, self.gen)
        assert abs(shifted.var().item() - 1.0) < 0.05
        with self.assertRaises(ValueError):
            rectified_gn_sample((3,), 1.5)

    def test_rdm_reg(self):
        from pyod.models._jepa_variants import rdm_reg, rectified_gn_sample
        torch = self.torch
        match = rectified_gn_sample((3, 400, 16), 1.0, 0.0, self.gen)

        def seeded():
            return torch.Generator().manual_seed(5)

        value = rdm_reg(match, generator=seeded()).item()
        assert value < rdm_reg(torch.zeros(3, 400, 16),
                               generator=seeded()).item() / 5
        assert value < rdm_reg(match + 1.0, generator=seeded()).item() / 5
        assert value == rdm_reg(match, generator=seeded()).item()

    def test_sac_reg(self):
        from pyod.models._jepa_variants import sac_reg
        torch = self.torch
        z = torch.randn(6000, 16, generator=self.gen)
        assert sac_reg(z).item() < 0.01
        for scale in (0.3, 2.0):
            expected = 0.5 * (scale ** 2 - math.log(scale ** 2) - 1)
            assert abs(sac_reg(scale * z).item() - expected) < 0.02
        assert abs(sac_reg(z + 1.0).item() - 0.5) < 0.03
        # fewer samples than twice the dimension: random projection first
        small = sac_reg(torch.randn(20, 16, generator=self.gen),
                        generator=self.gen)
        assert torch.isfinite(small)
        assert sac_reg(z[:1]).item() == 0.0  # one sample: no estimate

    def test_gradients(self):
        from pyod.models._jepa_variants import (
            covariance_loss, cw_mmd, rdm_reg, sac_reg, variance_loss)
        torch = self.torch
        w = torch.randn(3, 64, 16, generator=self.gen, requires_grad=True)
        loss = (variance_loss(w) + covariance_loss(w) + cw_mmd(w)
                + rdm_reg(torch.relu(w), generator=self.gen)
                + sac_reg(w.reshape(-1, 16)))
        loss.backward()
        assert torch.isfinite(w.grad).all()
        assert w.grad.abs().sum().item() > 0


@unittest.skipUnless(_torch_available(), "torch not installed")
class TestTSJEPAFurtherVariants(unittest.TestCase):
    """The twelve versions implemented as recipes."""

    @classmethod
    def setUpClass(cls):
        from pyod.models.ts_jepa import TSJEPA
        cls.TSJEPA = TSJEPA
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = \
            generate_ts_data(n_train=1000, n_test=600, n_channels=3,
                             contamination=0.05, anomaly_type='both',
                             random_state=1)
        cls.params = dict(window_size=40, patch_len=5, epochs=2,
                          score='combined', random_state=3)
        cls.fitted = {
            variant: TSJEPA(variant=variant, **cls.params).fit(cls.X_train)
            for variant in NEW}

    def _fit(self, variant, X=None, **kwargs):
        params = dict(self.params, **kwargs)
        return self.TSJEPA(variant=variant, **params).fit(
            self.X_train if X is None else X)

    def test_registry(self):
        from pyod.models.ts_jepa import _BUILT_IN, _VARIANTS
        from pyod.models._jepa_variants import RECIPES
        # the names the detector lists without torch are the recipes
        assert _VARIANTS == _BUILT_IN + NEW
        assert tuple(RECIPES) == NEW
        assert len(set(_VARIANTS)) == 16
        for variant, recipe in RECIPES.items():
            assert recipe.__doc__, variant
            assert isinstance(recipe.params, dict), variant
            assert recipe.masked == (variant not in MASK_FREE), variant
            assert (recipe.ema_schedule is not None) == (variant in EMA)
            assert recipe.ema_momentum == MOMENTUM.get(variant), variant
            assert recipe.encoder_norm is False, variant

    def test_scores(self):
        for variant, clf in self.fitted.items():
            assert clf.decision_scores_.shape == (1000,), variant
            assert np.isfinite(clf.decision_scores_).all(), variant
            assert set(np.unique(clf.labels_)) <= {0, 1}, variant
            for score in ('pred', 'embed', 'combined'):
                clf.score = score
                scores = clf.decision_function(self.X_test)
                assert scores.shape == (600,), (variant, score)
                assert np.isfinite(scores).all(), (variant, score)
            clf.score = 'combined'
            scores = clf.decision_function(self.X_test)
            assert roc_auc_score(self.y_test, scores) >= 0.7, variant

    def test_variants_differ(self):
        seen = [self._fit('lejepa').decision_scores_]
        for variant in NEW:
            scores = self.fitted[variant].decision_scores_
            for other in seen:
                assert not np.allclose(scores, other), variant
            seen.append(scores)

    def test_resolved_defaults(self):
        from pyod.models._jepa_variants import RECIPES
        for variant, clf in self.fitted.items():
            assert clf.encoder_norm_ is False, variant
            assert clf.ema_momentum_ == MOMENTUM.get(variant), variant
            assert clf.variant_params_ == RECIPES[variant].params, variant
            # the constructor arguments are stored as given
            assert clf.variant_params is None
            assert clf.ema_momentum is None
        clf = self._fit('vjepa', ema_momentum=0.9, encoder_norm=True)
        assert (clf.encoder_norm_, clf.ema_momentum_) == (True, 0.9)
        assert np.isfinite(clf.decision_scores_).all()
        assert not np.allclose(clf.decision_scores_,
                               self.fitted['vjepa'].decision_scores_)

    def test_variant_params(self):
        from pyod.models._jepa_variants import RECIPES
        given = {'cov_weight': 5.0}
        before = dict(RECIPES['vicreg'].params)
        clf = self._fit('vicreg', variant_params=given)
        assert clf.variant_params_ == {'var_weight': 1.0, 'cov_weight': 5.0}
        # neither the argument nor the recipe is changed
        assert given == {'cov_weight': 5.0}
        assert RECIPES['vicreg'].params == before
        assert not np.allclose(clf.decision_scores_,
                               self.fitted['vicreg'].decision_scores_)
        # constants that change the network
        clf = self._fit('sjepa', variant_params={'latent_dim': 5}, epochs=1)
        assert clf.model_.latent_dim == 5
        clf = self._fit('cfjepa', epochs=1,
                        variant_params={'n_blocks': 2, 'n_crops': 2})
        assert len(clf.model_.encoder.blocks) == 2
        for variant, bad in (('vicreg', {'no_such': 1}),
                             ('lejepa', {'var_weight': 1.0}),
                             ('tsjepa', {'var_weight': 1.0}),
                             ('vicreg', [('cov_weight', 5.0)])):
            with self.assertRaises(ValueError):
                self._fit(variant, variant_params=bad)
        # an empty dict is the same as None, for every variant
        clf = self._fit('lejepa', variant_params={}, epochs=1)
        assert clf.variant_params_ == {}

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            self._fit('xjepa')
        for variant in ('vjepa', 'kerjepa', 'cfjepa'):
            with self.assertRaises(ValueError):
                # the reconstruction head exists for two variants only
                self._fit(variant, recon_weight=0.1)

    def test_verbose(self):
        for variant, lines in (('salt', ['Teacher epoch 1/1', 'Epoch 1/1']),
                               ('lewm', ['Epoch 1/2', 'Epoch 2/2'])):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self._fit(variant, verbose=1)
            printed = out.getvalue().splitlines()
            assert len(printed) == 2, variant
            for line, start in zip(printed, lines):
                assert line.startswith(start), (variant, line)

    def test_momentum_schedules(self):
        from pyod.models._jepa_utils import momentum_at
        assert momentum_at(0.9, 0.5, 'fixed') == 0.9
        assert abs(momentum_at(0.9, 0.5) - 0.95) < 1e-12
        assert abs(momentum_at(0.9, 0.5, 'linear') - 0.95) < 1e-12
        assert abs(momentum_at(0.9, 0.0, 'cosine') - 0.9) < 1e-12
        assert abs(momentum_at(0.9, 0.5, 'cosine') - 0.95) < 1e-12
        assert abs(momentum_at(0.9, 1.0, 'cosine') - 1.0) < 1e-12

    def test_zero_epochs_and_ema_target(self):
        import torch
        for variant in NEW:
            initial = self._fit(variant, epochs=0)
            torch.manual_seed(self.params['random_state'])
            fresh = initial._build_model(initial.n_tokens_,
                                         initial.n_channels_)
            for a, b in zip(initial.model_.parameters(),
                            fresh.parameters()):
                assert torch.equal(a, b), variant
            assert np.isfinite(initial.decision_scores_).all(), variant
            assert not initial.model_.training, variant

            model = self.fitted[variant].model_
            if variant not in EMA:
                assert getattr(model, 'target_encoder', None) is None
                continue
            target = model.target_encoder
            assert not any(p.requires_grad for p in target.parameters())
            moved = max((a - b).abs().max().item() for a, b in zip(
                target.parameters(),
                initial.model_.target_encoder.parameters()))
            lag = max((a - b).abs().max().item() for a, b in zip(
                target.parameters(), model.encoder.parameters()))
            assert moved > 0, variant  # the target was updated
            assert lag > 0, variant    # and is not the online encoder

    def test_variants_without_masks(self):
        n_patches, patch_len = 8, 5
        for variant in MASK_FREE:
            clf = self.fitted[variant]
            other = self._fit(variant, mask_mode='forecast', mask_ratio=0.6)
            assert_array_equal(other.decision_scores_, clf.decision_scores_)
            ctx, tgt = clf._eval_mask_set()
            first = 1 if variant != 'cfjepa' else clf.model_.n_ctx_tokens
            assert_array_equal(tgt.numpy(),
                               np.arange(first, n_patches)[None])
            # the leading patches of a series have no prediction score
            clf.score = 'pred'
            scores = clf.decision_function(self.X_test)
            assert_array_equal(scores[:first * patch_len],
                               np.full(first * patch_len, clf.threshold_))
            assert not np.any(scores[first * patch_len:] == clf.threshold_)
            clf.score = 'combined'
        assert self.fitted['cfjepa'].model_.n_ctx_tokens == 6

    def test_mask_modes(self):
        for variant in NEW:
            if variant in MASK_FREE:
                continue
            for mask_mode in ('forecast', 'random'):
                clf = self._fit(variant, mask_mode=mask_mode, epochs=1)
                scores = clf.decision_function(self.X_test)
                assert np.isfinite(scores).all(), (variant, mask_mode)

    def test_univariate_and_short_windows(self):
        X_train, X_test, _, _ = generate_ts_data(
            n_train=600, n_test=300, n_channels=1, contamination=0.05,
            anomaly_type='point', random_state=0)
        for variant in NEW:
            clf = self._fit(variant, X=X_train, epochs=1)
            assert clf.n_channels_ == 1
            scores = clf.decision_function(X_test)
            assert scores.shape == (300,), variant
            assert np.isfinite(scores).all(), variant
            # two patches of two timestamps
            tiny = self.TSJEPA(window_size=4, patch_len=2, epochs=1,
                               variant=variant, score='combined',
                               random_state=0).fit(X_train[:300])
            assert np.isfinite(tiny.decision_scores_).all(), variant

    def test_refit_clone_and_pickle(self):
        for variant, clf in self.fitted.items():
            cloned = clone(clf)
            assert cloned.get_params() == clf.get_params()
            assert cloned.variant == variant
            refit = cloned.fit(self.X_train)
            assert_allclose(refit.decision_scores_, clf.decision_scores_,
                            rtol=1e-4, atol=1e-5)
            restored = pickle.loads(pickle.dumps(clf))
            assert_allclose(restored.decision_function(self.X_test),
                            clf.decision_function(self.X_test))

    def test_salt_stages(self):
        import torch
        from pyod.models._jepa_variants import RECIPES
        salt = RECIPES['salt']
        for epochs, teacher, student in ((10, 2, 8), (3, 1, 2), (2, 1, 1),
                                         (1, 1, 1), (0, 0, 0)):
            det = self.TSJEPA(variant='salt', epochs=epochs)
            assert salt._teacher_epochs(det, salt.params) == teacher
            assert salt.train_epochs(det, salt.params) == student
        model = self.fitted['salt'].model_
        assert not any(p.requires_grad for p in model.teacher_parameters())
        assert all(p.grad is None for p in model.teacher_parameters())
        assert all(p.requires_grad for p in model.encoder.parameters())
        # one epoch of two goes to the teacher. A run with one epoch
        # has the same first stage, so its teacher must be the same:
        # the second stage does not touch the teacher.
        short = self._fit('salt', epochs=1).model_
        initial = self._fit('salt', epochs=0).model_
        assert not torch.equal(model.teacher.pos, initial.teacher.pos)
        for a, b in zip(short.teacher_parameters(),
                        model.teacher_parameters()):
            assert torch.equal(a, b)
        # the student is what is embedded
        x = torch.randn(4, 8, 15)
        _, h_all = model.multi_mask_errors(
            x, *self.fitted['salt']._eval_mask_set())
        with torch.no_grad():
            assert torch.equal(h_all, model.encoder(model.embed(x)))

    def test_next_latent_is_causal(self):
        import torch
        from pyod.models._jepa_variants import NextLatentCore
        torch.manual_seed(0)
        for style in ('lenepa', 'lewm'):
            net = NextLatentCore(6, 10, d_model=16, n_heads=2, n_layers=2,
                                 style=style).eval()
            x = torch.randn(3, 6, 10)
            y = x.clone()
            y[:, 4:] += 1.0  # change patches 4 and 5
            with torch.no_grad():
                pred_x, tgt_x, _, out_x = net(x)
                pred_y, tgt_y, _, out_y = net(y)
            # the predictions of patches 1 to 4 use patches 0 to 3 only
            assert torch.allclose(pred_x[:, :4], pred_y[:, :4], atol=1e-6)
            assert not torch.allclose(pred_x[:, 4], pred_y[:, 4])
            assert torch.allclose(tgt_x[:, :3], tgt_y[:, :3], atol=1e-6)
            assert torch.allclose(out_x[:, :4], out_y[:, :4], atol=1e-6)
            err, h_all = net.multi_mask_errors(x, *net.eval_mask_set())
            assert err.shape == (3, 1, 5)
            assert torch.allclose(err[:, 0],
                                  (pred_x - tgt_x).pow(2).mean(-1),
                                  atol=1e-6)
            assert h_all.shape == (3, 6, 16)
        with self.assertRaises(ValueError):
            NextLatentCore(6, 10, style='other')

    def test_cfjepa_network(self):
        import torch
        import torch.nn.functional as F
        from pyod.models._jepa_variants import CFJEPACore, _DilatedBlock
        torch.manual_seed(0)

        def reference(block, x):
            # the same block with PyTorch's own convolutions
            h, dilation = x.transpose(1, 2), block.dilation
            y = sum(F.conv1d(h, conv.weight, conv.bias,
                             padding=dilation * (conv.kernel_size[0] - 1)
                             // 2, dilation=dilation, groups=h.size(1))
                    for conv in block.convs)
            return (h + block.point(F.gelu(block.bn(y)))).transpose(1, 2)

        for dilation, length in ((1, 23), (4, 23), (16, 9), (16, 2)):
            block = _DilatedBlock(8, dilation).eval()
            x = torch.randn(3, length, 8)
            assert torch.allclose(reference(block, x), block(x),
                                  atol=1e-5), (dilation, length)

        net = CFJEPACore(n_tokens=12, patch_len=5, n_channels=2, d_model=16)
        assert net.crop_ratio == (0.367, 0.418)
        assert net.n_ctx_tokens == 5
        assert CFJEPACore(8, 5, 2, d_model=16).crop_ratio == (0.6, 0.8)
        for predictor in net.predictors:  # near the identity
            assert (predictor.weight - torch.eye(16)).abs().max() < 0.01
        x = torch.randn(4, 12, 10)
        # patches are time-major: a reshape recovers the window
        assert torch.equal(net.series(x)[:, 5:10].reshape(4, 10), x[:, 1])
        horizon, crops = net.training_terms(
            x, torch.Generator().manual_seed(1))
        assert len(crops) == 4
        for z in crops:
            assert z.size(1) in (22, 23, 24, 25)
        assert torch.isfinite(horizon)
        net.eval()
        err, h_all = net.multi_mask_errors(x, *net.eval_mask_set())
        assert err.shape == (4, 1, 7)
        assert h_all.shape == (4, 12, 16)
        assert err.min().item() >= 0.0
        # unit vectors: the mean absolute difference is bounded
        assert err.max().item() <= 2.0 / 16 ** 0.5 + 1e-6

    def test_dense_levels(self):
        import torch
        from pyod.models._jepa_variants import DenseJEPACore
        for n_layers, levels in ((1, [1]), (2, [1, 2]), (3, [1, 2, 3]),
                                 (8, [2, 4, 6, 8])):
            net = DenseJEPACore(6, 10, d_model=16, n_heads=2,
                                n_layers=n_layers)
            assert net.levels == levels
        net = self.fitted['vjepa21'].model_
        x = torch.randn(5, 8, 15)
        ctx = torch.arange(0, 6).repeat(5, 1)
        tgt = torch.arange(6, 8).repeat(5, 1)
        with torch.no_grad():
            preds, targets = net(x, ctx, tgt)
        assert len(preds) == 2
        assert preds[0].shape == (5, 8, 64)  # every patch is predicted
        assert targets[1].shape == (5, 8, 64)

    def test_stochastic_latent(self):
        import torch
        net = self.fitted['sjepa'].model_
        assert net.latent_dim == 16
        x = torch.randn(5, 8, 15)
        ctx = torch.arange(0, 6).repeat(5, 1)
        tgt = torch.arange(6, 8).repeat(5, 1)
        with torch.no_grad():
            h_ctx = net.encoder(net.embed(x)[:, :6])
            mu, _ = net.prior(h_ctx)
            # scoring uses the prior mean
            assert torch.equal(net.predict(h_ctx, ctx, tgt),
                               net.predict(h_ctx, ctx, tgt, mu))
            assert not torch.allclose(net.predict(h_ctx, ctx, tgt),
                                      net.predict(h_ctx, ctx, tgt, mu + 1.0))
            kl = net.forward_latent(x, ctx, tgt, torch.zeros(5, 16))[3]
        assert kl.item() >= 0.0

    def test_regularised_variants(self):
        import torch
        patches = torch.randn(32, 8, 15)
        masks = self.fitted['kerjepa']._eval_mask_set()
        # Rectified LpJEPA: non-negative embeddings with exact zeros
        _, emb = self.fitted['lpjepa'].model_.multi_mask_errors(
            patches, *masks)
        assert emb.min().item() >= 0.0
        assert (emb == 0).float().mean().item() > 0.05
        # KerJEPA: running target statistics, frozen when scoring
        model = self.fitted['kerjepa'].model_
        initial = self._fit('kerjepa', epochs=0).model_
        assert torch.equal(initial.tgt_mean, torch.zeros(64))
        assert initial.tgt_var.item() == 1.0
        assert model.tgt_mean.abs().max().item() > 0
        before = model.tgt_mean.clone()
        model.multi_mask_errors(patches, *masks)
        assert torch.equal(before, model.tgt_mean)
        # C-JEPA has the projector for its VICReg terms
        assert hasattr(self.fitted['cjepa'].model_, 'projector')


if __name__ == '__main__':
    unittest.main()
