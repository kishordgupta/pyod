# -*- coding: utf-8 -*-
"""Tests for TSJEPA."""

import os
import pickle
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose
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


@unittest.skipUnless(_torch_available(), "torch not installed")
class TestTSJEPA(unittest.TestCase):
    def setUp(self):
        from pyod.models.ts_jepa import TSJEPA
        self.TSJEPA = TSJEPA
        self.X_train, self.X_test, self.y_train, self.y_test = \
            generate_ts_data(n_train=1000, n_test=400, contamination=0.05,
                             random_state=42)
        self.clf = TSJEPA(window_size=40, patch_len=5, epochs=3,
                          contamination=0.05)
        self.clf.fit(self.X_train)

    def test_fit(self):
        assert len(self.clf.decision_scores_) == 1000
        assert len(self.clf.labels_) == 1000
        assert np.isfinite(self.clf.decision_scores_).all()
        assert set(np.unique(self.clf.labels_)) <= {0, 1}
        for attr in ('model_', 'threshold_', 'n_channels_', 'X_mean_',
                     'X_std_', 'err_mean_', 'err_std_', 'emb_mean_',
                     'emb_prec_'):
            assert hasattr(self.clf, attr), attr

    def test_decision_function(self):
        scores = self.clf.decision_function(self.X_test)
        assert scores.shape == (400,)
        assert np.isfinite(scores).all()
        assert roc_auc_score(self.y_test, scores) >= 0.8

    def test_decision_function_deterministic(self):
        # the scoring masks are seeded, so repeated calls must agree
        assert_allclose(self.clf.decision_function(self.X_test),
                        self.clf.decision_function(self.X_test))

    def test_predict(self):
        labels = self.clf.predict(self.X_test)
        assert labels.shape == self.y_test.shape
        assert set(np.unique(labels)) <= {0, 1}

    def test_predict_proba(self):
        proba = self.clf.predict_proba(self.X_test)
        assert proba.min() >= 0
        assert proba.max() <= 1

    def test_multivariate(self):
        X_train, X_test, _, _ = generate_ts_data(
            n_train=600, n_test=300, n_channels=3, contamination=0.05,
            random_state=42)
        clf = self.TSJEPA(window_size=40, patch_len=5, epochs=1)
        clf.fit(X_train)
        assert clf.n_channels_ == 3
        assert clf.X_mean_.shape == (3,)
        assert len(clf.decision_scores_) == 600
        scores = clf.decision_function(X_test)
        assert scores.shape == (300,)
        assert np.isfinite(scores).all()

    def test_mask_modes_and_scores(self):
        X_train, X_test, _, _ = generate_ts_data(
            n_train=400, n_test=300, n_channels=2, contamination=0.05,
            anomaly_type='both', random_state=1)
        for mask_mode in ('block', 'forecast', 'random', 'mixed'):
            for score in ('pred', 'embed', 'combined'):
                clf = self.TSJEPA(window_size=40, patch_len=5, epochs=1,
                                  mask_mode=mask_mode, score=score)
                clf.fit(X_train)
                scores = clf.decision_function(X_test)
                assert scores.shape == (300,), (mask_mode, score)
                assert np.isfinite(scores).all(), (mask_mode, score)

    def test_causal_boundary(self):
        # 'forecast' hides the last 2 of 8 patches of every window, so
        # the first 30 timestamps are never hidden and get threshold_
        clf = self.TSJEPA(window_size=40, patch_len=5, epochs=1,
                          mask_mode='forecast')
        clf.fit(self.X_train)
        assert np.allclose(clf.decision_scores_[:30], clf.threshold_)
        assert not np.allclose(clf.decision_scores_[30:60], clf.threshold_)
        scores = clf.decision_function(self.X_test)
        assert np.allclose(scores[:30], clf.threshold_)

    def test_step_leaves_no_gaps_unfilled(self):
        # with step > 1 the tail may not be covered by any window; those
        # timestamps must be filled, not left as NaN
        clf = self.TSJEPA(window_size=40, patch_len=5, epochs=1, step=7)
        clf.fit(self.X_train)
        assert np.isfinite(clf.decision_scores_).all()
        assert np.isfinite(clf.decision_function(self.X_test)).all()

    def test_ema_target_and_max_aggregation(self):
        clf = self.TSJEPA(window_size=40, patch_len=5, epochs=1,
                          variant='ijepa', recon_weight=0.1,
                          score_aggregation='max', n_mask_samples=3)
        clf.fit(self.X_train)
        assert clf.model_.target_encoder is not None
        assert np.isfinite(clf.decision_function(self.X_test)).all()

    def test_max_train_windows(self):
        clf = self.TSJEPA(window_size=40, patch_len=5, epochs=1,
                          max_train_windows=100)
        clf.fit(self.X_train)
        assert len(clf.decision_scores_) == 1000

    def test_model_clone(self):
        clone_clf = clone(self.clf)
        assert clone_clf.get_params() == self.clf.get_params()

    def test_refit_reproduces_scores(self):
        refit = clone(self.clf).fit(self.X_train)
        assert_allclose(refit.decision_scores_, self.clf.decision_scores_,
                        rtol=1e-4, atol=1e-5)

    def test_pickle_roundtrip(self):
        restored = pickle.loads(pickle.dumps(self.clf))
        assert_allclose(restored.decision_function(self.X_test),
                        self.clf.decision_function(self.X_test))

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            # window_size is not a multiple of patch_len
            self.TSJEPA(window_size=45, patch_len=10).fit(self.X_train)
        with self.assertRaises(ValueError):
            # a window must hold at least two patches
            self.TSJEPA(window_size=10, patch_len=10).fit(self.X_train)
        with self.assertRaises(ValueError):
            self.TSJEPA(window_size=40, patch_len=5,
                        mask_mode='something').fit(self.X_train)
        with self.assertRaises(ValueError):
            self.TSJEPA(window_size=40, patch_len=5,
                        score='something').fit(self.X_train)
        with self.assertRaises(ValueError):
            self.TSJEPA(window_size=40, patch_len=5,
                        score_aggregation='something').fit(self.X_train)
        with self.assertRaises(ValueError):
            self.TSJEPA(window_size=40, patch_len=5,
                        mask_ratio=0.0).fit(self.X_train)
        with self.assertRaises(ValueError):
            self.TSJEPA(window_size=40, patch_len=5,
                        variant='something').fit(self.X_train)
        for variant in ('tsjepa', 'tjepa'):
            with self.assertRaises(ValueError):
                # the reconstruction head exists for two variants only
                self.TSJEPA(window_size=40, patch_len=5, variant=variant,
                            recon_weight=0.1).fit(self.X_train)

    def test_short_series_raises(self):
        with self.assertRaises(ValueError):
            self.TSJEPA(window_size=40, patch_len=5).fit(self.X_train[:30])
        with self.assertRaises(ValueError):
            self.clf.decision_function(self.X_test[:30])

    def test_channel_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.clf.decision_function(np.random.randn(500, 2))


VARIANTS = ('lejepa', 'ijepa', 'tsjepa', 'tjepa')
EMA_VARIANTS = ('ijepa', 'tsjepa', 'tjepa')


@unittest.skipUnless(_torch_available(), "torch not installed")
class TestTSJEPAVariants(unittest.TestCase):
    """The JEPA versions selected with ``variant``."""

    @classmethod
    def setUpClass(cls):
        from pyod.models.ts_jepa import TSJEPA
        cls.TSJEPA = TSJEPA
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = \
            generate_ts_data(n_train=1000, n_test=400, contamination=0.05,
                             random_state=42)
        cls.params = dict(window_size=40, patch_len=5, epochs=2,
                          score='combined', random_state=3)
        cls.fitted = {
            variant: TSJEPA(variant=variant, **cls.params).fit(cls.X_train)
            for variant in VARIANTS}

    def _fit(self, variant, **kwargs):
        params = dict(self.params, **kwargs)
        return self.TSJEPA(variant=variant, **params).fit(self.X_train)

    def test_default_variant(self):
        assert self.TSJEPA().variant == 'lejepa'
        default = self.TSJEPA(**self.params).fit(self.X_train)
        assert_allclose(default.decision_scores_,
                        self.fitted['lejepa'].decision_scores_,
                        rtol=1e-4, atol=1e-5)

    def test_scores(self):
        for variant, clf in self.fitted.items():
            assert clf.decision_scores_.shape == (1000,), variant
            assert np.isfinite(clf.decision_scores_).all(), variant
            for score in ('pred', 'embed', 'combined'):
                clf.score = score
                scores = clf.decision_function(self.X_test)
                assert scores.shape == (400,), (variant, score)
                assert np.isfinite(scores).all(), (variant, score)
            clf.score = 'combined'
            scores = clf.decision_function(self.X_test)
            assert roc_auc_score(self.y_test, scores) >= 0.8, variant

    def test_variants_differ(self):
        reference = self.fitted['lejepa'].decision_scores_
        for variant in EMA_VARIANTS:
            assert not np.allclose(reference,
                                   self.fitted[variant].decision_scores_)

    def test_resolved_defaults(self):
        expected = {'lejepa': (False, None), 'ijepa': (False, 0.996),
                    'tsjepa': (True, 0.998), 'tjepa': (True, 0.996)}
        for variant, (norm, momentum) in expected.items():
            clf = self.fitted[variant]
            assert clf.encoder_norm_ == norm, variant
            assert clf.ema_momentum_ == momentum, variant
            # the constructor arguments are stored as given
            assert clf.encoder_norm is None
            assert clf.ema_momentum is None
        # explicit values override the published ones
        for variant in ('tsjepa', 'tjepa'):
            clf = self._fit(variant, encoder_norm=False, ema_momentum=0.99,
                            epochs=1)
            assert clf.encoder_norm_ is False
            assert clf.ema_momentum_ == 0.99
            assert np.isfinite(clf.decision_scores_).all(), variant
            assert not np.allclose(clf.decision_scores_,
                                   self.fitted[variant].decision_scores_)

    def test_networks(self):
        from pyod.models._jepa_utils import (
            ConvPatchTokenizer, JEPACore, TJEPACore, TSJEPACore)
        types = {'lejepa': JEPACore, 'ijepa': JEPACore,
                 'tsjepa': TSJEPACore, 'tjepa': TJEPACore}
        for variant, cls in types.items():
            assert type(self.fitted[variant].model_) is cls, variant
        assert self.fitted['lejepa'].model_.target_encoder is None
        for variant in EMA_VARIANTS:
            target = self.fitted[variant].model_.target_encoder
            assert not any(p.requires_grad for p in target.parameters())
        # the EMA covers the tokenizer for TS-JEPA and T-JEPA only
        assert not hasattr(self.fitted['ijepa'].model_.target_encoder,
                           'tokenizer')
        for variant in ('tsjepa', 'tjepa'):
            assert hasattr(self.fitted[variant].model_.target_encoder,
                           'tokenizer')
        assert isinstance(self.fitted['tsjepa'].model_.encoder.tokenizer,
                          ConvPatchTokenizer)
        assert self.fitted['tjepa'].model_.encoder.reg.shape[0] == 1

    def test_ema_target_follows_encoder(self):
        for variant in EMA_VARIANTS:
            model = self.fitted[variant].model_
            initial = self._fit(variant, epochs=0).model_
            moved = max((a - b).abs().max().item() for a, b in zip(
                model.target_encoder.parameters(),
                initial.target_encoder.parameters()))
            lag = max((a - b).abs().max().item() for a, b in zip(
                model.target_encoder.parameters(),
                model.encoder.parameters()))
            assert moved > 0, variant  # the target was updated
            assert lag > 0, variant    # and is not the online encoder

    def test_zero_epochs_keeps_initial_weights(self):
        import torch
        for variant in VARIANTS:
            clf = self._fit(variant, epochs=0)
            torch.manual_seed(self.params['random_state'])
            fresh = clf._build_model(clf.n_tokens_, clf.n_channels_)
            # the fresh network is on the CPU, the fitted one where
            # device='auto' put it
            for a, b in zip(clf.model_.parameters(), fresh.parameters()):
                assert torch.equal(a.cpu(), b), variant
            assert np.isfinite(clf.decision_scores_).all(), variant

    def test_mask_modes(self):
        for variant in EMA_VARIANTS:
            for mask_mode in ('forecast', 'random'):
                clf = self._fit(variant, mask_mode=mask_mode, epochs=1)
                scores = clf.decision_function(self.X_test)
                assert np.isfinite(scores).all(), (variant, mask_mode)
        # the masking of the TS-JEPA paper: 70 % of the patches, at random
        clf = self._fit('tsjepa', mask_mode='random', mask_ratio=0.7,
                        window_size=50, epochs=1)
        assert (clf.n_tokens_, clf.n_mask_) == (10, 7)
        assert np.isfinite(clf.decision_function(self.X_test)).all()

    def test_multivariate(self):
        X_train, X_test, _, _ = generate_ts_data(
            n_train=600, n_test=300, n_channels=3, contamination=0.05,
            anomaly_type='both', random_state=1)
        for variant in EMA_VARIANTS:
            clf = self.TSJEPA(window_size=40, patch_len=5, epochs=1,
                              variant=variant).fit(X_train)
            assert clf.n_channels_ == 3
            scores = clf.decision_function(X_test)
            assert scores.shape == (300,), variant
            assert np.isfinite(scores).all(), variant

    def test_refit_clone_and_pickle(self):
        for variant in EMA_VARIANTS:
            clf = self.fitted[variant]
            cloned = clone(clf)
            assert cloned.get_params() == clf.get_params()
            assert cloned.variant == variant
            refit = cloned.fit(self.X_train)
            assert_allclose(refit.decision_scores_, clf.decision_scores_,
                            rtol=1e-4, atol=1e-5)
            restored = pickle.loads(pickle.dumps(clf))
            assert_allclose(restored.decision_function(self.X_test),
                            clf.decision_function(self.X_test))

    def test_conv_tokenizer_layout(self):
        import torch
        from pyod.models._jepa_utils import ConvPatchTokenizer
        # a patch arrives flattened time-major: [t0c0, t0c1, t1c0, ...]
        torch.manual_seed(0)
        tokenizer = ConvPatchTokenizer(patch_len=6, n_channels=2, out_dim=8)
        x = torch.randn(4, 5, 12)
        out = tokenizer(x)
        assert out.shape == (4, 5, 8)
        one = x[1, 3].view(6, 2).t().unsqueeze(0)  # (1, C, L)
        expected = tokenizer.fc(tokenizer.conv(one).flatten(1))[0]
        assert torch.allclose(out[1, 3], expected, atol=1e-6)
        # the kernel is capped at the patch length
        short = ConvPatchTokenizer(patch_len=2, n_channels=1, out_dim=8)
        assert short.conv.kernel_size == (2,)
        assert short(torch.randn(3, 4, 2)).shape == (3, 4, 8)


if __name__ == '__main__':
    unittest.main()
