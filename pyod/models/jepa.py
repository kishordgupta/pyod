# -*- coding: utf-8 -*-
"""Outlier detection with a Joint-Embedding Predictive Architecture (JEPA)
"""
# Author: Kishor Datta Gupta
# License: BSD 2 clause


try:
    import torch
except ImportError:
    print('please install torch first')

import math

import numpy as np
import torch.nn.functional as F

from pyod.models._jepa_utils import JEPACore, MASK_MODES
from pyod.models._jepa_utils import ema_update, eval_masks
from pyod.models._jepa_utils import sample_masks, sigreg_loss
from pyod.models.base_dl import BaseDeepLearningDetector

_SCORES = ('pred', 'embed', 'combined')


class JEPA(BaseDeepLearningDetector):
    """
    Joint-Embedding Predictive Architecture (JEPA) for outlier detection.

    A JEPA learns to predict the *latent representation* of hidden parts
    of an input from its visible parts :cite:`lecun2022path,assran2023self`.
    Here every feature (or small group of consecutive features) of a row
    is a token. During training a random subset of tokens is hidden, a
    Transformer encoder embeds the visible tokens, and a light predictor
    forecasts the embeddings of the hidden tokens. Normal rows have
    feature dependencies the predictor can model, so a row whose hidden
    features cannot be predicted from the others in representation space
    is an outlier. Unlike reconstruction error (AutoEncoder), the error
    is measured in a learned latent space, so unpredictable noise in the
    raw features is not penalised.

    Representation collapse is prevented with SIGReg, the Sketched
    Isotropic Gaussian Regularization of LeJEPA
    :cite:`balestriero2025lejepa`, which pushes the embeddings of the
    training data towards an isotropic Gaussian and needs no teacher
    network. This also gives a second score, the Mahalanobis distance of
    each token embedding to the training Gaussian.

    Three outlier scores are available through ``score``:

    - ``'pred'``: latent prediction error, z-normalised per token and
      averaged over ``n_mask_samples`` masks.
    - ``'embed'``: Mahalanobis distance of each token embedding to the
      Gaussian fitted on the training embeddings, averaged over tokens.
    - ``'combined'``: the mean of both after z-normalisation with their
      training statistics.

    Parameters
    ----------
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set.
        Used when fitting to define the threshold on the decision function.

    preprocessing : bool, optional (default=True)
        If True, apply the preprocessing procedure before training models.

    lr : float, optional (default=1e-3)
        The initial learning rate for the optimizer.

    epoch_num : int, optional (default=30)
        The number of epochs for training.

    batch_size : int, optional (default=64)
        The batch size for training.

    optimizer_name : str, optional (default='adam')
        The name of the optimizer used to train the model.

    criterion_name : str, optional (default='mse')
        The name of the criterion applied to the latent prediction error.

    device : str, optional (default=None)
        The device to use for the model. If None, it will be decided
        automatically. If you want to use MPS, set it to 'mps'.

    random_state : int, optional (default=42)
        The random seed for reproducibility.

    use_compile : bool, optional (default=False)
        Whether to compile the model.
        If True, the model will be compiled before training.
        This is only available for
        PyTorch version >= 2.0.0. and Python < 3.12.

    compile_mode : str, optional (default='default')
        The mode to compile the model.
        Can be either "default", "reduce-overhead",
        "max-autotune" or "max-autotune-no-cudagraphs".
        See https://pytorch.org/docs/stable/generated/torch.compile.html#torch-compile for details.

    verbose : int, optional (default=1)
        Verbosity mode.
        - 0 = silent
        - 1 = progress bar
        - 2 = one line per epoch.

    optimizer_params : dict, optional (default={'weight_decay': 1e-4})
        Additional parameters for the optimizer.
        For example, `optimizer_params={'weight_decay': 1e-4}`.

    criterion_params : dict, optional (default={})
        Additional parameters for the criterion.
        For example, `criterion_params={'reduction': 'sum'}`.

    d_model : int, optional (default=64)
        The width of the token embeddings.

    n_heads : int, optional (default=4)
        The number of attention heads of the encoder.
        Must divide ``d_model`` evenly.

    n_layers : int, optional (default=2)
        The number of Transformer layers of the encoder.

    pred_layers : int, optional (default=1)
        The number of Transformer layers of the predictor, which has
        width ``d_model // 2``.

    dropout : float in [0., 1), optional (default=0.0)
        The dropout rate of the encoder and the predictor.

    max_tokens : int, optional (default=64)
        The maximum number of tokens. With more than ``max_tokens``
        features, consecutive features are grouped so that each token
        holds ``ceil(n_features / max_tokens)`` features.

    mask_ratio : float in (0., 1.), optional (default=0.3)
        The fraction of tokens hidden from the encoder. At least one
        token is hidden and at least one stays visible.

    mask_mode : str, optional (default='random')
        The masking pattern, one of 'random', 'block', 'forecast' and
        'mixed'. 'random' hides a uniformly random subset of tokens and
        suits tabular data, where the feature order carries no meaning.
        'block' hides contiguous tokens, 'forecast' always hides the
        last tokens, and 'mixed' alternates between these two.

    n_mask_samples : int, optional (default=8)
        The number of masks each sample is scored under. The mask set
        is drawn once from ``random_state``, so scoring is deterministic.

    target_mode : str, optional (default='sigreg')
        How representation collapse is prevented. 'sigreg' uses one
        shared encoder and adds SIGReg to the loss. 'ema' uses an
        exponential-moving-average target encoder as in I-JEPA.

    sigreg_weight : float in [0., 1.], optional (default=0.05)
        The weight ``w`` of SIGReg. The loss is
        ``(1 - w) * prediction + w * SIGReg``.
        Only used when ``target_mode='sigreg'``.

    sigreg_slices : int, optional (default=128)
        The number of random projections per SIGReg evaluation.
        Only used when ``target_mode='sigreg'``.

    ema_momentum : float in [0., 1.], optional (default=0.996)
        The initial momentum of the target encoder, annealed to 1 over
        training. Only used when ``target_mode='ema'``.

    recon_weight : float, optional (default=0.0)
        The weight of an auxiliary loss that linearly reconstructs each
        raw token from its embedding. 0 disables it.

    score : str, optional (default='combined')
        The outlier score, one of 'pred', 'embed' and 'combined' as
        described above. The embedding distance is sensitive to a shift
        between training and test data, so prefer 'pred' when the test
        data may be shifted.

    error_norm : bool, optional (default=True)
        If True, z-normalise the latent error of every token with its
        training mean and standard deviation, so that features that are
        hard to predict for all samples do not dominate the score.

    lr_schedule : bool, optional (default=True)
        If True, decay the learning rate with a cosine schedule over
        ``epoch_num``.

    grad_clip : float or None, optional (default=1.0)
        The maximum gradient norm. None or 0 disables clipping.

    encoder_norm : bool, optional (default=False)
        If True, apply a LayerNorm to the encoder output. It is off by
        default because it lets extreme inputs saturate to a direction
        that is easy to predict, so that strong outliers can receive
        low scores.

    Attributes
    ----------
    model : torch.nn.Module
        The underlying JEPA model.

    optimizer : torch.optim
        The optimizer used to train the model.

    criterion : torch.nn.modules
        The loss function applied to the latent prediction error.

    n_tokens_ : int
        The number of tokens each sample is split into.

    token_dim_ : int
        The number of features per token.

    err_mean_ : numpy array of shape (n_tokens,)
        The mean latent prediction error of every token on the
        training data.

    err_std_ : numpy array of shape (n_tokens,)
        The standard deviation of the latent prediction error of every
        token on the training data.

    emb_mean_ : numpy array of shape (n_tokens, d_model)
        The mean training embedding of every token.

    emb_prec_ : numpy array of shape (d_model, d_model)
        The inverse of the covariance of the training embeddings around
        their token means, shared by all tokens.

    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : numpy array of shape (n_samples,)
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self,
                 contamination=0.1, preprocessing=True,
                 lr=1e-3, epoch_num=30, batch_size=64,
                 optimizer_name='adam', criterion_name='mse',
                 device=None, random_state=42,
                 use_compile=False, compile_mode='default',
                 verbose=1,
                 optimizer_params: dict = {'weight_decay': 1e-4},
                 criterion_params: dict = {},
                 d_model=64, n_heads=4, n_layers=2, pred_layers=1,
                 dropout=0.0, max_tokens=64,
                 mask_ratio=0.3, mask_mode='random', n_mask_samples=8,
                 target_mode='sigreg', sigreg_weight=0.05,
                 sigreg_slices=128, ema_momentum=0.996,
                 recon_weight=0.0,
                 score='combined', error_norm=True,
                 lr_schedule=True, grad_clip=1.0, encoder_norm=False):
        super(JEPA, self).__init__(contamination=contamination,
                                   preprocessing=preprocessing,
                                   lr=lr, epoch_num=epoch_num,
                                   batch_size=batch_size,
                                   optimizer_name=optimizer_name,
                                   criterion_name=criterion_name,
                                   device=device,
                                   random_state=random_state,
                                   use_compile=use_compile,
                                   compile_mode=compile_mode,
                                   verbose=verbose,
                                   optimizer_params=optimizer_params,
                                   criterion_params=criterion_params)
        self.criterion_name = criterion_name
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.pred_layers = pred_layers
        self.dropout = dropout
        self.max_tokens = max_tokens
        self.mask_ratio = mask_ratio
        self.mask_mode = mask_mode
        self.n_mask_samples = n_mask_samples
        self.target_mode = target_mode
        self.sigreg_weight = sigreg_weight
        self.sigreg_slices = sigreg_slices
        self.ema_momentum = ema_momentum
        self.recon_weight = recon_weight
        self.score = score
        self.error_norm = error_norm
        self.lr_schedule = lr_schedule
        self.grad_clip = grad_clip
        self.encoder_norm = encoder_norm

    def build_model(self):
        if self.feature_size < 2:
            raise ValueError("JEPA needs at least 2 features to form a "
                             "context / target split; got %d."
                             % self.feature_size)
        if self.score not in _SCORES:
            raise ValueError("score must be one of %s; got %r"
                             % (_SCORES, self.score))
        if self.mask_mode not in MASK_MODES:
            raise ValueError("mask_mode must be one of %s; got %r"
                             % (MASK_MODES, self.mask_mode))
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1); got %r"
                             % (self.mask_ratio,))
        if self.max_tokens < 2:
            raise ValueError("max_tokens must be at least 2; got %r"
                             % (self.max_tokens,))

        # group features into at most max_tokens tokens of equal size
        n_tokens = min(self.feature_size, self.max_tokens)
        token_dim = int(math.ceil(self.feature_size / n_tokens))
        n_tokens = int(math.ceil(self.feature_size / token_dim))
        self.n_tokens_ = n_tokens
        self.token_dim_ = token_dim
        self.pad_ = n_tokens * token_dim - self.feature_size
        self.n_mask_ = int(np.clip(round(self.mask_ratio * n_tokens),
                                   1, n_tokens - 1))
        self.model = JEPACore(
            n_tokens=n_tokens, in_dim=token_dim, d_model=self.d_model,
            n_heads=self.n_heads, n_layers=self.n_layers,
            pred_layers=self.pred_layers, dropout=self.dropout,
            target_mode=self.target_mode,
            recon_dim=token_dim if self.recon_weight > 0 else None,
            encoder_norm=self.encoder_norm)

    def training_prepare(self):
        super(JEPA, self).training_prepare()
        self._train_gen = torch.Generator().manual_seed(self.random_state)
        self.scheduler = None
        if self.lr_schedule and self.epoch_num > 1:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epoch_num, eta_min=self.lr * 0.05)
        steps_per_epoch = max(1, self.data_num // self.batch_size)
        self._total_steps = max(1, self.epoch_num * steps_per_epoch)
        self._step = 0

    def _to_tokens(self, x):
        """Reshape (B, n_features) to (B, n_tokens, token_dim).

        Features are zero-padded when ``n_features`` is not a multiple
        of the token size.
        """
        if self.pad_ > 0:
            x = F.pad(x, (0, self.pad_))
        return x.view(x.size(0), self.n_tokens_, self.token_dim_)

    def training_forward(self, batch_data):
        x = batch_data.to(self.device)
        tokens = self._to_tokens(x)
        ctx, tgt = sample_masks(tokens.size(0), self.n_tokens_,
                                self.n_mask_, self.mask_mode,
                                self._train_gen)
        ctx, tgt = ctx.to(self.device), tgt.to(self.device)

        self.optimizer.zero_grad()
        pred, h_tgt, h_all = self.model(tokens, ctx, tgt)
        # The targets are detached: the encoder must not lower the loss
        # by making its own targets easier to predict.
        loss_pred = self.criterion(pred, h_tgt.detach())
        loss = loss_pred
        parts = [loss_pred.item()]
        if self.target_mode == 'sigreg':
            # one group per token position: (T, B, D)
            reg = sigreg_loss(h_all.transpose(0, 1),
                              num_slices=self.sigreg_slices,
                              generator=self._train_gen)
            loss = (1.0 - self.sigreg_weight) * loss_pred \
                   + self.sigreg_weight * reg
            parts.append(reg.item())
        if self.recon_weight > 0:
            loss_rec = F.mse_loss(self.model.recon_head(h_all), tokens)
            loss = loss + self.recon_weight * loss_rec
            parts.append(loss_rec.item())
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.grad_clip)
        self.optimizer.step()

        if self.target_mode == 'ema':
            frac = min(1.0, self._step / self._total_steps)
            momentum = self.ema_momentum + (1.0 - self.ema_momentum) * frac
            ema_update(self.model.target_encoder, self.model.encoder,
                       momentum)
        self._step += 1
        return tuple(parts) if len(parts) > 1 else parts[0]

    def epoch_update(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def train(self, train_loader):
        super(JEPA, self).train(train_loader)
        self._calibrate(train_loader)

    def evaluating_prepare(self):
        self.model.eval()
        gen = torch.Generator().manual_seed(self.random_state + 1)
        self._eval_ctx, self._eval_tgt = eval_masks(
            self.n_tokens_, self.n_mask_, self.mask_mode,
            self.n_mask_samples, gen)

    def evaluating_forward(self, batch_data):
        x = batch_data.to(self.device)
        s_pred, s_emb = self._component_scores(x)
        return self._combine(s_pred, s_emb)

    def _raw(self, x):
        """Latent errors and embeddings of one batch, on CPU.

        Returns
        -------
        err : torch.Tensor of shape (B, n_masks, n_mask)
            Latent prediction error per mask and hidden token.

        tgt : torch.LongTensor of shape (n_masks, n_mask)
            Hidden token indices per mask.

        h_all : torch.Tensor of shape (B, n_tokens, d_model)
            Token embeddings.
        """
        err, h_all = self.model.multi_mask_errors(
            self._to_tokens(x), self._eval_ctx, self._eval_tgt)
        return err.cpu(), self._eval_tgt, h_all.cpu()

    def _component_scores(self, x):
        """Return the 'pred' and 'embed' scores of one batch."""
        err, tgt, h_all = self._raw(x)
        if self.error_norm:
            mu = torch.as_tensor(self.err_mean_, dtype=err.dtype)[tgt]
            sd = torch.as_tensor(self.err_std_, dtype=err.dtype)[tgt]
            err = (err - mu) / sd
        s_pred = err.mean(dim=(1, 2)).numpy()
        diff = h_all.numpy() - self.emb_mean_  # (B, T, D)
        s_emb = np.einsum('bti,ij,btj->bt', diff, self.emb_prec_,
                          diff).mean(1)
        return s_pred, s_emb

    def _combine(self, s_pred, s_emb):
        """Select or fuse the component scores according to ``score``."""
        if self.score == 'pred':
            return s_pred
        if self.score == 'embed':
            return s_emb
        z_pred = (s_pred - self.comp_mean_[0]) / self.comp_std_[0]
        z_emb = (s_emb - self.comp_mean_[1]) / self.comp_std_[1]
        return 0.5 * (z_pred + z_emb)

    @torch.no_grad()
    def _calibrate(self, train_loader):
        """Fit the scoring statistics on the training data.

        Sets the per-token error statistics (``err_mean_``,
        ``err_std_``), the embedding Gaussian (``emb_mean_``,
        ``emb_prec_``) and the statistics used to z-normalise the two
        component scores (``comp_mean_``, ``comp_std_``).
        """
        self.evaluating_prepare()
        n_tokens, d_model = self.n_tokens_, self.d_model
        err_sum = torch.zeros(n_tokens, dtype=torch.float64)
        err_sq = torch.zeros(n_tokens, dtype=torch.float64)
        err_cnt = torch.zeros(n_tokens, dtype=torch.float64)
        emb_sum = torch.zeros(n_tokens, d_model, dtype=torch.float64)
        emb_sq = torch.zeros(d_model, d_model, dtype=torch.float64)
        n_rows = 0
        for batch in train_loader:
            err, tgt, h_all = self._raw(batch.to(self.device))
            h = h_all.double()
            emb_sum += h.sum(0)
            emb_sq += torch.einsum('bti,btj->ij', h, h)
            n_rows += h.size(0)
            e = err.double().reshape(err.size(0), -1)  # (B, K * m)
            idx = tgt.reshape(-1).expand(e.size(0), -1).reshape(-1)
            e = e.reshape(-1)
            err_sum.index_add_(0, idx, e)
            err_sq.index_add_(0, idx, e ** 2)
            err_cnt.index_add_(0, idx, torch.ones_like(e))

        # tokens never hidden by the mask set fall back to global stats
        seen = err_cnt > 0
        g_mean = (err_sum.sum() / err_cnt.sum()).item()
        g_var = max((err_sq.sum() / err_cnt.sum()).item() - g_mean ** 2,
                    1e-12)
        mean = torch.full((n_tokens,), g_mean, dtype=torch.float64)
        var = torch.full((n_tokens,), g_var, dtype=torch.float64)
        mean[seen] = err_sum[seen] / err_cnt[seen]
        var[seen] = (err_sq[seen] / err_cnt[seen]
                     - mean[seen] ** 2).clamp_min(0)
        std = var.sqrt()
        std = torch.where(std > 1e-8, std,
                          torch.full_like(std, g_var ** 0.5))
        self.err_mean_ = mean.numpy()
        self.err_std_ = np.maximum(std.numpy(), 1e-8)

        # covariance of the token embeddings around their token means,
        # pooled over tokens
        token_mean = emb_sum / max(n_rows, 1)  # (T, D)
        cov = (emb_sq - n_rows * token_mean.t() @ token_mean) \
              / max(n_rows * n_tokens - 1, 1)
        cov = cov.numpy()
        ridge = 1e-3 * max(np.trace(cov) / d_model, 1e-6)
        self.emb_mean_ = token_mean.numpy()
        self.emb_prec_ = np.linalg.inv(cov + ridge * np.eye(d_model))

        s_pred, s_emb = [], []
        for batch in train_loader:
            a, b = self._component_scores(batch.to(self.device))
            s_pred.append(a)
            s_emb.append(b)
        s_pred, s_emb = np.concatenate(s_pred), np.concatenate(s_emb)
        self.comp_mean_ = np.array([s_pred.mean(), s_emb.mean()])
        self.comp_std_ = np.maximum(
            np.array([s_pred.std(), s_emb.std()]), 1e-8)
