# -*- coding: utf-8 -*-
"""TSJEPA: time series anomaly detection with a Joint-Embedding
Predictive Architecture (JEPA).

A sliding window is cut into patches of ``patch_len`` timestamps, some
patches are hidden, and a predictor forecasts the *latent embedding* of
the hidden patches from the visible ones. Timestamps whose patch
embedding cannot be predicted from its context are anomalous.

``variant`` selects the JEPA that trains the network: LeJEPA (SIGReg),
I-JEPA (EMA target encoder), TS-JEPA or T-JEPA.

PyTorch is imported lazily so that it remains an optional dependency.
"""
# Author: Kishor Datta Gupta
# License: BSD 2 clause

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector
from ._ts_utils import validate_ts_input

_SCORES = ('pred', 'embed', 'combined')
_AGGREGATIONS = ('mean', 'max')
_VARIANTS = ('lejepa', 'ijepa', 'tsjepa', 'tjepa')
# What None means for the two parameters whose published value differs
# between the variants: (encoder_norm, ema_momentum).
_VARIANT_DEFAULTS = {'lejepa': (False, None), 'ijepa': (False, 0.996),
                     'tsjepa': (True, 0.998), 'tjepa': (True, 0.996)}


def _require_torch():
    """Import torch or raise an informative ImportError."""
    try:
        import torch
    except ImportError:
        raise ImportError(
            "TSJEPA requires PyTorch. "
            "Install it with: pip install torch")
    return torch


class TSJEPA(BaseDetector):
    """Joint-Embedding Predictive Architecture for time series anomaly
    detection.

    A JEPA learns to predict the latent representation of hidden parts
    of an input from its visible parts [1]_ [2]_. Each sliding window of
    ``window_size`` timestamps is cut into ``window_size // patch_len``
    patches, and all channels of a patch form one token. During training
    some patches are hidden, a Transformer encoder embeds the visible
    patches, and a light predictor forecasts the embeddings of the
    hidden ones. At inference the latent prediction error of every patch
    is mapped back to the timestamps it covers and aggregated over all
    windows and masks that cover a timestamp.

    Because the error is measured in representation space, unpredictable
    noise in the raw values is not penalised, unlike raw-space
    forecasting or reconstruction.

    ``variant`` selects the JEPA that trains the network:

    - ``'lejepa'`` (default): one shared encoder with a per-patch linear
      tokenizer and learned positions. Representation collapse is
      prevented with SIGReg, the Sketched Isotropic Gaussian
      Regularization of LeJEPA [3]_, which pushes the embeddings of the
      training data towards an isotropic Gaussian and needs no teacher
      network.
    - ``'ijepa'``: the same network with an exponential-moving-average
      (EMA) target encoder and normalised targets, as in I-JEPA [2]_.
    - ``'tsjepa'``: the network of TS-JEPA [4]_, with a convolutional
      patch tokenizer, sinusoidal positions, a normalised encoder
      output, an EMA copy of the whole encoder as target and the
      absolute error as training loss.
    - ``'tjepa'``: the network of T-JEPA [5]_, a tabular method, applied
      to patch tokens: sinusoidal positions, a regularisation token, a
      post-norm encoder with a normalised output and an EMA copy of the
      whole encoder as target.

    A variant contributes its network, its target mechanism and its
    training loss. Masking, model sizes, optimizer, training budget and
    the anomaly scores are those of this detector for every variant;
    none of the cited papers defines an anomaly score. The variants
    with normalised targets or outputs (``'ijepa'``, ``'tsjepa'``,
    ``'tjepa'``) can lose accuracy when the test series follows a
    different operating regime than the training series; ``'lejepa'``
    is the default for that reason.

    Three anomaly scores are available through ``score``:

    - ``'pred'``: z-normalised latent prediction error.
    - ``'embed'``: Mahalanobis distance of each patch embedding to the
      Gaussian fitted on the training embeddings.
    - ``'combined'``: the mean of both after z-normalisation with their
      training statistics.

    Parameters
    ----------
    window_size : int, optional (default=100)
        Length of the sliding window. Must be a multiple of
        ``patch_len`` and hold at least two patches.

    patch_len : int, optional (default=10)
        Number of timestamps per patch (token).

    step : int, optional (default=1)
        Step size between consecutive scoring windows.

    train_step : int or None, optional (default=None)
        Step size between consecutive training windows.
        None uses ``max(1, patch_len // 2)``.

    max_train_windows : int, optional (default=5000)
        Maximum number of training windows. Longer series are randomly
        subsampled. Together with ``epochs`` this bounds the training
        budget: with long training on a contaminated series the model
        can learn to predict a sustained anomalous regime, which then
        scores as normal.

    d_model : int, optional (default=64)
        Width of the patch embeddings.

    n_heads : int, optional (default=4)
        Number of attention heads. Must divide ``d_model`` evenly.

    n_layers : int, optional (default=2)
        Number of Transformer layers of the encoder.

    pred_layers : int, optional (default=1)
        Number of Transformer layers of the predictor. Its width is
        ``d_model // 2`` for 'lejepa' and 'ijepa' and ``d_model`` for
        'tsjepa' and 'tjepa'.

    dropout : float, optional (default=0.0)
        Dropout rate of the encoder and the predictor.

    mask_mode : str, optional (default='block')
        Masking pattern, one of 'block', 'forecast', 'mixed' and
        'random'. 'block' hides one contiguous run of patches anywhere
        in the window, so the context lies on both sides. 'forecast'
        always hides the last patches, which makes the detector causal.
        'mixed' trains on both. 'random' hides scattered patches.

    mask_ratio : float in (0., 1.), optional (default=0.3)
        Fraction of patches hidden. At least one patch is hidden and at
        least one stays visible.

    n_mask_samples : int or None, optional (default=None)
        Number of masks each window is scored under. None enumerates
        every block position for 'block' and 'mixed' and uses 8 masks
        for 'random'. Ignored for 'forecast', which has a single mask.

    variant : str, optional (default='lejepa')
        The JEPA that trains the network, one of 'lejepa', 'ijepa',
        'tsjepa' and 'tjepa' as described above.

    sigreg_weight : float in [0., 1.], optional (default=0.05)
        Weight ``w`` of SIGReg. The loss is
        ``(1 - w) * prediction + w * SIGReg``.
        Only used when ``variant='lejepa'``.

    sigreg_slices : int, optional (default=128)
        Number of random projections per SIGReg evaluation.
        Only used when ``variant='lejepa'``.

    ema_momentum : float in [0., 1.] or None, optional (default=None)
        Initial momentum of the target encoder, annealed to 1 over
        training. None uses 0.996 for 'ijepa' and 'tjepa' and 0.998 for
        'tsjepa'. Not used when ``variant='lejepa'``.

    recon_weight : float, optional (default=0.0)
        Weight of an auxiliary loss that linearly reconstructs each raw
        patch from its embedding. 0 disables it. Only available for
        'lejepa' and 'ijepa'.

    score : str, optional (default='pred')
        Anomaly score, one of 'pred', 'embed' and 'combined' as
        described above. The embedding distance is sensitive to a shift
        between the training and the test series, so 'pred' is the
        default.

    score_aggregation : str, optional (default='mean')
        How the scores of all (window, mask, patch) combinations that
        cover a timestamp are aggregated, 'mean' or 'max'.

    epochs : int, optional (default=10)
        Number of training epochs.

    lr : float, optional (default=1e-3)
        Learning rate of the AdamW optimizer, decayed with a cosine
        schedule.

    weight_decay : float, optional (default=1e-4)
        Weight decay of the AdamW optimizer.

    batch_size : int, optional (default=64)
        Mini-batch size for training.

    grad_clip : float or None, optional (default=1.0)
        Maximum gradient norm. None or 0 disables clipping.

    encoder_norm : bool or None, optional (default=None)
        If True, normalise the encoder output. None uses False for
        'lejepa' and 'ijepa' and True, the published form, for 'tsjepa'
        and 'tjepa'. A normalised output lets shifted or extreme inputs
        saturate to a direction that is easy to predict, which degrades
        the scores when the test series follows a different operating
        regime than the training series.

    device : str, optional (default='auto')
        Device to use ('cpu', 'cuda', 'mps', or 'auto').
        'auto' picks CUDA when available and the CPU otherwise.

    random_state : int, optional (default=42)
        Random seed for the initial weights, the training order, the
        training masks and the scoring masks.

    verbose : int, optional (default=0)
        If non-zero, print the latent prediction loss after every epoch.

    contamination : float, optional (default=0.1)
        Expected proportion of outliers. Must be in (0, 0.5].

    Attributes
    ----------
    model_ : torch.nn.Module
        The fitted JEPA model.

    encoder_norm_ : bool
        The value of ``encoder_norm`` used by the fitted model.

    ema_momentum_ : float or None
        The initial target-encoder momentum used during fit. None for
        ``variant='lejepa'``.

    n_channels_ : int
        Number of channels seen during fit.

    X_mean_ : numpy array of shape (n_channels,)
        Per-channel mean of the training series, used to standardise
        the input.

    X_std_ : numpy array of shape (n_channels,)
        Per-channel standard deviation of the training series.

    decision_scores_ : numpy array of shape (n_timestamps,)
        Outlier scores of the training data. Higher is more abnormal.
        Timestamps not covered by any hidden patch are filled with
        ``threshold_``.

    threshold_ : float
        Score threshold based on ``contamination``.

    labels_ : numpy array of shape (n_timestamps,)
        Binary labels of training data (0: inlier, 1: outlier).

    Examples
    --------
    >>> from pyod.models.ts_jepa import TSJEPA
    >>> from pyod.utils.data import generate_ts_data
    >>> X_train, X_test, _, _ = generate_ts_data(
    ...     n_train=1000, n_test=400, random_state=42)
    >>> clf = TSJEPA(window_size=40, patch_len=5, epochs=5)
    >>> clf.fit(X_train)
    >>> scores = clf.decision_function(X_test)

    References
    ----------
    .. [1] LeCun, Y. (2022). A Path Towards Autonomous Machine
       Intelligence. OpenReview.
    .. [2] Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P.,
       Rabbat, M., LeCun, Y., & Ballas, N. (2023). Self-Supervised
       Learning from Images with a Joint-Embedding Predictive
       Architecture. CVPR 2023.
    .. [3] Balestriero, R., & LeCun, Y. (2025). LeJEPA: Provable and
       Scalable Self-Supervised Learning Without the Heuristics.
       arXiv:2511.08544.
    .. [4] Ennadir, S., Golkar, S., & Sarra, L. (2025). Joint Embeddings
       Go Temporal. arXiv:2509.25449.
    .. [5] Thimonier, H., De Melo Costa, J. L., Popineau, F., Rimmel, A.,
       & Doan, B.-L. (2025). T-JEPA: Augmentation-Free Self-Supervised
       Learning for Tabular Data. ICLR 2025.
    """

    def __init__(self, window_size=100, patch_len=10, step=1,
                 train_step=None, max_train_windows=5000,
                 d_model=64, n_heads=4, n_layers=2, pred_layers=1,
                 dropout=0.0,
                 mask_mode='block', mask_ratio=0.3, n_mask_samples=None,
                 variant='lejepa', sigreg_weight=0.05,
                 sigreg_slices=128, ema_momentum=None, recon_weight=0.0,
                 score='pred', score_aggregation='mean',
                 epochs=10, lr=1e-3, weight_decay=1e-4, batch_size=64,
                 grad_clip=1.0, encoder_norm=None, device='auto',
                 random_state=42, verbose=0, contamination=0.1):
        super(TSJEPA, self).__init__(contamination=contamination)
        self.window_size = window_size
        self.patch_len = patch_len
        self.step = step
        self.train_step = train_step
        self.max_train_windows = max_train_windows
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.pred_layers = pred_layers
        self.dropout = dropout
        self.mask_mode = mask_mode
        self.mask_ratio = mask_ratio
        self.n_mask_samples = n_mask_samples
        self.variant = variant
        self.sigreg_weight = sigreg_weight
        self.sigreg_slices = sigreg_slices
        self.ema_momentum = ema_momentum
        self.recon_weight = recon_weight
        self.score = score
        self.score_aggregation = score_aggregation
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.encoder_norm = encoder_norm
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _get_min_length(self):
        """Return the minimum time series length required.

        One patch beyond a full window, so that there are several
        training windows.

        Returns
        -------
        min_length : int
        """
        return self.window_size + self.patch_len

    def _resolve_device(self):
        """Resolve the device string to a torch.device."""
        torch = _require_torch()
        if self.device == 'auto':
            return torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)

    def _validate_params(self):
        """Check the constructor parameters that can be checked early."""
        from ._jepa_utils import MASK_MODES

        if self.patch_len < 1 or self.window_size % self.patch_len != 0:
            raise ValueError(
                "window_size (%d) must be a multiple of patch_len (%d)"
                % (self.window_size, self.patch_len))
        if self.window_size // self.patch_len < 2:
            raise ValueError(
                "window_size (%d) must hold at least 2 patches of "
                "patch_len (%d)" % (self.window_size, self.patch_len))
        if self.variant not in _VARIANTS:
            raise ValueError("variant must be one of %s; got %r"
                             % (_VARIANTS, self.variant))
        if self.recon_weight > 0 and self.variant not in ('lejepa',
                                                          'ijepa'):
            raise ValueError("recon_weight is only available for the "
                             "'lejepa' and 'ijepa' variants; got "
                             "variant=%r" % (self.variant,))
        if self.mask_mode not in MASK_MODES:
            raise ValueError("mask_mode must be one of %s; got %r"
                             % (MASK_MODES, self.mask_mode))
        if self.score not in _SCORES:
            raise ValueError("score must be one of %s; got %r"
                             % (_SCORES, self.score))
        if self.score_aggregation not in _AGGREGATIONS:
            raise ValueError("score_aggregation must be one of %s; got %r"
                             % (_AGGREGATIONS, self.score_aggregation))
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1); got %r"
                             % (self.mask_ratio,))

    @staticmethod
    def _windows(X, window_size, step):
        """Cut sliding windows.

        Parameters
        ----------
        X : np.ndarray of shape (n_timestamps, n_channels)
            Time series data.

        window_size : int
            Window length.

        step : int
            Step between consecutive windows.

        Returns
        -------
        windows : np.ndarray of shape (n_windows, window_size, n_channels)
        """
        n_windows = (X.shape[0] - window_size) // step + 1
        idx = (np.arange(window_size)[None, :]
               + step * np.arange(n_windows)[:, None])
        return np.ascontiguousarray(X[idx], dtype=np.float32)

    def _to_patches(self, windows):
        """Reshape (n_windows, window_size, C) to
        (n_windows, n_patches, patch_len * C)."""
        n_windows, window_size, n_channels = windows.shape
        return windows.reshape(n_windows, window_size // self.patch_len,
                               self.patch_len * n_channels)

    def _build_model(self, n_tokens, n_channels):
        """Build the network of ``variant``.

        Parameters
        ----------
        n_tokens : int
            Patches per window.

        n_channels : int
            Channels of the series.

        Returns
        -------
        model : torch.nn.Module
            A network with the ``forward`` and ``multi_mask_errors``
            interface of ``_jepa_utils.JEPACore``.
        """
        from ._jepa_utils import JEPACore, TJEPACore, TSJEPACore

        token_dim = self.patch_len * n_channels
        size = dict(d_model=self.d_model, n_heads=self.n_heads,
                    n_layers=self.n_layers, pred_layers=self.pred_layers,
                    dropout=self.dropout, encoder_norm=self.encoder_norm_)
        if self.variant == 'tsjepa':
            return TSJEPACore(n_tokens=n_tokens, patch_len=self.patch_len,
                              n_channels=n_channels, **size)
        if self.variant == 'tjepa':
            return TJEPACore(n_tokens=n_tokens, in_dim=token_dim, **size)
        return JEPACore(
            n_tokens=n_tokens, in_dim=token_dim,
            target_mode='sigreg' if self.variant == 'lejepa' else 'ema',
            recon_dim=token_dim if self.recon_weight > 0 else None, **size)

    def fit(self, X, y=None):
        """Fit detector on time series data.

        Parameters
        ----------
        X : array-like of shape (n_timestamps,) or (n_timestamps, n_channels)
            Training time series data.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        torch = _require_torch()

        self._validate_params()
        X = validate_ts_input(X)
        n_timestamps, n_channels = X.shape
        min_len = self._get_min_length()
        if n_timestamps < min_len:
            raise ValueError(
                "Time series length %d is shorter than minimum "
                "required length %d (window_size=%d)"
                % (n_timestamps, min_len, self.window_size))

        self._set_n_classes(y)

        torch.manual_seed(self.random_state)
        rng = np.random.RandomState(self.random_state)
        device = self._resolve_device()

        # Step 1: standardise every channel with training statistics
        self.n_channels_ = n_channels
        self.X_mean_ = X.mean(axis=0)
        self.X_std_ = X.std(axis=0) + 1e-8
        X_std = (X - self.X_mean_) / self.X_std_

        n_patches = self.window_size // self.patch_len
        self.n_tokens_ = n_patches
        self.n_mask_ = int(np.clip(round(self.mask_ratio * n_patches),
                                   1, n_patches - 1))
        norm, momentum = _VARIANT_DEFAULTS[self.variant]
        self.encoder_norm_ = (norm if self.encoder_norm is None
                              else bool(self.encoder_norm))
        self.ema_momentum_ = (momentum if self.ema_momentum is None
                              else self.ema_momentum)

        # Step 2: training windows, subsampled to the training budget
        train_step = self.train_step or max(1, self.patch_len // 2)
        windows = self._windows(X_std, self.window_size, train_step)
        if len(windows) > self.max_train_windows:
            keep = rng.choice(len(windows), self.max_train_windows,
                              replace=False)
            windows = windows[np.sort(keep)]
        train_patches = torch.from_numpy(self._to_patches(windows))

        # Step 3: build and train the JEPA, then fit the scoring stats
        self.model_ = self._build_model(n_patches, n_channels).to(device)
        self._train_model(train_patches, device)
        self._calibrate(train_patches, device)

        # Step 4: masked-score workflow -- compute the threshold on the
        # timestamps that received a score, then fill the others.
        scores, valid = self._score_series(X_std, device)
        self.decision_scores_ = scores[valid]
        self._process_decision_scores()

        full_scores = scores.copy()
        full_scores[~valid] = self.threshold_
        self.decision_scores_ = full_scores
        self.labels_ = (full_scores > self.threshold_).astype(int)

        return self

    def decision_function(self, X):
        """Predict raw anomaly scores for time series X.

        Parameters
        ----------
        X : array-like of shape (n_timestamps,) or (n_timestamps, n_channels)
            Test time series data. Must have at least ``window_size``
            timestamps and the same number of channels as the training
            data.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_timestamps,)
            Anomaly scores. Higher is more abnormal. Timestamps not
            covered by any hidden patch are filled with ``threshold_``.
        """
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])

        X = validate_ts_input(X)
        if X.shape[1] != self.n_channels_:
            raise ValueError("Expected %d channels, got %d"
                             % (self.n_channels_, X.shape[1]))
        if X.shape[0] < self.window_size:
            raise ValueError(
                "Time series length %d is shorter than window_size %d"
                % (X.shape[0], self.window_size))

        X_std = (X - self.X_mean_) / self.X_std_
        scores, valid = self._score_series(X_std, self._resolve_device())
        scores[~valid] = self.threshold_
        return scores

    def _train_model(self, patches, device):
        """Train ``model_`` on the training patches.

        Parameters
        ----------
        patches : torch.Tensor of shape (n_windows, n_patches, token_dim)
            Training windows cut into patches.

        device : torch.device
            Device to train on.
        """
        torch = _require_torch()
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
        from ._jepa_utils import ema_update, sample_masks, sigreg_loss

        model = self.model_
        gen = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(
            TensorDataset(patches), batch_size=self.batch_size,
            shuffle=True, drop_last=len(patches) >= self.batch_size,
            generator=gen)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.epochs), eta_min=self.lr * 0.05)
        total_steps = max(1, self.epochs * len(loader))
        step = 0

        model.train()
        for epoch in range(self.epochs):
            epoch_losses = []
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                ctx, tgt = sample_masks(batch_x.size(0), self.n_tokens_,
                                        self.n_mask_, self.mask_mode, gen)
                ctx, tgt = ctx.to(device), tgt.to(device)

                optimizer.zero_grad()
                pred, h_tgt, h_all = model(batch_x, ctx, tgt)
                # The targets are detached: the encoder must not lower
                # the loss by making its own targets easier to predict.
                if self.variant == 'tsjepa':
                    loss_pred = F.l1_loss(pred, h_tgt.detach())
                else:
                    loss_pred = F.mse_loss(pred, h_tgt.detach())
                loss = loss_pred
                if self.variant == 'lejepa':
                    # one group per patch position: (P, B, D)
                    reg = sigreg_loss(h_all.transpose(0, 1),
                                      num_slices=self.sigreg_slices,
                                      generator=gen)
                    loss = (1 - self.sigreg_weight) * loss_pred \
                           + self.sigreg_weight * reg
                if self.recon_weight > 0:
                    loss = loss + self.recon_weight * F.mse_loss(
                        model.recon_head(h_all), batch_x)
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   self.grad_clip)
                optimizer.step()

                if self.variant != 'lejepa':
                    frac = min(1.0, step / total_steps)
                    momentum = self.ema_momentum_ \
                               + (1 - self.ema_momentum_) * frac
                    ema_update(model.target_encoder, model.encoder,
                               momentum)
                step += 1
                epoch_losses.append(loss_pred.item())
            scheduler.step()
            if self.verbose:
                print('Epoch %d/%d, latent prediction loss=%.4f'
                      % (epoch + 1, self.epochs,
                         float(np.mean(epoch_losses))))
        model.eval()

    def _eval_mask_set(self):
        """Return the seeded (context, target) mask set used for scoring."""
        torch = _require_torch()
        from ._jepa_utils import eval_masks

        gen = torch.Generator().manual_seed(self.random_state + 1)
        return eval_masks(self.n_tokens_, self.n_mask_, self.mask_mode,
                          self.n_mask_samples, gen)

    def _raw(self, patches, device, ctx, tgt):
        """Yield latent errors and embeddings batch by batch.

        Parameters
        ----------
        patches : torch.Tensor of shape (n_windows, n_patches, token_dim)
            Windows cut into patches.

        device : torch.device
            Device to run the model on.

        ctx : torch.LongTensor of shape (n_masks, n_patches - n_mask)
            Visible patch indices per mask.

        tgt : torch.LongTensor of shape (n_masks, n_mask)
            Hidden patch indices per mask.

        Yields
        ------
        err : np.ndarray of shape (batch, n_masks, n_mask)
            Latent prediction error per mask and hidden patch.

        h_all : np.ndarray of shape (batch, n_patches, d_model)
            Patch embeddings.
        """
        self.model_.eval()
        batch_size = max(8, self.batch_size)
        for start in range(0, len(patches), batch_size):
            batch_x = patches[start:start + batch_size].to(device)
            # multi_mask_errors runs without gradients
            err, h_all = self.model_.multi_mask_errors(batch_x, ctx, tgt)
            yield err.cpu().numpy(), h_all.cpu().numpy()

    def _calibrate(self, patches, device):
        """Fit the scoring statistics on the training patches.

        Sets the latent error statistics (``err_mean_``, ``err_std_``),
        the embedding Gaussian (``emb_mean_``, ``emb_prec_``) and the
        statistics used to z-normalise the two component scores
        (``comp_mean_``, ``comp_std_``).
        """
        ctx, tgt = self._eval_mask_set()
        errors, embeddings = [], []
        for err, h_all in self._raw(patches, device, ctx, tgt):
            errors.append(err.reshape(-1))
            embeddings.append(h_all.reshape(-1, h_all.shape[-1]))
        errors = np.concatenate(errors)
        self.err_mean_ = float(errors.mean())
        self.err_std_ = float(max(errors.std(), 1e-8))

        embeddings = np.concatenate(embeddings, axis=0)
        self.emb_mean_ = embeddings.mean(axis=0)
        cov = np.atleast_2d(np.cov(embeddings, rowvar=False))
        ridge = 1e-3 * max(np.trace(cov) / cov.shape[0], 1e-6)
        self.emb_prec_ = np.linalg.inv(cov + ridge * np.eye(cov.shape[0]))

        s_pred, s_emb = [], []
        for err, h_all in self._raw(patches, device, ctx, tgt):
            s_pred.append(
                ((err - self.err_mean_) / self.err_std_).reshape(-1))
            s_emb.append(self._mahalanobis(h_all).reshape(-1))
        s_pred, s_emb = np.concatenate(s_pred), np.concatenate(s_emb)
        self.comp_mean_ = np.array([s_pred.mean(), s_emb.mean()])
        self.comp_std_ = np.maximum(
            np.array([s_pred.std(), s_emb.std()]), 1e-8)

    def _mahalanobis(self, h_all):
        """Squared Mahalanobis distance of embeddings (..., d_model) to
        the training Gaussian."""
        diff = h_all - self.emb_mean_
        return np.einsum('...i,ij,...j->...', diff, self.emb_prec_, diff)

    def _score_series(self, X_std, device):
        """Score every timestamp of a standardised series.

        Parameters
        ----------
        X_std : np.ndarray of shape (n_timestamps, n_channels)
            Standardised time series.

        device : torch.device
            Device to run the model on.

        Returns
        -------
        scores : np.ndarray of shape (n_timestamps,)
            Anomaly scores; NaN where no score is available.

        valid : np.ndarray of shape (n_timestamps,), dtype bool
            True where a score is available.
        """
        torch = _require_torch()

        n_timestamps = X_std.shape[0]
        windows = self._windows(X_std, self.window_size, self.step)
        patches = torch.from_numpy(self._to_patches(windows))
        patch_len = self.patch_len
        ctx, tgt = self._eval_mask_set()
        tgt_np = tgt.numpy()  # (K, m)
        use_max = self.score_aggregation == 'max'
        init = -np.inf if use_max else 0.0
        acc_pred = np.full(n_timestamps, init)
        cnt_pred = np.zeros(n_timestamps)
        acc_emb = np.full(n_timestamps, init)
        cnt_emb = np.zeros(n_timestamps)
        in_patch = np.arange(patch_len)
        patch_offsets = np.arange(self.n_tokens_) * patch_len  # (P,)

        start = 0
        for err, h_all in self._raw(patches, device, ctx, tgt):
            batch = err.shape[0]
            # first timestamp of every window in the batch
            w0 = np.arange(start, start + batch) * self.step
            start += batch
            if self.score in ('pred', 'combined'):
                z = (err - self.err_mean_) / self.err_std_  # (B, K, m)
                base = w0[:, None, None] + tgt_np[None] * patch_len
                idx = (base[..., None] + in_patch).reshape(-1)
                val = np.broadcast_to(
                    z[..., None], z.shape + (patch_len,)).reshape(-1)
                self._accumulate(acc_pred, cnt_pred, idx, val, use_max)
            if self.score in ('embed', 'combined'):
                dist = self._mahalanobis(h_all)  # (B, P)
                base = w0[:, None] + patch_offsets[None]
                idx = (base[..., None] + in_patch).reshape(-1)
                val = np.broadcast_to(
                    dist[..., None], dist.shape + (patch_len,)).reshape(-1)
                self._accumulate(acc_emb, cnt_emb, idx, val, use_max)

        def _finish(acc, cnt):
            out = np.full(n_timestamps, np.nan)
            ok = cnt > 0
            out[ok] = acc[ok] if use_max else acc[ok] / cnt[ok]
            return out, ok

        if self.score == 'pred':
            return _finish(acc_pred, cnt_pred)
        if self.score == 'embed':
            return _finish(acc_emb, cnt_emb)

        s_pred, ok_pred = _finish(acc_pred, cnt_pred)
        s_emb, ok_emb = _finish(acc_emb, cnt_emb)
        z_pred = (s_pred - self.comp_mean_[0]) / self.comp_std_[0]
        z_emb = (s_emb - self.comp_mean_[1]) / self.comp_std_[1]
        scores = np.where(ok_pred & ok_emb, 0.5 * (z_pred + z_emb),
                          np.where(ok_pred, z_pred, z_emb))
        return scores, ok_pred | ok_emb

    @staticmethod
    def _accumulate(acc, cnt, idx, val, use_max):
        """Scatter ``val`` into ``acc`` at ``idx`` (sum or max) and
        count the contributions in ``cnt``."""
        if use_max:
            np.maximum.at(acc, idx, val)
        else:
            np.add.at(acc, idx, val)
        np.add.at(cnt, idx, 1.0)
