# -*- coding: utf-8 -*-
"""The further JEPA versions of the time series detector.

:class:`pyod.models.ts_jepa.TSJEPA` trains its network with the JEPA
named by ``variant``. The first four versions (``'lejepa'``,
``'ijepa'``, ``'tsjepa'``, ``'tjepa'``) are implemented in
``ts_jepa.py`` and ``_jepa_utils.py``. This module holds the twelve
others, each written from the paper named in its docstring, in three
parts:

* the anti-collapse regularisers that are not SIGReg;
* the networks, which offer the scoring interface of
  :class:`pyod.models._jepa_utils.JEPACore` (``multi_mask_errors``).
  Networks that do not hide patches also define ``eval_mask_set``, the
  single (context, target) split they are scored with;
* the recipes: one object per version that builds its network and
  computes its training loss (:data:`RECIPES`).

A version contributes its network, its target mechanism and its loss.
The masking, the model sizes, the optimizer, the training budget and
the anomaly scores stay the detector's own.

This module imports torch at load time; the detector imports it lazily.
"""
# Author: Kishor Datta Gupta
# License: BSD 2 clause

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._jepa_utils import (JEPACore, Tokenizer, TransformerStack,
                          _gather_tokens, sample_masks, sigreg_loss)


# ----------------------------------------------------------------------
# Regularisers
# ----------------------------------------------------------------------
# All take embeddings of shape (N, D) or (G, N, D). With a leading group
# dimension (one group per patch position) the term is computed per
# group and averaged, as sigreg_loss does.
def _groups(z):
    return z.unsqueeze(0) if z.dim() == 2 else z


def _covariance(z):
    """(G, N, D) -> (G, D, D) unbiased covariance over the N samples."""
    zc = z - z.mean(dim=1, keepdim=True)
    return zc.transpose(1, 2) @ zc / max(1, z.size(1) - 1)


def variance_loss(z, gamma=1.0, eps=1e-4):
    """Variance term of VICReg.

    A hinge on the standard deviation of every embedding dimension:
    ``mean_j max(0, gamma - sqrt(Var(z_j) + eps))``.

    Parameters
    ----------
    z : torch.Tensor of shape (N, D) or (G, N, D)
        Embeddings.

    gamma : float, optional (default=1.0)
        Standard deviation below which a dimension is penalised.

    eps : float, optional (default=1e-4)
        Added to the variance for a finite gradient at zero.

    Returns
    -------
    loss : torch.Tensor (scalar)
    """
    z = _groups(z)
    std = torch.sqrt(z.var(dim=1) + eps)
    return F.relu(gamma - std).mean()


def covariance_loss(z, pairwise=False):
    """Covariance term of VICReg: the squared off-diagonal covariances.

    Parameters
    ----------
    z : torch.Tensor of shape (N, D) or (G, N, D)
        Embeddings.

    pairwise : bool, optional (default=False)
        If False the sum is divided by ``D``, as in VICReg, C-JEPA and
        CF-JEPA. If True it is divided by ``D * (D - 1)``, as in
        EB-JEPA.

    Returns
    -------
    loss : torch.Tensor (scalar)
    """
    z = _groups(z)
    n_dims = z.size(-1)
    cov = _covariance(z)
    off = cov.pow(2).sum(dim=(1, 2)) \
        - cov.diagonal(dim1=1, dim2=2).pow(2).sum(1)
    return (off / (n_dims * max(1, n_dims - 1) if pairwise
                   else n_dims)).mean()


def cw_mmd(z, sigma=1.0, gamma=None):
    """Cramer-Wold MMD between the embeddings and ``N(0, sigma^2 I)``.

    The closed form of KerJEPA: the Gaussian-kernel MMD of every 1-D
    projection, averaged over all directions, with the average over
    directions replaced by its large-dimension inverse-multiquadric
    approximation. No random projections are drawn. The estimate of the
    first term leaves out the diagonal, so the value is about zero for
    Gaussian embeddings and can be slightly negative.

    Parameters
    ----------
    z : torch.Tensor of shape (N, D) or (G, N, D)
        Embeddings.

    sigma : float, optional (default=1.0)
        Standard deviation of the Gaussian prior.

    gamma : float or None, optional (default=None)
        Kernel bandwidth parameter. None follows Silverman's rule for
        the group size ``N``.

    Returns
    -------
    loss : torch.Tensor (scalar)
    """
    z = _groups(z)
    n_samples, n_dims = z.shape[1], z.shape[2]
    if n_samples < 2:
        return z.sum() * 0.0
    if gamma is None:
        h = (4.0 / (3.0 * n_samples)) ** 0.2 * sigma
        gamma = 1.0 / (2.0 * h * h)
    a = 4.0 * gamma / max(1, 2 * n_dims - 3)
    s = 1.0 + 2.0 * gamma * sigma ** 2
    sq = z.pow(2).sum(-1)  # (G, N)
    d2 = (sq.unsqueeze(2) + sq.unsqueeze(1)
          - 2.0 * z @ z.transpose(1, 2)).clamp_min(0.0)  # (G, N, N)
    k = (1.0 + a * d2).rsqrt()
    within = (k.sum(dim=(1, 2)) - k.diagonal(dim1=1, dim2=2).sum(1)) \
        / (n_samples * (n_samples - 1))
    cross = (1.0 + (a / s) * sq).rsqrt().mean(1) / math.sqrt(s)
    const = 1.0 / math.sqrt(1.0 + 4.0 * gamma * sigma ** 2)
    return (within - 2.0 * cross + const).mean()


def rectified_gn_sample(shape, p=1.0, mean_shift=0.0, generator=None):
    """Samples of the target distribution of Rectified LpJEPA.

    ``relu(mean_shift + e)`` with ``e`` of unit variance: Laplace for
    ``p = 1`` and Gaussian for ``p = 2``.

    Parameters
    ----------
    shape : tuple of int
        Shape of the sample.

    p : float, optional (default=1.0)
        1.0 or 2.0.

    mean_shift : float, optional (default=0.0)
        Location before the rectification.

    generator : torch.Generator or None, optional (default=None)
        CPU generator.

    Returns
    -------
    sample : torch.Tensor of shape ``shape``, on the CPU
    """
    if p == 2.0:
        e = torch.randn(shape, generator=generator)
    elif p == 1.0:
        u = torch.rand(shape, generator=generator).clamp(1e-7, 1 - 1e-7) \
            - 0.5
        e = -torch.sign(u) * torch.log1p(-2.0 * u.abs()) / math.sqrt(2.0)
    else:
        raise ValueError("p must be 1.0 or 2.0; got %r" % (p,))
    return F.relu(mean_shift + e)


def rdm_reg(z, p=1.0, mean_shift=0.0, num_slices=128, generator=None):
    """RDMReg of Rectified LpJEPA.

    A sliced two-sample match between the rectified embeddings and
    samples of :func:`rectified_gn_sample`: both are projected on random
    unit directions, every projection is sorted over the samples, and
    the squared differences are averaged.

    Parameters
    ----------
    z : torch.Tensor of shape (N, D) or (G, N, D)
        Non-negative embeddings.

    p : float, optional (default=1.0)
        Shape of the target distribution, 1.0 or 2.0.

    mean_shift : float, optional (default=0.0)
        Location of the target distribution before the rectification.

    num_slices : int, optional (default=128)
        Number of random unit directions, resampled on every call.

    generator : torch.Generator or None, optional (default=None)
        CPU generator for the directions and the target samples.

    Returns
    -------
    loss : torch.Tensor (scalar)
    """
    z = _groups(z)
    directions = torch.randn(z.size(-1), num_slices, generator=generator)
    directions = (directions
                  / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
                  ).to(z)
    y = rectified_gn_sample(tuple(z.shape), p, mean_shift, generator).to(z)
    z_sorted = (z @ directions).sort(dim=1).values
    y_sorted = (y @ directions).sort(dim=1).values
    return (z_sorted - y_sorted).pow(2).mean()


def sac_reg(z, eps=1e-4, generator=None):
    """SACReg of lambda-JEPA.

    The divergence between the Gaussian fitted to the embeddings and
    ``N(0, I)``, per dimension:
    ``0.5 * (|mu|^2 + tr(S) - logdet(S + eps I) - D) / D``. With fewer
    than ``2 D`` samples the embeddings are first projected on
    ``N // 2`` random orthonormal directions so that the covariance has
    full rank.

    Parameters
    ----------
    z : torch.Tensor of shape (N, D)
        Embeddings.

    eps : float, optional (default=1e-4)
        Ridge added to the covariance.

    generator : torch.Generator or None, optional (default=None)
        CPU generator for the random projection.

    Returns
    -------
    loss : torch.Tensor (scalar)
    """
    n_samples, n_dims = z.shape
    if n_samples < 2:
        return z.sum() * 0.0
    if n_samples < 2 * n_dims:
        n_dims = max(1, n_samples // 2)
        q, _ = torch.linalg.qr(
            torch.randn(z.size(1), n_dims, generator=generator))
        z = z @ q.to(z)
    mu = z.mean(0)
    cov = _covariance(z.unsqueeze(0))[0]
    cov = cov + eps * torch.eye(n_dims, device=z.device, dtype=z.dtype)
    logdet = torch.linalg.eigvalsh(cov).clamp_min(1e-12).log().sum()
    return 0.5 * (mu.pow(2).sum() + cov.diagonal().sum() - logdet
                  - n_dims) / n_dims


# ----------------------------------------------------------------------
# Networks
# ----------------------------------------------------------------------
def _frozen_copy(module):
    """A copy of ``module`` whose parameters take no gradient."""
    target = copy.deepcopy(module)
    for p in target.parameters():
        p.requires_grad_(False)
    return target


def _repeat_masks(x, ctx_idx, tgt_idx):
    """Repeat the rows of ``x`` once per mask of a shared mask set.

    Returns ``x`` of shape (B * K, ...) and the matching index tensors
    of shape (B * K, c) and (B * K, m).
    """
    batch_size = x.size(0)
    ctx = ctx_idx.to(x.device).repeat(batch_size, 1)
    tgt = tgt_idx.to(x.device).repeat(batch_size, 1)
    return x.repeat_interleave(tgt_idx.size(0), dim=0), ctx, tgt


class CJEPACore(JEPACore):
    """The network of the ``'cjepa'`` version.

    The network of ``'ijepa'`` with the projector on which C-JEPA (Mo
    and Tong, arXiv:2410.19560) computes its VICReg terms.

    Parameters
    ----------
    n_tokens : int
        Patches per window.

    in_dim : int
        Raw values per patch.

    kwargs : dict
        Sizes, as for :class:`pyod.models._jepa_utils.JEPACore`.
    """

    def __init__(self, n_tokens, in_dim, **kwargs):
        super().__init__(n_tokens, in_dim, target_mode='ema', **kwargs)
        d_model = self.d_model
        self.projector = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(),
            nn.Linear(2 * d_model, d_model))

    def forward_views(self, x, ctx_idx, tgt_idx):
        """Training pass that also returns the projected mean context
        embedding, of shape (B, D), as fourth value."""
        tokens = self.embed(x)
        h_ctx = self.encoder(_gather_tokens(tokens, ctx_idx))
        h_all = self.encode_target(tokens)
        pred = self.predict(h_ctx, ctx_idx, tgt_idx)
        return (pred, _gather_tokens(h_all, tgt_idx), h_all,
                self.projector(h_ctx.mean(1)))


class StochasticJEPACore(JEPACore):
    """The network of the ``'sjepa'`` version.

    The network of ``'ijepa'`` whose predictor also receives a Gaussian
    latent, after Giakoumoglou et al. (arXiv:2611.09302): a prior
    ``p(z | context)`` and a posterior ``q(z | context, target)``, the
    latent enters the predictor as a prefix token, and the loss adds
    ``KL(q || p)``. Scoring uses the prior mean, that is, only what the
    context says.

    Parameters
    ----------
    n_tokens : int
        Patches per window.

    in_dim : int
        Raw values per patch.

    latent_dim : int or None, optional (default=None)
        Width of the latent. None uses ``max(8, d_model // 4)``.

    kwargs : dict
        Sizes, as for :class:`pyod.models._jepa_utils.JEPACore`.
    """

    def __init__(self, n_tokens, in_dim, latent_dim=None, **kwargs):
        super().__init__(n_tokens, in_dim, target_mode='ema', **kwargs)
        d_model = self.d_model
        self.latent_dim = int(latent_dim or max(8, d_model // 4))
        self.prior_net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 2 * self.latent_dim))
        self.post_net = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 2 * self.latent_dim))
        self.latent_in = nn.Linear(self.latent_dim,
                                   self.pred_in.out_features)

    @staticmethod
    def _gaussian(out):
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 8.0)

    def prior(self, h_ctx):
        """Mean and log-variance of ``p(z | context)``."""
        return self._gaussian(self.prior_net(h_ctx.mean(1)))

    def posterior(self, h_ctx, h_tgt):
        """Mean and log-variance of ``q(z | context, target)``."""
        return self._gaussian(self.post_net(
            torch.cat([h_ctx.mean(1), h_tgt.mean(1)], dim=-1)))

    def predict(self, h_ctx, ctx_idx, tgt_idx, z=None):
        """Predict target-patch representations from encoded context
        and the latent ``z`` (the prior mean when None)."""
        if z is None:
            z = self.prior(h_ctx)[0]
        c = self.pred_in(h_ctx) + self.pred_pos[ctx_idx]
        q = self.mask_token + self.pred_pos[tgt_idx]
        out = self.predictor(
            torch.cat([self.latent_in(z).unsqueeze(1), c, q], dim=1))
        return self.pred_out(out[:, 1 + c.size(1):])

    def forward_latent(self, x, ctx_idx, tgt_idx, noise):
        """Training pass with a posterior sample.

        Parameters
        ----------
        x : torch.Tensor of shape (B, T, in_dim)
            Raw tokens.

        ctx_idx : torch.LongTensor of shape (B, T - m)
            Visible token indices per row.

        tgt_idx : torch.LongTensor of shape (B, m)
            Hidden token indices per row.

        noise : torch.Tensor of shape (B, latent_dim)
            Standard normal draws.

        Returns
        -------
        pred, h_tgt, h_all : torch.Tensor
            As :meth:`forward`.

        kl : torch.Tensor (scalar)
            ``KL(q || p)`` summed over the latent dimensions and
            averaged over the batch.
        """
        tokens = self.embed(x)
        h_ctx = self.encoder(_gather_tokens(tokens, ctx_idx))
        h_all = self.encode_target(tokens)
        h_tgt = _gather_tokens(h_all, tgt_idx)
        mu_p, lv_p = self.prior(h_ctx)
        mu_q, lv_q = self.posterior(h_ctx, h_tgt)
        z = mu_q + noise * torch.exp(0.5 * lv_q)
        pred = self.predict(h_ctx, ctx_idx, tgt_idx, z)
        kl = 0.5 * (lv_p - lv_q + (lv_q.exp() + (mu_q - mu_p).pow(2))
                    / lv_p.exp() - 1.0).sum(-1).mean()
        return pred, h_tgt, h_all, kl


class DenseJEPACore(nn.Module):
    """The network of the ``'vjepa21'`` version.

    After V-JEPA 2.1 (arXiv:2603.14482): the outputs of several encoder
    depths are fused before the predictor, the predictor returns one
    prediction per depth for every token, the visible ones included,
    and the targets are the outputs of the EMA encoder at the same
    depths.

    Parameters
    ----------
    n_tokens : int
        Patches per window.

    in_dim : int
        Raw values per patch.

    d_model, n_heads, n_layers, pred_layers, dropout, encoder_norm
        As for :class:`pyod.models._jepa_utils.JEPACore`.

    max_levels : int, optional (default=4)
        Number of depths used, equally spaced and ending at the last
        layer; never more than ``n_layers``.
    """

    def __init__(self, n_tokens, in_dim, d_model=64, n_heads=4, n_layers=2,
                 pred_layers=1, dropout=0.0, encoder_norm=False,
                 max_levels=4):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.n_layers = n_layers
        n_levels = max(1, min(int(max_levels), n_layers))
        self.levels = sorted({int(round(n_layers * (i + 1) / n_levels))
                              for i in range(n_levels)})
        pred_dim = max(16, d_model // 2)
        pred_heads = max(1, min(n_heads, pred_dim // 16))

        self.tokenizer = Tokenizer(n_tokens, in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.encoder = TransformerStack(d_model, n_heads, n_layers, dropout,
                                        final_norm=encoder_norm)
        self.target_encoder = _frozen_copy(self.encoder)
        self.fuse = nn.Sequential(
            nn.Linear(len(self.levels) * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, pred_dim))
        self.mask_token = nn.Parameter(torch.randn(pred_dim) * 0.02)
        self.pred_pos = nn.Parameter(torch.randn(n_tokens, pred_dim) * 0.02)
        self.predictor = TransformerStack(pred_dim, pred_heads, pred_layers,
                                          dropout)
        self.pred_out = nn.ModuleList(
            [nn.Linear(pred_dim, d_model) for _ in self.levels])

    def embed(self, x):
        """Raw tokens (B, T, in_dim) -> positional token embeddings."""
        return self.tokenizer(x) + self.pos

    def _level_outputs(self, stack, tokens):
        """Layer-normalised outputs of ``stack`` after the chosen
        depths."""
        outs, h = [], tokens
        for i, layer in enumerate(stack.layers.layers, 1):
            h = layer(h)
            if i in self.levels:
                o = stack.norm(h) if i == self.n_layers else h
                outs.append(F.layer_norm(o, (o.size(-1),)))
        return outs

    @torch.no_grad()
    def target_levels(self, tokens):
        """EMA embeddings of all tokens, one (B, T, D) per depth."""
        return self._level_outputs(self.target_encoder, tokens)

    def predict_levels(self, ctx_levels, ctx_idx, tgt_idx):
        """One (B, c + m, D) prediction per depth, context tokens
        first."""
        c = self.fuse(torch.cat(ctx_levels, dim=-1)) + self.pred_pos[ctx_idx]
        q = self.mask_token + self.pred_pos[tgt_idx]
        out = self.predictor(torch.cat([c, q], dim=1))
        return [head(out) for head in self.pred_out]

    def forward(self, x, ctx_idx, tgt_idx):
        """Training pass.

        Returns
        -------
        preds : list of torch.Tensor of shape (B, c + m, D)
            Predictions of the visible and then the hidden tokens, one
            entry per depth.

        targets : list of torch.Tensor of shape (B, T, D)
            EMA embeddings of all tokens, one entry per depth.
        """
        tokens = self.embed(x)
        ctx_levels = self._level_outputs(self.encoder,
                                         _gather_tokens(tokens, ctx_idx))
        return (self.predict_levels(ctx_levels, ctx_idx, tgt_idx),
                self.target_levels(tokens))

    @torch.no_grad()
    def multi_mask_errors(self, x, ctx_idx, tgt_idx):
        """Squared error of the hidden tokens at the last depth, of
        shape (B, K, m), and the target embeddings (B, T, D)."""
        n_masks, n_mask = tgt_idx.shape
        tokens = self.embed(x)
        h_all = self.target_levels(tokens)[-1]
        tokens_rep, ctx, tgt = _repeat_masks(tokens, ctx_idx, tgt_idx)
        ctx_levels = self._level_outputs(self.encoder,
                                         _gather_tokens(tokens_rep, ctx))
        pred = self.predict_levels(ctx_levels, ctx, tgt)[-1][:, ctx.size(1):]
        h_tgt = _gather_tokens(h_all.repeat_interleave(n_masks, dim=0), tgt)
        err = (pred - h_tgt).pow(2).mean(-1)
        return err.view(x.size(0), n_masks, n_mask), h_all


class _TokenEncoder(nn.Module):
    """Tokenizer, learned positions and Transformer in one module."""

    def __init__(self, n_tokens, in_dim, d_model, n_heads, n_layers, dropout,
                 final_norm):
        super().__init__()
        self.tokenizer = Tokenizer(n_tokens, in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.body = TransformerStack(d_model, n_heads, n_layers, dropout,
                                     final_norm=final_norm)

    def forward(self, x, idx=None):
        # encode the patches idx (all patches when None) of x
        tok = self.tokenizer(x) + self.pos
        if idx is not None:
            tok = _gather_tokens(tok, idx)
        return self.body(tok)


class SALTCore(JEPACore):
    """The network of the ``'salt'`` version.

    A student, a teacher and the teacher's reconstruction decoder,
    after SALT (Li et al., arXiv:2509.24317). Stage 1 trains the
    teacher with :meth:`reconstruct`, the raw values of the hidden
    patches from the visible ones. Stage 2 freezes it and trains the
    student, the inherited encoder and predictor, to predict the
    teacher's layer-normalised embeddings of the hidden patches. The
    embeddings returned for scoring are the student's.

    Parameters
    ----------
    n_tokens : int
        Patches per window.

    in_dim : int
        Raw values per patch.

    d_model, n_heads, n_layers, pred_layers, dropout, encoder_norm
        As for :class:`pyod.models._jepa_utils.JEPACore`.
    """

    def __init__(self, n_tokens, in_dim, d_model=64, n_heads=4, n_layers=2,
                 pred_layers=1, dropout=0.0, encoder_norm=False):
        super().__init__(n_tokens, in_dim, d_model=d_model, n_heads=n_heads,
                         n_layers=n_layers, pred_layers=pred_layers,
                         dropout=dropout, target_mode='sigreg',
                         encoder_norm=encoder_norm)
        pred_dim = self.pred_in.out_features
        self.teacher = _TokenEncoder(n_tokens, in_dim, d_model, n_heads,
                                     n_layers, dropout, encoder_norm)
        self.dec_in = nn.Linear(d_model, pred_dim)
        self.dec_mask = nn.Parameter(torch.randn(pred_dim) * 0.02)
        self.dec_pos = nn.Parameter(torch.randn(n_tokens, pred_dim) * 0.02)
        self.decoder = TransformerStack(
            pred_dim, max(1, min(n_heads, pred_dim // 16)), pred_layers,
            dropout)
        self.dec_out = nn.Linear(pred_dim, in_dim)

    def teacher_parameters(self):
        """Parameters of the teacher and of its decoder."""
        modules = (self.teacher, self.dec_in, self.decoder, self.dec_out)
        return [p for m in modules for p in m.parameters()] \
            + [self.dec_mask, self.dec_pos]

    def freeze_teacher(self):
        """End of stage 1: the teacher takes no gradient any more."""
        for p in self.teacher_parameters():
            p.requires_grad_(False)
            # no stale gradient for the optimizer of stage 2
            p.grad = None

    def reconstruct(self, x, ctx_idx, tgt_idx):
        """Stage 1: raw values of the hidden patches, (B, m, in_dim)."""
        h = self.teacher(x, ctx_idx)
        c = self.dec_in(h) + self.dec_pos[ctx_idx]
        q = self.dec_mask + self.dec_pos[tgt_idx]
        out = self.decoder(torch.cat([c, q], dim=1))
        return self.dec_out(out[:, c.size(1):])

    @torch.no_grad()
    def teacher_embeddings(self, x):
        """Layer-normalised teacher embeddings of all patches."""
        h = self.teacher(x)
        return F.layer_norm(h, (h.size(-1),))

    def forward(self, x, ctx_idx, tgt_idx):
        """Stage 2 training pass: ``pred``, ``h_tgt`` and ``h_all``
        with the teacher's embeddings as targets."""
        tokens = self.embed(x)
        h_ctx = self.encoder(_gather_tokens(tokens, ctx_idx))
        pred = self.predict(h_ctx, ctx_idx, tgt_idx)
        t_all = self.teacher_embeddings(x)
        return pred, _gather_tokens(t_all, tgt_idx), t_all

    @torch.no_grad()
    def multi_mask_errors(self, x, ctx_idx, tgt_idx):
        """Squared error against the teacher's embeddings, of shape
        (B, K, m), and the student's embeddings (B, T, D)."""
        n_masks, n_mask = tgt_idx.shape
        tokens = self.embed(x)
        t_all = self.teacher_embeddings(x)
        tokens_rep, ctx, tgt = _repeat_masks(tokens, ctx_idx, tgt_idx)
        h_ctx = self.encoder(_gather_tokens(tokens_rep, ctx))
        pred = self.predict(h_ctx, ctx, tgt)
        h_tgt = _gather_tokens(t_all.repeat_interleave(n_masks, dim=0), tgt)
        err = (pred - h_tgt).pow(2).mean(-1)
        return err.view(x.size(0), n_masks, n_mask), self.encoder(tokens)


class KerJEPACore(JEPACore):
    """The network of the ``'kerjepa'`` version.

    The network of ``'lejepa'`` whose prediction targets are
    standardised with running estimates of the embedding mean and
    variance, which KerJEPA (Zimmermann et al., arXiv:2512.19605)
    reports as necessary.

    Parameters
    ----------
    n_tokens : int
        Patches per window.

    in_dim : int
        Raw values per patch.

    stat_momentum : float, optional (default=0.99)
        Momentum of the running statistics.

    kwargs : dict
        Sizes, as for :class:`pyod.models._jepa_utils.JEPACore`.
    """

    def __init__(self, n_tokens, in_dim, stat_momentum=0.99, **kwargs):
        super().__init__(n_tokens, in_dim, target_mode='sigreg', **kwargs)
        self.stat_momentum = stat_momentum
        self.register_buffer('tgt_mean', torch.zeros(self.d_model))
        self.register_buffer('tgt_var', torch.ones(()))

    def standardise(self, h):
        """Embeddings minus the running mean, over the running
        standard deviation."""
        return (h - self.tgt_mean) / self.tgt_var.clamp_min(1e-8).sqrt()

    @torch.no_grad()
    def _update_stats(self, h_all):
        flat = h_all.detach().reshape(-1, h_all.size(-1))
        momentum = self.stat_momentum
        self.tgt_mean.mul_(momentum).add_(flat.mean(0), alpha=1.0 - momentum)
        self.tgt_var.mul_(momentum).add_(flat.var(0).mean(),
                                         alpha=1.0 - momentum)

    def forward(self, x, ctx_idx, tgt_idx):
        """Training pass with standardised targets. In training mode
        it also updates the running statistics."""
        pred, h_tgt, h_all = super().forward(x, ctx_idx, tgt_idx)
        if self.training:
            self._update_stats(h_all)
        return pred, self.standardise(h_tgt), h_all

    @torch.no_grad()
    def multi_mask_errors(self, x, ctx_idx, tgt_idx):
        """Squared error against the standardised targets, of shape
        (B, K, m), and the embeddings (B, T, D)."""
        n_masks, n_mask = tgt_idx.shape
        tokens = self.embed(x)
        h_all = self.encode_target(tokens)
        tokens_rep, ctx, tgt = _repeat_masks(tokens, ctx_idx, tgt_idx)
        pred = self.predict(self.encoder(_gather_tokens(tokens_rep, ctx)),
                            ctx, tgt)
        h_tgt = self.standardise(
            _gather_tokens(h_all.repeat_interleave(n_masks, dim=0), tgt))
        err = (pred - h_tgt).pow(2).mean(-1)
        return err.view(x.size(0), n_masks, n_mask), h_all


class _Rectified(nn.Module):
    """``relu(body(x))``."""

    def __init__(self, body):
        super().__init__()
        self.body = body

    def forward(self, x):
        return F.relu(self.body(x))


class RectifiedJEPACore(JEPACore):
    """The network of the ``'lpjepa'`` version.

    The network of ``'lejepa'`` with a ReLU on the encoder output, as
    Rectified LpJEPA (Kuang et al., arXiv:2602.01456) requires.

    Parameters
    ----------
    n_tokens : int
        Patches per window.

    in_dim : int
        Raw values per patch.

    kwargs : dict
        Sizes, as for :class:`pyod.models._jepa_utils.JEPACore`.
    """

    def __init__(self, n_tokens, in_dim, **kwargs):
        super().__init__(n_tokens, in_dim, target_mode='sigreg', **kwargs)
        self.encoder = _Rectified(self.encoder)


class NextLatentCore(nn.Module):
    """The network of the ``'lenepa'`` and ``'lewm'`` versions.

    Causal prediction of the next patch in latent space, without masks.

    ``style='lenepa'`` follows LeNEPA (Chemeris, Jin and Balestriero,
    arXiv:2607.00958): one linear patch embedding ``z``, a causal
    Transformer over it, and a shared projector; position ``t`` of the
    Transformer predicts ``z`` of patch ``t + 1`` in the space of the
    projector. The embeddings returned for scoring are the outputs of
    the Transformer.

    ``style='lewm'`` follows the two-term objective of LeWorldModel
    (Maes et al., arXiv:2603.19312) without its action input: an
    encoder applied to every patch gives ``z`` and a causal Transformer
    predicts ``z`` of the next patch directly. The embeddings returned
    for scoring are ``z``.

    Parameters
    ----------
    n_tokens : int
        Patches per window.

    in_dim : int
        Raw values per patch.

    d_model, n_heads, n_layers, dropout, encoder_norm
        As for :class:`pyod.models._jepa_utils.JEPACore`.

    style : str, optional (default='lenepa')
        'lenepa' or 'lewm'.

    proj_dim : int, optional (default=64)
        Output width of the projector of 'lenepa'.
    """

    def __init__(self, n_tokens, in_dim, d_model=64, n_heads=4, n_layers=2,
                 dropout=0.0, encoder_norm=False, style='lenepa',
                 proj_dim=64):
        super().__init__()
        if style not in ('lenepa', 'lewm'):
            raise ValueError("style must be 'lenepa' or 'lewm'; got %r"
                             % (style,))
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.style = style
        if style == 'lenepa':
            self.tokenizer = nn.Linear(in_dim, d_model)
            self.projector = nn.Sequential(
                nn.Linear(d_model, 2 * d_model), nn.BatchNorm1d(2 * d_model),
                nn.ReLU(), nn.Linear(2 * d_model, proj_dim))
        else:
            self.tokenizer = nn.Sequential(
                nn.Linear(in_dim, d_model), nn.GELU(),
                nn.Linear(d_model, d_model))
            self.head = nn.Linear(d_model, d_model)
        self.pos = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.backbone = TransformerStack(d_model, n_heads, n_layers, dropout,
                                         final_norm=encoder_norm)
        self.register_buffer('causal', torch.triu(
            torch.full((n_tokens, n_tokens), float('-inf')), diagonal=1))

    def eval_mask_set(self):
        """The scoring split: every patch but the first is predicted
        from the patches before it."""
        return (torch.zeros(1, 1, dtype=torch.long),
                torch.arange(1, self.n_tokens).unsqueeze(0))

    def _causal(self, z):
        h = self.backbone.layers(z + self.pos, mask=self.causal)
        return self.backbone.norm(h)

    def _project(self, h):
        batch_size, n_tokens, width = h.shape
        return self.projector(h.reshape(batch_size * n_tokens, width)).view(
            batch_size, n_tokens, -1)

    def forward(self, x):
        """Training pass.

        Parameters
        ----------
        x : torch.Tensor of shape (B, T, in_dim)
            Raw tokens.

        Returns
        -------
        pred : torch.Tensor of shape (B, T - 1, E)
            Predictions of patches 1 to T - 1.

        target : torch.Tensor of shape (B, T - 1, E)
            What they are compared with (not detached).

        reg : list of torch.Tensor of shape (B, T, E)
            The embeddings to regularise.

        y : torch.Tensor of shape (B, T, D)
            Outputs of the Transformer.
        """
        z = self.tokenizer(x)
        y = self._causal(z)
        if self.style == 'lenepa':
            u_in, u_out = self._project(z), self._project(y)
            return u_out[:, :-1], u_in[:, 1:], [u_in, u_out], y
        return self.head(y)[:, :-1], z[:, 1:], [z], y

    @torch.no_grad()
    def multi_mask_errors(self, x, ctx_idx, tgt_idx):
        """Next-latent error of the patches in ``tgt_idx`` (all at
        least 1), of shape (B, K, m), and the embeddings (B, T, D).
        ``ctx_idx`` is not used."""
        pred, target, reg, y = self.forward(x)
        err = (pred - target).pow(2).mean(-1)  # (B, T - 1)
        h_all = y if self.style == 'lenepa' else reg[0]
        return err[:, tgt_idx.to(x.device) - 1], h_all


class _DilatedBlock(nn.Module):
    """Parallel depthwise convolutions (kernels 3, 9, 15), batch norm,
    GELU, a pointwise convolution and a residual connection.

    Input and output are (B, L, C). The convolutions are computed as
    sums over the filter taps, because the grouped dilated ``Conv1d``
    of PyTorch is more than ten times slower on a CPU. The ``Conv1d``
    modules only hold the parameters.
    """

    def __init__(self, width, dilation, kernels=(3, 9, 15)):
        super().__init__()
        self.dilation = dilation
        self.convs = nn.ModuleList([
            nn.Conv1d(width, width, k, groups=width) for k in kernels])
        self.bn = nn.BatchNorm1d(width)
        self.point = nn.Conv1d(width, width, 1)

    def forward(self, x):
        batch_size, length, width = x.shape
        dilation, y = self.dilation, 0.0
        for conv in self.convs:
            k = conv.kernel_size[0]
            pad = dilation * (k - 1) // 2  # zero 'same' padding
            xp = F.pad(x, (0, 0, pad, pad))
            w = conv.weight[:, 0]  # (C, k)
            y = y + conv.bias
            for j in range(k):
                # the other taps only see the padding
                if abs(j * dilation - pad) < length:
                    y = y + xp[:, j * dilation:j * dilation + length] \
                        * w[:, j]
        y = F.gelu(self.bn(y.reshape(batch_size * length, width))).view(
            batch_size, length, width)
        return x + F.linear(y, self.point.weight[:, :, 0], self.point.bias)


class _ConvEncoder(nn.Module):
    """(B, L, C) series -> (B, L, D): one embedding per timestamp."""

    def __init__(self, n_channels, width, out_dim, n_blocks):
        super().__init__()
        self.inp = nn.Linear(n_channels, width)
        self.blocks = nn.Sequential(
            *[_DilatedBlock(width, 2 ** i) for i in range(n_blocks)])
        self.out = nn.Linear(width, out_dim)

    def forward(self, x):
        return self.out(self.blocks(self.inp(x)))


class CFJEPACore(nn.Module):
    """The network of the ``'cfjepa'`` version.

    After CF-JEPA (Lee and Sim, arXiv:2606.07031). There are no masks:
    a random crop of the window is the context, the part of the window
    after the crop is cut into three zones, and one linear predictor
    per zone maps the embeddings of the crop, one per timestamp, to the
    EMA encoder's embeddings of that zone. The paper does not say how a
    prediction of the crop's length is laid over a zone of another
    length; here it is average-pooled along time to the zone's length.
    One crop position is drawn per crop and step and shared by the
    batch.

    For scoring, the context is the leading ``n_ctx_tokens`` patches
    and every later patch gets the mean error of its timestamps. The
    embeddings returned for scoring are the EMA encoder's, averaged
    within each patch.

    Parameters
    ----------
    n_tokens : int
        Patches per window.

    patch_len : int
        Timestamps per patch.

    n_channels : int
        Channels of the series.

    d_model : int, optional (default=64)
        Width of the encoder and of its embeddings.

    n_blocks : int, optional (default=5)
        Dilated convolution blocks; block ``i`` has dilation ``2 ** i``.

    n_crops : int, optional (default=4)
        Crops per window and training step.

    crop_ratio : tuple of float or None, optional (default=None)
        Range of the crop length as a share of the window. None uses
        the published ranges, (0.367, 0.418) for windows of at least 50
        timestamps and (0.6, 0.8) for shorter ones.
    """

    def __init__(self, n_tokens, patch_len, n_channels, d_model=64,
                 n_blocks=5, n_crops=4, crop_ratio=None):
        super().__init__()
        self.n_tokens = n_tokens
        self.patch_len = patch_len
        self.window = n_tokens * patch_len
        self.d_model = d_model
        self.n_crops = n_crops
        if crop_ratio is None:
            crop_ratio = (0.367, 0.418) if self.window >= 50 else (0.6, 0.8)
        self.crop_ratio = tuple(crop_ratio)
        mid = 0.5 * (self.crop_ratio[0] + self.crop_ratio[1])
        self.n_ctx_tokens = int(min(n_tokens - 1,
                                    max(1, round(mid * n_tokens))))
        self.encoder = _ConvEncoder(n_channels, d_model, d_model, n_blocks)
        self.predictors = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(3)])
        for predictor in self.predictors:  # near the identity
            with torch.no_grad():
                predictor.weight.copy_(
                    torch.eye(d_model)
                    + 1e-3 * torch.randn(d_model, d_model))
                predictor.bias.zero_()
        self.target_encoder = _frozen_copy(self.encoder)

    def eval_mask_set(self):
        """The scoring split: the leading ``n_ctx_tokens`` patches are
        the context and the others are predicted."""
        n_ctx = self.n_ctx_tokens
        return (torch.arange(n_ctx).unsqueeze(0),
                torch.arange(n_ctx, self.n_tokens).unsqueeze(0))

    def series(self, x):
        """Patches (B, T, patch_len * C) back to the window (B, L, C)."""
        return x.reshape(x.size(0), self.window, -1)

    def _zone_errors(self, z_crop, z_future):
        """Absolute difference per timestamp between the unit-normalised
        zone predictions and targets, (B, R); None if nothing follows
        the crop. A future shorter than three timestamps has one zone
        per timestamp."""
        n_future = z_future.size(1)
        if n_future < 1:
            return None
        out = []
        zones = torch.tensor_split(torch.arange(n_future),
                                   min(3, n_future))
        for predictor, idx in zip(self.predictors, zones):
            p = F.adaptive_avg_pool1d(predictor(z_crop).transpose(1, 2),
                                      len(idx)).transpose(1, 2)
            t = z_future[:, idx[0]:idx[-1] + 1]
            out.append((F.normalize(p, dim=-1)
                        - F.normalize(t, dim=-1)).abs().mean(-1))
        return torch.cat(out, dim=1)

    def training_terms(self, x, generator=None):
        """Training pass.

        Parameters
        ----------
        x : torch.Tensor of shape (B, T, patch_len * C)
            Raw tokens.

        generator : torch.Generator or None, optional (default=None)
            CPU generator for the crop lengths and positions.

        Returns
        -------
        horizon : torch.Tensor (scalar)
            Prediction loss over the zones after the crops.

        crops : list of torch.Tensor of shape (B, L_i, D)
            Embeddings of the crops, for the regularisers.
        """
        xs = self.series(x)
        window = self.window
        with torch.no_grad():
            z_full = self.target_encoder(xs)
        low, high = self.crop_ratio
        crops, horizon = [], []
        for _ in range(self.n_crops):
            ratio = low + (high - low) * torch.rand(
                (), generator=generator).item()
            length = int(min(window - 1, max(2, int(ratio * window))))
            start = int(torch.randint(0, window - length + 1, (),
                                      generator=generator))
            z_crop = self.encoder(xs[:, start:start + length])
            crops.append(z_crop)
            err = self._zone_errors(z_crop, z_full[:, start + length:])
            if err is not None:
                horizon.append(err.mean())
        loss = torch.stack(horizon).mean() if horizon else xs.sum() * 0.0
        return loss, crops

    @torch.no_grad()
    def multi_mask_errors(self, x, ctx_idx, tgt_idx):
        """Error of the patches after the leading crop, of shape
        (B, 1, m), and the embeddings (B, T, D). The split is fixed;
        ``ctx_idx`` and ``tgt_idx`` are not used."""
        xs = self.series(x)
        length = self.n_ctx_tokens * self.patch_len
        z_full = self.target_encoder(xs)
        err = self._zone_errors(self.encoder(xs[:, :length]),
                                z_full[:, length:])
        err = err.view(x.size(0), self.n_tokens - self.n_ctx_tokens,
                       self.patch_len).mean(-1)
        h_all = z_full.view(x.size(0), self.n_tokens, self.patch_len,
                            -1).mean(2)
        return err.unsqueeze(1), h_all


# ----------------------------------------------------------------------
# Recipes
# ----------------------------------------------------------------------
class Recipe(object):
    """Base class of a JEPA version.

    A recipe is stateless. The detector passes itself as ``det``, from
    which a recipe reads the masking, the budget and ``sigreg_slices``,
    and the resolved constants of the version as ``hp``.

    Attributes
    ----------
    encoder_norm : bool
        What ``encoder_norm=None`` means for this version.

    ema_momentum : float or None
        What ``ema_momentum=None`` means for this version. None for a
        version without an EMA target.

    ema_schedule : str or None
        How the momentum develops during training, see
        :func:`pyod.models._jepa_utils.momentum_at`. None for a version
        without an EMA target. With a schedule, the network has an
        ``encoder`` and a ``target_encoder`` that follows it.

    masked : bool
        False for a version that does not hide patches; ``mask_mode``
        and ``mask_ratio`` are then ignored.

    params : dict
        Constants of the version: the published value where the paper
        gives one, otherwise the value named in the docstring of the
        class. ``TSJEPA(variant_params={...})`` overrides them.
    """
    encoder_norm = False
    ema_momentum = None
    ema_schedule = None
    masked = True
    params = {}

    def build(self, det, n_tokens, n_channels, size, hp):
        """Build the network.

        Parameters
        ----------
        det : TSJEPA
            The detector.

        n_tokens : int
            Patches per window.

        n_channels : int
            Channels of the series.

        size : dict
            ``d_model``, ``n_heads``, ``n_layers``, ``pred_layers``,
            ``dropout`` and ``encoder_norm``.

        hp : dict
            Constants of the version.

        Returns
        -------
        model : torch.nn.Module
        """
        raise NotImplementedError

    def loss(self, det, model, x, ctx, tgt, gen, state, hp):
        """Loss of one batch.

        Parameters
        ----------
        det : TSJEPA
            The detector.

        model : torch.nn.Module
            The network returned by :meth:`build`.

        x : torch.Tensor of shape (B, T, in_dim)
            Raw tokens.

        ctx, tgt : torch.LongTensor or None
            Visible and hidden token indices per row. None when
            ``masked`` is False.

        gen : torch.Generator
            CPU generator of the training run.

        state : dict
            ``frac``, the share of the training steps done, and
            ``epoch`` and ``n_epochs``.

        hp : dict
            Constants of the version.

        Returns
        -------
        loss : torch.Tensor (scalar)
            The loss to minimise.

        loss_pred : torch.Tensor (scalar)
            Its latent prediction term, for the log.
        """
        raise NotImplementedError

    def train_epochs(self, det, hp):
        """Number of epochs of the main training loop."""
        return det.epochs

    def pretrain(self, det, model, make_loader, device, gen, hp):
        """Run what has to happen before the main training loop."""


def _token_dim(det, n_channels):
    return det.patch_len * n_channels


class VJEPA(Recipe):
    """V-JEPA (Bardes et al., arXiv:2404.08471): an EMA target with
    momentum 0.998 to 1 and the absolute error as loss. Network and
    target normalisation are those of ``'ijepa'``."""
    ema_momentum, ema_schedule = 0.998, 'linear'

    def build(self, det, n_tokens, n_channels, size, hp):
        return JEPACore(n_tokens, _token_dim(det, n_channels),
                        target_mode='ema', **size)

    def loss(self, det, model, x, ctx, tgt, gen, state, hp):
        pred, h_tgt, _ = model(x, ctx, tgt)
        loss = F.l1_loss(pred, h_tgt.detach())
        return loss, loss


class VJEPA21(Recipe):
    """V-JEPA 2.1 (arXiv:2603.14482): the absolute error on the hidden
    tokens plus a loss on the visible tokens weighted by
    ``context_weight / sqrt(distance to the nearest hidden token)``,
    both at several encoder depths, with a fixed EMA momentum of
    0.99925. The paper ramps the context weight during training without
    giving the shape; here it rises linearly from 0 over the first
    ``warmup`` share of the steps, and the losses of the depths are
    averaged."""
    ema_momentum, ema_schedule = 0.99925, 'fixed'
    params = dict(context_weight=0.5, warmup=0.5, max_levels=4)

    def build(self, det, n_tokens, n_channels, size, hp):
        return DenseJEPACore(n_tokens, _token_dim(det, n_channels),
                             max_levels=hp['max_levels'], **size)

    def loss(self, det, model, x, ctx, tgt, gen, state, hp):
        preds, targets = model(x, ctx, tgt)
        n_ctx = ctx.size(1)
        ramp = 1.0 if hp['warmup'] <= 0 \
            else min(1.0, state['frac'] / hp['warmup'])
        dist = (ctx.unsqueeze(2) - tgt.unsqueeze(1)).abs().min(dim=2).values
        weight = (hp['context_weight'] * ramp
                  / dist.clamp_min(1).to(x.dtype).sqrt()).unsqueeze(-1)
        loss_pred = loss_ctx = 0.0
        for pred, h_all in zip(preds, targets):
            loss_pred = loss_pred + F.l1_loss(
                pred[:, n_ctx:], _gather_tokens(h_all, tgt))
            loss_ctx = loss_ctx + (weight * (
                pred[:, :n_ctx] - _gather_tokens(h_all, ctx)).abs()).mean()
        loss_pred, loss_ctx = loss_pred / len(preds), loss_ctx / len(preds)
        return loss_pred + loss_ctx, loss_pred


class CJEPA(Recipe):
    """C-JEPA (Mo and Tong, arXiv:2410.19560): I-JEPA plus VICReg at an
    overall weight of 0.001 with VICReg's own weights 25, 25 and 1. The
    paper applies it to the context embeddings of the mask blocks of
    one image; here it is applied to the projected mean context
    embeddings of two mask draws of one window."""
    ema_momentum, ema_schedule = 0.996, 'linear'
    params = dict(vicreg_weight=0.001, sim_weight=25.0, var_weight=25.0,
                  cov_weight=1.0)

    def build(self, det, n_tokens, n_channels, size, hp):
        return CJEPACore(n_tokens, _token_dim(det, n_channels), **size)

    def loss(self, det, model, x, ctx, tgt, gen, state, hp):
        ctx2, tgt2 = sample_masks(x.size(0), det.n_tokens_, det.n_mask_,
                                  det.mask_mode, gen)
        ctx2, tgt2 = ctx2.to(x.device), tgt2.to(x.device)
        pred1, h_tgt1, _, z1 = model.forward_views(x, ctx, tgt)
        pred2, h_tgt2, _, z2 = model.forward_views(x, ctx2, tgt2)
        loss_pred = 0.5 * (F.mse_loss(pred1, h_tgt1.detach())
                           + F.mse_loss(pred2, h_tgt2.detach()))
        reg = (hp['sim_weight'] * F.mse_loss(z1, z2)
               + hp['var_weight'] * 0.5 * (variance_loss(z1)
                                           + variance_loss(z2))
               + hp['cov_weight'] * (covariance_loss(z1)
                                     + covariance_loss(z2)))
        return loss_pred + hp['vicreg_weight'] * reg, loss_pred


class SJEPA(Recipe):
    """Stochastic-latent JEPA (Giakoumoglou et al., arXiv:2611.09302):
    I-JEPA with a Gaussian latent for the predictor and a KL term. The
    objective is the summed squared error plus ``kl_weight * KL``. The
    weight of 1 and the latent width ``d_model // 4`` (``latent_dim``
    None) are choices of this implementation."""
    ema_momentum, ema_schedule = 0.996, 'linear'
    params = dict(kl_weight=1.0, latent_dim=None)

    def build(self, det, n_tokens, n_channels, size, hp):
        return StochasticJEPACore(n_tokens, _token_dim(det, n_channels),
                                  latent_dim=hp['latent_dim'], **size)

    def loss(self, det, model, x, ctx, tgt, gen, state, hp):
        noise = torch.randn(x.size(0), model.latent_dim,
                            generator=gen).to(x)
        pred, h_tgt, _, kl = model.forward_latent(x, ctx, tgt, noise)
        loss_pred = F.mse_loss(pred, h_tgt.detach())
        # the squared error is averaged over m * D elements here
        scale = pred.size(1) * pred.size(2)
        return loss_pred + hp['kl_weight'] * kl / scale, loss_pred


class SALT(Recipe):
    """SALT (Li et al., arXiv:2509.24317): a teacher is trained by
    masked reconstruction and then frozen, and the student predicts its
    latents with the absolute error. No EMA and no regulariser. The
    teacher gets ``teacher_share`` of the epochs, at least one; 1/6 is
    the 1 : 5 split the paper found best."""
    params = dict(teacher_share=1.0 / 6.0)

    def build(self, det, n_tokens, n_channels, size, hp):
        return SALTCore(n_tokens, _token_dim(det, n_channels), **size)

    @staticmethod
    def _teacher_epochs(det, hp):
        if det.epochs <= 0:
            return 0
        return max(1, int(round(det.epochs * hp['teacher_share'])))

    def train_epochs(self, det, hp):
        if det.epochs <= 0:
            return 0
        return max(1, det.epochs - self._teacher_epochs(det, hp))

    def pretrain(self, det, model, make_loader, device, gen, hp):
        n_epochs = self._teacher_epochs(det, hp)
        if n_epochs > 0:
            loader = make_loader()
            optimizer = torch.optim.AdamW(
                model.teacher_parameters(), lr=det.lr,
                weight_decay=det.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs, eta_min=det.lr * 0.05)
            model.train()
            for epoch in range(n_epochs):
                epoch_losses = []
                for (batch_x,) in loader:
                    batch_x = batch_x.to(device)
                    ctx, tgt = sample_masks(
                        batch_x.size(0), det.n_tokens_, det.n_mask_,
                        det.mask_mode, gen)
                    ctx, tgt = ctx.to(device), tgt.to(device)
                    optimizer.zero_grad()
                    loss = F.mse_loss(model.reconstruct(batch_x, ctx, tgt),
                                      _gather_tokens(batch_x, tgt))
                    loss.backward()
                    if det.grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            model.teacher_parameters(), det.grad_clip)
                    optimizer.step()
                    epoch_losses.append(loss.item())
                scheduler.step()
                if det.verbose:
                    print('Teacher epoch %d/%d, reconstruction loss=%.4f'
                          % (epoch + 1, n_epochs,
                             sum(epoch_losses) / max(1, len(epoch_losses))))
        model.freeze_teacher()

    def loss(self, det, model, x, ctx, tgt, gen, state, hp):
        pred, h_tgt, _ = model(x, ctx, tgt)
        loss = F.l1_loss(pred, h_tgt.detach())
        return loss, loss


class _Regularised(Recipe):
    """Versions with one shared encoder: the squared error against
    stop-gradient targets, as ``'lejepa'``, plus the regulariser of the
    version."""
    core = JEPACore

    def build(self, det, n_tokens, n_channels, size, hp):
        if self.core is JEPACore:
            return JEPACore(n_tokens, _token_dim(det, n_channels),
                            target_mode='sigreg', **size)
        return self.core(n_tokens, _token_dim(det, n_channels), **size)

    def regulariser(self, det, h_all, gen, hp):
        raise NotImplementedError

    def loss(self, det, model, x, ctx, tgt, gen, state, hp):
        pred, h_tgt, h_all = model(x, ctx, tgt)
        loss_pred = F.mse_loss(pred, h_tgt.detach())
        return loss_pred + self.regulariser(det, h_all, gen, hp), loss_pred


class VICRegJEPA(_Regularised):
    """A JEPA regularised by the variance and covariance terms of
    VICReg, with the weights 1 and 80 that EB-JEPA (Terver et al.,
    arXiv:2602.03604) uses for images. Computed per patch position over
    the batch."""
    params = dict(var_weight=1.0, cov_weight=80.0)

    def regulariser(self, det, h_all, gen, hp):
        z = h_all.transpose(0, 1)
        return (hp['var_weight'] * variance_loss(z)
                + hp['cov_weight'] * covariance_loss(z, pairwise=True))


class KerJEPA(_Regularised):
    """KerJEPA (Zimmermann et al., arXiv:2512.19605): the closed-form
    Cramer-Wold MMD to a Gaussian prior at weight 7.5, with
    standardised prediction targets. Computed per patch position over
    the batch."""
    core = KerJEPACore
    params = dict(reg_weight=7.5, prior_scale=1.0)

    def regulariser(self, det, h_all, gen, hp):
        return hp['reg_weight'] * cw_mmd(h_all.transpose(0, 1),
                                         sigma=hp['prior_scale'])


class LpJEPA(_Regularised):
    """Rectified LpJEPA (Kuang et al., arXiv:2602.01456): non-negative
    embeddings matched to a rectified Laplace distribution (``p=1``)
    by RDMReg, with ``sigreg_slices`` projections. The weight of 5 is
    the ratio of the two weights of the released code and a choice of
    this implementation."""
    core = RectifiedJEPACore
    params = dict(reg_weight=5.0, p=1.0, mean_shift=0.0)

    def regulariser(self, det, h_all, gen, hp):
        return hp['reg_weight'] * rdm_reg(
            h_all.transpose(0, 1), p=hp['p'], mean_shift=hp['mean_shift'],
            num_slices=det.sigreg_slices, generator=gen)


class LambdaJEPA(_Regularised):
    """lambda-JEPA (Demirel et al., arXiv:2609.35288): SACReg on the
    encoder output, over all patch embeddings of the batch. The paper
    calibrates its weights per model; the weight of 1 on the value per
    dimension is a choice of this implementation."""
    params = dict(reg_weight=1.0, eps=1e-4)

    def regulariser(self, det, h_all, gen, hp):
        return hp['reg_weight'] * sac_reg(
            h_all.reshape(-1, h_all.size(-1)), eps=hp['eps'], generator=gen)


class _NextLatent(Recipe):
    """Versions that predict the next patch and hide none."""
    masked = False
    style = None

    def build(self, det, n_tokens, n_channels, size, hp):
        return NextLatentCore(
            n_tokens, _token_dim(det, n_channels), d_model=size['d_model'],
            n_heads=size['n_heads'], n_layers=size['n_layers'],
            dropout=size['dropout'], encoder_norm=size['encoder_norm'],
            style=self.style)


class LeNEPA(_NextLatent):
    """LeNEPA (Chemeris, Jin and Balestriero, arXiv:2607.00958): causal
    next-latent prediction with the summed squared error in the space
    of the projector, plus SIGReg at weight 20, computed for every
    window over its own patches on the projected input and output of
    the Transformer. No EMA, no stop-gradient and no masks."""
    style = 'lenepa'
    params = dict(sigreg_weight=20.0)

    def loss(self, det, model, x, ctx, tgt, gen, state, hp):
        pred, target, reg, _ = model(x)
        loss_pred = (pred - target).pow(2).sum(-1).mean()
        loss_reg = sum(sigreg_loss(u, num_slices=det.sigreg_slices,
                                   generator=gen) for u in reg) / len(reg)
        return loss_pred + hp['sigreg_weight'] * loss_reg, loss_pred


class LeWM(_NextLatent):
    """LeWorldModel (Maes et al., arXiv:2603.19312) without its action
    input: the squared error of the next embedding plus SIGReg at
    weight 0.1, computed over the batch at every patch position. No
    EMA, no stop-gradient and no masks."""
    style = 'lewm'
    params = dict(sigreg_weight=0.1)

    def loss(self, det, model, x, ctx, tgt, gen, state, hp):
        pred, target, reg, _ = model(x)
        loss_pred = F.mse_loss(pred, target)
        loss_reg = sigreg_loss(reg[0].transpose(0, 1),
                               num_slices=det.sigreg_slices, generator=gen)
        return loss_pred + hp['sigreg_weight'] * loss_reg, loss_pred


class CFJEPA(Recipe):
    """CF-JEPA (Lee and Sim, arXiv:2606.07031): crops as context, three
    forward zones, linear predictors, the absolute error on
    unit-normalised embeddings with a weight that falls to zero over
    the epochs, the variance and covariance terms of VICReg on the mean
    crop embeddings, a multi-scale invariance term across the crops of
    a window, and an EMA target with a cosine momentum schedule from
    0.983. The constants are the published ones."""
    ema_momentum, ema_schedule = 0.983, 'cosine'
    masked = False
    params = dict(n_crops=4, var_weight=0.081, cov_weight=0.076,
                  inv_weight=1.101, n_blocks=5, crop_ratio=None)

    def build(self, det, n_tokens, n_channels, size, hp):
        return CFJEPACore(n_tokens, det.patch_len, n_channels,
                          d_model=size['d_model'], n_blocks=hp['n_blocks'],
                          n_crops=hp['n_crops'], crop_ratio=hp['crop_ratio'])

    def loss(self, det, model, x, ctx, tgt, gen, state, hp):
        horizon, crops = model.training_terms(x, gen)
        pooled = torch.cat([z.mean(1) for z in crops], dim=0)
        inv = 0.0
        for n_bins in (2, 4, 8):
            views = torch.stack(
                [F.adaptive_avg_pool1d(z.transpose(1, 2), n_bins)
                 for z in crops], dim=0)
            inv = inv + views.var(dim=0, unbiased=False).mean() / 3.0
        weight = 1.0 - (state['epoch'] + 1) / max(1, state['n_epochs'])
        loss = (weight * horizon
                + hp['var_weight'] * variance_loss(pooled)
                + hp['cov_weight'] * covariance_loss(pooled)
                + hp['inv_weight'] * inv)
        return loss, horizon


#: The recipes by ``variant`` name, in the order of the documentation.
RECIPES = {
    'vjepa': VJEPA(), 'vjepa21': VJEPA21(), 'cjepa': CJEPA(),
    'sjepa': SJEPA(), 'salt': SALT(), 'vicreg': VICRegJEPA(),
    'kerjepa': KerJEPA(), 'lpjepa': LpJEPA(), 'lambdajepa': LambdaJEPA(),
    'lenepa': LeNEPA(), 'lewm': LeWM(), 'cfjepa': CFJEPA(),
}
