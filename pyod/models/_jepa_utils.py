# -*- coding: utf-8 -*-
"""Shared PyTorch building blocks of the JEPA detectors.

Used by :class:`pyod.models.jepa.JEPA` (tabular) and
:class:`pyod.models.ts_jepa.TSJEPA` (time series). Both present their
data to the network as a sequence of *tokens*:

* tabular rows -> one token per feature (or per small group of
  consecutive features);
* time series  -> one token per patch of consecutive timestamps (all
  channels of the patch are flattened into the token).

:class:`JEPACore` implements a Joint-Embedding Predictive Architecture
on top of these tokens::

    context tokens --encoder--> h_ctx --predictor(+mask tokens)--> pred
    all tokens     --target encoder--> h_all --gather--> h_tgt

and the training signal is ``||pred - h_tgt||`` in *latent* space.
Representation collapse is prevented either with SIGReg
(:func:`sigreg_loss`) or with an EMA target encoder (:func:`ema_update`).

This module imports torch at load time; detectors that must stay
importable without torch should import it lazily.
"""
# Author: Kishor Datta Gupta
# License: BSD 2 clause

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_MODES = ('random', 'block', 'forecast', 'mixed')


# ----------------------------------------------------------------------
# Collapse prevention
# ----------------------------------------------------------------------
def sigreg_loss(z, num_slices=128, n_points=17, t_max=5.0, generator=None):
    """Sketched Isotropic Gaussian Regularization (SIGReg).

    Epps-Pulley characteristic-function statistic on random 1-D
    projections of the embeddings, as used by LeJEPA. Minimising it
    pushes the embedding distribution towards an isotropic Gaussian,
    which rules out collapsed representations without a teacher network.

    Parameters
    ----------
    z : torch.Tensor of shape (N, K) or (G, N, K)
        Embeddings. With a leading group dimension ``G`` (for example
        one group per token position) the statistic is computed per
        group and averaged.

    num_slices : int, optional (default=128)
        Number of random unit directions, resampled on every call.

    n_points : int, optional (default=17)
        Trapezoidal quadrature knots on ``[-t_max, t_max]``.

    t_max : float, optional (default=5.0)
        Half-width of the integration domain.

    generator : torch.Generator or None, optional (default=None)
        CPU generator for the projection directions.

    Returns
    -------
    loss : torch.Tensor (scalar)
        ``N * integral |ecf(t) - exp(-t^2/2)|^2 exp(-t^2/2) dt`` averaged
        over directions and groups.
    """
    if z.dim() == 2:
        z = z.unsqueeze(0)
    n_samples, n_dims = z.shape[1], z.shape[2]
    # directions are sampled on CPU for device-independent reproducibility
    directions = torch.randn(n_dims, num_slices, generator=generator)
    directions = (directions
                  / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
                  ).to(device=z.device, dtype=z.dtype)
    t = torch.linspace(-t_max, t_max, n_points, device=z.device,
                       dtype=z.dtype)
    window = torch.exp(-0.5 * t ** 2)  # CF of N(0, 1) and the weight
    x_t = (z @ directions).unsqueeze(-1) * t  # (G, N, M, T)
    ecf_re = torch.cos(x_t).mean(1)  # (G, M, T)
    ecf_im = torch.sin(x_t).mean(1)
    err = ((ecf_re - window) ** 2 + ecf_im ** 2) * window
    stat = torch.trapezoid(err, t, dim=-1) * n_samples  # (G, M)
    return stat.mean()


@torch.no_grad()
def ema_update(target, online, momentum):
    """In-place update ``target <- m * target + (1 - m) * online``.

    Parameters
    ----------
    target : torch.nn.Module
        The EMA (teacher) network, updated in place.

    online : torch.nn.Module
        The trained (student) network.

    momentum : float
        EMA momentum ``m`` in [0, 1].
    """
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.mul_(momentum).add_(p_o.detach(), alpha=1.0 - momentum)
    for b_t, b_o in zip(target.buffers(), online.buffers()):
        b_t.copy_(b_o)


# ----------------------------------------------------------------------
# Network blocks
# ----------------------------------------------------------------------
class Tokenizer(nn.Module):
    """Per-token affine map ``x[:, t] @ W[t] + b[t]``.

    Parameters
    ----------
    n_tokens : int
        Number of tokens per sample.

    in_dim : int
        Raw values per token.

    out_dim : int
        Output width per token.
    """

    def __init__(self, n_tokens, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(n_tokens, in_dim, out_dim) / math.sqrt(in_dim))
        self.bias = nn.Parameter(torch.zeros(n_tokens, out_dim))

    def forward(self, x):
        # x: (B, T, in_dim) -> (B, T, out_dim)
        return torch.einsum('bti,tio->bto', x, self.weight) + self.bias


class TransformerStack(nn.Module):
    """Pre-norm Transformer encoder with an optional final LayerNorm.

    Parameters
    ----------
    d_model : int
        Token width.

    n_heads : int
        Number of attention heads.

    n_layers : int
        Number of encoder layers.

    dropout : float, optional (default=0.0)
        Dropout rate.

    ff_mult : int, optional (default=2)
        Feed-forward width as a multiple of ``d_model``.

    final_norm : bool, optional (default=True)
        Apply a LayerNorm to the output.
    """

    def __init__(self, d_model, n_heads, n_layers, dropout=0.0, ff_mult=2,
                 final_norm=True):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ff_mult * d_model, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, num_layers=n_layers,
                                            enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model) if final_norm else nn.Identity()

    def forward(self, x):
        return self.norm(self.layers(x))


# ----------------------------------------------------------------------
# Masking
# ----------------------------------------------------------------------
def _complement(tgt, n_tokens):
    """Sorted context indices complementary to ``tgt`` (rows of size m)."""
    batch_size, n_mask = tgt.shape
    is_tgt = torch.zeros(batch_size, n_tokens, dtype=torch.bool)
    is_tgt.scatter_(1, tgt, True)
    ctx = torch.arange(n_tokens).unsqueeze(0).expand(batch_size, n_tokens)
    return ctx[~is_tgt].view(batch_size, n_tokens - n_mask)


def sample_masks(batch_size, n_tokens, n_mask, mode='random',
                 generator=None):
    """Sample one (context, target) split per row for training.

    Parameters
    ----------
    batch_size : int
        Number of rows.

    n_tokens : int
        Tokens per row.

    n_mask : int
        Target tokens per row, ``1 <= n_mask <= n_tokens - 1``.

    mode : str, optional (default='random')
        One of ``'random'`` (uniformly random token subset), ``'block'``
        (one contiguous block at a random position), ``'forecast'``
        (always the last ``n_mask`` tokens) or ``'mixed'`` (per-row coin
        flip between block and forecast).

    generator : torch.Generator or None, optional (default=None)
        CPU generator.

    Returns
    -------
    ctx_idx : torch.LongTensor of shape (batch_size, n_tokens - n_mask)
        Indices of the visible tokens.

    tgt_idx : torch.LongTensor of shape (batch_size, n_mask)
        Indices of the hidden tokens.
    """
    if mode == 'random':
        tgt = torch.rand(batch_size, n_tokens,
                         generator=generator).argsort(dim=1)[:, :n_mask]
        tgt, _ = tgt.sort(dim=1)
    elif mode in ('block', 'forecast', 'mixed'):
        last = n_tokens - n_mask
        start = torch.randint(0, last + 1, (batch_size,),
                              generator=generator)
        if mode == 'forecast':
            start = torch.full((batch_size,), last, dtype=torch.long)
        elif mode == 'mixed':
            flip = torch.rand(batch_size, generator=generator) < 0.5
            start = torch.where(flip, torch.full_like(start, last), start)
        tgt = start.unsqueeze(1) + torch.arange(n_mask).unsqueeze(0)
    else:
        raise ValueError("mask_mode must be one of %s; got %r"
                         % (MASK_MODES, mode))
    return _complement(tgt, n_tokens), tgt


def eval_masks(n_tokens, n_mask, mode='random', n_samples=8,
               generator=None):
    """Build the mask set shared by all rows at inference.

    * ``'block'`` and ``'mixed'`` enumerate all block positions when
      ``n_samples`` is None or large enough (complete, deterministic
      coverage); otherwise a random subset of positions.
    * ``'forecast'`` is the single last block.
    * ``'random'`` chains random permutations cut into groups of
      ``n_mask`` tokens, so every token is masked about equally often.

    Parameters
    ----------
    n_tokens : int
        Tokens per row.

    n_mask : int
        Target tokens per mask.

    mode : str, optional (default='random')
        Masking pattern, see :func:`sample_masks`.

    n_samples : int or None, optional (default=8)
        Number of masks. Ignored for ``'forecast'``.

    generator : torch.Generator or None, optional (default=None)
        CPU generator.

    Returns
    -------
    ctx_idx : torch.LongTensor of shape (n_masks, n_tokens - n_mask)
        Indices of the visible tokens per mask.

    tgt_idx : torch.LongTensor of shape (n_masks, n_mask)
        Indices of the hidden tokens per mask.
    """
    if mode == 'forecast':
        tgt = torch.arange(n_tokens - n_mask, n_tokens).unsqueeze(0)
    elif mode in ('block', 'mixed'):
        starts = torch.arange(0, n_tokens - n_mask + 1)
        if n_samples is not None and n_samples < len(starts):
            keep = torch.randperm(len(starts),
                                  generator=generator)[:n_samples]
            starts = starts[keep.sort().values]
        tgt = starts.unsqueeze(1) + torch.arange(n_mask).unsqueeze(0)
    elif mode == 'random':
        n_masks = 8 if n_samples is None else int(n_samples)
        rows = []
        while len(rows) < n_masks:
            perm = torch.randperm(n_tokens, generator=generator)
            for s in range(0, n_tokens - n_mask + 1, n_mask):
                rows.append(perm[s:s + n_mask].sort().values)
                if len(rows) == n_masks:
                    break
        tgt = torch.stack(rows, 0)
    else:
        raise ValueError("mask_mode must be one of %s; got %r"
                         % (MASK_MODES, mode))
    return _complement(tgt, n_tokens), tgt


def _gather_tokens(x, idx):
    """Gather tokens ``x[b, idx[b, j]]`` -> (B, idx.shape[1], D)."""
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))


# ----------------------------------------------------------------------
# JEPA
# ----------------------------------------------------------------------
class JEPACore(nn.Module):
    """Joint-Embedding Predictive Architecture over a token sequence.

    Parameters
    ----------
    n_tokens : int
        Tokens per sample.

    in_dim : int
        Raw values per token.

    d_model : int, optional (default=64)
        Token embedding width.

    n_heads : int, optional (default=4)
        Attention heads of the encoder.

    n_layers : int, optional (default=2)
        Depth of the encoder.

    pred_layers : int, optional (default=1)
        Depth of the predictor, which has width ``d_model // 2``.

    dropout : float, optional (default=0.0)
        Dropout rate.

    target_mode : str, optional (default='sigreg')
        ``'sigreg'``: one shared encoder produces both context and
        target embeddings and collapse is prevented by SIGReg in the
        loss. ``'ema'``: frozen EMA target encoder with LayerNorm'ed
        targets.

    recon_dim : int or None, optional (default=None)
        If given, a per-token linear head decodes each target embedding
        back to its raw token.

    encoder_norm : bool, optional (default=False)
        Final LayerNorm on the encoder output. With it, extreme inputs
        saturate to a fixed direction that is trivially predictable and
        can score as normal, so it is off by default.
    """

    def __init__(self, n_tokens, in_dim, d_model=64, n_heads=4, n_layers=2,
                 pred_layers=1, dropout=0.0, target_mode='sigreg',
                 recon_dim=None, encoder_norm=False):
        super().__init__()
        if target_mode not in ('sigreg', 'ema'):
            raise ValueError("target_mode must be 'sigreg' or 'ema'; "
                             "got %r" % (target_mode,))
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.target_mode = target_mode
        pred_dim = max(16, d_model // 2)
        pred_heads = max(1, min(n_heads, pred_dim // 16))

        self.tokenizer = Tokenizer(n_tokens, in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.encoder = TransformerStack(d_model, n_heads, n_layers, dropout,
                                        final_norm=encoder_norm)
        if target_mode == 'ema':
            self.target_encoder = copy.deepcopy(self.encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad_(False)
        else:
            self.target_encoder = None

        self.pred_in = nn.Linear(d_model, pred_dim)
        self.mask_token = nn.Parameter(torch.randn(pred_dim) * 0.02)
        self.pred_pos = nn.Parameter(torch.randn(n_tokens, pred_dim) * 0.02)
        self.predictor = TransformerStack(pred_dim, pred_heads, pred_layers,
                                          dropout)
        self.pred_out = nn.Linear(pred_dim, d_model)
        self.recon_head = (Tokenizer(n_tokens, d_model, recon_dim)
                           if recon_dim else None)

    def embed(self, x):
        """Raw tokens (B, T, in_dim) -> positional token embeddings."""
        return self.tokenizer(x) + self.pos

    def encode_target(self, tokens):
        """Target representation of all tokens, shape (B, T, D)."""
        if self.target_mode == 'ema':
            with torch.no_grad():
                h = self.target_encoder(tokens)
                return F.layer_norm(h, (h.size(-1),))
        return self.encoder(tokens)

    def predict(self, h_ctx, ctx_idx, tgt_idx):
        """Predict target-token representations from encoded context."""
        c = self.pred_in(h_ctx) + self.pred_pos[ctx_idx]
        q = self.mask_token + self.pred_pos[tgt_idx]
        out = self.predictor(torch.cat([c, q], dim=1))
        return self.pred_out(out[:, c.size(1):])

    def forward(self, x, ctx_idx, tgt_idx):
        """Training pass.

        Parameters
        ----------
        x : torch.Tensor of shape (B, T, in_dim)
            Raw tokens.

        ctx_idx : torch.LongTensor of shape (B, T - m)
            Visible token indices per row.

        tgt_idx : torch.LongTensor of shape (B, m)
            Hidden token indices per row.

        Returns
        -------
        pred : torch.Tensor of shape (B, m, D)
            Predicted target embeddings.

        h_tgt : torch.Tensor of shape (B, m, D)
            Actual target embeddings (not detached).

        h_all : torch.Tensor of shape (B, T, D)
            Target-encoder embeddings of all tokens.
        """
        tokens = self.embed(x)
        h_ctx = self.encoder(_gather_tokens(tokens, ctx_idx))
        h_all = self.encode_target(tokens)
        h_tgt = _gather_tokens(h_all, tgt_idx)
        pred = self.predict(h_ctx, ctx_idx, tgt_idx)
        return pred, h_tgt, h_all

    @torch.no_grad()
    def multi_mask_errors(self, x, ctx_idx, tgt_idx):
        """Latent prediction error of every row under a shared mask set.

        Parameters
        ----------
        x : torch.Tensor of shape (B, T, in_dim)
            Raw tokens.

        ctx_idx : torch.LongTensor of shape (K, T - m)
            Visible token indices per mask, from :func:`eval_masks`.

        tgt_idx : torch.LongTensor of shape (K, m)
            Hidden token indices per mask.

        Returns
        -------
        err : torch.Tensor of shape (B, K, m)
            Squared error per target token, averaged over embedding
            dimensions.

        h_all : torch.Tensor of shape (B, T, D)
            Target embeddings of all tokens (mask independent).
        """
        batch_size = x.shape[0]
        n_masks, n_mask = tgt_idx.shape
        tokens = self.embed(x)
        h_all = self.encode_target(tokens)
        ctx = ctx_idx.to(x.device).repeat(batch_size, 1)
        tgt = tgt_idx.to(x.device).repeat(batch_size, 1)
        tokens_rep = tokens.repeat_interleave(n_masks, dim=0)
        h_all_rep = h_all.repeat_interleave(n_masks, dim=0)
        h_ctx = self.encoder(_gather_tokens(tokens_rep, ctx))
        pred = self.predict(h_ctx, ctx, tgt)
        h_tgt = _gather_tokens(h_all_rep, tgt)
        err = (pred - h_tgt).pow(2).mean(-1)  # (B * K, m)
        return err.view(batch_size, n_masks, n_mask), h_all
