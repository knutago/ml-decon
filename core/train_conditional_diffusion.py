"""
train_conditional_diffusion.py

Trains a *conditional* DDPM whose denoiser is a flat (no-downsampling) CNN,
learning p(ideal | observed) from paired astronomical patches.

Why conditional?
----------------
An unconditional prior trained only on sharp patches knows what star fields
look like, but nothing about how a *specific* blurry observation constrains
the reconstruction. By concatenating the blurry counterpart as a conditioning
channel at every diffusion step, the model learns the joint statistics of the
(sharp, blurry) pairing. When later used as the proximal operator / prior in
the PnP loop, this suppresses hallucinated sources: the prior itself has been
taught which sharp structures are consistent with a given blurry input.

Why a flat CNN (not U-Net)?
---------------------------
Sharp patches have sub-pixel spatial autocorrelation and an approximately
flat power spectrum -- there is no multi-scale structure for a U-Net's
encoder/decoder pyramid to exploit, and downsampling would destroy the
point-source statistics we care about. Instead we use a DnCNN-style stack
with a dilation pyramid (1,2,3,4,4,3,2,1) giving a 41-pixel receptive field
on 64x64 patches, with FiLM (scale/shift) timestep conditioning.

Data handling
-------------
* Input is a generated patch dataset directory (ml-decon gen_data output):
      {train,val}_observed.npy   (N, 1, 64, 64) float32  -- blurry conditioning
      {train,val}_ideal.npy      (N, 1, 64, 64) float32  -- sharp target
      norm.json                  the normalization the generator applied
* Patches are consumed EXACTLY as stored. NO transform is applied here: no
  flux-ratio correction, no asinh stretch, no domain change of any kind.
  The dataset generator already did sky subtraction, flux alignment, and its
  own normalization; whatever domain the .npy files are in IS the training
  domain. norm.json is copied into every checkpoint as `dataset_norm` so
  downstream consumers can invert to physical flux -- the trainer itself
  never crosses that seam.
  (The previous version fitted its own asinh with b ~ 1.3e-6, which crushed
  the bright end so hard that MSE barely saw errors on source peaks -- one
  of the two causes of the smoothing / dynamic-range loss observed when the
  checkpoint was used inside RED. Gone by construction now.)
* Train/val split is the dataset's own spatial-block split (the generator
  guarantees it); no scene logic here.
* Augmentation is on the fly: a random D4 transform (rot90 k + optional
  flip) applied JOINTLY to each (observed, ideal) pair. The old pipeline
  baked 7 variants into filenames; the .npy dataset stores each patch once.

Usage
-----
    cd ~/noir_ml/mycode
    python train_conditional_diffusion.py \
        --data-dir /home/alex/noir_ml/global/ml-decon/data/m31bK50 \
        --epochs 100 --batch-size 32

    # Sanity check first (memorize a single batch; loss should -> ~0):
    python train_conditional_diffusion.py --overfit-one-batch

Requires: torch, numpy. Optional: ema_pytorch (falls back to a built-in
EMA if absent).
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

class SkyAug:
    """Random sky pedestal added to the OBSERVED channel only.

    Each 64px patch of M32 sits at a different galactocentric radius and so
    carries a different diffuse pedestal -- measured 0.0001 to 0.052 in flux
    across the field, a 250x range. The conditional model has to infer that
    level from the patch alone, and it gets only 87% of the way: the residue
    left in the reconstruction tracks the true sky at r = +0.93, and IS the
    non-zero background floor (bkg0 0.00072 / zero% 15.2 against the truth's
    0.00000 / 94.7).

    Adding a random constant to the observed while leaving the ideal
    untouched breaks the association: the same target now appears under many
    pedestals, so "read the sky off the context" stops minimizing the loss
    and the only remaining strategy is to subtract whatever pedestal is
    present. The invariance being trained is f(observed + c) = f(observed).

    A CONSTANT is the physically right shape. The diffuse component varies by
    1.74x across the whole 2048^2 frame, so over one 64px patch it really is
    near-constant.

    TWO THINGS THIS CLASS EXISTS TO GET RIGHT
    -----------------------------------------
    1. THE OFFSET IS ADDED IN FLUX SPACE, NOT IN THE STORED z DOMAIN. The
       domain is asinh, so those are not the same operation -- not even
       close. For m32_klong_nosky, c = 0.01 moves the background (z = 0.15)
       by +0.188 but a bright peak (z = 0.90) by only +0.0017, a factor of
       ~110. That compression is exactly what real sky does: it lifts the
       background and leaves stellar peaks alone. Adding c directly in z
       would instead brighten every star, which is a different and wrong
       augmentation.

    2. THE NORMALIZATION IS FROZEN. These params come from the dataset's
       norm.json once, at construction. They must never be refitted per
       augmentation: `fit_normalization` derives median and beta FROM the
       data, so an affine change of the input produces the exact affine
       image of the transform and (x - median) / beta is invariant -- the
       offset would cancel and the augmentation would silently do nothing.
       That is not hypothetical; it is why removing gen_data's global
       gain/sky step was verified byte-level a no-op (max diff 1.8e-07).

    Note the augmentation only ever ADDS, on top of each patch's native sky,
    so the training distribution is shifted upward relative to the field. It
    teaches invariance over [native, native + hi], which is the reachable
    direction -- there is no way to subtract a pedestal that was never
    measured per patch.

    Arithmetic mirrors red_pnp_deconvolve.TorchNorm exactly, in float64.
    """

    def __init__(self, norm_params: dict, lo: float, hi: float):
        method = norm_params.get("method")
        if method not in ("asinh", "linear"):
            sys.exit(f"[sky-aug] unsupported normalization {method!r}; "
                     f"the flux<->z map is required to place an offset in "
                     f"flux space")
        if not 0 < lo <= hi:
            sys.exit(f"[sky-aug] need 0 < --sky-aug-lo <= --sky-aug-hi, "
                     f"got {lo} and {hi}")
        self.method = method
        self.p = {k: float(v) for k, v in norm_params.items() if k != "method"}
        self.lo, self.hi = float(lo), float(hi)
        self.log_lo, self.log_hi = math.log(self.lo), math.log(self.hi)

    def to_flux(self, z):
        p = self.p
        if self.method == "asinh":
            s = z * (p["hi_s"] - p["lo_s"]) + p["lo_s"]
            return p["median"] + p["beta"] * np.sinh(s)
        return z * (p["hi"] - p["lo"]) + p["lo"]

    def to_z(self, x):
        p = self.p
        if self.method == "asinh":
            s = np.arcsinh((x - p["median"]) / p["beta"])
            return (s - p["lo_s"]) / (p["hi_s"] - p["lo_s"])
        return (x - p["lo"]) / (p["hi"] - p["lo"])

    def draw(self) -> float:
        """Log-uniform over [lo, hi] -- the pedestal spans 250x, so uniform
        would put almost every draw in the top decade."""
        # torch's RNG is seeded per DataLoader worker; numpy's is not.
        u = float(torch.rand(1).item())
        return math.exp(self.log_lo + u * (self.log_hi - self.log_lo))

    def apply(self, observed: np.ndarray) -> np.ndarray:
        z = observed.astype(np.float64)
        return self.to_z(self.to_flux(z) + self.draw()).astype(np.float32)

    def report(self) -> str:
        """The z-shift this actually injects, at four brightness levels.

        Printed at setup because the flux->z compression is the whole point
        and is not something to take on trust.
        """
        lines = [f"[sky-aug] flux offset log-uniform in "
                 f"[{self.lo:.5g}, {self.hi:.5g}]; resulting z shift:",
                 f"          {'z_in':>6} {'flux':>10}  "
                 + "  ".join(f"c={c:<8.4g}" for c in (self.lo, self.hi))]
        for z in (0.15, 0.30, 0.60, 0.90):
            x = float(self.to_flux(np.float64(z)))
            shifts = "  ".join(
                f"{float(self.to_z(np.float64(x + c))) - z:>+10.4f}"
                for c in (self.lo, self.hi))
            lines.append(f"          {z:>6.2f} {x:>10.5f}  {shifts}")
        lines.append("          (background lifts, bright peaks barely move "
                     "-- that is the asinh doing its job)")
        return "\n".join(lines)


class NpyPairDataset(Dataset):
    """Yields (ideal, observed) tensors straight from the generated arrays.

    ideal    -> the sharp target the diffusion model learns to generate
    observed -> the blurry conditioning channel

    Values pass through untouched -- the arrays on disk are already in their
    final (generator-normalized) domain and this class must never rescale,
    stretch, or otherwise re-map them. Augmentation (if enabled) is a random
    D4 element applied identically to both members of the pair, so the
    observed/ideal registration is preserved.

    `sky_aug` is the one deliberate exception to "values pass through
    untouched", and it is ASYMMETRIC by design: a random pedestal is added to
    the observed only, never to the ideal. See SkyAug.
    """

    def __init__(self, observed: np.ndarray, ideal: np.ndarray,
                 augment: bool = False, sky_aug: "SkyAug | None" = None):
        if observed.shape != ideal.shape:
            sys.exit(f"[data] observed {observed.shape} and ideal "
                     f"{ideal.shape} arrays must have identical shapes")
        self.observed = np.ascontiguousarray(observed, dtype=np.float32)
        self.ideal = np.ascontiguousarray(ideal, dtype=np.float32)
        self.augment = augment
        self.sky_aug = sky_aug

    def __len__(self):
        return len(self.observed)

    def __getitem__(self, idx):
        observed = self.observed[idx]   # (1, H, W)
        ideal = self.ideal[idx]
        if self.augment:
            # torch's RNG is seeded per DataLoader worker, unlike numpy's.
            k = int(torch.randint(0, 8, (1,)).item())
            if k % 4:
                observed = np.rot90(observed, k % 4, axes=(-2, -1))
                ideal = np.rot90(ideal, k % 4, axes=(-2, -1))
            if k >= 4:
                observed = np.flip(observed, axis=-1)
                ideal = np.flip(ideal, axis=-1)
        if self.sky_aug is not None:
            # Observed only. A constant commutes with D4, so the order here
            # is presentation, not correctness.
            observed = self.sky_aug.apply(observed)
        return (
            torch.from_numpy(np.ascontiguousarray(ideal)),
            torch.from_numpy(np.ascontiguousarray(observed)),
        )


def load_split(data_dir: Path, split: str):
    observed = np.load(data_dir / f"{split}_observed.npy")
    ideal = np.load(data_dir / f"{split}_ideal.npy")
    print(f"[data] {split}: {len(observed)} pairs {tuple(observed.shape[1:])} "
          f"observed[{observed.min():+.4f}, {observed.max():+.4f}] "
          f"ideal[{ideal.min():+.4f}, {ideal.max():+.4f}]")
    return observed, ideal


# ----------------------------------------------------------------------------
# Cosine noise schedule (Nichol & Dhariwal 2021, as used in the DPS paper)
# ----------------------------------------------------------------------------

def cosine_alpha_bar(T: int, s: float = 0.008,
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """alpha_bar[t], t = 0..T-1. Always COMPUTED in float64.

    `dtype` sets only the return dtype. It defaults to float32 so training is
    bit-for-bit unchanged; the solvers pass float64 so that the schedule they
    derive sigma(t) from is not pre-rounded to ~1e-7 relative. That rounding
    matters downstream because a normalized-domain error is amplified by
    dmag/dz = 14.475 at the bright end of this dataset's asinh transform.
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    betas = betas.clamp(1e-8, 0.999)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0).to(dtype)


# ----------------------------------------------------------------------------
# Model: flat CNN denoiser with FiLM timestep conditioning
# ----------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # This module has no parameters, so `model.to(float64)` had no way to
        # reach it and the hardcoded .float() below produced a float32 embedding
        # that then hit float64 Linear weights ("mat1 and mat2 must have the
        # same dtype"). A non-persistent buffer tracks the module dtype without
        # entering state_dict, so existing checkpoints load unchanged.
        self.register_buffer("_dtype_probe", torch.zeros(1), persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        dt = self._dtype_probe.dtype
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=dt) / half
        )
        args = t.to(dt)[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FiLMConvBlock(nn.Module):
    """Conv -> norm -> FiLM(gamma, beta from t-embedding) -> SiLU.

    norm="none" replaces the GroupNorm with nn.Identity.  GroupNorm normalizes
    per-sample over (C/8, H, W), which has two measured costs (see
    diag_scale_invariance.py):

      * it divides out ABSOLUTE SCALE.  A 4x input reaches blocks.0.norm at
        std ratio 3.74 and leaves it at 1.005 -- gone at the first block.  What
        survives arrives only via the residual skip below, and the net responds
        x2.03 to a x4 input instead of x4.
      * it makes behaviour depend on H,W.  The same patch run alone vs as a
        quadrant of a 128x128 mosaic differs by 3.7% over the central region
        the 41px receptive field cannot see the seam from, so this otherwise
        fully-convolutional net does not reproduce patch results on a frame.

    Identity is used rather than deleting the attribute so that the same
    forward hooks (and the same diagnostic) work unchanged on both variants.
    """

    def __init__(self, channels: int, dilation: int, t_dim: int,
                 norm: str = "group"):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3,
                              padding=dilation, dilation=dilation)
        self.norm = (nn.GroupNorm(8, channels) if norm == "group"
                     else nn.Identity())
        self.film = nn.Linear(t_dim, 2 * channels)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)  # start as identity modulation

    def forward(self, x, t_emb):
        h = self.norm(self.conv(x))
        gamma, beta = self.film(t_emb).chunk(2, dim=-1)
        h = h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return F.silu(h) + x  # residual


class ConditionalFlatCNN(nn.Module):
    """DnCNN-style flat denoiser, dilation pyramid 1,2,3,4,4,3,2,1
    (41-pixel receptive field), no downsampling. Predicts the noise eps.

    Input channels: [x_t (noisy sharp), y (clean blurry conditioning)].
    The blurry channel is NEVER noised -- it is a fixed conditioning signal
    at every timestep, which is what injects the pairing statistics.
    """

    DILATIONS = (1, 2, 3, 4, 4, 3, 2, 1)

    def __init__(self, channels: int = 64, t_dim: int = 128,
                 norm: str = "group"):
        super().__init__()
        # Recorded so consumers can rebuild the right variant from a
        # checkpoint's arch dict; "group" is the pre-flag behaviour.
        self.norm_kind = norm
        self.t_embed = nn.Sequential(
            SinusoidalTimeEmbedding(t_dim),
            nn.Linear(t_dim, t_dim), nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self.head = nn.Conv2d(2, channels, 3, padding=1)  # 2 in-channels
        self.blocks = nn.ModuleList(
            FiLMConvBlock(channels, d, t_dim, norm=norm) for d in self.DILATIONS
        )
        self.tail = nn.Conv2d(channels, 1, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)  # predict ~0 noise at init

    def forward(self, x_t, y_cond, t):
        t_emb = self.t_embed(t)
        h = self.head(torch.cat([x_t, y_cond], dim=1))
        for block in self.blocks:
            h = block(h, t_emb)
        return self.tail(h)


class ResBlock(nn.Module):
    """U-Net residual block. Zero-init second conv => starts as identity.

    Ported from train_psf_conditional_diffusion.py with the PSF conditioning
    removed; `cond` here is the time embedding alone.
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int,
                 norm: str = "none"):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch) if norm == "group" else nn.Identity()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.film = nn.Linear(cond_dim, 2 * out_ch)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.norm2 = nn.GroupNorm(8, out_ch) if norm == "group" else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = (nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch
                     else nn.Identity())

    def forward(self, x, cond):
        h = self.conv1(F.silu(self.norm1(x)))
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ConditionalUNet(nn.Module):
    """Downsampling denoiser, drop-in for ConditionalFlatCNN.

    Same contract: forward(x_t, y_cond, t) -> predicted eps, 2 input channels,
    zero-init tail. Ported from PsfConditionalUNet with the PSF encoder,
    cross-attention and psf_map channel stripped out, so the ONLY difference
    from the flat control is the backbone.

    WHY IT MIGHT WIN. The flat stack's dilation pyramid gives a 41-px receptive
    field. The bottleneck here sees the whole 64-px patch. The one place that
    should matter on m32 is the sky leak -- the model leaks 13% of each patch's
    sky into its sources, and a 41-px window genuinely cannot tell a smooth
    pedestal from faint extended flux. That is the metric that would justify
    this arch; not val loss, and not completeness.

    WHY IT MIGHT LOSE, stated in advance so the result is falsifiable:

      * The zero floor. The whole normless finding is an EXACTLY-zero
        background (72.3% of pixels). Every output pixel here is a blend of
        upsampled features, so exact zeros are much less likely to survive.
        PREDICTION: zero% comes in BELOW the flat CNN's. If it does not, that
        is the interesting result.
      * Shift-equivariance. Stride-2 convs are equivariant only to EVEN
        translations, so a source sits at a different phase relative to the
        pooling grid in a 64-px patch than in the 512-px frame. The flat CNN's
        patchwise/global agreement (fwd_resid 0.1762 vs 0.1705) is a property
        of that arch and should not be expected to carry over.
      * Size divisibility. len(ch_mult)=3 means H,W must be divisible by 4.
        64 and 512 both are; an odd crop would not be.

    Nearest-neighbour upsampling (not transposed conv) avoids checkerboard
    artifacts, which on a star field would be indistinguishable from faint
    sources.
    """

    # Defaults are PARAM-MATCHED to ConditionalFlatCNN(channels=64): 498k vs
    # 462k (1.08x). A closer fit exists at base=12, ch_mult=(1,2,4),
    # blocks_per_level=1 (451k, 0.97x) but chans would be 12/24/48 and
    # GroupNorm(8, 12) is invalid, so the --norm group ablation could not run.
    # Every channel count here (16/32/32) is divisible by 8.
    def __init__(self, base: int = 16, ch_mult=(1, 2, 2), t_dim: int = 128,
                 norm: str = "none", blocks_per_level: int = 2):
        super().__init__()
        self.norm_kind = norm
        self.base = base
        self.ch_mult = tuple(ch_mult)
        self.blocks_per_level = blocks_per_level
        self.t_embed = nn.Sequential(
            SinusoidalTimeEmbedding(t_dim),
            nn.Linear(t_dim, t_dim), nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        chans = [base * m for m in self.ch_mult]
        self.head = nn.Conv2d(2, chans[0], 3, padding=1)  # [x_t, y_cond]

        self.down = nn.ModuleList()
        self.downsample = nn.ModuleList()
        skip_chans = [chans[0]]
        cur = chans[0]
        for li, ch in enumerate(chans):
            level = nn.ModuleList()
            for _ in range(blocks_per_level):
                level.append(ResBlock(cur, ch, t_dim, norm=norm))
                cur = ch
                skip_chans.append(cur)
            self.down.append(level)
            if li < len(chans) - 1:
                self.downsample.append(nn.Conv2d(cur, cur, 3, stride=2,
                                                 padding=1))
                skip_chans.append(cur)
            else:
                self.downsample.append(None)

        self.mid1 = ResBlock(cur, cur, t_dim, norm=norm)
        self.mid2 = ResBlock(cur, cur, t_dim, norm=norm)

        self.up = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for li, ch in reversed(list(enumerate(chans))):
            level = nn.ModuleList()
            for _ in range(blocks_per_level + 1):
                level.append(ResBlock(cur + skip_chans.pop(), ch, t_dim,
                                      norm=norm))
                cur = ch
            self.up.append(level)
            self.upsample.append(nn.Conv2d(cur, cur, 3, padding=1)
                                 if li > 0 else None)

        self.tail = nn.Conv2d(cur, 1, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x_t, y_cond, t):
        cond = self.t_embed(t)
        z = self.head(torch.cat([x_t, y_cond], dim=1))
        skips = [z]
        for level, ds in zip(self.down, self.downsample):
            for block in level:
                z = block(z, cond)
                skips.append(z)
            if ds is not None:
                z = ds(z)
                skips.append(z)

        z = self.mid1(z, cond)
        z = self.mid2(z, cond)

        for level, us in zip(self.up, self.upsample):
            for block in level:
                z = block(torch.cat([z, skips.pop()], dim=1), cond)
            if us is not None:
                z = us(F.interpolate(z, scale_factor=2, mode="nearest"))
        return self.tail(z)


def build_model(kind: str, **kw) -> nn.Module:
    """Single construction point for both architectures.

    Consumers MUST go through this (or model_from_checkpoint) rather than
    naming a class: guessing the arch wrong gives either a load error or, with
    strict=False anywhere, a silently partly-random network.
    """
    if kind == "flat":
        kw.pop("base", None); kw.pop("ch_mult", None)
        kw.pop("blocks_per_level", None)
        return ConditionalFlatCNN(**kw)
    if kind == "unet":
        kw.pop("channels", None)
        return ConditionalUNet(**kw)
    raise SystemExit(f"[model] unknown arch {kind!r}")


def model_from_checkpoint(ckpt: dict) -> nn.Module:
    """Rebuild the exact architecture a checkpoint was trained with.

    `kind` is absent from every pre-flag checkpoint and those are all flat, so
    the default preserves existing behaviour exactly -- the same convention
    `norm` already uses.
    """
    a = dict(ckpt.get("arch", {}))
    kind = a.pop("kind", "flat")
    a.pop("in_channels", None)
    a.pop("dilations", None)
    a.setdefault("norm", "group")   # pre-flag checkpoints are GroupNorm
    if kind == "unet" and isinstance(a.get("ch_mult"), list):
        a["ch_mult"] = tuple(a["ch_mult"])
    return build_model(kind, **a)


# ----------------------------------------------------------------------------
# EMA (uses ema_pytorch if installed, else a minimal fallback)
# ----------------------------------------------------------------------------

class SimpleEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(),
                                                     alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self):
        return self.shadow


def make_ema(model):
    try:
        from ema_pytorch import EMA
        return EMA(model, beta=0.999, update_every=1), True
    except ImportError:
        return SimpleEMA(model), False


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def diffusion_loss(model, sharp, blurry, alpha_bar, device, p_uncond=0.0,
                   identity_frac=0.0, identity_t_max=100,
                   low_t_frac=0.0, low_t_max=100):
    """eps-prediction DDPM loss, conditioned on blurry.

    p_uncond > 0 enables CLASSIFIER-FREE GUIDANCE training: that fraction of
    the batch has its conditioning channel replaced by the NULL token (all
    zeros), so the same weights learn BOTH
        eps_theta(x_t, y, t)     the conditional/posterior score, and
        eps_theta(x_t, 0, t)     a genuine UNCONDITIONAL prior score.

    Why this matters for the PnP use: RED / Graikos regularizers assume the
    denoiser models the PRIOR p(x). A purely conditional denoiser gives the
    POSTERIOR score, which already contains the likelihood -- using it next
    to an explicit ||y - Ax||^2 term double-counts the measurement. With
    condition dropout you can call the model with the null token to get the
    clean prior score, and keep the conditional path for sampling/anchoring.

    All-zeros is a safe null token here: real blurry patches are never
    identically zero, so it is unambiguously distinguishable.
    """
    bsz = sharp.shape[0]
    T = alpha_bar.shape[0]
    t = torch.randint(0, T, (bsz,), device=device)
    if low_t_frac > 0:
        # LOW-t OVERSAMPLING: the RED/PnP prox only ever calls the denoiser
        # at t <= ~75, but uniform t gives that regime just low_t_max/T
        # (~10%) of the gradient signal. The smoothing seen in RED is a
        # low-t accuracy problem: at small t the model must resolve
        # point-source peaks against faint noise, the hardest regime and the
        # one uniform sampling trains least. Redirect a fraction of the
        # batch there. (Unlike the identity rows below, these keep their
        # noise -- they are ordinary denoising problems, just concentrated
        # where the model is actually used.)
        low = (torch.rand(bsz, device=device) < low_t_frac)
        t_low = torch.randint(0, max(int(low_t_max), 1), (bsz,),
                              device=device)
        t = torch.where(low, t_low, t)
    ab = alpha_bar[t][:, None, None, None]
    eps = torch.randn_like(sharp)
    if identity_frac > 0:
        # IDENTITY / fixed-point term: this fraction of the batch gets ZERO
        # noise, so x_t = sqrt(abar)*sharp and the target eps is exactly 0.
        # It teaches D(x, y) ~ x on a clean sharp field.
        #
        # Why it matters here: the RED gradient is x - D(x), so if the model
        # is not a fixed point on the sharp manifold, RED pulls AWAY from the
        # truth even when x is already correct. Measured on the current
        # conditional checkpoint (trained WITHOUT this term):
        #     ||D(x)-x||/||x|| on a true sharp patch = 0.109 at t=5,
        #     0.190 at t=20, 0.400 at t=50; flux shrinks to 0.917 / 0.857 /
        #     0.678. The unconditional v3 model, trained WITH the term,
        #     gives 0.003 / 0.023 / 0.120 and flux 1.002 / 1.016 / 1.083.
        # That bias is baked into the denoiser and no choice of lambda in the
        # RED objective can remove it.
        #
        # The identity rows also get their t RESAMPLED into [0, identity_t_max).
        # A zero-noise draw at t=900 is just 0.03*sharp -- a near-black frame
        # that teaches nothing about the sharp manifold and never occurs in
        # the prox, which runs at t<=~75. Only the non-identity rows keep the
        # uniform t; the rest of the batch's distribution is untouched.
        idm = (torch.rand(bsz, device=device) < identity_frac)
        eps = torch.where(idm[:, None, None, None],
                          torch.zeros_like(eps), eps)
        t_low = torch.randint(0, max(int(identity_t_max), 1), (bsz,),
                              device=device)
        t = torch.where(idm, t_low, t)
        ab = alpha_bar[t][:, None, None, None]
    x_t = ab.sqrt() * sharp + (1 - ab).sqrt() * eps
    if p_uncond > 0:
        drop = (torch.rand(bsz, device=device) < p_uncond)
        blurry = torch.where(drop[:, None, None, None],
                             torch.zeros_like(blurry), blurry)
    eps_pred = model(x_t, blurry, t)
    return F.mse_loss(eps_pred, eps)


@torch.no_grad()
def validate(model, loader, alpha_bar, device):
    model.eval()
    total, n = 0.0, 0
    for sharp, blurry in loader:
        sharp, blurry = sharp.to(device), blurry.to(device)
        loss = diffusion_loss(model, sharp, blurry, alpha_bar, device)
        total += loss.item() * sharp.shape[0]
        n += sharp.shape[0]
    model.train()
    return total / max(n, 1)


def save_checkpoint(path, model, ema, ema_is_lib, optimizer, epoch,
                    dataset_norm, args):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "ema_state": (ema.ema_model.state_dict() if ema_is_lib
                      else ema.state_dict()),
        "optimizer_state": optimizer.state_dict(),
        # NO trainer-side transform exists any more ("transform" key is
        # deliberately absent so stale consumers fail loudly instead of
        # silently applying the old asinh seam). The model operates in the
        # dataset's stored domain; `dataset_norm` is the generator's
        # norm.json, which is the ONLY map between that domain and physical
        # flux. Inverting it is the consumer's job.
        "dataset_norm": dataset_norm,
        "data_dir": str(args.data_dir),
        # "norm" MUST stay in here: a normless checkpoint loaded into a
        # GroupNorm model has no blocks.*.norm.* keys, and any consumer that
        # loads with strict=False would silently run a partly-random network.
        # Absent (pre-flag checkpoints) means "group".
        # "kind" MUST stay in here for the same reason "norm" does: a unet
        # state_dict loaded into a flat CNN shares almost no keys, and any
        # consumer using strict=False would run a nearly-random network.
        # Absent (pre-flag checkpoints) means "flat", as absent norm means
        # "group". Only the keys the chosen arch actually takes are written,
        # so build_model never receives a value it would have to guess at.
        "arch": ({"kind": "flat", "channels": args.channels,
                  "dilations": list(ConditionalFlatCNN.DILATIONS),
                  "t_dim": 128, "in_channels": 2, "norm": args.norm}
                 if args.arch == "flat" else
                 {"kind": "unet", "base": args.base,
                  "ch_mult": [int(v) for v in args.ch_mult.split(",")],
                  "blocks_per_level": args.blocks_per_level,
                  "t_dim": 128, "in_channels": 2, "norm": args.norm}),
        "diffusion": {"timesteps": args.timesteps, "schedule": "cosine"},
        # p_uncond > 0 means model(x_t, zeros, t) is a valid UNCONDITIONAL
        # prior score (classifier-free guidance); consumers should check this
        # before using the null-token path.
        "p_uncond": args.p_uncond,
        "identity_frac": args.identity_frac,
        "identity_t_max": args.identity_t_max,
        "low_t_frac": args.low_t_frac,
        "low_t_max": args.low_t_max,
        # A sky-augmented model was trained to be INVARIANT to an additive
        # flux pedestal on its conditioning channel. That is a different
        # function from the un-augmented one, so record it: it is the
        # explanation a consumer needs when the background floor moves.
        "sky_aug": ({"lo": args.sky_aug_lo, "hi": args.sky_aug_hi,
                     "space": "flux", "draw": "log-uniform"}
                    if args.sky_aug else None),
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
    }
    torch.save(ckpt, path)


def main():
    ap = argparse.ArgumentParser(
        description="Train a conditional flat-CNN DDPM on observed/ideal "
                    ".npy patch pairs (ml-decon gen_data output).")
    ap.add_argument("--data-dir", type=Path,
                    default=Path("/home/alex/noir_ml/global/ml-decon/data/m31bK50"),
                    help="directory holding {train,val}_{observed,ideal}.npy "
                         "and norm.json")
    # New default dir: the old checkpoints_cond_diffusion/*.pt use the
    # retired 'transform' (flux_ratio, asinh_b, asinh_A) format and would be
    # clobbered by the new-format files.
    ap.add_argument("--checkpoint-dir", type=Path,
                    default=Path("checkpoints_cond_diffusion_npy"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--channels", type=int, default=64,
                    help="flat arch: width of every block.")
    ap.add_argument("--arch", choices=["flat", "unet"], default="flat",
                    help="'flat' is the dilated stack every previous result "
                         "was measured on: 41-px receptive field, no "
                         "downsampling, shift-equivariant, applies unchanged "
                         "to a whole frame. 'unet' downsamples so its "
                         "bottleneck sees the entire patch. Defaults are "
                         "param-matched (462k vs 498k) so a win cannot be "
                         "capacity. Compare on photometry and the sky-leak "
                         "slope, NOT on val loss -- the two archs do not have "
                         "the same loss landscape.")
    ap.add_argument("--base", type=int, default=16,
                    help="unet arch: width at full resolution.")
    ap.add_argument("--ch-mult", type=str, default="1,2,2",
                    help="unet arch: per-level channel multipliers. Length "
                         "sets the number of levels, so H,W must be divisible "
                         "by 2**(len-1).")
    ap.add_argument("--blocks-per-level", type=int, default=2,
                    help="unet arch: ResBlocks per encoder level.")
    ap.add_argument("--norm", choices=["group", "none"], default="group",
                    help="normalization inside FiLMConvBlock. 'group' is the "
                         "existing GroupNorm(8, C). 'none' replaces it with "
                         "Identity: GroupNorm divides out absolute scale at "
                         "the FIRST block (measured 3.74 in, 1.005 out for a "
                         "4x input) and makes behaviour depend on H,W (3.7%% "
                         "difference between a patch run alone and the same "
                         "patch inside a 128px mosaic). film and tail are "
                         "already zero-init'd, so the residual stack trains "
                         "without it. Verify with diag_scale_invariance.py.")
    ap.add_argument("--p-uncond", type=float, default=0.15,
                    help="classifier-free-guidance dropout: fraction of "
                         "training samples whose blurry conditioning is "
                         "replaced by the null token (zeros). >0 makes the "
                         "SAME weights usable as an unconditional prior via "
                         "model(x_t, zeros, t) -- required if you want to "
                         "use this model in a RED/Graikos regularizer "
                         "alongside an explicit ||y-Ax||^2 term without "
                         "double-counting the likelihood. 0 = purely "
                         "conditional (old behaviour).")
    ap.add_argument("--identity-frac", type=float, default=0.15,
                    help="fraction of training draws with ZERO noise and "
                         "target eps=0, teaching D(x,y) ~ x on clean sharp "
                         "fields. Required for the RED gradient x - D(x) to "
                         "vanish on the truth; without it the denoiser has a "
                         "biased fixed point (measured: it removes 8-32%% of "
                         "the flux from a true sharp patch at t=5..50) and no "
                         "lambda can correct that. 0 = old behaviour.")
    ap.add_argument("--identity-t-max", type=int, default=100,
                    help="identity draws are placed at t ~ U[0, this). The "
                         "PnP/RED prox runs at t<=~75, so that is where the "
                         "fixed point has to hold.")
    ap.add_argument("--low-t-frac", type=float, default=0.25,
                    help="fraction of NOISY training draws whose t is "
                         "resampled into [0, --low-t-max). The RED prox only "
                         "calls the denoiser at t<=~75; uniform t starves "
                         "that regime, and low-t point-source recovery is "
                         "exactly where the observed smoothing lives. "
                         "0 = old uniform behaviour.")
    ap.add_argument("--low-t-max", type=int, default=100,
                    help="upper bound (exclusive) for the oversampled low-t "
                         "range; match to the largest t the prox uses.")
    ap.add_argument("--no-augment", action="store_true",
                    help="disable the on-the-fly D4 (rot90/flip) pair "
                         "augmentation of the training split.")
    ap.add_argument("--sky-aug", action="store_true",
                    help="add a random sky pedestal to the OBSERVED channel "
                         "only (ideal untouched), teaching the invariance "
                         "f(observed + c) = f(observed). Targets the "
                         "background floor: the model currently removes only "
                         "87%% of each patch's sky and the 13%% it leaves "
                         "tracks the true sky at r=+0.93, holding the "
                         "reconstruction's background off zero (bkg0 0.00072 "
                         "/ zero%% 15.2 vs the truth's 0.00000 / 94.7). The "
                         "offset is applied in FLUX space through the "
                         "dataset's FROZEN norm.json -- see SkyAug for why "
                         "both of those words are load-bearing. Requires "
                         "norm.json. Train split only; val is never shifted, "
                         "so val loss stays comparable across runs.")
    ap.add_argument("--sky-aug-lo", type=float, default=1e-4,
                    help="low end of the log-uniform flux offset. The default "
                         "matches the faintest per-patch pedestal measured on "
                         "the m32 field (0.0001) and is close enough to a "
                         "no-op that the un-augmented regime stays in the "
                         "training distribution.")
    ap.add_argument("--sky-aug-hi", type=float, default=5e-2,
                    help="high end of the log-uniform flux offset; the "
                         "default matches the brightest per-patch pedestal "
                         "measured on the m32 field (0.052). Draws are "
                         "log-uniform because the range spans 250x and a "
                         "uniform draw would put nearly every sample in the "
                         "top decade.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--overfit-one-batch", action="store_true",
                    help="Sanity check: memorize a single batch and track a "
                         "deterministic fixed-noise eval loss.")
    ap.add_argument("--sanity-steps", type=int, default=3000,
                    help="Steps for the overfit-one-batch check.")
    ap.add_argument("--sanity-lr", type=float, default=3e-4,
                    help="Learning rate for the overfit-one-batch check "
                         "(hotter than the training default, appropriate "
                         "for pure memorization).")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)
    print(f"[setup] device = {device}"
          + ("  (CPU: fine for --overfit-one-batch verification; use the GPU "
             "cluster for the full run)" if device.type == "cpu" else ""))

    # --- data -----------------------------------------------------------
    # Arrays are used exactly as stored: the generator owns the domain.
    # norm.json is read BEFORE the datasets are built: --sky-aug needs the
    # frozen flux<->z map to place its offset in flux space.
    norm_path = args.data_dir / "norm.json"
    if norm_path.exists():
        dataset_norm = json.loads(norm_path.read_text())
        print(f"[data] dataset norm ({norm_path}): {dataset_norm}")
    else:
        dataset_norm = None
        print(f"[data] WARNING: no norm.json in {args.data_dir}; checkpoints "
              f"will not carry the normalized->physical map and downstream "
              f"photometry cannot be inverted to flux.")

    sky_aug = None
    if args.sky_aug:
        if dataset_norm is None:
            sys.exit(f"[sky-aug] --sky-aug requires norm.json in "
                     f"{args.data_dir}: the offset is a FLUX offset and there "
                     f"is no way to map it into the stored domain without the "
                     f"generator's transform.")
        sky_aug = SkyAug(dataset_norm["observed"],
                         args.sky_aug_lo, args.sky_aug_hi)
        print(sky_aug.report())

    train_ds = NpyPairDataset(*load_split(args.data_dir, "train"),
                              augment=not args.no_augment, sky_aug=sky_aug)
    # val is deliberately NOT sky-shifted: it is the fixed yardstick across
    # runs, and the effect this augmentation targets is measured end-to-end
    # by score_recon.py's bkg0/zero% columns, not by val loss.
    val_ds = NpyPairDataset(*load_split(args.data_dir, "val"), augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"),
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # --- model / optim --------------------------------------------------
    ch_mult = tuple(int(v) for v in args.ch_mult.split(","))
    model = build_model(args.arch, channels=args.channels, norm=args.norm,
                        base=args.base, ch_mult=ch_mult,
                        blocks_per_level=args.blocks_per_level).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if args.arch == "flat":
        print(f"[model] ConditionalFlatCNN, {n_params/1e6:.3f}M params, "
              f"dilations {ConditionalFlatCNN.DILATIONS} (41-px receptive "
              f"field), norm={args.norm}")
    else:
        div = 2 ** (len(ch_mult) - 1)
        print(f"[model] ConditionalUNet, {n_params/1e6:.3f}M params, base "
              f"{args.base}, ch_mult {ch_mult}, {args.blocks_per_level} "
              f"blocks/level, norm={args.norm}")
        print(f"[model] H,W must be divisible by {div}. Downsampling is NOT "
              f"shift-equivariant, so patchwise and whole-frame inference are "
              f"no longer expected to agree the way the flat arch does.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-5)
    ema, ema_is_lib = make_ema(model)
    alpha_bar = cosine_alpha_bar(args.timesteps).to(device)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Convenience copy of the dataset's own normalization next to the
    # checkpoints (replaces the retired trainer-fitted transform.json).
    if dataset_norm is not None:
        with open(args.checkpoint_dir / "dataset_norm.json", "w") as f:
            json.dump(dataset_norm, f, indent=2)

    # --- overfit-one-batch sanity check ---------------------------------
    if args.overfit_one_batch:
        sharp, blurry = next(iter(train_loader))
        sharp, blurry = sharp.to(device), blurry.to(device)

        # Deterministic evaluation pack: FROZEN noise + FIXED timestep grid.
        # The per-step training loss resamples t and eps every iteration, so
        # it bounces around as it draws easy (high-t) or hard (low-t)
        # subproblems -- it is NOT a reliable convergence signal on its own.
        # This eval number is comparable across steps.
        gen = torch.Generator(device="cpu").manual_seed(0)
        eval_eps = torch.randn(sharp.shape, generator=gen).to(device)
        eval_ts = [int(f * args.timesteps) for f in (0.05, 0.25, 0.5, 0.75, 0.95)]

        @torch.no_grad()
        def fixed_eval():
            model.eval()
            losses = []
            for ti in eval_ts:
                t = torch.full((sharp.shape[0],), ti, device=device,
                               dtype=torch.long)
                ab = alpha_bar[t][:, None, None, None]
                x_t = ab.sqrt() * sharp + (1 - ab).sqrt() * eval_eps
                losses.append(F.mse_loss(model(x_t, blurry, t),
                                         eval_eps).item())
            model.train()
            return float(np.mean(losses)), losses

        # A hotter LR is appropriate for pure memorization.
        sanity_opt = torch.optim.AdamW(model.parameters(), lr=args.sanity_lr)
        print(f"[sanity] Overfitting a single batch of {sharp.shape[0]} "
              f"pairs for {args.sanity_steps} steps (lr {args.sanity_lr}).")
        print("[sanity] Reference points: untrained model scores ~1.0 "
              "(variance of eps). Healthy memorization: fixed-eval loss "
              "steadily decreasing, reaching <0.05 by the end. The low-t "
              "entries of the per-t breakdown are the hardest and fall last.")
        model.train()
        running = None
        for step in range(1, args.sanity_steps + 1):
            sanity_opt.zero_grad()
            loss = diffusion_loss(model, sharp, blurry, alpha_bar, device)
            loss.backward()
            sanity_opt.step()
            running = (loss.item() if running is None
                       else 0.98 * running + 0.02 * loss.item())
            if step % 100 == 0:
                ev_mean, ev_per_t = fixed_eval()
                per_t = "  ".join(f"t={t}:{l:.3f}"
                                  for t, l in zip(eval_ts, ev_per_t))
                print(f"  step {step:5d}  train(avg) {running:.4f}  "
                      f"fixed-eval {ev_mean:.4f}   [{per_t}]")
        ev_mean, _ = fixed_eval()
        verdict = ("PASS" if ev_mean < 0.05 else
                   "MARGINAL -- decreasing but not converged; rerun with "
                   "more --sanity-steps" if ev_mean < 0.2 else
                   "FAIL -- check the data ranges printed above")
        print(f"[sanity] Final fixed-eval loss {ev_mean:.4f}: {verdict}")
        return

    # --- full training loop ---------------------------------------------
    best_val = float("inf")
    model.train()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        running, n_seen = 0.0, 0
        for sharp, blurry in train_loader:
            sharp, blurry = sharp.to(device), blurry.to(device)
            optimizer.zero_grad()
            loss = diffusion_loss(model, sharp, blurry, alpha_bar, device,
                                  p_uncond=args.p_uncond,
                                  identity_frac=args.identity_frac,
                                  identity_t_max=args.identity_t_max,
                                  low_t_frac=args.low_t_frac,
                                  low_t_max=args.low_t_max)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if ema_is_lib:
                ema.update()
            else:
                ema.update(model)
            running += loss.item() * sharp.shape[0]
            n_seen += sharp.shape[0]

        train_loss = running / max(n_seen, 1)
        val_loss = validate(model, val_loader, alpha_bar, device)
        dt = time.time() - t0
        print(f"[epoch {epoch:3d}/{args.epochs}] train {train_loss:.5f}  "
              f"val {val_loss:.5f}  ({dt:.1f}s)")

        save_checkpoint(args.checkpoint_dir / "last.pt", model, ema,
                        ema_is_lib, optimizer, epoch, dataset_norm, args)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(args.checkpoint_dir / "best.pt", model, ema,
                            ema_is_lib, optimizer, epoch, dataset_norm, args)
            print(f"          -> new best val loss, saved best.pt")

    print(f"[done] best val loss {best_val:.5f}. Use the 'ema_state' weights "
          f"from best.pt as the prior in the PnP loop. The model works in the "
          f"dataset's stored domain; 'dataset_norm' in the checkpoint is the "
          f"only map back to physical flux.")


# Mandatory guard: this file contains a training loop and must never
# retrain on import (e.g. when pnp_deconvolve.py imports ConditionalFlatCNN).
if __name__ == "__main__":
    main()