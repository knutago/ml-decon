"""
train_psf_conditional_diffusion.py

Conditional DDPM that is conditioned on BOTH the blurry observation AND the
PSF that produced it, so one checkpoint serves any PSF instead of needing a
retrain per instrument/epoch.

What changed vs train_conditional_diffusion.py
----------------------------------------------
1. PSF CONDITIONING.  The denoiser input is
       [ x_t , y_observed , psf_channel(s) ]
   The PSF stamp is rendered into full-patch-sized channels (see
   `psf_to_channels`) so the network sees the actual beam shape -- core width,
   ellipticity, wing slope -- rather than inferring it from y alone.  Train on
   a mix of data directories with different PSFs and the model learns the
   PSF-conditional posterior p(ideal | observed, psf); a new dataset with a
   new PSF is then a forward pass, not a training run.

2. PSF ENCODING + CROSS-ATTENTION (--psf-cross-attn).  In parallel with the
   image channels, a small encoder turns the PSF stamp into a set of tokens.
   The pooled token modulates every block through FiLM alongside the timestep
   embedding; with --psf-cross-attn the spatial features additionally attend
   over the full token set.  This is the forward-compatible path: for a
   SPATIALLY VARYING PSF you keep the identical model and simply feed several
   PSF stamps -- one per field position -- as several token groups carrying
   field-position embeddings (see `PSFEncoder.forward`, which already accepts
   a stamp batch of shape (B, K, C, P, P)).  Cross-attention then lets each
   output pixel select the beam appropriate to its location, which is what
   removes the need for patch-wise deconvolution.  Nothing about the channel
   path has to change for that upgrade.

3. NO GroupNorm.  GroupNorm normalizes per-sample over (C/8, H, W), which
   divides out ABSOLUTE SCALE at the very first block (measured 3.74x in,
   1.005x out for a 4x input) and makes behaviour depend on H,W (3.7%
   difference between a patch run alone and the same patch as a quadrant of a
   128px mosaic).  Both are fatal for photometry, so --norm defaults to
   "none" here.  The flag is retained ONLY so the ablation can still be run;
   it is not the intended setting.  Stability without it comes from zero-init
   output projections, which make every residual block start as the identity.

4. LARGER RECEPTIVE FIELD, and a U-Net option (--arch unet).  Conditioning on
   the blurry image makes the target depend on a neighbourhood as wide as the
   PSF wings, not just the point-source core the old (1,2,3,4,4,3,2,1) stack
   was sized for.  The flat arch's default dilation pyramid is widened to an
   85-pixel receptive field, and --arch unet offers a downsampling backbone
   whose bottleneck sees the whole patch.  Both are provided so they can be
   A/B'd on identical data; `flat` remains the control the previous results
   were measured on.

Data layout
-----------
Pass one or more dataset directories.  Each must contain the ml-decon
gen_data output plus the PSF that dataset's observed images were blurred with:

    <dir>/{train,val}_observed.npy   (N, 1, H, W) float32  blurry conditioning
    <dir>/{train,val}_ideal.npy      (N, 1, H, W) float32  sharp target
    <dir>/norm.json                  the normalization the generator applied
    <dir>/psf.fits | psf.npy | psf*.fits    the PSF for this dataset

If the PSF file is named something else, or lives elsewhere, use the
"DIR:PSF_PATH" form:

    --data-dir data/m31bK50:psf_k50.fits data/m32_klong:psf_klong_epsf.fits

Patches are consumed EXACTLY as stored -- no stretch, no rescale, no domain
change.  The generator owns the domain; each dir's norm.json is carried into
the checkpoint under `dataset_norms` keyed by directory.  MIXING DIRS WITH
DIFFERENT norm.json MEANS MIXING DOMAINS: the trainer prints a loud warning
because the model cannot tell the two apart from pixel values alone, and the
resulting flux calibration is silently wrong downstream.

Augmentation applies a random D4 element (rot90 k + optional flip) JOINTLY to
observed, ideal AND the PSF stamp -- rotating the pair without rotating the
beam would teach the model a false (image, psf) correspondence.  This is why
--psf-size must be odd: for odd P the stamp centre is a fixed point of every
D4 element, for even P the flips shift it by half a pixel.

Usage
-----
    cd ~/noir_ml/mycode
    python train_psf_conditional_diffusion.py \
        --data-dir data/setA data/setB data/setC \
        --arch flat --epochs 100 --batch-size 32

    # sanity check first (memorize one batch; fixed-eval loss should -> ~0):
    python train_psf_conditional_diffusion.py --data-dir data/setA \
        --overfit-one-batch

Requires: torch, numpy, astropy (only for .fits PSFs).
Optional: ema_pytorch (falls back to a built-in EMA if absent).
"""

import argparse
import glob
import json
import math
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
# PSF handling
# ----------------------------------------------------------------------------

PSF_TOKEN_GRID = 4          # PSF encoder emits GRID*GRID + 1 tokens
PSF_LOG_FLOOR = 1e-6        # dynamic range of the log PSF channel, peak-relative


def load_psf_array(path: Path) -> np.ndarray:
    """Read a PSF stamp from .fits or .npy as a 2-D float64 array."""
    if path.suffix.lower() in (".fits", ".fit"):
        from astropy.io import fits
        data = fits.getdata(str(path))
    else:
        data = np.load(path)
    data = np.squeeze(np.asarray(data, dtype=np.float64))
    if data.ndim != 2:
        sys.exit(f"[psf] {path}: expected a 2-D stamp, got shape {data.shape}")
    return data


def find_psf(data_dir: Path) -> Path:
    """Locate the PSF that belongs to a dataset directory.

    Preference order is exact names first, then a unique psf* glob.  An
    ambiguous glob is a hard error rather than a guess: silently picking the
    wrong PSF would train the model on a false (image, beam) pairing and the
    failure would only show up as bad photometry much later.
    """
    for name in ("psf.fits", "psf.npy"):
        if (data_dir / name).exists():
            return data_dir / name
    hits = sorted(glob.glob(str(data_dir / "psf*.fits"))
                  + glob.glob(str(data_dir / "psf*.npy")))
    if len(hits) == 1:
        return Path(hits[0])
    if not hits:
        sys.exit(f"[psf] no PSF found in {data_dir}. Put psf.fits (or psf.npy) "
                 f"there, or use the DIR:PSF_PATH form of --data-dir.")
    sys.exit(f"[psf] {data_dir} has {len(hits)} candidate PSF files "
             f"({', '.join(Path(h).name for h in hits)}); disambiguate with "
             f"the DIR:PSF_PATH form of --data-dir.")


def canonicalize_psf(psf: np.ndarray, size: int) -> np.ndarray:
    """Centre-crop or zero-pad a stamp to (size, size) and renormalize to sum 1.

    Every PSF the model ever sees must live on the same grid, otherwise the
    psf channel means different things for different datasets.  Cropping is
    centred on the stamp centre, so the peak stays at index size//2 as long as
    the input stamp was itself centred (all psf_*.fits in this repo are).
    """
    psf = np.asarray(psf, dtype=np.float64)
    h, w = psf.shape
    out = np.zeros((size, size), dtype=np.float64)
    # source window centred on the stamp centre, clipped to both grids
    sy = (h - size) // 2
    sx = (w - size) // 2
    src_y0, dst_y0 = max(sy, 0), max(-sy, 0)
    src_x0, dst_x0 = max(sx, 0), max(-sx, 0)
    ny = min(h - src_y0, size - dst_y0)
    nx = min(w - src_x0, size - dst_x0)
    out[dst_y0:dst_y0 + ny, dst_x0:dst_x0 + nx] = \
        psf[src_y0:src_y0 + ny, src_x0:src_x0 + nx]
    total = out.sum()
    if not np.isfinite(total) or total <= 0:
        sys.exit("[psf] stamp has non-positive or non-finite sum after "
                 "cropping; check the file and --psf-size")
    return out / total


def psf_to_channels(psf: np.ndarray, repr_kind: str) -> np.ndarray:
    """Turn a unit-sum stamp into the (C, P, P) tensor the network consumes.

    A raw PSF is numerically a delta: peak ~5e-2, wings ~1e-5.  Fed in linearly
    the wings are below the noise floor of the first conv's weights and the
    network effectively sees "a dot" for every beam.  The wings are exactly
    what distinguishes one PSF from another and exactly what drives the
    long-range pixel dependence this model exists to capture, so the default
    representation is peak-normalized log.  "both" keeps the linear core too,
    at the cost of one extra channel.
    """
    p = np.asarray(psf, dtype=np.float32)
    peak = float(np.max(p))
    if peak <= 0:
        sys.exit("[psf] stamp peak is non-positive")
    lin = p / peak                                        # [~0, 1], peak = 1
    log = (np.log10(np.clip(lin, PSF_LOG_FLOOR, None))
           - math.log10(PSF_LOG_FLOOR)) / (-math.log10(PSF_LOG_FLOOR))
    if repr_kind == "linear":
        return lin[None]
    if repr_kind == "log":
        return log.astype(np.float32)[None]
    if repr_kind == "both":
        return np.stack([lin, log.astype(np.float32)], axis=0)
    sys.exit(f"[psf] unknown --psf-repr {repr_kind!r}")


def n_psf_channels(repr_kind: str) -> int:
    return 2 if repr_kind == "both" else 1


def render_psf_map(psf: torch.Tensor, h: int, w: int, mode: str) -> torch.Tensor:
    """Render a (B, C, P, P) stamp into the (B, C, h, w) input channel(s).

    "center" places the stamp in the middle of the canvas.  It is the faithful
    picture of the beam and is correct whenever the input is the size the
    model was trained at -- which for a patch-wise pipeline is always.  It is
    NOT size-independent: on a 128px frame the stamp sits at the frame centre,
    so a corner pixel's receptive field never reaches it and the same quadrant
    scores differently alone than inside the mosaic (measured 50% relative
    difference).

    "tile" repeats the stamp from the (0,0) anchor to fill the canvas, so
    every pixel has a full copy of the beam within P pixels regardless of how
    big the input is.  Use it for whole-frame inference.  It is consistent
    only for crop offsets that are multiples of P, and it paints a periodic
    texture the network must learn to ignore -- which it can, because the
    period is fixed and known.

    Either way the PSF also reaches every block through FiLM (and, with
    --psf-cross-attn, through attention); those paths are spatially uniform
    and completely independent of input size, so PSF information is never
    lost even where this channel is uninformative.
    """
    p = psf.shape[-1]
    if p == h and p == w:
        return psf
    if mode == "tile":
        ry, rx = -(-h // p), -(-w // p)      # ceil-div
        return psf.repeat(1, 1, ry, rx)[..., :h, :w]
    # "center": negative padding crops, so a stamp larger than the canvas is
    # handled by the same expression.
    dy, dx = h - p, w - p
    return F.pad(psf, (dx // 2, dx - dx // 2, dy // 2, dy - dy // 2))


def d4(arr: np.ndarray, k: int) -> np.ndarray:
    """Apply D4 element k in {0..7}: rot90 (k%4) then flip if k>=4.

    Used identically on observed, ideal and the PSF stamp so the three stay
    mutually consistent.
    """
    if k % 4:
        arr = np.rot90(arr, k % 4, axes=(-2, -1))
    if k >= 4:
        arr = np.flip(arr, axis=-1)
    return arr


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

class PsfPairDataset(Dataset):
    """Yields (ideal, observed, psf_channels) across one or more datasets.

    Pixel values pass through untouched -- the arrays on disk are already in
    their final generator-normalized domain and this class must never rescale
    or stretch them.  Each sample carries the PSF of the dataset it came from,
    broadcast from a per-dataset stamp rather than stored per-patch (N copies
    of an identical 31x31 array would be pure memory waste).
    """

    def __init__(self, sources, augment: bool = False):
        """sources: list of (observed, ideal, psf_channels, tag)."""
        self.observed, self.ideal, self.psfs, self.tags = [], [], [], []
        self.index = []                      # (source_i, row_i)
        for si, (observed, ideal, psf_ch, tag) in enumerate(sources):
            if observed.shape != ideal.shape:
                sys.exit(f"[data] {tag}: observed {observed.shape} and ideal "
                         f"{ideal.shape} must have identical shapes")
            self.observed.append(np.ascontiguousarray(observed, dtype=np.float32))
            self.ideal.append(np.ascontiguousarray(ideal, dtype=np.float32))
            self.psfs.append(np.ascontiguousarray(psf_ch, dtype=np.float32))
            self.tags.append(tag)
            self.index.extend((si, ri) for ri in range(len(observed)))
        self.augment = augment

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        si, ri = self.index[idx]
        observed = self.observed[si][ri]      # (1, H, W)
        ideal = self.ideal[si][ri]
        psf = self.psfs[si]                   # (C, P, P)
        if self.augment:
            # torch's RNG is seeded per DataLoader worker, unlike numpy's.
            k = int(torch.randint(0, 8, (1,)).item())
            # The SAME element goes on all three: rotating the image pair
            # without rotating the beam would teach a false correspondence.
            observed, ideal, psf = d4(observed, k), d4(ideal, k), d4(psf, k)
        return (
            torch.from_numpy(np.ascontiguousarray(ideal)),
            torch.from_numpy(np.ascontiguousarray(observed)),
            torch.from_numpy(np.ascontiguousarray(psf)),
        )


def parse_data_spec(spec: str):
    """'dir' or 'dir:psf_path' -> (Path(dir), Path(psf) or None).

    Split on the LAST colon so Windows-style or otherwise colon-bearing
    directory names do not break, and only treat it as a separator if the
    right-hand side actually looks like a file that exists.
    """
    if ":" in spec:
        head, _, tail = spec.rpartition(":")
        if head and Path(tail).exists():
            return Path(head), Path(tail)
    return Path(spec), None


def load_sources(specs, split: str, psf_size: int, psf_repr: str):
    """Load every dataset dir for one split. Returns (sources, norms)."""
    sources, norms = [], {}
    for spec in specs:
        data_dir, psf_path = parse_data_spec(spec)
        if not data_dir.is_dir():
            sys.exit(f"[data] {data_dir} is not a directory")
        psf_path = psf_path or find_psf(data_dir)
        psf_raw = load_psf_array(psf_path)
        psf = canonicalize_psf(psf_raw, psf_size)
        psf_ch = psf_to_channels(psf, psf_repr)

        observed = np.load(data_dir / f"{split}_observed.npy")
        ideal = np.load(data_dir / f"{split}_ideal.npy")
        tag = data_dir.name
        print(f"[data] {split}/{tag}: {len(observed)} pairs "
              f"{tuple(observed.shape[1:])}  "
              f"observed[{observed.min():+.4f}, {observed.max():+.4f}]  "
              f"ideal[{ideal.min():+.4f}, {ideal.max():+.4f}]  "
              f"psf={psf_path.name} {psf_raw.shape}->{psf.shape} "
              f"peak={psf.max():.4g}")
        sources.append((observed, ideal, psf_ch, tag))

        norm_path = data_dir / "norm.json"
        norms[tag] = (json.loads(norm_path.read_text())
                      if norm_path.exists() else None)
        if norms[tag] is None:
            print(f"[data] WARNING: no norm.json in {data_dir}; checkpoints "
                  f"will not carry the normalized->physical map for {tag} and "
                  f"downstream photometry cannot be inverted to flux.")
    return sources, norms


def warn_on_mixed_norms(norms: dict):
    """A model cannot see which normalization a patch came from.

    Two dirs whose norm.json differ put the SAME pixel value at two different
    physical fluxes.  The network averages them, and every flux the model
    produces is then wrong by an amount no downstream inversion can recover,
    because the checkpoint has two candidate maps and no way to choose.
    """
    present = {k: v for k, v in norms.items() if v is not None}
    if len(present) < 2:
        return
    keys = list(present)
    ref = json.dumps(present[keys[0]], sort_keys=True)
    odd = [k for k in keys[1:]
           if json.dumps(present[k], sort_keys=True) != ref]
    if odd:
        print("[data] " + "!" * 68)
        print(f"[data] WARNING: norm.json differs across dataset dirs "
              f"({keys[0]} vs {', '.join(odd)}).")
        print("[data] The model sees pixel values only -- it cannot tell the "
              "domains apart, so it will average two different "
              "normalized->flux maps and every reconstructed flux will be "
              "biased. Regenerate the datasets with a SHARED normalization "
              "before trusting photometry from this checkpoint.")
        print("[data] " + "!" * 68)


# ----------------------------------------------------------------------------
# Cosine noise schedule (Nichol & Dhariwal 2021, as used in the DPS paper)
# ----------------------------------------------------------------------------

def cosine_alpha_bar(T: int, s: float = 0.008,
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """alpha_bar[t], t = 0..T-1. Always COMPUTED in float64.

    `dtype` sets only the return dtype. It defaults to float32 so training is
    unchanged from the previous trainer; solvers pass float64 so the schedule
    they derive sigma(t) from is not pre-rounded to ~1e-7 relative.
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    betas = betas.clamp(1e-8, 0.999)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0).to(dtype)


# ----------------------------------------------------------------------------
# Conditioning modules
# ----------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # A non-persistent buffer tracks the module dtype (so model.to(float64)
        # reaches this parameter-free module) without entering state_dict, so
        # checkpoints are unaffected.
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


class PSFEncoder(nn.Module):
    """PSF stamp(s) -> (pooled global embedding, token sequence).

    The pooled embedding is added to the timestep embedding, so every FiLM
    block is modulated by the beam as well as by t.  The token sequence is
    what cross-attention reads.

    SPATIALLY VARYING PSF (the planned upgrade): forward() accepts a stamp
    batch of shape (B, K, C, P, P).  Today K == 1.  To handle a field-varying
    beam, pass K stamps sampled across the field and add a field-position
    embedding to each stamp's tokens before they are concatenated -- the
    cross-attention below then lets every output pixel weight the beams by
    relevance, which is what makes patch-wise deconvolution unnecessary.  The
    hook is `pos_emb` in forward(); nothing else in the model changes.
    """

    def __init__(self, in_ch: int, embed_dim: int, width: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width * 2, embed_dim, 3, stride=2, padding=1), nn.SiLU(),
        )
        # Fixed token count regardless of --psf-size, so the token sequence
        # length (and hence the attention cost) does not depend on the stamp.
        self.pool = nn.AdaptiveAvgPool2d(PSF_TOKEN_GRID)
        self.token_proj = nn.Linear(embed_dim, embed_dim)
        self.global_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, psf, pos_emb=None):
        """psf: (B, C, P, P) or (B, K, C, P, P) -> (global (B,D), tokens (B,N,D))."""
        if psf.dim() == 4:
            psf = psf[:, None]                       # K = 1
        b, k, c, p, _ = psf.shape
        h = self.stem(psf.reshape(b * k, c, p, p))
        h = self.pool(h)                             # (B*K, D, G, G)
        d = h.shape[1]
        # (B*K, D, G, G) -> (B, K*G*G, D): each stamp contributes G**2 tokens,
        # kept adjacent so a per-stamp position embedding can be broadcast
        # over them below.
        tokens = h.flatten(2).transpose(1, 2).reshape(b, -1, d)
        tokens = self.token_proj(tokens)
        if pos_emb is not None:
            # (B, K, D) -> one field-position embedding per stamp, broadcast
            # over that stamp's PSF_TOKEN_GRID**2 tokens.
            tokens = tokens + pos_emb.repeat_interleave(
                PSF_TOKEN_GRID * PSF_TOKEN_GRID, dim=1)
        glob = self.global_proj(tokens.mean(dim=1))
        # The global embedding is prepended as a token too, giving attention a
        # "whole beam" summary alongside the spatial detail tokens.
        return glob, torch.cat([glob[:, None, :], tokens], dim=1)


class CrossAttention(nn.Module):
    """Spatial features attend over PSF tokens.

    Zero-init on the output projection makes the whole module start as a
    no-op, so turning --psf-cross-attn on does not perturb the early training
    dynamics of the rest of the network.
    """

    def __init__(self, channels: int, ctx_dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.scale = (channels // heads) ** -0.5
        self.q = nn.Conv2d(channels, channels, 1)
        self.kv = nn.Linear(ctx_dim, 2 * channels)
        self.out = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, ctx):
        b, c, h, w = x.shape
        n = ctx.shape[1]
        q = self.q(x).reshape(b, self.heads, c // self.heads, h * w)
        q = q.transpose(-2, -1)                       # (B, H, HW, Ch)
        k, v = self.kv(ctx).chunk(2, dim=-1)
        k = k.reshape(b, n, self.heads, c // self.heads).transpose(1, 2)
        v = v.reshape(b, n, self.heads, c // self.heads).transpose(1, 2)
        att = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        o = (att @ v).transpose(-2, -1).reshape(b, c, h, w)
        return x + self.out(o)


class FiLMConvBlock(nn.Module):
    """Conv -> [norm] -> FiLM(gamma, beta from t+psf embedding) -> SiLU -> +x.

    norm defaults to "none" (nn.Identity).  GroupNorm is retained behind the
    flag ONLY so the ablation can still be run: it divides out absolute scale
    at the first block and makes behaviour depend on H,W, both of which break
    photometry.  Identity is used rather than deleting the attribute so the
    same forward hooks and the same diagnostics work on both variants.
    """

    def __init__(self, channels: int, dilation: int, cond_dim: int,
                 norm: str = "none"):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3,
                              padding=dilation, dilation=dilation)
        self.norm = (nn.GroupNorm(8, channels) if norm == "group"
                     else nn.Identity())
        self.film = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)  # start as identity modulation

    def forward(self, x, cond):
        h = self.norm(self.conv(x))
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return F.silu(h) + x  # residual


# ----------------------------------------------------------------------------
# Backbones
# ----------------------------------------------------------------------------

DEFAULT_DILATIONS = (1, 2, 3, 4, 6, 8, 6, 4, 3, 2, 1)


def receptive_field(dilations) -> int:
    """3x3 head + dilated 3x3 blocks + 3x3 tail."""
    return 1 + 2 + 2 * sum(dilations) + 2


class PsfConditionalFlatCNN(nn.Module):
    """DnCNN-style flat denoiser, no downsampling. Predicts the noise eps.

    Input channels: [x_t (noisy sharp), y (blurry conditioning), psf...].
    Neither the blurry channel nor the PSF channels are ever noised -- they
    are fixed conditioning at every timestep, which is what injects the
    (image, beam) pairing statistics.

    With --norm none the block stack itself is fully convolutional and free of
    any operation that couples a pixel to the patch boundary.  The one part
    that is NOT size-independent is the psf_map channel under
    psf_map="center"; run whole frames with psf_map="tile" (see
    render_psf_map) or the beam channel goes out of reach of the edges.
    """

    def __init__(self, channels: int = 64, t_dim: int = 128,
                 norm: str = "none", psf_ch: int = 1, psf_size: int = 31,
                 psf_embed_dim: int = 128, psf_cross_attn: bool = False,
                 psf_map: str = "center",
                 dilations=DEFAULT_DILATIONS, xattn_every: int = 3):
        super().__init__()
        self.norm_kind = norm
        self.dilations = tuple(dilations)
        self.psf_cross_attn = psf_cross_attn
        self.psf_map = psf_map
        self.t_embed = nn.Sequential(
            SinusoidalTimeEmbedding(t_dim),
            nn.Linear(t_dim, t_dim), nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self.psf_encoder = PSFEncoder(psf_ch, psf_embed_dim)
        self.psf_to_cond = nn.Linear(psf_embed_dim, t_dim)
        in_ch = 2 + psf_ch
        self.head = nn.Conv2d(in_ch, channels, 3, padding=1)
        self.blocks = nn.ModuleList(
            FiLMConvBlock(channels, d, t_dim, norm=norm) for d in self.dilations
        )
        # Attention is expensive at full resolution, so it is inserted on a
        # stride rather than after every block. None entries keep the module
        # list index-aligned with self.blocks.
        self.attns = nn.ModuleList(
            CrossAttention(channels, psf_embed_dim)
            if (psf_cross_attn and i % xattn_every == xattn_every - 1)
            else None
            for i in range(len(self.dilations))
        )
        self.tail = nn.Conv2d(channels, 1, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)  # predict ~0 noise at init

    def forward(self, x_t, y_cond, psf, t):
        psf_glob, psf_tokens = self.psf_encoder(psf)
        cond = self.t_embed(t) + self.psf_to_cond(psf_glob)
        # The stamp is smaller than the patch, so it is rendered into a
        # full-size channel that reads the beam in the same pixel units as
        # the image.
        h, w = x_t.shape[-2:]
        psf_map = render_psf_map(psf, h, w, self.psf_map)
        z = self.head(torch.cat([x_t, y_cond, psf_map], dim=1))
        for block, attn in zip(self.blocks, self.attns):
            z = block(z, cond)
            if attn is not None:
                z = attn(z, psf_tokens)
        return self.tail(z)


class ResBlock(nn.Module):
    """U-Net residual block. Zero-init second conv => starts as identity."""

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


class PsfConditionalUNet(nn.Module):
    """Encoder/decoder denoiser for the PSF-conditional problem.

    Rationale for offering it: conditioning on the blurry image means the
    target at a pixel depends on the observation across the whole PSF
    footprint, wings included, and a downsampling backbone reaches that extent
    far more cheaply than dilations do.  The counter-argument that motivated
    the original flat CNN still stands -- downsampling is lossy for
    point-source statistics, and strided convs are not shift-equivariant --
    which is why this is a flag, not a replacement.  Measure both.

    Nearest-neighbour upsampling (not transposed conv) avoids checkerboard
    artifacts, which on a star field would be indistinguishable from faint
    sources.
    """

    def __init__(self, base: int = 64, ch_mult=(1, 2, 4), t_dim: int = 128,
                 norm: str = "none", psf_ch: int = 1, psf_size: int = 31,
                 psf_embed_dim: int = 128, psf_cross_attn: bool = False,
                 psf_map: str = "center", blocks_per_level: int = 2):
        super().__init__()
        self.norm_kind = norm
        self.ch_mult = tuple(ch_mult)
        self.blocks_per_level = blocks_per_level
        self.psf_cross_attn = psf_cross_attn
        self.psf_map = psf_map
        self.t_embed = nn.Sequential(
            SinusoidalTimeEmbedding(t_dim),
            nn.Linear(t_dim, t_dim), nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self.psf_encoder = PSFEncoder(psf_ch, psf_embed_dim)
        self.psf_to_cond = nn.Linear(psf_embed_dim, t_dim)

        in_ch = 2 + psf_ch
        chans = [base * m for m in self.ch_mult]
        self.head = nn.Conv2d(in_ch, chans[0], 3, padding=1)

        self.down = nn.ModuleList()
        self.down_attn = nn.ModuleList()
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
            # Attend only at reduced resolution: at full resolution the query
            # count (H*W) makes this the dominant cost for no extra reach.
            self.down_attn.append(
                CrossAttention(cur, psf_embed_dim)
                if (psf_cross_attn and li > 0) else None)
            if li < len(chans) - 1:
                self.downsample.append(nn.Conv2d(cur, cur, 3, stride=2,
                                                 padding=1))
                skip_chans.append(cur)
            else:
                self.downsample.append(None)

        self.mid1 = ResBlock(cur, cur, t_dim, norm=norm)
        self.mid_attn = (CrossAttention(cur, psf_embed_dim)
                         if psf_cross_attn else None)
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

    def forward(self, x_t, y_cond, psf, t):
        psf_glob, psf_tokens = self.psf_encoder(psf)
        cond = self.t_embed(t) + self.psf_to_cond(psf_glob)

        h, w = x_t.shape[-2:]
        psf_map = render_psf_map(psf, h, w, self.psf_map)

        z = self.head(torch.cat([x_t, y_cond, psf_map], dim=1))
        skips = [z]
        for level, attn, ds in zip(self.down, self.down_attn, self.downsample):
            for block in level:
                z = block(z, cond)
                skips.append(z)
            if attn is not None:
                z = attn(z, psf_tokens)
            if ds is not None:
                z = ds(z)
                skips.append(z)

        z = self.mid1(z, cond)
        if self.mid_attn is not None:
            z = self.mid_attn(z, psf_tokens)
        z = self.mid2(z, cond)

        for level, us in zip(self.up, self.upsample):
            for block in level:
                z = block(torch.cat([z, skips.pop()], dim=1), cond)
            if us is not None:
                z = us(F.interpolate(z, scale_factor=2, mode="nearest"))
        return self.tail(z)


def build_model(arch: str, **kw) -> nn.Module:
    if arch == "flat":
        kw.pop("base", None); kw.pop("ch_mult", None)
        kw.pop("blocks_per_level", None)
        return PsfConditionalFlatCNN(**kw)
    if arch == "unet":
        kw.pop("channels", None); kw.pop("dilations", None)
        kw.pop("xattn_every", None)
        return PsfConditionalUNet(**kw)
    sys.exit(f"[model] unknown --arch {arch!r}")


def model_from_checkpoint(ckpt: dict, use_ema: bool = True) -> nn.Module:
    """Rebuild the exact architecture a checkpoint was trained with.

    Downstream solvers should call this rather than hardcoding constructor
    arguments -- every architectural choice that affects the weights layout is
    recorded in ckpt["arch"], and guessing one wrong gives either a load error
    or, worse, a silently partly-random network.
    """
    a = dict(ckpt["arch"])
    arch = a.pop("kind")
    a.pop("in_channels", None)
    model = build_model(arch, **a)
    state = ckpt["ema_state"] if (use_ema and "ema_state" in ckpt) \
        else ckpt["model_state"]
    # ema_pytorch wraps the model, so its keys carry an "ema_model." prefix.
    state = {k.replace("ema_model.", "", 1): v for k, v in state.items()
             if not k.startswith(("initted", "step"))}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint/model mismatch: missing={missing} "
                           f"unexpected={unexpected}")
    return model


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

def diffusion_loss(model, sharp, blurry, psf, alpha_bar, device,
                   p_uncond=0.0, identity_frac=0.0, identity_t_max=100,
                   low_t_frac=0.0, low_t_max=100):
    """eps-prediction DDPM loss, conditioned on (blurry, psf).

    p_uncond > 0 enables CLASSIFIER-FREE GUIDANCE training: that fraction of
    the batch has BOTH conditioning inputs replaced by the NULL token (all
    zeros), so the same weights learn
        eps_theta(x_t, y, psf, t)   the conditional/posterior score, and
        eps_theta(x_t, 0, 0, t)     a genuine UNCONDITIONAL prior score.
    Both are dropped together: dropping y while keeping the PSF would leave a
    "prior" that still knows the beam, which is not the unconditional score
    RED/Graikos regularizers assume.

    All-zeros is a safe null token for both: real blurry patches are never
    identically zero, and a unit-sum PSF cannot be.
    """
    bsz = sharp.shape[0]
    T = alpha_bar.shape[0]
    t = torch.randint(0, T, (bsz,), device=device)
    if low_t_frac > 0:
        # LOW-t OVERSAMPLING: the RED/PnP prox only ever calls the denoiser at
        # t <= ~75, but uniform t gives that regime only low_t_max/T (~10%) of
        # the gradient signal, and low-t point-source recovery is exactly
        # where the observed smoothing lives. These keep their noise -- they
        # are ordinary denoising problems, just concentrated where the model
        # is actually used.
        low = (torch.rand(bsz, device=device) < low_t_frac)
        t_low = torch.randint(0, max(int(low_t_max), 1), (bsz,), device=device)
        t = torch.where(low, t_low, t)
    ab = alpha_bar[t][:, None, None, None]
    eps = torch.randn_like(sharp)
    if identity_frac > 0:
        # IDENTITY / fixed-point term: zero noise, so x_t = sqrt(abar)*sharp
        # and the target eps is exactly 0. Teaches D(x, y, psf) ~ x on a clean
        # sharp field. Without it the RED gradient x - D(x) does not vanish on
        # the truth and pulls away from it even when x is already correct
        # (measured on the old conditional checkpoint: 8-32% of the flux
        # removed from a true sharp patch at t=5..50). No lambda fixes that.
        # Their t is resampled into [0, identity_t_max) because a zero-noise
        # draw at t=900 is just 0.03*sharp -- a near-black frame that teaches
        # nothing and never occurs in the prox.
        idm = (torch.rand(bsz, device=device) < identity_frac)
        eps = torch.where(idm[:, None, None, None], torch.zeros_like(eps), eps)
        t_low = torch.randint(0, max(int(identity_t_max), 1), (bsz,),
                              device=device)
        t = torch.where(idm, t_low, t)
        ab = alpha_bar[t][:, None, None, None]
    x_t = ab.sqrt() * sharp + (1 - ab).sqrt() * eps
    if p_uncond > 0:
        drop = (torch.rand(bsz, device=device) < p_uncond)
        blurry = torch.where(drop[:, None, None, None],
                             torch.zeros_like(blurry), blurry)
        psf = torch.where(drop[:, None, None, None], torch.zeros_like(psf), psf)
    eps_pred = model(x_t, blurry, psf, t)
    return F.mse_loss(eps_pred, eps)


@torch.no_grad()
def validate(model, loader, alpha_bar, device):
    model.eval()
    total, n = 0.0, 0
    for sharp, blurry, psf in loader:
        sharp, blurry, psf = (sharp.to(device), blurry.to(device),
                              psf.to(device))
        loss = diffusion_loss(model, sharp, blurry, psf, alpha_bar, device)
        total += loss.item() * sharp.shape[0]
        n += sharp.shape[0]
    model.train()
    return total / max(n, 1)


def save_checkpoint(path, model, ema, ema_is_lib, optimizer, epoch,
                    dataset_norms, arch_cfg, args):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "ema_state": (ema.ema_model.state_dict() if ema_is_lib
                      else ema.state_dict()),
        "optimizer_state": optimizer.state_dict(),
        # NO trainer-side transform exists ("transform" is deliberately absent
        # so stale consumers fail loudly instead of applying the retired asinh
        # seam). The model operates in the dataset's stored domain;
        # `dataset_norms` maps dataset tag -> that dir's norm.json, the ONLY
        # map between this domain and physical flux. Inverting it is the
        # consumer's job. More than one entry with differing content means the
        # domains were mixed -- see warn_on_mixed_norms.
        "dataset_norms": dataset_norms,
        "data_dirs": [str(d) for d in args.data_dir],
        # Everything needed to rebuild the network. model_from_checkpoint()
        # consumes this dict directly, so any new constructor argument must be
        # added here too or old checkpoints will load into a wrong model.
        "arch": arch_cfg,
        "diffusion": {"timesteps": args.timesteps, "schedule": "cosine"},
        # p_uncond > 0 means model(x_t, 0, 0, t) is a valid UNCONDITIONAL
        # prior score (classifier-free guidance); consumers should check this
        # before using the null-token path.
        "p_uncond": args.p_uncond,
        "identity_frac": args.identity_frac,
        "identity_t_max": args.identity_t_max,
        "low_t_frac": args.low_t_frac,
        "low_t_max": args.low_t_max,
        "args": {k: ([str(x) for x in v] if isinstance(v, list)
                     else str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
    }
    torch.save(ckpt, path)


def main():
    ap = argparse.ArgumentParser(
        description="Train a PSF-conditional DDPM on observed/ideal .npy "
                    "patch pairs from one or more dataset dirs, each with "
                    "its own PSF.")
    ap.add_argument("--data-dir", type=str, nargs="+", required=True,
                    help="one or more dataset directories, each holding "
                         "{train,val}_{observed,ideal}.npy, norm.json and a "
                         "PSF (psf.fits / psf.npy / a unique psf*.fits). Use "
                         "DIR:PSF_PATH to name the PSF explicitly. Mixing "
                         "dirs is the point: PSF variety in training is what "
                         "removes the retrain-per-PSF requirement.")
    ap.add_argument("--checkpoint-dir", type=Path,
                    default=Path("checkpoints_psf_cond"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--timesteps", type=int, default=1000)

    # --- architecture ---------------------------------------------------
    ap.add_argument("--arch", choices=["flat", "unet"], default="flat",
                    help="'flat' is the dilated fully-convolutional stack "
                         "(the control the previous results were measured "
                         "on): shift-equivariant, no downsampling, applies "
                         "unchanged to a whole frame. 'unet' downsamples, so "
                         "its bottleneck sees the entire patch -- cheaper "
                         "long-range reach for the PSF-wing dependence the "
                         "blurry conditioning introduces, at the cost of "
                         "shift-equivariance and some point-source detail. "
                         "Train both and compare on photometry, not on val "
                         "loss.")
    ap.add_argument("--channels", type=int, default=64,
                    help="flat arch: width of every block.")
    ap.add_argument("--dilations", type=str,
                    default=",".join(str(d) for d in DEFAULT_DILATIONS),
                    help="flat arch: comma-separated dilation pyramid. The "
                         "default widens the old (1,2,3,4,4,3,2,1) 41-px "
                         "receptive field to 85 px, because the blurry "
                         "conditioning channel makes each output pixel depend "
                         "on the observation across the whole PSF footprint.")
    ap.add_argument("--base-channels", type=int, default=64,
                    help="unet arch: channels at the finest level.")
    ap.add_argument("--ch-mult", type=str, default="1,2,4",
                    help="unet arch: per-level channel multipliers; length "
                         "sets the depth (3 levels = /4 bottleneck).")
    ap.add_argument("--blocks-per-level", type=int, default=2,
                    help="unet arch: residual blocks per resolution level.")
    ap.add_argument("--norm", choices=["none", "group"], default="none",
                    help="normalization inside the blocks. Default 'none'. "
                         "'group' restores GroupNorm(8, C) and exists ONLY "
                         "for ablation: it divides out absolute scale at the "
                         "FIRST block (measured 3.74x in, 1.005x out for a 4x "
                         "input) and makes behaviour depend on H,W (3.7%% "
                         "difference between a patch alone and the same patch "
                         "inside a 128px mosaic). Both break photometry. "
                         "Verify with diag_scale_invariance.py.")

    # --- PSF conditioning ------------------------------------------------
    ap.add_argument("--psf-size", type=int, default=31,
                    help="every PSF is centre-cropped/zero-padded to this odd "
                         "size and renormalized to unit sum. Must be odd: the "
                         "D4 augmentation flips the stamp, and only an odd "
                         "grid has its centre as a fixed point.")
    ap.add_argument("--psf-repr", choices=["log", "linear", "both"],
                    default="log",
                    help="how the stamp becomes input channels. Raw PSFs span "
                         "peak ~5e-2 to wings ~1e-5, so a linear channel is "
                         "numerically just a delta and every beam looks alike; "
                         "'log' is peak-normalized log10 over a 1e-6 floor, "
                         "which is what makes the wings -- the part that "
                         "actually distinguishes PSFs -- visible. 'both' adds "
                         "the linear channel back for the core.")
    ap.add_argument("--psf-map", choices=["center", "tile"], default="center",
                    help="how the stamp is rendered into the input channel. "
                         "'center' is the faithful picture of the beam and is "
                         "right whenever inference runs at the training patch "
                         "size. 'tile' repeats the stamp across the canvas so "
                         "every pixel sees a full copy at ANY input size -- "
                         "use it if you intend to run the checkpoint on whole "
                         "frames, because a centred stamp on a 128px frame is "
                         "out of reach of the corner pixels (measured: 50%% "
                         "relative difference between a quadrant run alone "
                         "and the same quadrant inside the mosaic). The FiLM "
                         "and cross-attention PSF paths are size-independent "
                         "either way.")
    ap.add_argument("--psf-embed-dim", type=int, default=128,
                    help="width of the PSF encoder's tokens and of its pooled "
                         "embedding, which is added to the timestep embedding "
                         "so every FiLM block is beam-modulated.")
    ap.add_argument("--psf-cross-attn", action="store_true",
                    help="let spatial features cross-attend over the PSF "
                         "tokens. Zero-init'd, so it starts as a no-op. This "
                         "is the path to SPATIALLY VARYING PSFs: the encoder "
                         "already accepts K stamps per sample, so a "
                         "field-varying beam becomes K tokens groups with "
                         "position embeddings and no other model change.")
    ap.add_argument("--xattn-every", type=int, default=3,
                    help="flat arch: insert cross-attention after every Nth "
                         "block. Full-resolution attention is the dominant "
                         "cost, so this is a stride rather than every block.")

    # --- loss shaping (carried over unchanged) ---------------------------
    ap.add_argument("--p-uncond", type=float, default=0.15,
                    help="classifier-free-guidance dropout: fraction of "
                         "samples whose blurry AND psf conditioning are "
                         "replaced by the null token (zeros). >0 makes the "
                         "SAME weights usable as an unconditional prior via "
                         "model(x_t, 0, 0, t) -- required to use this model "
                         "in a RED/Graikos regularizer alongside an explicit "
                         "||y-Ax||^2 term without double-counting the "
                         "likelihood. 0 = purely conditional.")
    ap.add_argument("--identity-frac", type=float, default=0.15,
                    help="fraction of draws with ZERO noise and target eps=0, "
                         "teaching D(x,y,psf) ~ x on clean sharp fields. "
                         "Required for the RED gradient x - D(x) to vanish on "
                         "the truth.")
    ap.add_argument("--identity-t-max", type=int, default=100,
                    help="identity draws are placed at t ~ U[0, this). The "
                         "PnP/RED prox runs at t<=~75, so that is where the "
                         "fixed point has to hold.")
    ap.add_argument("--low-t-frac", type=float, default=0.25,
                    help="fraction of NOISY draws whose t is resampled into "
                         "[0, --low-t-max), where the prox actually runs.")
    ap.add_argument("--low-t-max", type=int, default=100,
                    help="upper bound (exclusive) for the oversampled low-t "
                         "range; match to the largest t the prox uses.")

    ap.add_argument("--no-augment", action="store_true",
                    help="disable the on-the-fly D4 (rot90/flip) augmentation "
                         "of the training split. The PSF is rotated with the "
                         "pair, so this also removes the beam-orientation "
                         "variety the augmentation supplies.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--overfit-one-batch", action="store_true",
                    help="Sanity check: memorize a single batch and track a "
                         "deterministic fixed-noise eval loss.")
    ap.add_argument("--sanity-steps", type=int, default=3000)
    ap.add_argument("--sanity-lr", type=float, default=3e-4,
                    help="hotter than the training default, appropriate for "
                         "pure memorization.")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.psf_size % 2 == 0:
        sys.exit(f"[psf] --psf-size must be odd (got {args.psf_size}); an "
                 f"even grid's centre is not a fixed point of the D4 flips "
                 f"used for augmentation, so the stamp would drift half a "
                 f"pixel relative to the image pair.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)
    print(f"[setup] device = {device}"
          + ("  (CPU: fine for --overfit-one-batch verification; use the GPU "
             "cluster for the full run)" if device.type == "cpu" else ""))

    # --- data -----------------------------------------------------------
    # Arrays are used exactly as stored: the generator owns the domain.
    train_sources, norms = load_sources(args.data_dir, "train",
                                        args.psf_size, args.psf_repr)
    val_sources, _ = load_sources(args.data_dir, "val",
                                  args.psf_size, args.psf_repr)
    warn_on_mixed_norms(norms)

    train_ds = PsfPairDataset(train_sources, augment=not args.no_augment)
    val_ds = PsfPairDataset(val_sources, augment=False)
    print(f"[data] total: {len(train_ds)} train / {len(val_ds)} val pairs "
          f"across {len(train_sources)} PSF(s)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"),
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # --- model / optim --------------------------------------------------
    psf_ch = n_psf_channels(args.psf_repr)
    dilations = tuple(int(d) for d in args.dilations.split(",") if d.strip())
    ch_mult = tuple(int(m) for m in args.ch_mult.split(",") if m.strip())
    arch_cfg = {
        "kind": args.arch,
        "t_dim": 128,
        "norm": args.norm,
        "psf_ch": psf_ch,
        "psf_size": args.psf_size,
        "psf_embed_dim": args.psf_embed_dim,
        "psf_cross_attn": args.psf_cross_attn,
        "psf_map": args.psf_map,
        "in_channels": 2 + psf_ch,          # informational; not a ctor arg
    }
    if args.arch == "flat":
        arch_cfg.update({"channels": args.channels,
                         "dilations": list(dilations),
                         "xattn_every": args.xattn_every})
    else:
        arch_cfg.update({"base": args.base_channels,
                         "ch_mult": list(ch_mult),
                         "blocks_per_level": args.blocks_per_level})

    ctor = {k: v for k, v in arch_cfg.items()
            if k not in ("kind", "in_channels")}
    model = build_model(args.arch, **ctor).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if args.arch == "flat":
        print(f"[model] PsfConditionalFlatCNN, {n_params/1e6:.2f}M params, "
              f"dilations {dilations} "
              f"({receptive_field(dilations)}-px receptive field)")
    else:
        print(f"[model] PsfConditionalUNet, {n_params/1e6:.2f}M params, "
              f"base {args.base_channels}, ch_mult {ch_mult}, "
              f"{args.blocks_per_level} blocks/level")
    print(f"[model] in_channels {2 + psf_ch} = [x_t, observed, "
          f"{psf_ch} psf ({args.psf_repr})], norm={args.norm}, "
          f"psf_embed_dim={args.psf_embed_dim}, "
          f"cross_attn={args.psf_cross_attn}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-5)
    ema, ema_is_lib = make_ema(model)
    alpha_bar = cosine_alpha_bar(args.timesteps).to(device)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(args.checkpoint_dir / "dataset_norms.json", "w") as f:
        json.dump(norms, f, indent=2)

    # --- overfit-one-batch sanity check ---------------------------------
    if args.overfit_one_batch:
        sharp, blurry, psf = next(iter(train_loader))
        sharp, blurry, psf = (sharp.to(device), blurry.to(device),
                              psf.to(device))

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
                losses.append(F.mse_loss(model(x_t, blurry, psf, t),
                                         eval_eps).item())
            model.train()
            return float(np.mean(losses)), losses

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
            loss = diffusion_loss(model, sharp, blurry, psf, alpha_bar, device)
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
        for sharp, blurry, psf in train_loader:
            sharp, blurry, psf = (sharp.to(device), blurry.to(device),
                                  psf.to(device))
            optimizer.zero_grad()
            loss = diffusion_loss(model, sharp, blurry, psf, alpha_bar, device,
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
                        ema_is_lib, optimizer, epoch, norms, arch_cfg, args)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(args.checkpoint_dir / "best.pt", model, ema,
                            ema_is_lib, optimizer, epoch, norms, arch_cfg,
                            args)
            print(f"          -> new best val loss, saved best.pt")

    print(f"[done] best val loss {best_val:.5f}. Use the 'ema_state' weights "
          f"from best.pt; rebuild the network with model_from_checkpoint() so "
          f"the arch matches. The model works in the dataset's stored domain; "
          f"'dataset_norms' in the checkpoint is the only map back to "
          f"physical flux. At inference the PSF must be prepared exactly as "
          f"here: canonicalize_psf(psf, {args.psf_size}) then "
          f"psf_to_channels(psf, '{args.psf_repr}').")


# Mandatory guard: this file contains a training loop and must never retrain
# on import (consumers import the model classes and model_from_checkpoint).
if __name__ == "__main__":
    main()
