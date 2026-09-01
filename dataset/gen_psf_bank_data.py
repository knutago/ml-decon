"""
gen_psf_bank_data.py

Build a PSF-CONDITIONAL training set: one truth frame rendered through a bank
of PSFs, every variant sharing ONE frozen normalization, each output directory
carrying the exact PSF that produced it.

Consumed by train_psf_conditional_diffusion.py:
    python gen_psf_bank_data.py --mode jitter ... --out-root data/f555_jitter
    python train_psf_conditional_diffusion.py \
        --data-dir data/f555_jitter/var_* \
        --val-data-dir data/f555_jitter/holdout_*

Why this exists instead of running gen_data.py N times
------------------------------------------------------
gen_data.py fits a SEPARATE asinh normalization per dataset, and AsinhNorm.fit
anchors hi_s on pixels.max().  Blur lowers the max, so the fit silently
renormalizes every field's peak back to z = 1.0 -- it divides out the single
most obvious PSF cue.  Measured on an m31bK50 crop blurred with five Gaussians
and fitted the way gen_data.py fits:

    FWHM   obs.max   median     beta     hi_s
     1.5   472.04   0.01401  0.00981   11.475
     2.0   265.91   0.01584  0.00945   10.938
     2.8   135.74   0.01762  0.00881   10.336
     3.5    86.93   0.01865  0.00856    9.918
     4.5    52.63   0.01984  0.00914    9.352

Worse, the same physical flux then lands at a different z in each dataset
(flux 1.0 -> z 0.520 at FWHM 1.5 but z 0.633 at FWHM 4.5, a 22% spread, with
the median drifting 42% across the bank).  A network asked to learn
p(ideal | observed, psf) across that is being shown a target whose units move
with the conditioning.  So: ONE normalization, fitted on the sharpest variant
(blur only lowers the max, so every other variant then lands below 1.0 rather
than clipping above it) and frozen for the whole bank.

Why the FULL FRAME is re-blurred, not the stored patches
--------------------------------------------------------
A 31x31 PSF has a 15 px half-width.  Convolving a 64x64 patch in isolation
gets the border wrong wherever flux should have leaked in from outside it --
that is (64^2 - 34^2)/64^2 = 72% of the patch area.  The truth frame is
2048^2, so blurring the frame and cutting afterwards costs nothing and is
correct by construction.

How big must the jitter be?
---------------------------
Big.  Measured on sim_f555w (gain 0.8284, sky 278, noise sigma 52.4, from the
fit this script's --fit-forward reproduces), as RMS(blurred_jittered -
blurred_nominal) in units of the noise:

    size x1.01   0.12 sigma    invisible
    size x1.02   0.21 sigma    invisible
    size x1.05   0.50 sigma    marginal
    size x1.10   0.95 sigma    marginal
    size x1.20   1.73 sigma    marginal
    size x1.40   2.90 sigma    learnable
    size x0.80   2.64 sigma    learnable
    size x0.70   4.51 sigma    learnable
    elong 1.05   0.38 sigma    invisible
    elong 1.20   1.40 sigma    marginal

A +-1-2% jitter changes the observation by a fifth of the noise: there is no
gradient signal that could teach the network to read the PSF channel, and it
will correctly learn to ignore it.  Hence --scale-range defaults to a factor
of ~1.4 either way, and note the asymmetry -- NARROWING is more visible than
broadening at equal factor, and elongation is a weaker cue than size, so shape
jitter needs a larger amplitude than it intuitively seems to.

Two modes
---------
--mode jitter   perturb a reference PSF (scale / elongation / angle).  The
                ideal is byte-identical across variants and only the beam
                changes, so it is a perfectly controlled ablation -- the right
                FIRST experiment to answer "is the PSF channel used at all?".
                It buys robustness to PSF MISESTIMATION. It does NOT buy
                transfer to a different instrument: f555w's beam vs m31f11's
                (1.15-1.47 px, MTF@Nyquist 0.073-0.234) is a different regime,
                not a jitter.

--mode bank     a wide parametric bank (Moffat/Gaussian, log-spaced FWHM,
                varying wing slope) plus any real measured PSFs passed with
                --extra-psf.  This is what delivers "a new PSF is a forward
                pass, not a retrain".  Include the PSFs you will deploy on so
                that deployment is interpolation, not extrapolation.

Sky and noise are randomized INDEPENDENTLY of the PSF (--sky-jitter,
--noise-jitter).  If they are constant per blur level the network can read the
blur off the noise-to-peak ratio and ignore the PSF channel; if they correlate
with blur it learns the correlation instead of the beam.  Decoupling them is
what forces the PSF channel to be the only reliable cue.
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import affine_transform
from scipy.signal import fftconvolve


def _find_ml_decon() -> Path:
    """Locate the shared ml-decon checkout.

    Hardcoding an absolute path makes this script unrunnable anywhere but the
    machine it was written on, which is the wrong property for something that
    has to run on the training cluster. Order: $ML_DECON, then the layout this
    repo sits in (../global/ml-decon next to mycode), then a sibling.
    """
    env = os.environ.get("ML_DECON")
    if env:
        # An explicit setting is strict: falling back to some other checkout
        # after a typo would silently swap the norm.json schema this script
        # single-sources, and the damage would only surface as bad photometry.
        c = Path(env).expanduser()
        if (c / "core" / "normalize.py").exists():
            return c
        sys.exit(f"[setup] ML_DECON={env} does not contain core/normalize.py. "
                 f"Point it at an ml-decon checkout, or unset it to use the "
                 f"default sibling layout.")
    here = Path(__file__).resolve().parent
    candidates = [here.parent / "global" / "ml-decon", here.parent / "ml-decon"]
    for c in candidates:
        if (c / "core" / "normalize.py").exists():
            return c
    sys.exit(
        "[setup] cannot find the ml-decon checkout (looked for "
        "core/normalize.py in: " + ", ".join(str(c) for c in candidates) +
        "). Clone git@github.com:knutago/ml-decon.git and either place it at "
        "../global/ml-decon relative to this script or set ML_DECON=/path/to/"
        "ml-decon.")


ML_DECON = _find_ml_decon()
sys.path.insert(0, str(ML_DECON))
# Single-sourced from the shared repo on purpose: norm.json's schema is a
# contract with every downstream solver, and a local reimplementation would
# drift from it silently.
from core.normalize import AsinhNorm, LinearNorm            # noqa: E402
from dataset.gen_data import (assign_splits, cut_patches,   # noqa: E402
                              grid_corners, load_fits)

PSF_SIZE = 31       # matches every psf_*.fits in the tree and the trainer default


# ----------------------------------------------------------------------------
# PSF construction
# ----------------------------------------------------------------------------

def _unit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 0, None)
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("PSF has non-positive sum")
    return p / s


def moffat_psf(fwhm, beta=3.5, ellip=1.0, theta=0.0, n=PSF_SIZE):
    """Moffat profile: I(r) = (1 + (r/alpha)^2)^-beta.

    The standard atmospheric-seeing profile. beta is the WING SLOPE and is the
    knob that varies the wings at fixed core width -- exactly what the log PSF
    channel exists to expose, and what distinguishes two PSFs of equal FWHM.
    beta -> inf recovers a Gaussian.
    """
    alpha = fwhm / (2.0 * math.sqrt(2.0 ** (1.0 / beta) - 1.0))
    y, x = np.mgrid[:n, :n] - (n - 1) / 2.0
    ct, st = math.cos(theta), math.sin(theta)
    xr, yr = ct * x + st * y, -st * x + ct * y
    r2 = (xr / ellip) ** 2 + (yr * ellip) ** 2
    return _unit((1.0 + r2 / alpha ** 2) ** (-beta))


def gaussian_psf(fwhm, ellip=1.0, theta=0.0, n=PSF_SIZE):
    sigma = fwhm / 2.3548200450309493
    y, x = np.mgrid[:n, :n] - (n - 1) / 2.0
    ct, st = math.cos(theta), math.sin(theta)
    xr, yr = ct * x + st * y, -st * x + ct * y
    r2 = (xr / ellip) ** 2 + (yr * ellip) ** 2
    return _unit(np.exp(-r2 / (2.0 * sigma ** 2)))


def affine_psf(psf, scale=1.0, ellip=1.0, theta=0.0):
    """Resample a measured PSF: isotropic scale, then an area-preserving stretch.

    Uses cubic interpolation about the stamp centre, so the peak stays at
    index n//2 and the D4 augmentation in the trainer remains valid. Negative
    interpolation undershoot is clipped before renormalizing -- a PSF with
    negative wings is not physical and breaks the log channel.
    """
    n = psf.shape[0]
    c = (n - 1) / 2.0
    ct, st = math.cos(theta), math.sin(theta)
    rot = np.array([[ct, -st], [st, ct]])
    m = rot @ np.diag([1.0 / ellip, ellip]) @ rot.T / scale
    offset = np.array([c, c]) - m @ np.array([c, c])
    out = affine_transform(psf, m, offset=offset, order=3,
                           mode="constant", cval=0.0)
    return _unit(out)


def second_moment_fwhm(psf):
    """FWHM from the second moment. NOTE this is wing-weighted and reads much
    larger than a core fit (8.43 vs 3.49 px on psf_m32sim_f555w); it is used
    here only as a consistent relative ordering, never as a quoted seeing."""
    n = psf.shape[0]
    y, x = np.mgrid[:n, :n] - n // 2
    return 2.3548200450309493 * math.sqrt(
        float((psf * (x * x + y * y)).sum() / psf.sum() / 2.0))


def build_bank(args, rng):
    """-> list of dicts with 'name', 'psf', and the parameters that made it."""
    bank = []
    if args.mode == "jitter":
        ref = _unit(load_psf(args.psf))
        if args.include_nominal:
            bank.append({"name": "nominal", "psf": ref, "family": "reference",
                         "scale": 1.0, "ellip": 1.0, "theta": 0.0})
        lo, hi = args.scale_range
        for i in range(args.n_variants):
            # Log-uniform in scale so narrowing and broadening are sampled
            # symmetrically; narrowing is the more visible half (x0.80 reads
            # 2.64 sigma against x1.20's 1.73) so a linear range would
            # over-weight the easy direction.
            scale = float(np.exp(rng.uniform(math.log(lo), math.log(hi))))
            ellip = float(rng.uniform(1.0, args.max_ellip))
            # D4 augmentation in the trainer supplies the other 8 orientations
            # for free, so theta only has to cover a quarter turn.
            theta = float(rng.uniform(0.0, math.pi / 4))
            bank.append({"name": f"var_{i:03d}", "family": "jitter",
                         "psf": affine_psf(ref, scale, ellip, theta),
                         "scale": scale, "ellip": ellip, "theta": theta})
    else:
        lo, hi = args.fwhm_range
        for i in range(args.n_variants):
            fwhm = float(np.exp(rng.uniform(math.log(lo), math.log(hi))))
            beta = float(rng.uniform(*args.moffat_beta))
            ellip = float(rng.uniform(1.0, args.max_ellip))
            theta = float(rng.uniform(0.0, math.pi / 4))
            gauss = rng.random() < args.gaussian_frac
            psf = (gaussian_psf(fwhm, ellip, theta) if gauss
                   else moffat_psf(fwhm, beta, ellip, theta))
            bank.append({"name": f"var_{i:03d}",
                         "family": "gaussian" if gauss else "moffat",
                         "psf": psf, "fwhm": fwhm,
                         "beta": None if gauss else beta,
                         "ellip": ellip, "theta": theta})
    for path in args.extra_psf or []:
        p = Path(path)
        bank.append({"name": f"real_{p.stem}", "family": "measured",
                     "psf": _unit(load_psf(p)), "source": str(p)})
    if not bank:
        sys.exit("[bank] empty PSF bank; raise --n-variants or pass --extra-psf")
    return bank


def load_psf(path):
    path = Path(path)
    data = (fits.getdata(str(path)) if path.suffix.lower() in (".fits", ".fit")
            else np.load(path))
    data = np.squeeze(np.asarray(data, dtype=np.float64))
    if data.ndim != 2:
        sys.exit(f"[psf] {path}: expected a 2-D stamp, got {data.shape}")
    if data.shape != (PSF_SIZE, PSF_SIZE):
        sys.exit(f"[psf] {path}: expected {PSF_SIZE}x{PSF_SIZE}, got "
                 f"{data.shape}; regrid it first so the whole bank shares a "
                 f"stamp geometry")
    return data


# ----------------------------------------------------------------------------
# Forward model
# ----------------------------------------------------------------------------

def fit_forward(truth, observed_path, psf, crop=None):
    """Recover (gain, sky, noise_sigma) from a real observed frame.

    Least squares on observed ~ gain*(psf*truth) + sky over every pixel, then
    the noise from the sigma-clipped std of the residual. On sim_f555w this
    returns gain 0.8284, sky 277.6, sigma 52.4, with the residual sitting
    BELOW the frame's own background scatter (99.4) -- i.e. the forward model
    explains the frame, which is what licenses re-blurring this truth at all.
    """
    obs = load_fits(observed_path).astype(np.float64)
    if crop:
        # The truth was already cropped; the fit frame has to be cut the same
        # way or the two are not the same pixels.
        y0, y1, x0, x1 = crop
        obs = obs[y0:y1, x0:x1]
    if obs.shape != truth.shape:
        sys.exit(f"[forward] observed {obs.shape} and truth {truth.shape} "
                 f"must share a pixel grid")
    pred = fftconvolve(truth, psf, mode="same")
    a = np.vstack([pred.ravel(), np.ones(pred.size)]).T
    gain, sky = np.linalg.lstsq(a, obs.ravel(), rcond=None)[0]
    _, _, sigma = sigma_clipped_stats(obs - (gain * pred + sky), sigma=3.0)
    return float(gain), float(sky), float(sigma)


def render(truth, psf, gain, sky, sigma, poisson_gain, rng):
    """observed = Poisson[gain*(psf*truth) + sky] + N(0, sigma)."""
    lam = np.clip(gain * fftconvolve(truth, psf, mode="same") + sky, 0, None)
    if poisson_gain > 0:
        lam = rng.poisson(lam * poisson_gain) / poisson_gain
    if sigma > 0:
        lam = lam + rng.normal(0.0, sigma, lam.shape)
    return lam.astype(np.float32)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Render one truth frame through a bank of PSFs into "
                    "PSF-conditional training directories sharing one frozen "
                    "normalization.")
    ap.add_argument("--truth-fits", type=Path, required=True,
                    help="the ideal/truth frame, e.g. "
                         "ml-decon/data/m32_sim/sim_f555w_truth.fits")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--mode", choices=["jitter", "bank"], default="jitter")
    ap.add_argument("--psf", type=Path,
                    help="reference PSF (required for --mode jitter, and used "
                         "to fit the forward model with --fit-forward)")
    ap.add_argument("--n-variants", type=int, default=24)
    ap.add_argument("--holdout", type=int, default=4,
                    help="this many variants are written as HOLDOUT dirs whose "
                         "PSF appears nowhere in training. Without them the "
                         "val split shares its PSFs with train and measures "
                         "nothing about the only claim that matters -- that a "
                         "NEW PSF needs no retrain. Point the trainer's "
                         "--val-data-dir at these.")

    # jitter mode
    ap.add_argument("--scale-range", type=float, nargs=2, default=(0.70, 1.40),
                    help="jitter: log-uniform isotropic scale factor. The "
                         "default spans a factor ~1.4 either way because "
                         "smaller is unlearnable: a +-2%% jitter moves the "
                         "observation by 0.21 of the noise sigma.")
    ap.add_argument("--include-nominal", action="store_true",
                    help="jitter: also emit the unmodified reference PSF.")

    # bank mode
    ap.add_argument("--fwhm-range", type=float, nargs=2, default=(1.2, 5.0),
                    help="bank: log-uniform core FWHM in pixels. The default "
                         "brackets the real 2-3.4 px PSFs and the 1.5 px beam "
                         "the reconstruction delivers.")
    ap.add_argument("--moffat-beta", type=float, nargs=2, default=(2.5, 5.0),
                    help="bank: Moffat wing slope range. This varies the WINGS "
                         "at fixed core width, which is what distinguishes two "
                         "PSFs of equal FWHM and what the log PSF channel is "
                         "built to show.")
    ap.add_argument("--gaussian-frac", type=float, default=0.25,
                    help="bank: fraction of variants drawn Gaussian rather "
                         "than Moffat.")

    ap.add_argument("--max-ellip", type=float, default=1.15,
                    help="axis-ratio bound for the area-preserving stretch. "
                         "Elongation is a WEAKER cue than size (1.20 reads "
                         "1.40 sigma against size 1.20's 1.73), so it needs a "
                         "larger amplitude than it looks like it should.")
    ap.add_argument("--extra-psf", type=Path, nargs="*",
                    help="real measured PSFs to include verbatim, so "
                         "deployment on them is interpolation rather than "
                         "extrapolation.")

    # forward model
    ap.add_argument("--fit-forward", type=Path,
                    help="a real observed frame to fit (gain, sky, sigma) "
                         "from, e.g. sim_f555w_conv.fits. Overrides the "
                         "explicit values below.")
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--sky", type=float, default=0.0)
    ap.add_argument("--noise-sigma", type=float, default=0.0)
    ap.add_argument("--poisson-gain", type=float, default=0.0,
                    help="electrons per count for the shot-noise term; 0 "
                         "disables it (Gaussian read noise only).")
    ap.add_argument("--sky-jitter", type=float, default=0.30,
                    help="per-variant fractional spread of the sky pedestal, "
                         "drawn INDEPENDENTLY of the PSF. If sky and noise "
                         "track the blur the network reads blur off them and "
                         "ignores the PSF channel; decoupling makes the PSF "
                         "channel the only reliable cue. It also supplies the "
                         "sky augmentation that fixed the 13%% sky leak.")
    ap.add_argument("--noise-jitter", type=float, default=0.30,
                    help="per-variant fractional spread of the noise sigma, "
                         "likewise independent of the PSF.")

    # patching (mirrors gen_data.py so the split semantics are identical)
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--split-block-size", type=int, default=512)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--asinh-softening", type=float, default=1.0)
    ap.add_argument("--crop", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.mode == "jitter" and args.psf is None:
        sys.exit("[args] --mode jitter needs --psf (the reference PSF)")
    if args.holdout >= args.n_variants + len(args.extra_psf or []):
        sys.exit("[args] --holdout must leave at least one training variant")

    rng = np.random.default_rng(args.seed)
    truth = load_fits(args.truth_fits).astype(np.float64)
    if args.crop:
        y0, y1, x0, x1 = args.crop
        truth = truth[y0:y1, x0:x1]
    print(f"[truth] {args.truth_fits} {truth.shape}  "
          f"exact zeros {np.mean(truth == 0):.2%}  max {truth.max():.4g}")

    bank = build_bank(args, rng)
    print(f"[bank] {len(bank)} PSFs ({args.mode} mode)")

    # --- forward model ---------------------------------------------------
    gain, sky, sigma = args.gain, args.sky, args.noise_sigma
    if args.fit_forward:
        ref = _unit(load_psf(args.psf)) if args.psf else bank[0]["psf"]
        gain, sky, sigma = fit_forward(truth, args.fit_forward, ref, args.crop)
        print(f"[forward] fitted on {args.fit_forward.name}: "
              f"observed = {gain:.4f}*(psf*truth) + {sky:.2f},  "
              f"noise sigma {sigma:.2f}")
    else:
        print(f"[forward] gain {gain:.4f}  sky {sky:.4g}  sigma {sigma:.4g}")

    # --- patch grid and split (identical for every variant) --------------
    h, w = truth.shape
    corners = grid_corners(h, w, args.patch_size, args.stride)
    splits, block_of = assign_splits(corners, args.patch_size,
                                     args.split_block_size, args.val_fraction,
                                     np.random.default_rng(args.seed))
    splits = np.array(splits)
    train_mask = splits == "train"
    n_val = int((~train_mask).sum())
    print(f"[patch] {len(corners)} patches  "
          f"{int(train_mask.sum())} train / {n_val} val")
    if n_val == 0:
        # assign_splits sends WHOLE BLOCKS to val, so a small frame divided by
        # a large --split-block-size has too few blocks for round(val_fraction
        # * n_blocks) to reach 1. Silently continuing would write holdout dirs
        # with no patches at all, and the held-out-PSF experiment -- the only
        # one that tests the model's reason for existing -- would score
        # nothing while looking like it ran.
        n_blocks = len({((y + args.patch_size // 2) // args.split_block_size,
                         (x + args.patch_size // 2) // args.split_block_size)
                        for y, x in corners})
        sys.exit(
            f"[patch] 0 val patches: the frame gives {n_blocks} split "
            f"block(s) of {args.split_block_size} px and "
            f"round({args.val_fraction} * {n_blocks}) = 0. Lower "
            f"--split-block-size, raise --val-fraction, or crop less. "
            f"(Blocks must stay big enough that train and val pixels never "
            f"share a patch.)")
    if n_val < 32:
        print(f"[patch] WARNING: only {n_val} val patches; the held-out-PSF "
              f"val loss will be noisy.")

    # --- ideal: cut and encode ONCE, shared byte-identically -------------
    ideal_raw = cut_patches(truth.astype(np.float32), corners, args.patch_size)
    ideal_norm = AsinhNorm.fit(ideal_raw[train_mask],
                               asinh_softening=args.asinh_softening)
    ideal = ideal_norm.forward(ideal_raw).astype(np.float32)
    print(f"[norm]    ideal: {ideal_norm.to_dict()}")

    # --- freeze the observed norm on the SHARPEST variant ----------------
    # Blur only lowers the peak, so anchoring on the sharpest PSF puts every
    # other variant BELOW z = 1 instead of clipping above it. PSF peak height
    # orders sharpness monotonically, so no extra convolutions are needed.
    anchor = max(range(len(bank)), key=lambda i: bank[i]["psf"].max())
    anchor_obs = render(truth, bank[anchor]["psf"], gain, sky, sigma,
                        args.poisson_gain, np.random.default_rng(args.seed))
    obs_norm = AsinhNorm.fit(
        cut_patches(anchor_obs, corners, args.patch_size)[train_mask],
        asinh_softening=args.asinh_softening)
    print(f"[norm] observed: {obs_norm.to_dict()}")
    print(f"[norm] frozen from the sharpest variant "
          f"'{bank[anchor]['name']}' and reused for all {len(bank)}")
    norm_json = json.dumps({"observed": obs_norm.to_dict(),
                            "ideal": ideal_norm.to_dict()}, indent=2)

    # --- render every variant -------------------------------------------
    args.out_root.mkdir(parents=True, exist_ok=True)
    order = list(rng.permutation(len(bank)))
    hold = set(order[:args.holdout])
    rows = []
    for i, entry in enumerate(bank):
        is_hold = i in hold
        name = entry["name"]
        vr = np.random.default_rng(args.seed * 100003 + i)
        # Sky and noise are drawn per variant, INDEPENDENTLY of the PSF.
        v_sky = sky * (1.0 + args.sky_jitter * vr.normal()) if sky else sky
        v_sig = abs(sigma * (1.0 + args.noise_jitter * vr.normal()))
        obs = render(truth, entry["psf"], gain, v_sky, v_sig,
                     args.poisson_gain, vr)
        observed = obs_norm.forward(
            cut_patches(obs, corners, args.patch_size)).astype(np.float32)

        # Two subtrees rather than a name prefix, so the trainer invocation is
        # an unambiguous glob and a holdout dir cannot be swept into training
        # by a careless pattern.
        d = args.out_root / ("holdout" if is_hold else "train") / name
        d.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val"):
            m = splits == split
            # A holdout variant contributes ONLY val patches: its PSF must not
            # reach the optimizer through any split.
            np.save(d / f"{split}_observed.npy",
                    observed[m] if not is_hold or split == "val"
                    else observed[:0])
            np.save(d / f"{split}_ideal.npy",
                    ideal[m] if not is_hold or split == "val" else ideal[:0])
        (d / "norm.json").write_text(norm_json)
        fits.PrimaryHDU(entry["psf"].astype(np.float32)).writeto(
            d / "psf.fits", overwrite=True)
        row = {k: v for k, v in entry.items() if k != "psf"}
        row.update({"dir": str(d.relative_to(args.out_root)),
                    "holdout": int(is_hold),
                    "psf_peak": float(entry["psf"].max()),
                    "sm_fwhm": second_moment_fwhm(entry["psf"]),
                    "sky": float(v_sky), "noise_sigma": float(v_sig),
                    "obs_z_max": float(observed.max())})
        rows.append(row)
        print(f"  [{i+1:3d}/{len(bank)}] {name:22s} peak {entry['psf'].max():.5f} "
              f"sm_fwhm {row['sm_fwhm']:5.2f}  sky {v_sky:8.1f} sig {v_sig:6.1f} "
              f"z_max {row['obs_z_max']:.3f}" + ("   HOLDOUT" if is_hold else ""))

    fields = sorted({k for r in rows for k in r})
    with open(args.out_root / "bank_manifest.csv", "w", newline="") as fh:
        wri = csv.DictWriter(fh, fieldnames=fields)
        wri.writeheader()
        wri.writerows(rows)

    zmax = max(r["obs_z_max"] for r in rows)
    print(f"\n[done] wrote {len(bank)} dirs to {args.out_root}")
    print(f"[check] max observed z across the bank = {zmax:.3f} "
          + ("(<=1, the frozen anchor brackets the bank)" if zmax <= 1.01 else
             "(>1: the anchor did NOT bracket the bank -- a variant is "
             "sharper than the anchor, check --extra-psf)"))
    print(f"[next] python train_psf_conditional_diffusion.py \\\n"
          f"         --data-dir {args.out_root}/train/* \\\n"
          f"         --val-data-dir {args.out_root}/holdout/*")


if __name__ == "__main__":
    main()
