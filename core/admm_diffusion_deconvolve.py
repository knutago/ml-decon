"""
admm_diffusion_deconvolve.py

PnP-ADMM deconvolution with a conditional diffusion denoising prior, following
Deutsch, "ADMM-Based Image Deconvolution with Conditioned Diffusion Prior"
(and Venkatakrishnan/Bouman/Wohlberg 2013 for the PnP-ADMM framework itself).

    minimize  0.5||Ax - b||^2 + lam * Psi(z)     s.t.  z - x = 0

ADMM splits this into three cheap steps per outer iteration:

    x^{k+1} = argmin_x 0.5||Ax - b||^2 + (rho/2)||x - (z^k - u^k)||^2
            = F^-1[ (conj(K) B + rho F(z^k - u^k)) / (|K|^2 + rho) ]   EXACT
    z^{k+1} = D_sigma( x^{k+1} + u^k )        <- diffusion prior plugged in here
    u^{k+1} = u^k + x^{k+1} - z^{k+1}         <- dual ascent

How this compares to the RED scheme in red_pnp_deconvolve.py
------------------------------------------------------------
Originally this script existed because RED was losing badly. That gap turned
out to be mostly RED's denoiser, not the RED framework: RED was calling a
single Tweedie step, which supplies almost no sharpening at any timestep.
red_pnp_deconvolve.py now runs a --denoise-steps chain (and needed its lambda
re-tuned 0.03 -> 0.3 as a result), and the two schemes are close. Scored
head-to-head by compare_solvers.py on 16 m31bK50 val patches:

                    conc  int_rmse  edge_rmse  floor%   flux   data_res
    RED (1-step)   0.621    0.6699     3.0258   12.85  0.944    0.1482
    RED (chain)    0.770    0.0753     0.1122    7.49  0.968    0.0478
    ADMM           0.717    0.0388     0.2012    0.06  1.021    0.0761
    truth          0.749    0.0000     0.0000    7.28  1.000    0.0589

ADMM still wins where its structure earns it: interior RMSE (0.0388 vs 0.0753,
driven by source photometry -- RMSE at source pixels is 0.085 vs 0.173) and it
does not overfit the noise (RED's data residual 0.0478 sits BELOW the 0.0589
noise floor, ADMM's 0.0761 above it). Its weaknesses are a +2% flux bias and a
floor fraction of 0.06% against the truth's 7.28%, i.e. it holds the background
off zero where the truth sits on it.

The structural reason ADMM's x-update matters is unchanged: gradient descent
on 0.5||Ax-b||^2 with a band-limited A amplifies exactly the frequencies the
PSF suppresses (with lam=0, 43% of pixels clamped and background scatter 540x
the truth). ADMM's x-update is a single exact Fourier solve with a rho ridge,
so the (|K|^2 + rho) denominator is bounded away from zero and those modes are
damped rather than amplified. What changed is that a strong enough prior can
also hold that instability down, which is what RED at lam=0.3 is doing.

Timestep selection follows the paper's heuristic: pick the t whose noise
variance is closest to the noise variance of the input to the denoiser. In the
DDPM parameterization x_t = sqrt(abar_t) x_0 + sqrt(1-abar_t) eps, dividing
through by sqrt(abar_t) gives an equivalent additive noise std of
    sigma(t) = sqrt((1 - abar_t) / abar_t)
so t is chosen by matching sigma(t) to the ADMM denoising level sqrt(lam/rho).

Domains
-------
x, z, u all live in PHYSICAL FLUX, because A is linear only there. The denoiser
wraps norm.json's forward/inverse around the model call, and clamps the x0
estimate to [0,1] in normalized units before inverting -- the asinh inverse is
explosive (normalized 1.2 -> 14x flux error, 2.0 -> inf in float32).

Usage:
python admm_diffusion_deconvolve.py \
    --ckpt /home/alex/noir_ml/global/ml-decon/checkpoints_cond_diffusion_npy/best.pt \
    --data-dir /home/alex/noir_ml/global/ml-decon/data/m31bK50 \
    --split val --indices 0:16 --psf-file psf_50_true.fits \
    --iters 200 --denoise-steps 4 --rho 3.0 --rho-scale 1.0 --lam 0.01 \
    --eta 0.0 --renoise --t-min 1 --t-max 75 \
    --n-show 0 --log-every 50 --out-dir admm_fft_best
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from red_pnp_deconvolve import (TorchNorm, load_checkpoint, load_kernel,
                                gaussian_kernel, make_otf, parse_indices, show,
                                row_norm, tweedie_chain)
from cond_sample_npy import find_peaks, concentration
from train_conditional_diffusion import cosine_alpha_bar


# ----------------------------------------------------------------------------
# The paper's timestep heuristic
# ----------------------------------------------------------------------------
def sigma_of_t(alpha_bar: torch.Tensor) -> torch.Tensor:
    """Equivalent additive-noise std of each timestep: sqrt((1-abar)/abar)."""
    return ((1.0 - alpha_bar) / alpha_bar).sqrt()


def fit_gain_sky(observed, truth, block=128, margin=16, verbose=True):
    """Fit observed ~ gain*truth + sky on block means, as gen_data.py does.

    This is the REAL preprocessing: ml-decon maps the observed into the truth's
    photometric units with (observed - sky)/gain before any asinh is fitted.
    When a matched truth exists this replaces --match-flux entirely, and unlike
    the background-statistics proxy it recovers the gain rather than assuming
    the noise level carries it.

    Block means (not per-pixel) because individual pixels are noise-dominated
    and would bias the slope toward zero.
    """
    h, w = observed.shape[-2:]
    o = observed.reshape(h, w)[margin:h - margin, margin:w - margin]
    t = truth.reshape(h, w)[margin:h - margin, margin:w - margin]
    # Shrink the block until there are enough samples to fit 2 parameters. On a
    # 256px region the 128px default yields ONE block, which fits the affine
    # exactly and reports a meaningless gain with residual 0.0000.
    MIN_BLOCKS = 36
    block = int(min(block, max(min(o.shape) // 6, 2)))
    while block > 2 and (o.shape[0] // block) * (o.shape[1] // block) < MIN_BLOCKS:
        block //= 2
    nb = (o.shape[0] // block) * (o.shape[1] // block)
    if nb < 8:
        raise SystemExit(f"--fit-gain-sky needs a bigger --region: only {nb} "
                         f"blocks available, which cannot constrain gain/sky")
    hh = (o.shape[0] // block) * block
    ww = (o.shape[1] // block) * block
    om = o[:hh, :ww].reshape(hh // block, block, ww // block, block).mean((1, 3))
    tm = t[:hh, :ww].reshape(hh // block, block, ww // block, block).mean((1, 3))
    A = np.stack([tm.ravel(), np.ones(tm.size)], axis=1)
    (gain, sky), *_ = np.linalg.lstsq(A, om.ravel(), rcond=None)
    if verbose:
        pred = gain * tm.ravel() + sky
        rr = (np.linalg.norm(om.ravel() - pred)
              / max(np.linalg.norm(om.ravel()), 1e-30))
        print(f"[gain] observed ~ {gain:.6g} * truth + {sky:.6g}   "
              f"(block-mean residual {rr:.4f}, {tm.size} blocks of {block}px)")
    return float(gain), float(sky)


def match_flux_distribution(d, obs_norm, softening=1.0, verbose=True):
    """Put a foreign frame into the checkpoint's flux domain, via the BACKGROUND.

    ml-decon's gen_data.py does NOT min-max. It fits observed ~ gain*ideal + sky
    against the paired ideal and stores (observed - sky)/gain, i.e. the observed
    in the IDEAL's photometric units, and only then fits the asinh whose
    parameters are background statistics:

        median = sigma_clipped_median,   beta = asinh_softening * clipped_std

    A foreign frame has no paired ideal, so gain/sky cannot be fit that way. But
    the two numbers the transform actually depends on ARE recoverable from the
    frame itself: match its sigma-clipped median and std to the training ones
    and the stretch lands where the model expects.

    Do NOT match a bright percentile instead. m32's p99.9/p50 is 2.25 against
    m31's 354 -- a RATIO mismatch no affine map can absorb. Matching p99.9
    needed a 123x scale, drove everything below the median to negative flux
    (conditioning p1 = -0.33, outside [0,1]) and made the solve diverge.
    """
    from astropy.stats import sigma_clipped_stats
    _, med_in, std_in = sigma_clipped_stats(d, sigma=3.0, maxiters=5)
    med_t = obs_norm.p["median"]
    std_t = obs_norm.p["beta"] / max(softening, 1e-30)
    scale = std_t / max(float(std_in), 1e-30)
    out = (d - float(med_in)) * scale + med_t
    if verbose:
        print(f"[input] --match-flux (background stats): "
              f"median {float(med_in):.6g}->{med_t:.6g}, "
              f"std {float(std_in):.6g}->{std_t:.6g}  (scale {scale:.6g})")
    return out


def load_single_image(path, hdu=0, region=None, minmax=True, verbose=True):
    """Load ONE .fits/.npy image and put it in the pipeline's flux domain.

    The dataset path stores patches already asinh-normalized, and the solver
    recovers flux with obs_norm.inverse(). A raw file has to enter the SAME
    domain, which means reproducing load_astro.py's per-image MIN-MAX step:

        raw counts --min-max--> "flux" --obs_norm.forward--> z (conditioning)

    `minmax=False` skips that and treats the file as already being in the flux
    domain -- correct only for a file cut from an already-normalized global.

    NOTE the min-max is per-image, so a file whose dynamic range differs from
    m31bK50's lands on a different flux scale even though the arithmetic is
    identical. That is the "21x flux" normalization artifact in another guise,
    and it is why cross-field transfer needs the z-histogram check that the
    caller prints.
    """
    path = str(path)
    if path.endswith(".npy"):
        d = np.load(path)
    else:
        from astropy.io import fits as _f
        with _f.open(path) as hl:
            d = hl[hdu].data
    d = np.asarray(d, dtype=np.float64).squeeze()
    if d.ndim != 2:
        raise SystemExit(f"{path}: expected a 2-D image, got shape {d.shape}")
    if not np.isfinite(d).all():
        n = int((~np.isfinite(d)).sum())
        d = np.nan_to_num(d, nan=float(np.nanmedian(d)), posinf=0.0, neginf=0.0)
        if verbose:
            print(f"[input] replaced {n} non-finite pixels with the median")
    # NORMALIZE ON THE FULL FRAME, THEN CROP -- load_astro.py min-max normalizes
    # each GLOBAL and cuts patches afterwards. Cropping first would rescale by
    # the crop's own max (for m32 that is 2187 vs the frame's 16852, a 7.7x
    # error) and silently move the data to a different flux scale per region.
    lo, hi = float(d.min()), float(d.max())
    if minmax:
        d = (d - lo) / max(hi - lo, 1e-30)
    if region is not None:
        y0, y1, x0, x1 = region
        d = d[y0:y1, x0:x1]
    if verbose:
        print(f"[input] {path}  shape {d.shape}  frame min {lo:.6g} "
              f"max {hi:.6g}"
              f"{'  -> min-max on the FULL frame' if minmax else '  (used as-is)'}")
    return d[None, None]                      # (1, 1, H, W)


def t_for_sigma(sigmas: torch.Tensor, sigma: float, t_max: int,
                t_min: int = 1) -> int:
    """t whose noise variance is closest to sigma^2, clamped to [t_min, t_max].

    t_max matters: the checkpoint's identity/fixed-point term was trained with
    identity_t_max=100 and the operator degrades sharply past it (||D(x)-x||
    on true patches is 0.015 at t=50, 0.216 at t=100, 46.6 at t=200).

    t_min matters for a different reason. With --rho-scale > 1, rho grows every
    outer iteration, so sigma = sqrt(lam/rho) shrinks and the matched t walks
    down (29 -> 17 -> 9 -> 4 -> 1 in the reference run). But a chain started at
    t=1 is a near-identity operator -- the denoiser stops contributing exactly
    during the late iterations that are supposed to be refining the solution.
    Flooring t keeps the prior alive to the end.
    """
    t = int(torch.argmin((sigmas - float(sigma)).abs()).item())
    return max(int(t_min), min(t, int(t_max)))


# ----------------------------------------------------------------------------
# z-update: a short conditional reverse chain (the paper's "set number of
# denoising steps"), NOT a single Tweedie step
# ----------------------------------------------------------------------------
@torch.no_grad()
def diffusion_denoise(model, v_flux, y_z, sigma_z, n_steps, alpha_bar, sigmas,
                      ideal_norm, t_max=75, t_min=1, guidance=1.0,
                      has_null=False, eta=0.0, generator=None,
                      clamp=(0.0, 1.0), renoise=False, trace=None,
                      inject_scale=1.0):
    """Treat v as a noisy image at normalized-domain noise level sigma_z, map
    that to a timestep, and run the reverse chain from there down to ~0.

    v already CARRIES noise of std sigma_z, so tweedie_chain scales it into the
    diffusion parameterization rather than injecting fresh noise:
        x_t = sqrt(abar_t) * (x_0 + sigma(t) eps) = sqrt(abar_t) * z
    """
    z = ideal_norm.forward(v_flux)
    t0 = t_for_sigma(sigmas, sigma_z, t_max, t_min)
    z0 = tweedie_chain(model, z, y_z, t0, n_steps, alpha_bar,
                       guidance=guidance, has_null=has_null, eta=eta,
                       generator=generator, clamp=clamp, renoise=renoise,
                       trace=trace, inject_scale=inject_scale)
    return ideal_norm.inverse(z0), t0


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="/home/alex/noir_ml/global/ml-decon/"
                                     "checkpoints_cond_diffusion_npy/best.pt")
    p.add_argument("--prior", choices=["cond", "flat", "cond31"], default="cond",
                   help="WHICH PRIOR sits in the z-update. 'cond' is the "
                        "conditional diffusion checkpoint at --ckpt (the "
                        "measurement is fed to the network as a second "
                        "channel). 'flat' is the UNCONDITIONAL flat-CNN "
                        "checkpoint at --flat-ckpt, driven through the "
                        "identical DDIM chain via flat_prior.FlatPrior -- so "
                        "the two differ in the network and its normalization, "
                        "and in nothing else. With 'flat', --ckpt is still "
                        "loaded, for its dataset_norm only (nothing else in "
                        "the loop needs the conditional weights).")
    p.add_argument("--flat-ckpt",
                   default="checkpoints/flat_cnn_stars_v3_best.pt",
                   help="unconditional checkpoint for --prior flat")
    p.add_argument("--drop-conditioning", action="store_true", default=False,
                   help="--prior cond only: feed the NULL token instead of the "
                        "measurement, turning the conditional checkpoint into "
                        "an unconditional prior with the SAME weights and the "
                        "SAME training data. Valid only for a checkpoint "
                        "trained with p_uncond > 0 (these were, at 0.15), "
                        "otherwise the null token is untrained. This is the "
                        "cell that completes the 2x2 against --prior flat: it "
                        "isolates CONDITIONING from the training field, which "
                        "comparing an m32-trained conditional against an "
                        "m31-trained unconditional cannot do.")
    p.add_argument("--cond31-ckpt", default="checkpoints_cond_diffusion/best.pt",
                   help="CONTROL arm for --prior cond31: a CONDITIONAL prior "
                        "trained on the same mycode/patches m31 data as "
                        "--flat-ckpt. Against 'flat' it isolates conditioning "
                        "(same training field); against 'cond' it isolates the "
                        "training field (both conditional). It shares "
                        "--flat-scale and uses --flat-gain as its conditioning "
                        "divisor.")
    p.add_argument("--flat-transform",
                   default="checkpoints/flat_cnn_stars_v3_transform.json",
                   help="AsinhTransform JSON matching --flat-ckpt; asserted "
                        "against the constants stored in the checkpoint")
    p.add_argument("--flat-scale", default="auto", metavar="S|auto",
                   help="flux scale mapping THIS dataset's flux into the flat "
                        "checkpoint's training flux domain (see flat_prior.py "
                        "and m32_prior_domain_check.py). 'auto' = the "
                        "truth-free mean-flux anchor: mean(observed)/--flat-"
                        "gain matched to --flat-train-mean. A scale fitted "
                        "from the bright end instead is ~30x too high on an "
                        "ml-decon dataset, because the flat prior's A carries "
                        "mycode/patches' 22.8x blurry/sharp artifact.")
    p.add_argument("--flat-gain", type=float, default=1.0, metavar="G",
                   help="forward-model gain of THIS dataset, observed ~ G * "
                        "(psf*ideal), used only by --flat-scale auto. On "
                        "m32_klong_nosky the per-patch median is 0.81.")
    p.add_argument("--flat-train-mean", type=float, default=9.8058e-06,
                   help="mean pixel flux of the flat checkpoint's TRAINING "
                        "sharp patches (mycode/patches/sharp_patches.npy). "
                        "The other half of the mean-flux anchor.")
    p.add_argument("--flat-clamp", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="clamp on the flat chain's x0 estimate, in NET units "
                        "(default: the transform's own [net_floor, net_ceil], "
                        "i.e. the support of its training data -- the direct "
                        "analogue of --clamp 0 1 on the conditional path)")
    p.add_argument("--data-dir", type=Path,
                   default=Path("/home/alex/noir_ml/global/ml-decon/data/m31bK50"))
    p.add_argument("--split", choices=["val", "train"], default="val")
    p.add_argument("--indices", default="0:16")
    p.add_argument("--psf-file", default="psf_50_true.fits")
    p.add_argument("--psf-sigma", type=float, default=None)

    p.add_argument("--iters", type=int, default=30,
                   help="ADMM outer iterations. Far fewer than RED needs "
                        "because the x-update is exact, but each one costs "
                        "--denoise-steps model calls.")
    p.add_argument("--denoise-steps", type=int, default=8,
                   help="reverse-chain steps in the z-update. The paper's "
                        "'set number of denoising steps' per outer iteration. "
                        "1 reproduces the one-step RED behaviour (and its "
                        "0.24 sharpening ceiling); 8-20 is the useful range.")
    p.add_argument("--rho", type=float, default=1.0,
                   help="ADMM penalty. Large rho = trust the prior/split, "
                        "small rho = trust the data solve.")
    p.add_argument("--rho-scale", type=float, default=1.0,
                   help="multiply rho by this each outer iteration "
                        "(continuation; >1 tightens the split over the run). "
                        "COMPOUNDS: 1.1 over 400 iters ends at rho=4e15. "
                        "Prefer --rho-end, which is independent of --iters.")
    p.add_argument("--rho-end", type=float, default=None, metavar="R",
                   help="ramp rho GEOMETRICALLY from --rho to R over the run, "
                        "deriving the per-iteration ratio (overrides "
                        "--rho-scale). This is the log noise anneal of Park et "
                        "al. 2026: sigma_z = sqrt(lam/rho) decays log-linearly, "
                        "and their Thm 2 needs an annealed sigma -> 0 to "
                        "converge to a critical point of the EXACT objective "
                        "rather than the Gaussian-smoothed one. Endpoint "
                        "parameterisation keeps the schedule fixed when you "
                        "change --iters.")
    p.add_argument("--lam", type=float, default=0.01,
                   help="prior weight. Enters ONLY through the denoising "
                        "level sigma = sqrt(lam/rho), i.e. the paper's "
                        "noise-variance-matching heuristic.")
    p.add_argument("--t-max", type=int, default=75,
                   help="cap on the matched timestep; keep <= the "
                        "checkpoint's identity_t_max")
    p.add_argument("--t-min", type=int, default=1,
                   help="FLOOR on the matched timestep. With --rho-scale > 1 "
                        "the matched t decays to 1, where the chain is a "
                        "near-identity and the prior stops acting. Raising "
                        "this keeps the denoiser contributing to the end.")
    p.add_argument("--eta", type=float, default=0.0,
                   help="DDIM stochasticity, MUST be in [0, 1]. 0 = fully "
                        "deterministic (measured best for photometry: flux "
                        "0.995 vs 0.972 at eta=1), 1 = full ancestral. Values "
                        "> 1 are NOT 'more exploration' -- they make "
                        "1 - abar_next - s^2 negative, which clamps the "
                        "deterministic direction term to zero and deletes "
                        "every refinement step in the chain.")
    p.add_argument("--avg-last", type=int, default=0,
                   help="average the final K z-iterates into the output "
                        "(Polyak averaging). The z-update is a STOCHASTIC, "
                        "non-contractive operator, so the iteration has no "
                        "fixed point -- it samples a neighbourhood. Measured "
                        "relative oscillation of the primal residual is ~0.5 "
                        "with --renoise. Averaging is the standard variance "
                        "reduction for that and costs no extra model calls. "
                        "0 = off (use the final iterate). Only the OUTPUT is "
                        "averaged; the ADMM recursion itself is untouched.")
    p.add_argument("--renoise", action="store_true",
                   help="do a PROPER forward diffusion at the chain entry, "
                        "x_t = sqrt(abar)*z + sqrt(1-abar)*eps, instead of the "
                        "deterministic rescale. Late in an ADMM run z is much "
                        "cleaner than sigma(t0), so the model sees a too-clean "
                        "input, does nothing, and preserves blur. This is the "
                        "calibrated way to get the regeneration that eta > 1 "
                        "produces by accident -- strength is set by t0 (see "
                        "--t-min), not by an unbounded multiplier.")
    p.add_argument("--eta-end", type=float, default=None,
                   help="anneal eta linearly from --eta to this over the outer "
                        "iterations (also must be in [0,1]). Default: constant.")
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--input-file", default=None, metavar="PATH",
                   help="run on ONE .fits/.npy image instead of the "
                        "--data-dir patch arrays. Truth-dependent metrics are "
                        "skipped unless --truth-file is also given. The file "
                        "is min-max normalized to enter the same flux domain "
                        "the patches live in (see --no-input-minmax).")
    p.add_argument("--input-hdu", type=int, default=0,
                   help="FITS HDU index for --input-file / --truth-file")
    p.add_argument("--truth-file", default=None, metavar="PATH",
                   help="a GROUND TRUTH matched to --input-file, enabling the "
                        "usual accuracy metrics. Only pass a real truth; for "
                        "another solver's output use --compare-file.")
    p.add_argument("--compare-file", default=None, metavar="PATH",
                   help="a RIVAL RECONSTRUCTION to be scored against, not a "
                        "truth. Nothing is called accuracy: the comparison is "
                        "(a) forward residual ||g*A x - b||/||b|| with g fit "
                        "per image, which asks which reconstruction explains "
                        "the OBSERVED data better and needs no truth, and (b) "
                        "point-source concentration, i.e. which is sharper. "
                        "Use this to beat an existing deconvolution.")
    p.add_argument("--compare-label", default=None, metavar="NAME",
                   help="name for --compare-file in the printout and figure "
                        "(default: the file's basename)")
    p.add_argument("--region", default=None, metavar="Y0:Y1,X0:X1",
                   help="crop --input-file before solving, e.g. 512:768,512:768"
                        ". A full 2048x2048 frame is ~64x the cost of a "
                        "16-patch run; crop first to size a job.")
    p.add_argument("--match-flux", action="store_true", default=False,
                   help="REQUIRED for a frame from another field. Affinely maps "
                        "the input's sigma-clipped median and std onto the "
                        "checkpoint's, which is exactly what the asinh "
                        "transform's median/beta encode. Without it the frame "
                        "sits wherever its own min-max puts it and the denoiser "
                        "runs out of distribution (m32: a 256x256 region spans "
                        "0.083 of the normalized range vs 0.473 for a training "
                        "patch, and the output is speckle).")
    p.add_argument("--fit-gain-sky", action="store_true", default=False,
                   help="with --truth-file, fit observed ~ gain*truth + sky on "
                        "block means and store (observed-sky)/gain -- exactly "
                        "gen_data.py's preprocessing. This is the CORRECT "
                        "normalization whenever a matched truth exists, and it "
                        "supersedes --match-flux (which is the no-truth "
                        "fallback and only matches background statistics).")
    p.add_argument("--asinh-softening", type=float, default=1.0,
                   help="the dataset's asinh_softening, used by --match-flux to "
                        "recover the training noise std as beta/softening. "
                        "m31bK50's resolved_config.yaml says 1.0.")
    p.add_argument("--no-input-minmax", dest="input_minmax",
                   action="store_false", default=True,
                   help="treat --input-file as ALREADY in the flux domain "
                        "(skip the per-image min-max that load_astro.py "
                        "applies to every global)")
    p.add_argument("--gain", type=float, default=1.0, metavar="G",
                   help="forward-model gain: b = G * (psf * x) + sky. A pure "
                        "scaling of x, so it only changes what flux_ratio "
                        "means. Measured 0.822 on m32_klong_nosky, 17.83 on "
                        "the m32b_phase_v2 ef/l160 pair.")
    p.add_argument("--sky", default="0.0", metavar="S|auto",
                   help="additive sky in the forward model. 'auto' = each "
                        "patch's own sigma-clipped median of the OBSERVED, "
                        "which is truth-free. Unlike --gain this is a "
                        "structural correction: when x is a point-source "
                        "catalogue the diffuse light has nowhere to live, so "
                        "leaving it out makes the solver manufacture it out of "
                        "sources. Note the CONDITIONING keeps the sky either "
                        "way -- the network was trained on observed patches "
                        "that have one.")
    p.add_argument("--nonneg", action="store_true", default=False)
    p.add_argument("--sparse-tau", type=float, default=2.0, metavar="K",
                   help="SPARSITY PROX composed after the diffusion chain in the "
                        "z-update: threshold at K * the sigma-clipped std of "
                        "that iterate's own background, in flux. 0 disables. "
                        "The target field is ~95%% exact zeros and NOTHING else "
                        "in this pipeline penalises a non-zero, which is the "
                        "leading suspect for the 75%% spurious rate. Note the "
                        "data term cannot do this job: deleting the spurious "
                        "detections from a finished reconstruction makes "
                        "||Az-b|| WORSE by 4.4%%, because they are fitting "
                        "noise -- so no reweighting of the forward model "
                        "removes them, and a prior-side penalty is the only "
                        "lever. K is in units of the iterate's own background "
                        "sigma so it transfers across the very different flux "
                        "domains each checkpoint imposes. Do not raise it to 3: "
                        "that zeroes the DENSE patches' real background too "
                        "(idx 846 truth sigma 0.0071 -> 0.0).\n"
                        "DEFAULT 2.0, BUT IT IS A PHOTOMETRY/PURITY TRADE AND "
                        "THE CHOICE IS APPLICATION-DEPENDENT. Measured on 128 "
                        "m32 val patches, tau 2.0 against tau 0 "
                        "(full_benchmark.sh):\n"
                        "  WINS  background floor sigma 0.00054 -> 0.00032 and "
                        "exact zeros 33.8%% -> 82.1%% on the flat patches "
                        "(truth 0.0 / 95.0%%); purity 55.9%% -> 70.6%%; "
                        "concentration 0.857 -> 0.877; and the data residual "
                        "moves TOWARD the truth's own 0.5195 (|delta| 0.0173 "
                        "-> 0.0066).\n"
                        "  LOSES per-source photometry, at the faint end, "
                        "monotonically. Median flux ratio 1.001 -> 0.902, and "
                        "the magnitude-binned bias goes from FLAT (base is "
                        "within 0.05 mag in all 8 bins) to +0.04, +0.03, "
                        "+0.07, +0.06, +0.17, +0.19, +0.16, +0.42 mag. "
                        "Scatter is worse in every bin (e.g. 0.243 -> 0.404 at "
                        "mag 1.86). Cause: the cut clips the WINGS of faint "
                        "sources, so their 3x3 box flux drops. That bias is "
                        "magnitude-dependent, so it CANNOT be calibrated out "
                        "with a constant.\n"
                        "So: use 2.0 when spurious sources are the problem, "
                        "and 0 when the deliverable is photometry (a "
                        "luminosity function or a CMD). Do not trust the "
                        "8-patch flat-patch TOTAL flux of 1.003 that first "
                        "motivated this default -- total flux includes the "
                        "removed floor and hid the per-source deficit.")
    p.add_argument("--sparse-mode", choices=["hard", "soft"], default="hard",
                   help="'soft' is the true L1 prox, x -> sign(x)*max(|x|-tau, 0), "
                        "and shrinks EVERY source including the bright ones -- "
                        "on a field already carrying a +2 mag faint bias that "
                        "makes the photometry worse. 'hard' zeroes below tau "
                        "and leaves survivors untouched, which is not a proper "
                        "prox but does not add bias. Try hard first.")
    p.add_argument("--inject-scale", type=float, default=1.0, metavar="S",
                   help="sigma_inject / sigma_cond for --renoise. 1.0 (default) "
                        "is the MATCHED setting: the entry noise equals the "
                        "noise the conditioning timestep implies. Park et al. "
                        "2026 (arXiv:2604.03603, App. C.1) show matched is "
                        "suboptimal because the iterate also carries residual "
                        "measurement noise and forward-operator artifacts, so "
                        "the denoiser should be conditioned HIGHER than it is "
                        "perturbed: their CS-MRI ADMM row uses S=0.01 and gains "
                        "+1.88 dB, though their deblurring row stays matched. "
                        "No effect without --renoise.")
    p.add_argument("--weights", choices=["ema", "raw"], default="ema")
    p.add_argument("--net-float64", action="store_true", default=False,
                   help="also run the NETWORK's forward pass in float64. "
                        "Everything else (normalization, schedule, FFT solve, "
                        "ADMM state, chain arithmetic) is float64 "
                        "unconditionally. This flag only casts the trained "
                        "weights, which costs ~3x runtime on CPU and cannot "
                        "make the weights themselves more accurate -- eps is a "
                        "learned quantity carrying ~1e-2 error, six orders "
                        "above fp32 epsilon. Use it to VERIFY the boundary is "
                        "not the limiter, not as a default.")
    p.add_argument("--norm-from-data-dir", action="store_true", default=False,
                   help="take the normalization from <data-dir>/norm.json "
                        "instead of the checkpoint's baked-in dataset_norm. "
                        "REQUIRED whenever the dataset has been regenerated or "
                        "repaired since the checkpoint was trained: the "
                        "default silently prefers the checkpoint, so the "
                        "network's z is read in a domain it was never trained "
                        "in. The two are compared and any disagreement is "
                        "reported either way -- read that line before "
                        "trusting a flux ratio.")
    p.add_argument("--split-norms", action="store_true", default=False,
                   help="decode the STORED patches with <data-dir>/norm.json "
                        "while the NETWORK keeps the checkpoint's dataset_norm "
                        "(conditioning encoded into it, its z output decoded "
                        "out of it). This is the correct handling when only "
                        "the ENCODING of the dataset changed since training "
                        "-- e.g. m32_klong_nosky, whose ideal beta was "
                        "repaired from the 1.0 fallback to 0.0175 after the "
                        "checkpoint was trained. --norm-from-data-dir and the "
                        "default each pick ONE norm for both jobs and so get "
                        "one of them wrong; this gets both right. Both "
                        "encodings decode to the same flux, which is what "
                        "makes it valid -- check the printed flux stats.")
    p.add_argument("--clamp", type=float, nargs=2, default=(0.0, 1.0),
                   metavar=("LO", "HI"),
                   help="clamp the chain's x0 estimate in NORMALIZED units. "
                        "The default (0, 1) is the training range. Widen it "
                        "(e.g. -0.5 2.0) to let the prior overshoot and see "
                        "what it would have done unconstrained; the clamp "
                        "hides divergence rather than fixing it.")
    p.add_argument("--no-fits", dest="write_fits", action="store_false",
                   help="skip writing <out-dir>/fits/{recon,truth}/*.fits "
                        "(they are what the external photometry tools consume)")
    p.add_argument("--plot-linear", action="store_true", default=False,
                   help="also write patches_linear.png, the same panels on a "
                        "LINEAR stretch clipped at --linear-pct. asinh is "
                        "honest about faint structure but flatters the bright "
                        "end; a linear panel shows whether a peak is actually "
                        "there. Read the two together.")
    p.add_argument("--linear-pct", type=float, default=99.5, metavar="P",
                   help="upper percentile of the reference panel used as vmax "
                        "for --plot-linear.")
    p.add_argument("--plot-combined", action="store_true", default=False,
                   help="also write patches_combined.png: asinh and linear "
                        "rows interleaved per patch, for a single figure that "
                        "shows both stretches at once.")
    p.add_argument("--n-show", type=int, default=6,
                   help="rows in patches.png, one per patch. 0 or negative = "
                        "ALL the patches in --indices.")
    p.add_argument("--crop", type=int, default=4)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("admm_diff_out"))
    p.add_argument("--trace-chain", type=int, default=0, metavar="N",
                   help="record per-chain-step diagnostics for the first N "
                        "outer iterations and every --log-every-th one after, "
                        "into <out-dir>/chain_trace.csv. The column that "
                        "matters is dz0_rms: how far the chain's x0 estimate "
                        "moves at each step, in normalized units. If it decays "
                        "to ~0 over the chain, the late steps are dead and "
                        "entry-only --renoise cannot reach them (which is what "
                        "eta > 1 was compensating for). 1 normalized unit is "
                        "~14.5 mag at the bright end, so dz0_rms=0.001 is "
                        "~0.015 mag of movement.")
    args = p.parse_args()

    # eta outside [0,1] over-noises relative to the DDIM derivation. NOTE the
    # old claim here -- that it "deletes every refinement step" -- was measured
    # FALSE on 2026-08-11: at t0=75/n_steps=20 the deterministic term is zeroed
    # in 0/19 steps at eta=1.0, 2/19 at eta=1.5, 6/19 at eta=2.0, and only at
    # the tail of the chain where the term is already ~0.006. eta=1.5 injects
    # ~5.6x the total noise of --renoise alone with 17 intact steps. Warn, but
    # do not mis-describe it. See tweedie_chain's docstring.
    for nm, v in (("--eta", args.eta), ("--eta-end", args.eta_end)):
        if v is not None and not (0.0 <= v <= 1.0):
            print(f"warning: {nm}={v} is outside [0, 1]. DDIM's eta is a "
                  f"fraction of the ancestral noise, not a gain, so this "
                  f"over-noises every step and zeroes the deterministic term "
                  f"in the last few. If it helps, that is evidence the chain "
                  f"wants more noise than t_max allows (sigma=sqrt(lam/rho) "
                  f"asks for t=188 at lam=0.01/rho=0.1 and is clamped to "
                  f"t_max); prefer raising --t-max or --renoise, and use "
                  f"--trace-chain to see which steps are actually moving.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Dump the exact config next to the outputs. Without this a directory of
    # FITS is unattributable after the fact, which has already cost one
    # unreproducible tuning result.
    (args.out_dir / "config.json").write_text(json.dumps(
        {k: (str(v) if isinstance(v, Path) else v)
         for k, v in vars(args).items()}, indent=2, sort_keys=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    # DTYPE is the dtype of EVERYTHING outside the network: the normalization,
    # the schedule, the ADMM state, the FFT solve, the reverse-chain
    # arithmetic. The network's own weights are set separately by
    # --net-float64, because doubling them costs ~3x runtime and buys nothing
    # (eps is a learned quantity with ~1e-2 error of its own).
    DTYPE = torch.float64
    net_dtype = torch.float64 if args.net_float64 else torch.float32
    model, T, ck = load_checkpoint(args.ckpt, device, weights=args.weights,
                                   dtype=net_dtype)
    alpha_bar = cosine_alpha_bar(T, dtype=DTYPE).to(device)
    sigmas = sigma_of_t(alpha_bar)                  # float64, follows alpha_bar
    # The checkpoint carries the normalization it was TRAINED with; the data dir
    # carries the one its .npy files are CURRENTLY encoded with. They diverge the
    # moment a dataset is regenerated or repaired, and the network only speaks the
    # former -- so a silent preference for either one can be wrong. Compare them
    # loudly and let --norm-from-data-dir decide.
    ck_norms = ck["dataset_norm"]
    dir_path = args.data_dir / "norm.json"
    dir_norms = json.loads(dir_path.read_text()) if dir_path.exists() else None
    if ck_norms and dir_norms:
        for ch in ("observed", "ideal"):
            a, b_ = ck_norms.get(ch, {}), dir_norms.get(ch, {})
            keys = sorted((set(a) | set(b_)) - {"method"})
            bad = [k for k in keys
                   if not np.isclose(float(a.get(k, np.nan)),
                                     float(b_.get(k, np.nan)), rtol=1e-4)]
            if bad:
                print(f"[norm] *** MISMATCH on '{ch}' *** checkpoint vs "
                      f"{dir_path}: " + ", ".join(
                          f"{k} {a.get(k)!r} vs {b_.get(k)!r}" for k in bad))
                print(f"[norm]     the network emits z in the CHECKPOINT's "
                      f"domain. Using the "
                      + ("data dir's" if args.norm_from_data_dir
                         else "checkpoint's")
                      + " -> " + ("truth decodes correctly, the recon does NOT"
                                  if args.norm_from_data_dir else
                                  "the recon decodes correctly, the truth does "
                                  "NOT")
                      + ". Either way the flux ratio is meaningless; retrain "
                        "or regenerate so they agree.")
    if args.split_norms:
        if dir_norms is None:
            raise SystemExit(f"--split-norms but {dir_path} does not exist")
        if not ck_norms:
            raise SystemExit("--split-norms but the checkpoint has no "
                             "dataset_norm to split against")
        norms = ck_norms                    # the network's domain
        print(f"[norm] --split-norms: network <- checkpoint dataset_norm, "
              f"stored patches <- {dir_path}")
    elif args.norm_from_data_dir:
        if dir_norms is None:
            raise SystemExit(f"--norm-from-data-dir but {dir_path} does not exist")
        norms = dir_norms
        print(f"[norm] using {dir_path} (--norm-from-data-dir)")
    else:
        norms = ck_norms or dir_norms
        print("[norm] using the checkpoint's dataset_norm"
              if ck_norms else f"[norm] using {dir_path} (checkpoint has none)")
    if norms is None:
        raise SystemExit("no normalization: the checkpoint has no dataset_norm "
                         f"and {dir_path} does not exist")
    obs_norm, ideal_norm = TorchNorm(norms["observed"]), TorchNorm(norms["ideal"])
    # The norms the STORED .npy patches are encoded with. Same objects unless
    # --split-norms, in which case they are the data dir's and the pair above
    # stays the network's.
    if args.split_norms:
        store_obs_norm = TorchNorm(dir_norms["observed"])
        store_ideal_norm = TorchNorm(dir_norms["ideal"])
    else:
        store_obs_norm, store_ideal_norm = obs_norm, ideal_norm
    has_null = float(ck.get("p_uncond") or 0.0) > 0.0

    # ---- the unconditional prior, if that is the one being compared --------
    flat_prior = None
    if args.prior in ("flat", "cond31"):
        from flat_prior import FlatPrior, CondM31Prior, mean_flux_scale
        # The scale needs the data's mean flux, which is not loaded yet, so
        # build the prior after the data block below. Defer by remembering the
        # constructor arguments.
        flat_prior = ("deferred",
                      FlatPrior if args.prior == "flat" else CondM31Prior,
                      mean_flux_scale, net_dtype)

    if args.input_file:
        region = None
        if args.region:
            try:
                ys, xs = args.region.split(",")
                region = (int(ys.split(":")[0]), int(ys.split(":")[1]),
                          int(xs.split(":")[0]), int(xs.split(":")[1]))
            except Exception:
                raise SystemExit("--region must look like Y0:Y1,X0:X1")
        # --fit-gain-sky derives the scaling from the pair, so a prior min-max
        # would be a second, conflicting transform of the same data.
        use_minmax = args.input_minmax and not args.fit_gain_sky
        if args.input_minmax and args.fit_gain_sky:
            print("[input] --fit-gain-sky supersedes the min-max; skipping it "
                  "(they are two different scalings of the same frame)")
        b_np = load_single_image(args.input_file, args.input_hdu, region,
                                 use_minmax)
        if args.match_flux and not args.fit_gain_sky:
            b_np = match_flux_distribution(b_np, obs_norm,
                                           softening=args.asinh_softening)
        b = torch.from_numpy(b_np).to(DTYPE).to(device)   # flux domain
        y_z = obs_norm.forward(b)                         # conditioning
        idx = [0]
        if args.truth_file:
            t_np = load_single_image(args.truth_file, args.input_hdu, region,
                                     minmax=False)
            if args.fit_gain_sky:
                # gen_data.py's actual preprocessing: put the observed into the
                # TRUTH's photometric units. Do not also --match-flux; that is
                # the no-truth fallback and would undo this.
                gain, sky = fit_gain_sky(b_np, t_np)
                b_np = (b_np - sky) / max(gain, 1e-30)
                b = torch.from_numpy(b_np).to(DTYPE).to(device)
                y_z = obs_norm.forward(b)
            elif args.match_flux:
                # No gain fit: the truth still needs SOME common footing, so
                # match its background to ideal_norm's. Guard the degenerate
                # case of a point-source model, whose clipped std is ~0.
                from astropy.stats import sigma_clipped_stats as _scs
                _, _, _sd = _scs(t_np, sigma=3.0, maxiters=5)
                if float(_sd) > 1e-12:
                    t_np = match_flux_distribution(t_np, ideal_norm,
                                                   softening=args.asinh_softening)
                else:
                    print("[input] truth has ~zero clipped std (a point-source "
                          "model): skipping --match-flux on it and leaving it "
                          "in native units. Flux ratios will not be "
                          "meaningful -- use --fit-gain-sky instead.")
            truth = torch.from_numpy(t_np).to(DTYPE).to(device)
        else:
            truth = None
        if args.compare_file:
            # NOT normalized to the training stats: a rival reconstruction has
            # its own units, and the metrics below fit a per-image gain, which
            # makes the comparison invariant to that choice.
            cmp_np = load_single_image(args.compare_file, args.input_hdu,
                                       region, minmax=False)
        else:
            cmp_np = None
        # The checkpoint's normalization was FIT to m31bK50. A file from
        # another field lands wherever its own min-max puts it, so print the
        # conditioning histogram against the training range instead of
        # silently running out of distribution.
        zq = np.percentile(y_z.cpu().numpy(), [1, 50, 99, 99.9, 100])
        print("[input] conditioning z percentiles "
              "1/50/99/99.9/100: " + " ".join(f"{v:.4f}" for v in zq))
        print("[input]   m31bK50 val_observed for reference: "
              "0.1223 / 0.3208 / 0.6155 / 0.7212 / 0.9972")
        if zq[1] < 0.1 or zq[1] > 0.6 or zq[4] > 1.05:
            print("[input] WARNING: conditioning distribution is far from the "
                  "training range -- the prior is being used out of "
                  "distribution and fluxes may be badly scaled "
                  "(cf. the 1.9x error seen transferring to held-out m31bK50).")
    else:
        observed = np.load(args.data_dir / f"{args.split}_observed.npy")
        ideal = np.load(args.data_dir / f"{args.split}_ideal.npy")
        idx = parse_indices(args.indices, len(observed))
        # .to(DTYPE) not .float(): y_z is the model conditioning AND the
        # argument to obs_norm.inverse, whose sinh amplifies any rounding here.
        y_z = torch.from_numpy(observed[idx]).to(DTYPE).to(device)
        # FLUX is the common ground: decode each stored array with the norm it
        # was ENCODED with, then re-encode the conditioning into the domain the
        # NETWORK speaks. Without --split-norms both norms are the same object
        # and this is bit-identical to the old two lines.
        b = store_obs_norm.inverse(y_z)             # measurement, flux units
        truth = store_ideal_norm.inverse(
            torch.from_numpy(ideal[idx]).to(DTYPE).to(device))
        if args.split_norms:
            y_z = obs_norm.forward(b)               # conditioning, ckpt domain
            zq = np.percentile(y_z.cpu().numpy(), [1, 50, 99, 100])
            print("[norm] conditioning re-encoded into the checkpoint's "
                  "observed domain: z 1/50/99/100 = "
                  + " ".join(f"{v:.4f}" for v in zq))
        # Field-level mean flux, for --flat-scale auto. It must NOT come from
        # the selected patches: patch means are dominated by whether a bright
        # star happens to land in them (on m32_klong_nosky val, indices 0:2 sit
        # 18x below the field mean), and the prior's calibration would then
        # change with --indices. Subsample to keep the decode cheap.
        _step = max(1, len(observed) // 512)
        field_obs_mean = float(store_obs_norm.inverse(
            torch.from_numpy(observed[::_step]).to(DTYPE)).mean())
        cmp_np = None
    B, _, H, W = y_z.shape

    # ---- build the unconditional prior now that the data's scale is known --
    if args.prior in ("flat", "cond31"):
        _, PriorClass, mean_flux_scale, _dt = flat_prior
        if str(args.flat_scale).lower() == "auto":
            s = mean_flux_scale(field_obs_mean, args.flat_gain,
                                args.flat_train_mean)
            print(f"[flat]   (field mean over {args.split}_observed; the "
                  f"selected patches' own mean is {float(b.mean()):.6g}, "
                  f"{float(b.mean()) / field_obs_mean:.2f}x the field -- the "
                  f"field value is the one used, so the scale does not move "
                  f"with --indices)")
        else:
            s = float(args.flat_scale)
            print(f"[flat] flux scale s={s:.6g} (given)")
        if args.prior == "flat":
            flat_prior = PriorClass(args.flat_ckpt, args.flat_transform,
                                    flux_scale=s, device=device,
                                    weights=args.weights, dtype=_dt)
        else:
            flat_prior = PriorClass(args.cond31_ckpt, flux_scale=s,
                                    cond_divisor=args.flat_gain, device=device,
                                    weights=args.weights, dtype=_dt)
            flat_prior.set_conditioning(b)
        # What the prior will actually SEE, against what it was trained on.
        # A z-histogram far from the training one is the single most reliable
        # predictor of a runaway prior on a foreign field.
        _to_z = (flat_prior.T.to_z if args.prior == "flat" else flat_prior.to_z)
        _zb = _to_z(b * s).cpu().numpy()
        print("[flat] z of the OBSERVED under s, 50/99/99.9/100: "
              + " ".join(f"{v:.3f}" for v in np.percentile(_zb, [50, 99, 99.9, 100])))
        if truth is not None:
            _zt = _to_z(truth * s).cpu().numpy()
            print("[flat] z of the TRUTH    under s, 50/99/99.9/100: "
                  + " ".join(f"{v:.3f}" for v in np.percentile(_zt, [50, 99, 99.9, 100]))
                  + ("   | training sharp patches: 0.000 0.396 0.611 1.362"
                     if args.prior == "flat" else ""))
            if args.prior == "flat":
                over = float((flat_prior.T.to_z(truth * s)
                              >= flat_prior.T.z_ceil).float().mean())
                print(f"[flat] truth pixels at/above z_ceil="
                      f"{flat_prior.T.z_ceil}: {100 * over:.4f}%  (flux the "
                      f"prior cannot represent)")

    kernel = (gaussian_kernel(args.psf_sigma) if args.psf_sigma is not None
              else load_kernel(args.psf_file))
    K = make_otf(kernel, (H, W), device)

    # ---- gain and sky IN THE OPERATOR, not in the data ---------------------
    # The forward model is  b = gain * (psf * x) + sky, and the two terms are
    # not interchangeable:
    #   gain  is a pure scaling of x. It changes what flux_ratio MEANS and
    #         nothing else about the solution's structure.
    #   sky   is additive and has no representation in x whenever x is a
    #         point-source catalogue. Leaving it out forces the solver to
    #         manufacture that flux out of sources, which is a structural
    #         error, not a calibration one.
    # Putting them here rather than pre-dividing the data matters for the
    # CONDITIONAL prior: the network was trained on observed patches WITH
    # their sky, so `y_z` must keep it. Only the data term is corrected.
    #
    #   x-update:  x = F^-1[ (g conj(K) F(b - sky) + rho F(v))
    #                        / (g^2 |K|^2 + rho) ]
    gain = float(args.gain)
    if str(args.sky).lower() == "auto":
        from astropy.stats import sigma_clipped_stats
        sky_np = np.array([sigma_clipped_stats(b.cpu().numpy()[i, 0], sigma=3.0,
                                               maxiters=5)[1]
                           for i in range(B)], dtype=np.float64)
        print("[fwd]  --sky auto (per-patch sigma-clipped median): "
              + " ".join(f"{v:.5g}" for v in sky_np[:8])
              + (" ..." if B > 8 else ""))
    else:
        sky_np = np.full(B, float(args.sky), dtype=np.float64)
    sky = torch.from_numpy(sky_np).to(DTYPE).to(device)[:, None, None, None]
    if gain != 1.0 or float(sky_np.max()) != 0.0 or float(sky_np.min()) != 0.0:
        print(f"[fwd]  forward model b = {gain:g} * (psf * x) + sky   "
              f"(sky mean {float(sky_np.mean()):.6g}); flux_ratio is now "
              f"directly comparable to 1.0")
    K2 = gain * gain * (K.conj() * K).real
    Kb = gain * K.conj() * torch.fft.rfft2(b - sky)   # g conj(K) F(b - sky)

    # --rho-end sets the schedule by its ENDPOINT and derives the per-iteration
    # ratio, instead of --rho-scale which compounds and is entangled with
    # --iters (rho-scale 1.1 over 400 iters ends at rho = 4e15).
    if args.rho_end is not None:
        if args.rho_end <= 0 or args.rho <= 0:
            raise SystemExit("--rho and --rho-end must both be > 0")
        rho_scale = (args.rho_end / args.rho) ** (1.0 / max(args.iters - 1, 1))
        print(f"[rho]  {args.rho:g} -> {args.rho_end:g} over {args.iters} iters "
              f"(derived rho-scale {rho_scale:.6f}; --rho-scale ignored)")
    else:
        rho_scale = float(args.rho_scale)
        print(f"[rho]  {args.rho:g} * {rho_scale:g}^k -> "
              f"{args.rho * rho_scale ** (args.iters - 1):.4g} at iter {args.iters - 1}")

    print(f"[data] {args.split}: {len(idx)} patches")
    print(f"[psf]  {'gaussian %.2f' % args.psf_sigma if args.psf_sigma else args.psf_file}")
    print(f"[admm] iters={args.iters} denoise_steps={args.denoise_steps} "
          f"rho={args.rho} lam={args.lam} t_max={args.t_max}")

    # The timestep is NOT set by --t-max; it is sigma_z = sqrt(lam/rho) matched
    # against the schedule, and t_max only CLAMPS that from above. Print the
    # trajectory up front so a t that ignores --t-max is self-evidently the
    # lam/rho ratio rather than a bug.
    _rp = args.rho
    _traj, _sig = [], []
    for _k in range(args.iters):
        _s = float(np.sqrt(max(args.lam, 1e-12) / max(_rp, 1e-12)))
        if args.prior in ("flat", "cond31"):
            _t, _sm = flat_prior.t_for_sigma_z(_s, args.t_max, args.t_min)
            _sig.append(flat_prior.sigmas[_t].item())
        else:
            _t = t_for_sigma(sigmas, _s, args.t_max, args.t_min)
            _sig.append(sigmas[_t].item())
        _traj.append(_t)
        _rp *= rho_scale
    _pts = [0, args.iters // 4, args.iters // 2, 3 * args.iters // 4,
            args.iters - 1]
    print("[t]    trajectory " + "  ".join(f"k={i}:t={_traj[i]}" for i in _pts)
          + f"   | clamped at t_max for {sum(t >= args.t_max for t in _traj)} "
            f"iters, at t_min for {sum(t <= args.t_min for t in _traj)}")
    # The integer t is NOT comparable between the two priors -- cosine and
    # linear schedules put different noise at the same index. sigma(t) is.
    print("[t]    sigma(t) at those points " +
          "  ".join(f"k={i}:{_sig[i]:.4f}" for i in _pts)
          + f"   ({'linear' if args.prior == 'flat' else 'cosine'} schedule; "
            f"compare arms on THIS row, not on t)")

    # ADMM state, all in flux units
    x = b.clone()
    z = b.clone()
    u = torch.zeros_like(b)
    rho = float(args.rho)
    bn = float(torch.linalg.vector_norm(b))
    # With --input-file and no --truth-file there is no reference. Fall back to
    # peaks of the OBSERVED so `concentration` still tracks sharpening (it is a
    # ratio of a source pixel to its own neighbourhood, so it needs positions,
    # not truth values); every metric that genuinely compares against a
    # reference is skipped below.
    have_truth = truth is not None
    truth_np = truth.cpu().numpy()[:, 0] if have_truth else None
    peaks = find_peaks(truth_np if have_truth else b.cpu().numpy()[:, 0])
    if not have_truth:
        print(f"[metrics] no --truth-file: concentration is measured at "
              f"{len(peaks)} peaks of the OBSERVED; flux/rmse vs truth skipped")
    # The conditioning actually handed to the network. Identical to y_z unless
    # --drop-conditioning, which substitutes the null token the checkpoint was
    # trained to accept (p_uncond > 0).
    y_z_model = y_z
    if args.drop_conditioning:
        if args.prior != "cond":
            raise SystemExit("--drop-conditioning applies to --prior cond")
        if not has_null:
            raise SystemExit("--drop-conditioning needs a checkpoint trained "
                             "with p_uncond > 0; this one has "
                             f"{ck.get('p_uncond')!r}, so its null token is "
                             "untrained and the run would be meaningless")
        y_z_model = torch.zeros_like(y_z)
        print("[cond] --drop-conditioning: feeding the NULL token; this "
              "checkpoint is being used as an UNCONDITIONAL prior")

    hist = {"data": [], "prim": [], "dual": [], "conc": []}
    z_acc, n_acc = None, 0
    chain_trace = []

    for k in range(args.iters):
        # ---- x-update: EXACT Fourier solve (this is the whole point) ----
        rhs = Kb + rho * torch.fft.rfft2(z - u)
        x = torch.fft.irfft2(rhs / (K2 + rho), s=(H, W))

        # ---- z-update: short conditional reverse chain ----
        sigma_z = float(np.sqrt(max(args.lam, 1e-12) / max(rho, 1e-12)))
        z_prev = z
        # linear anneal eta -> eta_end over the run. Values above 1 over-noise
        # every step (see the --eta help and tweedie_chain's docstring); they
        # do NOT delete the chain's deterministic path except at its tail.
        eta_k = (args.eta if args.eta_end is None
                 else args.eta + (args.eta_end - args.eta) * k / max(args.iters - 1, 1))
        # Trace the first N iterations densely, then sample -- the interesting
        # contrast is early (iterate still blurry) vs late (iterate converged
        # and, without --renoise, far cleaner than sigma(t0) implies).
        tr = ([] if args.trace_chain and
              (k < args.trace_chain or k % max(args.log_every, 1) == 0)
              else None)
        if args.prior in ("flat", "cond31"):
            # Same chain, same sigma, same renoise/eta/clamp semantics; the
            # network is unconditional and its normalization is the flat
            # checkpoint's asinh + net-space affine instead of dataset_norm.
            z, t_used = flat_prior.denoise(
                x + u, sigma_z, args.denoise_steps,
                t_max=args.t_max, t_min=args.t_min, eta=eta_k, generator=gen,
                renoise=args.renoise, trace=tr,
                inject_scale=args.inject_scale,
                clamp=None if args.flat_clamp is None else tuple(args.flat_clamp))
        else:
            z, t_used = diffusion_denoise(
                model, x + u, y_z_model, sigma_z, args.denoise_steps, alpha_bar, sigmas,
                ideal_norm, t_max=args.t_max, t_min=args.t_min,
                guidance=args.guidance, has_null=has_null, eta=eta_k,
                generator=gen, renoise=args.renoise, trace=tr,
                inject_scale=args.inject_scale, clamp=tuple(args.clamp))
        if tr is not None:
            for rec in tr:
                chain_trace.append({"iter": k, "eta": eta_k,
                                    "sigma_z": sigma_z, **rec})
        if args.nonneg:
            z = z.clamp(min=0.0)

        # ---- sparsity prox, composed after the diffusion chain ----
        # prox_{f+g} != prox_f o prox_g in general, so this is a heuristic
        # composition, not an exact proximal step -- the standard practice when
        # plugging two priors into one splitting scheme. Threshold PER PATCH on
        # that patch's own background sigma: the patches span a wide brightness
        # range and one absolute cut would over-clean the faint ones.
        if args.sparse_tau > 0:
            zc = z.cpu().numpy()
            taus = []
            for _i in range(zc.shape[0]):
                v = zc[_i, 0]
                m, s = float(np.median(v)), float(v.std())
                for _ in range(5):                       # sigma clip
                    keep = np.abs(v - m) < 3.0 * s
                    if not keep.any():
                        break
                    m, s = float(np.median(v[keep])), float(v[keep].std())
                taus.append(m + args.sparse_tau * s)
            tau = torch.tensor(taus, dtype=z.dtype, device=z.device)[:, None, None, None]
            if args.sparse_mode == "soft":
                z = torch.sign(z) * torch.clamp(z.abs() - tau.abs(), min=0.0)
            else:
                z = torch.where(z > tau, z, torch.zeros_like(z))

        # Polyak averaging of the OUTPUT only -- the recursion below still uses
        # the instantaneous z, so the dynamics are unchanged.
        if args.avg_last > 0 and k >= args.iters - args.avg_last:
            z_acc = z.clone() if z_acc is None else z_acc + z
            n_acc += 1

        # ---- u-update ----
        u = u + x - z

        data_r = float(torch.linalg.vector_norm(
            gain * torch.fft.irfft2(torch.fft.rfft2(z) * K, s=(H, W))
            + sky - b)) / max(bn, 1e-30)
        prim = float(torch.linalg.vector_norm(x - z))
        dual = float(rho * torch.linalg.vector_norm(z - z_prev))
        c = concentration(z.cpu().numpy()[:, 0], peaks)
        hist["data"].append(data_r); hist["prim"].append(prim)
        hist["dual"].append(dual); hist["conc"].append(c)
        if k % args.log_every == 0 or k == args.iters - 1:
            fr = (f"flux/truth={float(z.sum() / truth.sum()):.3f}"
                  if have_truth else f"flux/obs={float(z.sum() / b.sum()):.3f}")
            print(f"  it {k:3d}  t={t_used:3d} sigma={sigma_z:.4f}  "
                  f"||Az-b||/||b||={data_r:.4f}  prim={prim:.3e} "
                  f"dual={dual:.3e}  conc={c:.3f}  " + fr, flush=True)
        rho *= rho_scale

    if n_acc:
        z = z_acc / n_acc
        print(f"[avg] output = mean of the final {n_acc} z-iterates")

    # ---- metrics ----
    c0 = args.crop
    crop = np.s_[c0:H - c0, c0:W - c0]
    z_np = z.cpu().numpy()
    b_np = b.cpu().numpy()
    rows = []
    for i2, kk in enumerate(idx):
        xf = z_np[i2, 0][crop]
        tf = truth_np[i2][crop] if have_truth else b_np[i2, 0][crop]
        key = "flux_ratio" if have_truth else "flux_over_obs"
        rows.append({"index": kk,
                     key: float(xf.sum() / max(tf.sum(), 1e-30)),
                     "peak_ratio": float(xf.max() / max(tf.max(), 1e-30)),
                     "rmse": (float(np.sqrt(np.mean((xf - tf) ** 2)))
                              if have_truth else float("nan"))})
    with open(args.out_dir / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader()
        w.writerows(rows)
    # float64: the flux scale spans ~6 decades after the asinh inverse, and
    # this array is what the photometry benchmarks measure. Do not truncate it
    # back to fp32 after solving in fp64.
    np.save(args.out_dir / f"{args.split}_recon.npy", z_np.astype(np.float64))
    # convergence history, so the residual behaviour can be analysed instead of
    # only eyeballed in convergence.png
    np.savez(args.out_dir / "history.npz",
             **{k: np.asarray(v, np.float64) for k, v in hist.items()})

    if chain_trace:
        with open(args.out_dir / "chain_trace.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(chain_trace[0]))
            w.writeheader(); w.writerows(chain_trace)
        # Per-step summary averaged over the traced outer iterations: this is
        # the H1-vs-H2 read. A dz0_rms profile that decays to ~0 by the end of
        # the chain means the late steps do nothing and the fix is per-step
        # noise (eta) rather than more entry noise (t_max).
        last_it = max(r["iter"] for r in chain_trace)
        tail = [r for r in chain_trace if r["iter"] == last_it]
        print(f"\n[chain] per-step movement at outer iter {last_it} "
              f"(eta={tail[0]['eta']:.2f}, t0={tail[0]['t']}):")
        print("   step    t   dz0_rms   ~mag   det_coef  noise_coef  clamp_hi%")
        for r in tail:
            dz = r["dz0_rms"]
            print(f"  {r['step']:5d} {r['t']:4d}  {dz:8.5f} "
                  f"{14.475 * dz if dz == dz else float('nan'):6.3f} "
                  f"  {r['det_coef']:8.4f}  {r['noise_coef']:9.4f}  "
                  f"{100 * r['clamp_frac_hi']:8.4f}")
        print(f"[chain] full trace -> {args.out_dir / 'chain_trace.csv'}")

    # ---- FITS, for the external photometry tools ----
    # completeness_analysis.py / final_photometry_table.py glob FITS and pair a
    # reconstruction with its truth BY FILENAME, so both trees use identical
    # basenames. Truth is written alongside so this directory is self-contained
    # and can be pointed at directly:
    #     python final_photometry_table.py --fits-dir <out-dir>/fits \
    #            --methods recon
    if args.write_fits:
        from astropy.io import fits as _fits
        trees = [("recon", z_np[:, 0])]
        if have_truth:
            trees.append(("truth", truth_np))
        for sub, arr in trees:
            d = args.out_dir / "fits" / sub
            d.mkdir(parents=True, exist_ok=True)
            # DELETE stale patches first. Without this, re-running with fewer
            # --indices leaves the previous run's extra patch_NN.fits in place
            # and the photometry tools, which just glob the directory, silently
            # pool two different configurations into one score.
            stale = sorted(d.glob("patch_*.fits"))
            for f in stale:
                f.unlink()
            if stale and len(stale) != arr.shape[0]:
                print(f"[fits] cleared {len(stale)} stale patch files in {d} "
                      f"(previous run had a different --indices)")
            for i in range(arr.shape[0]):
                # float64 (FITS BITPIX -64) for the same reason as the .npy:
                # these are what benchmark_eval.py measures fluxes from.
                _fits.writeto(d / f"patch_{i:02d}.fits",
                              arr[i].astype(np.float64), overwrite=True)
        print(f"[fits] wrote {args.out_dir}/fits/"
              f"{{{','.join(t[0] for t in trees)}}}/patch_*.fits "
              f"({B} patches each)")

    if have_truth:
        inner = np.s_[:, 16:H - 16, 16:W - 16]
        interior_rmse = float(np.sqrt(
            ((z_np[:, 0][inner] - truth_np[inner]) ** 2).mean()))
        print(f"\n[result] flux_ratio="
              f"{np.mean([r['flux_ratio'] for r in rows]):.4f}"
              f"  rmse={np.mean([r['rmse'] for r in rows]):.4e}")
        print(f"[result] concentration={concentration(z_np[:, 0], peaks):.3f}"
              f"   (truth {concentration(truth_np, peaks):.3f})")
        print(f"[result] interior rmse={interior_rmse:.4f}")
    else:
        print(f"\n[result] flux(recon)/flux(observed)="
              f"{np.mean([r['flux_over_obs'] for r in rows]):.4f}")
        print(f"[result] concentration={concentration(z_np[:, 0], peaks):.3f} "
              f"at observed peaks  (blurry input "
              f"{concentration(b_np[:, 0], peaks):.3f}; higher = sharper. "
              f"No truth available, so this is the sharpening it achieved, "
              f"NOT accuracy.)")
    print(f"[result] pixels at the zero floor: {100*(z_np <= 0).mean():.2f}%   "
          + (f"(truth {100*(truth_np <= 0).mean():.2f}% -- MATCH this, do not "
             f"minimize it; see solver_metrics.floor_frac)" if have_truth
             else "(no truth to match it against)"))
    # ---- head-to-head against a rival reconstruction (no truth needed) ----
    if cmp_np is not None:
        label = args.compare_label or os.path.basename(args.compare_file)
        if cmp_np.shape[-2:] != z_np.shape[-2:]:
            raise SystemExit(f"--compare-file is {cmp_np.shape[-2:]} but the "
                             f"solve is {z_np.shape[-2:]}; use a matching "
                             f"--region")

        def fwd_residual(arr):
            """min_g ||g*(A arr) - b|| / ||b||.

            The gain is fit per image, so an arbitrary flux calibration in
            either reconstruction cannot flatter or penalise it -- this scores
            STRUCTURE against the measured data, which is the only objective
            comparison available when neither image is a truth.
            """
            t = torch.from_numpy(np.ascontiguousarray(arr)).to(DTYPE).to(device)
            if t.dim() == 2:
                t = t[None, None]
            Ax = torch.fft.irfft2(K * torch.fft.rfft2(t), s=(H, W))
            g = float((Ax * b).sum() / torch.clamp((Ax * Ax).sum(), min=1e-30))
            return (float(torch.linalg.vector_norm(g * Ax - b))
                    / max(float(torch.linalg.vector_norm(b)), 1e-30), g)

        r_ours, g_ours = fwd_residual(z_np[:, 0])
        r_them, g_them = fwd_residual(cmp_np[:, 0])
        c_ours = concentration(z_np[:, 0], peaks)
        c_them = concentration(cmp_np[:, 0], peaks)
        c_blur = concentration(b_np[:, 0], peaks)
        print(f"\n[compare] this run  vs  {label}   "
              f"(NEITHER is a truth -- no accuracy is claimed)")
        print(f"{'':<14}{'this run':>12}{label[:16]:>18}{'blurry in':>12}")
        print(f"{'fwd residual':<14}{r_ours:12.4f}{r_them:18.4f}"
              f"{'--':>12}   lower = explains the OBSERVED data better")
        print(f"{'concentration':<14}{c_ours:12.4f}{c_them:18.4f}{c_blur:12.4f}"
              f"   higher = sharper")
        print(f"{'zero-floor %':<14}{100*(z_np<=0).mean():12.2f}"
              f"{100*(cmp_np<=0).mean():18.2f}{'--':>12}")
        print(f"{'fitted gain':<14}{g_ours:12.4g}{g_them:18.4g}{'--':>12}"
              f"   (divided out above; units are not comparable)")
        verdict = ("BETTER" if r_ours < r_them else
                   "WORSE" if r_ours > r_them else "EQUAL")
        print(f"[compare] data fidelity: this run is {verdict} "
              f"({r_ours:.4f} vs {r_them:.4f}); sharpness "
              f"{'higher' if c_ours > c_them else 'lower'} "
              f"({c_ours:.4f} vs {c_them:.4f}).")
        print("[compare] CAVEAT: forward residual rewards agreeing with the "
              "measurement, so a solver that under-deconvolves can score well "
              "on it while being blurrier. Read it WITH concentration, never "
              "alone.")

    print("[note] for a like-for-like comparison against the sampler and RED, "
          "run compare_solvers.py -- it scores every method with one metric "
          "definition. The numbers printed here use this script's own peak "
          "list and crop and are NOT comparable to RED's own printout.")

    nshow = B if args.n_show <= 0 else min(B, args.n_show)
    ncol = 2 + int(have_truth) + int(cmp_np is not None)
    fig, ax = plt.subplots(nshow, ncol, figsize=(3.7 * ncol, 3.6 * nshow),
                           squeeze=False)
    for i2 in range(nshow):
        # ONE stretch per row, from the row's background scale (row_norm).
        #
        # Neither of the two obvious alternatives works on m32. shared_norm
        # (truth) collapses, because the ideal patches are bare point sources
        # on an exactly-zero background so sig_std(truth) is 0.0 -- the asinh
        # becomes a black/white step and every panel loses its faint structure
        # at once. But the per-panel default this file used until now fails
        # the OTHER way: only the truth's width is degenerate, so it takes the
        # p99.9/100 fallback while the ADMM panel takes its own sig_std, and
        # the truth renders as a binary mask beside a recon showing real
        # texture. Both artifacts have already produced wrong written
        # conclusions. row_norm takes the median of the well-defined clipped
        # stds instead, which is the scale the panels genuinely share.
        #
        # The observed panel is excluded from that width and keeps its own
        # stretch: it is blurred and sits at a different peak.
        nrm = row_norm([z_np[i2, 0]] + ([truth_np[i2]] if have_truth else [])
                       + ([cmp_np[i2, 0] * (g_them / max(g_ours, 1e-30))]
                          if cmp_np is not None else []))
        show(ax[i2, 0], b_np[i2, 0], f"observed (idx {idx[i2]})")
        lbl = (f"flux/truth={rows[i2]['flux_ratio']:.3f}" if have_truth
               else f"flux/obs={rows[i2]['flux_over_obs']:.3f}")
        show(ax[i2, 1], z_np[i2, 0], f"ADMM+diffusion  {lbl}", nrm)
        col = 2
        if cmp_np is not None:
            # Put the rival on THIS run's scale. Both g_ours*(A z) and
            # g_them*(A c) approximate b, so g_ours*z and g_them*c share a
            # scale; showing z unscaled therefore needs c * (g_them/g_ours).
            # The inverse ratio is a 166x over-brightening on m32 and renders
            # the panel solid white -- get this the right way round. (Kept
            # even under a per-panel stretch: it makes the FITTED gains
            # comparable, which the display normalization no longer does.)
            show(ax[i2, col], cmp_np[i2, 0] * (g_them / max(g_ours, 1e-30)),
                 f"{args.compare_label or os.path.basename(args.compare_file)}"
                 f"\n(gain-matched to this run; NOT a truth)", nrm)
            col += 1
        if have_truth:
            show(ax[i2, col], truth_np[i2], "ideal (truth)", nrm)
    fig.suptitle("one asinh stretch per row, set by the row's background "
                 "scale -- observed keeps its own (it is blurred)",
                 fontsize=9, y=1.0)
    fig.tight_layout(); fig.savefig(args.out_dir / "patches.png", dpi=130)
    plt.close(fig)

    # ---- the same panels on a LINEAR stretch -------------------------------
    # asinh is the honest default for faint structure, but it compresses the
    # bright end by construction, so a peak deficit that matters photometrically
    # can be invisible in it. A linear panel clipped at --linear-pct of the
    # reference shows the bright end at face value. Neither stretch is correct
    # on its own; disagreement between them IS the diagnostic.
    def _panels(i2):
        """(reference, [(image, title), ...]) for one patch row."""
        ref = truth_np[i2] if have_truth else b_np[i2, 0]
        items = [(b_np[i2, 0], f"observed (idx {idx[i2]})")]
        lbl = (f"flux/truth={rows[i2]['flux_ratio']:.3f}" if have_truth
               else f"flux/obs={rows[i2]['flux_over_obs']:.3f}")
        items.append((z_np[i2, 0], f"ADMM+diffusion  {lbl}"))
        if cmp_np is not None:
            items.append((cmp_np[i2, 0] * (g_them / max(g_ours, 1e-30)),
                          f"{args.compare_label or os.path.basename(args.compare_file)}"
                          f"\n(gain-matched; NOT a truth)"))
        if have_truth:
            items.append((truth_np[i2], "ideal (truth)"))
        return ref, items

    def _linear_row(axrow, i2):
        ref, items = _panels(i2)
        vmax = float(np.percentile(ref, args.linear_pct))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = float(max(ref.max(), 1e-30))
        for a, (img, title) in zip(axrow, items):
            a.imshow(img, origin="lower", cmap="gray", vmin=0.0, vmax=vmax)
            a.set_title(title, fontsize=9); a.axis("off")
        return vmax

    if args.plot_linear:
        fig, ax = plt.subplots(nshow, ncol, figsize=(3.7 * ncol, 3.6 * nshow),
                               squeeze=False)
        for i2 in range(nshow):
            vmax = _linear_row(ax[i2], i2)
        fig.suptitle(f"LINEAR stretch, vmax = p{args.linear_pct:g} of the "
                     + ("truth" if have_truth else "observed")
                     + f" panel ({vmax:.4g}) -- the bright end at face value",
                     fontsize=9, y=1.0)
        fig.tight_layout()
        fig.savefig(args.out_dir / "patches_linear.png", dpi=130)
        plt.close(fig)

    if args.plot_combined:
        fig, ax = plt.subplots(2 * nshow, ncol,
                               figsize=(3.7 * ncol, 3.6 * 2 * nshow),
                               squeeze=False)
        for i2 in range(nshow):
            _ref, items = _panels(i2)
            for a, (img, title) in zip(ax[2 * i2], items):
                show(a, img, title + "  [asinh]")
            _linear_row(ax[2 * i2 + 1], i2)
            for a in ax[2 * i2 + 1]:
                a.set_title(a.get_title() + "  [linear]", fontsize=9)
        fig.suptitle("each patch twice: per-panel asinh stretch above, linear "
                     f"(p{args.linear_pct:g}, shared across the row) below",
                     fontsize=9, y=1.0)
        fig.tight_layout()
        fig.savefig(args.out_dir / "patches_combined.png", dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].semilogy(hist["data"]); ax[0].set_title("||Az-b||/||b||", fontsize=9)
    ax[1].semilogy(hist["prim"], label="primal ||x-z||")
    ax[1].semilogy(hist["dual"], label="dual"); ax[1].legend(fontsize=8)
    ax[1].set_title("ADMM residuals", fontsize=9)
    ax[2].plot(hist["conc"])
    ax[2].axhline(concentration(truth_np if have_truth else b_np[:, 0], peaks),
                  color="g", ls="--", lw=0.9,
                  label="truth" if have_truth else "blurry input")
    ax[2].legend(fontsize=8); ax[2].set_title("point-source concentration",
                                              fontsize=9)
    for a in ax:
        a.set_xlabel("ADMM iteration"); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out_dir / "convergence.png", dpi=130)
    plt.close(fig)
    extra = ("".join(f", {n}" for n, on in
                     (("patches_linear.png", args.plot_linear),
                      ("patches_combined.png", args.plot_combined)) if on))
    print(f"Wrote {args.out_dir}/patches.png ({nshow} of {B} patches), "
          f"convergence.png, metrics.csv{extra}")


if __name__ == "__main__":
    main()
