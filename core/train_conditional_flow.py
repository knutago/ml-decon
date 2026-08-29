"""
train_conditional_flow.py

Flow-matching (rectified-flow) counterpart to train_conditional_diffusion.py.
Same data, same network, same checkpoint conventions -- only the generative
parameterization changes: the net predicts a VELOCITY along a straight path
instead of the noise eps along a cosine-schedule diffusion.

What is actually different (and what is not)
--------------------------------------------
NOT different: the architecture (ConditionalFlatCNN is imported, not
re-declared, so weights are shape-identical and the two runs are a controlled
comparison), the dataset, the D4 pair augmentation, the "no trainer-side
transform" rule, `dataset_norm` passthrough, EMA, and the null-token /
identity-fixed-point tricks that the DDPM run needed.

Different:
  * PATH.  x_t = (1-t) * x0 + t * x1,  t in [0,1],  x1 = noise (or the blurry
    frame -- see --pairing).  Straight line, no cosine schedule, no 1000-bin
    discretization of t.
  * TARGET.  v = x1 - x0, the constant velocity of that line.  The net's
    output is v_theta(x_t, y, t).
  * EXTRACTION.  x0_hat = x_t - t * v_theta,  exactly, at any t.  No
    division by sqrt(alpha_bar), so the bright-end amplification that
    dps_cond_m32.py documents at high t does not arise.
  * NOISE LEVEL IS THE TIME.  Writing x_t/(1-t) = x0 + sigma*eps gives
    sigma(t) = t/(1-t) and t(sigma) = sigma/(1+sigma), in closed form.  The
    solvers' `t_for_sigma` argmin-over-a-table lookup becomes one line.

Why bother, for THIS problem
----------------------------
Be clear about what this does and does not buy, because the measured
bottlenecks here (psf_true's ~2% MTF, the confusion limit, the asinh top-end
halo, the prior's sky leak) are physics/data problems that no reparameteri-
zation touches.  The honest case for flow matching is:

  1. NFE.  The prior is not used as a generator -- it is called as a prox
     inside ADMM, 20-step chain x 200+ iterations = ~4000 network evals per
     solve.  Straight paths integrate accurately in far fewer steps, and Heun
     (2nd order, --solver heun) roughly halves again.  A 4-8 step chain at
     equal quality is a 3-5x wall-clock win, which buys back the ADMM
     iterations that dominate completeness.
  2. LOSS WEIGHTING BECOMES A KNOB.  --low-t-frac in the DDPM trainer is a
     hand-built fix for eps-MSE under-weighting the low-noise regime the prox
     actually runs in.  Here that is --loss-weight / --t-sampling, i.e. the
     principled version of the same intervention (v-loss, logit-normal t).
  3. AN EXACT ONE-STEP x0.  Useful because every consumer here ultimately
     wants E[x0 | x_t, y], not samples.

What it does NOT buy: capacity, resolution, or any escape from the forward
model.  Expect a speed and stability win, plus whatever the reweighting is
worth.  Do not expect a step change in completeness.  Measure it.

Comparing the two runs HONESTLY
-------------------------------
v-MSE and eps-MSE are different numbers on different scales; comparing the
two trainers' printed losses is meaningless.  So validation here reports a
PARAMETERIZATION-INDEPENDENT metric: build x_noisy = x0 + sigma*eps with
frozen eps on a fixed sigma grid, ask each model for its x0 estimate, and
score MSE(x0_hat, x0).  Pass --baseline-ckpt <ddpm.pt> and this script scores
the DDPM checkpoint on the IDENTICAL pack and prints both, side by side.

    --pairing noise    standard conditional rectified flow; a drop-in
                       replacement for the DDPM prior (still has a valid
                       null-token/unconditional path via --p-uncond).
    --pairing bridge   x1 = the OBSERVED frame, so the flow transports
                       blurry -> sharp directly, never starting from pure
                       noise.  This removes the hallucination pathway that
                       the spurious-source rate keeps punishing, and is the
                       maximal form of the conditioning that (per every
                       ablation here) is the thing that works.  Cost: it is
                       a transport map, not a prior -- its residual is NOT a
                       prior score, so it belongs in ADMM as an anchor/prox,
                       never inside a RED term next to ||y - Ax||^2.

Usage
-----
    cd ~/noir_ml/mycode
    # sanity first: one batch, loss -> ~0
    python train_conditional_flow.py --overfit-one-batch

    # the drop-in run, matched to the DDPM trainer's defaults
    python train_conditional_flow.py \
        --data-dir /home/alex/noir_ml/global/ml-decon/data/m31bK50 \
        --epochs 100 --batch-size 32 \
        --baseline-ckpt checkpoints_cond_diffusion_npy/best.pt

    # the deblurring-bridge experiment
    python train_conditional_flow.py --pairing bridge \
        --checkpoint-dir checkpoints_cond_flow_bridge

Downstream consumers import `flow_chain`, `flow_x0`, `t_of_sigma` and
`sigma_of_t` from here, exactly as they import `cosine_alpha_bar` from
train_conditional_diffusion.  Checkpoints carry
`"parameterization": "flow_v"` and deliberately OMIT the "diffusion" key so
that a consumer written for the eps model cannot silently treat a velocity
as a noise prediction.
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
from torch.utils.data import DataLoader

# Reused verbatim so that the flow and diffusion runs differ ONLY in the
# generative parameterization. Anything redefined here would be a confound.
from train_conditional_diffusion import (
    ConditionalFlatCNN,
    NpyPairDataset,
    cosine_alpha_bar,
    load_split,
    make_ema,
)


# ----------------------------------------------------------------------------
# Path algebra.  x_t = (1-t) x0 + t x1,  v = x1 - x0,  x0 = x_t - t v.
# ----------------------------------------------------------------------------

# t in [0,1] is fed to the (reused) sinusoidal embedding multiplied by this,
# so it spans the same numeric range the embedding's frequency ladder was
# designed around when it saw integer timesteps 0..999. Changing it changes
# the effective conditioning resolution; it is stored in the checkpoint.
T_EMBED_SCALE = 1000.0


def sigma_of_t(t):
    """Noise-to-signal level of the linear path: x_t/(1-t) = x0 + sigma*eps."""
    return t / (1.0 - t)


def t_of_sigma(sigma):
    """Inverse of sigma_of_t. Closed form -- no table, no argmin, no binning."""
    return sigma / (1.0 + sigma)


def _bcast(t, ref):
    """(B,) -> (B,1,1,1) so it broadcasts against (B,C,H,W)."""
    return t.reshape(-1, *([1] * (ref.dim() - 1)))


def flow_x0(model, x_t, y_cond, t, t_embed_scale=T_EMBED_SCALE):
    """One-step x0 estimate. Exact given a perfect v: x0 = x_t - t*v."""
    v = model(x_t, y_cond, t * t_embed_scale)
    return x_t - _bcast(t, x_t) * v, v


# ----------------------------------------------------------------------------
# Sampler / prox: integrate dx/dt = v(x, t) from t0 down to 0.
# ----------------------------------------------------------------------------

@torch.no_grad()
def flow_chain(model, z, y_cond, t0, n_steps, *, x1=None, guidance=1.0,
               null_token=0.0, has_null=False, renoise=False, generator=None,
               clamp=None, solver="heun", t_embed_scale=T_EMBED_SCALE,
               trace=None):
    """Reverse-time integration, the flow analogue of `tweedie_chain`.

    `z` is the current estimate living at t=0 (the ADMM iterate, or a clean
    patch). It is lifted onto the path at t0 and integrated back down:

        x_t0 = (1 - t0) * z + t0 * x1

    with x1 = 0 by default (the deterministic rescale that tweedie_chain uses,
    chosen there because injecting fresh noise makes D stochastic and breaks
    the RED gradient), x1 = eps when `renoise` (use when the iterate is
    CLEANER than t0 implies, which is the late-ADMM situation), or x1 = the
    observed frame for a bridge checkpoint.

    solver="heun" is a 2nd-order (midpoint-corrected) step. On a straight
    path the truncation error is what forces DDIM to take many steps; Heun
    removes most of it, so n_steps ~ 4-8 here is comparable to 20+ Euler/DDIM
    steps. That is where the wall-clock win actually comes from -- measure
    both before committing.

    `clamp` bounds the running x0 ESTIMATE (not the state), mirroring the
    clip_z guard in cond_denoiser.sample: the velocity is rebuilt from the
    clamped x0 so the trajectory stays consistent rather than being kicked
    off-path.
    """
    dt_type = z.dtype
    t0 = float(t0)
    if x1 is None:
        x1 = (torch.randn(z.shape, device=z.device, generator=generator,
                          dtype=dt_type) if renoise
              else torch.zeros_like(z))
    x = (1.0 - t0) * z + t0 * x1

    null = torch.full_like(y_cond, float(null_token))
    grid = np.linspace(t0, 0.0, max(int(n_steps), 1) + 1)

    def velocity(state, t_scalar):
        tt = torch.full((state.shape[0],), float(t_scalar),
                        device=state.device, dtype=dt_type)
        v = model(state, y_cond, tt * t_embed_scale)
        if has_null and guidance != 1.0:
            v_u = model(state, null, tt * t_embed_scale)
            v = v_u + guidance * (v - v_u)
        if clamp is not None and t_scalar > 1e-6:
            x0 = (state - t_scalar * v).clamp(clamp[0], clamp[1])
            v = (state - x0) / t_scalar
        return v

    for i in range(len(grid) - 1):
        t_cur, t_next = float(grid[i]), float(grid[i + 1])
        h = t_next - t_cur                      # negative: integrating down
        v1 = velocity(x, t_cur)
        if solver == "heun" and t_next > 1e-6:
            x_pred = x + h * v1
            v2 = velocity(x_pred, t_next)
            x = x + h * 0.5 * (v1 + v2)
        else:
            x = x + h * v1
        if trace is not None:
            trace.append(float(x.abs().mean()))
    if clamp is not None:
        x = x.clamp(clamp[0], clamp[1])
    return x


# ----------------------------------------------------------------------------
# Timestep sampling
# ----------------------------------------------------------------------------

def sample_t(bsz, device, *, scheme, logit_mean, logit_std,
             low_t_frac, low_t_max):
    """Draw the path time for each row.

    "uniform"      t ~ U(0,1). The plain rectified-flow choice.
    "logit_normal" t = sigmoid(N(m, s)) (Esser et al. 2024). Concentrates
                   samples in the middle of the path where the velocity field
                   is hardest and spends fewer on the near-trivial endpoints;
                   this is the standard recipe and generally beats uniform.

    --low-t-frac then redirects a fraction of rows into [0, low_t_max)
    regardless of scheme. It is the direct port of the DDPM trainer's low-t
    oversampling, and exists for the same reason: the ADMM/RED prox only ever
    calls the model at low noise (DDPM t<=~75-100, i.e. sigma<=0.13-0.17,
    i.e. t<=0.116-0.146 here), and uniform sampling starves that regime.
    Prefer fixing this with --loss-weight/--t-sampling if you can; keep the
    knob so the two trainers stay comparable.
    """
    if scheme == "logit_normal":
        t = torch.sigmoid(torch.randn(bsz, device=device) * logit_std
                          + logit_mean)
    else:
        t = torch.rand(bsz, device=device)
    if low_t_frac > 0:
        low = torch.rand(bsz, device=device) < low_t_frac
        t = torch.where(low, torch.rand(bsz, device=device) * low_t_max, t)
    # Keep strictly interior: t=1 gives a target that ignores x0 entirely and
    # t=0 is a zero-information row; both are measure-zero anyway.
    return t.clamp(1e-4, 1.0 - 1e-4)


def loss_weight(t, kind):
    """w(t) multiplying the squared v-error.

    The three parameterizations differ ONLY by this weight, because the
    errors are proportional:
        x0_hat - x0   = -t       * (v_hat - v)
        eps_hat - eps = (1 - t)  * (v_hat - v)
    so w=t^2 is x0-space MSE, w=(1-t)^2 is eps-space MSE (what the DDPM
    trainer minimizes, up to its schedule), and w=1 is the plain
    rectified-flow objective. "snr" = t^2/(1-t)^2 * ... is deliberately not
    offered; it diverges at the endpoints.

    Default is "v". Switch to "eps" if you want the closest possible match to
    the DDPM run's emphasis when attributing a difference to the path rather
    than the weighting.
    """
    if kind == "v":
        return torch.ones_like(t)
    if kind == "x0":
        return t ** 2
    if kind == "eps":
        return (1.0 - t) ** 2
    raise ValueError(f"unknown loss weight {kind!r}")


# ----------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------

def flow_loss(model, sharp, blurry, device, *, pairing="noise",
              t_scheme="logit_normal", logit_mean=0.0, logit_std=1.0,
              low_t_frac=0.0, low_t_max=0.15, weight_kind="v",
              p_uncond=0.0, null_token=0.0, identity_frac=0.0,
              identity_t_max=0.15, bridge_sigma=0.0,
              t_embed_scale=T_EMBED_SCALE):
    """Conditional flow-matching loss, one batch.

    pairing="noise":   x1 ~ N(0, I).   Prior-like; the null-token path is a
                       genuine unconditional model when p_uncond > 0.
    pairing="bridge":  x1 = the observed (blurry) frame. The learned field
                       transports the observed distribution onto the ideal
                       one. There is no noise anywhere unless --bridge-sigma
                       > 0, so nothing can be hallucinated out of a random
                       draw -- the model can only move what the data shows.

    IDENTITY ROWS (--identity-frac) carry over from the DDPM trainer, where
    they were needed because the RED gradient x - D(x) must vanish on the
    truth. Here they fall out of the same code with a substitution:
        noise pairing:  x1 <- 0      => x_t = (1-t) x0, target v = -x0,
                                        hence x0_hat = x_t - t*v = x0 exactly.
        bridge pairing: x1 <- x0     => x_t = x0 for all t, target v = 0,
                                        i.e. "already sharp: do not move".
    Both say D(sharp) = sharp, which is the property that matters.

    --bridge-sigma adds gamma(t)*z with gamma(t) = s*sqrt(t(1-t)) to the
    bridge path (a stochastic interpolant), which widens the transported
    distribution so the map is not degenerate. The velocity target picks up
    gamma'(t)*z accordingly; gamma' diverges at the endpoints, so those rows
    are held away from t in {0,1}. Off by default.
    """
    bsz = sharp.shape[0]
    t = sample_t(bsz, device, scheme=t_scheme, logit_mean=logit_mean,
                 logit_std=logit_std, low_t_frac=low_t_frac,
                 low_t_max=low_t_max)

    if pairing == "bridge":
        x1 = blurry
    else:
        x1 = torch.randn_like(sharp)

    if identity_frac > 0:
        idm = torch.rand(bsz, device=device) < identity_frac
        id_x1 = sharp if pairing == "bridge" else torch.zeros_like(sharp)
        x1 = torch.where(_bcast(idm.float(), sharp) > 0.5, id_x1, x1)
        # Identity rows are re-placed at low t for the same reason as in the
        # DDPM trainer: a zero-noise row at t=0.9 is a near-black frame that
        # teaches nothing about the regime the prox runs in.
        t_id = torch.rand(bsz, device=device) * identity_t_max
        t = torch.where(idm, t_id.clamp(1e-4, 1.0 - 1e-4), t)
    else:
        idm = torch.zeros(bsz, dtype=torch.bool, device=device)

    tb = _bcast(t, sharp)
    x_t = (1.0 - tb) * sharp + tb * x1
    v_target = x1 - sharp

    if pairing == "bridge" and bridge_sigma > 0:
        t_s = t.clamp(1e-3, 1.0 - 1e-3)
        ts = _bcast(t_s, sharp)
        z = torch.randn_like(sharp)
        gamma = bridge_sigma * (ts * (1.0 - ts)).sqrt()
        dgamma = bridge_sigma * (1.0 - 2.0 * ts) / (2.0 * (ts * (1.0 - ts)).sqrt())
        keep = _bcast((~idm).float(), sharp)          # identity rows stay clean
        x_t = x_t + keep * gamma * z
        v_target = v_target + keep * dgamma * z

    y_cond = blurry
    if p_uncond > 0:
        drop = torch.rand(bsz, device=device) < p_uncond
        y_cond = torch.where(_bcast(drop.float(), blurry) > 0.5,
                             torch.full_like(blurry, float(null_token)),
                             y_cond)

    v_pred = model(x_t, y_cond, t * t_embed_scale)
    w = _bcast(loss_weight(t, weight_kind), sharp)
    return (w * (v_pred - v_target) ** 2).mean()


# ----------------------------------------------------------------------------
# Parameterization-independent validation
# ----------------------------------------------------------------------------

# The regimes that matter: the first four bracket where the ADMM/RED prox
# actually calls the denoiser (DDPM t = 5, 20, 50, 100 on the cosine
# schedule), the last two are the generative regime.
EVAL_SIGMAS = (0.018, 0.043, 0.091, 0.171, 0.337, 1.016)


def build_eval_pack(loader, device, n_batches=4, seed=0):
    """Fixed patches + frozen noise, shared by every model being compared."""
    sharp, blurry = [], []
    for i, (s, b) in enumerate(loader):
        if i >= n_batches:
            break
        sharp.append(s)
        blurry.append(b)
    sharp = torch.cat(sharp).to(device)
    blurry = torch.cat(blurry).to(device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    eps = torch.randn(sharp.shape, generator=gen).to(device)
    return sharp, blurry, eps


@torch.no_grad()
def x0_mse_flow(model, pack, sigmas=EVAL_SIGMAS,
                t_embed_scale=T_EMBED_SCALE):
    """MSE of the one-step x0 estimate at each sigma, flow parameterization.

    The noisy input is constructed in the SHARED convention
    x_noisy = x0 + sigma*eps, then mapped onto this model's own path
    (x_t = x_noisy/(1+sigma), since 1-t = 1/(1+sigma)). Identical eps and
    identical sigma as the DDPM scorer below, so the numbers are comparable.
    """
    sharp, blurry, eps = pack
    model.eval()
    out = []
    for sigma in sigmas:
        t = t_of_sigma(sigma)
        x_t = (sharp + sigma * eps) * (1.0 - t)
        tt = torch.full((sharp.shape[0],), t, device=sharp.device)
        x0_hat, _ = flow_x0(model, x_t, blurry, tt, t_embed_scale)
        out.append(F.mse_loss(x0_hat, sharp).item())
    model.train()
    return out


@torch.no_grad()
def x0_mse_ddpm(model, alpha_bar, pack, sigmas=EVAL_SIGMAS):
    """Same metric for an eps-prediction DDPM checkpoint (the baseline)."""
    sharp, blurry, eps = pack
    sig_table = ((1 - alpha_bar) / alpha_bar.clamp_min(1e-12)).sqrt()
    model.eval()
    out = []
    for sigma in sigmas:
        t = int(torch.argmin((sig_table - sigma).abs()).item())
        ab = alpha_bar[t]
        x_t = ab.sqrt() * (sharp + sigma * eps)
        tt = torch.full((sharp.shape[0],), t, device=sharp.device,
                        dtype=torch.long)
        eps_hat = model(x_t, blurry, tt)
        x0_hat = (x_t - (1 - ab).sqrt() * eps_hat) / ab.sqrt()
        out.append(F.mse_loss(x0_hat, sharp).item())
    return out


@torch.no_grad()
def bridge_transport_mse(model, pack, n_steps=8, solver="heun",
                         t_embed_scale=T_EMBED_SCALE):
    """Bridge-only: integrate observed -> sharp and score the endpoint.

    This is the number a bridge checkpoint should be judged on; its x0-MSE at
    injected Gaussian sigma is off-distribution (it never saw Gaussian noise).
    """
    sharp, blurry, _ = pack
    model.eval()
    # Start the state AT the observed frame: with x1 = blurry and t0 -> 1,
    # x_t0 = (1-t0)*z + t0*x1 -> blurry regardless of z.
    x = flow_chain(model, torch.zeros_like(blurry), blurry, 1.0 - 1e-4,
                   n_steps, x1=blurry, solver=solver,
                   t_embed_scale=t_embed_scale)
    model.train()
    return F.mse_loss(x, sharp).item(), F.mse_loss(blurry, sharp).item()


def load_baseline(ckpt_path, device):
    """Rebuild a train_conditional_diffusion.py checkpoint for scoring."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ck.get("parameterization") == "flow_v":
        sys.exit(f"[baseline] {ckpt_path} is a FLOW checkpoint, not a DDPM one")
    arch = ck["arch"]
    m = ConditionalFlatCNN(channels=arch["channels"],
                           t_dim=arch.get("t_dim", 128),
                           norm=arch.get("norm", "group")).to(device)
    state = ck.get("ema_state") or ck.get("model_state") or ck.get("model")
    m.load_state_dict(state)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    T = int(ck.get("diffusion", {}).get("timesteps", 1000))
    return m, cosine_alpha_bar(T).to(device)


# ----------------------------------------------------------------------------
# Checkpointing
# ----------------------------------------------------------------------------

def save_checkpoint(path, model, ema, ema_is_lib, optimizer, epoch,
                    dataset_norm, args, best_val):
    ckpt = {
        "epoch": epoch,
        "best_val": best_val,
        "model_state": model.state_dict(),
        "ema_state": (ema.ema_model.state_dict() if ema_is_lib
                      else ema.state_dict()),
        "optimizer_state": optimizer.state_dict(),
        # Same contract as the DDPM trainer: no trainer-side transform exists,
        # the model lives in the dataset's stored domain, and dataset_norm is
        # the only map back to physical flux.
        "dataset_norm": dataset_norm,
        "data_dir": str(args.data_dir),
        "arch": {"channels": args.channels,
                 "dilations": list(ConditionalFlatCNN.DILATIONS),
                 "t_dim": 128, "in_channels": 2, "norm": args.norm},
        # The output of this network is a VELOCITY, not eps. Any consumer
        # written against the DDPM checkpoints must refuse to load this; the
        # "diffusion" key is deliberately absent so a KeyError fires instead
        # of a silent misinterpretation.
        "parameterization": "flow_v",
        "flow": {"path": "linear",
                 "pairing": args.pairing,
                 "t_embed_scale": args.t_embed_scale,
                 "sigma_of_t": "t/(1-t)",
                 "x0_from_v": "x_t - t*v",
                 "bridge_sigma": args.bridge_sigma},
        "p_uncond": args.p_uncond,
        "null_token": args.null_token,
        "identity_frac": args.identity_frac,
        "identity_t_max": args.identity_t_max,
        "low_t_frac": args.low_t_frac,
        "low_t_max": args.low_t_max,
        "loss_weight": args.loss_weight,
        "t_sampling": args.t_sampling,
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
    }
    torch.save(ckpt, path)


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Train a conditional flow-matching model on observed/ideal "
                    ".npy patch pairs. Architecture and data handling are "
                    "identical to train_conditional_diffusion.py; only the "
                    "generative parameterization differs.")
    ap.add_argument("--data-dir", type=Path,
                    default=Path("/home/alex/noir_ml/global/ml-decon/data/m31bK50"))
    ap.add_argument("--checkpoint-dir", type=Path,
                    default=Path("checkpoints_cond_flow"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--norm", choices=["group", "none"], default="group",
                    help="normalization in FiLMConvBlock. Same flag, same "
                         "caveat as the DDPM trainer: 'group' divides out "
                         "absolute scale at the first block and makes "
                         "behaviour depend on H,W. Keep it matched to "
                         "whatever the baseline checkpoint used, or the "
                         "flow-vs-diffusion comparison is confounded.")

    ap.add_argument("--pairing", choices=["noise", "bridge"], default="noise",
                    help="'noise': x1 ~ N(0,I), a conditional prior and a "
                         "drop-in replacement for the DDPM checkpoint. "
                         "'bridge': x1 = the observed frame, so the flow maps "
                         "blurry -> sharp with no noise in the path at all. "
                         "The bridge cannot hallucinate from a random draw, "
                         "but its residual is NOT a prior score -- use it as "
                         "an ADMM anchor/prox, never inside a RED term.")
    ap.add_argument("--bridge-sigma", type=float, default=0.0,
                    help="stochastic-interpolant width for --pairing bridge: "
                         "adds s*sqrt(t(1-t))*z to the path (and the matching "
                         "s*(1-2t)/(2 sqrt(t(1-t)))*z to the target). 0 = a "
                         "purely deterministic transport map.")

    ap.add_argument("--t-sampling", choices=["uniform", "logit_normal"],
                    default="logit_normal",
                    help="path-time distribution. logit_normal (SD3-style) "
                         "concentrates training in the middle of the path "
                         "and is the standard recipe.")
    ap.add_argument("--logit-mean", type=float, default=0.0)
    ap.add_argument("--logit-std", type=float, default=1.0)
    ap.add_argument("--loss-weight", choices=["v", "x0", "eps"], default="v",
                    help="weight on the squared velocity error: 'v'=1 (plain "
                         "rectified flow), 'x0'=t^2 (x0-space MSE), "
                         "'eps'=(1-t)^2 (matches the DDPM trainer's "
                         "emphasis). Use 'eps' if you want to attribute a "
                         "difference to the PATH rather than the weighting.")
    ap.add_argument("--low-t-frac", type=float, default=0.25,
                    help="fraction of rows redirected into [0, --low-t-max). "
                         "Port of the DDPM trainer's low-t oversampling; the "
                         "prox only ever runs at low noise. Default matches "
                         "that run so the comparison is controlled -- try 0 "
                         "once --t-sampling/--loss-weight are tuned.")
    ap.add_argument("--low-t-max", type=float, default=0.15,
                    help="t is the noise level here: sigma = t/(1-t). 0.15 "
                         "<-> sigma 0.176 <-> DDPM cosine t~103, matching the "
                         "DDPM trainer's --low-t-max 100.")
    ap.add_argument("--identity-frac", type=float, default=0.15,
                    help="fraction of rows with a degenerate path (x1 <- 0 "
                         "for noise pairing, x1 <- sharp for bridge), which "
                         "makes the one-step x0 estimate exactly the identity "
                         "on a clean sharp field. Same purpose as in the DDPM "
                         "trainer: the RED/ADMM residual must vanish on the "
                         "truth.")
    ap.add_argument("--identity-t-max", type=float, default=0.15)
    ap.add_argument("--p-uncond", type=float, default=0.15,
                    help="classifier-free dropout: fraction of rows whose "
                         "conditioning channel is replaced by the null token, "
                         "making model(x_t, null, t) a genuine unconditional "
                         "velocity. Largely pointless for --pairing bridge, "
                         "where the observed frame also enters through x_t.")
    ap.add_argument("--null-token", type=float, default=0.0,
                    help="value of the null conditioning channel. 0.0 is "
                         "correct in the asinh domain, where real patches are "
                         "far from zero. It is WRONG in the linear domain "
                         "(background sits at ~2e-4, so a zeros channel is "
                         "just a faint conditional one) -- use -1.0 there, "
                         "matching admm_diffusion_linear.py.")
    ap.add_argument("--t-embed-scale", type=float, default=T_EMBED_SCALE,
                    help="t in [0,1] is multiplied by this before the "
                         "sinusoidal embedding, so it spans the range that "
                         "embedding saw as integer DDPM timesteps.")

    ap.add_argument("--eval-batches", type=int, default=4,
                    help="val batches in the frozen comparison pack.")
    ap.add_argument("--baseline-ckpt", type=Path, default=None,
                    help="a train_conditional_diffusion.py checkpoint to "
                         "score on the IDENTICAL frozen pack. This is the "
                         "only fair flow-vs-diffusion comparison: the two "
                         "training losses live on different scales and must "
                         "never be compared directly.")
    ap.add_argument("--solver", choices=["heun", "euler"], default="heun")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--overfit-one-batch", action="store_true")
    ap.add_argument("--sanity-steps", type=int, default=3000)
    ap.add_argument("--sanity-lr", type=float, default=3e-4)
    ap.add_argument("--limit-train", type=int, default=0,
                    help="debug only: truncate the training split.")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    print(f"[setup] device = {device}"
          + ("  (CPU: fine for --overfit-one-batch; use the cluster for the "
             "full run)" if device.type == "cpu" else ""))

    if args.pairing == "bridge" and args.p_uncond > 0:
        print("[setup] NOTE: --p-uncond with --pairing bridge only masks the "
              "conditioning CHANNEL; the observed frame still enters through "
              "the path itself, so the null-token branch is not an "
              "unconditional model. Guidance on a bridge is not meaningful.")

    # --- data (identical handling to the DDPM trainer) --------------------
    tr_obs, tr_ideal = load_split(args.data_dir, "train")
    if args.limit_train:
        tr_obs, tr_ideal = tr_obs[:args.limit_train], tr_ideal[:args.limit_train]
    train_ds = NpyPairDataset(tr_obs, tr_ideal, augment=not args.no_augment)
    val_ds = NpyPairDataset(*load_split(args.data_dir, "val"), augment=False)

    norm_path = args.data_dir / "norm.json"
    if norm_path.exists():
        dataset_norm = json.loads(norm_path.read_text())
        print(f"[data] dataset norm ({norm_path}): {dataset_norm}")
    else:
        dataset_norm = None
        print(f"[data] WARNING: no norm.json in {args.data_dir}; checkpoints "
              f"will not carry the normalized->physical map.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"),
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    # --- model / optim ----------------------------------------------------
    model = ConditionalFlatCNN(channels=args.channels, norm=args.norm).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] ConditionalFlatCNN (velocity head), {n_params/1e6:.2f}M "
          f"params, dilations {ConditionalFlatCNN.DILATIONS}, norm={args.norm}")
    print(f"[flow]  path=linear  pairing={args.pairing}  "
          f"t~{args.t_sampling}  weight={args.loss_weight}  "
          f"sigma(t)=t/(1-t)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-5)
    ema, ema_is_lib = make_ema(model)

    start_epoch, best_val = 1, float("inf")
    if args.resume is not None:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        if ck.get("parameterization") != "flow_v":
            sys.exit(f"[resume] {args.resume} is not a flow checkpoint")
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        if ema_is_lib:
            ema.ema_model.load_state_dict(ck["ema_state"])
        else:
            ema.shadow = {k: v.to(device) for k, v in ck["ema_state"].items()}
        start_epoch = int(ck["epoch"]) + 1
        best_val = float(ck.get("best_val", float("inf")))
        print(f"[resume] from {args.resume} at epoch {start_epoch}, "
              f"best {best_val:.6f}")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if dataset_norm is not None:
        with open(args.checkpoint_dir / "dataset_norm.json", "w") as f:
            json.dump(dataset_norm, f, indent=2)

    loss_kwargs = dict(
        pairing=args.pairing, t_scheme=args.t_sampling,
        logit_mean=args.logit_mean, logit_std=args.logit_std,
        low_t_frac=args.low_t_frac, low_t_max=args.low_t_max,
        weight_kind=args.loss_weight, p_uncond=args.p_uncond,
        null_token=args.null_token, identity_frac=args.identity_frac,
        identity_t_max=args.identity_t_max, bridge_sigma=args.bridge_sigma,
        t_embed_scale=args.t_embed_scale)

    # --- overfit-one-batch sanity check -----------------------------------
    if args.overfit_one_batch:
        sharp, blurry = next(iter(train_loader))
        sharp, blurry = sharp.to(device), blurry.to(device)
        pack = (sharp, blurry,
                torch.randn(sharp.shape,
                            generator=torch.Generator(device="cpu"
                                                      ).manual_seed(0)).to(device))
        sanity_opt = torch.optim.AdamW(model.parameters(), lr=args.sanity_lr)
        print(f"[sanity] memorizing one batch of {sharp.shape[0]} pairs for "
              f"{args.sanity_steps} steps (lr {args.sanity_lr}).")
        print("[sanity] Watch the x0-MSE row, not the train loss: the train "
              "loss resamples t every step so it bounces. x0-MSE should fall "
              "monotonically, and the LOW-sigma entries are the ones the prox "
              "depends on.")
        model.train()
        running = None
        for step in range(1, args.sanity_steps + 1):
            sanity_opt.zero_grad()
            loss = flow_loss(model, sharp, blurry, device, **loss_kwargs)
            loss.backward()
            sanity_opt.step()
            running = (loss.item() if running is None
                       else 0.98 * running + 0.02 * loss.item())
            if step % 100 == 0:
                mses = x0_mse_flow(model, pack,
                                   t_embed_scale=args.t_embed_scale)
                cells = "  ".join(f"s={s:.3f}:{m:.2e}"
                                  for s, m in zip(EVAL_SIGMAS, mses))
                print(f"  step {step:5d}  train(avg) {running:.5f}  "
                      f"x0-MSE {np.mean(mses):.3e}   [{cells}]")
        final = float(np.mean(x0_mse_flow(model, pack,
                                          t_embed_scale=args.t_embed_scale)))
        var = float(sharp.var())
        print(f"[sanity] final mean x0-MSE {final:.3e} vs patch variance "
              f"{var:.3e}  (ratio {final/max(var,1e-12):.3e}). Ratio well "
              f"below 1e-2 = healthy memorization.")
        return

    # --- frozen comparison pack + optional baseline -----------------------
    pack = build_eval_pack(val_loader, device, n_batches=args.eval_batches)
    print(f"[eval] frozen pack: {pack[0].shape[0]} val patches, sigmas "
          f"{list(EVAL_SIGMAS)}")
    if args.baseline_ckpt is not None:
        base, base_ab = load_baseline(args.baseline_ckpt, device)
        base_mse = x0_mse_ddpm(base, base_ab, pack)
        print(f"[eval] BASELINE {args.baseline_ckpt} x0-MSE  "
              + "  ".join(f"s={s:.3f}:{m:.3e}"
                          for s, m in zip(EVAL_SIGMAS, base_mse))
              + f"   mean {np.mean(base_mse):.3e}")
        del base
    else:
        base_mse = None

    # --- training loop ----------------------------------------------------
    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        running, n_seen = 0.0, 0
        for sharp, blurry in train_loader:
            sharp, blurry = sharp.to(device), blurry.to(device)
            optimizer.zero_grad()
            loss = flow_loss(model, sharp, blurry, device, **loss_kwargs)
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

        mses = x0_mse_flow(model, pack, t_embed_scale=args.t_embed_scale)
        val_metric = float(np.mean(mses))
        dt = time.time() - t0
        cells = "  ".join(f"{m:.2e}" for m in mses)
        line = (f"[epoch {epoch:3d}/{args.epochs}] train(v) {train_loss:.5f}  "
                f"x0-MSE {val_metric:.3e}  [{cells}]  ({dt:.1f}s)")
        if base_mse is not None:
            line += f"  vs baseline {np.mean(base_mse):.3e}"
        print(line)
        if args.pairing == "bridge":
            got, start = bridge_transport_mse(
                model, pack, solver=args.solver,
                t_embed_scale=args.t_embed_scale)
            print(f"           bridge transport MSE {got:.3e} "
                  f"(observed-vs-sharp {start:.3e}, so the map must beat that)")

        save_checkpoint(args.checkpoint_dir / "last.pt", model, ema,
                        ema_is_lib, optimizer, epoch, dataset_norm, args,
                        best_val)
        if val_metric < best_val:
            best_val = val_metric
            save_checkpoint(args.checkpoint_dir / "best.pt", model, ema,
                            ema_is_lib, optimizer, epoch, dataset_norm, args,
                            best_val)
            print("          -> new best x0-MSE, saved best.pt")

    print(f"[done] best mean x0-MSE {best_val:.3e}. Selection is on the "
          f"x0 metric, NOT the velocity loss, so best.pt is comparable across "
          f"parameterizations. Use 'ema_state'; the model works in the "
          f"dataset's stored domain and 'dataset_norm' is the only map back "
          f"to physical flux.")


# Mandatory guard: never retrain on import (consumers import flow_chain /
# flow_x0 / t_of_sigma from this module).
if __name__ == "__main__":
    main()
