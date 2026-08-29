"""
dps_cond_m32.py

Diffusion Posterior Sampling (Chung et al. 2022, arXiv:2209.14687) with the
CONDITIONAL m32 checkpoint, on the same data, operator and output convention as
admm_diffusion_deconvolve.py -- so the two can be scored side by side.

Why guidance is an ADDITION, not a CFG-style subtraction
--------------------------------------------------------
Classifier-free guidance forms eps_c + w(eps_c - eps_u) because BOTH terms are
learned scores and the difference isolates the direction conditioning added. A
measurement needs no such trick: Bayes gives the term additively,

    grad log p(x_t | y)  =  grad log p(x_t)  +  grad log p(y | x_t)
                            [ the network ]      [ the operator ]

and for Gaussian noise the second term is A^T(y - Ax)/sigma_y^2. There is
nothing to subtract; the "unconditional model" of the CFG analogy is just the
score network itself. What DOES carry over is the guidance SCALE: Bayes fixes
the likelihood weight at 1/sigma_y^2, but nothing stops us over-weighting it.
That knob is --zeta.

p(y | x_t) is intractable for t > 0 -- y was measured on the clean x_0, not on
the noisy x_t -- so DPS substitutes the Tweedie estimate, p(y|x_t) ~ p(y|x0(x_t)),
and differentiates THROUGH the network. That Jacobian is the whole method and
the reason this is not just PnP with extra steps.

This script stacks that on the conditional prior, which is the combination the
repo has not measured: every unconditional prior tried in the ADMM lost to no
prior at all (see --drop-conditioning there), so the guidance mechanism is only
interesting on top of the model that actually works on this field.

    for t in t_start .. 0:
        eps    = model(z_t, y_z, t)                     [+ CFG if --guidance]
        z0     = (z_t - sqrt(1-abar_t) eps) / sqrt(abar_t)      (Tweedie)
        x      = ideal_norm.inverse(clamp(z0))                  (flux seam)
        r      = gain * (psf * x) + sky - b
        z_{t-1} = DDIM(z_t, z0, eps) - zeta * grad_{z_t} ||r|| / ||b||

Three details that are specific to this problem, not to DPS
-----------------------------------------------------------
1. The residual is normalized PER PATCH, by that patch's own ||b||. The eight
   val patches span a wide brightness range (indices 0:2 sit 18x below the
   field mean); one pooled norm would hand essentially all the guidance to the
   brightest patch, because DPS's 1/||r|| factor would then be global. Summing
   per-patch norms and taking a single autograd.grad gives each patch its own
   1/||r_i|| -- the patches are independent in the graph, so this is exact.
   It also makes --zeta dimensionless and makes the logged residual directly
   comparable to the ADMM's ||Az-b||/||b|| column.

2. The x0 clamp is STRAIGHT-THROUGH. The flux seam is beta*sinh(z*span) with
   span 6.47, so an unclamped z0 of 2 overflows float32 and z0 of 5 overflows
   float64. A hard clamp would be safe but would ZERO the guidance gradient on
   every out-of-range pixel, which early in the chain is most of them, leaving
   the likelihood unable to pull them back. Forward value clamped, gradient
   passed through; the cosh in the backward pass is evaluated at the clamped
   value, so it stays bounded.

3. --step-cap is a trust region on the guidance step, and it is NOT cosmetic:
   without it this problem diverges in the first step. Two factors multiply at
   high t. The Tweedie estimate divides by sqrt(abar_t), which on the cosine
   schedule is ~3e-3 at t=999, so the Jacobian dz0/dz_t carries a factor ~300;
   and the flux seam's derivative is beta*span*cosh(z*span), which at the
   clamp ceiling z=1 is ~2.1e3. The product overwhelms z (measured: z_rms
   19776 after one step, against a target z whose 99th percentile is 0.077).
   The cap bounds the step's per-patch L2 norm to --step-cap times the
   iterate's own, leaving the DPS DIRECTION untouched and only its length
   bounded. The log prints how often it binds; if it binds for the whole run
   then zeta is doing nothing and only the cap is setting the step.

4. --split-norms, same as the ADMM. The network emits z in the CHECKPOINT's
   ideal domain (beta=1.0, span 6.472); the stored .npy patches are encoded
   with the data dir's (beta=0.0175, span 10.52). Flux is the only common
   ground. Getting this wrong is silent -- both z arrays look like [0,1].

The residual is NOT the thing to minimize. Pushing the truth itself through
this operator leaves 0.78 on these patches; anything far below that is fitting
forward-model error. Read |resid - 0.78|, and check it against the accuracy
columns rather than trusting it.

Usage (matches admm_sweep_m32.py's data/operator settings exactly):

  python dps_cond_m32.py \
      --ckpt /home/alex/noir_ml/global/ml-decon/checkpoints_cond_diffusion_npy_m32/best.pt \
      --data-dir /home/alex/noir_ml/global/ml-decon/data/m32_klong_nosky \
      --split val --indices 94,282,470,658,846,1034,1222,1410 \
      --psf-file psf_klong_epsf.fits --split-norms \
      --steps 300 --zeta 1.0 --out-dir dps_m32/z1.0

Then score it against the ADMM sweep with:

  python score_recon.py dps_m32/* sweep_admm/ds4_eta1_rho0.03
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from red_pnp_deconvolve import (TorchNorm, AsinhNorm, load_checkpoint,
                                load_kernel, gaussian_kernel, make_otf,
                                parse_indices, show, shared_norm, sig_std)
from train_conditional_diffusion import cosine_alpha_bar


# ----------------------------------------------------------------------------
def panel_norm(img):
    """A per-panel asinh stretch that survives an all-zero background.

    shared_norm(truth) is unusable on THIS data and silently so. It sets
    linear_width = max(sig_std(ref), 1e-30), and the m32 truth is a
    point-source catalogue whose background is EXACTLY zero -- sig_std is
    0.0 on every patch, so the width collapses to 1e-30 and asinh degenerates
    into a step function at zero. Every panel then renders as pure black and
    white and shows only "is this pixel nonzero", which reads as a filled or
    broken reconstruction regardless of what the values actually are.

    So: fall back to a fraction of the bright end when the clipped std is
    degenerate, and cut vmax at p99.9 so one saturated star does not flatten
    everything else to black.

    The cost, which shared_norm's docstring is right about, is that panels are
    no longer strictly comparable by eye -- a faint wing can render at
    different greys in two panels holding near-identical data. Use
    --shared-stretch when comparing panels quantitatively; do not use it to
    judge whether a reconstruction is structurally wrong.
    """
    a = np.asarray(img, dtype=np.float64)
    w = sig_std(a)
    hi = float(np.percentile(a, 99.9))
    if not np.isfinite(w) or w <= 0:
        w = max(hi, 1e-30) / 100.0
    return AsinhNorm(linear_width=w, vmin=float(a.min()),
                     vmax=max(hi, float(a.min()) + 1e-30))


def straight_through_clamp(z, lo, hi):
    """Forward: z.clamp(lo, hi). Backward: identity. See docstring note 2."""
    return z + (z.clamp(lo, hi) - z).detach()


# ----------------------------------------------------------------------------
# Data-fidelity terms. Both return a PER-PATCH dimensionless misfit on the same
# scale, so --zeta and --step-cap mean the same thing under either.
# ----------------------------------------------------------------------------
def l2_rel(Ax, b, bn, _c, _tau):
    """||Ax - b|| / ||b||, per patch. The classic DPS term."""
    return torch.linalg.vector_norm((Ax - b).reshape(Ax.shape[0], -1), dim=1) / bn


def poisson_rel(Ax, b, bn, c, tau):
    """Per-patch Poisson misfit, offset-shifted and quadratically extended.

    Why bother. The L2 term weights every pixel equally, so its gradient is
    set by the pixels with the largest ABSOLUTE residual -- the bright stars.
    A faint source sitting 0.01 above sky contributes ~1e-4 to a sum whose
    bright pixels contribute ~1. Poisson replaces that with an inverse-variance
    weight: near its minimum the deviance below is sum (Ax-b)^2/(b+c), i.e.
    weight 1/(b+c), which is ~(b_bright/c) times heavier on empty sky. c sets
    how much heavier, and c -> inf recovers L2 exactly.

    Two things make the textbook form unusable here and both are fixed:

      offset  This data is sky-subtracted: 8% of pixels are negative (min
              -1.9e-3) and -y*log(mu) is not defined for y < 0. So the loss is
              built on shifted variables y = b + c, mu = Ax + c with c > 0
              large enough that y > 0 everywhere. This is not a fudge -- c is
              the variance floor the Poisson model is missing, the counts
              offset that sky subtraction removed.

      piecewise  f(mu) = mu - y log mu has f'(mu) = 1 - y/mu, which diverges as
              mu -> 0, and mu DOES reach 0 here: the x0 clamp floor maps to
              flux 0 exactly, so early in the chain most pixels sit there. So
              f is replaced below mu = tau by its own second-order Taylor
              expansion about tau. That is C^2 at the seam, has a LINEAR (hence
              bounded) gradient below it, and is finite for mu <= 0 too, which
              the PSF's small negative lobes need.

    Returned as sqrt(deviance * mean(y)) / ||b|| so it reduces EXACTLY to
    l2_rel when y is constant -- that is what lets --zeta carry over between
    the two losses instead of needing a fresh sweep.
    """
    B = Ax.shape[0]
    y = b + c                                  # > 0 by construction of c
    mu = Ax + c
    d = mu - tau
    # clamp_min keeps the unselected branch finite: torch.where propagates NaN
    # through the branch it did NOT take, so log(mu) must never see mu <= 0.
    f_hi = mu.clamp_min(tau) - y * torch.log(mu.clamp_min(tau))
    f_lo = (tau - y * torch.log(tau)) + (1 - y / tau) * d + (y / (2 * tau ** 2)) * d * d
    f = torch.where(mu < tau, f_lo, f_hi)
    # deviance: 2*(f(mu) - f(y)), zero exactly at Ax = b. The Taylor extension
    # is a lower bound on f, so clamp the tiny negative values it can produce.
    dev = (2.0 * (f - (y - y * torch.log(y)))).clamp_min(0.0)
    s = y.reshape(B, -1).mean(dim=1)           # per-patch scale, mean(y)
    S = dev.reshape(B, -1).sum(dim=1) * s
    return S.clamp_min(1e-30).sqrt() / bn


DATA_LOSS = {"l2": l2_rel, "poisson": poisson_rel}


def dps_sample(model, alpha_bar, ideal_norm, y_z, b, K, gain, sky, *,
               zeta, steps, t_start, eta, guidance, has_null, clamp,
               misfit=l2_rel, pois_c=None, pois_tau=None,
               step_cap=0.1, init_z=None, generator=None, log_every=25,
               verbose=True):
    """One DPS posterior draw for the whole batch. Returns flux (B,1,H,W) f64.

    y_z : conditioning in the checkpoint's OBSERVED domain, network dtype.
    b   : measurement in FLUX units, float64.
    """
    B, _, H, W = y_z.shape
    dev, ndtype = y_z.device, y_z.dtype
    bn = torch.linalg.vector_norm(b.reshape(B, -1), dim=1).clamp_min(1e-30)

    ts = np.unique(np.linspace(0, int(t_start), int(steps)).astype(int))[::-1]
    null = torch.zeros_like(y_z) if has_null else None

    if init_z is None:
        z = torch.randn(B, 1, H, W, device=dev, dtype=ndtype,
                        generator=generator)
    else:
        # warm start: the observed, pushed to the entry timestep's noise level
        ab0 = alpha_bar[int(ts[0])].to(ndtype)
        z = (ab0.sqrt() * init_z
             + (1 - ab0).sqrt() * torch.randn(init_z.shape, device=dev,
                                              dtype=ndtype, generator=generator))

    hist = []
    for i, t_cur in enumerate(ts):
        t_next = int(ts[i + 1]) if i + 1 < len(ts) else -1
        ab_t = alpha_bar[int(t_cur)].to(ndtype)
        ab_n = (alpha_bar[t_next].to(ndtype) if t_next >= 0
                else torch.ones((), device=dev, dtype=ndtype))

        z = z.detach().requires_grad_(True)
        tt = torch.full((B,), int(t_cur), device=dev, dtype=torch.long)
        eps = model(z, y_z, tt)
        if guidance != 1.0:
            if not has_null:
                raise SystemExit("--guidance != 1 needs a checkpoint trained "
                                 "with p_uncond > 0")
            eps_u = model(z, null, tt)
            eps = eps_u + guidance * (eps - eps_u)
        z0 = (z - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
        z0c = straight_through_clamp(z0, clamp[0], clamp[1])

        # ---- likelihood, through the flux seam ----
        x_flux = ideal_norm.inverse(z0c)                       # -> float64
        Ax = gain * torch.fft.irfft2(torch.fft.rfft2(x_flux) * K, s=(H, W)) + sky
        rel = misfit(Ax, b, bn, pois_c, pois_tau)              # per patch
        grad = torch.autograd.grad(rel.sum(), z)[0]
        with torch.no_grad():                                  # always logged
            rel_l2 = l2_rel(Ax, b, bn, None, None)

        with torch.no_grad():
            if t_next >= 0:
                sigma = (eta * ((1 - ab_n) / (1 - ab_t)).sqrt()
                         * (1 - ab_t / ab_n).clamp(min=0).sqrt())
                dir_c = (1 - ab_n - sigma ** 2).clamp(min=0).sqrt()
                z_prev = ab_n.sqrt() * z0c.detach() + dir_c * eps.detach()
                if eta > 0:
                    z_prev = z_prev + sigma * torch.randn(
                        z.shape, device=dev, dtype=ndtype, generator=generator)
            else:
                z_prev = z0c.detach()

            # trust region on the step LENGTH only; the direction is DPS's.
            step = zeta * grad
            if step_cap > 0:
                sn = torch.linalg.vector_norm(step.reshape(B, -1), dim=1)
                zn = torch.linalg.vector_norm(z_prev.reshape(B, -1), dim=1)
                scale = (step_cap * zn.clamp_min(1e-30)
                         / sn.clamp_min(1e-30)).clamp(max=1.0)
                step = step * scale[:, None, None, None]
            else:
                scale = torch.ones(B, device=dev, dtype=ndtype)
            z = z_prev - step

        rec = {"step": i, "t": int(t_cur),
               "resid": float(rel_l2.mean()),
               "misfit": float(rel.detach().mean()),
               "step_rms": float(step.pow(2).mean().sqrt()),
               "capped": float((scale < 1.0).to(torch.float64).mean()),
               "z_rms": float(z.pow(2).mean().sqrt()),
               "clamp_hi": float((z0 > clamp[1]).to(torch.float64).mean())}
        hist.append(rec)
        if verbose and (i % log_every == 0 or t_next < 0):
            print(f"  step {i:4d}/{len(ts)}  t={rec['t']:4d}  "
                  f"resid={rec['resid']:.4f}  misfit={rec['misfit']:.4f}  "
                  f"|step|_rms={rec['step_rms']:.3e}"
                  f"  capped={100 * rec['capped']:3.0f}%  "
                  f"z_rms={rec['z_rms']:.3f}  "
                  f"clamp_hi={100 * rec['clamp_hi']:.3f}%", flush=True)

    with torch.no_grad():
        return ideal_norm.inverse(
            straight_through_clamp(z.detach(), clamp[0], clamp[1])), hist


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="/home/alex/noir_ml/global/ml-decon/"
                                     "checkpoints_cond_diffusion_npy_m32/best.pt")
    p.add_argument("--data-dir", type=Path,
                   default=Path("/home/alex/noir_ml/global/ml-decon/data/"
                                "m32_klong_nosky"))
    p.add_argument("--split", default="val")
    p.add_argument("--indices", default="94,282,470,658,846,1034,1222,1410")
    p.add_argument("--psf-file", default="psf_klong_epsf.fits")
    p.add_argument("--psf-sigma", type=float, default=None)
    p.add_argument("--weights", default="ema", choices=["ema", "model"])
    p.add_argument("--split-norms", action="store_true", default=False,
                   help="network <- checkpoint dataset_norm, stored patches "
                        "<- data-dir norm.json. Required for the m32 "
                        "checkpoints; see docstring note 3.")
    p.add_argument("--gain", type=float, default=1.0,
                   help="forward model b = gain*(psf*x) + sky. 1.0 matches "
                        "admm_sweep_m32.py; 0.822 is the measured value.")
    p.add_argument("--sky", default="0.0", metavar="S|auto")
    # ---- DPS proper ----
    p.add_argument("--steps", type=int, default=300,
                   help="reverse steps spaced over [0, t-start]")
    p.add_argument("--t-start", type=int, default=None,
                   help="entry timestep (default T-1, i.e. the full chain)")
    p.add_argument("--zeta", type=float, default=1.0,
                   help="guidance scale. The step is zeta * grad of the "
                        "RELATIVE residual, so this is dimensionless.")
    p.add_argument("--data-loss", default="l2", choices=sorted(DATA_LOSS),
                   help="'l2' is textbook DPS: grad of ||Ax-b||/||b||, which "
                        "is driven by the bright stars. 'poisson' weights each "
                        "pixel by 1/(b+c), which is what gives the faint end "
                        "any say. See poisson_rel's docstring.")
    p.add_argument("--pois-c", default="auto", metavar="C|auto",
                   help="Poisson offset in FLUX units: the variance floor sky "
                        "subtraction removed. Small c = aggressive faint-end "
                        "reweighting; c >> b recovers L2. 'auto' = per patch, "
                        "3x the sigma-clipped sky noise (raised if that leaves "
                        "b+c <= 0 anywhere).")
    p.add_argument("--pois-tau", type=float, default=0.5, metavar="F",
                   help="the piecewise seam, as a fraction of c. Below "
                        "mu = F*c the loss switches to its own quadratic "
                        "Taylor expansion, so the gradient stays linear "
                        "instead of going as 1/mu. F must be > 0.")
    p.add_argument("--step-cap", type=float, default=0.1, metavar="F",
                   help="cap the guidance step's per-patch L2 norm at F times "
                        "the iterate's own. 0 disables it -- which diverges on "
                        "the first step at t=999; see docstring note 3.")
    p.add_argument("--eta", type=float, default=0.0,
                   help="0 = deterministic DDIM, 1 = ancestral DDPM")
    p.add_argument("--guidance", type=float, default=1.0,
                   help="classifier-free guidance on the CONDITIONING (this "
                        "is the eps_u + w(eps_c - eps_u) knob, orthogonal to "
                        "--zeta, which weights the measurement)")
    p.add_argument("--init", default="noise", choices=["noise", "observed"],
                   help="'observed' warm-starts the chain from b in the ideal "
                        "domain, noised to --t-start")
    p.add_argument("--clamp", type=float, nargs=2, default=(0.0, 1.0),
                   metavar=("LO", "HI"))
    p.add_argument("--n-samples", type=int, default=1,
                   help=">1: output is the posterior mean, and a per-pixel "
                        "std is saved as the hallucination diagnostic")
    p.add_argument("--no-fits", dest="write_fits", action="store_false",
                   default=True,
                   help="skip the per-patch FITS trees. Same flag name and "
                        "same layout as admm_diffusion_deconvolve.py, so "
                        "final_photometry_table.py --fits-dir <out-dir>/fits "
                        "works on either solver's output.")
    p.add_argument("--n-show", type=int, default=4, metavar="N",
                   help="write compare.png for the first N patches (0 = none)")
    p.add_argument("--shared-stretch", action="store_true", default=False,
                   help="one stretch per row taken from the truth. UNUSABLE on "
                        "the m32 point-source truths, whose background is "
                        "exactly zero -- see panel_norm's docstring.")
    p.add_argument("--net-float64", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--crop", type=int, default=4)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--out-dir", type=Path, default=Path("dps_m32_out"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps(
        {k: (str(v) if isinstance(v, Path) else v)
         for k, v in vars(args).items()}, indent=2, sort_keys=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    DTYPE = torch.float64
    net_dtype = torch.float64 if args.net_float64 else torch.float32

    model, T, ck = load_checkpoint(args.ckpt, device, weights=args.weights,
                                   dtype=net_dtype)
    for prm in model.parameters():          # grads are w.r.t. the INPUT only
        prm.requires_grad_(False)
    alpha_bar = cosine_alpha_bar(T, dtype=DTYPE).to(device)
    has_null = float(ck.get("p_uncond") or 0.0) > 0.0
    t_start = (T - 1) if args.t_start is None else min(int(args.t_start), T - 1)

    # ---- normalizations (see docstring note 3) ----
    ck_norms = ck["dataset_norm"]
    dir_path = args.data_dir / "norm.json"
    dir_norms = json.loads(dir_path.read_text()) if dir_path.exists() else None
    if args.split_norms:
        if dir_norms is None or not ck_norms:
            raise SystemExit("--split-norms needs both norm.json and a "
                             "checkpoint dataset_norm")
        norms = ck_norms
        store_obs_norm = TorchNorm(dir_norms["observed"])
        store_ideal_norm = TorchNorm(dir_norms["ideal"])
        print(f"[norm] --split-norms: network <- checkpoint dataset_norm, "
              f"stored patches <- {dir_path}")
    else:
        norms = ck_norms or dir_norms
        store_obs_norm = TorchNorm(norms["observed"])
        store_ideal_norm = TorchNorm(norms["ideal"])
    obs_norm, ideal_norm = TorchNorm(norms["observed"]), TorchNorm(norms["ideal"])

    # ---- data ----
    observed = np.load(args.data_dir / f"{args.split}_observed.npy")
    ideal = np.load(args.data_dir / f"{args.split}_ideal.npy")
    idx = parse_indices(args.indices, len(observed))
    y_z_stored = torch.from_numpy(observed[idx]).to(DTYPE).to(device)
    b = store_obs_norm.inverse(y_z_stored)                    # flux
    truth = store_ideal_norm.inverse(
        torch.from_numpy(ideal[idx]).to(DTYPE).to(device))
    y_z = obs_norm.forward(b) if args.split_norms else y_z_stored
    B, _, H, W = y_z.shape
    zq = np.percentile(y_z.cpu().numpy(), [1, 50, 99, 100])
    print("[norm] conditioning z 1/50/99/100 = "
          + " ".join(f"{v:.4f}" for v in zq))

    # ---- operator ----
    kernel = (gaussian_kernel(args.psf_sigma) if args.psf_sigma is not None
              else load_kernel(args.psf_file))
    K = make_otf(kernel, (H, W), device)
    gain = float(args.gain)
    if str(args.sky).lower() == "auto":
        from astropy.stats import sigma_clipped_stats
        sky_np = np.array([sigma_clipped_stats(b.cpu().numpy()[i, 0], sigma=3.0,
                                               maxiters=5)[1]
                           for i in range(B)], dtype=np.float64)
    else:
        sky_np = np.full(B, float(args.sky), dtype=np.float64)
    sky = torch.from_numpy(sky_np).to(DTYPE).to(device)[:, None, None, None]

    # The number every accuracy claim below has to be read against: the TRUTH's
    # own residual through this operator. Driving 'resid' below it is fitting
    # forward-model error, not signal.
    with torch.no_grad():
        At = gain * torch.fft.irfft2(torch.fft.rfft2(truth) * K, s=(H, W)) + sky
        bn = torch.linalg.vector_norm(b.reshape(B, -1), dim=1)
        truth_resid = float((torch.linalg.vector_norm((At - b).reshape(B, -1),
                                                      dim=1) / bn).mean())
    print(f"[fwd]  b = {gain:g}*(psf*x) + sky (mean {sky_np.mean():.6g})")
    print(f"[fwd]  the TRUTH's own relative residual = {truth_resid:.4f}  "
          f"<- the target for 'resid', not zero")
    # ---- Poisson offset (see poisson_rel's docstring) ----
    pois_c = pois_tau = None
    if args.data_loss == "poisson":
        b_np = b.cpu().numpy()
        if str(args.pois_c).lower() == "auto":
            from astropy.stats import sigma_clipped_stats
            c_np = np.array([3.0 * sigma_clipped_stats(b_np[i, 0], sigma=3.0,
                                                       maxiters=5)[2]
                             for i in range(B)], dtype=np.float64)
        else:
            c_np = np.full(B, float(args.pois_c), dtype=np.float64)
        # y = b + c must be strictly positive, or the log is undefined and the
        # minimum of the loss stops being at Ax = b.
        floor = -b_np.min(axis=(1, 2, 3)) * 1.05
        bumped = c_np < floor
        if bumped.any():
            print(f"[pois] raised c on {int(bumped.sum())}/{B} patch(es) to "
                  f"keep b+c > 0")
            c_np = np.maximum(c_np, floor)
        if args.pois_tau <= 0:
            raise SystemExit("--pois-tau must be > 0")
        pois_c = torch.from_numpy(c_np).to(DTYPE).to(device)[:, None, None, None]
        pois_tau = float(args.pois_tau) * pois_c
        print(f"[pois] c per patch = "
              + " ".join(f"{v:.4g}" for v in c_np)
              + f"  (tau = {args.pois_tau:g}*c)")
        print(f"[pois] pixel weight 1/(b+c) spans "
              f"{float((b_np.max() + c_np.max()) / c_np.min()):.0f}x "
              f"between the brightest pixel and empty sky")
    print(f"[dps]  data-loss={args.data_loss}, "
          f"{args.steps} steps from t={t_start}, zeta={args.zeta}, "
          f"eta={args.eta}, guidance={args.guidance}, init={args.init}, "
          f"{args.n_samples} sample(s)")

    init_z = (ideal_norm.forward(b).to(net_dtype)
              if args.init == "observed" else None)

    samples, hist = [], None
    for s in range(args.n_samples):
        x_flux, h = dps_sample(
            model, alpha_bar, ideal_norm, y_z.to(net_dtype), b, K, gain, sky,
            zeta=args.zeta, steps=args.steps, t_start=t_start, eta=args.eta,
            guidance=args.guidance, has_null=has_null, clamp=tuple(args.clamp),
            misfit=DATA_LOSS[args.data_loss], pois_c=pois_c, pois_tau=pois_tau,
            step_cap=args.step_cap, init_z=init_z, generator=gen,
            log_every=args.log_every, verbose=(s == 0))
        samples.append(x_flux.cpu().numpy())
        hist = hist or h
        if args.n_samples > 1:
            print(f"  sample {s + 1}/{args.n_samples} done", flush=True)
    z_np = (samples[0] if args.n_samples == 1
            else np.mean(samples, axis=0)).astype(np.float64)

    # ---- outputs, in admm_diffusion_deconvolve.py's convention ----
    np.save(args.out_dir / f"{args.split}_recon.npy", z_np)
    if args.n_samples > 1:
        np.save(args.out_dir / f"{args.split}_std.npy",
                np.std(samples, axis=0).astype(np.float64))
    with open(args.out_dir / "history.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hist[0]))
        w.writeheader(); w.writerows(hist)

    c0 = args.crop
    crop = np.s_[c0:H - c0, c0:W - c0]
    truth_np = truth.cpu().numpy()[:, 0]
    rows = []
    for i2, kk in enumerate(idx):
        xf, tf = z_np[i2, 0][crop], truth_np[i2][crop]
        rows.append({"index": kk,
                     "flux_ratio": float(xf.sum() / max(tf.sum(), 1e-30)),
                     "peak_ratio": float(xf.max() / max(tf.max(), 1e-30)),
                     "rmse": float(np.sqrt(np.mean((xf - tf) ** 2)))})
    with open(args.out_dir / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # ---- per-patch FITS, in admm_diffusion_deconvolve.py's exact layout ----
    # The photometry tools glob a directory and pair recon with truth BY
    # FILENAME, so both trees use identical basenames and stale files from a
    # previous --indices must go, or two configurations get pooled silently.
    if args.write_fits:
        from astropy.io import fits as _fits
        for sub, arr in (("recon", z_np[:, 0]), ("truth", truth_np)):
            d = args.out_dir / "fits" / sub
            d.mkdir(parents=True, exist_ok=True)
            stale = sorted(d.glob("patch_*.fits"))
            for f in stale:
                f.unlink()
            if stale and len(stale) != arr.shape[0]:
                print(f"[fits] cleared {len(stale)} stale patches in {d}")
            for i in range(arr.shape[0]):
                _fits.writeto(d / f"patch_{i:02d}.fits",
                              arr[i].astype(np.float64), overwrite=True)
        print(f"[fits] {args.out_dir}/fits/{{recon,truth}}/patch_*.fits "
              f"({B} each)")

    # ---- figure ----
    if args.n_show > 0:
        b_np = b.cpu().numpy()
        ns = min(args.n_show, B)
        fig, axes = plt.subplots(ns, 4, figsize=(13, 3.2 * ns), squeeze=False)
        for i in range(ns):
            # per-panel by default: shared_norm(truth) degenerates to a step
            # function on this data. See panel_norm's docstring.
            if args.shared_stretch:
                nb = nr = nt = shared_norm(truth_np[i])
            else:
                nb, nr, nt = (panel_norm(b_np[i, 0]), panel_norm(z_np[i, 0]),
                              panel_norm(truth_np[i]))
            show(axes[i][0], b_np[i, 0], f"observed (idx {idx[i]})", nb)
            show(axes[i][1], z_np[i, 0], "DPS recon", nr)
            show(axes[i][2], truth_np[i], "truth", nt)
            r = (gain * np.fft.irfft2(np.fft.rfft2(z_np[i, 0])
                                      * K.cpu().numpy(), s=(H, W))
                 + sky_np[i] - b_np[i, 0])
            v = max(float(np.percentile(np.abs(r), 99.5)), 1e-30)
            axes[i][3].imshow(r, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
            axes[i][3].set_title("A(x) - b", fontsize=9)
            axes[i][3].axis("off")
        fig.tight_layout()
        fig.savefig(args.out_dir / "compare.png", dpi=140)
        plt.close(fig)
        print(f"[fig]  {args.out_dir}/compare.png")

    # Score the SAVED array, not hist[-1]: the last logged residual belongs to
    # the final z0 estimate, and one more guidance step is applied after it.
    with torch.no_grad():
        xt = torch.from_numpy(z_np).to(DTYPE).to(device)
        Axo = gain * torch.fft.irfft2(torch.fft.rfft2(xt) * K, s=(H, W)) + sky
        out_resid = float((torch.linalg.vector_norm((Axo - b).reshape(B, -1),
                                                    dim=1) / bn).mean())
    print(f"\n[out]  resid of the SAVED image = {out_resid:.4f}  "
          f"(truth {truth_resid:.4f}, |delta| "
          f"{abs(out_resid - truth_resid):.4f}; last chain step logged "
          f"{hist[-1]['resid']:.4f} before its own guidance step)")
    print(f"[out]  flux/truth = "
          f"{float(z_np[:, 0][:, crop[0], crop[1]].sum() / truth_np[:, crop[0], crop[1]].sum()):.4f}")
    print(f"[out]  {args.out_dir}/{args.split}_recon.npy")


if __name__ == "__main__":
    main()
