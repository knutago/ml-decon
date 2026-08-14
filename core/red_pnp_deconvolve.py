"""
red_pnp_deconvolve.py

PnP/RED deconvolution driven by the CONDITIONAL diffusion model, running
directly on the generated .npy patch datasets (data/m31bK50 style: the
output of ml-decon's dataset/gen_data.py).

Objective (RED, Romano/Elad/Milanfar 2017; conditional variant):

    min_x  0.5 ||A x - y||^2  +  (lam/2) ||x - D(x; y)||^2

solved by gradient descent with the standard RED gradient

    grad = A^T (A x - y)  +  lam * (x - D(x; y))

where D is a SHORT DETERMINISTIC REVERSE CHAIN of the conditional eps-model,
started at an annealed timestep t:

    z    = ideal_norm.forward(x)                    # flux -> model domain
    x_t  = sqrt(abar_t) * z                         # NO injected noise (RED
                                                    # wants a deterministic
                                                    # denoiser, not a sampler)
    repeat --denoise-steps times, t -> 0:
        eps = eps_theta(x_t, y_norm, t)             # conditioned on observed
        z0  = (x_t - sqrt(1 - abar_t) * eps) / sqrt(abar_t)
        x_t = sqrt(abar_next) * z0 + sqrt(1 - abar_next) * eps   # DDIM, eta=0
    D(x) = ideal_norm.inverse(z0)                   # back to flux

WHY A CHAIN AND NOT ONE TWEEDIE STEP.  The original version of this script ran
exactly one step. That is a correct posterior-mean estimator but a useless RED
denoiser, for a reason that is structural rather than a tuning problem: at
small t, E[x_0 | x_t] is close to the identity by construction, and this
checkpoint's --identity-frac term trains it to be *exactly* the identity on the
sharp manifold. So ||x - D(x)|| collapses, the prior force vanishes, and the
data term runs unopposed -- which is what produced the ringing and the 43%
clamped-to-zero pixels. Measured concentration lift of D on a blurry input:

    1 step,  t=5     0.067 -> 0.244
    1 step,  t>=20   0.067 -> ~0.08
    250-step chain   0.067 -> 0.762      (truth 0.761)

Sharpness in a diffusion model lives in the ITERATED chain. --denoise-steps > 1
is therefore not an optimization, it is what makes the operator a denoiser at
all. The zero-noise entry point is still the configuration --identity-frac
optimizes, so the fixed-point property RED requires is preserved.

Domains
-------
The dataset stores patches in the generator's normalized domain (norm.json,
separate asinh maps for observed and ideal). The model lives there; the
convolution forward model is only linear in PHYSICAL flux. So:
  * conditioning channel  = stored observed patch, used as-is;
  * data term + optimizer = physical flux, via norm.json inverse();
  * D(x) wraps forward/inverse around the model call.
Per RED convention the normalization Jacobian is ignored in the gradient.

Diagnostics
-----------
Every log interval the script prints  cos(x - D(x), x - truth)  averaged over
the batch. This is the go/no-go signal from the earlier RED runs: if the RED
gradient is not positively aligned with the direction toward truth, no lambda
will save the run -- kill it and fix the denoiser instead.

Caveats
-------
  * A is applied by FFT, i.e. periodic boundaries; patch edges see flux from
    outside the patch that the model cannot explain. Metrics crop 4 px.
  * The PSF is the estimated one (psf_true.fits by default); its bandwidth
    bounds what any solver can recover.

Usage (after training a checkpoint):
  python red_pnp_deconvolve.py \
      --ckpt checkpoints_cond_diffusion_npy/best.pt \
      --data-dir /home/alex/noir_ml/global/ml-decon/data/m31bK50 \
      --split val --indices 0:8 \
      --psf-file psf_true.fits \
      --lam 0.1 --iters 300 --t-start 60 --t-end 5
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
from matplotlib.colors import AsinhNorm

# Architecture + schedule come from the trainer so they cannot drift.
from train_conditional_diffusion import ConditionalFlatCNN, cosine_alpha_bar


# ----------------------------------------------------------------------------
# Normalization (torch port of ml-decon core/normalize.py, from norm.json)
# ----------------------------------------------------------------------------
class TorchNorm:
    """forward: physical flux -> model domain; inverse: back. Exact, unclipped.

    ALWAYS COMPUTES IN float64 AND RETURNS float64, whatever the input dtype.

    This is not defensive style, it is required by the transform's own
    conditioning. `inverse` is median + beta*sinh(y*span + lo_s) with
    span = 13.33 for m31bK50, so d(flux)/dy = beta*span*cosh(...) -- the
    derivative grows exponentially in y. Near the bright end one normalized
    unit is ~14.475 mag, so a float32 rounding of ~6e-8 in y is ~1e-6 mag
    directly, and the sinh amplifies any error accumulated upstream by a
    factor that reaches ~1e6 at y = 1.5. float32 also OVERFLOWS to inf at
    y ~ 2.0, which float64 does not until y ~ 7.
    """

    DTYPE = torch.float64

    def __init__(self, params: dict):
        self.method = params["method"]
        # json floats are already Python (64-bit) floats; keep them that way
        # and only ever build tensors from them at DTYPE.
        self.p = {k: float(v) for k, v in params.items() if k != "method"}
        if self.method not in ("asinh", "linear"):
            raise ValueError(f"unsupported normalization: {self.method}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p
        x = x.to(self.DTYPE)
        if self.method == "asinh":
            s = torch.asinh((x - p["median"]) / p["beta"])
            return (s - p["lo_s"]) / (p["hi_s"] - p["lo_s"])
        return (x - p["lo"]) / (p["hi"] - p["lo"])

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        p = self.p
        y = y.to(self.DTYPE)
        if self.method == "asinh":
            s = y * (p["hi_s"] - p["lo_s"]) + p["lo_s"]
            return p["median"] + p["beta"] * torch.sinh(s)
        return y * (p["hi"] - p["lo"]) + p["lo"]


# ----------------------------------------------------------------------------
# Forward operator: FFT convolution (periodic), batched (B, 1, H, W)
# ----------------------------------------------------------------------------
def make_otf(kernel: np.ndarray, shape, device):
    k = np.asarray(kernel, dtype=np.float64)
    k = k / k.sum()
    H, W = shape
    pad = np.zeros((H, W))
    kh, kw = k.shape
    pad[:kh, :kw] = k
    pad = np.roll(pad, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    # complex128, not complex64: the x-update divides by (|K|^2 + rho) every
    # outer iteration, and psf_50_true's MTF falls to ~2% at the band edge, so
    # the quotient is ill-conditioned exactly where the OTF is smallest.
    K = torch.from_numpy(np.fft.rfft2(pad).astype(np.complex128)).to(device)
    return K


def gaussian_kernel(sigma, truncate=4.0):
    r = max(1, int(truncate * sigma + 0.5))
    ax = np.arange(-r, r + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return (k / k.sum()).astype(np.float64)


def load_kernel(path):
    if path.endswith(".npy"):
        k = np.load(path)
    else:
        from astropy.io import fits
        k = fits.getdata(path)
    return np.asarray(k, dtype=np.float64).squeeze()


# ----------------------------------------------------------------------------
# The denoiser D(x; y): a short deterministic conditional reverse chain
# ----------------------------------------------------------------------------
def model_dtype(model) -> torch.dtype:
    """The dtype the network's weights actually live in."""
    return next(model.parameters()).dtype


def model_eps(model, x_t, y_z, tt):
    """Call the network across a possible dtype boundary.

    Everything OUTSIDE the network runs in float64 (see TorchNorm). The network
    itself runs in whatever its weights are -- float32 for the existing
    checkpoints, float64 if the caller has run `model.double()`. Cast in, cast
    the prediction straight back, so the chain arithmetic (which is where error
    accumulates over hundreds of outer iterations) never silently downcasts.

    Note the fp32 network is not a precision leak worth fixing by doubling the
    weights: its output eps is a LEARNED quantity whose own error is ~1e-2, six
    orders above fp32 epsilon. What float64 protects is the arithmetic wrapped
    around it -- the asinh inverse, the FFT solve, and the ADMM recursion.
    """
    md = model_dtype(model)
    out_dtype = x_t.dtype
    return model(x_t.to(md), y_z.to(md), tt).to(out_dtype)


@torch.no_grad()
def tweedie_chain(model, z, y_z, t0, n_steps, alpha_bar, guidance=1.0,
                  has_null=False, eta=0.0, generator=None, clamp=(0.0, 0.9),
                  renoise=False, trace=None, inject_scale=1.0):
    """DDIM reverse chain from t0 down to 0, in the model's normalized domain.

    n_steps=1 is the bare Tweedie step  z0 = (x_t - sqrt(1-abar) eps)/sqrt(abar).
    That step is NOT a usable RED denoiser: measured on blurry input it lifts
    point-source concentration from 0.067 to only 0.244 at t=5 and ~0.08 at
    t>=20, because E[x_0 | x_t] at small t is nearly the identity by
    construction (and this checkpoint's identity_frac term trains it to be
    exactly that on the sharp manifold). Sharpness in a diffusion model is a
    property of the ITERATED chain -- the same 250-step chain reaches 0.762.
    So RED needs n_steps > 1 for D to actually move x toward the prior.

    The input is scaled as x_t = sqrt(abar_t0) * z rather than having noise
    injected: the RED iterate already carries its own error, and injecting
    fresh noise on top makes D stochastic, which breaks the RED gradient (it
    is a valid gradient only for a deterministic, locally homogeneous D).

    inject_scale DECOUPLES the injected noise level from the conditioning
    level, which is the central prescription of Park et al. 2026 ("Stochastic
    Generative Plug-and-Play Priors", arXiv:2604.03603), Sec. 3.2 / App. C.1:

        z_k = DC_k(x_k; y),   x_{k+1} = D_theta(z_k + sigma_inject * n;
                                                sigma_cond)

    renoise=True with inject_scale=1.0 is the MATCHED setting sigma_inject =
    sigma_cond, i.e. exactly SNORE's assumption. The paper's point is that the
    match is wrong: an ADMM iterate carries residual measurement noise and
    forward-operator artifacts on top of the injected noise, so the denoiser
    should be conditioned at a HIGHER level than it is perturbed at. Setting
    inject_scale < 1 gives sigma_inject < sigma_cond. Their CS-MRI ADMM row
    uses sigma_cond/sigma_inject = 100 (inject_scale 0.01) and gains +1.88 dB
    over matched; their deblurring row stays matched. Worth a sweep, not a
    default -- so this defaults to 1.0, which is the previous behaviour.

    renoise=True overrides that and does a PROPER forward diffusion to level
    t0:   x_t = sqrt(abar)*z + sqrt(1-abar)*eps.
    Use it when the input is CLEANER than its timestep implies -- which is the
    case late in an ADMM run, where z has nearly converged and carries far
    less noise than sigma(t0). Feeding a too-clean input to the model is
    out-of-distribution in the direction that makes it do nothing, so the
    chain preserves the blurry coarse structure instead of regenerating sharp
    sources. Renoising restores the training-time relationship between the
    input and the timestep embedding, and sets the amount from t0, a quantity
    with meaning, rather than an unbounded multiplier.

    CORRECTION (measured 2026-08-11): the claim that eta > 1 "deletes every
    refinement step" by driving 1 - abar_next - s^2 negative is FALSE at the
    settings actually used. That term only goes negative near the END of the
    chain, where 1 - abar_next is already tiny. At t0=75, n_steps=20 the count
    of zeroed steps is 0/19 at eta=1.0, 2/19 at eta=1.5, 6/19 at eta=2.0 -- so
    eta=1.5 runs 17 intact DDIM steps and simply injects ~5.6x more total noise
    than renoise alone (0.735 vs 0.130). It over-noises, which may or may not be
    wanted, but it is not structurally broken. The zeroing only bites hard with
    LARGE step gaps (few steps over a long range). See --eta's help text in
    admm_diffusion_deconvolve.py, which still states the old claim.

    trace: optional list. If given, one dict per chain step is appended,
    recording how far z0 moves per step (in normalized units, RMS) and whether
    the deterministic direction term survived. Use it to test whether the late
    chain is contributing at all -- at eta=0 the deterministic coefficient
    decays 0.124 -> 0.006 across a t0=75 chain, and if dz0 decays with it the
    last steps are dead weight and entry-only renoising cannot reach them.
    """
    # The whole chain runs at z's dtype (float64 when reached through
    # TorchNorm.forward); alpha_bar is indexed into that dtype so no product
    # below silently promotes or demotes.
    dt = z.dtype
    alpha_bar = alpha_bar.to(dt)
    x_t = alpha_bar[int(t0)].sqrt() * z
    if renoise:
        # inject_scale = sigma_inject / sigma_cond. 1.0 reproduces matched
        # forward diffusion; < 1 is the paper's decoupled regime.
        x_t = x_t + inject_scale * (1 - alpha_bar[int(t0)]).sqrt() * torch.randn(
            z.shape, device=z.device, generator=generator, dtype=dt)
    grid = np.unique(np.linspace(0, int(t0), max(int(n_steps), 1)).astype(int))[::-1]
    null = torch.zeros_like(y_z)
    z0_prev = None

    for i, t_cur in enumerate(grid):
        ab_t = alpha_bar[int(t_cur)]
        tt = torch.full((z.shape[0],), int(t_cur), device=z.device,
                        dtype=torch.long)
        eps = model_eps(model, x_t, y_z, tt)
        if guidance != 1.0:
            if not has_null:
                raise ValueError("--guidance != 1 needs a checkpoint trained "
                                 "with p_uncond > 0 (null token is untrained "
                                 "otherwise)")
            eps = (model_eps(model, x_t, null, tt) * (1 - guidance)
                   + eps * guidance)
        z0 = (x_t - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
        if clamp is not None:
            # The asinh inverse is EXPLOSIVE at the bright end: for m31bK50 the
            # s-span is 13.3, so normalized 1.0 -> flux 1780 but 1.2 -> 2.6e4
            # and 1.5 -> 1.4e6. A 20% overshoot in normalized units is a 14x
            # flux error, and 2.0 overflows float32 to inf. Training data is
            #exactly [0, 1], so clamping there is the data's own support.
            frac_lo = float((z0 < clamp[0]).float().mean())
            frac_hi = float((z0 > clamp[1]).float().mean())
            z0 = z0.clamp(clamp[0], clamp[1])
        else:
            frac_lo = frac_hi = 0.0

        last = (i + 1 == len(grid))
        if not last:
            ab_n = alpha_bar[int(grid[i + 1])]
            s = eta * ((1 - ab_n) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_n).sqrt()
            det_raw = float(1 - ab_n - s ** 2)      # BEFORE the clamp at 0
            det = (1 - ab_n - s ** 2).clamp(min=0).sqrt()

        if trace is not None:
            # RMS, not the raw norm, so the numbers are per-pixel normalized
            # units and directly comparable across patch counts. In this
            # dataset's asinh domain 1 normalized unit ~ 14.475 mag at the
            # bright end, so dz0 = 0.001 is ~0.0145 mag.
            n = z0.numel()
            trace.append({
                "step": i,
                "t": int(t_cur),
                "t_next": int(grid[i + 1]) if not last else 0,
                "dz0_rms": (float(torch.linalg.vector_norm(z0 - z0_prev))
                            / n ** 0.5) if z0_prev is not None else float("nan"),
                "z0_rms": float(torch.linalg.vector_norm(z0)) / n ** 0.5,
                "eps_rms": float(torch.linalg.vector_norm(eps)) / n ** 0.5,
                "det_coef": float(det) if not last else float("nan"),
                "noise_coef": float(s) if not last else float("nan"),
                "det_zeroed": bool(det_raw < 0) if not last else False,
                "clamp_frac_lo": frac_lo,
                "clamp_frac_hi": frac_hi,
            })

        if last:
            return z0
        z0_prev = z0
        x_t = ab_n.sqrt() * z0 + det * eps
        if eta > 0:
            x_t = x_t + s * torch.randn(x_t.shape, device=x_t.device,
                                        generator=generator, dtype=dt)
    return z0


@torch.no_grad()
def red_denoise(model, x_phys, y_z, t, alpha_bar, ideal_norm, n_steps=1,
                guidance=1.0, has_null=False, add_noise=False, generator=None,
                clamp=None, eta=0.0):
    """D(x; y) in PHYSICAL FLUX: normalize -> reverse chain -> denormalize."""
    z = ideal_norm.forward(x_phys)          # float64 from here on
    if add_noise:
        ab = alpha_bar.to(z.dtype)[int(t)]
        z = z + ((1 - ab).sqrt() / ab.sqrt()) * torch.randn(
            z.shape, device=z.device, generator=generator, dtype=z.dtype)
    z0 = tweedie_chain(model, z, y_z, t, n_steps, alpha_bar, guidance=guidance,
                       has_null=has_null, eta=eta, generator=generator,
                       clamp=clamp)
    return ideal_norm.inverse(z0)


# ----------------------------------------------------------------------------
# Checkpoint / data loading
# ----------------------------------------------------------------------------
def load_checkpoint(path, device, weights="ema", dtype=torch.float32):
    """`dtype` casts the NETWORK's weights. Default float32 = as trained.

    float64 makes the forward pass exact to the weights but does not make the
    weights themselves more accurate, and roughly triples CPU inference cost.
    Everything around the network is float64 regardless -- see model_eps.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    if "dataset_norm" not in ck:
        raise KeyError(
            f"{path} has no 'dataset_norm' -- this is an OLD-format checkpoint "
            "(trainer-side asinh transform). This script only supports the "
            "rewritten trainer's checkpoints.")
    arch = ck.get("arch", {})
    model = ConditionalFlatCNN(channels=arch.get("channels", 64),
                               t_dim=arch.get("t_dim", 128)).to(device)
    state_key = "ema_state" if (weights == "ema" and "ema_state" in ck) \
        else "model_state"
    state = ck[state_key]
    # ema_pytorch prefixes keys; strip any wrapper prefix defensively
    state = {k.split("ema_model.")[-1]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    model.to(dtype)
    T = ck["diffusion"]["timesteps"]
    print(f"[ckpt] {path}  weights={state_key}  T={T}  "
          f"net_dtype={str(dtype).replace('torch.', '')}  "
          f"identity_frac={ck.get('identity_frac')}  "
          f"low_t_frac={ck.get('low_t_frac')}  p_uncond={ck.get('p_uncond')}")
    return model, T, ck


def parse_indices(spec, n):
    if ":" in spec:
        lo, hi = spec.split(":")
        return list(range(int(lo or 0), min(int(hi or n), n)))
    return [int(s) for s in spec.split(",")]


def sig_std(a, iters=5, k=3.0):
    a = np.asarray(a, np.float64).ravel().copy()
    for _ in range(iters):
        m, s = np.median(a), np.std(a)
        keep = np.abs(a - m) < k * s
        if keep.sum() in (0, a.size):
            break
        a = a[keep]
    return float(np.std(a))


def shared_norm(ref, vmax=None):
    """One AsinhNorm for every panel of a figure, derived from `ref`.

    Pass the TRUTH panel as `ref`. Without this, `show` recomputes the stretch
    per panel from that panel's own min/max/noise, so two panels holding nearly
    identical data can look completely different -- a wing at a few
    ten-thousandths of the peak renders at mid-grey in one panel and black in
    the next. That artifact has repeatedly read as "the reconstruction is
    blurred at bright stars" when the measured radial profiles were identical
    to truth within 3e-4.

    `vmax` defaults to ref.max(); pass an explicit value when a reconstruction
    can legitimately overshoot the truth and you want the excess visible rather
    than saturated.
    """
    return AsinhNorm(linear_width=max(sig_std(ref), 1e-30),
                     vmin=float(ref.min()),
                     vmax=float(ref.max()) if vmax is None else float(vmax))


def show(ax, img, title, norm=None):
    """`norm=None` keeps the old per-panel stretch (every existing caller).

    Pass a `shared_norm(truth)` to make panels comparable by eye.
    """
    if norm is None:
        norm = AsinhNorm(linear_width=max(sig_std(img), 1e-30),
                         vmin=float(img.min()), vmax=float(img.max()))
    ax.imshow(img, origin="lower", cmap="gray", norm=norm)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints_cond_diffusion_npy/best.pt")
    p.add_argument("--data-dir", type=Path,
                   default=Path("/home/alex/noir_ml/global/ml-decon/data/m31bK50"))
    p.add_argument("--split", choices=["val", "train"], default="val")
    p.add_argument("--indices", default="0:8",
                   help="patches to solve: '0,5,12' or a slice '0:8'")
    p.add_argument("--psf-file", default="psf_true.fits")
    p.add_argument("--psf-sigma", type=float, default=None,
                   help="use a Gaussian PSF of this sigma instead of --psf-file")
    p.add_argument("--lam", type=float, default=0.1,
                   help="RED weight lambda")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--t-start", type=int, default=60,
                   help="denoiser timestep at iteration 0 (anneals down). Keep "
                        "<= the checkpoint's low_t_max: that is the regime the "
                        "low-t oversampling actually trained.")
    p.add_argument("--t-end", type=int, default=5)
    p.add_argument("--guidance", type=float, default=1.0,
                   help="classifier-free guidance weight; 1.0 = plain "
                        "conditional (needs p_uncond > 0 in the ckpt otherwise)")
    p.add_argument("--weights", choices=["ema", "raw"], default="ema")
    p.add_argument("--clamp", type=float, nargs=2, default=[0.0, 1.0],
                   metavar=("LO", "HI"),
                   help="clamp D's x0 estimate in NORMALIZED units before the "
                        "asinh inverse. Default [0,1] = the training data's "
                        "exact support; the inverse is explosive past it "
                        "(1.2 -> 14x flux error). Pass a wider range to "
                        "disable the guard.")
    p.add_argument("--denoise-steps", type=int, default=4,
                   help="reverse-chain steps inside D. 1 = the bare Tweedie "
                        "step, which is NOT a usable RED denoiser: at small t "
                        "E[x0|x_t] is nearly the identity, so ||x - D(x)|| is "
                        "tiny and the prior barely acts (measured concentration "
                        "lift on blurry input: 0.067 -> 0.244). >1 makes D an "
                        "actual projection toward the prior manifold. Cost is "
                        "linear in this, so 3-8 is the practical range.")
    p.add_argument("--add-noise", action="store_true",
                   help="inject noise inside D (default OFF -- RED wants a "
                        "deterministic denoiser)")
    p.add_argument("--step-scale", type=float, default=1.0,
                   help=">1 shrinks the gradient step")
    p.add_argument("--no-nonneg", dest="nonneg", action="store_false")
    p.add_argument("--crop", type=int, default=4,
                   help="border crop (px) for metrics, edge/wrap effects")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("red_pnp_out"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    # --- model + norms ------------------------------------------------------
    model, T, ck = load_checkpoint(args.ckpt, device, weights=args.weights)
    alpha_bar = cosine_alpha_bar(T).to(device)
    norms = ck["dataset_norm"]
    if norms is None:
        norms = json.loads((args.data_dir / "norm.json").read_text())
        print(f"[norm] checkpoint carried none; using {args.data_dir}/norm.json")
    obs_norm = TorchNorm(norms["observed"])
    ideal_norm = TorchNorm(norms["ideal"])

    # --- data ---------------------------------------------------------------
    observed = np.load(args.data_dir / f"{args.split}_observed.npy")
    ideal = np.load(args.data_dir / f"{args.split}_ideal.npy")
    idx = parse_indices(args.indices, len(observed))
    print(f"[data] {args.split}: solving {len(idx)} patches {idx} "
          f"of {len(observed)}")

    # .double(): the .npy is float32 on disk so the cast is exact, but it keeps
    # every downstream op (norm inverse, FFT, RED recursion) in float64.
    y_z = torch.from_numpy(observed[idx]).double().to(device)     # conditioning
    y_phys = obs_norm.inverse(y_z)                                # data term
    truth = ideal_norm.inverse(
        torch.from_numpy(ideal[idx]).double().to(device))
    B, _, H, W = y_z.shape

    # --- forward operator ---------------------------------------------------
    if args.psf_sigma is not None:
        kernel = gaussian_kernel(args.psf_sigma)
        print(f"[psf] Gaussian sigma={args.psf_sigma}")
    else:
        kernel = load_kernel(args.psf_file)
        print(f"[psf] {args.psf_file}  {kernel.shape}  sum={kernel.sum():.6f}")
    K = make_otf(kernel, (H, W), device)
    K2 = (K.conj() * K).real

    def A(u):
        return torch.fft.irfft2(torch.fft.rfft2(u) * K, s=(H, W))

    def AT(u):
        return torch.fft.irfft2(torch.fft.rfft2(u) * K.conj(), s=(H, W))

    # --- schedule + step size ----------------------------------------------
    t_sched = np.geomspace(max(args.t_start, 1), max(args.t_end, 1),
                           args.iters).astype(int)
    L = float(K2.max())                     # ||A^T A|| (=1 for a unit-sum PSF)
    step = 1.0 / ((L + args.lam) * args.step_scale)
    print(f"[opt] lam={args.lam}  step={step:.4f}  t {args.t_start}->{args.t_end}"
          f"  iters={args.iters}  add_noise={args.add_noise}")

    has_null = float(ck.get("p_uncond") or 0.0) > 0.0
    x = y_phys.clone()                      # init at the observation
    hist = {"data": [], "red": [], "cos": []}
    yn = float(torch.linalg.vector_norm(y_phys))

    for i in range(args.iters):
        t = int(t_sched[i])
        Dx = red_denoise(model, x, y_z, t, alpha_bar, ideal_norm,
                         n_steps=args.denoise_steps,
                         guidance=args.guidance, has_null=has_null,
                         add_noise=args.add_noise, generator=gen,
                         clamp=tuple(args.clamp))
        red = x - Dx
        resid = A(x) - y_phys
        x = x - step * (AT(resid) + args.lam * red)
        if args.nonneg:
            x = x.clamp(min=0.0)

        data_r = float(torch.linalg.vector_norm(resid)) / max(yn, 1e-30)
        red_n = float(torch.linalg.vector_norm(red))
        # go/no-go: is the RED gradient pointed toward the truth?
        u = red.reshape(B, -1)
        v = (x - truth).reshape(B, -1)
        cos = float((torch.sum(u * v, 1)
                     / (u.norm(dim=1) * v.norm(dim=1) + 1e-30)).mean())
        hist["data"].append(data_r)
        hist["red"].append(red_n)
        hist["cos"].append(cos)
        if i % args.log_every == 0 or i == args.iters - 1:
            fr = float(x.sum() / truth.sum())
            print(f"  it {i:4d}  t={t:3d}  ||Ax-y||/||y||={data_r:.4f}  "
                  f"||x-Dx||={red_n:.3e}  cos(red, x-truth)={cos:+.3f}  "
                  f"flux/truth={fr:.3f}", flush=True)
            xn = float(torch.linalg.vector_norm(x))
            # only meaningful when the RED residual is above float32
            # round-trip noise from the asinh forward/inverse pair
            if (i == 3 * args.log_every and np.mean(hist["cos"]) < 0
                    and red_n > 1e-3 * xn):
                print("  [WARN] cos(x-D(x), x-truth) is NEGATIVE on average: "
                      "the RED gradient points AWAY from truth. Past runs with "
                      "this signature never beat the no-prior baseline -- "
                      "consider stopping and re-examining the denoiser.")

    # --- metrics + outputs --------------------------------------------------
    c = args.crop
    crop = np.s_[c:H - c, c:W - c]
    x_np = x.cpu().numpy()
    tr_np = truth.cpu().numpy()
    y_np = y_phys.cpu().numpy()
    Dx_np = Dx.cpu().numpy()

    rows = []
    for b, k in enumerate(idx):
        xf, tf = x_np[b, 0][crop], tr_np[b, 0][crop]
        rows.append({
            "index": k,
            "flux_ratio": float(xf.sum() / max(tf.sum(), 1e-30)),
            "peak_ratio": float(xf.max() / max(tf.max(), 1e-30)),
            "rmse": float(np.sqrt(np.mean((xf - tf) ** 2))),
            "data_resid": hist["data"][-1],
        })
    with open(args.out_dir / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[result] mean flux_ratio={np.mean([r['flux_ratio'] for r in rows]):.4f}"
          f"  mean rmse={np.mean([r['rmse'] for r in rows]):.4e}"
          f"  final cos={hist['cos'][-1]:+.3f}")

    np.save(args.out_dir / f"{args.split}_recon.npy", x_np.astype(np.float32))
    (args.out_dir / "run_config.json").write_text(json.dumps(
        {k: (str(v) if isinstance(v, Path) else v)
         for k, v in vars(args).items()} | {"indices_resolved": idx}, indent=2))

    # per-patch figure (first 6): observed / recon / D(x) / truth
    nshow = min(B, 6)
    fig, ax = plt.subplots(nshow, 4, figsize=(14, 3.6 * nshow), squeeze=False)
    for b in range(nshow):
        show(ax[b, 0], y_np[b, 0], f"observed (idx {idx[b]})")
        show(ax[b, 1], x_np[b, 0],
             f"RED recon  flux/truth={rows[b]['flux_ratio']:.3f}")
        show(ax[b, 2], Dx_np[b, 0], "D(x) final")
        show(ax[b, 3], tr_np[b, 0], "ideal (truth)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "patches.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].semilogy(hist["data"]); ax[0].set_title("||Ax-y|| / ||y||", fontsize=9)
    ax[1].semilogy(hist["red"]);  ax[1].set_title("||x - D(x)||", fontsize=9)
    ax[2].plot(hist["cos"]); ax[2].axhline(0, color="r", lw=0.8)
    ax[2].set_title("cos(x - D(x), x - truth)", fontsize=9)
    for a in ax:
        a.set_xlabel("iteration"); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "convergence.png", dpi=130)
    plt.close(fig)
    print(f"Wrote {args.out_dir}/patches.png, convergence.png, "
          f"{args.split}_recon.npy, metrics.csv")


if __name__ == "__main__":
    main()
