"""The conditional diffusion denoiser: loading it, and running it.

Everything here is the DENOISER D(x; y) and nothing else -- the reverse chain,
the dtype boundary around the network, and rebuilding a trained denoiser from a
checkpoint. It was extracted verbatim from red_pnp_deconvolve.py, which had
accumulated into the repo's shared library and was an odd place for a solver to
reach for its prior.

MOVED, NOT COPIED. red_pnp_deconvolve.py re-imports these names, so the ~40
modules that do `from red_pnp_deconvolve import tweedie_chain, load_checkpoint`
keep working unchanged and there is still exactly ONE definition of each. Do
not paste a second copy anywhere: tweedie_chain carries a documented bug
history (the n_steps=1 linspace bug that invalidated every one-step result in
the repo) and load_checkpoint carries the arch-dispatch that keeps a UNet
checkpoint from being loaded into a flat CNN. A fork of either would silently
lose those.

The normalization (TorchNorm), the PSF/OTF helpers and the plotting utilities
deliberately stay in red_pnp_deconvolve.py -- they are not denoiser code.
"""
import numpy as np
import torch

from train_conditional_diffusion import model_from_checkpoint


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
                  renoise=False, trace=None, inject_scale=1.0, eps_fn=None):
    """DDIM reverse chain from t0 down to 0, in the model's normalized domain.

    n_steps=1 is the bare Tweedie step  z0 = (x_t - sqrt(1-abar) eps)/sqrt(abar).

    BUG, found 2026-08-29 and fixed below: until now n_steps=1 did NOT compute
    that. `np.linspace(0, t0, 1)` returns [0], not [t0], so the single-step
    path evaluated the model at t=0 -- telling it the input was already clean
    -- and returned ~0.99*z - 0.0064*eps, a near-identity, for any t0.

    Every "one Tweedie step does not sharpen" result in this repo went through
    that path and measures the near-identity, NOT Tweedie:
      - the 0.067 -> 0.244 / ~0.08 numbers this docstring used to quote;
      - the "flat in t" observation (concentration 0.118 at t=5, 20 AND 60),
        which the bug predicts exactly -- the only t-dependence left is
        sqrt(abar[t0]), 0.9998 -> 0.9943, and concentration is scale-blind;
      - the `RED (1-step) conc 0.621` row in admm_diffusion_deconvolve.py's
        header table, via compare_solvers.py:210;
      - the ds1_* arm of admm_sweep_m32.py;
      - the n_steps=1 rows in diffusion_crash_course.py:1198.
    All of those need re-measuring before "the chain beats one step" can be
    claimed again. The chain may still win -- but not for the stated reason,
    and not by the stated margin.

    Sharpness in a diffusion model is still expected to be a property of the
    ITERATED chain (a 250-step chain reaches concentration 0.762), so
    n_steps > 1 remains the default. That claim is now untested at n_steps=1.

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

    eps_fn: optional callable (x_t, tt) -> eps, used INSTEAD of the
    conditional model_eps(model, x_t, y_z, tt). This is what lets an
    UNCONDITIONAL prior (flat_cnn_diffusion.FlatCNN, eps_theta(x_t, t) with no
    conditioning channel) run through the identical DDIM/renoise/eta/clamp
    arithmetic as the conditional one, so a cond-vs-uncond comparison differs
    only in the network. `guidance` is meaningless without a null token and
    must be left at 1.0 when eps_fn is given.

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
    # n_steps=1 MUST evaluate at t0. np.linspace(0, t0, 1) returns [0] -- only
    # the start point -- so the old expression queried the model at t=0, told
    # it the input was already clean, and returned
    #     z0 = (sqrt(abar[t0]) z - sqrt(1-abar[0]) eps) / sqrt(abar[0])
    #        ~= 0.99 z - 0.0064 eps
    # i.e. a near-identity, for ANY t0. That is a scale factor, not a denoise,
    # and it is why every 1-step measurement on record came out "flat in t":
    # sqrt(abar[t0]) only moves 0.9998 -> 0.9943 across t0=5..60, and
    # concentration is blind to a global scale. See the BUG note in the
    # docstring above -- those numbers do not measure Tweedie.
    # n_steps>=2 is unaffected (linspace already spans 0..t0), so every tuned
    # run in the repo is byte-identical across this fix.
    grid = (np.array([int(t0)]) if int(n_steps) <= 1 else
            np.unique(np.linspace(0, int(t0), int(n_steps)).astype(int))[::-1])
    if eps_fn is not None and guidance != 1.0:
        raise ValueError("guidance != 1 is meaningless with eps_fn (an "
                         "unconditional network has no null token to guide "
                         "away from)")
    null = None if y_z is None else torch.zeros_like(y_z)
    z0_prev = None

    for i, t_cur in enumerate(grid):
        ab_t = alpha_bar[int(t_cur)]
        tt = torch.full((z.shape[0],), int(t_cur), device=z.device,
                        dtype=torch.long)
        eps = (eps_fn(x_t, tt) if eps_fn is not None
               else model_eps(model, x_t, y_z, tt))
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


# --------------------------------------------------------------------------
# Rebuilding a trained denoiser
# --------------------------------------------------------------------------
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
    # Rebuild whatever the checkpoint was trained with. "kind" and "norm" are
    # both absent from pre-flag checkpoints, and those are all flat/GroupNorm,
    # so the defaults inside model_from_checkpoint keep existing checkpoints
    # loading exactly as before. This is the path --prior cond takes, i.e. the
    # one every ef and m32 run goes through.
    model = model_from_checkpoint(ck).to(device)
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
