"""
flat_prior.py

Adapter that lets the UNCONDITIONAL flat-CNN diffusion prior
(flat_cnn_diffusion.FlatCNN, checkpoints/flat_cnn_stars_v3*.pt) be plugged
into admm_diffusion_deconvolve.py's z-update in place of the conditional
prior, so that a cond-vs-uncond comparison changes the NETWORK and nothing
else: same exact Fourier x-update, same rho schedule, same DDIM chain
(tweedie_chain, reached through its eps_fn hook), same metrics, same outputs.

Three things have to be reconciled, and each one is a place a silent factor
can hide:

1. FLUX DOMAIN.  The conditional checkpoints carry a `dataset_norm` fitted to
   the ml-decon dataset they were trained on; the flat checkpoint carries an
   AsinhTransform (b, A, z_ceil, net_scale) fitted to mycode/patches, which is
   m31bK100 min-max normalized as a whole frame. Those are different flux
   scales, so driving the flat prior with ml-decon flux needs a scale s:

       net = T.to_net(x_flux * s),      x_flux = T.net_to_flux(net) / s

   Anchoring s is NOT a detail. flat_cnn's A is asinh(p99.9(BLURRY)/b), and in
   the mycode pipeline blurry = 22.8 * (psf*sharp) + sky, whereas ml-decon's
   gen_data.py puts observed and ideal on ONE scale (blurry/sharp ~ 0.8). So a
   scale fitted from the observed bright end -- the natural-looking choice, and
   what --match-flux does on the conditional path -- lands ~30x too high for a
   prior whose input is a SHARP image. m32_prior_domain_check.py prints the
   candidates; the defensible truth-free one is mean flux (observed/gain
   matched to the training sharp mean), which agrees with the oracle
   ideal-p99.9 anchor to 25%.

2. NOISE LEVEL.  ADMM's z-update denoises at sigma = sqrt(lam/rho). On the
   conditional path that number is read in the normalized [0,1] domain the
   network trains in. The flat network trains in NET space, x = z/net_scale - 1,
   which is 1/net_scale = 5x wider than raw asinh z, so the same ADMM sigma is
   5x larger there:  sigma_model = sigma_z / net_scale. Skipping that
   conversion picks timesteps ~5x too early (the mistake
   AsinhTransform.sigma_flux_to_model exists to prevent).

3. SCHEDULE.  The conditional checkpoints use a COSINE alpha_bar; the flat
   checkpoint was trained on a LINEAR beta schedule (1e-4 -> 0.02, T=1000).
   sigma(t) = sqrt((1-abar_t)/abar_t) therefore means different things in the
   two: at t=75 the cosine schedule sits at sigma 0.35 and the linear one at
   0.26. Compare t_max between the two arms only after mapping through sigma;
   `t_for_sigma` on each arm's own sigma table is what makes the two runs
   equivalent, not equal integer t.
"""
import json
import math

import numpy as np
import torch

from astro_transforms import AsinhTransform
from flat_cnn_diffusion import FlatCNN


def linear_alpha_bar(T: int, dtype=torch.float64, device=None) -> torch.Tensor:
    """The schedule flat_cnn_diffusion.Schedule trains with (betas 1e-4..0.02).

    Built in float64 to match everything outside the network; the training-time
    version is float32 but the cumprod is the same quantity.
    """
    betas = torch.linspace(1e-4, 0.02, int(T), dtype=dtype, device=device)
    return torch.cumprod(1.0 - betas, dim=0)


class FlatPrior:
    """Unconditional diffusion denoiser, flux in -> flux out.

    Mirrors admm_diffusion_deconvolve.diffusion_denoise's contract exactly:
        z_flux, t_used = prior.denoise(v_flux, sigma_z, n_steps, ...)
    """

    def __init__(self, ckpt_path, transform_json=None, flux_scale=1.0,
                 device=None, weights="ema", dtype=torch.float32,
                 verbose=True):
        self.device = device or torch.device("cpu")
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        cfg = ck["config"]
        dil = tuple(cfg.get("dilations") or (1, 2, 3, 4, 4, 3, 2, 1))
        self.model = FlatCNN(1, cfg["base"], dil, 128).to(self.device)
        state_key = "ema" if (weights == "ema" and ck.get("ema")) else "model"
        self.model.load_state_dict(ck[state_key])
        self.model.eval().to(dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)

        # The transform must be the one this checkpoint trained with. The JSON
        # beside it is the same object; assert_matches fires if they drifted
        # (e.g. a z_ceil=1.05 JSON against a z_ceil=1.45 checkpoint, which
        # would silently clip the top of the flux range).
        d = (json.load(open(transform_json)) if transform_json
             else cfg["transform"])
        self.T = AsinhTransform.from_dict(d)
        if transform_json and "transform" in cfg:
            self.T.assert_matches(cfg["transform"])
        if self.T.net_scale is None:
            raise SystemExit(
                f"{ckpt_path}: transform has net_scale=None (a legacy raw-z "
                "checkpoint). The sigma conversion below assumes the net-space "
                "affine; use a v2/v3 checkpoint.")

        self.s = float(flux_scale)
        self.timesteps = int(cfg["timesteps"])
        self.alpha_bar = linear_alpha_bar(self.timesteps, device=self.device)
        self.sigmas = ((1.0 - self.alpha_bar) / self.alpha_bar).sqrt()
        self.net_clamp = (self.T.net_floor, self.T.net_ceil)
        if verbose:
            print(f"[ckpt] {ckpt_path}  weights={state_key}  T={self.timesteps} "
                  f" base={cfg['base']}  dilations={len(dil)} blocks  "
                  f"net_dtype={str(dtype).replace('torch.', '')}  "
                  f"identity_frac={cfg.get('identity_frac')}  "
                  f"low_t_max={cfg.get('low_t_max')}")
            print(f"[flat] {self.T}")
            print(f"[flat] flux scale s={self.s:.6g}  "
                  f"(prior sees x_flux * s; flux ceiling in ML-DECON units = "
                  f"{self.T.to_flux(self.T.z_ceil) / self.s:.6g})")
            print(f"[flat] schedule LINEAR betas: sigma(t=1)={self.sigmas[1]:.4f} "
                  f"sigma(75)={self.sigmas[75]:.4f} "
                  f"sigma(200)={self.sigmas[200]:.4f} "
                  f"sigma(T-1)={self.sigmas[-1]:.2f}")

    # -- the eps hook tweedie_chain calls --------------------------------
    def _eps(self, x_t, tt):
        md = next(self.model.parameters()).dtype
        return self.model(x_t.to(md), tt).to(x_t.dtype)

    def t_for_sigma_z(self, sigma_z, t_max, t_min=1):
        """ADMM sigma (asinh-z units) -> timestep on THIS schedule."""
        sigma_model = float(sigma_z) / self.T.net_scale
        t = int(torch.argmin((self.sigmas - sigma_model).abs()).item())
        return max(int(t_min), min(t, int(t_max))), sigma_model

    @torch.no_grad()
    def denoise(self, v_flux, sigma_z, n_steps, t_max=75, t_min=1, eta=0.0,
                generator=None, renoise=False, trace=None, inject_scale=1.0,
                clamp=None):
        from red_pnp_deconvolve import tweedie_chain
        t0, _ = self.t_for_sigma_z(sigma_z, t_max, t_min)
        net = self.T.to_net(v_flux * self.s)
        out = tweedie_chain(self.model, net, None, t0, n_steps, self.alpha_bar,
                            guidance=1.0, has_null=False, eta=eta,
                            generator=generator,
                            clamp=self.net_clamp if clamp is None else clamp,
                            renoise=renoise, trace=trace,
                            inject_scale=inject_scale, eps_fn=self._eps)
        return self.T.net_to_flux(out) / self.s, t0


class CondM31Prior:
    """CONTROL arm: a CONDITIONAL prior trained on the SAME m31 patches as the
    unconditional one (mycode/checkpoints_cond_diffusion, an old-format
    checkpoint carrying `transform` = {flux_ratio, asinh_b, asinh_A}).

    Why it exists: the m32-trained conditional and the m31-trained
    unconditional differ in TWO ways at once -- conditioning, and whether the
    prior ever saw this field. Comparing them alone cannot attribute a win to
    either. This arm holds the training data fixed against FlatPrior and
    changes only the conditioning, so the two comparisons together separate the
    factors.

    Its transform convention differs from flat_cnn's (see cond_denoiser.py):
    raw asinh z, no clipping, no net-space affine, and the CONDITIONING is
    divided by the blurry/sharp flux ratio before the asinh so both channels
    live in one space. On m31 that divisor is the pipeline's 21.07; on a
    dataset where gen_data already put observed and ideal on one scale the
    analogous divisor is that dataset's forward gain.
    """

    def __init__(self, ckpt_path, flux_scale=1.0, cond_divisor=None,
                 device=None, weights="ema", dtype=torch.float32, verbose=True):
        from train_conditional_diffusion import (cosine_alpha_bar,
                                                 model_from_checkpoint)
        self.device = device or torch.device("cpu")
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if "transform" not in ck:
            raise SystemExit(f"{ckpt_path}: no 'transform' -- this is a "
                             "dataset_norm checkpoint; run it with --prior cond")
        # arch (both "kind" and "norm") must come from the checkpoint: a net
        # trained with --norm none has nn.Identity where GroupNorm's affine
        # parameters would be, and a unet shares almost no keys with the flat
        # stack, so rebuilding the wrong variant fails the strict load. Absent
        # means flat/group -- i.e. every pre-flag checkpoint.
        self.model = model_from_checkpoint(ck).to(self.device)
        state = (ck.get("ema_state") if weights == "ema" else None) \
            or ck.get("model_state") or ck.get("model")
        self.model.load_state_dict(state)
        self.model.eval().to(dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)
        tr = ck["transform"]
        self.b, self.A = float(tr["asinh_b"]), float(tr["asinh_A"])
        self.flux_ratio = float(tr["flux_ratio"])
        self.cond_divisor = float(cond_divisor if cond_divisor is not None
                                  else self.flux_ratio)
        self.s = float(flux_scale)
        self.timesteps = int(ck["diffusion"]["timesteps"])
        self.alpha_bar = cosine_alpha_bar(self.timesteps,
                                          dtype=torch.float64).to(self.device)
        self.sigmas = ((1.0 - self.alpha_bar) / self.alpha_bar).sqrt()
        self.p_uncond = float(ck.get("p_uncond") or 0.0)
        self._y_z = None
        if verbose:
            print(f"[ckpt] {ckpt_path}  weights={'ema' if weights == 'ema' else 'raw'}"
                  f"  T={self.timesteps}  channels={arch['channels']}  "
                  f"p_uncond={self.p_uncond}")
            print(f"[c31]  asinh b={self.b:.4g} A={self.A:.4g} (no clip, no net "
                  f"affine)  training flux_ratio={self.flux_ratio:.2f}  "
                  f"conditioning divisor={self.cond_divisor:.3f}")
            print(f"[c31]  flux scale s={self.s:.6g}")

    def to_z(self, x):
        return torch.asinh(x / self.b) / self.A

    def to_flux(self, z):
        return self.b * torch.sinh(z * self.A)

    def set_conditioning(self, b_flux):
        """The measurement, in the SAME z-space as the signal channel."""
        self._y_z = self.to_z(b_flux * self.s / self.cond_divisor)

    def t_for_sigma_z(self, sigma_z, t_max, t_min=1):
        t = int(torch.argmin((self.sigmas - float(sigma_z)).abs()).item())
        return max(int(t_min), min(t, int(t_max))), float(sigma_z)

    @torch.no_grad()
    def denoise(self, v_flux, sigma_z, n_steps, t_max=75, t_min=1, eta=0.0,
                generator=None, renoise=False, trace=None, inject_scale=1.0,
                clamp=None):
        from red_pnp_deconvolve import tweedie_chain
        if self._y_z is None:
            raise RuntimeError("call set_conditioning(b) first")
        t0, _ = self.t_for_sigma_z(sigma_z, t_max, t_min)
        z = self.to_z(v_flux * self.s)
        out = tweedie_chain(self.model, z, self._y_z, t0, n_steps,
                            self.alpha_bar, guidance=1.0,
                            has_null=self.p_uncond > 0, eta=eta,
                            generator=generator,
                            clamp=(-2.0, 4.0) if clamp is None else clamp,
                            renoise=renoise, trace=trace,
                            inject_scale=inject_scale)
        return self.to_flux(out) / self.s, t0


def mean_flux_scale(observed_flux_mean, gain, train_sharp_mean, verbose=True):
    """The truth-free anchor: match mean surface brightness.

    observed/gain estimates the SHARP-domain mean of the target field (a
    unit-sum PSF preserves total flux), and the prior's own training mean is
    known, so the ratio is the scale that puts the field at the surface
    brightness the prior was trained at. Unlike a percentile anchor it is not
    destroyed by saturation, which censors exactly the bright tail a
    percentile anchor keys on (Klong is hard-clipped at 16 bits).

    Feed it a FIELD-level mean. A mean over a handful of patches is not a
    property of the field: it swings by an order of magnitude depending on
    whether a bright star is in frame.
    """
    m = float(np.asarray(observed_flux_mean).mean()) / float(gain)
    s = float(train_sharp_mean) / max(m, 1e-30)
    if verbose:
        print(f"[flat] mean-flux anchor: observed mean/gain = {m:.6g} vs "
              f"training sharp mean {train_sharp_mean:.6g}  ->  s = {s:.6g}")
    return s
