"""
cond_denoiser.py

Wrapper around a train_conditional_diffusion.py checkpoint so it can be used
as a denoiser inside the deconvolution loops (graikos_deconvolve.py).

Two call modes, and the distinction matters mathematically:

  denoise(x, y=<blurry>)   CONDITIONAL. Approximates E[x0 | x_t, y], i.e. the
                           POSTERIOR mean. Its residual x - D(x,y) is the
                           POSTERIOR score, which already contains the
                           likelihood -- pairing it with an explicit
                           ||y - Ax||^2 term double-counts the measurement.
                           Use it as an ANCHOR toward the model's answer,
                           not as a prior, and tune its weight empirically.

  denoise(x, y=None)       UNCONDITIONAL, via the null token (zeros). Only
                           valid if the checkpoint was trained with
                           classifier-free guidance (--p-uncond > 0); this
                           is checked and warned about. Its residual is a
                           genuine PRIOR score, safe to use in RED/Graikos
                           next to the data term.

NOTE the transform convention differs from flat_cnn_diffusion:
  * z = asinh(x / b) / A   with NO clipping and NO net_scale affine
  * the blurry conditioning is divided by flux_ratio BEFORE the asinh
    (so both channels live in the same sharp-flux asinh space)
"""

import numpy as np
import torch

from train_conditional_diffusion import ConditionalFlatCNN, cosine_alpha_bar


class ConditionalDenoiser:
    def __init__(self, ckpt_path, device=None, prefer_ema=True):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        arch = ck["arch"]
        # "norm" is absent from every pre---norm-flag checkpoint, and those are
        # all GroupNorm, so the default preserves existing behaviour exactly.
        self.model = ConditionalFlatCNN(
            channels=arch["channels"], t_dim=arch.get("t_dim", 128),
            norm=arch.get("norm", "group")
        ).to(self.device)
        state = None
        if prefer_ema:
            state = ck.get("ema_state")
        state = state or ck.get("model_state") or ck.get("model")
        self.model.load_state_dict(state)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        tr = ck["transform"]
        self.flux_ratio = float(tr["flux_ratio"])
        self.b = float(tr["asinh_b"])
        self.A = float(tr["asinh_A"])
        self.T = int(ck["diffusion"]["timesteps"])
        self.alpha_bar = cosine_alpha_bar(self.T).to(self.device)
        self.p_uncond = float(ck.get("p_uncond",
                                     ck.get("args", {}).get("p_uncond", 0.0)))
        if self.p_uncond <= 0:
            print("[cond] WARNING: this checkpoint was trained WITHOUT "
                  "classifier-free dropout (p_uncond=0). The unconditional "
                  "path (y=None) is NOT a valid prior -- the model never saw "
                  "the null token. Use conditional mode, or retrain with "
                  "--p-uncond 0.15.")

    # -- transform ------------------------------------------------------------
    def to_z(self, x):
        return torch.asinh(x / self.b) / self.A

    def to_flux(self, z):
        return self.b * torch.sinh(z * self.A)

    def t_for_sigma_z(self, sigma_z):
        """Timestep whose marginal SNR matches a z-space noise level."""
        ab = self.alpha_bar
        sig = ((1 - ab) / ab).sqrt()
        return int(torch.argmin((sig - float(sigma_z)).abs()).item())

    # -- standalone generation: blurry -> sharp -------------------------------
    @torch.no_grad()
    def sample(self, y_flux, steps=200, eta=0.0, seed=None, guidance=0.0,
               clip_z=True):
        """Generate a SHARP image from a blurry one by running the reverse
        chain conditioned on y -- i.e. draw from p(sharp | blurry).

        This uses the model as a standalone conditional generator: no PSF
        operator, no data-fidelity term, no optimization. Strided DDIM.

        eta=0 is deterministic DDIM; eta=1 recovers ancestral DDPM sampling
        (different draws = different posterior samples).

        guidance > 0 applies classifier-free guidance,
            eps = eps_uncond + w * (eps_cond - eps_uncond),
        which sharpens adherence to y. Requires a checkpoint trained with
        --p-uncond > 0.
        """
        if seed is not None:
            torch.manual_seed(seed)
        y = torch.from_numpy(np.asarray(y_flux, dtype=np.float32).copy()
                             )[None, None].to(self.device)
        y_z = self.to_z(y / self.flux_ratio)
        null = torch.zeros_like(y_z)

        grid = np.unique(np.linspace(0, self.T - 1, steps).astype(int))[::-1]
        z = torch.randn_like(y_z)                    # start from pure noise
        for i, t_cur in enumerate(grid):
            t_next = int(grid[i + 1]) if i + 1 < len(grid) else -1
            ab_t = self.alpha_bar[int(t_cur)]
            tt = torch.full((1,), int(t_cur), device=self.device,
                            dtype=torch.long)
            eps = self.model(z, y_z, tt)
            if guidance > 0:
                eps_u = self.model(z, null, tt)
                eps = eps_u + guidance * (eps - eps_u)
            z0 = (z - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            if clip_z:
                # keep the x0 estimate in a sane asinh range; without this
                # the early (high-t) estimates can blow up through sinh
                z0 = z0.clamp(-2.0, 4.0)
            if t_next < 0:
                z = z0
                break
            ab_n = self.alpha_bar[t_next]
            sigma = (eta * ((1 - ab_n) / (1 - ab_t)).sqrt()
                     * (1 - ab_t / ab_n).sqrt())
            dir_c = (1 - ab_n - sigma ** 2).clamp(min=0).sqrt()
            z = ab_n.sqrt() * z0 + dir_c * eps
            if eta > 0:
                z = z + sigma * torch.randn_like(z)
        return self.to_flux(z)[0, 0].cpu().numpy()

    # -- the denoiser ---------------------------------------------------------
    @torch.no_grad()
    def denoise(self, x_flux, t, y_flux=None, add_noise=True, seed=None):
        """x_flux: 2-D numpy (sharp flux units). y_flux: 2-D numpy blurry in
        ORIGINAL (un-divided) flux units, or None for the null token.
        Returns the x0 estimate as 2-D numpy in sharp flux units."""
        if seed is not None:
            torch.manual_seed(seed)
        t = int(np.clip(t, 0, self.T - 1))
        x = torch.from_numpy(np.asarray(x_flux, dtype=np.float32).copy()
                             )[None, None].to(self.device)
        z = self.to_z(x)

        if y_flux is None:
            y_z = torch.zeros_like(z)                       # null token
        else:
            y = torch.from_numpy(np.asarray(y_flux, dtype=np.float32).copy()
                                 )[None, None].to(self.device)
            y_z = self.to_z(y / self.flux_ratio)

        ab = self.alpha_bar[t]
        if add_noise:
            eps = torch.randn_like(z)
            z_t = ab.sqrt() * z + (1 - ab).sqrt() * eps
        else:
            z_t = z
        tt = torch.full((1,), t, device=self.device, dtype=torch.long)
        eps_hat = self.model(z_t, y_z, tt)
        z0 = (z_t - (1 - ab).sqrt() * eps_hat) / ab.sqrt()
        return self.to_flux(z0)[0, 0].cpu().numpy()
