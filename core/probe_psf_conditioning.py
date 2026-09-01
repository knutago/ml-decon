"""
probe_psf_conditioning.py

Go/no-go diagnostic for a PSF-conditional checkpoint: IS THE PSF CHANNEL
ACTUALLY USED, AND IS IT USED CORRECTLY?

Run this before spending a long solver sweep on any checkpoint from
train_psf_conditional_diffusion.py. It is ~a dozen forward passes and answers
in about a minute -- the same economics as the cos(g, x-truth) probe that
correctly predicted a 100-minute sweep's outcome.

What it measures
----------------
For each timestep on a grid, the model's one-step Tweedie estimate

    x0_hat = (x_t - sqrt(1-abar)*eps_theta(x_t, y, psf, t)) / sqrt(abar)

is computed three ways on the SAME x_t and the SAME observation y:

  RIGHT   the PSF that actually produced y (the dir's own psf.fits)
  WRONG   another PSF from the bank
  NULL    the all-zeros null token (only meaningful if p_uncond > 0)

Two numbers come out, and they answer different questions:

  SENSITIVITY  ||x0(right) - x0(wrong)|| / ||x0(right)||
      Does the channel change anything at all? Near zero means the network
      learned to ignore it -- usually because the training bank's PSFs were
      too similar to distinguish. gen_psf_bank_data.py's docstring has the
      measured floor: a +-2% size jitter moves the observation by 0.21 of the
      noise sigma, which is not a learnable signal.

  USEFULNESS   MSE(x0, ideal) with the right PSF vs with the wrong one
      Sensitivity alone is not enough: a model can be sensitive to the channel
      and still use it wrongly. The right PSF must score BETTER. If the two
      are equal the channel is decorative; if wrong scores better, something
      is mislabelled -- most likely a (patch, psf) pairing bug in the data
      generator.

The verdict line combines them. Only a checkpoint that is both sensitive AND
better-with-the-right-PSF is doing what the architecture claims.

Usage
-----
    python probe_psf_conditioning.py \
        --ckpt checkpoints_psf_cond/best.pt \
        --data-dir data/f555_jitter/holdout/var_005 \
        --wrong-psf data/f555_jitter/train/var_003/psf.fits

    # sweep every holdout dir, each probed against every other bank PSF:
    python probe_psf_conditioning.py --ckpt ... \
        --data-dir data/f555_jitter/holdout/* --wrong-psf data/f555_jitter/train/*/psf.fits

Prefer a HELD-OUT dir: on a PSF that was in training the model may be
recognising the beam from the image itself rather than reading the channel.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from train_psf_conditional_diffusion import (canonicalize_psf,
                                             cosine_alpha_bar, find_psf,
                                             load_psf_array,
                                             model_from_checkpoint,
                                             psf_to_channels)


def tweedie(model, x_t, y, psf, t, alpha_bar):
    """One-step posterior mean E[x_0 | x_t]. Exact, not an approximation."""
    ab = alpha_bar[t][:, None, None, None]
    eps = model(x_t, y, psf, t)
    return (x_t - (1 - ab).sqrt() * eps) / ab.sqrt()


def main():
    ap = argparse.ArgumentParser(
        description="Is the PSF channel used, and used correctly?")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, nargs="+", required=True,
                    help="dataset dir(s) to probe; prefer HELD-OUT ones.")
    ap.add_argument("--wrong-psf", type=Path, nargs="*",
                    help="PSF file(s) to substitute. Default: a 1.4x-rescaled "
                         "copy of the correct one, which the measured "
                         "identifiability table puts safely above the noise.")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--n-patches", type=int, default=32)
    ap.add_argument("--timesteps", type=int, nargs="*",
                    default=[5, 20, 50, 100, 200, 400])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = model_from_checkpoint(ckpt).to(device).eval()
    arch = ckpt["arch"]
    p_uncond = ckpt.get("p_uncond", 0.0)
    psf_size = arch["psf_size"]
    psf_repr = ckpt["args"].get("psf_repr", "log")
    alpha_bar = cosine_alpha_bar(ckpt["diffusion"]["timesteps"]).to(device)
    print(f"[ckpt] {args.ckpt}  arch={arch['kind']} psf_ch={arch['psf_ch']} "
          f"repr={psf_repr} cross_attn={arch['psf_cross_attn']} "
          f"p_uncond={p_uncond}")

    def prep(path_or_arr, scale=None):
        raw = (load_psf_array(Path(path_or_arr))
               if not isinstance(path_or_arr, np.ndarray) else path_or_arr)
        if scale is not None:
            from scipy.ndimage import affine_transform
            n = raw.shape[0]; c = (n - 1) / 2.0
            m = np.eye(2) / scale
            raw = affine_transform(raw, m, offset=np.array([c, c]) - m @ [c, c],
                                   order=3, mode="constant", cval=0.0)
            raw = np.clip(raw, 0, None)
        return torch.from_numpy(
            psf_to_channels(canonicalize_psf(raw, psf_size), psf_repr))

    for data_dir in args.data_dir:
        observed = np.load(data_dir / f"{args.split}_observed.npy")
        ideal = np.load(data_dir / f"{args.split}_ideal.npy")
        if len(observed) == 0:
            print(f"\n[{data_dir.name}] 0 patches in the {args.split} split, "
                  f"skipped")
            continue
        n = min(args.n_patches, len(observed))
        y = torch.from_numpy(observed[:n]).to(device)
        x0 = torch.from_numpy(ideal[:n]).to(device)

        right_path = find_psf(data_dir)
        right = prep(right_path)[None].repeat(n, 1, 1, 1).to(device)
        wrongs = []
        if args.wrong_psf:
            for p in args.wrong_psf:
                if Path(p).resolve() == right_path.resolve():
                    continue
                wrongs.append((Path(p).parent.name or Path(p).stem,
                               prep(p)[None].repeat(n, 1, 1, 1).to(device)))
        if not wrongs:
            wrongs = [("rescaled x1.4",
                       prep(load_psf_array(right_path), scale=1.4)[None]
                       .repeat(n, 1, 1, 1).to(device))]
        null = torch.zeros_like(right)

        print(f"\n[{data_dir.name}] {n} {args.split} patches, "
              f"correct PSF = {right_path.name}, "
              f"{len(wrongs)} wrong PSF(s)")
        print(f"  {'t':>5} {'sens(right,wrong)':>18} {'mse_right':>11} "
              f"{'mse_wrong':>11} {'mse_null':>11}  {'verdict':>9}")

        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        eps = torch.randn(x0.shape, generator=gen).to(device)
        agg = []
        for ti in args.timesteps:
            t = torch.full((n,), ti, device=device, dtype=torch.long)
            ab = alpha_bar[t][:, None, None, None]
            x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
            with torch.no_grad():
                xr = tweedie(model, x_t, y, right, t, alpha_bar)
                xw = torch.stack([tweedie(model, x_t, y, w, t, alpha_bar)
                                  for _, w in wrongs]).mean(0)
                xn = (tweedie(model, x_t, y, null, t, alpha_bar)
                      if p_uncond > 0 else None)
            sens = ((xr - xw).norm() / xr.norm()).item()
            mr = torch.mean((xr - x0) ** 2).item()
            mw = torch.mean((xw - x0) ** 2).item()
            mn = torch.mean((xn - x0) ** 2).item() if xn is not None else float("nan")
            # "used" needs BOTH: the channel must move the output, and the
            # correct PSF must move it in the right direction.
            v = ("ignored" if sens < 0.01 else
                 "used" if mr < mw * 0.98 else
                 "MISUSED" if mr > mw * 1.02 else "neutral")
            agg.append((sens, mr, mw, v))
            print(f"  {ti:5d} {sens:18.4f} {mr:11.5f} {mw:11.5f} "
                  f"{mn:11.5f}  {v:>9}")

        msens = float(np.mean([a[0] for a in agg]))
        better = sum(1 for a in agg if a[1] < a[2] * 0.98)
        worse = sum(1 for a in agg if a[1] > a[2] * 1.02)
        print(f"  -> mean sensitivity {msens:.4f}; the right PSF wins at "
              f"{better}/{len(agg)} timesteps, loses at {worse}")
        if msens < 0.01:
            print("  -> VERDICT: PSF CHANNEL IGNORED. The bank's PSFs are "
                  "probably too similar to be identifiable -- widen "
                  "--scale-range in gen_psf_bank_data.py and check the "
                  "measured floor in its docstring before retraining.")
        elif worse > better:
            print("  -> VERDICT: PSF CHANNEL MISUSED. The wrong PSF scores "
                  "better more often than the right one; suspect a "
                  "(patch, psf) pairing bug in the generator, or a psf_repr / "
                  "psf_size mismatch between training and this probe.")
        elif better == 0:
            print("  -> VERDICT: channel moves the output but does not help. "
                  "Necessary but not sufficient; do not read this as working.")
        else:
            print("  -> VERDICT: PSF conditioning is live and helping.")


if __name__ == "__main__":
    main()
