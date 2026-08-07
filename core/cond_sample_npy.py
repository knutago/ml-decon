"""
cond_sample_npy.py

Run the conditional diffusion model AS A PURE GENERATIVE MODEL on the .npy
patch dataset -- no data-fidelity term, no PnP, no forward operator at all.

    x_T ~ N(0, I)
    for t = T-1 ... 0:
        eps  = eps_theta(x_t, y, t)                    # y = observed patch
        x0   = (x_t - sqrt(1-abar_t) eps) / sqrt(abar_t)
        x_{t-1} = sqrt(abar_{t-1}) x0
                  + sqrt(1 - abar_{t-1} - sigma^2) eps + sigma * z

This isolates what the MODEL knows from what the OPTIMIZATION contributes.
Reading the two side by side tells you where a deficiency lives:

  * sampler sharp, RED recon smooth  -> the optimization is the problem
    (the RED pull is too weak, or gradient descent cannot reach the high
    frequencies the data term leaves undetermined)
  * sampler also smooth              -> the model is the problem; no PnP
    scheme built on it will be sharper than this
  * sampler sharp but flux wrong     -> the model hallucinates plausible
    structure; it NEEDS the data term to stay honest (this is what the
    held-out m31bK50 transfer test showed: 1.9x flux error unassisted)

--eta 0 gives deterministic DDIM (repeatable, tends smoother); --eta 1 gives
full ancestral sampling (stochastic, sharper, mode-seeking). Sampling several
draws with --n-draws shows the model's posterior spread: where draws disagree,
the data genuinely does not constrain the answer.

Everything happens in the dataset's stored (normalized) domain -- the model's
native domain. Flux-space metrics are computed at the end by inverting
norm.json, since only physical flux is comparable to the truth.

Usage:
  python cond_sample_npy.py \
      --ckpt /home/alex/noir_ml/global/ml-decon/checkpoints_cond_diffusion_npy/best.pt \
      --data-dir /home/alex/noir_ml/global/ml-decon/data/m31bK50 \
      --split val --indices 0:8 --steps 250 --eta 1.0 --n-draws 3
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

from red_pnp_deconvolve import (TorchNorm, load_checkpoint, parse_indices,
                                show)
from train_conditional_diffusion import cosine_alpha_bar


@torch.no_grad()
def sample(model, y_z, alpha_bar, steps, eta, guidance, has_null,
           generator, clamp=None):
    """DDIM / ancestral reverse chain in the model's normalized domain."""
    device = y_z.device
    T = alpha_bar.shape[0]
    grid = np.unique(np.linspace(0, T - 1, steps).astype(int))[::-1]
    z = torch.randn(y_z.shape, device=device, generator=generator)
    null = torch.zeros_like(y_z)

    for i, t_cur in enumerate(grid):
        ab_t = alpha_bar[int(t_cur)]
        tt = torch.full((y_z.shape[0],), int(t_cur), device=device,
                        dtype=torch.long)
        eps = model(z, y_z, tt)
        if guidance != 1.0:
            if not has_null:
                raise ValueError("--guidance != 1 needs p_uncond > 0")
            eps = model(z, null, tt) * (1 - guidance) + eps * guidance
        z0 = (z - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
        if clamp is not None:
            z0 = z0.clamp(clamp[0], clamp[1])
        if i + 1 == len(grid):
            return z0
        ab_n = alpha_bar[int(grid[i + 1])]
        # DDIM sigma; eta=0 deterministic, eta=1 ancestral
        sigma = (eta * ((1 - ab_n) / (1 - ab_t)).sqrt()
                 * (1 - ab_t / ab_n).sqrt())
        dir_c = (1 - ab_n - sigma ** 2).clamp(min=0).sqrt()
        z = ab_n.sqrt() * z0 + dir_c * eps
        if eta > 0:
            z = z + sigma * torch.randn(z.shape, device=device,
                                        generator=generator)
    return z0


def concentration(img, peaks):
    """mean fraction of a 5x5 box's flux held by the central pixel."""
    vals = []
    for b, i, j in peaks:
        box = img[b, i - 2:i + 3, j - 2:j + 3]
        if box.size == 25 and box.sum() > 0:
            vals.append(box[2, 2] / box.sum())
    return float(np.mean(vals)) if vals else float("nan")


def find_peaks(truth, thresh_pct=99.5):
    peaks = []
    for b in range(truth.shape[0]):
        t = truth[b]
        cut = np.percentile(t, thresh_pct)
        for i in range(3, t.shape[0] - 3):
            for j in range(3, t.shape[1] - 3):
                if t[i, j] > cut and t[i, j] == t[i - 1:i + 2, j - 1:j + 2].max():
                    peaks.append((b, i, j))
    return peaks


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="/home/alex/noir_ml/global/ml-decon/"
                                     "checkpoints_cond_diffusion_npy/best.pt")
    p.add_argument("--data-dir", type=Path,
                   default=Path("/home/alex/noir_ml/global/ml-decon/data/m31bK50"))
    p.add_argument("--split", choices=["val", "train"], default="val")
    p.add_argument("--indices", default="0:8")
    p.add_argument("--steps", type=int, default=250,
                   help="reverse steps (subsampled from T)")
    p.add_argument("--eta", type=float, default=1.0,
                   help="0 = deterministic DDIM, 1 = full ancestral")
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--n-draws", type=int, default=3,
                   help="independent samples per patch; their spread is the "
                        "model's posterior uncertainty")
    p.add_argument("--clamp", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="clamp the x0 estimate in normalized units, e.g. "
                        "-0.5 2.0. Off by default so nothing is hidden.")
    p.add_argument("--weights", choices=["ema", "raw"], default="ema")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("cond_sample_npy_out"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, T, ck = load_checkpoint(args.ckpt, device, weights=args.weights)
    alpha_bar = cosine_alpha_bar(T).to(device)
    norms = ck["dataset_norm"] or json.loads(
        (args.data_dir / "norm.json").read_text())
    obs_norm, ideal_norm = TorchNorm(norms["observed"]), TorchNorm(norms["ideal"])
    has_null = float(ck.get("p_uncond") or 0.0) > 0.0

    observed = np.load(args.data_dir / f"{args.split}_observed.npy")
    ideal = np.load(args.data_dir / f"{args.split}_ideal.npy")
    idx = parse_indices(args.indices, len(observed))
    y_z = torch.from_numpy(observed[idx]).float().to(device)
    truth_z = torch.from_numpy(ideal[idx]).float().to(device)
    truth = ideal_norm.inverse(truth_z).cpu().numpy()[:, 0]
    y_phys = obs_norm.inverse(y_z).cpu().numpy()[:, 0]
    print(f"[data] {args.split}: {len(idx)} patches, {args.n_draws} draws each")
    print(f"[sample] steps={args.steps} eta={args.eta} guidance={args.guidance}")

    draws = []
    for d in range(args.n_draws):
        gen = torch.Generator(device=device).manual_seed(args.seed + d)
        z0 = sample(model, y_z, alpha_bar, args.steps, args.eta,
                    args.guidance, has_null, gen,
                    clamp=tuple(args.clamp) if args.clamp else None)
        x = ideal_norm.inverse(z0).cpu().numpy()[:, 0]
        draws.append(x)
        print(f"  draw {d}: flux/truth={x.sum()/truth.sum():.4f}  "
              f"max={x.max():.3f} (truth {truth.max():.3f})", flush=True)
    draws = np.stack(draws)                       # (D, B, H, W)
    mean = draws.mean(0)

    # ---- sharpness: how concentrated are point sources, vs truth ----
    peaks = find_peaks(truth)
    print(f"\n[sharpness] {len(peaks)} true point sources, central-pixel "
          f"concentration in a 5x5 box (higher = sharper)")
    rows = []
    for nm, img in ([("truth", truth), ("observed", y_phys)]
                    + [(f"sample draw {d}", draws[d]) for d in range(args.n_draws)]
                    + [("draw mean", mean)]):
        c = concentration(img, peaks)
        pk = float(np.mean([img[b, i, j] for b, i, j in peaks]))
        rmse = float(np.sqrt(np.mean((img[:, 4:-4, 4:-4]
                                      - truth[:, 4:-4, 4:-4]) ** 2)))
        fr = float(img[:, 4:-4, 4:-4].sum()
                   / truth[:, 4:-4, 4:-4].sum())
        rows.append({"image": nm, "concentration": round(c, 4),
                     "peak_at_sources": round(pk, 4),
                     "flux_ratio": round(fr, 4), "rmse": rmse})
        print(f"  {nm:<16} conc={c:.3f}  peak={pk:8.4f}  "
              f"flux/truth={fr:6.3f}  rmse={rmse:.4e}")

    with open(args.out_dir / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    np.save(args.out_dir / "draws.npy", draws.astype(np.float32))

    # ---- figure: observed | draws | draw-spread | truth ----
    nshow = min(len(idx), 6)
    ncol = 2 + args.n_draws + 1
    fig, ax = plt.subplots(nshow, ncol, figsize=(3.1 * ncol, 3.3 * nshow),
                           squeeze=False)
    for b in range(nshow):
        show(ax[b, 0], y_phys[b], f"observed (idx {idx[b]})")
        for d in range(args.n_draws):
            show(ax[b, 1 + d], draws[d, b], f"sample {d}")
        show(ax[b, 1 + args.n_draws], draws[:, b].std(0),
             "std across draws")
        show(ax[b, ncol - 1], truth[b], "ideal (truth)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "samples.png", dpi=125)
    plt.close(fig)
    print(f"\nWrote {args.out_dir}/samples.png, metrics.csv, draws.npy")


if __name__ == "__main__":
    main()
