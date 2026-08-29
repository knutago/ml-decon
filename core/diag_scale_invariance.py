#!/usr/bin/env python3
"""
diag_scale_invariance.py

Does ConditionalFlatCNN preserve ABSOLUTE SCALE, or does GroupNorm divide it out?

Motivation
----------
FiLMConvBlock is Conv -> GroupNorm(8, C) -> FiLM -> SiLU (+residual).  GroupNorm
computes mean/var per-sample over (C/8, H, W).  Two consequences, both of which
would limit how broadly the model transfers:

  (1) SCALE.  A bright field and a faint field get renormalized toward the same
      statistics, so the network may be unable to tell them apart.  That is a
      candidate mechanism for the 1.9x flux error seen on held-out m31bK50.
  (2) SPATIAL EXTENT.  The statistics depend on H,W.  The net is otherwise fully
      convolutional (flat, no downsampling), so it *could* run on a full frame --
      but only if its behaviour does not change with input size.

Three tests
-----------
A. End-to-end scale response.  Scale the signal by c, measure how much the
   implied x0_hat scales.  Perfect model -> log-log slope 1.  Scale-blind model
   -> slope < 1.  NOTE: x_t itself carries c, so at LOW t the slope tends to 1
   even for a broken model; the informative regime is HIGH t, where the
   eps_hat error term is amplified by sqrt((1-abar)/abar).

B. Per-layer localisation (the decisive test).  Scale the WHOLE input (x_t and
   y jointly) by c, hook every GroupNorm, and compare activation std entering
   and leaving each norm.  If std_in scales with c but std_out does not, scale
   information dies at that layer.  Immune to the "input is out of
   distribution" objection because it measures the normalisation directly.

   Do NOT scale only the signal here: at t=500 the diffusion noise in x_t has
   std ~0.70 and a sparse star field's std is far smaller, so a 4x signal
   scaling moves the input std by only ~1.1x and nothing can be attributed.

C. Spatial-extent sensitivity.  Run one patch alone, then as the top-left
   quadrant of a 128x128 mosaic, and compare outputs over the central region
   that the 41px receptive field cannot see the seam from.  Any difference is
   attributable to non-local normalisation statistics.

Usage
-----
    /opt/conda/miniconda3/envs/py313/bin/python diag_scale_invariance.py
    ... --ckpt checkpoints_cond_diffusion/best.pt --n 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from train_conditional_diffusion import ConditionalFlatCNN, cosine_alpha_bar

DATA = Path("/home/alex/noir_ml/global/ml-decon/data/m31bK50")


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", {})
    # Checkpoints written before --norm existed are all GroupNorm.  A normless
    # checkpoint has no blocks.*.norm.* keys, so getting this wrong and loading
    # with strict=False would silently run a partly-random network -- hence
    # every load_state_dict below is strict.
    norm = arch.get("norm", "group")
    model = ConditionalFlatCNN(channels=arch.get("channels", 64),
                               t_dim=arch.get("t_dim", 128), norm=norm)
    state, which = ck.get("ema_state"), "ema"
    if state is None:
        state, which = ck["model_state"], "model"
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # ema_pytorch stores prefixed keys; fall back to the raw weights.
        model.load_state_dict(ck["model_state"])
        which = "model (ema_state keys did not match)"
    model.eval().to(device)
    print(f"loaded {ckpt_path}  epoch={ck.get('epoch')}  weights={which}")
    print(f"  arch={arch}  diffusion={ck.get('diffusion')}")
    if "norm" not in arch:
        print("  note: no 'norm' in arch -> assuming 'group' (pre-flag checkpoint)")
    return model, ck


def load_pairs(n: int, seed: int = 0):
    obs = np.load(DATA / "val_observed.npy", mmap_mode="r")
    ide = np.load(DATA / "val_ideal.npy", mmap_mode="r")
    idx = np.random.default_rng(seed).choice(len(obs), size=n, replace=False)
    idx.sort()
    y = torch.from_numpy(np.ascontiguousarray(obs[idx])).float()
    x0 = torch.from_numpy(np.ascontiguousarray(ide[idx])).float()
    return x0, y


def ls_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    """Least-squares k minimising ||a - k*b||: 'a is b scaled by k'."""
    a, b = a.flatten().double(), b.flatten().double()
    return float((a @ b) / (b @ b))


# ---------------------------------------------------------------------------
# Test A -- end-to-end scale response of x0_hat
# ---------------------------------------------------------------------------

def test_a(model, x0, y, alpha_bar, device, timesteps=(100, 500, 800, 950)):
    print("\n" + "=" * 74)
    print("TEST A: does the reconstruction honour absolute scale?")
    print("  k = least-squares ratio  x0_hat(c*signal) / x0_hat(signal)")
    print("  ideal k = c exactly  ->  log-log slope 1.0")
    print("=" * 74)

    x0, y = x0.to(device), y.to(device)
    eps = torch.randn(x0.shape, generator=torch.Generator(device="cpu").manual_seed(1234)).to(device)

    narrow = [0.8, 0.9, 1.0, 1.11, 1.25]
    wide = [0.25, 0.5, 1.0, 2.0, 4.0]

    for t_int in timesteps:
        ab = alpha_bar[t_int].to(device)
        s_ab, s_1mab = ab.sqrt(), (1 - ab).sqrt()
        t = torch.full((x0.shape[0],), t_int, device=device, dtype=torch.long)

        def x0_hat(c):
            xt = s_ab * (c * x0) + s_1mab * eps
            with torch.no_grad():
                eh = model(xt, c * y, t)
            return (xt - s_1mab * eh) / s_ab

        base = x0_hat(1.0)
        out = {}
        for label, grid in (("narrow", narrow), ("wide", wide)):
            ks, cs = [], []
            for c in grid:
                ks.append(ls_ratio(x0_hat(c), base))
                cs.append(c)
            slope = float(np.polyfit(np.log(cs), np.log(np.maximum(ks, 1e-8)), 1)[0])
            out[label] = (cs, ks, slope)

        print(f"\n  t = {t_int:4d}   (alpha_bar = {float(ab):.4f})")
        for label in ("narrow", "wide"):
            cs, ks, slope = out[label]
            body = "  ".join(f"c={c:<5g}k={k:7.3f}" for c, k in zip(cs, ks))
            print(f"    {label:6s} {body}")
            print(f"    {'':6s} log-log slope = {slope:.3f}"
                  f"   {'<-- scale preserved' if slope > 0.85 else ''}"
                  f"{'<-- SCALE LOST' if slope < 0.5 else ''}")


# ---------------------------------------------------------------------------
# Test B -- per-layer localisation
# ---------------------------------------------------------------------------

def test_b(model, x0, y, alpha_bar, device, t_int=500, c=4.0):
    """Scale the WHOLE network input by c, not just the signal component.

    Scaling only the signal is useless here: at t=500 the diffusion noise in
    x_t has std ~0.70 while a sparse star field's std is far smaller, so a 4x
    signal scaling moves the input std by ~1.1x and nothing downstream can be
    attributed.  Scaling (x_t, y) jointly makes the input std ratio exactly c
    by construction, so any collapse below c is the network's doing.
    """
    kinds = sorted({type(blk.norm).__name__ for blk in model.blocks})
    print("\n" + "=" * 74)
    print(f"TEST B: where does scale information die?  (t={t_int}, input x {c:g})")
    print("  ratio = std(activations at c) / std(activations at c=1)")
    print(f"  entering a GroupNorm we expect ~{c:g}; leaving it, ~1 means scale was divided out")
    print(f"  norm layers: {', '.join(kinds)}"
          + ("   <- --norm none: in and out ratios should be EQUAL at every row"
             if kinds == ["Identity"] else ""))
    print("=" * 74)

    x0, y = x0.to(device), y.to(device)
    ab = alpha_bar[t_int].to(device)
    s_ab, s_1mab = ab.sqrt(), (1 - ab).sqrt()
    eps = torch.randn(x0.shape, generator=torch.Generator(device="cpu").manual_seed(1234)).to(device)
    t = torch.full((x0.shape[0],), t_int, device=device, dtype=torch.long)

    rec: dict[str, tuple[float, float]] = {}

    def mk_hook(name):
        def hook(_mod, inp, outp):
            rec[name] = (float(inp[0].std()), float(outp.std()))
        return hook

    handles = [blk.norm.register_forward_hook(mk_hook(f"blocks.{i}.norm"))
               for i, blk in enumerate(model.blocks)]

    xt_base = s_ab * x0 + s_1mab * eps
    runs = {}
    for cc in (1.0, c):
        rec.clear()
        with torch.no_grad():
            outp = model(cc * xt_base, cc * y, t)   # scale the FULL input
        runs[cc] = (dict(rec), float(outp.std()))
    for h in handles:
        h.remove()
    print(f"\n  input std ratio (by construction): {c:g}")

    base, _ = runs[1.0]
    scaled, _ = runs[c]
    print(f"\n  {'layer':16s} {'std_in ratio':>14s} {'std_out ratio':>14s}   verdict")
    for name in base:
        ri = scaled[name][0] / max(base[name][0], 1e-12)
        ro = scaled[name][1] / max(base[name][1], 1e-12)
        if ri > 1.5 * ro and ro < 1.5:
            verdict = "scale divided out here"
        elif ro > 0.7 * c:
            verdict = "scale passes through"
        else:
            verdict = ""
        print(f"  {name:16s} {ri:14.3f} {ro:14.3f}   {verdict}")
    print(f"\n  network output std ratio: {runs[c][1] / max(runs[1.0][1], 1e-12):.3f}"
          f"  (a fully homogeneous net would give {c:g}; 1.0 would mean fully scale-blind)")
    print("  Whatever exceeds 1.0 arrives via the residual skip in FiLMConvBlock,")
    print("  which bypasses the norm -- that path is the only route absolute scale has.")


# ---------------------------------------------------------------------------
# Test C -- spatial-extent sensitivity
# ---------------------------------------------------------------------------

def test_c(model, x0, y, alpha_bar, device, t_int=500, margin=21, n_tiles=8):
    print("\n" + "=" * 74)
    print(f"TEST C: does behaviour change with input size?  (t={t_int})")
    print("  same patch run alone (64x64) vs as a quadrant of a 128x128 mosaic;")
    print(f"  compared over the central {64 - 2 * margin}x{64 - 2 * margin} region, which the 41px")
    print("  receptive field cannot see the mosaic seam from.")
    print("=" * 74)

    ab = alpha_bar[t_int].to(device)
    s_ab, s_1mab = ab.sqrt(), (1 - ab).sqrt()
    g = torch.Generator(device="cpu").manual_seed(1234)

    rels = []
    for k in range(n_tiles):
        q = x0[4 * k:4 * k + 4].to(device), y[4 * k:4 * k + 4].to(device)
        if q[0].shape[0] < 4:
            break
        eps = torch.randn(q[0].shape, generator=g).to(device)
        xt = s_ab * q[0] + s_1mab * eps
        yy = q[1]

        def mosaic(a):  # (4,1,64,64) -> (1,1,128,128)
            top = torch.cat([a[0], a[1]], dim=-1)
            bot = torch.cat([a[2], a[3]], dim=-1)
            return torch.cat([top, bot], dim=-2).unsqueeze(0)

        t1 = torch.full((4,), t_int, device=device, dtype=torch.long)
        t2 = torch.full((1,), t_int, device=device, dtype=torch.long)
        with torch.no_grad():
            alone = model(xt, yy, t1)[0, 0]
            big = model(mosaic(xt), mosaic(yy), t2)[0, 0, :64, :64]

        sl = slice(margin, 64 - margin)
        a_c, b_c = alone[sl, sl], big[sl, sl]
        rels.append(float((a_c - b_c).norm() / a_c.norm()))

    rels = np.array(rels)
    print(f"\n  relative difference over {len(rels)} tiles:"
          f"  median {np.median(rels):.4f}   max {rels.max():.4f}")
    if np.median(rels) < 0.01:
        print("  -> size-independent; the net can be applied to full frames as-is.")
    else:
        print("  -> behaviour DEPENDS on input size; full-frame inference would")
        print("     not reproduce patch-wise behaviour until the norm is changed.")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints_cond_diffusion/best.pt"))
    ap.add_argument("--n", type=int, default=64, help="number of val patches")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model, ck = load_model(args.ckpt, device)
    T = ck.get("diffusion", {}).get("timesteps", 1000)
    alpha_bar = cosine_alpha_bar(T)
    if isinstance(alpha_bar, tuple):
        alpha_bar = alpha_bar[0]

    x0, y = load_pairs(args.n, args.seed)
    print(f"data: {DATA}  n={args.n}  "
          f"x0 range [{float(x0.min()):.4f}, {float(x0.max()):.4f}]  "
          f"y range [{float(y.min()):.4f}, {float(y.max()):.4f}]")

    test_a(model, x0, y, alpha_bar, device)
    test_b(model, x0, y, alpha_bar, device)
    test_c(model, x0, y, alpha_bar, device)


if __name__ == "__main__":
    main()
