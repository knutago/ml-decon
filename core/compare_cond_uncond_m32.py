"""
compare_cond_uncond_m32.py

Score ADMM reconstructions of the SAME m32 patches, with the SAME forward
operator, under a CONDITIONAL and an UNCONDITIONAL diffusion prior, plus the
two references that bound them: the blurry input, and the no-prior Wiener
solve that the ADMM x-update reduces to when the prior is removed.

Everything goes through solver_metrics.evaluate, so the numbers are the same
definitions used for every other solver in this project (16 px crop,
concentration at true source positions, floor_frac to be MATCHED not
minimized).

Three m32-specific readings that do not apply to the m31 runs:

  ||Az-b|| IS NOT TO BE MINIMIZED, and lower than the truth's is overfitting.
  Pushing the TRUTH itself through this forward operator leaves
  ||A*truth - b||/||b|| = 0.78 on these patches; fitting a global gain first
  only brings it to 0.47, and excluding the saturated pixels 0.41. The
  operator explains ~60% of the measurement, so any solver whose residual is
  far BELOW the truth's row is fitting model error, not signal. Read the
  ||Az-b|| column against the truth's value, not against zero.

  flux_ratio is not supposed to reach 1.0 either. The per-patch affine fit
  gives observed = 0.82 * (psf*ideal) + sky with a strikingly stable gain
  (0.80-0.84 on 7 of 8 patches; the eighth, patch 1034, drops to 0.41 because
  23 of its pixels are saturated at the 16-bit ceiling while its truth peaks
  87x above it). The fitted sky is NOT zero: +0.0001 to +0.052, i.e. M32's own
  smooth halo light, which the catalogue truth does not contain and the
  unit-gain, no-sky operator used here cannot absorb. That unmodelled flux has
  to go somewhere in every reconstruction.

  floor_frac's target is ~84%, not m31's 7%. m3201newK is a catalogue
  rendering -- only the listed sources, exact zeros everywhere else -- while
  both priors were trained on FIELDS whose sky is a populated noise floor.
  A prior that "fills in" the sky is doing what it was trained to do.

Run:
    python compare_cond_uncond_m32.py --runs m32_cmp/cond m32_cmp/flat
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/alex/noir_ml/mycode")
from red_pnp_deconvolve import TorchNorm, load_kernel, make_otf, parse_indices  # noqa: E402
import solver_metrics as sm  # noqa: E402

# observed = GAIN * (psf*ideal) + sky, from the per-patch affine fit. Median
# over the 8 comparison patches, excluding the saturation-wrecked patch 1034.
GAIN = 0.822


def detection_scores(img, truth, nsigma=3.0):
    """Completeness / spurious, with every method detected at the TRUTH's cut.

    benchmark_eval.py derives the threshold from EACH image's own sky, which
    silently rewards whichever method has the smoothest background -- the same
    per-image-threshold bias that once handed FISTA-L1 1.57 mag for free. Here
    the threshold is computed once, on the truth, and applied to every
    reconstruction, so the catalogues are built on one common cut.
    """
    from benchmark_eval import detect_and_measure, match_catalogues, sky_stats
    import benchmark_eval as be
    n_in = n_meas = n_rec = 0
    dmag = []
    for k in range(truth.shape[0]):
        sky, sig = sky_stats(truth[k])
        thr = sky + nsigma * sig
        # feed the same absolute cut to both by shifting nsigma to match: the
        # helper recomputes sky/sigma internally, so call it on the truth for
        # the input list and on a rescaled copy for the measured list.
        ip, ifx = detect_and_measure(truth[k], nsigma)
        s2, g2 = sky_stats(img[k])
        ns2 = (thr - s2) / max(g2, 1e-30)          # same absolute threshold
        mp, mfx = detect_and_measure(img[k], ns2)
        n_in += len(ip); n_meas += len(mp)
        pairs = match_catalogues(ip, ifx, mp, mfx) if len(ip) and len(mp) else []
        n_rec += len(pairs)
        for a, b_ in pairs:
            if ifx[a] > 0 and mfx[b_] > 0:
                dmag.append(-2.5 * np.log10(mfx[b_] / ifx[a]))
    dmag = np.asarray(dmag)
    return {"completeness": 100.0 * n_rec / max(n_in, 1),
            "spurious": 100.0 * (n_meas - n_rec) / max(n_meas, 1),
            "sig_mag": float(np.median(np.abs(dmag - np.median(dmag))) * 1.4826)
            if dmag.size else float("nan")}


def data_residual(x, b, K, H, W):
    """||A x - b|| / ||b||, the truth-free consistency number."""
    t = torch.from_numpy(np.ascontiguousarray(x))[:, None]
    Ax = torch.fft.irfft2(K * torch.fft.rfft2(t), s=(H, W))
    return float(torch.linalg.vector_norm(Ax - b) / torch.linalg.vector_norm(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/home/alex/noir_ml/global/ml-decon/"
                                          "data/m32_klong_nosky")
    ap.add_argument("--split", default="val")
    ap.add_argument("--indices", required=True)
    ap.add_argument("--psf-file", default="psf_klong_epsf.fits")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run directories (each holds <split>_recon.npy); "
                         "label with dir=LABEL to rename a row")
    ap.add_argument("--out", default="m32_cmp")
    ap.add_argument("--n-show", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    DT = torch.float64

    # ---- data, in flux, decoded with the norm the .npy files carry ----
    dd = Path(args.data_dir)
    n = json.loads((dd / "norm.json").read_text())
    obs_n, ide_n = TorchNorm(n["observed"]), TorchNorm(n["ideal"])
    o = np.load(dd / f"{args.split}_observed.npy")
    i = np.load(dd / f"{args.split}_ideal.npy")
    idx = parse_indices(args.indices, len(o))
    b = obs_n.inverse(torch.from_numpy(o[idx]).to(DT))
    truth = ide_n.inverse(torch.from_numpy(i[idx]).to(DT))
    B, _, H, W = b.shape
    K = make_otf(load_kernel(args.psf_file), (H, W), torch.device("cpu"))
    K2 = (K.conj() * K).real
    Kb = K.conj() * torch.fft.rfft2(b)

    truth_np = truth.numpy()[:, 0]
    peaks = sm.find_peaks(truth_np)
    print(f"[data] {dd.name} {args.split} {args.indices}: {B} patches, "
          f"{len(sm.interior_peaks(peaks, H, W))} interior sources")

    rows = [("observed (input)", b.numpy()[:, 0]),
            ("truth", truth_np)]

    # ---- no-prior reference: the x-update alone (Tikhonov/Wiener) ----
    # Swept over rho and reported at the ORACLE best interior rmse, i.e. the
    # most generous no-prior baseline available. Anything a prior adds has to
    # beat this row to have earned its runtime.
    best = None
    for rho in np.logspace(-6, 1, 36):
        w = torch.fft.irfft2(Kb / (K2 + rho), s=(H, W)).numpy()[:, 0]
        r = np.sqrt(((w[:, 16:H - 16, 16:W - 16]
                      - truth_np[:, 16:H - 16, 16:W - 16]) ** 2).mean())
        if best is None or r < best[0]:
            best = (r, rho, w)
    print(f"[wiener] best rho = {best[1]:.4g} (oracle-tuned on interior rmse)")
    rows.append((f"no prior: Wiener rho={best[1]:.3g}", best[2]))

    # ---- the runs ----
    for spec in args.runs:
        d, _, label = spec.partition("=")
        d = Path(d)
        arr = np.load(d / f"{args.split}_recon.npy")
        if arr.shape[0] != B:
            raise SystemExit(f"{d}: {arr.shape[0]} patches but {B} requested")
        cfg = json.loads((d / "config.json").read_text())
        if not label:
            label = ("uncond flat_cnn" if cfg.get("prior") == "flat"
                     else "cond diffusion")
            label += f"  [{d.name}]"
        rows.append((label, arr[:, 0]))

    # ---- table ----
    table, results = [], {}
    for name, img in rows:
        m = sm.evaluate(img, truth_np, peaks)
        m["data_resid"] = data_residual(img, b, K, H, W)
        m["flux_over_fwd"] = m["flux_ratio"] / GAIN
        m.update(detection_scores(np.asarray(img, np.float64), truth_np))
        table.append((name, m))
        results[name] = m

    cols = [("completeness", "compl%", "{:.1f}"),
            ("spurious", "spur%", "{:.1f}"),
            ("sig_mag", "sig_mag", "{:.3f}"),
            ("concentration", "conc", "{:.3f}"),
            ("flux_ratio", "flux/truth", "{:.3f}"),
            ("flux_over_fwd", "flux/fwd", "{:.3f}"),
            ("phot_median", "phot med", "{:.3f}"),
            ("phot_scatter", "phot scat", "{:.3f}"),
            ("interior_rmse", "int rmse", "{:.4f}"),
            ("floor_frac", "floor%", "{:.2f}"),
            ("data_resid", "||Az-b||", "{:.4f}")]
    w0 = max(len(nm) for nm, _ in table) + 2
    print("\n" + " " * w0 + "  ".join(f"{h:>9s}" for _, h, _ in cols))
    for name, m in table:
        print(f"{name:<{w0}s}" + "  ".join(
            f"{f.format(m[k]):>9s}" for k, _, f in cols))
    print(f"\n  flux/fwd = flux_ratio / {GAIN} (the forward model's own gain: "
          f"a data-consistent solve cannot reach flux/truth = 1)")
    print(f"  floor% target is the truth's {results['truth']['floor_frac']:.2f}"
          f"% -- match it, do not minimize it")
    print(f"  ||Az-b|| target is the TRUTH's {results['truth']['data_resid']:.4f}"
          f" -- the operator explains only ~60% of this measurement, so a "
          f"residual far below that row is fitting model error")
    print(f"  the observed row's flux_ratio is {results['observed (input)']['flux_ratio']:.3f}"
          f", above 1 because of the unmodelled sky pedestal")
    (out / "metrics.json").write_text(json.dumps(
        {k: v for k, v in results.items()}, indent=2, sort_keys=True))

    # ---- figure ----
    ns = min(args.n_show, B)
    fig, axes = plt.subplots(ns, len(rows), figsize=(2.1 * len(rows), 2.1 * ns),
                             squeeze=False)
    for r in range(ns):
        hi = np.percentile(truth_np[r], 99.9) or 1.0
        for c, (name, img) in enumerate(rows):
            a = axes[r][c]
            a.imshow(np.arcsinh(img[r] / (0.05 * hi)), cmap="magma",
                     vmin=0, vmax=np.arcsinh(1 / 0.05))
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(name.split("  [")[0], fontsize=7)
        axes[r][0].set_ylabel(f"patch {idx[r]}", fontsize=7)
    fig.suptitle("m32 (Klong) ADMM: conditional vs unconditional prior "
                 "— asinh stretch, common scale per row", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "compare.png", dpi=150)
    print(f"\nwrote {out}/compare.png and {out}/metrics.json")


if __name__ == "__main__":
    main()
