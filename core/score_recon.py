"""
score_recon.py

Score any number of reconstruction directories through ONE metric path, so
solvers written in different scripts are comparable. A directory qualifies if
it contains {split}_recon.npy in admm_diffusion_deconvolve.py's convention
(B, 1, H, W) float64 in flux units.

This is admm_sweep_m32.py's scoring block lifted out so it is not welded to
that sweep's config list -- the ADMM sweep dirs and the DPS dirs go through the
same detection cut, the same photometry and the same operator.

    python score_recon.py dps_m32/* sweep_admm/ds4_eta1_rho0.03 sweep_admm/ds8_eta1

Columns
  resid    ||A x - b|| / ||b||, mean over patches
  |delta|  distance to the TRUTH's own residual through the same operator.
           That is the target: the truth leaves 0.78 on these patches, so a
           smaller resid means forward-model error is being fitted, not signal.
  compl%   completeness at the TRUTH's detection cut, applied to every method
  spur%    spurious fraction at that same cut
  phot sct per-star photometric scatter, over MATCHED detections only -- it is
           coupled to completeness by selection, so read the two together
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/alex/noir_ml/mycode")
from red_pnp_deconvolve import (TorchNorm, load_kernel, make_otf,
                                parse_indices, sig_std)
import solver_metrics as sm
from compare_cond_uncond_m32 import detection_scores

DATA = "/home/alex/noir_ml/global/ml-decon/data/m32_klong_nosky"
PSF = "psf_klong_epsf.fits"
IDX = "94,282,470,658,846,1034,1222,1410"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--psf-file", default=PSF)
    ap.add_argument("--indices", default=IDX)
    ap.add_argument("--split", default="val")
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    D = Path(args.data_dir)
    n = json.loads((D / "norm.json").read_text())
    idx = parse_indices(args.indices, 10 ** 9)
    o = np.load(D / f"{args.split}_observed.npy")[idx]
    i = np.load(D / f"{args.split}_ideal.npy")[idx]
    b = TorchNorm(n["observed"]).inverse(torch.from_numpy(o).double())
    truth = TorchNorm(n["ideal"]).inverse(torch.from_numpy(i).double())
    B, _, H, W = b.shape
    K = make_otf(load_kernel(args.psf_file), (H, W), torch.device("cpu"))
    tnp = truth.numpy()[:, 0]
    peaks = sm.find_peaks(tnp)
    bn = torch.linalg.vector_norm(b.reshape(B, -1), dim=1)

    def resid(a):
        t = torch.from_numpy(np.ascontiguousarray(a))[:, None].double()
        Ax = args.gain * torch.fft.irfft2(K * torch.fft.rfft2(t), s=(H, W))
        return float((torch.linalg.vector_norm((Ax - b).reshape(B, -1), dim=1)
                      / bn).mean())

    truth_resid = resid(tnp)

    # The background-floor columns. Quote them on the patches whose TRUTH has
    # an exactly-zero background: pooling those with the crowded patches
    # (where the truth's own std is 0.005) averages two opposite regimes and
    # hides the effect entirely.
    flat = [k for k in range(len(tnp)) if sig_std(tnp[k]) <= 0]
    if not flat:
        flat = list(range(len(tnp)))
        print("[floor] no patch has an exactly-zero truth background; "
              "floor columns are pooled over all patches")

    def floor_cols(a):
        return (float(np.mean([sig_std(a[k]) for k in flat])),
                100.0 * float(np.mean([(a[k] == 0).mean() for k in flat])))

    t_bkg, t_zero = floor_cols(tnp)

    rows = []
    for d in args.dirs:
        f = d / f"{args.split}_recon.npy"
        if not f.exists():
            print(f"[skip] {d}: no {args.split}_recon.npy")
            continue
        img = np.load(f)[:, 0]
        m = sm.evaluate(img, tnp, peaks)
        m.update(detection_scores(np.asarray(img, np.float64), tnp))
        m["resid"] = resid(img)
        m["dresid"] = abs(m["resid"] - truth_resid)
        m["bkg0"], m["zero_pct"] = floor_cols(img)
        m["tag"] = str(d)
        rows.append(m)

    if not rows:
        raise SystemExit("nothing to score")

    w = max(max(len(r["tag"]) for r in rows), 24)
    hdr = (f"{'run':<{w}}{'resid':>8}{'|delta|':>9}{'compl%':>8}{'spur%':>7}"
           f"{'sig_mag':>8}{'conc':>7}{'flux':>7}{'phot sct':>9}{'int rmse':>9}"
           f"{'bkg0':>9}{'zero%':>7}")
    print(f"\n  truth's own residual = {truth_resid:.4f}")
    print(f"  truth's own floor over the {len(flat)} flat-background patches: "
          f"bkg0 = {t_bkg:.5f}, zero% = {t_zero:.1f}\n")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: -r["completeness"]):
        print(f"{r['tag']:<{w}}{r['resid']:>8.4f}{r['dresid']:>9.4f}"
              f"{r['completeness']:>8.1f}{r['spurious']:>7.1f}"
              f"{r['sig_mag']:>8.3f}{r['concentration']:>7.3f}"
              f"{r['flux_ratio']:>7.3f}{r['phot_scatter']:>9.3f}"
              f"{r['interior_rmse']:>9.4f}{r['bkg0']:>9.5f}{r['zero_pct']:>7.1f}")
    print("\n  sorted by completeness. 'resid' is scored against the truth's "
          f"{truth_resid:.4f}, not against zero.")
    print(f"  bkg0/zero% are the background floor on the flat-background "
          f"patches only; the target is {t_bkg:.5f} / {t_zero:.1f}%.")
    if args.json_out:
        json.dump(rows, open(args.json_out, "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
