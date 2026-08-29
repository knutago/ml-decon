"""Score the m32_32patch runs through admm_sweep_m32's exact metric path.

Same PSF, same norm.json, same find_peaks/evaluate/detection_scores, so these
rows are directly comparable to sweep_admm/sweep_metrics.json -- which was run
on 8 patches, where this is 32.
"""
import json
from pathlib import Path

import numpy as np
import torch

import solver_metrics as sm
from red_pnp_deconvolve import TorchNorm, make_otf, load_kernel, parse_indices
from compare_cond_uncond_m32 import detection_scores

DATA = Path("/home/alex/noir_ml/global/ml-decon/data/m32_klong_nosky")
PSF = "psf_klong_epsf.fits"
IDX = ",".join(str(23 + 47 * k) for k in range(32))
OUT = Path("m32_32patch")
TAGS = ["ds4_eta1_rho0.03", "ds20_eta1"]

n = json.loads((DATA / "norm.json").read_text())
idx = parse_indices(IDX, 10 ** 9)
o = np.load(DATA / "val_observed.npy")[idx]
i = np.load(DATA / "val_ideal.npy")[idx]
b = TorchNorm(n["observed"]).inverse(torch.from_numpy(o).double())
truth = TorchNorm(n["ideal"]).inverse(torch.from_numpy(i).double())
B, _, H, W = b.shape
K = make_otf(load_kernel(PSF), (H, W), torch.device("cpu"))
tnp = truth.numpy()[:, 0]
peaks = sm.find_peaks(tnp)


def resid(a):
    t = torch.from_numpy(np.ascontiguousarray(a))[:, None].double()
    Ax = torch.fft.irfft2(K * torch.fft.rfft2(t), s=(H, W))
    return float(torch.linalg.vector_norm(Ax - b)
                 / torch.linalg.vector_norm(b))


truth_resid = resid(tnp)
rows = []
for tag in TAGS:
    f = OUT / tag / "val_recon.npy"
    if not f.exists():
        print(f"[skip] {tag}: no val_recon.npy")
        continue
    img = np.load(f)[:, 0]
    m = sm.evaluate(img, tnp, peaks)
    m.update(detection_scores(np.asarray(img, np.float64), tnp))
    m["resid"] = resid(img)
    m["dresid"] = abs(m["resid"] - truth_resid)
    cfg = json.loads((OUT / tag / "config.json").read_text())
    m["tag"] = tag
    m["cost"] = cfg["iters"] * cfg["denoise_steps"]
    rows.append(m)

hdr = (f"{'config':<18}{'resid':>8}{'|d|':>7}{'compl%':>8}{'spur%':>7}"
       f"{'sig_mag':>8}{'conc':>7}{'flux':>7}{'phot sct':>9}"
       f"{'int rmse':>9}{'calls':>7}")
print(f"\n  32 patches. truth's own residual = {truth_resid:.4f}  <- the "
      f"target for 'resid'; |d| is the distance to it\n")
print(hdr)
print("-" * len(hdr))
for r in sorted(rows, key=lambda r: r["dresid"]):
    print(f"{r['tag']:<18}{r['resid']:>8.4f}{r['dresid']:>7.4f}"
          f"{r['completeness']:>8.1f}{r['spurious']:>7.1f}"
          f"{r['sig_mag']:>8.3f}{r['concentration']:>7.3f}"
          f"{r['flux_ratio']:>7.3f}{r['phot_scatter']:>9.3f}"
          f"{r['interior_rmse']:>9.4f}{r['cost']:>7d}")
json.dump(rows, open(OUT / "metrics32.json", "w"), indent=2, default=float)
print(f"\nwrote {OUT / 'metrics32.json'}")
