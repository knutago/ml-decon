"""
admm_sweep_m32.py

Sweep the ADMM solver settings and report, for each, BOTH the data-fidelity
residual and the truth-based accuracy — because on this problem they do not
point the same way.

  ||A z - b|| / ||b|| is what the solver is minimizing, but its target is NOT
  zero. Pushing the TRUTH through this forward operator leaves 0.7797 on these
  patches; a global-gain fit only brings it to 0.47 and dropping the saturated
  pixels 0.41. The operator explains ~60% of the measurement, so a setting that
  drives the residual far below the truth's row is fitting forward-model error,
  not signal — the no-prior Wiener solve reaches 0.12 and rings visibly.

  So the column to read is |resid - 0.7797|, not resid. The accuracy columns
  (completeness, photometric scatter, concentration) are there to check whether
  that reasoning actually holds, rather than assuming it.

Runs are launched in a small process pool; each is an independent
admm_diffusion_deconvolve invocation, so a crashed config costs only itself.

Run:
    python admm_sweep_m32.py                 # the full grid
    python admm_sweep_m32.py --dry-run       # list the configs only
    python admm_sweep_m32.py --only ds1,ds4  # a subset by tag
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/alex/noir_ml/mycode")
from red_pnp_deconvolve import TorchNorm, load_kernel, make_otf, parse_indices  # noqa: E402
import solver_metrics as sm  # noqa: E402
from compare_cond_uncond_m32 import detection_scores  # noqa: E402

PY = "/opt/conda/miniconda3/envs/py313/bin/python"
CKPT = ("/home/alex/noir_ml/global/ml-decon/"
        "checkpoints_cond_diffusion_npy_m32/best.pt")
DATA = "/home/alex/noir_ml/global/ml-decon/data/m32_klong_nosky"
PSF = "psf_klong_epsf.fits"
IDX = "94,282,470,658,846,1034,1222,1410"
OUT = Path("sweep_admm")

BASE = dict(iters=200, denoise_steps=4, rho=0.1, rho_scale=1.05, lam=0.01,
            eta=0.0, t_min=1, t_max=75, renoise=True)


def configs():
    """(tag, overrides). The paper's prox is a SINGLE denoiser call, so the
    denoise-steps axis starts at 1; 4 is what every run this session used and
    20 is the m31 best-known setting."""
    c = []
    for ds in (1, 4, 20):
        c.append((f"ds{ds}", dict(denoise_steps=ds)))
    # rho at the two cheap chain lengths. rho sets how hard the split is pulled
    # toward the prior; it also sets sigma = sqrt(lam/rho) and therefore t.
    for ds in (1, 4):
        for rho in (0.03, 0.3, 1.0):
            c.append((f"ds{ds}_rho{rho}", dict(denoise_steps=ds, rho=rho)))
    # lam moves sigma without moving the data weighting
    for lam in (0.003, 0.03):
        c.append((f"ds1_lam{lam}", dict(denoise_steps=1, lam=lam)))
    # flat rho (no annealing) vs the compounding default
    c.append(("ds1_flatrho", dict(denoise_steps=1, rho_scale=1.0)))
    c.append(("ds4_flatrho", dict(denoise_steps=4, rho_scale=1.0)))
    # entry noise level
    for tm in (20, 150):
        c.append((f"ds1_tmax{tm}", dict(denoise_steps=1, t_max=tm)))
    # stochastic chain
    c.append(("ds4_eta1", dict(denoise_steps=4, eta=1.0)))
    # --- stage 2: eta was the only axis that improved accuracy, so follow it.
    # Ancestral noise in the chain (eta=1) beat the deterministic DDIM path on
    # every accuracy column at ds=4, which is the OPPOSITE of the m31 result
    # that made eta=0 the default. Pin down whether it is eta alone, and
    # whether it stacks with a flat rho or a longer chain.
    c.append(("ds4_eta0.5", dict(denoise_steps=4, eta=0.5)))
    c.append(("ds4_eta1_flat", dict(denoise_steps=4, eta=1.0, rho_scale=1.0)))
    c.append(("ds4_eta1_rho0.03", dict(denoise_steps=4, eta=1.0, rho=0.03)))
    c.append(("ds8_eta1", dict(denoise_steps=8, eta=1.0)))
    c.append(("ds20_eta1", dict(denoise_steps=20, eta=1.0)))
    return c


def run_one(tag, over):
    d = OUT / tag
    if (d / "val_recon.npy").exists():
        return tag, "cached"
    p = dict(BASE, **over)
    cmd = [PY, "-u", "admm_diffusion_deconvolve.py",
           "--prior", "cond", "--ckpt", CKPT, "--data-dir", DATA,
           "--split", "val", "--indices", IDX, "--psf-file", PSF,
           "--split-norms", "--n-show", "0", "--log-every", "100", "--no-fits",
           "--out-dir", str(d),
           "--iters", str(p["iters"]), "--denoise-steps", str(p["denoise_steps"]),
           "--rho", str(p["rho"]), "--rho-scale", str(p["rho_scale"]),
           "--lam", str(p["lam"]), "--eta", str(p["eta"]),
           "--t-min", str(p["t_min"]), "--t-max", str(p["t_max"])]
    if p["renoise"]:
        cmd.append("--renoise")
    env = {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PATH": "/usr/bin:/bin"}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode:
        return tag, f"FAILED: {r.stderr.strip().splitlines()[-1][:120]}"
    return tag, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cfgs = configs()
    if args.only:
        want = set(args.only.split(","))
        cfgs = [c for c in cfgs if c[0] in want]
    print(f"[sweep] {len(cfgs)} configs, {args.jobs} at a time, 2 threads each")
    for tag, over in cfgs:
        print(f"   {tag:16s} {dict(BASE, **over)}")
    if args.dry_run:
        return

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for tag, status in ex.map(lambda a: run_one(*a), cfgs):
            print(f"[done] {tag:16s} {status}", flush=True)

    # ---------------- score everything through one metric path -------------
    n = json.loads((Path(DATA) / "norm.json").read_text())
    idx = parse_indices(IDX, 10 ** 9)
    o = np.load(Path(DATA) / "val_observed.npy")[idx]
    i = np.load(Path(DATA) / "val_ideal.npy")[idx]
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
    for tag, _ in cfgs:
        f = OUT / tag / "val_recon.npy"
        if not f.exists():
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

    hdr = (f"{'config':<16}{'resid':>8}{'|Δ|':>7}{'compl%':>8}{'spur%':>7}"
           f"{'sig_mag':>8}{'conc':>7}{'flux':>7}{'phot sct':>9}"
           f"{'int rmse':>9}{'calls':>7}")
    print(f"\n  truth's own residual = {truth_resid:.4f}  <- the target for "
          f"'resid'; |Δ| is the distance to it\n")
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: r["dresid"]):
        print(f"{r['tag']:<16}{r['resid']:>8.4f}{r['dresid']:>7.4f}"
              f"{r['completeness']:>8.1f}{r['spurious']:>7.1f}"
              f"{r['sig_mag']:>8.3f}{r['concentration']:>7.3f}"
              f"{r['flux_ratio']:>7.3f}{r['phot_scatter']:>9.3f}"
              f"{r['interior_rmse']:>9.4f}{r['cost']:>7d}")
    print("\n  sorted by distance to the truth's residual. Check whether that "
          "ordering agrees with the accuracy columns before believing it.")
    json.dump(rows, open(OUT / "sweep_metrics.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
