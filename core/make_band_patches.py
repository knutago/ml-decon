"""
make_band_patches.py

make_ef_patches.py with the frame pair, PSF and checkpoint lifted out to
arguments, so the SAME cutting/normalization path can be pointed at either
HST band of the M32 field. Written for the two-band colour-magnitude work:
a CMD is a *difference* between bands, so any asymmetry in how the two
datasets are built shows up directly as a colour error.

WHICH FRAME IS WHICH BAND (measured 2026-09-10, four independent checks)
-----------------------------------------------------------------------
The file naming does not say, and the earlier `m32_ef_v2_f435w` /
`m32_ef_v2_f555w` dataset pair is NOT two bands -- both were cut from the same
`m32b_phase_v2ef.fits` with different `scale_a` (0.674 / 1.450), a
normalization trick from when only one real frame existed. Do not read them as
B and V.

                          m32b_phase_v2ef        m32v_phase_v2ef
                          (+ l160 rival)         (+ l640 rival)
    FITS header           FILTER2 = F435W        FILTER1 = F555W
                          PHOTPLAM 4311.0 A      PHOTPLAM 5356.0 A
                          j9h901ftq, 2005-09-20  j9h905kcq, 2005-09-22
    PSF core r50          2.246 px               2.884 px
      matching sim PSF    psf_m32sim_f435w       psf_m32sim_f555w
      (its r50)           2.231 px               2.888 px
    clipped bkg median    191.2                  364.5
      matching ckpt norm  f435w (207.0, b=46.6)  f555w (385.3, b=100.2)

    => m32b = F435W (B),  m32v = F555W (V).

The two frames land on the IDENTICAL pixel grid: cross-correlating a 256^2
central block gives best offset (0,0) at correlation 0.876. So catalogues cut
from the same `--region` in both bands cross-match by pixel position with no
registration step. Keep `--region` identical between bands or that breaks.

THE REFERENCE IS NOT A TRUTH. `val_ideal.npy` holds the rival deconvolution
(l160 for B, l640 for V) because that is the slot the solver reads. Every
metric against it is agreement-with-a-rival, not accuracy. Same warning as
make_ef_patches.py, repeated in meta.json.

SKY. Not written into the dataset -- it is a solver argument, and which value
is right depends on what x is asked to BE (see make_ef_patches.py's docstring
and the 2026-09-06 sky bug). This script PRINTS the two candidates:
  sky_const_for_reference_like_x  -- x contains M32's diffuse light, like the
                                     rival deconvolution does
  sky_reference_background        -- the rival's own sigma-clipped background
                                     in the flux domain; this is the one to
                                     use with a SPARSE point-source prior, and
                                     it is estimated from the rival rather
                                     than from our own output, so it is not
                                     circular.
`--matched-reference` is deliberately NOT offered: it folds the pedestal into
the stored reference and cost one round of wrong display panels.

Usage:
    python make_band_patches.py --band f555w
    python make_band_patches.py --band f435w
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from red_pnp_deconvolve import TorchNorm, load_kernel, make_otf  # noqa: E402

# Paths are env-overridable so the same file runs on Vista ($WORK/$SCRATCH)
# without editing. ML_DECON = the shared repo, MYCODE = this directory.
ML = Path(os.environ.get("ML_DECON", "/home/alex/noir_ml/global/ml-decon"))
MY = Path(os.environ.get("MYCODE", Path(__file__).resolve().parent))

# The measured band assignment above, as data. Each entry is a real HST frame
# plus somebody else's deconvolution of that same frame.
BANDS = {
    "f435w": dict(
        observed=ML / "data/m32b_phase_v2ef.fits",
        reference=ML / "data/m32b_phase_v2l160.fits",
        psf=MY / "psf_m32_v2_centred.fits",
        ckpt=ML / "checkpoints_cond_m32sim_f435w/best.pt",
        out=ML / "data/m32_band_f435w",
        photflam=5.3685924e-19, photzpt=-21.1, exptime=1279.0,
    ),
    "f555w": dict(
        observed=ML / "data/m32_l640/m32v_phase_v2ef.fits",
        reference=ML / "data/m32_l640/m32v_phase_v2l640.fits",
        psf=MY / "psf_m32v_v2_centred.fits",
        ckpt=ML / "checkpoints_cond_m32sim_f555w/best.pt",
        out=ML / "data/m32_band_f555w",
        photflam=3.020061e-19, photzpt=-21.1, exptime=1279.0,
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", choices=sorted(BANDS), required=True)
    ap.add_argument("--region", default="768:1280,768:1280",
                    help="Y0:Y1,X0:X1. MUST match between bands for the "
                         "catalogues to cross-match by pixel position.")
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--stride", type=int, default=64,
                    help="64 = non-overlapping, so patches tile the region "
                         "exactly and mosaic back for display")
    ap.add_argument("--out", default=None, help="override the default out dir")
    ap.add_argument("--ckpt", default=None,
                    help="override the band's default checkpoint. Use this to "
                         "normalize BOTH bands into ONE prior's observed "
                         "domain, so a single shared prior can be run on both "
                         "-- which makes any prior-induced scale error "
                         "COMMON-MODE and therefore cancelling in the colour. "
                         "The PSF and the photometric zeropoint stay "
                         "band-specific regardless: the operator must match "
                         "the data, and the zeropoint is physics.")
    ap.add_argument("--train-ref", default=None, metavar="DATASET",
                    help="which m32_sim_* dataset to print the in-distribution "
                         "check against (default: this band's own). Set it to "
                         "the dataset the --ckpt was actually trained on.")
    args = ap.parse_args()

    cfg = dict(BANDS[args.band])
    if args.ckpt:
        cfg["ckpt"] = Path(args.ckpt)
    ys, xs = args.region.split(",")
    y0, y1 = (int(v) for v in ys.split(":"))
    x0, x1 = (int(v) for v in xs.split(":"))
    out = Path(args.out) if args.out else cfg["out"]
    out.mkdir(parents=True, exist_ok=True)

    obs_img = fits.getdata(cfg["observed"]).astype(np.float64).squeeze()
    ref_img = fits.getdata(cfg["reference"]).astype(np.float64).squeeze()
    if obs_img.shape != ref_img.shape:
        raise SystemExit(f"shape mismatch {obs_img.shape} vs {ref_img.shape}")

    hdr = fits.getheader(cfg["observed"])
    filt = hdr.get("FILTER1", "")
    if filt in ("CLEAR1S", "CLEAR1L"):
        filt = hdr.get("FILTER2", "")
    print(f"[band] {args.band}: {cfg['observed'].name} header FILTER={filt} "
          f"PHOTPLAM={hdr.get('PHOTPLAM')}")
    if filt.lower() != args.band:
        raise SystemExit(
            f"REFUSING: --band {args.band} but the FITS header says {filt}. "
            "The band assignment is what the whole CMD rests on; fix the "
            "BANDS table rather than overriding this.")

    # ---- the measured relation between the two frames --------------------
    H, W = obs_img.shape
    K = make_otf(load_kernel(str(cfg["psf"])), (H, W), torch.device("cpu"))
    Aref = torch.fft.irfft2(K * torch.fft.rfft2(torch.from_numpy(ref_img)[None, None]),
                            s=(H, W)).numpy()[0, 0]
    bl = 128
    nb = H // bl
    om = obs_img[:nb * bl, :nb * bl].reshape(nb, bl, nb, bl).mean((1, 3)).ravel()
    tm = Aref[:nb * bl, :nb * bl].reshape(nb, bl, nb, bl).mean((1, 3)).ravel()
    G, sky_raw = np.linalg.lstsq(np.stack([tm, np.ones_like(tm)], 1), om,
                                 rcond=None)[0]
    pix_res = np.linalg.norm(obs_img - (G * Aref + sky_raw)) / np.linalg.norm(obs_img)
    print(f"[pair] observed = {G:.4f} * ({cfg['psf'].name} * {cfg['reference'].name}) "
          f"+ {sky_raw:.4f}   (full-pixel residual {pix_res:.4f})")

    # ---- background-statistics match into the checkpoint's observed domain --
    ck = torch.load(cfg["ckpt"], map_location="cpu", weights_only=False)
    onp = ck["dataset_norm"]["observed"]
    obs_norm = TorchNorm(onp)
    _, med_o, std_o = sigma_clipped_stats(obs_img, sigma=3.0, maxiters=5)
    a = float(onp["beta"]) / float(std_o)
    c = float(onp["median"])
    print(f"[norm] observed sigma-clipped median {med_o:.4f} std {std_o:.4f}  ->  "
          f"checkpoint median {c:.6g} beta {onp['beta']:.6g}   scale a={a:.6g}")

    # flux = a*(observed - med_o) + c, so with x_ref = a*G*reference the
    # forward model in this domain is  flux = (psf * x_ref) + sky_const, gain 1.
    sky_const = c - a * float(med_o) + a * float(sky_raw)
    # The rival is a full deconvolved IMAGE, so its own background still holds
    # M32's diffuse light. A sparse x does not, so that light has to move into
    # the operator on top of the reference-like offset. Clip over the REGION
    # being cut, not the whole frame -- the pedestal varies across the field.
    _, med_r, _ = sigma_clipped_stats(ref_img[y0:y1, x0:x1], sigma=3.0, maxiters=5)
    sky_sparse = sky_const + a * G * float(med_r)
    print(f"[sky ] --sky {sky_const:.6g}  -> x is REFERENCE-LIKE "
          f"(diffuse light in x)")
    print(f"[sky ] --sky {sky_sparse:.6g}  -> x is SPARSE (diffuse light in the "
          f"operator). USE THIS ONE with the point-source prior.")
    # cross-check in raw frame counts, which is normalization-independent and
    # is what makes this comparable between the two bands' different domains
    print(f"[sky ]   = {(sky_sparse - c) / a + float(med_o):.2f} counts in the raw "
          f"frame; reference clipped median over the region {med_r:.4f}")

    obs_flux = a * (obs_img - float(med_o)) + c
    ref_flux = a * G * ref_img          # rival deconvolution, same flux domain

    # ---- cut the patches --------------------------------------------------
    P, S = args.patch, args.stride
    obs, ref, corners = [], [], []
    for yy in range(y0, y1 - P + 1, S):
        for xx in range(x0, x1 - P + 1, S):
            obs.append(obs_flux[yy:yy + P, xx:xx + P])
            ref.append(ref_flux[yy:yy + P, xx:xx + P])
            corners.append((yy, xx))
    obs = np.stack(obs)[:, None]
    ref = np.stack(ref)[:, None]
    print(f"[cut]  {len(obs)} patches of {P}x{P} (stride {S}) from "
          f"[{y0}:{y1},{x0}:{x1}]")

    # observed is stored ENCODED (what the network conditions on); the rival is
    # stored in FLUX with an identity norm, so nothing re-stretches it
    z = obs_norm.forward(torch.from_numpy(obs)).numpy().astype(np.float32)
    ref32 = ref.astype(np.float32)

    np.save(out / "val_observed.npy", z)
    np.save(out / "val_ideal.npy", ref32)
    np.save(out / "train_observed.npy", z[:1])     # placeholder, unused
    np.save(out / "train_ideal.npy", ref32[:1])
    (out / "norm.json").write_text(json.dumps({
        "observed": onp,
        "ideal": {"method": "linear", "lo": 0.0, "hi": 1.0},
    }, indent=2))
    (out / "meta.json").write_text(json.dumps({
        "WARNING": f"val_ideal.npy is {cfg['reference'].name}, a RIVAL "
                   "DECONVOLUTION, not a truth. Metrics against it measure "
                   "agreement with the rival, not accuracy.",
        "band": args.band,
        "filter_header": filt,
        "observed_fits": str(cfg["observed"]),
        "reference_fits": str(cfg["reference"]),
        "psf": str(cfg["psf"]),
        "ckpt": str(cfg["ckpt"]),
        "photometry": {"photflam": cfg["photflam"], "photzpt": cfg["photzpt"],
                       "exptime": cfg["exptime"],
                       "stmag_zp": -2.5 * np.log10(cfg["photflam"]) + cfg["photzpt"]},
        "region": [y0, y1, x0, x1], "patch": P, "stride": S,
        "observed_to_reference": {"gain": float(G), "sky": float(sky_raw),
                                  "pixel_residual": float(pix_res)},
        "normalization": {"method": "background-statistics match",
                          "observed_clipped_median": float(med_o),
                          "observed_clipped_std": float(std_o),
                          "scale_a": a, "target_median": c,
                          "target_beta": float(onp["beta"]),
                          "sky_const_for_reference_like_x": float(sky_const),
                          "sky_sparse": float(sky_sparse),
                          "sky_sparse_raw_counts":
                              float((sky_sparse - c) / a + float(med_o)),
                          "reference_clipped_median_region": float(med_r)},
        "corners": corners,
    }, indent=2))

    # ---- is the prior in distribution? ------------------------------------
    q = [1, 50, 99, 99.9, 100]
    print("[check] conditioning z percentiles 1/50/99/99.9/100: "
          + " ".join(f"{v:.4f}" for v in np.percentile(z, q)))
    ref = args.train_ref or f"m32_sim_{args.band}"
    tr = np.load(ML / f"data/{ref}/val_observed.npy", mmap_mode="r")
    print(f"[check]   what it trained on ({ref}):"
          + " " * max(1, 24 - len(ref))
          + " ".join(f"{v:.4f}" for v in np.percentile(np.asarray(tr[::8]), q)))
    print(f"\nwrote {out}/  ({len(obs)} patches)")


if __name__ == "__main__":
    main()
