"""
completeness_analysis.py

The two diagnostics Knut asked for:

  1. COMPLETENESS vs INPUT FLUX -- the fraction of true sources recovered as a
     function of their true brightness. The headline number is the flux (and
     magnitude) at which completeness crosses 50%: that is the practical
     detection/confusion limit, and it is what you compare against the
     standard-photometry confusion limit.

  2. RECOVERED FLUX - INPUT FLUX vs INPUT FLUX -- the photometric bias and
     scatter. Shows whether faint sources are systematically over- or
     under-measured (Eddington-style bias from noise + blending) and where
     the photometry stops being trustworthy.

Sources are pooled over MANY patches: a single 64x64 patch holds ~120 sources,
far too few to bin. Fluxes are measured by circular aperture photometry with
a local annulus background, and the RECOVERED flux is measured by FORCED
photometry at the TRUE position, so the comparison is not confounded by
centroid disagreement.

Several methods can be overlaid on one figure for comparison.

Usage:
  # one method
  python completeness_analysis.py \
      --recon "cond_sample_out/*_cond.fits" --label "conditional sampler" \
      --truth-dir patches/sharp/fits

  # several methods on the same axes
  python completeness_analysis.py \
      --recon "cond_sample_out/*_cond.fits"      --label "cond sampler" \
      --recon "rw_best/*_reweighted.fits"        --label "reweighted L1" \
      --truth-dir patches/sharp/fits --out-dir completeness_out
"""

import argparse
import csv
import glob
import os
import re

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import find_peaks
from photutils.centroids import centroid_com
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry


# ----------------------------------------------------------------------------
# detection / photometry
# ----------------------------------------------------------------------------
def detect(img, nsigma=5.0, box=5, edge=4):
    _, med, std = sigma_clipped_stats(img, sigma=3.0)
    tbl = find_peaks(img, threshold=med + nsigma * std, box_size=box)
    if tbl is None or len(tbl) == 0:
        return np.empty((0, 2))
    H, W = img.shape
    r = box // 2
    out = []
    for row in tbl:
        x0, y0 = int(row["x_peak"]), int(row["y_peak"])
        if x0 < edge or x0 >= W - edge or y0 < edge or y0 >= H - edge:
            continue
        sub = np.clip(img[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1] - med, 0, None)
        if sub.sum() <= 0:
            out.append((x0, y0)); continue
        cx, cy = centroid_com(sub)
        out.append((x0 - r + cx, y0 - r + cy))
    return np.asarray(out, dtype=float)


def aper_flux(img, xy, r=3.0, r_in=5.0, r_out=8.0):
    """Background-subtracted circular aperture flux at each (x, y)."""
    if len(xy) == 0:
        return np.array([])
    ap = CircularAperture(xy, r=r)
    an = CircularAnnulus(xy, r_in=r_in, r_out=r_out)
    phot = aperture_photometry(img, ap)
    bkg = []
    for m in an.to_mask(method="center"):
        v = m.multiply(img)
        v = v[m.data > 0] if v is not None else np.array([])
        bkg.append(np.median(v) if len(v) else 0.0)
    return np.asarray(phot["aperture_sum"]) - np.asarray(bkg) * ap.area


def match(truth_xy, recon_xy, radius):
    pairs, used = [], set()
    for ti, t in enumerate(truth_xy):
        if len(recon_xy) == 0:
            break
        d = np.hypot(recon_xy[:, 0] - t[0], recon_xy[:, 1] - t[1])
        j = int(np.argmin(d))
        if d[j] <= radius and j not in used:
            pairs.append((ti, j)); used.add(j)
    return dict(pairs)


# ----------------------------------------------------------------------------
def collect(pattern, truth_dir, args):
    """Pool every truth source across all matching patches.

    Returns arrays: input flux, recovered flux (forced at the truth
    position), and a detected flag.
    """
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files matched {pattern}")
    f_in, f_rec, found = [], [], []
    n_patch = 0
    for rp in files:
        base = os.path.basename(rp)
        # strip any trailing _<suffix>.fits added by the reconstruction script
        stem = re.sub(r"_(cond|graikos|pnp|sgpnp|diffpnp|deconv|reweighted"
                      r"|l1only|spaced|linear|strength)\.fits$", ".fits", base)
        tp = os.path.join(truth_dir, stem)
        if not os.path.exists(tp):
            continue
        truth = fits.getdata(tp).astype(np.float64)
        recon = fits.getdata(rp).astype(np.float64)
        if truth.shape != recon.shape:
            continue
        # The INPUT catalogue and the DETECTION threshold must be set
        # independently. With one shared --nsigma the input list is only as
        # deep as the detection cut, so every source in the sample is one the
        # truth image already showed clearly and completeness cannot fall to
        # 50% -- measured, the faintest bin still sat at 76.7%. Deepening the
        # truth catalogue alone extends the faint end; the recon threshold
        # stays fixed so "detected" keeps its meaning across the curve.
        t_xy = detect(truth, args.truth_nsigma, args.box, args.edge)
        r_xy = detect(recon, args.nsigma, args.box, args.edge)
        if len(t_xy) == 0:
            continue
        m = match(t_xy, r_xy, args.match_radius)
        fi = aper_flux(truth, t_xy, args.aper, args.ann_in, args.ann_out)
        fr = aper_flux(recon, t_xy, args.aper, args.ann_in, args.ann_out)
        for i in range(len(t_xy)):
            f_in.append(fi[i]); f_rec.append(fr[i]); found.append(i in m)
        n_patch += 1
    print(f"  {n_patch} patches, {len(f_in)} truth sources pooled")
    return np.asarray(f_in), np.asarray(f_rec), np.asarray(found, dtype=bool)


def completeness_curve(f_in, found, nbins, fmin=None, fmax=None):
    ok = np.isfinite(f_in) & (f_in > 0)
    f_in, found = f_in[ok], found[ok]
    lo = fmin if fmin else np.percentile(f_in, 1)
    hi = fmax if fmax else np.percentile(f_in, 99.5)
    edges = np.geomspace(lo, hi, nbins + 1)
    cen, comp, err, n = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (f_in >= a) & (f_in < b)
        if m.sum() < 3:
            continue
        k = found[m].sum(); tot = m.sum()
        p = k / tot
        cen.append(np.sqrt(a * b)); comp.append(p); n.append(tot)
        err.append(np.sqrt(max(p * (1 - p), 1e-9) / tot))   # binomial
    return (np.asarray(cen), np.asarray(comp), np.asarray(err),
            np.asarray(n))


def fifty_percent_limit(cen, comp):
    """Flux at which completeness crosses 50%, by linear interpolation in
    log-flux (searching from the bright end downward)."""
    if len(cen) < 2:
        return np.nan
    order = np.argsort(cen)[::-1]          # bright -> faint
    c, p = cen[order], comp[order]
    for i in range(len(c) - 1):
        if p[i] >= 0.5 >= p[i + 1]:
            lc0, lc1 = np.log10(c[i]), np.log10(c[i + 1])
            t = (0.5 - p[i]) / (p[i + 1] - p[i] + 1e-12)
            return 10 ** (lc0 + t * (lc1 - lc0))
    return np.nan


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recon", action="append", required=True,
                   help="glob for reconstruction FITS (repeatable)")
    p.add_argument("--label", action="append", default=None,
                   help="legend label for each --recon (repeatable)")
    p.add_argument("--truth-dir", default="patches/sharp/fits")
    p.add_argument("--nsigma", type=float, default=5.0,
                   help="detection threshold applied to the RECONSTRUCTION")
    p.add_argument("--truth-nsigma", type=float, default=None,
                   help="threshold for building the INPUT catalogue from the "
                        "truth image. Lower than --nsigma to reach fainter "
                        "sources and pull the curve down toward the 50% "
                        "crossing. Defaults to --nsigma (old behaviour).")
    p.add_argument("--box", type=int, default=5)
    p.add_argument("--edge", type=int, default=4)
    p.add_argument("--match-radius", type=float, default=2.5)
    p.add_argument("--aper", type=float, default=3.0)
    p.add_argument("--ann-in", type=float, default=5.0)
    p.add_argument("--ann-out", type=float, default=8.0)
    p.add_argument("--nbins", type=int, default=12)
    p.add_argument("--zeropoint", type=float, default=0.0,
                   help="instrumental mag = -2.5*log10(flux) + zeropoint")
    p.add_argument("--mag-axis", action="store_true",
                   help="plot completeness against instrumental MAGNITUDE "
                        "(axis inverted, bright on the left) instead of flux")
    p.add_argument("--out-dir", default="completeness_out")
    args = p.parse_args()
    if args.truth_nsigma is None:
        args.truth_nsigma = args.nsigma
    os.makedirs(args.out_dir, exist_ok=True)
    labels = args.label or [f"run {i+1}" for i in range(len(args.recon))]
    if len(labels) != len(args.recon):
        raise SystemExit("give one --label per --recon")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
    rows = []
    for pat, lab in zip(args.recon, labels):
        print(f"[{lab}]  {pat}")
        f_in, f_rec, found = collect(pat, args.truth_dir, args)
        cen, comp, err, n = completeness_curve(f_in, found, args.nbins)
        lim = fifty_percent_limit(cen, comp)
        maglim = (-2.5 * np.log10(lim) + args.zeropoint
                  if np.isfinite(lim) else np.nan)
        print(f"    overall completeness {found.mean():.1%}")
        print(f"    50% completeness limit: flux {lim:.4e}"
              f"  (instr. mag {maglim:.3f})")

        # ---- panel 1: completeness vs input magnitude (or flux) ----
        # Instrumental mag: m = -2.5 log10(f) + ZP. Fainter is a LARGER
        # number, so the axis is inverted to keep bright on the left and the
        # curve falling left-to-right, as completeness curves are always read.
        if args.mag_axis:
            xc = -2.5 * np.log10(cen) + args.zeropoint
            xlim_ = maglim
            xlabel = f"input (true) instrumental magnitude (ZP={args.zeropoint:g})"
            tag = f"50% @ {maglim:.2f} mag" if np.isfinite(maglim) else "no 50% crossing"
        else:
            xc, xlim_ = cen, lim
            xlabel = "input (true) aperture flux"
            tag = f"50% @ {lim:.2e}" if np.isfinite(lim) else "no 50% crossing"
        ax[0].errorbar(xc, comp, yerr=err, marker="o", ms=4, lw=1.4,
                       capsize=2, label=f"{lab}  ({tag})")
        if np.isfinite(xlim_):
            ax[0].axvline(xlim_, ls=":", lw=1, alpha=0.5,
                          color=ax[0].lines[-1].get_color())

        # ---- panel 2: recovered - input vs input ----
        ok = np.isfinite(f_in) & (f_in > 0) & np.isfinite(f_rec) & found
        ax[1].plot(f_in[ok], (f_rec - f_in)[ok], ".", ms=3, alpha=0.35,
                   label=f"{lab} (detected)")
        # binned median of the residual
        e = np.geomspace(np.percentile(f_in[ok], 1),
                         np.percentile(f_in[ok], 99.5), args.nbins + 1)
        bc, bm = [], []
        for a, b in zip(e[:-1], e[1:]):
            m = ok & (f_in >= a) & (f_in < b)
            if m.sum() >= 3:
                bc.append(np.sqrt(a * b))
                bm.append(np.median((f_rec - f_in)[m]))
        ax[1].plot(bc, bm, "-", lw=2.2)

        for i in range(len(cen)):
            rows.append(dict(method=lab, flux=cen[i], completeness=comp[i],
                             err=err[i], n=n[i], limit50=lim, maglim=maglim))

    ax[0].axhline(0.5, color="k", ls="--", lw=1, alpha=0.6)
    if args.mag_axis:
        ax[0].invert_xaxis()
        ax[0].set_title("Completeness vs input magnitude\n"
                        "(dashed = 50%, the practical confusion limit)",
                        fontsize=10)
    else:
        ax[0].set_xscale("log")
        ax[0].set_title("Completeness vs input flux\n"
                        "(dashed = 50%, the practical confusion limit)",
                        fontsize=10)
    ax[0].set_ylim(-0.02, 1.02)
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel("completeness")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    ax[1].axhline(0, color="k", ls="--", lw=1, alpha=0.6)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("input (true) aperture flux")
    ax[1].set_ylabel("recovered - input flux")
    ax[1].set_title("Photometric residual vs input flux\n"
                    "(thick line = binned median; >0 = over-measured)",
                    fontsize=10)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    fig.tight_layout()
    png = os.path.join(args.out_dir, "completeness_and_flux.png")
    fig.savefig(png, dpi=140); plt.close(fig)

    csvp = os.path.join(args.out_dir, "completeness.csv")
    with open(csvp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "flux", "completeness",
                                           "err", "n", "limit50", "maglim"])
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {png}\nWrote {csvp}")


if __name__ == "__main__":
    main()
