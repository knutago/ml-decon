"""
solver_metrics.py

One metric definition shared by every solver, so sampler / RED / ADMM numbers
are actually comparable. Import this rather than re-deriving the metrics in
each script -- the earlier round of numbers was hard to compare because RED
reported flux-space RMSE over a 4 px crop while ADMM reported an interior RMSE
over a 16 px crop.

All inputs are PHYSICAL FLUX arrays of shape (B, H, W).

  concentration   fraction of a 5x5 box's flux held by the central pixel, at
                  true point-source positions. The sharpness number: blurry
                  ~0.07, truth ~0.76. This is the one that matters for
                  photometry.
  interior_rmse   RMSE over a 16 px border crop. A 31x31 PSF on a 64x64 patch
                  with FFT (periodic) boundaries leaves only a ~34x34
                  well-posed interior; anything reported over the full patch
                  is dominated by wraparound, not by solver quality.
  edge_rmse       RMSE over the discarded border, i.e. how bad the wraparound
                  actually is. Reported so the crop is not hiding anything.
  floor_frac      % of pixels sitting at the zero floor. MATCH the truth, do
                  NOT minimize -- the truth is 7.28% (the ideal patches were
                  min-max normalized, so their faintest sky pixels land exactly
                  on 0 and the asinh inverse maps that to flux ~0). Reading
                  "lower is better" here is a mistake I made once already:
                  ADMM's 0.06% is a deviation of the same order as a ringing
                  solver's 12.85%, just in the other direction. Too HIGH means
                  ringing sidelobes are being clamped; too LOW means the solver
                  is holding the background off the floor, which shows up as a
                  positive flux bias.
  flux_ratio      total interior flux / truth's. Photometric bias.
  peak_ratio      mean value at true source positions / truth's.
"""

import numpy as np

CROP = 16          # border discarded as boundary-contaminated
BOX = 2            # concentration half-box (5x5)


def find_peaks(truth, thresh_pct=99.5, edge=3):
    """Local maxima above a per-patch percentile, away from the border."""
    peaks = []
    for b in range(truth.shape[0]):
        t = truth[b]
        cut = np.percentile(t, thresh_pct)
        for i in range(edge, t.shape[0] - edge):
            for j in range(edge, t.shape[1] - edge):
                if t[i, j] > cut and t[i, j] == t[i - 1:i + 2, j - 1:j + 2].max():
                    peaks.append((b, i, j))
    return peaks


def interior_peaks(peaks, H, W, crop=CROP):
    """Peaks that survive the boundary crop -- use these for a fair
    concentration number, since edge sources are wraparound-corrupted."""
    return [(b, i, j) for b, i, j in peaks
            if crop <= i < H - crop and crop <= j < W - crop]


def concentration(img, peaks):
    vals = []
    for b, i, j in peaks:
        box = img[b, i - BOX:i + BOX + 1, j - BOX:j + BOX + 1]
        if box.size == (2 * BOX + 1) ** 2 and box.sum() > 0:
            vals.append(box[BOX, BOX] / box.sum())
    return float(np.mean(vals)) if vals else float("nan")


def star_photometry(img, truth, peaks, ap=1):
    """PER-STAR flux ratios, recon / truth, in a (2*ap+1)^2 aperture.

    This is the number that actually matters for the science: aggregate
    flux_ratio can sit at 1.00 while individual stars scatter by 50% in
    opposite directions, and peak_ratio only looks at one pixel so it rewards
    a solver for putting a source's flux in the wrong place as long as the
    centre is right.

    ap=1 (3x3) is deliberately tight. The ideal patches are essentially delta
    functions (see compare_solvers_out/source_profile.png), and every M31 patch
    is confusion-limited at ~120 sources, so a wide aperture just measures the
    neighbours.

    Returns median ratio (photometric BIAS), the 16-84 percentile half-spread
    (SCATTER), and the fraction of stars recovered to within 10% and 20%.
    """
    r = []
    for b, i, j in peaks:
        t = truth[b, i - ap:i + ap + 1, j - ap:j + ap + 1].sum()
        if t > 0:
            r.append(img[b, i - ap:i + ap + 1, j - ap:j + ap + 1].sum() / t)
    if not r:
        return {"phot_median": float("nan"), "phot_scatter": float("nan"),
                "phot_within10": float("nan"), "phot_within20": float("nan")}
    r = np.asarray(r)
    lo, hi = np.percentile(r, [16, 84])
    return {
        "phot_median": float(np.median(r)),
        "phot_scatter": float((hi - lo) / 2),
        "phot_within10": float(100.0 * (np.abs(r - 1) <= 0.10).mean()),
        "phot_within20": float(100.0 * (np.abs(r - 1) <= 0.20).mean()),
        "phot_n": len(r),
    }


def evaluate(img, truth, peaks, crop=CROP):
    """img, truth: (B, H, W) in physical flux. Returns a metrics dict."""
    img = np.asarray(img, np.float64)
    truth = np.asarray(truth, np.float64)
    B, H, W = truth.shape
    inner = np.s_[:, crop:H - crop, crop:W - crop]
    ipk = interior_peaks(peaks, H, W, crop)

    edge_mask = np.ones((H, W), bool)
    edge_mask[crop:H - crop, crop:W - crop] = False

    m = {
        "concentration": concentration(img, ipk),
        "interior_rmse": float(np.sqrt(((img[inner] - truth[inner]) ** 2).mean())),
        "edge_rmse": float(np.sqrt(
            ((img[:, edge_mask] - truth[:, edge_mask]) ** 2).mean())),
        "floor_frac": float(100.0 * (img[inner] <= 0).mean()),
        "flux_ratio": float(img[inner].sum() / max(truth[inner].sum(), 1e-30)),
        "peak_ratio": float(
            np.mean([img[b, i, j] for b, i, j in ipk])
            / max(np.mean([truth[b, i, j] for b, i, j in ipk]), 1e-30)),
        "n_peaks": len(ipk),
    }
    m.update(star_photometry(img, truth, ipk))
    return m


HEADER = ("method", "concentration", "interior_rmse", "edge_rmse", "floor_frac",
          "flux_ratio", "peak_ratio")


def format_table(rows):
    """rows: list of (name, metrics_dict)."""
    w = max(len(n) for n, _ in rows) + 2
    has_dr = any("data_resid" in m for _, m in rows)
    out = [f"{'method':<{w}}{'conc':>7}{'int_rmse':>10}{'edge_rmse':>11}"
           f"{'floor%':>7}{'flux':>8}"
           + (f"{'data_res':>10}" if has_dr else "")
           + f"{'ph_med':>8}{'ph_scat':>9}{'<10%':>7}{'<20%':>7}"]
    out.append("-" * len(out[0]))
    for n, m in rows:
        out.append(f"{n:<{w}}{m['concentration']:>7.3f}"
                   f"{m['interior_rmse']:>10.4f}{m['edge_rmse']:>11.4f}"
                   f"{m['floor_frac']:>7.2f}{m['flux_ratio']:>8.3f}"
                   + (f"{m.get('data_resid', float('nan')):>10.4f}"
                      if has_dr else "")
                   + f"{m['phot_median']:>8.3f}{m['phot_scatter']:>9.3f}"
                     f"{m['phot_within10']:>7.1f}{m['phot_within20']:>7.1f}")
    return "\n".join(out)
