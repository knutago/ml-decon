#!/usr/bin/env python3
"""
photometry_benchmark.py

The agreed photometric benchmark, run over SEVERAL methods at once and reported
in MAGNITUDE BINS. Steps are numbered as specified:

  0. Benchmark dataset: patch size is measured and reported.
  1. Measure the flux scaling between the blurry input and the sharp truth
     BEFORE trusting any photometry (--check-scaling, on by default).
  2. Detection threshold: --nsigma above the sigma-clipped sky.
  3. Identify pixels above that threshold.
  4. For each, flux = total value in a 3x3 box centred on it. A pixel counts as
     a peak only if it is the maximum of its own 3x3 neighbourhood, so fainter
     pixels inside the same box are not independent detections.
  5. Match derived positions against the input list within --match-radius px.
  6. Where several input sources match one measured source, keep the one with
     the most similar flux (compared in log flux, so the choice is scale-free).
     That input counts as DETECTED.
  7. Photometric error = scatter of the flux differences, IN BINS OF MAGNITUDE.
  8. Completeness = N_recovered / N_input, IN BINS OF MAGNITUDE.

Steps 2-6 are imported from benchmark_eval.py rather than reimplemented, so
these numbers are the same measurement the rest of the project uses.

The INPUT catalogue is built by running steps 2-4 on the truth image, so input
and measured catalogues are constructed identically and the comparison is not
biased by differing detection machinery.

Magnitudes are INSTRUMENTAL: mag = -2.5 log10(flux) + --zeropoint, and the
default zeropoint is 0. Only differences and bin edges are meaningful; the
absolute values are not calibrated to anything.

Usage:
  python photometry_benchmark.py \
      --method "L1 (FISTA)=l1_out/l1_recons.npz:fista_tau0.008" \
      --method "ADMM+diffusion (FFT)=admm_fft_best/val_recon.npy" \
      --method "Conditional sampler=cond_sample_npy_out/draws.npy:mean" \
      --out-dir photom_bench

  # a .npz needs a key; a 4-D draws array needs 'mean' or a draw index
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

from benchmark_eval import detect_and_measure, match_catalogues, sky_stats
from red_pnp_deconvolve import TorchNorm, load_kernel, make_otf, parse_indices


# Categorical slots 1-4 of the project palette. This order is the documented
# all-pairs-validated set, which is what a scatter needs; each series also gets
# its own marker so identity never rests on colour alone.
SERIES = [("#2a78d6", "o"), ("#008300", "s"), ("#e87ba4", "^"), ("#eda100", "D")]
GRID = dict(color="0.85", lw=0.6)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_recon(spec: str, n_expect: int) -> np.ndarray:
    """'path.npy', 'path.npz:key', 'draws.npy:mean', 'draws.npy:0' -> (B,H,W)."""
    path, _, key = spec.partition(":")
    p = Path(path)
    if p.suffix == ".npz":
        z = np.load(p)
        if not key:
            raise SystemExit(f"{p} is an .npz; append ':key' "
                             f"(available: {', '.join(z.files)})")
        if key not in z.files:
            raise SystemExit(f"{p} has no key '{key}' "
                             f"(available: {', '.join(z.files)})")
        a = z[key]
    else:
        a = np.load(p)

    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 4:
        if a.shape[1] == 1:                       # (B, 1, H, W)
            a = a[:, 0]
        else:                                     # (D, B, H, W) sampler draws
            if key in ("", "mean"):
                print(f"  [{p.name}] 4-D {a.shape}: averaging {a.shape[0]} draws. "
                      f"Append ':0' for a single draw -- draw-averaging and "
                      f"single draws do NOT agree on faint sources.")
                a = a.mean(0)
            else:
                a = a[int(key)]
    if a.ndim != 3:
        raise SystemExit(f"{p}: expected (B,H,W), got {a.shape}")
    if len(a) < n_expect:
        raise SystemExit(f"{p}: has {len(a)} patches, need {n_expect}")
    if len(a) != n_expect:
        print(f"  [{p.name}] has {len(a)} patches, using the first {n_expect} "
              f"-- check --indices matches the run that produced it")
    return a[:n_expect]


def check_alignment(recon, truth, label, min_corr=0.2):
    """Refuse to score a reconstruction that is not the SAME patches as truth.

    Reconstruction .npy files are written per-run into a fixed filename, so a
    later run with different --indices silently replaces an earlier one and the
    array still loads with a plausible shape and plausible pixel values. Every
    metric downstream then compares patch k of one field against patch k of
    another and reports confident nonsense -- completeness near the chance
    floor, flux ratios scattered over decades.

    Cheap detector: patch k of a real reconstruction correlates with truth k
    far better than with any other truth patch. If the best match is off the
    diagonal, or the diagonal correlation is at noise level, something is
    misaligned.
    """
    diag, offdiag_wins = [], 0
    for k in range(len(truth)):
        c = np.array([np.corrcoef(recon[k].ravel(), truth[j].ravel())[0, 1]
                      for j in range(len(truth))])
        c = np.nan_to_num(c)
        diag.append(c[k])
        if int(np.argmax(c)) != k:
            offdiag_wins += 1
    diag = np.asarray(diag)
    if np.median(diag) < min_corr or offdiag_wins > len(truth) // 4:
        raise SystemExit(
            f"\n[FATAL] '{label}' does not line up with the truth patches.\n"
            f"        median diagonal correlation {np.median(diag):.3f} "
            f"(want > {min_corr}), and {offdiag_wins}/{len(truth)} patches "
            f"correlate better with a DIFFERENT truth patch.\n"
            f"        The file is almost certainly from a run with different "
            f"--indices/--split/--data-dir. Regenerate it with the same "
            f"--indices used here, into its own --out-dir.")
    return float(np.median(diag))


# ---------------------------------------------------------------------------
# steps 1-8
# ---------------------------------------------------------------------------
def scaling_check(observed, truth, psf, device):
    """Step 1. Is the blurry input on the same flux scale as the sharp truth?

    Convolution conserves total flux for a unit-sum PSF, so sum(observed) and
    sum(PSF * truth) must agree. If they do not, every flux ratio below is
    multiplied by a constant that has nothing to do with the deconvolution, and
    the photometry is uninterpretable.

    NB this is measured over the POOLED PATCHES, which is the best available
    here. The specification asks for it on the full image before patching --
    do that too if the full frames are to hand, because a patch set can hide a
    position-dependent scale.
    """
    H, W = truth.shape[-2:]
    K = make_otf(psf, (H, W), device)
    t = torch.from_numpy(truth).float().to(device)[:, None]
    blurred = torch.fft.irfft2(K * torch.fft.rfft2(t), s=(H, W)).cpu().numpy()[:, 0]
    c = 16
    inner = np.s_[:, c:H - c, c:W - c]
    return {
        "sum_obs_over_sum_truth": float(observed.sum() / truth.sum()),
        "sum_obs_over_sum_blurred_truth": float(observed.sum() / blurred.sum()),
        "interior_obs_over_blurred_truth":
            float(observed[inner].sum() / blurred[inner].sum()),
        "interior_rmse_obs_vs_blurred_truth":
            float(np.sqrt(((observed[inner] - blurred[inner]) ** 2).mean())),
    }


def detect_at(img, thr, sky, edge, box):
    """detect_and_measure with an EXTERNALLY SUPPLIED threshold and sky.

    Mirrors benchmark_eval.detect_and_measure exactly except that `thr` and
    `sky` are given rather than measured from `img`. Needed because the
    per-image threshold is not a fair basis for comparing methods: it is
    med + nsigma*std of each image's OWN sigma-clipped background, so a sparse
    solver that emits mostly exact zeros collapses its own std and awards
    itself a deeper cut. Measured here at 3 sigma: the truth cuts at 0.0337
    while L1 (80% exact zeros) cuts at 0.0080 -- 1.57 magnitudes deeper, on the
    same field, for free.
    """
    from scipy.ndimage import maximum_filter
    above = img > thr
    ismax = img >= maximum_filter(img, size=box)
    peaks = above & ismax
    H, W = img.shape
    r = box // 2
    pos, flx = [], []
    ys, xs = np.where(peaks)
    for y0, x0 in zip(ys, xs):
        if x0 < edge or x0 >= W - edge or y0 < edge or y0 >= H - edge:
            continue
        sub = img[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1]
        pos.append((float(x0), float(y0)))
        flx.append(float(sub.sum() - sky * sub.size))
    return np.asarray(pos).reshape(-1, 2), np.asarray(flx)


def run_catalogue(recon, truth, nsigma, box, edge, radius, truth_nsigma=None,
                  common_threshold=False):
    """Steps 2-6 over every patch. Returns per-INPUT-SOURCE arrays.

    t_flux   truth aperture flux of every input source
    matched  whether that input source was recovered
    r_flux   measured aperture flux where matched (nan otherwise)
    n_meas   total measured peaks (for the spurious rate)

    `truth_nsigma` is the threshold for the INPUT catalogue and defaults to
    `nsigma`. Keeping them equal is what makes 100% completeness attainable:
    the truth image scored against itself then recovers every source. Setting
    truth_nsigma BELOW nsigma deliberately defines an input list that reaches
    fainter than the detection pass can, which imposes a completeness ceiling
    under 100% that no method can beat -- see the note in main().
    """
    tn = nsigma if truth_nsigma is None else truth_nsigma
    t_flux, matched, r_flux = [], [], []
    n_meas = 0
    for k in range(len(truth)):
        ip, ifx = detect_and_measure(truth[k], tn, edge, box)
        if common_threshold:
            # Same absolute cut for every method: the TRUTH's sky + nsigma*std.
            sky_t, sig_t = sky_stats(truth[k])
            mp, mfx = detect_at(recon[k], sky_t + nsigma * sig_t, sky_t,
                                edge, box)
        else:
            mp, mfx = detect_and_measure(recon[k], nsigma, edge, box)
        n_meas += len(mp)
        pairs = dict(match_catalogues(ip, ifx, mp, mfx, radius))
        for i in range(len(ifx)):
            t_flux.append(ifx[i])
            if i in pairs:
                matched.append(True)
                r_flux.append(mfx[pairs[i]])
            else:
                matched.append(False)
                r_flux.append(np.nan)
    return (np.asarray(t_flux), np.asarray(matched, bool),
            np.asarray(r_flux), n_meas)


def to_mag(flux, zp=0.0):
    """Instrumental magnitude; non-positive fluxes become nan."""
    f = np.asarray(flux, float)
    out = np.full(f.shape, np.nan)
    ok = f > 0
    out[ok] = -2.5 * np.log10(f[ok]) + zp
    return out


def bin_stats(t_mag, matched, dmag, edges, min_per_bin=5):
    """Steps 7-8 per magnitude bin."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (t_mag >= lo) & (t_mag < hi) & np.isfinite(t_mag)
        n_in = int(m.sum())
        if n_in == 0:
            continue
        n_rec = int((m & matched).sum())
        d = dmag[m & matched]
        d = d[np.isfinite(d)]
        rows.append(dict(
            mag_lo=float(lo), mag_hi=float(hi), mag_mid=float(0.5 * (lo + hi)),
            n_input=n_in, n_recovered=n_rec,
            completeness=100.0 * n_rec / n_in,
            # binomial error on the completeness fraction
            completeness_err=100.0 * np.sqrt(
                max(n_rec, 1) / n_in * (1 - n_rec / n_in) / n_in),
            # step 7: scatter. std is what was specified; the MAD version is
            # reported alongside because a handful of blended sources in a
            # confusion-limited field drag the std around badly.
            phot_scatter_mag=float(np.std(d)) if len(d) >= min_per_bin else np.nan,
            phot_scatter_mad=(float(1.4826 * np.median(np.abs(d - np.median(d))))
                              if len(d) >= min_per_bin else np.nan),
            phot_bias_mag=float(np.median(d)) if len(d) >= min_per_bin else np.nan,
            n_phot=len(d),
        ))
    return rows


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", action="append", default=[], metavar="LABEL=SPEC",
                   help="repeatable. SPEC is path.npy, path.npz:key, or "
                        "draws.npy:{mean|index}. Split on the LAST '=', so the "
                        "label may contain '=' but the path may not.")
    p.add_argument("--data-dir", type=Path,
                   default=Path("/home/alex/noir_ml/global/ml-decon/data/m31bK50"))
    p.add_argument("--split", default="val")
    p.add_argument("--indices", default="0:16")
    p.add_argument("--psf-file", default="psf_50_true.fits")
    p.add_argument("--nsigma", type=float, default=3.0,
                   help="detection threshold in sky sigma (step 2)")
    p.add_argument("--truth-nsigma", type=float, default=None,
                   help="threshold for the INPUT catalogue. Default: same as "
                        "--nsigma, which builds input and measured catalogues "
                        "identically (the specified procedure) and makes 100%% "
                        "completeness attainable -- the truth scored against "
                        "itself then recovers everything. Setting this BELOW "
                        "--nsigma defines an input list reaching fainter than "
                        "the detection pass can see, which caps completeness "
                        "below 100%% for every method including the truth. "
                        "Measured here: detect/input 3/3 -> 100.0%%, 3/2 -> "
                        "86.4%%, 5/3 -> 73.1%%, 5/2 -> 63.1%%. The ceiling is "
                        "printed every run; quote it alongside any "
                        "completeness taken from a run where these differ.")
    p.add_argument("--box", type=int, default=3, help="aperture box (step 4)")
    p.add_argument("--edge", type=int, default=4,
                   help="border excluded from detection. 64x64 patches carry a "
                        "31x31 PSF, so sources near the edge are boundary "
                        "contaminated for every method.")
    p.add_argument("--common-threshold", action="store_true",
                   help="detect every method at the TRUTH's absolute threshold "
                        "(sky + nsigma*sigma measured on the truth patch) "
                        "instead of each image's own. Off by default because "
                        "the specified procedure says 'N sigma above sky', "
                        "which reads as per-image. Turn it ON to compare "
                        "methods at matched DEPTH: per-image thresholds give a "
                        "sparse solver a 1.57 mag deeper cut for free, which "
                        "flatters its completeness and inflates its spurious "
                        "rate at the same time.")
    p.add_argument("--match-radius", type=float, default=1.0,
                   help="matching radius in pixels (step 5)")
    p.add_argument("--n-bins", type=int, default=8)
    p.add_argument("--zeropoint", type=float, default=0.0)
    p.add_argument("--min-per-bin", type=int, default=5)
    p.add_argument("--no-check-scaling", dest="check_scaling",
                   action="store_false")
    p.add_argument("--out-dir", type=Path, default=Path("photom_bench"))
    args = p.parse_args()
    if not args.method:
        p.error("give at least one --method LABEL=SPEC")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cpu")
    nrm = json.loads((args.data_dir / "norm.json").read_text())
    onorm, inorm = TorchNorm(nrm["observed"]), TorchNorm(nrm["ideal"])
    obs_z = np.load(args.data_dir / f"{args.split}_observed.npy")
    ide_z = np.load(args.data_dir / f"{args.split}_ideal.npy")
    idx = parse_indices(args.indices, len(obs_z))
    observed = onorm.inverse(torch.from_numpy(obs_z[idx]).float()).numpy()[:, 0]
    truth = inorm.inverse(torch.from_numpy(ide_z[idx]).float()).numpy()[:, 0]
    B, H, W = truth.shape

    print(f"[step 0] benchmark dataset: {B} {args.split} patches of {H}x{W} px "
          f"from {args.data_dir.name}")

    if args.check_scaling:
        psf = np.asarray(load_kernel(args.psf_file), np.float64)
        psf /= psf.sum()
        sc = scaling_check(observed, truth, psf, dev)
        print(f"[step 1] blurry-vs-sharp flux scaling")
        for k, v in sc.items():
            print(f"           {k:38s} {v:.5f}")
        off = abs(sc["interior_obs_over_blurred_truth"] - 1.0)
        print(f"           -> interior scale is {off*100:.1f}% off unity; "
              + ("OK, photometry below is interpretable."
                 if off < 0.05 else
                 "WARNING >5%: every flux ratio below carries this factor."))
        (args.out_dir / "scaling_check.json").write_text(json.dumps(sc, indent=2))

    # ---- truth catalogue defines the magnitude axis and the bins ----------
    tn = args.truth_nsigma if args.truth_nsigma is not None else args.nsigma
    ref_flux = []
    for k in range(B):
        _, ifx = detect_and_measure(truth[k], tn, args.edge, args.box)
        ref_flux.extend(ifx.tolist())
    ref_mag = to_mag(ref_flux, args.zeropoint)
    ref_mag = ref_mag[np.isfinite(ref_mag)]
    edges = np.linspace(np.percentile(ref_mag, 1), np.percentile(ref_mag, 99),
                        args.n_bins + 1)
    print(f"[input]  {len(ref_mag)} input sources at {tn:g} sigma; "
          f"instrumental mag {edges[0]:.2f} .. {edges[-1]:.2f}")

    # ---- per method ------------------------------------------------------
    results, all_rows = [], []
    print(f"\n[steps 2-8] nsigma={args.nsigma} box={args.box} "
          f"edge={args.edge} match_radius={args.match_radius}")
    hdr = f"{'method':<28}{'compl':>8}{'spur':>8}{'med ratio':>11}{'sct(mag)':>10}"
    print(hdr); print("-" * len(hdr))

    # The truth image scored against itself: the CEILING no method can beat.
    # It is 100% when the input list and the detection pass use the same
    # threshold (the specified procedure), and LESS than 100% as soon as
    # --truth-nsigma reaches fainter than --nsigma, because the input list then
    # contains sources the detection pass cannot see in ANY image. Printing it
    # every run stops a sub-100% ceiling from looking like a bug.
    _, ceil_matched, _, _ = run_catalogue(
        truth, truth, args.nsigma, args.box, args.edge, args.match_radius,
        truth_nsigma=tn, common_threshold=args.common_threshold)
    ceiling = 100.0 * ceil_matched.mean()
    print(f"{'truth vs itself (CEILING)':<28}{ceiling:>7.1f}%{0.0:>7.1f}%"
          f"{1.000:>11.3f}{0.000:>10.3f}")
    for spec in args.method:
        # rsplit, not split: labels routinely contain '=' ("L1 tau=0.008").
        # The cost is that a PATH may not contain '=' -- fine for these.
        if "=" not in spec:
            p.error(f"--method needs LABEL=SPEC, got '{spec}'")
        label, path = spec.rsplit("=", 1)
        recon = load_recon(path, B)
        check_alignment(recon, truth, label)
        t_flux, matched, r_flux, n_meas = run_catalogue(
            recon, truth, args.nsigma, args.box, args.edge, args.match_radius,
            truth_nsigma=tn, common_threshold=args.common_threshold)
        t_mag = to_mag(t_flux, args.zeropoint)
        dmag = to_mag(r_flux, args.zeropoint) - t_mag        # step 7 residual
        rows = bin_stats(t_mag, matched, dmag, edges, args.min_per_bin)
        for r in rows:
            r["method"] = label
        all_rows.extend(rows)
        ok = matched & np.isfinite(dmag)
        results.append(dict(label=label, t_flux=t_flux, r_flux=r_flux,
                            matched=matched, dmag=dmag, rows=rows))
        print(f"{label:<28}{100.0*matched.mean():>7.1f}%"
              f"{100.0*(n_meas-matched.sum())/max(n_meas,1):>7.1f}%"
              f"{np.median(r_flux[ok]/t_flux[ok]):>11.3f}"
              f"{np.std(dmag[ok]):>10.3f}")

    with open(args.out_dir / "binned.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "mag_lo", "mag_hi",
                                           "mag_mid", "n_input", "n_recovered",
                                           "completeness", "completeness_err",
                                           "phot_scatter_mag", "phot_scatter_mad",
                                           "phot_bias_mag", "n_phot"])
        w.writeheader(); w.writerows(all_rows)

    # ---- figure 1: flux-flux, one panel per method -----------------------
    n = len(results)
    fig, ax = plt.subplots(1, n, figsize=(4.3 * n, 4.4), squeeze=False)
    for i, res in enumerate(results):
        a = ax[0][i]
        col, mk = SERIES[i % len(SERIES)]
        ok = res["matched"] & np.isfinite(res["r_flux"]) & (res["t_flux"] > 0)
        ok &= res["r_flux"] > 0
        a.plot(res["t_flux"][ok], res["r_flux"][ok], mk, ms=5, alpha=0.55,
               color=col, mec="none")
        lim = [min(res["t_flux"][ok].min(), res["r_flux"][ok].min()),
               max(res["t_flux"][ok].max(), res["r_flux"][ok].max())]
        a.plot(lim, lim, "--", color="0.35", lw=1.2, label="1:1")
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel(f"truth {args.box}x{args.box}-box flux (sky-subtracted)")
        if i == 0:
            a.set_ylabel(f"recon {args.box}x{args.box}-box flux (sky-subtracted)")
        med = np.median(res["r_flux"][ok] / res["t_flux"][ok])
        a.set_title(f"{res['label']}\nmedian ratio {med:.3f}  "
                    f"({ok.sum()} matched)", fontsize=10)
        a.grid(True, which="major", **GRID); a.set_axisbelow(True)
        a.legend(loc="upper left", frameon=False, fontsize=9)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out_dir / "flux_flux.png", dpi=140)
    plt.close(fig)

    # ---- figure 2: completeness / scatter / bias vs magnitude ------------
    fig, ax = plt.subplots(3, 1, figsize=(7.6, 11.4), sharex=True)
    for i, res in enumerate(results):
        col, mk = SERIES[i % len(SERIES)]
        r = res["rows"]
        if not r:
            continue
        m = np.array([x["mag_mid"] for x in r])
        ax[0].errorbar(m, [x["completeness"] for x in r],
                       yerr=[x["completeness_err"] for x in r],
                       marker=mk, ms=6, lw=2, color=col, capsize=3,
                       label=res["label"])
        ax[1].plot(m, [x["phot_scatter_mag"] for x in r], marker=mk, ms=6,
                   lw=2, color=col, label=res["label"])
        ax[2].plot(m, [x["phot_bias_mag"] for x in r], marker=mk, ms=6,
                   lw=2, color=col, label=res["label"])
    ax[0].set_ylabel("completeness  N_rec / N_in  [%]")
    ax[0].set_title(f"Completeness vs magnitude "
                    f"({args.nsigma:g}$\\sigma$ detection, "
                    f"{args.match_radius:g} px match)", fontsize=11)
    ax[0].set_ylim(0, 105)
    ax[1].set_ylabel("photometric error  $\\sigma(\\Delta$mag$)$")
    ax[1].set_title("Photometric scatter vs magnitude (lower is better)",
                    fontsize=11)
    ax[2].axhline(0.0, color="0.35", ls="--", lw=1.2)
    ax[2].set_ylabel("photometric bias  median $\\Delta$mag")
    ax[2].set_title("Photometric bias vs magnitude (0 is unbiased; "
                    "positive = recon too faint)", fontsize=11)
    ax[2].set_xlabel(f"truth instrumental magnitude  "
                     f"$-2.5\\log_{{10}}(F)$ + {args.zeropoint:g}   "
                     f"(fainter $\\rightarrow$)")
    for a in ax:
        a.grid(True, **GRID); a.set_axisbelow(True)
        a.legend(frameon=False, fontsize=9)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out_dir / "completeness_and_error.png", dpi=140)
    plt.close(fig)

    print(f"\nWrote {args.out_dir}/flux_flux.png, completeness_and_error.png, "
          f"binned.csv" + (", scaling_check.json" if args.check_scaling else ""))


if __name__ == "__main__":
    main()
