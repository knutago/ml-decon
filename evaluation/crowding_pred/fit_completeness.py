#!/usr/bin/env python3
"""Fit a completeness-vs-crowding-error relation to Olsen et al. (2003) Fig. 14.

Rathore et al. (2025, ApJ 978, 55), their Equation (5), model the completeness
fraction f as a function of the crowding error sigma with a tanh curve:

    f(sigma) = f0 * (1 - tanh((sigma - sigma0) / sigma_s))

They fit this to the RIGHT panel of Fig. 14 of Olsen, Blum & Rigaut (2003),
which uses the color crowding error (sigma_color). This script fits the same
functional form to the LEFT panel, which uses the single-band photometric
(magnitude) crowding error, sigma_magnitude. Data points were digitized from
the filled circles (seeing-limited simulation) of that panel; see
olsen_fig14_left.txt for provenance.

By default f0 is held fixed at 0.5 so completeness saturates at exactly 1;
pass --fit-f0 to fit the amplitude as a free parameter instead.

Run:
    python fit_completeness.py                     # fits olsen_fig14_left.txt
    python fit_completeness.py --plot fit.png
    python fit_completeness.py --fit-f0            # free amplitude
"""

import argparse

import numpy as np
from scipy.optimize import curve_fit


def completeness_model(sigma, f0, sigma0, sigma_s):
    """Rathore et al. (2025) Eq. 5: tanh completeness law.

    f0 sets the amplitude (~0.5 for a curve running 1 -> 0), sigma0 is the
    crowding error at the midpoint (f = f0), and sigma_s is the transition
    width. Larger sigma means more crowding and lower completeness.
    """
    return f0 * (1.0 - np.tanh((sigma - sigma0) / sigma_s))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datafile", nargs="?", default="olsen_fig14_left.txt",
                   help="two-column file: crowding_error  completeness "
                        "(default: olsen_fig14_left.txt)")
    p.add_argument("--f0", type=float, default=0.5,
                   help="value at which f0 is held fixed (default: 0.5, which "
                        "makes completeness saturate at exactly 1)")
    p.add_argument("--fit-f0", action="store_true",
                   help="fit f0 as a free parameter instead of fixing it")
    p.add_argument("--xlabel", default=r"$\sigma_{\rm magnitude}$",
                   help="x-axis label for the plot")
    p.add_argument("--plot", default=None,
                   help="if given, save the data+fit plot to this file")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    sigma, completeness = np.loadtxt(args.datafile, unpack=True)

    # By default f0 is held fixed at args.f0 (0.5 makes completeness saturate at
    # exactly 1) and only sigma0, sigma_s are fit; --fit-f0 frees the amplitude.
    # sigma0 starts near the middle of the data, sigma_s at a modest fraction of
    # the sigma range.
    if args.fit_f0:
        p0 = [args.f0, np.median(sigma), 0.1]
        popt, pcov = curve_fit(completeness_model, sigma, completeness, p0=p0,
                               maxfev=10000)
        f0, sigma0, sigma_s = popt
        perr = np.sqrt(np.diag(pcov))
    else:
        f0 = args.f0
        fixed = lambda s, sigma0, sigma_s: completeness_model(s, f0, sigma0, sigma_s)
        p0 = [np.median(sigma), 0.1]
        popt, pcov = curve_fit(fixed, sigma, completeness, p0=p0, maxfev=10000)
        sigma0, sigma_s = popt
        perr = np.concatenate([[0.0], np.sqrt(np.diag(pcov))])
    popt = [f0, sigma0, sigma_s]

    resid = completeness - completeness_model(sigma, *popt)
    rms = float(np.sqrt(np.mean(resid**2)))

    # sigma at which completeness crosses 0.5, from the fitted curve.
    sigma_half = sigma0 + sigma_s * np.arctanh(1.0 - 0.5 / f0)

    print(f"Fit of f = f0 * (1 - tanh((sigma - sigma0)/sigma_s)) to "
          f"{len(sigma)} points from {args.datafile}")
    f0_note = f"+/- {perr[0]:.4f}" if args.fit_f0 else "(fixed)"
    print(f"  f0      = {f0:.4f} {f0_note}")
    print(f"  sigma0  = {sigma0:.4f} +/- {perr[1]:.4f}")
    print(f"  sigma_s = {sigma_s:.4f} +/- {perr[2]:.4f}")
    print(f"  RMS residual        = {rms:.4f}")
    print(f"  sigma(completeness=0.5) = {sigma_half:.4f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        grid = np.linspace(0.0, sigma.max() * 1.05, 400)
        plt.figure(figsize=(7, 5))
        plt.scatter(sigma, completeness, s=18, color="0.35",
                    label="Olsen+2003 Fig. 14 (left, digitized)")
        plt.plot(grid, completeness_model(grid, *popt), "r-", lw=2,
                 label="tanh fit (Rathore+2025 Eq. 5)")
        plt.axhline(0.5, color="0.7", ls=":", lw=1)
        plt.axvline(sigma_half, color="0.7", ls=":", lw=1)
        plt.text(0.98, 0.95,
                 f"$f_0$={f0:.3f}\n$\\sigma_0$={sigma0:.3f}\n"
                 f"$\\sigma_s$={sigma_s:.3f}",
                 transform=plt.gca().transAxes, va="top", ha="right",
                 bbox=dict(boxstyle="round", fc="white", ec="0.7"))
        plt.xlabel(args.xlabel)
        plt.ylabel("Completeness")
        plt.ylim(-0.05, 1.05)
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(args.plot, dpi=120)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
