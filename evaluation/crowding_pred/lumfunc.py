#!/usr/bin/env python3
"""Generate luminosity functions (star counts vs. magnitude) from a star list.

The input file (e.g. m31b.out) is whitespace-delimited with a header line:

    X  Y  I  J  H  K  mass

This script bins the J, H, and K magnitudes into histograms with a
user-specified bin size and writes the counts to text files (and,
optionally, a plot).

Examples
--------
    python lumfunc.py m31b.out --binsize 0.25
    python lumfunc.py m31b.out --binsize 0.5 --bands J K --plot lf.png
    python lumfunc.py m31b.out -b 0.2 --range 18 30
"""

import argparse
import sys

import numpy as np
import pandas as pd


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("infile", help="input star list (e.g. m31b.out)")
    p.add_argument("-b", "--binsize", type=float, required=True,
                   help="magnitude bin width")
    p.add_argument("--bands", nargs="+", default=["J", "H", "K"],
                   help="which band columns to histogram (default: J H K)")
    p.add_argument("--range", nargs=2, type=float, metavar=("MIN", "MAX"),
                   default=None,
                   help="magnitude range for the bins "
                        "(default: data min/max across the selected bands)")
    p.add_argument("--mass-min", type=float, default=None,
                   help="keep only stars with mass >= MASS_MIN (solar units)")
    p.add_argument("--mass-max", type=float, default=None,
                   help="keep only stars with mass <= MASS_MAX (solar units)")
    p.add_argument("--prefix", default=None,
                   help="output text-file prefix (default: input filename stem)")
    p.add_argument("--plot", default=None,
                   help="if given, also save a plot to this file (e.g. lf.png)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # The file has 7 data columns but only 6 header names (X Y I J H K plus an
    # unnamed 7th column, which appears to be stellar mass in solar units).
    # Supplying explicit names for all columns avoids pandas silently promoting
    # the first column to the index, which would shift every band over by one.
    # header=0 skips the file's own header row.
    colnames = ["X", "Y", "I", "J", "H", "K", "mass"]
    cut_mass = args.mass_min is not None or args.mass_max is not None
    read_cols = args.bands + ["mass"] if cut_mass else args.bands
    try:
        data = pd.read_csv(args.infile, sep=r"\s+", header=0,
                           names=colnames, usecols=read_cols)
    except ValueError as e:
        sys.exit(f"Error reading columns {read_cols} from {args.infile}: {e}")

    # Restrict to a stellar-mass range before histogramming, if requested.
    if cut_mass:
        keep = np.ones(len(data), dtype=bool)
        if args.mass_min is not None:
            keep &= data["mass"].to_numpy() >= args.mass_min
        if args.mass_max is not None:
            keep &= data["mass"].to_numpy() <= args.mass_max
        data = data[keep]
        lo = args.mass_min if args.mass_min is not None else "-inf"
        hi = args.mass_max if args.mass_max is not None else "+inf"
        print(f"Mass cut [{lo}, {hi}]: kept {len(data)} of {len(keep)} stars")

    # Determine common magnitude range and bin edges.
    if args.range is not None:
        mlo, mhi = args.range
    else:
        mlo = float(np.floor(min(data[b].min() for b in args.bands)))
        mhi = float(np.ceil(max(data[b].max() for b in args.bands)))

    # Bin edges from mlo to mhi (inclusive of mhi) at the requested step.
    nbins = int(np.ceil((mhi - mlo) / args.binsize))
    edges = mlo + args.binsize * np.arange(nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    prefix = args.prefix or args.infile.rsplit(".", 1)[0]
    mass_note = (f", mass in [{args.mass_min}, {args.mass_max}]"
                 if cut_mass else "")

    results = {}
    for band in args.bands:
        mags = data[band].to_numpy()
        mags = mags[np.isfinite(mags)]
        counts, _ = np.histogram(mags, bins=edges)
        results[band] = counts

        outname = f"{prefix}_{band}_lf.txt"
        header = (f"Luminosity function for band {band} from {args.infile}\n"
                  f"binsize = {args.binsize}, N = {counts.sum()}{mass_note}\n"
                  f"mag_center  count")
        np.savetxt(outname, np.column_stack([centers, counts]),
                   fmt=["%10.4f", "%12d"], header=header)
        print(f"Wrote {outname}  ({counts.sum()} stars)")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 5))
        for band in args.bands:
            plt.step(centers, results[band], where="mid", label=band)
        plt.xlabel("Magnitude")
        plt.ylabel(f"N per {args.binsize} mag bin")
        plt.yscale("log")
        plt.legend()
        plt.title(f"Luminosity functions: {args.infile}{mass_note}")
        plt.tight_layout()
        plt.savefig(args.plot, dpi=120)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
