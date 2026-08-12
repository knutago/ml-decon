#!/usr/bin/env python3
"""Generate luminosity functions from a star list and predict crowding errors.

For each requested band this script bins the magnitudes into a luminosity
function, converts the counts to a surface density (stars per square degree),
and feeds that to `compCrowdError` in crowding.py to predict the crowding
(confusion) error as a function of magnitude (Olsen, Blum & Rigaut 2003). It
also predicts the completeness from that crowding error using the tanh fit to
the left panel of their Fig. 14 (see fit_completeness.py).

Run:
    python crowding_lf.py m31b.out --binsize 0.25 --plot m31b_crowd.png
    python crowding_lf.py m31b.out --binsize 0.25 --color J K

Note: compCrowdError needs the area over which the luminosity function counts
were measured. We pass the raw histogram counts together with the field area
in arcsec² (from the pixel extent of the star list and the pixel scale) via its
lumAreaArcsec argument, and it normalizes internally.
"""

import argparse
import sys

import numpy as np
import pandas as pd

import crowding
from fit_completeness import completeness_model

# Band central wavelengths in microns, used only to derive the diffraction-
# limited seeing when the user does not supply one.
BAND_WAVELENGTH_MICRON = {"I": 0.90, "J": 1.25, "H": 1.65, "K": 2.20}

# Best-fit parameters of the completeness law completeness_model, fit with the
# amplitude fixed at f0=0.5 to the digitized LEFT panel (completeness vs
# single-band magnitude crowding error) of Fig. 14 of Olsen, Blum & Rigaut
# (2003); see fit_completeness.py and olsen_fig14_left.txt. Applicable to the
# per-band magnitude crowding error only, not to the color error.
COMPLETENESS_F0 = 0.5
COMPLETENESS_SIGMA0 = 0.2359
COMPLETENESS_SIGMA_S = 0.1238


def diffraction_limited_fwhm(band, diam_m):
    """Diffraction-limited FWHM in arcsec: FWHM = 1.028 * lambda / D.

    1.028 is the FWHM coefficient for an unobstructed circular aperture
    (e.g. Racine 1996). lambda from BAND_WAVELENGTH_MICRON, D in metres.
    """
    lam_m = BAND_WAVELENGTH_MICRON[band] * 1e-6
    radians = 1.028 * lam_m / diam_m
    return radians * 206265.0  # rad -> arcsec


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("infile", help="input star list (e.g. m31b.out)")
    p.add_argument("-b", "--binsize", type=float, default=0.25,
                   help="magnitude bin width (default: 0.25)")
    p.add_argument("--bands", nargs="+", default=["J", "H", "K"],
                   help="which band columns to use (default: J H K)")
    p.add_argument("--range", nargs=2, type=float, metavar=("MIN", "MAX"),
                   default=None,
                   help="magnitude range for the bins "
                        "(default: data min/max across the selected bands)")
    p.add_argument("--apix", type=float, default=0.005,
                   help="pixel scale in arcsec/pixel (default: 0.005, per "
                        "addnoise.pro)")
    p.add_argument("--diam", type=float, default=30.0,
                   help="telescope diameter in metres, for the diffraction-"
                        "limited seeing default (default: 30)")
    p.add_argument("--seeing", nargs="+", type=float, default=None,
                   help="seeing FWHM in arcsec, one value per band (default: "
                        "diffraction-limited per band from --diam)")
    p.add_argument("--color", nargs=2, metavar=("BAND1", "BAND2"), default=None,
                   help="also compute the color (BAND1-BAND2) crowding error "
                        "via compColorCrowdError, e.g. --color J K")
    p.add_argument("--prefix", default=None,
                   help="output text-file prefix (default: input filename stem)")
    p.add_argument("--plot", default=None,
                   help="if given, also save a plot to this file (e.g. crowd.png)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.seeing is not None and len(args.seeing) != len(args.bands):
        sys.exit(f"--seeing needs one value per band "
                 f"({len(args.bands)} bands, got {len(args.seeing)})")

    def seeing_for(band):
        """Per-band seeing: user value if supplied for it, else diffraction limit."""
        if args.seeing is not None and band in args.bands:
            return args.seeing[args.bands.index(band)]
        return diffraction_limited_fwhm(band, args.diam)

    # Bands whose magnitudes we need to read: the histogram bands plus any
    # color bands not already among them.
    bands_needed = list(args.bands)
    if args.color:
        bands_needed += [b for b in args.color if b not in bands_needed]

    # The file has 7 data columns but only 6 header names (X Y I J H K plus a
    # 7th column that is stellar mass). Naming all columns explicitly avoids
    # pandas promoting the first column to the index and shifting every band.
    colnames = ["X", "Y", "I", "J", "H", "K", "mass"]
    read_cols = ["X", "Y"] + bands_needed
    try:
        data = pd.read_csv(args.infile, sep=r"\s+", header=0,
                           names=colnames, usecols=read_cols)
    except ValueError as e:
        sys.exit(f"Error reading columns {read_cols} from {args.infile}: {e}")

    # Field area on the sky, from the pixel extent of the star positions and
    # the pixel scale. compCrowdError wants the LF per square degree.
    span_x = (data["X"].max() - data["X"].min()) * args.apix
    span_y = (data["Y"].max() - data["Y"].min()) * args.apix
    area_arcsec2 = span_x * span_y
    area_deg2 = area_arcsec2 / 3600.0**2
    print(f"Field: {span_x:.2f} x {span_y:.2f} arcsec = {area_deg2:.3e} deg² "
          f"(apix={args.apix})")

    # Magnitude range and bin edges, shared across all bands used.
    if args.range is not None:
        mlo, mhi = args.range
    else:
        mlo = float(np.floor(min(data[b].min() for b in bands_needed)))
        mhi = float(np.ceil(max(data[b].max() for b in bands_needed)))
    nbins = int(np.ceil((mhi - mlo) / args.binsize))
    edges = mlo + args.binsize * np.arange(nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    prefix = args.prefix or args.infile.rsplit(".", 1)[0]

    results = {}
    for band in args.bands:
        seeing = seeing_for(band)

        mags = data[band].to_numpy()
        mags = mags[np.isfinite(mags)]
        counts, _ = np.histogram(mags, bins=edges)

        # Pass raw counts with the field area; compCrowdError normalizes
        # internally. lum_func (per deg²) is reported for reference only.
        lum_func = counts / area_deg2
        crowd_error = crowding.compCrowdError(centers, counts, seeing,
                                              lumAreaArcsec=area_arcsec2)
        # Predicted completeness from the crowding error, via the Olsen Fig. 14
        # left-panel fit (magnitude error -> completeness). Empty bins get a
        # crowd_error of 0 from compCrowdError, which would spuriously read as
        # fully complete; mask them since completeness is undefined with no stars.
        completeness = completeness_model(crowd_error, COMPLETENESS_F0,
                                          COMPLETENESS_SIGMA0, COMPLETENESS_SIGMA_S)
        completeness[counts == 0] = np.nan
        results[band] = (counts, crowd_error, completeness, seeing)

        outname = f"{prefix}_{band}_crowd.txt"
        header = (f"Crowding errors for band {band} from {args.infile}\n"
                  f"binsize={args.binsize}, seeing={seeing:.5f} arcsec, "
                  f"area={area_deg2:.6e} deg², N={counts.sum()}\n"
                  f"Crowding error per Olsen, Blum & Rigaut 2003 (AJ 126, 452); "
                  f"completeness from their Fig. 14 (left) fit "
                  f"(sigma0={COMPLETENESS_SIGMA0}, sigma_s={COMPLETENESS_SIGMA_S})\n"
                  f"mag_center  count  N_per_deg2  crowd_error  completeness")
        np.savetxt(outname,
                   np.column_stack([centers, counts, lum_func, crowd_error,
                                    completeness]),
                   fmt=["%10.4f", "%12d", "%16.6e", "%14.6e", "%12.6f"],
                   header=header)
        print(f"Wrote {outname}  (seeing={seeing:.5f}\", N={counts.sum()})")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        ax2 = ax.twinx()  # completeness on a shared magnitude axis
        for band, color in zip(args.bands, ["C0", "C1", "C2", "C3"]):
            _, crowd_error, completeness, seeing = results[band]
            ax.plot(centers, crowd_error, color=color,
                    label=f"{band} (seeing {seeing:.4f}\")")
            ax2.plot(centers, completeness, color=color, ls="--", alpha=0.7)
        ax.set_xlabel("Magnitude")
        ax.set_ylabel("Crowding error (mag)")
        ax.set_yscale("log")
        ax2.set_ylabel("Predicted completeness (dashed)")
        ax2.set_ylim(-0.02, 1.02)
        ax.legend(loc="center left")
        ax.set_title(f"Predicted crowding errors and completeness: {args.infile}")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"Wrote {args.plot}")

    if args.color:
        band1, band2 = args.color
        s1, s2 = seeing_for(band1), seeing_for(band2)

        m1 = data[band1].to_numpy()
        m2 = data[band2].to_numpy()
        good = np.isfinite(m1) & np.isfinite(m2)

        # Joint LF with rows = band2, cols = band1, matching compColorCrowdError's
        # meshgrid convention (axis 0 -> magVector2, axis 1 -> magVector1).
        lum_func12, _, _ = np.histogram2d(m2[good], m1[good], bins=[edges, edges])
        _, _, color_error = crowding.compColorCrowdError(
            centers, centers, lum_func12, s1, s2, lumAreaArcsec=area_arcsec2)

        # Long-format grid: one row per (band1, band2) cell. M1/M2 share
        # color_error's (band2, band1) shape so the columns stay aligned.
        grid1, grid2 = np.meshgrid(centers, centers)
        table = np.column_stack([grid1.ravel(), grid2.ravel(),
                                 color_error.ravel()])
        n_finite = int(np.isfinite(color_error).sum())

        outname = f"{prefix}_{band1}m{band2}_colorcrowd.txt"
        header = (f"Color ({band1}-{band2}) crowding errors from {args.infile}\n"
                  f"binsize={args.binsize}, seeing_{band1}={s1:.5f}, "
                  f"seeing_{band2}={s2:.5f} arcsec, area={area_deg2:.6e} deg²\n"
                  f"Color crowding error per Olsen, Blum & Rigaut 2003 "
                  f"(AJ 126, 452)\n"
                  f"{band1}_mag  {band2}_mag  color_crowd_error")
        np.savetxt(outname, table,
                   fmt=["%10.4f", "%10.4f", "%14.6e"], header=header)
        print(f"Wrote {outname}  (seeing {band1}={s1:.5f}\", {band2}={s2:.5f}\", "
              f"{n_finite}/{color_error.size} finite cells)")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6.5, 5))
        mesh = plt.pcolormesh(centers, centers, color_error, shading="nearest")
        plt.colorbar(mesh, label=f"{band1}-{band2} crowding error (mag)")
        plt.xlabel(f"{band1} mag")
        plt.ylabel(f"{band2} mag")
        plt.title(f"Color crowding error: {args.infile}")
        plt.tight_layout()
        plotname = f"{prefix}_{band1}m{band2}_colorcrowd.png"
        plt.savefig(plotname, dpi=120)
        print(f"Wrote {plotname}")


if __name__ == "__main__":
    main()
