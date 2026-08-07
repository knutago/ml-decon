"""Verify the Python reproduction against every IDL _avg/_scl/_obs2/_trim pair and
build the scale/sky table.

    uv run python -m script.verify_pairs <data_dir> [out_csv]

<data_dir> holds the IDL FITS (the *_avg/_scl/_obs2/_trim.fits files, which live outside
the repo). out_csv defaults to <data_dir>/repro_table.csv. For each of the 12 (band,
config) tags it runs the three stages of dataset.idl_repro against the delivered files,
staged per the agreed contract:

    rescale  avg  -> scl   exact: scl must equal avg*k to floating tolerance
    addnoise scl  -> obs2  backbone from source constants; noise checked statistically
                           (mean~0), and scale/sky/nexp cross-checked against the files
    dotrim   obs2 -> trim  byte-identical center crop (the one truly reproducible step)

It prints a PASS/FAIL line per stage and writes the table with the rescale factor, the
source and file-recovered addnoise scale and sky, nexp, and the combined avg->trim affine
constants. All quantities are in raw FITS ADU. The pure-logic contract tests (no data) are
in test/test_idl_repro.py.
"""

import csv
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

from dataset import idl_repro

BANDS = ["J", "H", "K"]
CONFIGS = ["20", "30", "50", "100"]

# Tolerances. rescale is a pure multiply, so scl/avg should be constant to ~float32 eps
# (the FITS were written single precision). dotrim must be exact.
RESCALE_RTOL = 1e-5


def load(data_dir, tag, stage):
    return fits.getdata(data_dir / f"{tag}_{stage}.fits").astype(np.float64)


def verify_tag(data_dir, tag):
    """Run all three stages for one tag; return (row_dict, list_of_(stage, ok, note))."""
    avg = load(data_dir, tag, "avg")
    scl = load(data_dir, tag, "scl")
    obs2 = load(data_dir, tag, "obs2")
    trim = load(data_dir, tag, "trim")
    checks = []

    # --- rescale: scl == avg * k -------------------------------------------------
    k = idl_repro.rescale_factor(avg, scl)
    predicted_scl = idl_repro.apply_rescale(avg, k)
    nz = avg != 0
    rel = np.abs(predicted_scl[nz] - scl[nz]) / np.abs(scl[nz])
    rescale_max_rel = float(rel.max())
    rescale_ok = rescale_max_rel < RESCALE_RTOL
    checks.append(("rescale", rescale_ok, f"max rel dev {rescale_max_rel:.2e} (k={k:.6g})"))

    # --- addnoise: obs2 == (scale*scl + sky)/nexp + zero-mean shot/read noise ----
    # Backbone reproduced from the SOURCE constants; scale/sky also recovered from the
    # files independently, and the two must agree.
    band = tag.split("b")[1][0]
    config = tag.split("b")[1][1:]
    p = idl_repro.addnoise_params(band, config)
    scale_src, sky_src = p["scale"], p["sky"]
    scale_rec, sky_rec = idl_repro.fit_addnoise(scl, obs2)
    clean = idl_repro.apply_addnoise(scl, band, config)  # noise-free, from source
    resid = obs2 - clean
    resid_mean = float(resid.mean())
    resid_std = float(resid.std())
    noise = idl_repro.fit_noise_model(clean, resid)
    scale_dev = abs(scale_rec - scale_src) / scale_src
    sky_dev = abs(sky_rec - sky_src) / sky_src
    # source vs recovered agree; noise is zero-mean relative to its own scatter; the
    # recovered nexp (sqrt of the variance-fit divisor) matches the source nexp.
    nexp_recovered = noise["nexp"] ** 0.5
    # saturation: addnoise clips obs2 at SAT_LEVEL (1e5); flag how many pixels hit it.
    n_saturated = int(np.count_nonzero(obs2 >= idl_repro.SAT_LEVEL))
    addnoise_ok = (scale_dev < 1e-3 and sky_dev < 1e-3
                   and abs(resid_mean) < 1e-3 * resid_std
                   and abs(nexp_recovered - p["nexp"]) / p["nexp"] < 0.05)
    checks.append(("addnoise", addnoise_ok,
                   f"scale src={scale_src:.6g} rec={scale_rec:.6g} ({scale_dev:.1e})  "
                   f"sky src={sky_src:.5g} rec={sky_rec:.5g} ({sky_dev:.1e})  "
                   f"resid mean={resid_mean:.2e} std={resid_std:.4g}  "
                   f"nexp src={p['nexp']} rec={nexp_recovered:.0f}  "
                   f"saturated={n_saturated}"))

    # --- dotrim: trim == center crop of obs2 (byte identical) --------------------
    cropped = idl_repro.center_crop(obs2, trim.shape)
    dotrim_ok = np.array_equal(cropped, trim)
    checks.append(("dotrim", dotrim_ok,
                   f"dsz=({obs2.shape[0] - trim.shape[0]},{obs2.shape[1] - trim.shape[1]}) "
                   f"exact={dotrim_ok}"))

    row = {
        "tag": tag,
        "band": band,
        "config": config,
        "shape_full": f"{avg.shape[0]}x{avg.shape[1]}",
        "shape_trim": f"{trim.shape[0]}x{trim.shape[1]}",
        "exptime": idl_repro.EXPTIME,
        "nexp": p["nexp"],
        "rescale_k": k,
        "scale_src": scale_src,
        "scale_recovered": scale_rec,
        "sky_src": sky_src,
        "sky_recovered": sky_rec,
        "read_noise": idl_repro.READ_NOISE,
        # combined avg -> trim affine: trim = a*avg_crop + b (+noise)
        "avg_to_trim_a": k * scale_src,
        "avg_to_trim_b": sky_src,
        "resid_mean": resid_mean,
        "resid_std": resid_std,
        "n_saturated": n_saturated,
        "rescale_max_rel": rescale_max_rel,
        "dotrim_exact": dotrim_ok,
    }
    return row, checks


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    data_dir = Path(argv[0]).resolve()
    out = Path(argv[1]).resolve() if len(argv) > 1 else data_dir / "repro_table.csv"

    tags = [f"m31b{b}{c}" for c in CONFIGS for b in BANDS]
    rows = []
    all_ok = True
    for tag in tags:
        row, checks = verify_tag(data_dir, tag)
        rows.append(row)
        stages = "  ".join(f"{name}:{'PASS' if ok else 'FAIL'}" for name, ok, _ in checks)
        print(f"[{tag:10s}] {stages}")
        for name, ok, note in checks:
            print(f"    {name:8s} {note}")
            all_ok = all_ok and ok

    fields = list(rows[0].keys())
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {out}")
    print("\n=== scale / sky / rescale table (ADU; scale & sky are the per-pixel obs2 "
          "constants = raw/nexp) ===")
    hdr = (f"{'tag':10s} {'band':4s} {'cfg':4s} {'nexp':>5s} {'rescale_k':>12s} "
           f"{'scale':>10s} {'sky':>10s} {'trim=a*avg+b: a':>16s} {'b':>10s}")
    print(hdr)
    for r in rows:
        print(f"{r['tag']:10s} {r['band']:4s} {r['config']:4s} {r['nexp']:5d} "
              f"{r['rescale_k']:12.5g} {r['scale_src']:10.5g} {r['sky_src']:10.5g} "
              f"{r['avg_to_trim_a']:16.5g} {r['avg_to_trim_b']:10.5g}")

    print("\nALL STAGES PASS" if all_ok else "\nSOME STAGES FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
