"""Report the arithmetic that scales the original sharp image (avg) to a trimmed observation.

    uv run python -m script.report_scaling <trim_fits> [table_csv]

Given a *_trim.fits (e.g. m31bJ30_trim.fits), it parses the tag, reads the rescale factor k
from the scale/sky table (default: repro_table.csv next to the trim, produced by
script/verify_pairs.py), and prints the forward and inverse per-pixel operations from
dataset.idl_repro.scaling_recipe. Report only -- no pixels are modified; the trim is loaded
only to echo its shape and value range for identification.
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

from dataset import idl_repro

_TAG = re.compile(r"(m31b([JHK])(\d+))_trim\.fits$")


def parse_tag(path):
    """Return (tag, band, config) from a *_trim.fits filename, e.g. m31bJ30_trim.fits."""
    m = _TAG.search(Path(path).name)
    if not m:
        raise ValueError(f"not a recognized *_trim.fits name: {Path(path).name}")
    return m.group(1), m.group(2), m.group(3)


def read_rescale_k(table_csv, tag):
    """Look up rescale_k for tag in the CSV table written by verify_pairs.py."""
    with open(table_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["tag"] == tag:
                return float(row["rescale_k"])
    raise KeyError(f"tag {tag} not found in {table_csv}")


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    trim_path = Path(argv[0]).resolve()
    tag, band, config = parse_tag(trim_path)
    table_csv = Path(argv[1]).resolve() if len(argv) > 1 else trim_path.parent / "repro_table.csv"

    k = read_rescale_k(table_csv, tag)
    recipe = idl_repro.scaling_recipe(band, config, k)
    print(idl_repro.format_scaling_report(recipe, tag=tag))

    data = fits.getdata(trim_path)
    saturated = int(np.count_nonzero(data >= idl_repro.SAT_LEVEL))
    sat_frac = saturated / data.size
    print()
    print(f"  (trim {trim_path.name}: shape {data.shape[0]}x{data.shape[1]}, "
          f"range [{np.min(data):.6g}, {np.max(data):.6g}] ADU; k from {table_csv.name})")
    print(f"  saturated pixels (>= {idl_repro.SAT_LEVEL:g}): {saturated} "
          f"({100 * sat_frac:.4g}% of {data.size})")
    if saturated:
        print("  NOTE: saturated pixels are clipped; the inverse will not recover their "
              "true sharp-image value.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
