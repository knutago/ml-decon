"""Generate a LINEAR-domain patch dataset plus the asinh stretch used as a LOSS.

Run:
    python gen_data_linear.py <config>.yaml

This is a duplicate of ml-decon's `dataset/gen_data.py` with one change of intent:
the patches are stored in a LINEAR (affine) domain instead of an asinh-stretched
one, and the asinh map is emitted alongside them as `ideal_loss_stretch` so the
trainer can measure its loss in asinh space while the diffusion model itself
operates on linear flux.

Why split the two
-----------------
Storing patches in asinh makes the DDPM's state space compressed at the bright
end: an error dz in the normalized asinh value maps to dflux = beta*cosh(...)*dz,
which on this dataset reaches ~14.5 mag^-1 at the bright end. Every bright-source
artifact is therefore amplified on the way back to flux. Training on linear
patches removes that amplification -- the model adds and removes noise in units
of flux, the way a DDPM normally does.

The asinh map is still the right way to *weight* the loss (it is roughly uniform
in magnitude error, so faint sources are not drowned out by bright ones), so it
survives as the loss stretch rather than as the storage domain.

`ideal_loss_stretch` is the same asinh map that `gen_data.py` would have applied,
rebased onto the stored linear values, so
    stretch(linear_patch) == asinh_normalized_patch
to float32 precision. Loss numbers are therefore directly comparable between a
run of this pipeline and a run of the original asinh one.

The anchor
----------
`anchor_percentile` chooses which train-pixel flux maps to 1.0 (100.0 = the true
min-max of the original LinearNorm). It never clips and never removes
information: the map stays affine and exactly invertible whatever the anchor, and
pixels above the anchor simply land above 1.0. What it *does* control is where the
DDPM's fixed unit-variance noise sits relative to the data. On m31bK50 with
anchor 100.0 the noise at t=0 already corresponds to raw flux ~11.4 while the
99.9th percentile of the ideal image is 6.09, so the forward process erases all
but the brightest ~0.05% of pixels at the very first step. Lowering the anchor
rescales the data up relative to that fixed noise floor; it is the only knob that
changes which sources the diffusion process can represent at all.
"""

import csv
import json
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits

# This script lives in mycode/ but reuses ml-decon's normalization classes so the
# invertibility contract (inverse(forward(x)) == x) has a single tested source.
ML_DECON_ROOT = Path("/home/alex/noir_ml/global/ml-decon")
sys.path.insert(0, str(ML_DECON_ROOT))

from core.normalize import AsinhNorm, LinearNorm  # noqa: E402


@dataclass
class LinearDataConfig:
    """Same shape as ml-decon's DataConfig minus the normalization choices.

    `observed_normalize` / `ideal_normalize` are deliberately absent: this
    pipeline is linear by definition, and the asinh rebase below is only valid
    against an affine storage map. A config written for `dataset/gen_data.py`
    will fail to load here on those keys, which is the intended loud failure.
    """

    observed_fits: str  # input: the PSF-blurred, noisy observed image
    ideal_fits: str  # target: the underlying intensity field we are recovering
    out_dir: str
    patch_size: int = 64
    stride: int = 64
    anchor_percentile: float = 100.0  # train-pixel percentile mapped to 1.0; 100 = min-max
    loss_softening: float = 1.0  # asinh transition width, in sigma-clipped noise std
    split_block_size: int = 505
    val_fraction: float = 0.2


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text())
    values = raw.get("data", {})
    unknown = set(values) - {f.name for f in fields(LinearDataConfig)}
    if unknown:
        raise ValueError(f"unknown keys for LinearDataConfig: {sorted(unknown)}")
    return raw["seed"], LinearDataConfig(**values)


def dump_resolved_config(seed, data, path):
    Path(path).write_text(yaml.safe_dump({"seed": seed, "data": asdict(data)},
                                         sort_keys=False))


def load_fits(path):
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim == 2:
                return np.ascontiguousarray(hdu.data, dtype=np.float32)
    raise ValueError(f"no 2-d image HDU found in {path}")


def grid_corners(height, width, patch_size, stride):
    ys = range(0, height - patch_size + 1, stride)
    xs = range(0, width - patch_size + 1, stride)
    return [(y, x) for y in ys for x in xs]


def assign_splits(corners, patch_size, block_size, val_fraction, rng):
    """Map each patch to its (block_y, block_x); send a seeded subset of blocks to val."""
    blocks = sorted({((y + patch_size // 2) // block_size, (x + patch_size // 2) // block_size)
                     for y, x in corners})
    num_val = round(val_fraction * len(blocks))
    permuted = [blocks[i] for i in rng.permutation(len(blocks))]
    val_blocks = set(permuted[:num_val])

    block_of = {}
    splits = []
    for y, x in corners:
        block = ((y + patch_size // 2) // block_size, (x + patch_size // 2) // block_size)
        block_of[(y, x)] = block
        splits.append("val" if block in val_blocks else "train")
    return splits, block_of


def cut_patches(image, corners, patch_size):
    patches = np.empty((len(corners), 1, patch_size, patch_size), dtype=np.float32)
    for index, (y, x) in enumerate(corners):
        patches[index, 0] = image[y:y + patch_size, x:x + patch_size]
    return patches


def fit_linear_anchor(pixels, anchor_percentile):
    """LinearNorm mapping [min, percentile] -> [0, 1].

    At anchor_percentile = 100 this reproduces LinearNorm.fit exactly. Below 100
    the map is unchanged in form (still affine, still exactly invertible); only
    the scale differs, and pixels above the anchor exceed 1.0 rather than clip.
    """
    flat = np.asarray(pixels).reshape(-1)
    return LinearNorm(lo=float(flat.min()),
                      hi=float(np.percentile(flat, anchor_percentile)))


def rebase_asinh_onto_linear(asinh_norm, linear_norm):
    """Express an asinh map fit on RAW flux in terms of linear-normalized values.

    With u = (x - lo) / (hi - lo), substituting x = u*(hi - lo) + lo into
        s(x) = arcsinh((x - median) / beta)
    gives s(u) = arcsinh((u - median_u) / beta_u) with median_u and beta_u scaled
    by the same (hi - lo). lo_s/hi_s are already in s units and carry over
    unchanged, so the composed map returns exactly the value the asinh dataset
    would have stored.
    """
    scale = linear_norm.hi - linear_norm.lo
    return {"method": "asinh",
            "median": (asinh_norm.median - linear_norm.lo) / scale,
            "beta": asinh_norm.beta / scale,
            "lo_s": asinh_norm.lo_s,
            "hi_s": asinh_norm.hi_s}


def main(config_path):
    seed, data = load_config(config_path)
    out_dir = Path(data.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    observed_image = load_fits(data.observed_fits)
    ideal_image = load_fits(data.ideal_fits)
    if observed_image.shape != ideal_image.shape:
        raise ValueError(f"observed {observed_image.shape} and ideal {ideal_image.shape} "
                         "FITS must share a pixel grid")
    height, width = observed_image.shape

    patch_size = data.patch_size
    corners = grid_corners(height, width, patch_size, data.stride)
    splits, block_of = assign_splits(corners, patch_size, data.split_block_size,
                                     data.val_fraction, rng)

    observed_raw = cut_patches(observed_image, corners, patch_size)
    ideal_raw = cut_patches(ideal_image, corners, patch_size)

    splits = np.array(splits)
    train_mask = splits == "train"
    observed_norm = fit_linear_anchor(observed_raw[train_mask], data.anchor_percentile)
    ideal_norm = fit_linear_anchor(ideal_raw[train_mask], data.anchor_percentile)

    # The loss stretch is fit on the same TRAIN ideal pixels the original asinh
    # pipeline used, then rebased onto the linear storage domain.
    loss_asinh = AsinhNorm.fit(ideal_raw[train_mask].reshape(-1),
                               asinh_softening=data.loss_softening)
    loss_stretch = rebase_asinh_onto_linear(loss_asinh, ideal_norm)

    observed = observed_norm.forward(observed_raw).astype(np.float32)
    ideal = ideal_norm.forward(ideal_raw).astype(np.float32)

    for name in ("train", "val"):
        mask = splits == name
        np.save(out_dir / f"{name}_observed.npy", observed[mask])
        np.save(out_dir / f"{name}_ideal.npy", ideal[mask])
        print(f"{name}: {int(mask.sum())} patches  "
              f"observed{observed[mask].shape[1:]} ideal{ideal[mask].shape[1:]}")

    # Report where the DDPM's own noise lands relative to the stored data. A
    # forward process whose t=0 noise already exceeds most of the field cannot
    # represent that field at any timestep, whatever the loss says.
    ideal_scale = ideal_norm.hi - ideal_norm.lo
    percentiles = np.percentile(ideal[train_mask], [50, 99, 99.9, 100])
    print(f"anchor p{data.anchor_percentile} -> ideal lo={ideal_norm.lo:.6g} "
          f"hi={ideal_norm.hi:.6g}")
    print("ideal (stored) pct 50/99/99.9/100 = "
          + " ".join(f"{v:.6g}" for v in percentiles))
    print(f"DDPM noise at t=0 is ~0.0064 in these units "
          f"(= raw flux {0.0064 * ideal_scale:.4g}); pixels below it are erased "
          f"by the forward process at every timestep")

    (out_dir / "norm.json").write_text(json.dumps(
        {"observed": observed_norm.to_dict(),
         "ideal": ideal_norm.to_dict(),
         "ideal_loss_stretch": loss_stretch}, indent=2))
    with open(out_dir / "manifest.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "corner_y", "corner_x", "block_y", "block_x", "split"])
        for index, ((y, x), split) in enumerate(zip(corners, splits)):
            block_y, block_x = block_of[(y, x)]
            writer.writerow([index, y, x, block_y, block_x, split])
    dump_resolved_config(seed, data, out_dir / "resolved_config.yaml")
    print(f"wrote dataset to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
