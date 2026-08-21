"""Generate a deconvolution patch dataset from a paired FITS image.

Pipeline (run as `uv run python -m dataset.gen_data config/baseline.yaml`):

  1. Load the observed (trimmed) and ideal (untrimmed) FITS; they share a pixel grid.
  2. Lay a grid of patch top-left corners (step = data.stride).
  3. Assign each patch to a spatial block (data.split_block_size); whole blocks go to
     train or val (seeded), so train/val pixels never overlap.
  4. Cut observed and ideal patches at the SAME corners (they are co-registered).
  5. Fit a normalization on TRAIN pixels only, separately for observed and ideal, since
     the two live in very different value regimes. Apply to every patch.
  6. Save train/val observed+ideal arrays, the normalization params, a manifest, and the
     resolved config into data.out_dir.

Output arrays are float32, shape (N, 1, H, W), normalized to roughly [0, 1].
"""

import csv
import json
import sys
from pathlib import Path
from scipy.stats import theilslopes
import numpy as np
from astropy.io import fits

from core.config import dump_config, load_config
from core.normalize import fit_normalization


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

def fit_affine_photometry(observed, ideal, block=128, margin=64):
    """Fit observed ≈ gain * ideal + sky using block means.

    margin trims the border so PSF flux leaking off the frame doesn't
    bias the gain. Returns (gain, sky) in observed image units.
    """
    obs = observed[margin:-margin, margin:-margin]
    idl = ideal[margin:-margin, margin:-margin]

    h = (obs.shape[0] // block) * block
    w = (obs.shape[1] // block) * block
    obs = obs[:h, :w].reshape(h // block, block, w // block, block)
    idl = idl[:h, :w].reshape(h // block, block, w // block, block)

    obs_mean = np.nanmean(obs, axis=(1, 3)).ravel()
    idl_mean = np.nanmean(idl, axis=(1, 3)).ravel()

    good = np.isfinite(obs_mean) & np.isfinite(idl_mean)
    gain, sky, lo, hi = theilslopes(obs_mean[good], idl_mean[good])
    return gain, sky, (lo, hi)

def main(config_path):
    config = load_config(config_path)
    data = config.data
    out_dir = Path(data.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.seed)
    observed_image = load_fits(data.observed_fits)

    ideal_image = load_fits(data.ideal_fits) 
    ideal_image = load_fits(data.ideal_fits)
    if observed_image.shape != ideal_image.shape:
        raise ValueError(f"observed {observed_image.shape} and ideal {ideal_image.shape} "
                         "FITS must share a pixel grid")
    # Crop to the real footprint BEFORE anything is fitted. Both the photometry
    # fit and the normalization are fitted from pixel values, so zero padding
    # left in at this point corrupts them silently.
    if data.crop is not None:
        if data.crop == "auto":
            valid = (observed_image != 0) & (ideal_image != 0)
            ys, xs = np.where(valid)
            if not len(ys):
                raise ValueError("crop='auto' but no pixel is nonzero in both images")
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        else:
            y0, y1, x0, x1 = (int(v) for v in data.crop)
        before = observed_image.shape
        observed_image = observed_image[y0:y1, x0:x1]
        ideal_image = ideal_image[y0:y1, x0:x1]
        zf = 100.0 * float((observed_image == 0).mean())
        print(f"[crop] {before} -> [{y0}:{y1},{x0}:{x1}] = {observed_image.shape}"
              f"   ({zf:.1f}% of the cropped region is still exactly zero)")

    gain, sky, ci = fit_affine_photometry(observed_image, ideal_image,
                                      block=data.patch_size,
                                      margin=4 * data.patch_size) #compute scale flux difference between ideal and observed iamges
    # The fit is always REPORTED, so the two variants stay comparable and the
    # numbers that were applied (or deliberately not) are in the log either way.
    if data.fit_photometry:
        observed_image = (observed_image - sky) / gain
    else:
        print(f"[photometry] fit_photometry=false: observed left in NATIVE "
              f"units, sky included. Would have applied (obs - {sky:.6g}) / "
              f"{gain:.6g}.")
        print(f"[photometry]   the pair is therefore NOT photometrically "
              f"matched -- the network must absorb the gain, and flux ratios "
              f"against the ideal carry a factor {gain:.6g}.")
    height, width = observed_image.shape

    patch_size = data.patch_size
    corners = grid_corners(height, width, patch_size, data.stride)
    splits, block_of = assign_splits(corners, patch_size, data.split_block_size,
                                     data.val_fraction, rng)

    observed_raw = cut_patches(observed_image, corners, patch_size)
    ideal_raw = cut_patches(ideal_image, corners, patch_size)

    splits = np.array(splits)
    train_mask = splits == "train"
    observed_norm = fit_normalization(observed_raw[train_mask], data.observed_normalize,
                                      asinh_softening=data.asinh_softening,
                                      asinh_beta=data.observed_asinh_beta)
    ideal_norm = fit_normalization(ideal_raw[train_mask], data.ideal_normalize,
                                   asinh_softening=data.asinh_softening,
                                   asinh_beta=data.ideal_asinh_beta)

    # Report the fitted transforms. A bad normalization is invisible in the patches
    # themselves and only surfaces much later as a bad reconstruction, so put the numbers
    # that matter -- the knee and where the sources actually land in z -- in the log.
    print(f"gain={gain:.6g} sky={sky:.6g} (CI {ci[0]:.6g} .. {ci[1]:.6g})")
    for name, norm, raw in (("observed", observed_norm, observed_raw[train_mask]),
                            ("ideal", ideal_norm, ideal_raw[train_mask])):
        z = norm.forward(raw)
        print(f"{name:>8} norm: {norm.to_dict()}")
        print(f"{'':>8}  train z p1/p50/p99/p99.9 = {np.percentile(z, 1):.4f} "
              f"{np.percentile(z, 50):.4f} {np.percentile(z, 99):.4f} "
              f"{np.percentile(z, 99.9):.4f}   flux exactly zero: {np.mean(raw == 0):.2%}")

    observed = observed_norm.forward(observed_raw).astype(np.float32)
    ideal = ideal_norm.forward(ideal_raw).astype(np.float32)

    for name in ("train", "val"):
        mask = splits == name
        np.save(out_dir / f"{name}_observed.npy", observed[mask])
        np.save(out_dir / f"{name}_ideal.npy", ideal[mask])
        print(f"{name}: {int(mask.sum())} patches  "
              f"observed{observed[mask].shape[1:]} ideal{ideal[mask].shape[1:]}")

    (out_dir / "norm.json").write_text(json.dumps(
        {"observed": observed_norm.to_dict(), "ideal": ideal_norm.to_dict()}, indent=2))
    with open(out_dir / "manifest.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "corner_y", "corner_x", "block_y", "block_x", "split"])
        for index, ((y, x), split) in enumerate(zip(corners, splits)):
            block_y, block_x = block_of[(y, x)]
            writer.writerow([index, y, x, block_y, block_x, split])
    dump_config(config, out_dir / "resolved_config.yaml")
    print(f"wrote dataset to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
