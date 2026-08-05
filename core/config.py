"""Config schema and (de)serialization.

A run is fully specified by one YAML file: a top-level `seed` plus a `data` section.
Defaults live here in the dataclasses, so the *resolved* config (defaults filled in) is
what we dump alongside every run's outputs. Reproducing a run means reproducing these
parameters, not bytes. Unknown keys are a hard error, so a mistyped key fails loudly
instead of silently falling back to the default.

Model and training sections arrive in phase 2; the phase-1 evaluation workflow only
needs to describe the dataset.
"""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    # FITS paths are absolute and machine-specific: each member points at their own copy,
    # so these three fields do not transfer between machines. Never hardcode them in code.
    observed_fits: str  # input: the PSF-blurred, noisy observed image
    ideal_fits: str  # target: the underlying intensity field we are recovering
    out_dir: str
    patch_size: int = 64  # observed and ideal share a pixel grid; the model maps one to the other
    stride: int = 64  # patch grid step over the source image (>= patch_size => no overlap)
    observed_normalize: str = "asinh"  # asinh | log | linear — applied to the observed image
    ideal_normalize: str = "asinh"  # normalization for the sparse ideal field
    asinh_softening: float = 3.0  # asinh transition width in units of the sigma-clipped noise std
    split_block_size: int = 505  # side length (px) of blocks assigned wholesale to train or val
    val_fraction: float = 0.2


@dataclass
class Config:
    seed: int
    data: DataConfig


def _build(cls, values):
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**values)


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text())
    return Config(seed=raw["seed"], data=_build(DataConfig, raw.get("data", {})))


def dump_config(config, path):
    Path(path).write_text(yaml.safe_dump(asdict(config), sort_keys=False))
