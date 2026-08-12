#!/usr/bin/env bash
# Generate the linear-domain patch dataset at anchor_percentile 100.0.
#
#   ./run_gen_linear_p100.sh
#
# Writes ~500 MB (train/val observed+ideal .npy) to the out_dir named in
# config_linear_p100.yaml, plus norm.json, manifest.csv and resolved_config.yaml.
# CPU only, no torch needed. Re-running overwrites the output directory.
#
# Override the interpreter if you are not on this machine:
#   PYTHON=python ./run_gen_linear_p100.sh
set -euo pipefail

PYTHON="${PYTHON:-/opt/conda/miniconda3/envs/py313/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-$HERE/config_linear_p100.yaml}"

"$PYTHON" "$HERE/gen_data_linear.py" "$CONFIG"

# The generator prints where the DDPM's t=0 noise lands relative to the stored
# data. At this anchor it reports a noise floor around raw flux 11.4 against a
# 99.9th percentile of 6.09 -- that is expected here, not a failure, but it is
# the number to check first if the trained prior turns out to smooth the field.
