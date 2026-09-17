#!/usr/bin/env bash
# ONE prior over BOTH bands, on the shared asinh map.
#
# Replaces the pair of single-band runs (train_f435w_wide.sh + the f555w arm).
#
# WHY ONE PRIOR. Whatever bias the prior carries -- bright-end flux deficit,
# source splitting, a background floor off zero -- is COMMON-MODE across the
# bands and subtracts out of B-V. Two separately-trained priors have two
# different biases and the difference lands straight on the colour axis, which
# is the one axis a CMD actually measures. Measured on the 512^2 cut: one
# shared prior took the colour slope 0.156 -> 0.027, scatter 0.520 -> 0.347.
#
# WHY THE SHARED MAP IS A PREREQUISITE. beta is normally fit per band and, on a
# sparse catalogue render, falls back to the median non-zero pixel -- which
# tracks how bright THAT band's stars are (247.3 in F435W, 945.5 in F555W). So
# the per-band map divides the colour out: a measured 3.86x flux ratio reached
# the two networks as 0.87x in z. The *_p128_common datasets pin one map, so a
# given flux means a given z in both filters. Without that, concatenating the
# two bands trains the model on a contradiction, and the trainer refuses
# (load_dataset_norms).
#
# DO NOT point this at the old m32_sim_*_p128 datasets, and do not run an old
# checkpoint on the new ones -- either reproduces the cross-band bug with the
# sign flipped.
#
# Usage:  ./train_twoband_common.sh data     # build both datasets
#         ./train_twoband_common.sh train    # the joint run (needs a GPU)
#         ./train_twoband_common.sh          # print and exit
set -euo pipefail
ML=${ML_DECON:-$WORK/ml-decon}
PY=${PY:-$ML/.venv/bin/python}
OUT=${OUT:-$ML/m32sim_arms/ckpt_m32sim_twoband_common}

DB=$ML/data/m32_sim_f435w_p128_common
DV=$ML/data/m32_sim_f555w_p128_common

# --sky-aug-lo/-hi are a FLUX offset pushed through the dataset's own norm, so
# they are normalization-specific and had to be re-solved for the common map.
# Target = the z-shift the trained F555W arm actually applied (+0.1451 at
# z=0.15, +0.0937 at 0.30). Solved numerically, and under the common map the
# required offset came out IDENTICAL in both bands (117.6 / 101.2 / 118.8 at
# the three anchors) -- which is the shared map's invariance confirming itself,
# and why one range now serves both. Following the existing convention of
# taking the z=0.30 anchor for hi and hi/100 for lo:
SKY_LO=${SKY_LO:-1.0}
SKY_HI=${SKY_HI:-101.0}
# CHECK THE TRAINER'S OWN [sky-aug] TABLE at startup. It prints the resulting
# z shift and it should read ~+0.145 at z=0.15 and ~+0.094 at 0.30 for BOTH
# maps. If it does not, the datasets were regenerated with different constants
# and these need re-solving.

DATA_CMD=("$PY" "$ML/dataset/gen_data.py")

TRAIN_CMD=("$PY" "$ML/core/train_conditional_diffusion.py"
  --data-dir       "$DB"          # repeatable: this is what makes it two-band
  --data-dir       "$DV"
  --checkpoint-dir "$OUT"
  --epochs 600
  --batch-size 32
  --lr 2e-4
  --timesteps 1000
  --arch flat
  --channels 192
  --dilations 1,2,3,4,6,8,6,4,3,2,1
  --norm none                     # GroupNorm destroys absolute scale at block 0
  --p-uncond 0.15                 # keeps the null token trained, so --guidance works
  --identity-frac 0.15
  --identity-t-max 100
  --low-t-frac 0.25
  --low-t-max 100
  --sky-aug
  --sky-aug-lo "$SKY_LO"
  --sky-aug-hi "$SKY_HI")

case "${1:-show}" in
  data)
    for b in f435w f555w; do
      d=$ML/data/m32_sim_${b}_p128_common
      [ -f "$d/val_observed.npy" ] && { echo "$d exists, skipping"; continue; }
      echo "--- building $d"
      PYTHONPATH="$ML" "${DATA_CMD[@]}" "$ML/config/m32_sim_${b}_p128_common.yaml"
    done
    echo
    echo "Both norm.json must now differ ONLY in observed.median (the sky):"
    for b in f435w f555w; do
      echo -n "  $b: "; cat "$ML/data/m32_sim_${b}_p128_common/norm.json" | tr -d '\n '; echo
    done
    ;;
  train)
    [ -f "$DB/val_observed.npy" ] && [ -f "$DV/val_observed.npy" ] || {
      echo "datasets missing -- run './train_twoband_common.sh data' first" >&2; exit 1; }
    "$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),
              torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
    PYTHONPATH="$ML" "${TRAIN_CMD[@]}"
    ;;
  *)
    echo "Two-band joint prior on the shared asinh map"
    echo "  data   $DB"
    echo "         $DV"
    echo "  out    $OUT"
    echo "  sky-aug flux offset [$SKY_LO, $SKY_HI] (re-solved for the common map)"
    echo "  ~11450 train pairs (5725 per band, naturally balanced)"
    echo
    printf '  %q ' "${TRAIN_CMD[@]}"; echo
    ;;
esac
