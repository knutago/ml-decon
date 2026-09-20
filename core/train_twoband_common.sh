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
# Checkpoints go to $SCRATCH when it exists -- the convention train_f435w_wide.sh
# documents for Vista ("--checkpoint-dir $SCRATCH/m32sim_arms/..."). Scratch is
# the fast filesystem and these are 67 MB a side written every time val improves;
# $WORK has a quota and is not meant for that traffic. Scratch IS PURGED, so copy
# best.pt to $WORK/ml-decon/m32sim_arms/ when the run finishes.
if [ -n "${OUT:-}" ]; then :
elif [ -n "${SCRATCH:-}" ]; then OUT=$SCRATCH/m32sim_arms/ckpt_m32sim_twoband_common
else OUT=$ML/m32sim_arms/ckpt_m32sim_twoband_common; fi

# EPOCHS 300, not the 600 the single-band arms used, and this is deliberate.
# Two bands is 11450 train pairs against 5725, so an epoch is twice the work:
# 5725 x 600 = 11450 x 300 = 3.4M samples, i.e. THE SAME GRADIENT-STEP BUDGET as
# the F555W run, reached in half the wall time. At the measured 61.5 s/epoch for
# 5725 pairs, 600 two-band epochs would be ~21 h -- and the trainer has NO
# --resume, so a walltime kill loses everything. 300 lands near 10.5 h, which
# fits a 24 h job with real margin. Raise it only with a longer -t.
EPOCHS=${EPOCHS:-300}
WORKERS=${WORKERS:-8}

# T-MAX FOR THE TWO TRAINING TERMS -- raised from 100, and this is a consequence
# of the shared map, not a free knob.
#
# Both terms exist to cover the t range the PnP prox actually calls the denoiser
# in. Under the OLD per-band maps that was t <~ 75, so 100 covered it. The common
# map rescaled the ideal z-domain by a DIFFERENT factor per band -- F435W's
# target std went 0.0531 -> 0.0451 (0.849x), F555W's 0.0438 -> 0.0664 (1.518x) --
# so a fixed sigma_z is now 1.18x too strong for B, 0.66x too weak for V, and the
# useful range moved UP. Measured on the 2026-09-19 checkpoint (guidance_probe.py,
# completeness against the sim truth):
#
#     F435W  t=75 0.77 | 90 0.85 | 100 0.85    <- saturated
#     F555W  t=75 0.32 | 90 0.40 | 100 0.46    <- STILL CLIMBING at the ceiling
#
# F555W is clipped by t_max=100, and it cannot just be raised at inference: past
# identity_t_max the fixed-point term was never trained and ||D(x)-x|| on true
# patches blows up (0.216 at t=100, 46.6 at t=200). The ceiling has to move at
# TRAINING time. That band asymmetry is what produced flux 1.544 (B) vs 0.931 (V)
# = 0.55 mag of spurious colour, on the one axis this change was for.
#
# WATCH THE DILUTION. identity rows are spread over [0, T_ID), so doubling the
# range halves the identity density per timestep at fixed IDENT_FRAC, and the
# low-t oversampling likewise thins. Before trusting the next checkpoint, check
# ||D(x)-x|| on true patches at t=50/100/150 with denoise_probe.py; if the
# fixed-point behaviour is worse than this run's, raise IDENT_FRAC toward
# 0.25-0.30 rather than putting T_ID back to 100.
T_ID=${T_ID:-200}
T_LOW=${T_LOW:-200}
IDENT_FRAC=${IDENT_FRAC:-0.15}
LOW_FRAC=${LOW_FRAC:-0.25}

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
  --epochs "$EPOCHS"
  --num-workers "$WORKERS"
  --batch-size 32
  --lr 2e-4
  --timesteps 1000
  --arch flat
  --channels 192
  --dilations 1,2,3,4,6,8,6,4,3,2,1
  --norm none                     # GroupNorm destroys absolute scale at block 0
  --p-uncond 0.15                 # keeps the null token trained, so --guidance works
  --identity-frac "$IDENT_FRAC"
  --identity-t-max "$T_ID"
  --low-t-frac "$LOW_FRAC"
  --low-t-max "$T_LOW"
  --sky-aug
  --sky-aug-lo "$SKY_LO"
  --sky-aug-hi "$SKY_HI")

case "${1:-show}" in
  data)
    for b in f435w f555w; do
      d=$ML/data/m32_sim_${b}_p128_common
      c=$ML/config/m32_sim_${b}_p128_common.yaml
      # "exists, skipping" is only safe if what exists was built with the
      # constants the config NOW pins. Changing a pinned value and re-running
      # otherwise skips silently and trains on the old encoding -- which is
      # exactly the failure this whole change is about, and it bit me once
      # already during development.
      if [ -f "$d/val_observed.npy" ]; then
        PYTHONPATH="$ML" "$PY" - "$d/norm.json" "$c" <<'EOF' || exit 1
import json, sys, yaml
norm = json.load(open(sys.argv[1]))
cfg = yaml.safe_load(open(sys.argv[2]))["data"]
bad = []
for ch in ("observed", "ideal"):
    for k in ("beta", "lo_s", "hi_s"):
        want = cfg.get(f"{ch}_asinh_{k}")
        if want is None:
            continue
        got = norm[ch][k]
        if abs(float(got) - float(want)) > 1e-6 * max(1.0, abs(float(want))):
            bad.append(f"    {ch}.{k}: on disk {got!r}, config pins {want!r}")
if bad:
    sys.exit("STALE DATASET -- built with different asinh constants:\n"
             + "\n".join(bad)
             + f"\n    Delete {sys.argv[1].rsplit('/',1)[0]} and re-run; do NOT "
               "train on it.\n    (The two bands must share one map or the "
               "trainer's load_dataset_norms will refuse anyway.)")
print(f"    {sys.argv[1]}: constants match the config")
EOF
        echo "$d exists and matches, skipping"
        continue
      fi
      echo "--- building $d"
      PYTHONPATH="$ML" "${DATA_CMD[@]}" "$c"
    done
    echo
    echo "Both norm.json must now differ ONLY in observed.median (the sky):"
    for b in f435w f555w; do
      echo -n "  $b: "; cat "$ML/data/m32_sim_${b}_p128_common/norm.json" | tr -d '\n '; echo
    done
    ;;
  train)
    shift                     # drop "train"; anything left is passed THROUGH
    [ -f "$DB/val_observed.npy" ] && [ -f "$DV/val_observed.npy" ] || {
      echo "datasets missing -- run './train_twoband_common.sh data' first" >&2; exit 1; }
    "$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),
              torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
    # "$@" last so a caller can override any default above, e.g.
    #   ./train_twoband_common.sh train --lr 1e-4
    PYTHONPATH="$ML" "${TRAIN_CMD[@]}" "$@"
    ;;
  *)
    echo "Two-band joint prior on the shared asinh map"
    echo "  data   $DB"
    echo "         $DV"
    echo "  out    $OUT"
    echo "  sky-aug flux offset [$SKY_LO, $SKY_HI] (re-solved for the common map)"
    echo "  ~11450 train pairs (5725 per band, naturally balanced)"
    echo "  t-max   identity $T_ID  low-t $T_LOW  (raised from 100: the common map"
    echo "          moved the useful range up and F555W was clipped at 100)"
    echo "  epochs  $EPOCHS  = the F555W run's gradient-step budget (5725x600),"
    echo "          ~10.5 h at its measured 61.5 s per 5725-pair epoch"
    echo "  NOTE the trainer has no --resume: a walltime kill loses the run."
    echo "       Size -t from the line above, and copy best.pt off \$SCRATCH."
    echo
    printf '  %q ' "${TRAIN_CMD[@]}"; echo
    ;;
esac
