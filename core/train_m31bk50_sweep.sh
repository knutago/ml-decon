#!/bin/bash
# Retrain the conditional prior on m31bK50 with the current recipe.
#
# WHY ARM A IS NOT OPTIONAL
# -------------------------
# checkpoints_cond_diffusion_npy used to hold the m31bK50 conditional model and
# was overwritten on 2026-08-19 by an m32_klong run. There is therefore NO
# baseline checkpoint left to compare against -- only numbers in a document,
# produced by an older revision of this script. Arm A reproduces the retired
# recipe (GroupNorm, 64ch, 45-px RF) under today's code so "did the last month
# of changes help" is a controlled comparison rather than a comparison against
# prose.
#
# WHAT TO EXPECT, MEASURED BEFORE SPENDING THE GPU HOURS
# ------------------------------------------------------
# The two headline wins of the last month were both diagnosed on m32 and both
# target properties m31bK50 does not have:
#
#   --norm none   won +49.5 pp on the ZERO-PIXEL FLOOR, because m32's truth is
#                 a catalogue rendering that is ~95% exact zeros. m31bK50's
#                 ideal is 7.2% exact zeros / 92.8% populated -- it is a real
#                 unresolved-flux floor, not empty sky. There is no zero floor
#                 here to recover, so expect this to be neutral on the floor
#                 metric and to show up (if at all) on absolute scale/transfer.
#
#   --sky-aug     targets the sky leak: m32's per-patch pedestal varies 250x
#                 across the field. m31bK50's varies 2.0x (measured: per-patch
#                 background flux p1=0.0108, median 0.0142, p99=0.0213). There
#                 is very little for the model to be invariant to.
#
# A null result here is a real result -- it bounds where those fixes apply.
# What is genuinely untested on this field is capacity and receptive field,
# which is what arms C/D/E spend the compute on.
#
# SKY-AUG RANGE IS RESCALED AND MUST BE
# -------------------------------------
# The --sky-aug-lo/-hi defaults (1e-4 .. 5.2e-2) are the m32 field's flux
# pedestals. m31bK50 is min-max normalized to a completely different scale --
# 0.052 is 3.7x its entire background flux (median 0.0142). Measured on 512 val
# patches, median z shift:
#
#   m31bK50's real per-patch background spread   0.0642 in z
#   --sky-aug-hi 1e-2  (used below)              +0.0191
#   --sky-aug-hi 5.2e-2 (m32 default)            +0.0562, ~3x the real spread
#
# so the default teaches invariance to a pedestal range this field never
# exhibits. 1e-4 .. 1e-2 spans no-op to roughly the observed spread.
set -euo pipefail

PY=${PY:-python}
DATA=${DATA:-../global/ml-decon/data/m31bK50}
OUT=${OUT:-../global/ml-decon}

# 14153 train patches. batch 128 -> 110 steps/epoch; 1000 epochs ~ 110k steps,
# matching the step count the retired 250-epoch/batch-32 runs reached.
COMMON="--data-dir $DATA --epochs 1000 --batch-size 128 --lr 2e-4 \
        --p-uncond 0.15 --identity-frac 0.15 --low-t-frac 0.25 \
        --seed 1234 --num-workers 8"

# 85-px receptive field on a 64-px patch: every pixel finally sees the whole
# patch WITHOUT downsampling, which is the flat arch's answer to the U-Net.
WIDE="--channels 192 --dilations 1,2,3,4,6,8,6,4,3,2,1"

run () { echo "=== $1"; $PY -u train_conditional_diffusion.py $COMMON "${@:2}" \
         --checkpoint-dir "$OUT/$1" 2>&1 | tee "$1.log"; }

# A. baseline reproduction -- the retired recipe, today's code.
run ckpt_m31bK50_A_group        --norm group

# B. the norm fix alone, capacity held at the baseline.
run ckpt_m31bK50_B_normless     --norm none

# C. + capacity and receptive field (0.46M -> 4.23M params, 45 -> 85 px).
run ckpt_m31bK50_C_wide         --norm none $WIDE

# D. + sky augmentation, range rescaled to THIS field (see header).
run ckpt_m31bK50_D_wide_skyaug  --norm none $WIDE \
                                --sky-aug --sky-aug-lo 1e-4 --sky-aug-hi 1e-2

# E. the architecture control. Downsampling is the thing CLAUDE.md 4.3 argues
#    against for point sources; C is the same context budget without it, so
#    C-vs-E isolates downsampling rather than confounding it with reach.
run ckpt_m31bK50_E_unet         --norm none --arch unet --base 128 \
                                --ch-mult 1,2,4 --blocks-per-level 3

cat <<'EOF'

ALL ARMS DONE. Do NOT rank these on val loss:
  - the sky-aug arm trains on a shifted input distribution;
  - flat and unet do not share a loss landscape (the script says so itself);
  - and val loss has never predicted photometry on this problem.
Score them through the solver instead, e.g. for each checkpoint:

  python admm_psf_cond.py --cond-ckpt <ckpt>/best.pt \
    --data-dir ../global/ml-decon/data/m31bK50 --psf psf_50_true.fits \
    --out-dir admm_m31bK50_<arm> --n-patches 16 --iters 200

and compare real/false counts and dmag at matched completeness.
EOF
