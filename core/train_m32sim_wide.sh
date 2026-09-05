#!/bin/bash
# Wide conditional prior on the M32 F555W simulation.
#
# READ THIS BEFORE SPENDING THE GPU HOURS -- one of the two things this script
# was asked to target turns out not to be a problem on THIS dataset.
#
# THE SKY LEAK IS NOT AN M32-SIM PROBLEM. MEASURED.
# --------------------------------------------------
# The r=+0.93 / "13% of the sky leaks into the output" result was measured on
# the REAL Klong field, whose per-patch sky pedestal varies 250x. The
# SIMULATION does not:
#
#   per-patch background flux, m32_sim_f555w   284 .. 380   = 1.34x
#   per-patch background flux, m31bK50          0.012 .. 0.022 = 1.83x
#
# and on the existing m32_sim_f555w reconstructions the leak is already gone:
# median residual background 0.72 counts against a true sky of 283-342 (99.8%
# removed), correlation with the true per-patch sky +0.148, not +0.93.
#
# So --sky-aug below is NOT for this dataset. It is insurance for TRANSFER to
# the real ef/Klong frames, where the leak is real. Keep it, but do not expect
# it to move any number measured on the sim, and do not read a null result as
# a failed experiment.
#
# FRAGMENTATION HAS NO PROVEN TRAINING-SIDE LEVER
# -----------------------------------------------
# The over-fragmentation fix that IS measured is a SOLVE-time significance cut
# -- `--min-dchi2 4` against the OBSERVED frame's noise (purity 28% -> 71%, LF
# total ratio 3.11 -> 1.10). Nothing in this script reproduces that, and no
# training setting here is known to. Separately, deblending itself has no
# headroom: 93% of blended pairs sit at dchi2 < 1, i.e. already at the
# confusion ceiling. Widening the model cannot beat that bound.
#
# WHAT THIS SCRIPT ACTUALLY BETS ON, then, is capacity + receptive field +
# 6.6x the training pixels, which is the axis that won on m31bK50. That win is
# itself only partly attributed -- see the note at the bottom.
set -euo pipefail

PY=${PY:-python}
DATA=${DATA:-../global/ml-decon/data/m32_sim_f555w_p128}
OUT=${OUT:-../global/ml-decon}

# Build the dataset first:
#   cd ../global/ml-decon && uv run python -m dataset.gen_data \
#       config/m32_sim_f555w_p128.yaml
# 6561 patches of 128x128. Batch 32 not 128: a 128-px patch is 4x the
# activations of a 64-px one, so this holds roughly the old memory footprint.
# ~185 steps/epoch; 600 epochs ~ 111k steps, matching the m31bK50 arms.
COMMON="--data-dir $DATA --epochs 600 --batch-size 32 --lr 2e-4 \
        --p-uncond 0.15 --identity-frac 0.15 --low-t-frac 0.25 \
        --seed 1234 --num-workers 8"

# 4.235M params, 85-px receptive field -- a real neighbourhood on a 128-px
# patch, where on 64 px it saturated. --norm none is REQUIRED here, not
# optional: GroupNorm annihilates absolute scale at block 0 and makes the
# network input-size dependent, which would forfeit the whole point of
# training at 128 and block full-frame inference.
WIDE="--channels 192 --dilations 1,2,3,4,6,8,6,4,3,2,1 --norm none"

# Sky-aug range in FLUX (counts), calibrated against this frame's own sky.
# Measured median z shift for a pedestal added on top of the ~325-count base:
#     +10 -> +0.009    +50 -> +0.049    +100 -> +0.101    +300 -> +0.229
# The field's own real spread is 0.088 in z. 1..150 therefore spans no-op to
# ~1.7x the real spread -- enough to teach pedestal invariance for transfer
# without training on a sky this frame never shows. (The DEFAULT range,
# 1e-4..5.2e-2, is in m32_klong's normalized units and is a no-op at this
# frame's scale of hundreds of counts -- passing it here would silently
# disable the augmentation.)
SKYAUG="--sky-aug --sky-aug-lo 1.0 --sky-aug-hi 150.0"

run () { echo "=== $1"; $PY -u train_conditional_diffusion.py $COMMON "${@:2}" \
         --checkpoint-dir "$OUT/$1" 2>&1 | tee "$1.log"; }

# W. the bet: capacity + RF + 6.6x pixels, with transfer insurance.
W () { run ckpt_m32sim_f555w_wide      $WIDE $SKYAUG; }

# N. the control that makes W interpretable: same width, same data, NO
#    sky-aug. Without it, a W-vs-checkpoints_m32_normless comparison changes
#    width, patch size, dataset size AND augmentation at once.
N () { run ckpt_m32sim_f555w_wide_noaug $WIDE; }

# S. the small control: baseline width on the SAME 128-px data, so "did width
#    help" is answerable without re-running the old 64-px checkpoint.
S () { run ckpt_m32sim_f555w_small     --channels 64 --norm none; }

[ $# -gt 0 ] || set -- W N S
for arm in "$@"; do
    case "$arm" in
        W|N|S) "$arm" ;;
        *) echo "unknown arm '$arm' (want W N S)" >&2; exit 2 ;;
    esac
done

cat <<'EOF'

SCORING. Do not rank on val loss. Score through the solver, and note that
admm_psf_cond.py / admm_diffusion_deconvolve.py default --sparse-tau to 2.0,
which is tuned for exactly this kind of ~95%-zero truth -- so unlike m31bK50
it is defensible here, but it MUST be stated and held constant across arms.

  python admm_diffusion_deconvolve.py --prior cond --ckpt <ckpt>/best.pt \
    --data-dir ../global/ml-decon/data/m32_sim_f555w_p128 --split val \
    --indices 0:16 --psf-file data/m32_sim/psf_m32sim_f555w.fits \
    --iters 400 --denoise-steps 4 --rho 0.1 --rho-scale 1.1 --lam 0.01 \
    --eta 1.0 --t-max 75 --t-min 1 --renoise --avg-last 2 --sparse-tau 0 \
    --out-dir admm_m32sim_<arm>

Report real/false counts, not completeness; and the LF total ratio against the
truth, which is the metric over-fragmentation actually shows up in.
EOF
