#!/usr/bin/env bash
# Train the conditional DDPM on an M32 v2 simulation band (F435W / F555W).
#
# Hyperparameters are copied verbatim from the args stored in
# checkpoints_cond_diffusion_npy_m32/best.pt, so a run differs from the previous
# models in the DATASET and nothing else.
#
# One band per run: ConditionalFlatCNN is in_channels=2 (noisy ideal + the
# observed as conditioning), i.e. single-band. Train F435W and F555W separately.
#
# WHY THIS DATA SHOULD TRANSFER WHERE M31f11 DID NOT.
# The M31f11 sims were UNDERSAMPLED -- 1.47 px and 1.15 px FWHM, with 7.3% and
# 23.4% of the DC power still present at the Nyquist frequency. A model trained
# there learns to invert an ALIASED operator, which is specific to that
# sampling, and it under-sharpened badly on the measured ef frame
# (concentration 0.092 against l160's 0.297, completeness 63.7%/56.5%).
# This pair is sampled like the measurement:
#     sim F435W  FWHM 3.14 px, MTF@Nyquist 0.002
#     sim F555W  FWHM 3.49 px, MTF@Nyquist 0.001
#     ef frame   FWHM 3.10 px, MTF@Nyquist 0.003
# F435W is within 1% of the ef blur, so it is the one to train first and the
# one most likely to transfer.
#
# WHAT TO CHECK AFTERWARDS, in order:
#   1. concentration on the ef frame. The bar to beat is 0.236 (the old
#      Klong-trained model) and the target is l160's ~0.30. If the undersampling
#      diagnosis is right this is where the gain shows up.
#   2. the sky leak: correlate the reconstruction's per-patch background against
#      the true per-patch sky. It was +0.93 with 13% residual on Klong, and this
#      truth is a catalogue again (94.5% exact zeros) with a 163-240 count sky
#      in the conv, so the same mismatch is present. Do NOT expect it fixed.
#   3. catalogue size against l160 (was 2.1x over) via ef_lumfunc_compare.py.
#
# KNOWN LIMITATION: the truth is a catalogue rendering and the conv carries a
# sky the truth does not, so flux_ratio 1.0 is NOT the target -- measured
# flux(obs)/flux(truth) is 3.34 (F435W) and 1.95 (F555W) on val patches. The
# diffuse light has no home in the target and the model must infer and cancel
# it, which is the mechanism behind the sky leak and the over-fragmented
# catalogue seen previously.
#
# Usage on Vista, after generating the dataset there:
#   PYTHON=python BAND=f435w \
#   DATA_ROOT=/work/11702/alexwohlberg/vista/ml-decon/data \
#   CKPT_ROOT=/scratch/11702/alexwohlberg \
#   EPOCHS=300 ./core/run_train_m32sim.sh
#
# Locally, plumbing check only:
#   BAND=f435w SANITY=1 ./core/run_train_m32sim.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PYTHON="${PYTHON:-/opt/conda/miniconda3/envs/py313/bin/python}"
BAND="${BAND:-f435w}"
DATA_ROOT="${DATA_ROOT:-$REPO/data}"
CKPT_ROOT="${CKPT_ROOT:-$REPO}"
DATA_DIR="${DATA_DIR:-$DATA_ROOT/m32_sim_$BAND}"
CKPT_DIR="${CKPT_DIR:-$CKPT_ROOT/checkpoints_cond_m32sim_$BAND}"
EPOCHS="${EPOCHS:-300}"
BATCH="${BATCH:-64}"
WORKERS="${WORKERS:-16}"

case "$BAND" in
    f435w|f555w) ;;
    *) echo "BAND must be f435w or f555w (got '$BAND')" >&2; exit 2 ;;
esac

if [[ ! -f "$DATA_DIR/train_observed.npy" ]]; then
    echo "no dataset at $DATA_DIR -- generate it first, from the repo root:" >&2
    echo "  python -m dataset.gen_data config/m32_sim_$BAND.yaml" >&2
    exit 1
fi

echo "[run] band=$BAND  data=$DATA_DIR  ckpt=$CKPT_DIR  epochs=$EPOCHS"

if [[ "${SANITY:-0}" == "1" ]]; then
    # Memorize one batch. Checks plumbing, not learning.
    exec "$PYTHON" -u "$HERE/train_conditional_diffusion.py" \
        --data-dir "$DATA_DIR" \
        --checkpoint-dir "$CKPT_DIR" \
        --overfit-one-batch --sanity-steps 3000
fi

exec "$PYTHON" -u "$HERE/train_conditional_diffusion.py" \
    --data-dir "$DATA_DIR" \
    --checkpoint-dir "$CKPT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --lr 1e-4 \
    --timesteps 1000 \
    --channels 64 \
    --p-uncond 0.15 \
    --identity-frac 0.15 \
    --identity-t-max 100 \
    --low-t-frac 0.25 \
    --low-t-max 100 \
    --seed 1234 \
    --num-workers "$WORKERS"
