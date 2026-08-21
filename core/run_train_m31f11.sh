#!/usr/bin/env bash
# Train the conditional DDPM on an M31 field-11 simulation band (F475W / F814W).
#
# Hyperparameters are copied verbatim from the args stored in
# checkpoints_cond_diffusion_npy_m32/best.pt, so a run differs from the current
# m32 model in the DATASET and nothing else.
#
# One band per run: ConditionalFlatCNN is in_channels=2 (noisy ideal + the
# observed as conditioning), i.e. single-band. Train F475W and F814W separately.
#
# Why this dataset should behave better than the m32 one, and what to check:
#   * its IDEAL contains the diffuse light (0.01% exact zeros, ~5% negatives --
#     a field, not a catalogue). m3201newK was 91% zeros with no diffuse
#     component, which is what left the m32 model inferring and cancelling a
#     background that was not in its target: 13% sky leak correlated +0.93 with
#     the true per-patch sky, and a catalogue over-fragmented 2.1x against l160.
#     If that diagnosis is right, both should largely disappear here.
#   * gain 1.000 and sky ~0, verified by block-mean Theil-Sen in both bands, so
#     flux_ratio 1.0 is the correct target rather than 0.822.
#   * it is a real deconvolution task: concentration 0.093 -> 0.313 (F475W),
#     0.189 -> 0.488 (F814W).
#
# Usage on Vista, after generating the dataset there (see the header of
# config/m31f11_f475w.yaml):
#   PYTHON=python BAND=f475w \
#   DATA_ROOT=/work/11702/alexwohlberg/vista/ml-decon/data \
#   CKPT_ROOT=/scratch/11702/alexwohlberg \
#   EPOCHS=300 ./core/run_train_m31f11.sh
#
# Locally, plumbing check only:
#   BAND=f475w SANITY=1 ./core/run_train_m31f11.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PYTHON="${PYTHON:-/opt/conda/miniconda3/envs/py313/bin/python}"
BAND="${BAND:-f475w}"
DATA_ROOT="${DATA_ROOT:-$REPO/data}"
CKPT_ROOT="${CKPT_ROOT:-$REPO}"
DATA_DIR="${DATA_DIR:-$DATA_ROOT/m31f11_$BAND}"
CKPT_DIR="${CKPT_DIR:-$CKPT_ROOT/checkpoints_cond_m31f11_$BAND}"
EPOCHS="${EPOCHS:-300}"
BATCH="${BATCH:-64}"
WORKERS="${WORKERS:-16}"

case "$BAND" in
    f475w|f814w) ;;
    *) echo "BAND must be f475w or f814w (got '$BAND')" >&2; exit 2 ;;
esac

if [[ ! -f "$DATA_DIR/train_observed.npy" ]]; then
    echo "no dataset at $DATA_DIR -- generate it first, from the repo root:" >&2
    echo "  python -m dataset.gen_data config/m31f11_$BAND.yaml" >&2
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
