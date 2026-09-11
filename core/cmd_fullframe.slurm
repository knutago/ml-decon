#!/usr/bin/env bash
#SBATCH -J cmd_fullframe
#SBATCH -o cmd_fullframe.%j.out
#SBATCH -e cmd_fullframe.%j.err
#SBATCH -p gh                 # Vista GH200 queue -- check `sinfo` for yours
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A CHANGEME           # <-- your allocation; `sacctmgr show assoc user=$USER`
#
# Full-frame two-band reconstruction for the colour-magnitude diagram.
# Modelled on admm_tune.slurm, which is the template that already works here.
#
# Submit:   sbatch cmd_fullframe.slurm
# Interact: idev -p gh -N 1 -n 1 -t 2:00:00   then ./run_cmd_fullframe.sh solve
#
# RESUMABLE. Chunks that already have a val_recon.npy are skipped, so if this
# hits the walltime just resubmit the identical script and it continues. Only
# the chunk in flight is lost.
#
# WHAT IT BUYS: the CMD currently uses 6.2% of the frame (a 512^2 cut of a
# 2048^2 image). Full frame is ~16x the stars -- ~7,400 at sigma<0.05 against
# the present 461 -- which is what turns the per-magnitude scatter comparison
# from a 6-bin eyeball into a statistical claim.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# --- adjust these to Vista's layout (same three as admm_tune.slurm) --------
module load cuda || true
export PY=${PY:-$WORK/conda/envs/py313/bin/python}
export ML_DECON=${ML_DECON:-$WORK/ml-decon}
# --------------------------------------------------------------------------

export ITERS=${ITERS:-450}
export CHUNK=${CHUNK:-32}     # a GH200 has the memory for wider chunks than CPU
export OUT=${OUT:-cmd_bands_full}

echo "host=$(hostname)  iters=$ITERS  chunk=$CHUNK  out=$OUT"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# The datasets are small and CPU-bound to build; do it here so the job is
# self-contained, and it is a no-op if they already exist.
[ -d "$ML_DECON/data/m32_full_f555w" ] || ./run_cmd_fullframe.sh data

./run_cmd_fullframe.sh solve

echo
echo "Then, back on any machine:"
echo "  python cmd_forced.py --run-dir $OUT --data-prefix m32_full --min-dchi2 470 --out-dir cmd_out_full_snr470"
echo "  python cmd_quality_figure.py --run-dir $OUT --data-prefix m32_full"
