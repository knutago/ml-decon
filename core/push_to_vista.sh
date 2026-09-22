#!/usr/bin/env bash
# Push exactly what Vista needs for the CMD runs -- nothing more.
#
# The manifest is the transitive closure of what the solver actually imports,
# checked rather than guessed:
#   admm_diffusion_deconvolve.py
#     -> model_denoise.py        (load_checkpoint, tweedie_chain)
#          -> train_conditional_diffusion.py  (model_from_checkpoint)
#     -> cond_sample_npy.py      (find_peaks, concentration)
#     -> solver_metrics.py
#     -> red_pnp_deconvolve.py   (TorchNorm, load_kernel, make_otf)
#   make_band_patches.py -> red_pnp_deconvolve.py
#
# NOT pushed, deliberately:
#   * checkpoints -- already on Vista, that is where they were trained.
#     ONE EXCEPTION, below: the joint two-band prior was trained to $SCRATCH,
#     and SCRATCH IS PURGED. Pushed back when a local copy exists.
#   * the m32_sim_* datasets -- likewise
#   * cmd_two_band.py / cmd_forced.py / cmd_compare_arms.py and the figure
#     scripts -- analysis is seconds on a laptop; pull the reconstructions
#     back and run it here rather than burning GPU allocation on matplotlib
#
# All the pushed scripts take their paths from $PY / $ML_DECON / $MYCODE, so
# nothing needs editing on the far side.
#
# Usage:
#   ./push_to_vista.sh                       # dry run, shows what would move
#   REMOTE=you@vista ./push_to_vista.sh go   # actually push
#
# You run the ssh/rsync yourself -- I do not hold TACC credentials or MFA.
set -euo pipefail
cd "$(dirname "$0")"

REMOTE=${REMOTE:-CHANGEME@vista.tacc.utexas.edu}
RWORK=${RWORK:-\$WORK}            # expanded on the REMOTE side
ML=${ML_DECON:-/home/alex/noir_ml/global/ml-decon}

CODE=(admm_diffusion_deconvolve.py red_pnp_deconvolve.py model_denoise.py
      cond_sample_npy.py solver_metrics.py train_conditional_diffusion.py
      make_band_patches.py
      run_cmd_confirmed.sh run_cmd_fullframe.sh cmd_fullframe.slurm
      psf_m32_v2_centred.fits psf_m32v_v2_centred.fits)

# Resolved at run time: this repo has reorganised the m32 frames more than
# once (flat in data/, then data/m32/, now per-reference m32_l160 / m32_l640),
# so search rather than hardcode -- the same reason make_band_patches._find
# exists. Paths are emitted relative to $ML so rsync --relative recreates the
# layout on the far side.
DATA=()
for n in m32b_phase_v2ef.fits m32b_phase_v2l160.fits \
         m32v_phase_v2ef.fits m32v_phase_v2l640.fits; do
  found=""
  for sub in data/m32_l160 data/m32_l640 data data/m32; do
    [ -f "$ML/$sub/$n" ] && { found="$sub/$n"; break; }
  done
  [ -n "$found" ] || { echo "MISSING anywhere: $n" >&2; exit 1; }
  DATA+=("$found")
done

missing=0
for f in "${CODE[@]}"; do [ -e "$f" ] || { echo "MISSING: $f" >&2; missing=1; }; done
for f in "${DATA[@]}"; do [ -e "$ML/$f" ] || { echo "MISSING: $ML/$f" >&2; missing=1; }; done
[ "$missing" -eq 0 ] || exit 1

sz=$( { du -ch "${CODE[@]}" | tail -1; (cd "$ML" && du -ch "${DATA[@]}" | tail -1); } \
      | awk '{print $1}' | paste -sd+ )
CKPT=${CKPT:-/home/alex/noir_ml/results/twoband/best.pt}
CKPT_DEST=m32sim_arms/ckpt_m32sim_twoband_common

echo "code : ${#CODE[@]} files"
echo "data : ${#DATA[@]} FITS"
echo "total: $sz"
if [ -f "$CKPT" ]; then
  echo "ckpt : $CKPT ($(du -h "$CKPT" | cut -f1)) -> ml-decon/$CKPT_DEST/best.pt"
  echo "       (trained to \$SCRATCH, which is purged -- check the far side first:"
  echo "        ssh $REMOTE 'ls -l \$WORK/ml-decon/$CKPT_DEST/best.pt')"
else
  echo "ckpt : $CKPT not found locally -- the joint arm needs it on Vista already"
fi
echo
echo "destination: $REMOTE:$RWORK/{mycode,ml-decon/data,ml-decon/$CKPT_DEST}"

if [ "${1:-}" != "go" ]; then
  echo
  echo "DRY RUN. Re-run with:  REMOTE=you@vista.tacc.utexas.edu ./push_to_vista.sh go"
  exit 0
fi

# shellcheck disable=SC2029  # $RWORK is meant to expand remotely
ssh "$REMOTE" "mkdir -p $RWORK/mycode $RWORK/ml-decon/data/m32_l640 $RWORK/ml-decon/$CKPT_DEST"
rsync -avP "${CODE[@]}"                 "$REMOTE:$RWORK/mycode/"
rsync -avP --relative                   \
      "${DATA[@]/#/$ML/./}"             "$REMOTE:$RWORK/ml-decon/"
[ -f "$CKPT" ] && rsync -avP "$CKPT"    "$REMOTE:$RWORK/ml-decon/$CKPT_DEST/best.pt"

cat <<EOF

Pushed. On Vista:

  cd \$WORK/mycode                  # FLAT -- there is no mycode/core on Vista.
                                    # core/ is ml-decon's layout; this rsync
                                    # puts every pushed script at the top level.
  export ML_DECON=\$WORK/ml-decon
  export MYCODE=\$WORK/mycode
  export PY=\$WORK/ml-decon/.venv/bin/python       # uv venv, cpython 3.12 aarch64
                                                  # (NOT a conda py313 -- that
                                                  # path is this workstation's)

  # 0. confirm it will actually use the GPU (else you wait hours for nothing)
  \$PY -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

  # 1. the four arms  (~40 min total on a GH200)
  for pr in shared matched; do for pf in ef200x4 c1000x1; do
      PRIOR=\$pr PROFILE=\$pf ./run_cmd_confirmed.sh
  done; done

  # 2. the full frame -- the one that changes the science  (~1.5-3 h)
  sbatch cmd_fullframe.slurm        # after filling in -A <allocation>

Pull back only the reconstructions (small) and do the analysis locally:

  rsync -avP $REMOTE:$RWORK/mycode/'cmd_bands_*' .
  python cmd_forced.py --run-dir cmd_bands_ef200x4_matched --data-prefix m32_bm --min-dchi2 470 --out-dir cmd_out_matched
  python cmd_compare_arms.py --a cmd_out_ef200x4 --b cmd_out_matched
EOF
