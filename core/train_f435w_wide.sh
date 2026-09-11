#!/usr/bin/env bash
# Train the F435W twin of ckpt_m32sim_f555w_wide -- the 4.235M-parameter,
# 85-px receptive field, norm=none, sky-augmented prior that the two-band CMD
# work used for BOTH bands.
#
# Every hyperparameter below is read off the F555W checkpoint's stored `args`,
# not reconstructed from memory:
#
#   {'epochs': 600, 'batch_size': 32, 'lr': 0.0002, 'timesteps': 1000,
#    'dilations': '1,2,3,4,6,8,6,4,3,2,1', 'channels': 192, 'arch': 'flat',
#    'norm': 'none', 'p_uncond': 0.15, 'identity_frac': 0.15,
#    'identity_t_max': 100, 'low_t_frac': 0.25, 'low_t_max': 100,
#    'no_augment': False, 'sky_aug': True, 'sky_aug_lo': 1.0,
#    'sky_aug_hi': 150.0, 'seed': 1234, 'num_workers': 8}
#
# TWO THINGS DO NOT CARRY OVER FROM THE F555W RUN.
#
# 1. THE DATASET. data/m32_sim_f435w is 64 px, and the wide prior's 85-px
#    receptive field SATURATES there -- every output pixel already sees every
#    input pixel, so the extra capacity has nothing to learn with. Step 1
#    below builds data/m32_sim_f435w_p128 at the same 128/24 geometry the
#    F555W arm used. Skipping this is the single easiest way to waste the run.
#
# 2. --sky-aug-lo / --sky-aug-hi ARE IN FLUX THROUGH THE DATASET'S OWN
#    norm.json, so they are normalization-specific and DO NOT transfer between
#    bands. F555W's [1, 150] sits in a domain with median 385.7 / beta 100.1;
#    F435W's is median 207.0 / beta 46.6. Matching the z-shift the F555W run
#    actually applied (solved numerically, not scaled by eye):
#
#        z_in    f555w c=150      f435w c needed
#        0.15      +0.1451             58.9
#        0.30      +0.0937             42.1
#        0.60      +0.0093             24.2
#
#    -> lo 0.41, hi 42 (below). CHECK THE TRAINER'S OWN [sky-aug] TABLE on
#    startup: it prints the resulting z shift, and it should look like the
#    F555W one (+0.145 at z=0.15, +0.094 at 0.30). If it does not, the
#    regenerated p128 norm differs from the 64-px one these were solved
#    against and the numbers need redoing.
#
# WHAT THIS TEST CAN AND CANNOT SETTLE. The existing evidence says an in-band
# prior is NOT what limits the CMD: the out-of-band F555W wide prior measures
# F435W BETTER than the in-band GroupNorm F435W prior at every magnitude to
# 23.5 (0.060 vs 0.097 at 21.0-21.5), and the two bands' scatter already
# tracks to 10-20%, which caps any band-matching gain at ~5-13% in colour by
# quadrature. But that comparison was CONFOUNDED -- the out-of-band arm also
# changed norm, width and patch size. THIS run is the clean test, because it
# changes ONLY the band. To read it, compare against the F555W wide prior on
# the SAME F435W data:
#
#     python cmd_two_band.py --run-dir <new> --data-prefix m32_wide ...
#     python cmd_compare_arms.py --a cmd_out_wide --b <new out-dir>
#
# and look at the per-band F435W scatter table, not the pooled colour MAD.
#
# Usage:  ./train_f435w_wide.sh data      # step 1, build the p128 dataset
#         ./train_f435w_wide.sh train     # step 2, train (needs a GPU)
#         ./train_f435w_wide.sh           # print the commands and exit
set -euo pipefail
ML=/home/alex/noir_ml/global/ml-decon
PY=${PY:-python}

DATA_CMD=("$PY" "$ML/dataset/gen_data.py" "$ML/config/m32_sim_f435w_p128.yaml")

TRAIN_CMD=("$PY" "$ML/core/train_conditional_diffusion.py"
  --data-dir       "$ML/data/m32_sim_f435w_p128"
  --checkpoint-dir "$ML/m32sim_arms/ckpt_m32sim_f435w_wide"
  --epochs 600
  --batch-size 32
  --lr 2e-4
  --timesteps 1000
  --arch flat
  --channels 192
  --dilations 1,2,3,4,6,8,6,4,3,2,1
  --norm none
  --p-uncond 0.15
  --identity-frac 0.15
  --identity-t-max 100
  --low-t-frac 0.25
  --low-t-max 100
  --sky-aug
  --sky-aug-lo 0.41
  --sky-aug-hi 42
  --seed 1234
  --num-workers 8)

case "${1:-show}" in
  data)  echo "+ ${DATA_CMD[*]}";  "${DATA_CMD[@]}" ;;
  train)
    [ -d "$ML/data/m32_sim_f435w_p128" ] || {
      echo "data/m32_sim_f435w_p128 missing -- run './train_f435w_wide.sh data' first" >&2
      exit 1; }
    echo "+ ${TRAIN_CMD[*]}"; "${TRAIN_CMD[@]}" ;;
  *)
    echo "step 1 (dataset):"; echo "  ${DATA_CMD[*]}"; echo
    echo "step 2 (train, GPU):"; printf '  %s\n' "${TRAIN_CMD[*]}"; echo
    echo "On Vista the paths under $ML become the \$WORK/\$SCRATCH equivalents"
    echo "the F555W run used:"
    echo "  --data-dir       \$WORK/ml-decon/data/m32_sim_f435w_p128"
    echo "  --checkpoint-dir \$SCRATCH/m32sim_arms/ckpt_m32sim_f435w_wide"
    echo
    echo "F555W reference: 600 epochs at ~61.5 s/epoch = ~10.5 h on one GPU."
    ;;
esac
