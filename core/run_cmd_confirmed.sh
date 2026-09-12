#!/usr/bin/env bash
# Two-band shared-wide-prior run using ADMM settings COPIED VERBATIM from runs
# that were measured to work, rather than assembled from remembered rules.
#
# WHY THIS SCRIPT EXISTS -- the earlier runs had annealing switched OFF.
# run_cmd_wide_shared.sh and run_cmd_wide_v2.sh used `--rho-scale 1.0`. The
# timestep is chosen by matching sigma(t) to the ADMM denoising level
# sqrt(lam/rho); with rho constant that level never moves, so t sat pinned at
# t_max for every iteration -- the solver log said so plainly:
#
#     [t] trajectory k=0:t=55 ... k=449:t=55 | clamped at t_max for 450 iters
#
# EVERY config on disk that was actually validated anneals rho, and therefore t:
#
#   admm_fft_best         400 x 4  rho 0.1  scale 1.1    eta 1.0  t 75->1  avg 2
#   admm_m31bK50_C_tuned 1000 x 1  rho 0.05 scale 1.01   eta 1.0  t 80->5  avg 2
#   ef_W_alex_tmin1      1000 x 1  rho 0.1  scale 1.006  eta 1.0  t 80->1  avg 0
#   ef_out/v2_f555w       200 x 4  rho 0.1  scale 1.05   eta 0.0  t 75->1  avg 0
#
# The rho ramp is what drives the anneal:
#   200 x 4:  rho 0.1  -> 1729,  sigma 0.3162 -> 0.0024
#   1000 x 1: rho 0.05 -> 1048,  sigma 0.4472 -> 0.0031
# so t walks from t_max down to t_min instead of standing still. CLAUDE.md is
# explicit that annealing is not a refinement but the whole mechanism -- a
# flat-rho arm degrades every metric at once.
#
# The earlier "flat rho is endorsed" reading was a misapplication: that result
# was about ADMM's rho CONTINUATION being a crutch for running too few
# iterations, not about disabling the timestep schedule.
#
# PROFILES -- pick with PROFILE=, both named by the user as known-good:
#
#   ef200x4  (default)  200 iters x 4 denoise steps = 800 model calls.
#                       From ef_out/v2_f555w, i.e. validated on THIS data
#                       family (the m32 ef frames), and eta 0 keeps
#                       photometric scatter down, which is what a CMD is read
#                       on. Cheapest of the two.
#
#   c1000x1             1000 iters x 1 denoise step = 1000 model calls.
#                       From admm_m31bK50_C_tuned. Measured to match the old
#                       best at 5/8 the cost with better purity. eta 1.0 buys
#                       a ~5 pp completeness ceiling at ~34% more scatter.
#
# UNCHANGED AND DELIBERATE: the prior is ckpt_m32sim_f555w_wide SHARED across
# both bands (that is what made the colour survive -- differential slope
# 0.156 -> 0.027), the 128-px datasets, the band-specific PSFs, and the
# self-consistent RIVAL-FREE skies (F435W 256.87, F555W 231.12).
#
# COST. ~30 s per model call here (16 patches of 128^2, 4.2M-param net, 4 CPU
# threads), so ef200x4 is ~6.7 h/band and c1000x1 ~8.3 h/band. Both bands run
# concurrently. This is an overnight job on CPU and well under an hour on a
# GH200 -- see cmd_fullframe.slurm for the Vista pattern.
#
# Usage:  ./run_cmd_confirmed.sh                  # ef200x4, 4 threads/band
#         PROFILE=c1000x1 ./run_cmd_confirmed.sh
#         THREADS=2 ./run_cmd_confirmed.sh        # yield cores to a meeting
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-/opt/conda/miniconda3/envs/py313/bin/python}
ML=${ML_DECON:-/home/alex/noir_ml/global/ml-decon}
PROFILE=${PROFILE:-ef200x4}
# PRIOR=shared  : ckpt_m32sim_f555w_wide on BOTH bands. Makes prior-induced
#                 scale error common-mode so it cancels in the colour --
#                 measured to take the differential slope 0.156 -> 0.027.
# PRIOR=matched : each band gets its own in-band wide prior. The clean test of
#                 band-matching, but it gives up the common-mode cancellation,
#                 so watch the differential slope.
PRIOR=${PRIOR:-shared}
THREADS=${THREADS:-4}

case "$PROFILE" in
  ef200x4)  SET=(--iters 200  --denoise-steps 4 --rho 0.1  --rho-scale 1.05
                 --eta 0.0 --t-max 75 --t-min 1 --avg-last 0 --renoise) ;;
  c1000x1)  SET=(--iters 1000 --denoise-steps 1 --rho 0.05 --rho-scale 1.01
                 --eta 1.0 --t-max 80 --t-min 5 --avg-last 2 --renoise) ;;
  *) echo "unknown PROFILE=$PROFILE (ef200x4 | c1000x1)" >&2; exit 1 ;;
esac
OUT=${OUT:-cmd_bands_${PROFILE}_${PRIOR}}

case "$PRIOR" in
  shared)  CK_F435=$ML/m32sim_arms/ckpt_m32sim_f555w_wide/best.pt
           CK_F555=$ML/m32sim_arms/ckpt_m32sim_f555w_wide/best.pt
           DATA=m32_wide;  SKY_F435=256.87; SKY_F555=231.12 ;;
  matched) CK_F435=$ML/m32sim_arms/ckpt_m32sim_f435w_wide/best.pt
           CK_F555=$ML/m32sim_arms/ckpt_m32sim_f555w_wide/best.pt
           DATA_F435=m32_bm_f435w; DATA_F555=m32_wide_f555w
           SKY_F435=147.27; SKY_F555=231.12 ;;
  *) echo "unknown PRIOR=$PRIOR (shared | matched)" >&2; exit 1 ;;
esac
[ "$PRIOR" = shared ] && { DATA_F435=${DATA}_f435w; DATA_F555=${DATA}_f555w; }

COMMON=(--prior cond --split val --indices 0:16 --split-norms
        --gain 1.0 --sparse-tau 0 --lam 0.01 --n-show 0 --log-every 10)

mkdir -p "$OUT"
echo "=== profile $PROFILE  prior $PRIOR  threads/band $THREADS  -> $OUT/"
echo "    ${SET[*]}"
pids=()
for arm in "f435w:psf_m32_v2_centred.fits:$SKY_F435:$CK_F435:$DATA_F435" \
           "f555w:psf_m32v_v2_centred.fits:$SKY_F555:$CK_F555:$DATA_F555"; do
  IFS=: read -r band psf sky ck data <<<"$arm"
  echo "--- launching $band  psf=$psf  sky=$sky"
  OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
  OPENBLAS_NUM_THREADS=$THREADS \
  $PY -u admm_diffusion_deconvolve.py "${COMMON[@]}" "${SET[@]}" \
      --ckpt "$ck" --data-dir "$ML/data/$data" \
      --psf-file "$psf" --sky "$sky" --out-dir "$OUT/$band" \
      > "$OUT/$band.log" 2>&1 &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] || { echo "A BAND FAILED -- check $OUT/*.log" >&2; exit 1; }

echo
echo "=== done -> $OUT/{f435w,f555w}/val_recon.npy"
echo "  python cmd_forced.py --run-dir $OUT --data-prefix m32_wide --min-dchi2 470 --out-dir cmd_out_${PROFILE}_snr470"
echo "  python cmd_compare_arms.py --a cmd_out_wide --b <new cross-match dir>"
