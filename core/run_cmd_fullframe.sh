#!/usr/bin/env bash
# FULL-FRAME two-band reconstruction -- the 16x more data the 512^2 cut leaves
# on the table.
#
# WHY THIS IS THE BIGGEST AVAILABLE WIN. Every CMD so far comes from
# ef[768:1280, 768:1280]: 262,144 px of a 4,194,304 px frame, i.e. 6.2% of the
# data. Scaling the measured yields by 16x:
#
#     sigma<0.10    803 stars  ->  ~12,800
#     sigma<0.05    461        ->   ~7,400
#     sigma<0.03    244        ->   ~3,900
#
# That is a qualitative change, not an incremental one. The red clump becomes
# an unmistakable feature rather than a knot of ~150 stars, and every scatter
# number gets ~4x tighter error bars -- which is what would turn "we beat the
# reference brightward of 21.5" from a 6-bin eyeball into a statistical claim.
#
# WHY IT NEEDS A GPU. 16x the pixels is ~40 h on 8 CPU cores. On a GH200 it is
# well under an hour. See cmd_fullframe.slurm.
#
# TWO THINGS THIS HANDLES THAT THE 512^2 RUN DID NOT:
#
# 1. MEMORY. 256 patches of 128^2 through a 4.2M-parameter net is several GB of
#    activations in one batch. The solve is therefore CHUNKED by patch index
#    and the pieces concatenated at the end. Chunks are resumable: a chunk with
#    a val_recon.npy is skipped, so a walltime kill costs only the chunk in
#    flight.
#
# 2. SKY VARIATION. The 512^2 cut sits in a patch of sky flat enough for one
#    constant. Across the full frame the sigma-clipped block medians run
#    1.21x (F435W) and 1.27x (F555W) end to end -- modest and smooth, but a
#    single constant would be off by up to ~12% at the corners. Patches are
#    emitted row-major, so an index chunk of 16 is a horizontal strip, and
#    each strip gets its own sky via the first-order correction
#
#        sky_strip = sky_ref + a * (median_strip - median_ref)
#
#    in the flux domain (`a` = that band's scale_a). Residual within-strip
#    error is a few percent, which the sky sweep showed costs ~nothing:
#    247.5 vs the 250.0 optimum differed by 0.0002 in residual.
#
# THE SKY REFERENCE VALUES are the self-consistent, RIVAL-FREE estimates
# measured on the 512^2 v2 reconstruction -- clipped_median(observed - PSF*x)
# using our own x. Nothing here depends on the l160/l640 deconvolution.
#
# Usage:
#   ./run_cmd_fullframe.sh data     # build the two 256-patch datasets
#   ./run_cmd_fullframe.sh solve    # chunked solve + merge (GPU strongly advised)
#   ./run_cmd_fullframe.sh          # show the plan and exit
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-/opt/conda/miniconda3/envs/py313/bin/python}
ML=${ML_DECON:-/home/alex/noir_ml/global/ml-decon}
# PRIOR=shared  : ckpt_m32sim_f555w_wide on BOTH bands (what the first
#                 full-frame run used). Prior-induced scale error is then
#                 common-mode and cancels in the colour.
# PRIOR=matched : each band gets its own in-band wide prior. Only worth running
#                 now that ckpt_m32sim_f435w_wide is trained to epoch 102/600 --
#                 the earlier epoch-46 version failed its own sim check
#                 (flux 1.75x, ZERO zero-pixels against a 94.5%-zero truth).
# PRIOR=joint   : ONE prior trained jointly over BOTH bands on the shared asinh
#                 map. Keeps the shared arm's one-network property -- which is
#                 what makes prior bias common-mode and cancel in the colour --
#                 WITHOUT its off-band collapse. Measured on 8 sim val patches
#                 at t=188, detections matched to the truth's own within 1 px:
#
#                   band   arm                    compl  purity
#                   F435W  joint                   0.55    0.81
#                   F435W  f435w_wide (in band)    0.55    0.80
#                   F435W  f555w_wide (off band)   0.35    0.91
#                   F555W  joint                   0.61    0.90
#                   F555W  f555w_wide (in band)    0.63    0.87
#                   F555W  f435w_wide (off band)   0.46    0.75
#
#                 and on per-source B-V over the 227 sources every arm found,
#                 joint MAD 0.100 vs matched 0.112 vs shared 0.318. Joint wins
#                 the NON-calibratable axis (scatter) and loses the calibratable
#                 one (clump -0.087 vs matched +0.029) -- the right side of that
#                 trade, but the margin over matched is modest, so run both.
PRIOR=${PRIOR:-shared}
CK_JOINT=${CK_JOINT:-$ML/m32sim_arms/ckpt_m32sim_twoband_common/best.pt}
case "$PRIOR" in
  shared)  CK_F435=$ML/m32sim_arms/ckpt_m32sim_f555w_wide/best.pt
           DATA_F435=m32_full_f435w
           CK_F555=$ML/m32sim_arms/ckpt_m32sim_f555w_wide/best.pt
           DATA_F555=m32_full_f555w ;;
  matched) CK_F435=$ML/m32sim_arms/ckpt_m32sim_f435w_wide/best.pt
           DATA_F435=m32_fullbm_f435w
           CK_F555=$ML/m32sim_arms/ckpt_m32sim_f555w_wide/best.pt
           DATA_F555=m32_full_f555w ;;
  # Separate data dirs: the joint checkpoint's two observed norms differ in
  # median (207.17 B / 385.66 V), and the solver picks the band's entry BY that
  # median. Reusing m32_full_* -- both of which carry 385.66 -- would silently
  # hand F435W the F555W conditioning norm.
  joint)   CK_F435=$CK_JOINT; DATA_F435=m32_fullj_f435w
           CK_F555=$CK_JOINT; DATA_F555=m32_fullj_f555w ;;
  *) echo "unknown PRIOR=$PRIOR (shared | matched | joint)" >&2; exit 1 ;;
esac
OUT=${OUT:-cmd_bands_full_$PRIOR}
ITERS=${ITERS:-200}   # 200x4 = the confirmed ef200x4 schedule
CHUNK=${CHUNK:-16}          # patches per chunk = one 128-px row strip
REGION=${REGION:-0:2048,0:2048}
# T_MAX is the depth dial (noir-ml-admm-pareto-frontier): pick the LOWEST that
# reaches your target. Exposed because the 2026-09-15 sim comparison showed the
# two priors BRACKET the truth's concentration rather than one being right --
# f435w 0.524, f555w 0.995, truth 0.804 -- so the setting that lands on 0.804
# is an empirical question, not something either checkpoint gives for free.
# The joint prior defaults to 188, not 75, and that is a measured number rather
# than a guess: sigma_z = sqrt(lam/rho) = 0.3162 lands naturally at t=188, so
# above it the clamp stops binding at all. Sweeping it on the joint checkpoint
# (8 sim val patches, real/false matched) the B/V summed-flux ratio -- i.e. the
# spurious colour -- runs 0.64 mag at t_max 75, 0.52 at 100, 0.25 at 150, 0.16
# at 188. TMAX=75 is the WORST row, and every two-band CMD so far was made at
# it. Left at 75 for shared/matched so their existing runs stay reproducible.
if [ "$PRIOR" = joint ]; then TMAX=${TMAX:-188}; else TMAX=${TMAX:-75}; fi
SPARSE=${SPARSE:-0}
SEED=${SEED:-0}

# band : psf : sky measured on the 512^2 cut : that cut's clipped block median
# band : psf : sky at the reference strip : that strip's clipped median
# The F435W sky differs between priors because each normalizes the frame into
# its OWN observed domain -- 256.87 in the f555w domain is 147.27 in the f435w
# one, the SAME physical level (102.3 raw counts), not a different choice.
#
# `meta` for the joint arm: its observed domain is a THIRD one (the common map,
# beta 68.29), so neither hardcoded constant applies and inventing a number here
# is how a 12% sky error gets in silently. make_band_patches.py already computes
# the right value and writes it to meta.json as normalization.sky_sparse -- read
# it. The reference median is then the dataset's own observed_clipped_median by
# construction, so the per-strip offset below reduces to (med_flux - c).
# shared/matched keep their literals so their existing runs stay bit-identical.
case "$PRIOR" in
  shared)  SKY435=256.87; MED435=180.5; SKY555=231.12; MED555=354.5 ;;
  matched) SKY435=147.27; MED435=180.5; SKY555=231.12; MED555=354.5 ;;
  joint)   SKY435=meta;   MED435=meta;  SKY555=meta;   MED555=meta  ;;
esac
ARMS=("f435w:psf_m32_v2_centred.fits:$SKY435:$MED435:$CK_F435:$DATA_F435"
      "f555w:psf_m32v_v2_centred.fits:$SKY555:$MED555:$CK_F555:$DATA_F555")

build_data () {
  for arm in "${ARMS[@]}"; do
    IFS=: read -r band _ _ _ ck data <<<"$arm"
    echo "--- dataset $band over $REGION"
    tr=m32_sim_f555w_p128
    [ "$PRIOR" = matched ] && [ "$band" = f435w ] && tr=m32_sim_f435w_p128
    # the joint prior was trained on the *_common pair, and --train-ref is what
    # the in-distribution check compares against, so it has to name the right one
    [ "$PRIOR" = joint ] && tr=m32_sim_${band}_p128_common
    [ -f "$ML/data/$data/val_observed.npy" ] && { echo "    exists, skipping"; continue; }
    ML_DECON="$ML" $PY make_band_patches.py --band "$band" --patch 128 --stride 128 \
        --region "$REGION" --ckpt "$ck" --train-ref "$tr" \
        --out "$ML/data/$data"
  done
}

solve () {
  for arm in "${ARMS[@]}"; do
    IFS=: read -r band psf sky0 med0 ck data <<<"$arm"
    D=$ML/data/$data
    [ -f "$D/val_observed.npy" ] || { echo "missing $D -- run 'data' first" >&2; exit 1; }
    N=$($PY -c "import numpy as np;print(len(np.load('$D/val_observed.npy',mmap_mode='r')))")
    echo "=== $band: $N patches, chunks of $CHUNK, $ITERS iters"
    for (( s=0; s<N; s+=CHUNK )); do
      e=$(( s+CHUNK > N ? N : s+CHUNK ))
      CD=$OUT/$band/chunk_$(printf '%04d' "$s")
      [ -f "$CD/val_recon.npy" ] && { echo "  [$s:$e] done, skipping"; continue; }
      # this strip's own sky, first-order from its background level
      SKY=$($PY - "$D" "$s" "$e" "$sky0" "$med0" <<'EOF'
import sys, json, numpy as np, torch
sys.path.insert(0, __import__("os").environ.get("MYCODE", "."))
from red_pnp_deconvolve import TorchNorm
from astropy.stats import sigma_clipped_stats
d,s,e,sky0,med0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
n = json.load(open(d+"/norm.json")); m = json.load(open(d+"/meta.json"))
a = m["normalization"]["scale_a"]; mo = m["normalization"]["observed_clipped_median"]
z = np.load(d+"/val_observed.npy", mmap_mode="r")[s:e]
flux = TorchNorm(n["observed"]).inverse(torch.from_numpy(np.asarray(z)).double()).numpy()
_, med_flux, _ = sigma_clipped_stats(flux, sigma=3.0, maxiters=5)
if sky0 == "meta":
    # the dataset's own sky, in its own flux domain; its reference level is the
    # whole-region clipped median, which maps to exactly n.observed.median
    sky0 = m["normalization"]["sky_sparse"]
    med0_flux = n["observed"]["median"]
else:
    sky0 = float(sky0)
    # med0 is in raw counts; convert to this dataset's flux domain to compare
    med0_flux = a*(float(med0)-mo) + n["observed"]["median"]
print(f"{sky0 + (med_flux - med0_flux):.3f}")
EOF
)
      echo "  [$s:$e] sky=$SKY -> $CD"
      $PY -u admm_diffusion_deconvolve.py --prior cond --split val \
          --indices "$s:$e" --split-norms --gain 1.0 --sparse-tau "$SPARSE" \
          --rho 0.1 --rho-scale 1.05 --eta 0.0 --avg-last 0 --seed "$SEED" \
          --denoise-steps 4 --t-max "$TMAX" --t-min 1 --lam 0.01 --renoise \
          --iters "$ITERS" --n-show 0 --no-fits --log-every 100 \
          --ckpt "$ck" --data-dir "$D" --psf-file "$psf" --sky "$SKY" \
          --out-dir "$CD" > "$CD.log" 2>&1 || {
            echo "  CHUNK $s FAILED -- see $CD.log" >&2; exit 1; }
    done
    $PY - "$OUT/$band" <<'EOF'
import sys, numpy as np
from pathlib import Path
d = Path(sys.argv[1])
parts = sorted(d.glob("chunk_*/val_recon.npy"),
               key=lambda p: int(p.parent.name.split("_")[1]))
rec = np.concatenate([np.load(p) for p in parts], 0)
np.save(d/"val_recon.npy", rec)
print(f"  merged {len(parts)} chunks -> {d}/val_recon.npy  {rec.shape}")
EOF
  done
}

case "${1:-show}" in
  data)  build_data ;;
  solve) solve ;;
  *)
    echo "Full-frame plan:"
    echo "  region      $REGION  (16x the area of the 512^2 cut)"
    echo "  patches     256 per band at 128/128"
    echo "  chunks      $CHUNK patches each = one row strip, own sky, resumable"
    echo "  iters       $ITERS"
    echo "  prior       PRIOR=$PRIOR"
    echo "     F435W    $(basename $(dirname $CK_F435))   data $DATA_F435"
    echo "     F555W    $(basename $(dirname $CK_F555))   data $DATA_F555"
    echo "  t-max       $TMAX   sparse-tau $SPARSE   seed $SEED"
    echo "  sky         self-consistent, rival-free (F435W $SKY435, F555W $SKY555"
    echo "              at the reference strip; offset per strip)"
    [ -f "$CK_F435" ] || echo "  !! CK_F435 not found: $CK_F435"
    [ -f "$CK_F555" ] || echo "  !! CK_F555 not found: $CK_F555"
    echo
    echo "  ./run_cmd_fullframe.sh data     then     ./run_cmd_fullframe.sh solve"
    echo "  on Vista:  sbatch cmd_fullframe.slurm"
    ;;
esac
