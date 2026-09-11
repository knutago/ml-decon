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
CK=$ML/m32sim_arms/ckpt_m32sim_f555w_wide/best.pt
OUT=${OUT:-cmd_bands_full}
ITERS=${ITERS:-450}
CHUNK=${CHUNK:-16}          # patches per chunk = one 128-px row strip
REGION=${REGION:-0:2048,0:2048}

# band : psf : sky measured on the 512^2 cut : that cut's clipped block median
ARMS=("f435w:psf_m32_v2_centred.fits:256.87:180.5"
      "f555w:psf_m32v_v2_centred.fits:231.12:354.5")

build_data () {
  for arm in "${ARMS[@]}"; do
    IFS=: read -r band _ _ _ <<<"$arm"
    echo "--- dataset $band over $REGION"
    $PY make_band_patches.py --band "$band" --patch 128 --stride 128 \
        --region "$REGION" --ckpt "$CK" --train-ref m32_sim_f555w_p128 \
        --out "$ML/data/m32_full_$band"
  done
}

solve () {
  for arm in "${ARMS[@]}"; do
    IFS=: read -r band psf sky0 med0 <<<"$arm"
    D=$ML/data/m32_full_$band
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
sys.path.insert(0,"/home/alex/noir_ml/mycode")
from red_pnp_deconvolve import TorchNorm
from astropy.stats import sigma_clipped_stats
d,s,e,sky0,med0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
n = json.load(open(d+"/norm.json")); m = json.load(open(d+"/meta.json"))
a = m["normalization"]["scale_a"]; mo = m["normalization"]["observed_clipped_median"]
z = np.load(d+"/val_observed.npy", mmap_mode="r")[s:e]
flux = TorchNorm(n["observed"]).inverse(torch.from_numpy(np.asarray(z)).double()).numpy()
_, med_flux, _ = sigma_clipped_stats(flux, sigma=3.0, maxiters=5)
# med0 is in raw counts; convert to this dataset's flux domain to compare
med0_flux = a*(med0-mo) + n["observed"]["median"]
print(f"{sky0 + (med_flux - med0_flux):.3f}")
EOF
)
      echo "  [$s:$e] sky=$SKY -> $CD"
      $PY -u admm_diffusion_deconvolve.py --prior cond --split val \
          --indices "$s:$e" --split-norms --gain 1.0 --sparse-tau 0 \
          --rho 0.1 --rho-scale 1.0 --eta 0.0 --avg-last 10 \
          --denoise-steps 1 --t-max 55 --t-min 1 --lam 0.01 --renoise \
          --iters "$ITERS" --n-show 0 --no-fits --log-every 100 \
          --ckpt "$CK" --data-dir "$D" --psf-file "$psf" --sky "$SKY" \
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
    echo "  prior       ckpt_m32sim_f555w_wide, SHARED across both bands"
    echo "  sky         self-consistent, rival-free (F435W 256.87, F555W 231.12"
    echo "              at the reference strip; offset per strip)"
    echo
    echo "  ./run_cmd_fullframe.sh data     then     ./run_cmd_fullframe.sh solve"
    echo "  on Vista:  sbatch cmd_fullframe.slurm"
    ;;
esac
