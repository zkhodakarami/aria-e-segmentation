#!/usr/bin/env bash
# Predict ARIA-E lesion masks on new FLAIR scans with the trained 5-fold ensemble.
#
#   ./scripts/predict.sh <input_dir> <output_dir>
#
# input_dir must contain nnU-Net-style filenames: <case>_0000.nii.gz (one FLAIR
# per case, single channel). Example:
#   mkdir -p /tmp/in && cp scan.nii.gz /tmp/in/CASE001_0000.nii.gz
#   ./scripts/predict.sh /tmp/in /tmp/out
set -euo pipefail
ENV_DIR="${ENV_DIR:-$HOME/.conda/envs/nnunet_ariae}"
export PATH="$ENV_DIR/bin:$PATH"
export nnUNet_raw="${nnUNet_raw:-/path/to/nnunet/work_dir/nnUNet_raw_data_base}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/path/to/nnunet/work_dir/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-/path/to/nnunet/work_dir/nnUNet_trained_models}"

IN=${1:?usage: predict.sh <input_dir> <output_dir>}
OUT=${2:?usage: predict.sh <input_dir> <output_dir>}
mkdir -p "$OUT"

nnUNetv2_predict \
  -i "$IN" -o "$OUT" \
  -d 501 -c 3d_fullres \
  -tr nnUNetTrainer_500epochs \
  -f 0 1 2 3 4 \
  --save_probabilities

echo "predictions -> $OUT"
