#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 INPUT_FASTA OUTPUT_DIR [MMSEQS_BINARY]" >&2
  exit 2
fi

input_fasta=$1
output_dir=$2
mmseqs_binary=${3:-mmseqs}

if [[ ! -f "$input_fasta" ]]; then
  echo "Input FASTA not found: $input_fasta" >&2
  exit 2
fi
if ! command -v "$mmseqs_binary" >/dev/null 2>&1 && [[ ! -x "$mmseqs_binary" ]]; then
  echo "MMseqs2 executable not found: $mmseqs_binary" >&2
  exit 2
fi

mkdir -p "$output_dir"

for identity in 0.95 0.90; do
  label=${identity/0./}
  result_prefix="$output_dir/mmseqs_${label}"
  temporary_dir="$output_dir/tmp_${label}"
  "$mmseqs_binary" easy-cluster \
    "$input_fasta" \
    "$result_prefix" \
    "$temporary_dir" \
    --min-seq-id "$identity" \
    -c 0.8 \
    --cov-mode 0 \
    --threads 4
done

echo "MMseqs2 clustering finished: $output_dir"
