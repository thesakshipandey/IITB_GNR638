#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_LIB="${SCRIPT_DIR}/libgnr_cuda_backend.so"
SOURCE_FILE="${SCRIPT_DIR}/gnr_cuda_backend.cu"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found. Install CUDA toolkit and try again."
  exit 1
fi

nvcc -O3 -Xcompiler -fPIC -shared "${SOURCE_FILE}" -o "${OUTPUT_LIB}"
echo "Built CUDA backend: ${OUTPUT_LIB}"
