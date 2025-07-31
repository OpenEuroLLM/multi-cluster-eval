#!/usr/bin/env bash
# Build all Apptainer (.def) files in the apptainer/ directory into SIF images.
# Requires: Apptainer installed and configured (e.g., via Lima VM on macOS).
# Usage: ./build_sif_local.sh
set -euo pipefail

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
APPTAINER_DIR="${ROOT_DIR}/apptainer"

# Allow overriding the output directory; default to a writable location under HOME
# This avoids “read-only file system” errors when the project folder is mounted read-only in Lima.
OUTPUT_DIR="${SIF_OUTPUT_DIR:-$HOME/apptainer_images}"
mkdir -p "${OUTPUT_DIR}"

build_one() {
  local def_file="$1"
  local base_name="$(basename "${def_file%.def}")"
  local sif_name="eval_env-${base_name}.sif"
  local sif_path="${OUTPUT_DIR}/${sif_name}"
  echo "=== Building ${sif_name} from ${def_file}" >&2
  # --fakeroot lets unprivileged users build images (works inside Lima Apptainer setup)
  apptainer build --fakeroot "${sif_path}" "${def_file}"
  echo "=== Finished: ${sif_path}" >&2
}

for def in "${APPTAINER_DIR}"/*.def; do
  [ -e "${def}" ] || { echo "No .def files found in ${APPTAINER_DIR}" >&2; exit 1; }
  build_one "${def}"
done

echo "\nAll SIF images built successfully. Find them under: ${OUTPUT_DIR}" 