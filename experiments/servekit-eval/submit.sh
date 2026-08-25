#!/bin/bash
# MODEL=<apertus8b|llama70b|glm4.7> ./submit.sh <prepare|mmap|mmap2|nommap|fst|servekit|servekit-overlap> [sbatch args...]
#
# VERIFY_MODE=record ./submit.sh mmap    writes this model's gold logprobs
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd -P)"
arm="${1:?usage: MODEL=<model> ./submit.sh <prepare|mmap|mmap2|nommap|fst|servekit|servekit-overlap> [sbatch args...]}"
shift || true

source "${here}/scripts/models.sh"
mkdir -p "${here}/${RESULTS_DIR}"

case "${arm}" in
  prepare) script=scripts/prepare.sbatch ;;
  mmap|nommap|fst|servekit|servekit-overlap)
           script=scripts/serve.sbatch ;;
  # same loader as mmap, own job name so the summarizer keeps the timed baseline
  # and the determinism gate apart.
  mmap2)   script=scripts/serve.sbatch; export_arm=mmap ;;
  *) echo "unknown arm: ${arm}" >&2; exit 1 ;;
esac

# --nodes, --time and --job-name go here because #SBATCH lines cannot read models.sh.
exec sbatch \
  --job-name="${MODEL}-${arm}" \
  --nodes="${NNODES}" \
  --time="${TIME_LIMIT}" \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --output="${here}/${RESULTS_DIR}/%x-%j.out" \
  --export="ALL,MODEL=${MODEL},ARM=${export_arm:-${arm}}" \
  "$@" "${here}/${script}"
