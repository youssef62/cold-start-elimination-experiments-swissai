#!/bin/bash
# The chain: for each model, record the gold if it is missing, then run every
# other arm on nodes cold for the bytes IT reads. Page cache survives between
# jobs, so an arm landing on a node that already read its input measures a warm
# read. The engine-loader arms read the source checkpoint and the servekit arms
# read the artifact -- two disjoint file sets -- so they are tracked separately.
# Excluding every node any arm ever touched instead of every node that read the
# same bytes makes the last arms of a large model unschedulable on a busy
# cluster, for no measurement benefit.
#
# Order is fastest-first after the gold arm, so a sweep that runs out of time
# still has the interesting rows.
#
#   ./sweep.sh                                     # all three models
#   MODELS="llama70b glm4.7" ./sweep.sh
#   MODEL=apertus8b ARMS="mmap servekit" ./sweep.sh
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd -P)"
MODELS="${MODELS:-${MODEL:-apertus8b llama70b glm4.7}}"
ARMS="${ARMS:-mmap servekit-overlap servekit fst nommap mmap2}"

submit() {
  local model="$1" arm="$2" used="$3"
  local exclude=()
  [ -n "${used}" ] && exclude=(--exclude="${used}")
  MODEL="${model}" "${here}/submit.sh" "${arm}" "${exclude[@]}" --parsable
}

wait_for() {
  # squeue occasionally fails to fork on a busy login node; retry rather than
  # treat empty output as "job left the queue" (that misreads a job as
  # finished while it is still PENDING and silently drops it from tracking).
  local state
  while true; do
    state="$(squeue -h -j "$1" -o "%T" 2>/dev/null)" || { sleep 5; continue; }
    [[ "${state}" =~ PENDING|RUNNING|CONFIGURING|COMPLETING ]] || break
    sleep 30
  done
  # %300: the default NodeList width truncates a 4-node list to "nid[002280,002+",
  # which sbatch then rejects as an invalid --exclude.
  sacct -j "$1" -X -n -o NodeList%300 | tr -d ' '
}

for model in ${MODELS}; do
  # USED_SOURCE / USED_ARTIFACT can be pre-seeded with nodes an earlier job
  # already warmed the respective cache on.
  used_source="${USED_SOURCE:-}"
  used_artifact="${USED_ARTIFACT:-}"
  echo "=== ${model} ==="

  for arm in ${ARMS}; do
    # The gold is recorded once per model, by whichever mmap arm runs first.
    unset VERIFY_MODE
    if [ ! -f "${here}/results/${CLUSTER:-bristen}-${model}/${model}-gold.json" ]; then
      if [ "${arm}" = "mmap" ]; then
        export VERIFY_MODE=record
      else
        echo "$(date +%T) ${model}: no gold yet, skipping ${arm}"
        continue
      fi
    fi

    case "${arm}" in
      servekit*) reads=artifact; exclude="${used_artifact}" ;;
      *)         reads=source;   exclude="${used_source}" ;;
    esac

    jid="$(submit "${model}" "${arm}" "${exclude}")"
    echo "$(date +%T) submitted ${model}/${arm} as ${jid} (reads ${reads})${VERIFY_MODE:+ (recording gold)}"

    node="$(wait_for "${jid}")"
    echo "$(date +%T) ${model}/${arm} (${jid}) finished on ${node}"
    if [ "${reads}" = "artifact" ]; then
      used_artifact="${used_artifact:+${used_artifact},}${node}"
    else
      used_source="${used_source:+${used_source},}${node}"
    fi
  done

  echo "${model} read the source on: ${used_source:-none}"
  echo "${model} read the artifact on: ${used_artifact:-none}"
done
