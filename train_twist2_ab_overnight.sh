#!/usr/bin/env bash
# Overnight equal-budget A/B for the TWIST2 differentiable auxiliary objective.
#
# Runs sequentially, on the same motion dataset and seed (42):
#   1) baseline : plain PPO (aux disabled)
#   2) diff-aux : Twist2PPO with the learned world-model auxiliary objective
#
# Usage:
#   TWIST2_MOTION_FILE=/path/to/dataset.yaml bash train_twist2_ab_overnight.sh [ITERS] [GPU_ID]
#
# Defaults: ITERS=12000, GPU_ID=0.

set -euo pipefail

ITERS="${1:-12000}"
GPU_ID="${2:-0}"
MOTION_FILE="${TWIST2_MOTION_FILE:-/home/user/twist2_data/enriched/dataset.yaml}"
export TWIST2_MOTION_FILE="${MOTION_FILE}"

if [[ ! -f "${MOTION_FILE}" ]]; then
  echo "Error: motion file not found: ${MOTION_FILE}" >&2
  exit 1
fi

echo "[AB] iters=${ITERS} gpu=${GPU_ID}"
echo "[AB] motion=${MOTION_FILE}"

echo "[AB] === 1/2 baseline (aux off) ==="
TWIST2_ENABLE_AUX=0 bash train_twist2.sh "${GPU_ID}" \
  --agent.max-iterations "${ITERS}" \
  --agent.logger tensorboard

echo "[AB] === 2/2 differentiable aux (world_model) ==="
TWIST2_ENABLE_AUX=1 bash train_twist2.sh "${GPU_ID}" \
  --agent.max-iterations "${ITERS}" \
  --agent.algorithm.aux-mode world_model \
  --agent.algorithm.aux-coef 0.1 \
  --agent.algorithm.aux-coef-warmup-iters 1000 \
  --agent.algorithm.aux-model-lr 3e-4 \
  --agent.logger tensorboard

echo "[AB] done"
