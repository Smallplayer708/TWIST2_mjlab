#!/usr/bin/env bash
# TWIST2-mjlab sim2sim under real-time teleop control (Redis -> mjlab policy).
#
# Launches two nodes:
#   1. twist2_policy_redis.py  — reads 35D mimic from Redis (published by the
#       original TWIST2 teleop.sh), runs the mjlab ONNX, sends action via UDP.
#   2. sim_node.py             — MuJoCo physics + viewer with green ghost overlay.
#
# Usage:
#   ./deploy/play_sim_twist2_redis.sh /path/to/model.onnx [--redis-ip IP] [--redis-port PORT]
#
# Start the teleop first (separate terminal):  bash teleop.sh  (in TWIST2 repo)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"

ONNX_MODEL="${1:-}"
if [[ -z "${ONNX_MODEL}" ]]; then
  echo "Usage: $0 /path/to/model.onnx [--redis-ip IP] [--redis-port PORT]"
  echo "  e.g.  $0 resources/pretrained.onnx --redis-ip localhost"
  exit 1
fi
shift || true

cleanup() {
  trap '' SIGINT SIGTERM EXIT
  echo -e '\nStopping...'
  pkill -f "deploy/policy/twist2_policy_redis.py" 2>/dev/null || true
  pkill -f "deploy/sim/sim_node.py"              2>/dev/null || true
}
trap cleanup SIGINT SIGTERM EXIT

export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}:${ROOT_DIR}/src"
cd "${ROOT_DIR}"

# Optional tuning passthrough (env vars, empty = node defaults).
source "${SCRIPT_DIR}/common/forward_env_args.sh"

# Policy node (background)
echo "Starting TWIST2 Redis Policy Node (background)..."
uv run python deploy/policy/twist2_policy_redis.py "${ONNX_MODEL}" "$@" "${POLICY_TUNE_ARGS[@]}" &

sleep 2.0

# Sim node (foreground)
echo "Starting Simulation Node (foreground)..."
uv run python deploy/sim/sim_node.py "${LOWLEVEL_TUNE_ARGS[@]}"
