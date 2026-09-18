#!/usr/bin/env bash
# Shared env-var -> CLI-flag forwarding for the TWIST2 deploy launchers.
#
# Source this file, then expand "${POLICY_TUNE_ARGS[@]}" on the policy node and
# "${LOWLEVEL_TUNE_ARGS[@]}" on the sim / hardware node. Empty env vars are
# omitted so each node keeps its own argparse default.

POLICY_TUNE_ARGS=()
[[ -n "${TWIST2_LEG_SMOOTH_ALPHA:-}" ]] && POLICY_TUNE_ARGS+=(--leg_smooth_alpha "${TWIST2_LEG_SMOOTH_ALPHA}")
[[ -n "${TWIST2_ARM_SMOOTH_ALPHA:-}" ]] && POLICY_TUNE_ARGS+=(--arm_smooth_alpha "${TWIST2_ARM_SMOOTH_ALPHA}")
[[ -n "${TWIST2_SMOOTH_BODY:-}" ]]       && POLICY_TUNE_ARGS+=(--smooth_body "${TWIST2_SMOOTH_BODY}")
[[ -n "${TWIST2_SMOOTH_WINDOW:-}" ]]     && POLICY_TUNE_ARGS+=(--smooth_window_size "${TWIST2_SMOOTH_WINDOW}")

LOWLEVEL_TUNE_ARGS=()
[[ -n "${TWIST2_LEG_PD_GAIN:-}" ]]   && LOWLEVEL_TUNE_ARGS+=(--leg_pd_gain "${TWIST2_LEG_PD_GAIN}")
[[ -n "${TWIST2_ARM_PD_GAIN:-}" ]]   && LOWLEVEL_TUNE_ARGS+=(--arm_pd_gain "${TWIST2_ARM_PD_GAIN}")
[[ -n "${TWIST2_LEG_EMA_ALPHA:-}" ]] && LOWLEVEL_TUNE_ARGS+=(--leg_ema_alpha "${TWIST2_LEG_EMA_ALPHA}")

: # keep the sourced file's exit status 0 under `set -e`
