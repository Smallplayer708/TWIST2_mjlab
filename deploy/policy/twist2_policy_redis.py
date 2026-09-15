"""TWIST2 motion-tracking policy — Redis (teleop) + UDP (sim) bridge.

Reads the 35D mimic observation from Redis (published by the original TWIST2
teleop pipeline: ``xrobot_teleop_to_robot_w_hand.py``), builds the 1524D mjlab
observation, runs the mjlab ONNX, and sends the action + a ghost reference
pose to sim_node via UDP for visualization.

Redis key: ``action_body_unitree_g1_with_hands`` (JSON list of 35 floats).

Run via ``deploy/play_sim_twist2_redis.sh`` so PYTHONPATH is set correctly.
"""

import argparse
import json
import math
import socket
import time
from collections import deque

import numpy as np
import onnxruntime as ort
import redis

# Reuse training-matched constants / helpers from the pkl-based policy node
# (these were already validated against mjlab's sim2sim).
from deploy.policy.twist2_policy import (
    ACTOR_HISTORY_LENGTH,
    ACTOR_OBS_DIM,
    DEFAULT_POS,
    DEFAULT_STANDING_MIMIC,
    JOINT_SCALES,
    NUM_JOINTS,
    build_proprio,
)
from deploy.common.udp_sync import (
    UDP_HOST,
    UDP_POLICY_PORT,
    UDP_SIM_PORT,
    STATE_BYTES,
    pack_action,
    unpack_state,
)

REDIS_ACTION_KEY = "action_body_unitree_g1_with_hands"


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build a wxyz quaternion from ZYX euler angles (radians)."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_path", help="Path to twist2 .onnx model")
    parser.add_argument("--redis-ip", default="localhost", help="Redis host (teleop publisher)")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    args = parser.parse_args()

    session = ort.InferenceSession(args.onnx_path, providers=["CPUExecutionProvider"])
    inp_name = session.get_inputs()[0].name
    actual = tuple(session.get_inputs()[0].shape)
    print(f"ONNX input {actual} (expect (1, {ACTOR_OBS_DIM}))")
    if actual[1] != ACTOR_OBS_DIM:
        print(f"ERROR: ONNX obs dim {actual[1]} != {ACTOR_OBS_DIM}; wrong model?")

    r = redis.Redis(host=args.redis_ip, port=args.redis_port, db=0)

    last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
    mimic = DEFAULT_STANDING_MIMIC.copy()
    dt = 1.0 / 50.0
    history: deque[np.ndarray] = deque(maxlen=ACTOR_HISTORY_LENGTH)
    history_initialized = False

    # Ghost reference pose accumulation (integrate from mimic velocities).
    ref_yaw = 0.0
    ref_x = 0.0
    ref_y = 0.0

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_HOST, UDP_POLICY_PORT))
    sock.settimeout(1.0)
    sim_addr = (UDP_HOST, UDP_SIM_PORT)
    print(
        f"TWIST2 redis policy: redis={args.redis_ip}:{args.redis_port} "
        f"udp={UDP_HOST}:{UDP_POLICY_PORT} -> sim {UDP_HOST}:{UDP_SIM_PORT}"
    )

    next_tick = time.perf_counter()
    try:
        while True:
            now = time.perf_counter()
            if next_tick - now > 0:
                time.sleep(next_tick - now)
            next_tick += dt

            # 1. Latest mimic from Redis (teleop). Keep previous on timeout.
            msg = r.get(REDIS_ACTION_KEY)
            if msg is not None:
                try:
                    mimic = np.asarray(json.loads(msg), dtype=np.float32)
                except (json.JSONDecodeError, TypeError):
                    pass

            # 2. Drain to latest state packet from sim.
            latest_data = None
            sock.setblocking(False)
            try:
                while True:
                    latest_data, _ = sock.recvfrom(STATE_BYTES + 64)
            except BlockingIOError:
                pass
            sock.setblocking(True)
            sock.settimeout(1.0)
            if latest_data is None:
                continue

            step_id, root_quat, root_pos, body_lin_vel, body_ang_vel, \
                joint_pos, joint_vel = unpack_state(latest_data)

            # 3. Build 1524D observation.
            proprio = build_proprio(joint_pos, joint_vel, root_quat, body_ang_vel, last_action)
            actor_current = np.concatenate([mimic, proprio]).astype(np.float32)
            if not history_initialized:
                for _ in range(ACTOR_HISTORY_LENGTH):
                    history.append(actor_current.copy())
                history_initialized = True
            else:
                history.append(actor_current.copy())
            actor_history = np.concatenate(list(history)).astype(np.float32)
            obs = np.concatenate([actor_current, actor_history]).reshape(1, -1)

            # 4. Inference.
            raw_action = session.run(None, {inp_name: obs})[0][0]
            target_pos = (DEFAULT_POS + raw_action * JOINT_SCALES).astype(np.float32)
            last_action = raw_action.copy()

            # 5. Ghost reference pose (for the green overlay in sim_node).
            ref_yaw += float(mimic[5]) * dt
            vx, vy = float(mimic[0]), float(mimic[1])
            world_vx = vx * math.cos(ref_yaw) - vy * math.sin(ref_yaw)
            world_vy = vx * math.sin(ref_yaw) + vy * math.cos(ref_yaw)
            ref_x += world_vx * dt
            ref_y += world_vy * dt
            ref_root_pos = np.array([ref_x, ref_y, float(mimic[2])], dtype=np.float32)
            ref_root_quat = euler_to_quat(float(mimic[3]), float(mimic[4]), ref_yaw)
            ref_joint_pos = mimic[6:35].astype(np.float32)

            # 6. Send action + ghost pose.
            sock.sendto(
                pack_action(step_id, target_pos, ref_root_pos, ref_root_quat, ref_joint_pos),
                sim_addr,
            )

    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        print("TWIST2 redis policy stopped.")


if __name__ == "__main__":
    main()
