"""Real excitation + recording for the mjlab G1 (unitree_interface).

Mirrors `deploy/real/hardware_node.py` transport but sends a scripted excitation
instead of policy targets and records the full sysid schema, including the
`tau_est` / `voltage` / `temperature` fields hardware_node does not currently
read. Start suspended; SELECT aborts.

Example:
    uv run python -m deploy.sysid.record_real_sysid --net eth0 --joint 3 \
        --kind multisine --duration 20 --out deploy/sysid/data/susp_knee.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .constants import DEFAULT_POS, JOINT_NAMES, KD, KP, NUM_JOINTS
from .excitation import ExcitationSpec, SafetyLimiter, build_targets

CONTROLLER_MAPPING = {"select": 0x0008}


def _btn(ctrl, mask: int) -> bool:
    return bool(getattr(ctrl, "keys", 0) & mask)


def _send_pd(robot, q_target, kp, kd):
    cmd = robot.create_zero_command()
    cmd.q_target = list(np.asarray(q_target, dtype=float))
    cmd.dq_target = [0.0] * NUM_JOINTS
    cmd.kp = list(np.asarray(kp, dtype=float))
    cmd.kd = list(np.asarray(kd, dtype=float))
    cmd.tau_ff = [0.0] * NUM_JOINTS
    robot.write_low_command(cmd)


def record(args: argparse.Namespace) -> None:
    import unitree_interface

    robot = unitree_interface.UnitreeInterface.create_g1(args.net)
    robot.set_control_mode(unitree_interface.ControlMode.PR)

    hold = DEFAULT_POS.copy()
    limiter = SafetyLimiter(max_rate=args.max_rate, temp_limit=args.temp_limit, voltage_min=args.voltage_min)
    spec = ExcitationSpec(
        joint_index=args.joint, kind=args.kind, duration=args.duration, dt=args.dt,
        amplitude=args.amplitude, f_lo=args.f_lo, f_hi=args.f_hi, n_tones=args.n_tones, seed=args.seed,
    )
    targets = build_targets(spec, hold_pose=hold)

    state = robot.read_low_state()
    start = np.array(state.motor.q[:NUM_JOINTS], dtype=np.float64)
    limiter.reset(start)

    t_rec, rec = [], {k: [] for k in ("q", "dq", "tau", "qt", "vol", "temp", "quat", "w", "acc", "abort")}
    t0 = time.monotonic()
    aborted = False

    def emit(target):
        nonlocal aborted
        state = robot.read_low_state()
        q = np.array(state.motor.q[:NUM_JOINTS], dtype=np.float64)
        dq = np.array(state.motor.dq[:NUM_JOINTS], dtype=np.float64)
        tau = np.array(state.motor.tau_est[:NUM_JOINTS], dtype=np.float64)
        vol = np.array(state.motor.voltage[:NUM_JOINTS], dtype=np.float64)
        temp = np.array(state.motor.temperature[:NUM_JOINTS], dtype=np.float64)
        res = limiter.watchdog(temp, vol)
        if res is not None:
            aborted = True
        u = limiter(target, args.dt, temp, vol)
        _send_pd(robot, u, KP, KD)
        t_rec.append(time.monotonic() - t0)
        rec["q"].append(q); rec["dq"].append(dq); rec["tau"].append(tau)
        rec["qt"].append(u); rec["vol"].append(vol); rec["temp"].append(temp)
        rec["quat"].append(np.array(state.imu.quat, dtype=np.float64))
        rec["w"].append(np.array(state.imu.omega, dtype=np.float64))
        rec["acc"].append(np.array(state.imu.accel, dtype=np.float64))
        rec["abort"].append(1.0 if res is not None else 0.0)
        if res is not None:
            print(f"[record] WATCHDOG {res} -> abort")

    ramp_n = max(1, int(round(args.ramp / args.dt)))
    print(f"[record] ramp {args.ramp}s, excitation {args.duration}s on {JOINT_NAMES[args.joint]}")
    try:
        for i in range(ramp_n):
            a = (i + 1) / ramp_n
            emit(start * (1 - a) + hold * a)
            time.sleep(args.dt)
        for i in range(targets.shape[0]):
            if aborted or _btn(robot.read_wireless_controller(), CONTROLLER_MAPPING["select"]):
                aborted = True
                break
            emit(targets[i])
            time.sleep(args.dt)
        for i in range(ramp_n):
            emit(hold)
            time.sleep(args.dt)
    except KeyboardInterrupt:
        print("[record] interrupted")
    finally:
        _send_pd(robot, hold, np.zeros(NUM_JOINTS), KD)

    t_arr = np.asarray(t_rec)
    n = t_arr.size
    out = {
        "t": t_arr - t_arr[0],
        "q": np.asarray(rec["q"]).reshape(n, NUM_JOINTS),
        "dq": np.asarray(rec["dq"]).reshape(n, NUM_JOINTS),
        "tau_est": np.asarray(rec["tau"]).reshape(n, NUM_JOINTS),
        "q_target": np.asarray(rec["qt"]).reshape(n, NUM_JOINTS),
        "kp": np.tile(KP.reshape(1, -1), (n, 1)),
        "kd": np.tile(KD.reshape(1, -1), (n, 1)),
        "tau_ff": np.zeros((n, NUM_JOINTS)),
        "voltage": np.asarray(rec["vol"]).reshape(n, NUM_JOINTS),
        "temp": np.asarray(rec["temp"]).reshape(n, NUM_JOINTS),
        "quat": np.asarray(rec["quat"]).reshape(n, 4),
        "ang_vel": np.asarray(rec["w"]).reshape(n, 3),
        "accel": np.asarray(rec["acc"]).reshape(n, 3),
        "mode": np.array("susp"),
        "joint_index": np.array(args.joint),
        "safety_abort": np.asarray(rec["abort"]).reshape(n),
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    print(f"[record] saved {path} ({n} samples, aborted={aborted})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--net", default="eth0")
    parser.add_argument("--joint", type=int, default=3)
    parser.add_argument("--kind", default="multisine", choices=("multisine", "chirp", "prbs", "steps", "sweep", "quasi_static"))
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--amplitude", type=float, default=0.05)
    parser.add_argument("--f-lo", type=float, default=0.5)
    parser.add_argument("--f-hi", type=float, default=10.0)
    parser.add_argument("--n-tones", type=int, default=12)
    parser.add_argument("--ramp", type=float, default=2.0)
    parser.add_argument("--max-rate", type=float, default=8.0)
    parser.add_argument("--temp-limit", type=float, default=75.0)
    parser.add_argument("--voltage-min", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="deploy/sysid/data/susp_run.npz")
    args = parser.parse_args()
    record(args)


if __name__ == "__main__":
    main()
