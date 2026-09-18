"""Staged gradient-free sysid fit for the mjlab port.

    uv run python -m deploy.sysid.fit --space joint --runs data/*.npz --out out/joint.json

Input runs use the shared sysid schema (see the TWIST2 ``deploy_real/sysid/dataio``
for the field list): t, q, dq, tau_est, q_target, quat, ang_vel.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import DEFAULT_POS
from .plant import ModelBundle, fit_es, make_eval
from . import space as spaces


@dataclass
class Stage:
    name: str
    active: tuple[str, ...]
    iterations: int


def _stages(kind: str):
    if kind == "susp":
        return [Stage("susp", ("susp_k", "susp_c", "d_ctrl"), 200)]
    if kind == "softbase":
        return [Stage("softbase", ("susp_k", "susp_c", "d_ctrl"), 200),
                Stage("payload", ("payload_mass", "payload_com"), 150)]
    return [
        Stage("friction", ("frictionloss", "damping", "d_ctrl"), 250),
        Stage("inertia", ("armature", "effort_limit"), 250),
        Stage("lag", ("tau_m", "d_ctrl"), 150),
        Stage("joint", ("armature", "frictionloss", "damping", "effort_limit", "tau_m", "d_ctrl"), 300),
    ]


def _resample(t, x, dt):
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = int(np.floor((t[-1] - t[0]) / dt)) + 1
    t_dst = t[0] + np.arange(n) * dt
    flat = x.reshape(x.shape[0], -1)
    out = np.empty((n, flat.shape[1]))
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(t_dst, t, flat[:, c])
    return t_dst, out.reshape((n,) + x.shape[1:])


def load_batch(paths, window, stride, dt, warmup=0):
    pieces = []
    for p in paths:
        with np.load(p, allow_pickle=False) as d:
            t, q = _resample(d["t"], d["q"], dt)
            _, dq = _resample(d["t"], d["dq"], dt)
            _, tau = _resample(d["t"], d["tau_est"], dt)
            _, u = _resample(d["t"], d["q_target"], dt)
            _, quat = _resample(d["t"], d["quat"], dt) if "quat" in d else (t, np.tile([1.0, 0, 0, 0], (q.shape[0], 1)))
            _, w = _resample(d["t"], d["ang_vel"], dt) if "ang_vel" in d else (t, np.zeros((q.shape[0], 3)))
        pieces.append((q, dq, tau, u, quat, w))
    q = np.concatenate([p[0] for p in pieces], axis=0)
    dq = np.concatenate([p[1] for p in pieces], axis=0)
    tau = np.concatenate([p[2] for p in pieces], axis=0)
    u = np.concatenate([p[3] for p in pieces], axis=0)
    quat = np.concatenate([p[4] for p in pieces], axis=0)
    w = np.concatenate([p[5] for p in pieces], axis=0)
    n = q.shape[0]
    span = window + warmup
    starts = np.arange(1 + warmup, max(2 + warmup, n - window), stride, dtype=np.int64)
    idx = (starts - warmup)[:, None] + np.arange(span)[None, :]
    prev = starts - 1 - warmup
    score = starts[:, None] + np.arange(window)[None, :]
    batch = {
        "u": u[idx],
        "q0": q[prev],
        "v0": dq[prev],
        "q": q[score],
        "v": dq[score],
        "tau": tau[score],
        "quat0": quat[prev],
        "quat": quat[score],
        "w0": w[prev],
        "w": w[score],
        "base_pos": np.zeros((starts.size, 3)),
        "warmup": warmup,
    }
    return batch, starts.size


def build_space(kind: str):
    return {"susp": spaces.susp_space, "softbase": spaces.softbase_space}.get(kind, spaces.joint_space)()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", choices=("susp", "joint", "softbase"), default="joint")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--xml", default="deploy/sysid/g1_sysid.xml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--decimation", type=int, default=20)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--susp-params", default=None)
    parser.add_argument("--iters-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    space = build_space(args.space)
    stages = _stages(args.space)
    if args.iters_scale != 1.0:
        stages = [Stage(s.name, s.active, max(1, int(round(s.iterations * args.iters_scale)))) for s in stages]

    batch, n_windows = load_batch(args.runs, args.window, args.stride, args.dt, warmup=args.warmup)
    if n_windows == 0:
        raise SystemExit("no windows; run is shorter than --window")
    if args.max_windows and n_windows > args.max_windows:
        keep = np.linspace(0, n_windows - 1, args.max_windows).astype(int)
        batch = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in batch.items()}
        n_windows = args.max_windows
    print(f"[fit] space={args.space} windows={n_windows} window={args.window}")

    raw_init = None
    if args.susp_params:
        data = spaces.load_identified(args.susp_params)
        overrides = {k: np.asarray(data["theta"][k]) for k in ("susp_k", "susp_c") if k in data.get("theta", {})}
        if overrides:
            raw_init = space.init_raw_with(overrides)
            print(f"[fit] froze suspension from {args.susp_params}")

    bundle = ModelBundle(args.xml, timestep=args.dt)
    evaluate = make_eval(bundle, space, batch, decimation=args.decimation)
    print(f"[fit] initial loss {evaluate(space.init_raw() if raw_init is None else raw_init):.6e}")
    raw, history = fit_es(
        evaluate, space, stages,
        raw_init=raw_init, population=args.population, seed=args.seed, log_fn=print,
    )
    print(f"[fit] final loss {evaluate(raw):.6e}")
    spaces.save_identified(args.out, space, raw, meta={"space": args.space, "runs": args.runs, "windows": int(n_windows)})
    Path(args.out).with_suffix(".history.json").write_text(json.dumps(history, indent=2))
    print(f"[fit] wrote {args.out}")
    for name, values in space.unpack(raw).items():
        arr = np.asarray(values).reshape(-1)
        print(f"[fit]   {name}: mean={arr.mean():.6g} min={arr.min():.6g} max={arr.max():.6g}")


if __name__ == "__main__":
    main()
