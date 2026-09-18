"""Plain-MuJoCo plant + gradient-free fitting for the mjlab sysid port.

The sysid MJCF uses position actuators (as in mjlab training), so the command is
a joint target: delay -> first-order motor lag -> ``data.ctrl``. Joint
``armature``/``frictionloss``/``damping``, actuator ``forcerange`` and an
optional torso payload are the differentiable/identified quantities.

This backend runs with mujoco + numpy only. ``mujoco_warp`` batching is the
intended accelerator for the 29-joint joint stage (see README).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

import mujoco

from .constants import DEFAULT_POS, JOINT_NAMES, NUM_JOINTS
from .space import ParamSpace


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = q / max(1e-12, float(np.linalg.norm(q)))
    if q[0] < 0:
        q = -q
    v = q[1:4]
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.zeros(3)
    return v / s * (2.0 * np.arctan2(s, q[0]))


@dataclass
class RolloutResult:
    q: np.ndarray
    v: np.ndarray
    quat: np.ndarray
    ang_vel: np.ndarray
    tau: np.ndarray
    base_pos: np.ndarray
    base_lin_vel: np.ndarray


class ModelBundle:
    def __init__(self, xml_path: str, timestep: float = 0.001) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.model.opt.timestep = float(timestep)
        self.timestep = float(timestep)
        self.qpos_adr = np.array(
            [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES],
            dtype=np.int64,
        )
        self.dof_adr = np.array(
            [self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES],
            dtype=np.int64,
        )
        self.ctrl_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINT_NAMES],
            dtype=np.int64,
        )
        self.free_dof = int(self.model.jnt_dofadr[0])
        self.free_qpos = int(self.model.jnt_qposadr[0])
        self.torso_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"))
        self._defaults = {
            "dof_armature": self.model.dof_armature.copy(),
            "dof_frictionloss": self.model.dof_frictionloss.copy(),
            "dof_damping": self.model.dof_damping.copy(),
            "actuator_forcerange": self.model.actuator_forcerange.copy(),
            "body_mass": self.model.body_mass.copy(),
            "body_ipos": self.model.body_ipos.copy(),
        }

    def reset(self) -> None:
        for field, value in self._defaults.items():
            getattr(self.model, field)[:] = value

    def apply_theta(self, theta: Mapping[str, np.ndarray]) -> None:
        self.reset()
        for key, field in (("armature", "dof_armature"), ("frictionloss", "dof_frictionloss"), ("damping", "dof_damping")):
            if key in theta:
                getattr(self.model, field)[self.dof_adr] = np.asarray(theta[key], dtype=np.float64)
        if "effort_limit" in theta:
            limit = np.asarray(theta["effort_limit"], dtype=np.float64)
            self.model.actuator_forcerange[self.ctrl_ids, 0] = -limit
            self.model.actuator_forcerange[self.ctrl_ids, 1] = limit
        if self.torso_id >= 0 and "payload_mass" in theta:
            self.model.body_mass[self.torso_id] += float(np.asarray(theta["payload_mass"]).reshape(-1)[0])
        if self.torso_id >= 0 and "payload_com" in theta:
            self.model.body_ipos[self.torso_id] += np.asarray(theta["payload_com"], dtype=np.float64).reshape(3)


def _delayed(u: np.ndarray, delay_steps: float) -> np.ndarray:
    n = u.shape[0]
    idx = np.arange(n, dtype=np.float64) - float(delay_steps)
    i0 = np.clip(np.floor(idx), 0, n - 1).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, n - 1)
    frac = np.clip(idx - np.floor(idx), 0.0, 1.0).reshape(-1, 1)
    return (1.0 - frac) * u[i0] + frac * u[i1]


def _apply_suspension(data, free_dof, p_ref, k, c) -> None:
    p = data.qvel[free_dof : free_dof + 3]
    w = data.qvel[free_dof + 3 : free_dof + 6]
    pos = data.qpos[free_dof : free_dof + 3]
    quat = data.qpos[free_dof + 3 : free_dof + 7]
    data.qfrc_applied[free_dof : free_dof + 3] = -k[:3] * (pos - p_ref) - c[:3] * p
    data.qfrc_applied[free_dof + 3 : free_dof + 6] = -k[3:6] * quat_to_rotvec(quat) - c[3:6] * w


def rollout(
    bundle: ModelBundle,
    theta: Mapping[str, np.ndarray],
    u_target: np.ndarray,
    q0: np.ndarray | None = None,
    v0: np.ndarray | None = None,
    base_pos: np.ndarray | None = None,
    base_quat: np.ndarray | None = None,
    base_w: np.ndarray | None = None,
    base_v: np.ndarray | None = None,
    susp_ref: np.ndarray | None = None,
    decimation: int = 20,
) -> RolloutResult:
    bundle.reset()
    bundle.apply_theta(theta)
    data = bundle.new_data() if hasattr(bundle, "new_data") else mujoco.MjData(bundle.model)

    q0 = DEFAULT_POS.copy() if q0 is None else np.asarray(q0, dtype=np.float64)
    init_pos = np.zeros(3) if base_pos is None else np.asarray(base_pos, dtype=np.float64)
    ref_pos = np.zeros(3) if susp_ref is None else np.asarray(susp_ref, dtype=np.float64)
    b_quat = np.array([1.0, 0.0, 0.0, 0.0]) if base_quat is None else np.asarray(base_quat, dtype=np.float64)
    data.qpos[bundle.free_qpos : bundle.free_qpos + 3] = init_pos
    data.qpos[bundle.free_qpos + 3 : bundle.free_qpos + 7] = b_quat
    data.qpos[bundle.qpos_adr] = q0
    data.qvel[:] = 0.0
    if v0 is not None:
        data.qvel[bundle.dof_adr] = np.asarray(v0, dtype=np.float64)
    if base_w is not None:
        data.qvel[bundle.free_dof + 3 : bundle.free_dof + 6] = np.asarray(base_w, dtype=np.float64)
    if base_v is not None:
        data.qvel[bundle.free_dof : bundle.free_dof + 3] = np.asarray(base_v, dtype=np.float64)
    mujoco.mj_forward(bundle.model, data)

    tau_m = np.asarray(theta.get("tau_m", np.full(NUM_JOINTS, 1e-6)), dtype=np.float64)
    d_ctrl = float(np.asarray(theta.get("d_ctrl", [0.0])).reshape(-1)[0])
    k = np.asarray(theta.get("susp_k", np.zeros(6)), dtype=np.float64)
    c = np.asarray(theta.get("susp_c", np.zeros(6)), dtype=np.float64)
    suspended = bool(np.any(k > 0) or np.any(c > 0))

    u = _delayed(np.asarray(u_target, dtype=np.float64), d_ctrl * float(decimation))
    W = u.shape[0]
    dt = bundle.timestep
    alpha = 1.0 - np.exp(-dt / np.clip(tau_m, 1e-6, None))

    q_out = np.empty((W, NUM_JOINTS))
    v_out = np.empty((W, NUM_JOINTS))
    quat_out = np.empty((W, 4))
    w_out = np.empty((W, 3))
    tau_out = np.empty((W, NUM_JOINTS))
    base_pos_out = np.empty((W, 3))
    base_lin_vel_out = np.empty((W, 3))

    u_lag = np.array(u[0], dtype=np.float64)
    for i in range(W):
        u_lag = u_lag + alpha * (u[i] - u_lag)
        data.ctrl[bundle.ctrl_ids] = u_lag
        if suspended:
            _apply_suspension(data, bundle.free_dof, ref_pos, k, c)
        mujoco.mj_step(bundle.model, data)
        q_out[i] = data.qpos[bundle.qpos_adr]
        v_out[i] = data.qvel[bundle.dof_adr]
        quat_out[i] = data.qpos[bundle.free_qpos + 3 : bundle.free_qpos + 7]
        w_out[i] = data.qvel[bundle.free_dof + 3 : bundle.free_dof + 6]
        tau_out[i] = data.actuator_force[bundle.ctrl_ids]
        base_pos_out[i] = data.qpos[bundle.free_qpos : bundle.free_qpos + 3]
        base_lin_vel_out[i] = data.qvel[bundle.free_dof : bundle.free_dof + 3]
    return RolloutResult(
        q=q_out,
        v=v_out,
        quat=quat_out,
        ang_vel=w_out,
        tau=tau_out,
        base_pos=base_pos_out,
        base_lin_vel=base_lin_vel_out,
    )


def make_eval(bundle: ModelBundle, space: ParamSpace, batch: Mapping[str, np.ndarray], decimation: int = 20):
    q_ref = np.asarray(batch["q"], dtype=np.float64)
    v_ref = np.asarray(batch["v"], dtype=np.float64)
    quat_ref = np.asarray(batch["quat"], dtype=np.float64)
    w_ref = np.asarray(batch["w"], dtype=np.float64)
    tau_ref = np.asarray(batch["tau"], dtype=np.float64)
    tau_norm = max(float(np.std(tau_ref)), 1e-6)
    n = q_ref.shape[0]
    prior = space.init_raw()
    base_pos = batch.get("base_pos", np.zeros((n, 3)))
    base_v = batch.get("base_v0", np.zeros((n, 3)))
    wu = int(batch.get("warmup", 0))

    def evaluate(raw: np.ndarray) -> float:
        theta = space.unpack(raw)
        total = 0.0
        for b in range(n):
            out = rollout(bundle, theta, batch["u"][b], q0=batch["q0"][b], v0=batch["v0"][b],
                          base_pos=base_pos[b], base_quat=batch["quat0"][b], base_w=batch["w0"][b],
                          base_v=base_v[b], decimation=decimation)
            q_sim = out.q[wu:]
            v_sim = out.v[wu:]
            quat_sim = out.quat[wu:]
            gyro_sim = out.ang_vel[wu:]
            dot = np.sum(quat_sim * quat_ref[b], axis=-1)
            total += (
                np.mean((q_sim - q_ref[b]) ** 2)
                + 0.05 * np.mean((v_sim - v_ref[b]) ** 2)
                + np.mean(1.0 - np.abs(dot))
                + 0.05 * np.mean((gyro_sim - w_ref[b]) ** 2)
            )
        total /= n
        total += 1e-3 * float(np.mean((raw - prior) ** 2))
        return float(total) if np.isfinite(total) else 1e9

    return evaluate


def fit_es(
    evaluate,
    space: ParamSpace,
    stages,
    raw_init=None,
    sigma: float = 0.15,
    population: int = 16,
    seed: int = 0,
    log_fn=None,
):
    rng = np.random.default_rng(seed)
    raw = np.asarray(space.init_raw() if raw_init is None else raw_init, dtype=np.float64)
    history = []
    best = evaluate(raw)
    for stage in stages:
        mask = np.zeros(space.dim)
        for name in stage.active:
            mask[space.slice_of(name)] = 1.0
        for it in range(stage.iterations):
            noise = rng.standard_normal((population, space.dim)) * sigma * mask.reshape(1, -1)
            scores = np.array([evaluate(raw + noise[k]) for k in range(population)])
            order = np.argsort(scores)
            elite = order[: max(2, population // 4)]
            raw = raw + noise[elite].mean(axis=0)
            if scores[order[0]] < best:
                best = scores[order[0]]
            if log_fn and it % max(1, stage.iterations // 10) == 0:
                log_fn(f"[{stage.name}] iter {it:5d}  best {best:.6e}")
        best = evaluate(raw)
        history.append({"stage": stage.name, "iter": stage.iterations, "loss": float(best)})
        if log_fn:
            log_fn(f"[{stage.name}] done     best {best:.6e}")
    return raw, history
