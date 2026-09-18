"""Bounded parameter space for the mjlab sysid port.

kp/kd and armature are already physics-derived in mjlab, so kp/kd are fixed and
armature is only lightly refined. Identified groups: armature, frictionloss,
damping, effort_limit (actuator forcerange), motor lag, action delay, and the
suspension / soft-base stiffness and damping.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

from .constants import NUM_JOINTS


def _sigmoid(x):
    return 0.5 * (np.tanh(0.5 * x) + 1.0)


def _logit(p):
    return 2.0 * np.arctanh(2.0 * np.clip(p, 1e-6, 1 - 1e-6) - 1.0)


@dataclass(frozen=True)
class Group:
    name: str
    size: int
    lo: float
    hi: float
    init: float


class ParamSpace:
    def __init__(self, groups: Sequence[Group]) -> None:
        self.groups: List[Group] = list(groups)
        self.names = [g.name for g in self.groups]
        sizes = [g.size for g in self.groups]
        self._offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
        self.lo = np.concatenate([np.full(g.size, g.lo) for g in self.groups]).astype(np.float64)
        self.hi = np.concatenate([np.full(g.size, g.hi) for g in self.groups]).astype(np.float64)
        self.init = np.concatenate([np.full(g.size, g.init) for g in self.groups]).astype(np.float64)
        self.dim = int(self.lo.size)

    def slice_of(self, name: str) -> slice:
        i = self.names.index(name)
        return slice(int(self._offsets[i]), int(self._offsets[i + 1]))

    def unpack(self, raw: np.ndarray) -> Dict[str, np.ndarray]:
        raw = np.asarray(raw, dtype=np.float64).reshape(-1)
        x = self.lo + (self.hi - self.lo) * _sigmoid(raw)
        return {name: x[self.slice_of(name)] for name in self.names}

    def pack(self, values: Mapping[str, np.ndarray]) -> np.ndarray:
        flat = np.concatenate([np.asarray(values[g.name], dtype=np.float64).reshape(-1) for g in self.groups])
        return _logit((flat - self.lo) / (self.hi - self.lo))

    def init_raw(self) -> np.ndarray:
        return self.pack({g.name: np.full(g.size, g.init) for g in self.groups})

    def init_raw_with(self, values: Mapping[str, np.ndarray]) -> np.ndarray:
        theta = {g.name: np.full(g.size, g.init) for g in self.groups}
        for name, val in values.items():
            if name in theta:
                theta[name] = np.asarray(val, dtype=np.float64).reshape(theta[name].shape)
        return self.pack(theta)

    def as_dict(self):
        return {g.name: asdict(g) for g in self.groups}

    def save(self, path):
        payload = self.as_dict()
        payload["__order__"] = list(self.names)
        Path(path).write_text(json.dumps(payload, indent=2))


def susp_space() -> ParamSpace:
    return ParamSpace(
        [
            Group("susp_k", 6, 1.0, 2000.0, 500.0),
            Group("susp_c", 6, 0.1, 300.0, 60.0),
            Group("d_ctrl", 1, 0.0, 5.0, 1.0),
        ]
    )


def joint_space() -> ParamSpace:
    return ParamSpace(
        [
            Group("armature", NUM_JOINTS, 1e-4, 0.10, 0.02),
            Group("frictionloss", NUM_JOINTS, 0.0, 5.0, 0.10),
            Group("damping", NUM_JOINTS, 0.0, 10.0, 0.50),
            Group("effort_limit", NUM_JOINTS, 5.0, 200.0, 88.0),
            Group("tau_m", NUM_JOINTS, 5e-4, 0.05, 0.005),
            Group("d_ctrl", 1, 0.0, 5.0, 1.0),
            Group("susp_k", 6, 1.0, 2000.0, 500.0),
            Group("susp_c", 6, 0.1, 300.0, 60.0),
        ]
    )


def softbase_space() -> ParamSpace:
    return ParamSpace(
        [
            Group("susp_k", 6, 1.0, 5000.0, 1500.0),
            Group("susp_c", 6, 0.1, 500.0, 120.0),
            Group("d_ctrl", 1, 0.0, 5.0, 1.0),
            Group("payload_mass", 1, -3.0, 3.0, 0.0),
            Group("payload_com", 3, -0.05, 0.05, 0.0),
        ]
    )


def save_identified(path, space: ParamSpace, raw: np.ndarray, meta: Mapping | None = None) -> None:
    theta = space.unpack(raw)
    payload = {
        "meta": dict(meta or {}),
        "space": space.as_dict(),
        "space_order": list(space.names),
        "theta": {k: np.asarray(v).tolist() for k, v in theta.items()},
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_identified(path) -> dict:
    return json.loads(Path(path).read_text())
