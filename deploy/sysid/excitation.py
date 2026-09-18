"""Excitation signals + safety limiting for the mjlab sysid port."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .constants import DEFAULT_POS, G1_JOINT_RANGES, KP, NUM_JOINTS

KINDS: tuple[str, ...] = ("multisine", "chirp", "prbs", "steps", "sweep", "quasi_static")


@dataclass
class ExcitationSpec:
    joint_index: int
    kind: str = "multisine"
    duration: float = 20.0
    dt: float = 0.001
    amplitude: float = 0.05
    f_lo: float = 0.5
    f_hi: float = 10.0
    n_tones: int = 12
    f_bit: float = 2.0
    period: float = 4.0
    speeds: Sequence[float] = field(default_factory=lambda: (0.05, 0.2, 0.5))
    ramp_time: float = 1.0
    seed: int = 0


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _cosine_ramp(n: int, dt: float, ramp_time: float) -> np.ndarray:
    ramp_n = max(1, int(round(ramp_time / dt)))
    ramp_n = min(ramp_n, n // 2 if n >= 2 else 1)
    w = np.ones(n, dtype=np.float64)
    if ramp_n <= 0:
        return w
    t = (np.arange(ramp_n) + 0.5) / ramp_n
    half = 0.5 * (1.0 - np.cos(np.pi * t))
    w[:ramp_n] = half
    w[-ramp_n:] = half[::-1]
    return w


def schroeder_multisine(n, dt, f_lo, f_hi, n_tones, amplitude, seed=0):
    freqs = np.linspace(float(f_lo), float(f_hi), int(n_tones))
    time = np.arange(n, dtype=np.float64) * dt
    sig = np.zeros(n, dtype=np.float64)
    for k, f in enumerate(freqs):
        sig += np.sin(2.0 * np.pi * f * time - np.pi * k * (k + 1) / max(1, n_tones))
    peak = np.max(np.abs(sig))
    return sig * (amplitude / peak) if peak > 0 else sig


def chirp(n, dt, f0, f1, amplitude):
    time = np.arange(n, dtype=np.float64) * dt
    duration = max(dt, (n - 1) * dt)
    rate = (f1 - f0) / duration
    sig = np.sin(2.0 * np.pi * (f0 * time + 0.5 * rate * time**2))
    peak = np.max(np.abs(sig))
    return sig * (amplitude / peak) if peak > 0 else sig


def prbs(n, dt, f_bit, amplitude, seed=0):
    rng = _rng(seed)
    hold = max(1, int(round(1.0 / (f_bit * dt))))
    bits = rng.choice(np.array([-1.0, 1.0]), size=int(np.ceil(n / hold)) + 1)
    return np.repeat(bits, hold)[:n] * amplitude


def step_train(n, dt, amplitude, period, seed=0):
    rng = _rng(seed)
    n_period = max(2, int(round(period / dt)))
    cycle = np.concatenate([np.full(n_period // 2, amplitude), np.full(n_period - n_period // 2, -amplitude)])
    return np.resize(cycle, n) * rng.choice(np.array([-1.0, 1.0]))


def constant_velocity_sweep(n, dt, speeds, amplitude):
    speed_cycle = [0.0] + list(speeds) + [0.0] + [-s for s in reversed(speeds)] + [0.0]
    seg_len = max(1, n // max(1, len(speed_cycle)))
    pos, idx = 0.0, 0
    out = np.empty(n, dtype=np.float64)
    for speed in speed_cycle:
        for _ in range(seg_len):
            if idx >= n:
                break
            pos += speed * dt
            out[idx] = pos
            idx += 1
    while idx < n:
        out[idx] = out[idx - 1] if idx else 0.0
        idx += 1
    peak = np.max(np.abs(out))
    return out * min(1.0, amplitude / peak) if peak > 0 else out


def quasi_static(n, dt, amplitude):
    time = np.arange(n, dtype=np.float64) * dt
    sig = np.sin(2.0 * np.pi * 0.05 * time)
    peak = np.max(np.abs(sig))
    return sig * (amplitude / peak) if peak > 0 else sig


_SIGNAL_FNS: dict[str, Callable] = {
    "multisine": lambda n, dt, s: schroeder_multisine(n, dt, s.f_lo, s.f_hi, s.n_tones, s.amplitude, s.seed),
    "chirp": lambda n, dt, s: chirp(n, dt, s.f_lo, s.f_hi, s.amplitude),
    "prbs": lambda n, dt, s: prbs(n, dt, s.f_bit, s.amplitude, s.seed),
    "steps": lambda n, dt, s: step_train(n, dt, s.amplitude, s.period, s.seed),
    "sweep": lambda n, dt, s: constant_velocity_sweep(n, dt, s.speeds, s.amplitude),
    "quasi_static": lambda n, dt, s: quasi_static(n, dt, s.amplitude),
}


def generate_signal(spec: ExcitationSpec) -> np.ndarray:
    if spec.kind not in _SIGNAL_FNS:
        raise ValueError(f"unknown excitation kind {spec.kind!r}; choose from {KINDS}")
    n = max(2, int(round(spec.duration / spec.dt)))
    return _SIGNAL_FNS[spec.kind](n, spec.dt, spec) * _cosine_ramp(n, spec.dt, spec.ramp_time)


def clamp_to_ranges(targets, margin=0.05, ranges=None):
    rng = G1_JOINT_RANGES if ranges is None else np.asarray(ranges, dtype=np.float64)
    lo = rng[:, 0] + margin
    hi = rng[:, 1] - margin
    return np.clip(targets, lo.reshape(1, -1), hi.reshape(1, -1))


def build_targets(spec: ExcitationSpec, hold_pose=None, safety_margin=0.05) -> np.ndarray:
    hold = DEFAULT_POS if hold_pose is None else np.asarray(hold_pose, dtype=np.float64)
    sig = generate_signal(spec)
    targets = np.tile(hold.reshape(1, -1), (sig.size, 1))
    targets[:, int(spec.joint_index)] = hold[int(spec.joint_index)] + sig
    return clamp_to_ranges(targets, margin=safety_margin)


def rate_limit(u, dt, max_rate, u0=None):
    u = np.asarray(u, dtype=np.float64)
    out = np.empty_like(u)
    prev = u[0].copy() if u0 is None else np.asarray(u0, dtype=np.float64).copy()
    step = max_rate * dt
    for i in range(u.shape[0]):
        prev = prev + np.clip(u[i] - prev, -step, step)
        out[i] = prev
    return out


def estimate_peak_torque(targets, hold_pose=None, kp=None) -> np.ndarray:
    hold = DEFAULT_POS if hold_pose is None else np.asarray(hold_pose, dtype=np.float64)
    gains = KP if kp is None else np.asarray(kp, dtype=np.float64)
    return np.max(np.abs(targets - hold.reshape(1, -1)), axis=0) * gains


class SafetyLimiter:
    def __init__(self, max_rate=8.0, margin=0.05, temp_limit=75.0, voltage_min=40.0, ranges=None):
        self.max_rate = float(max_rate)
        self.margin = float(margin)
        self.temp_limit = float(temp_limit)
        self.voltage_min = float(voltage_min)
        self.ranges = G1_JOINT_RANGES if ranges is None else np.asarray(ranges, dtype=np.float64)
        self._prev = None

    def reset(self, u0):
        self._prev = np.asarray(u0, dtype=np.float64).copy()

    def __call__(self, target, dt, temperature=None, voltage=None):
        u = clamp_to_ranges(np.asarray(target, dtype=np.float64), self.margin, self.ranges)
        if self._prev is None:
            self._prev = u.copy()
        u = self._prev + np.clip(u - self._prev, -self.max_rate * dt, self.max_rate * dt)
        self._prev = u.copy()
        return u

    def watchdog(self, temperature=None, voltage=None):
        if temperature is not None and np.max(np.asarray(temperature)) > self.temp_limit:
            return "temperature"
        if voltage is not None and np.min(np.asarray(voltage)) < self.voltage_min:
            return "voltage"
        return None
