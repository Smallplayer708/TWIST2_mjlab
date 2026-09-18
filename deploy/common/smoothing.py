"""Shared runtime smoothing filters and tuning validation for the TWIST2 deploy nodes.

The original TWIST2 deploy scripts use several first-order EMA filters with
*different* conventions, plus a sliding-window mean.  To keep parity we expose
each convention explicitly instead of hiding it behind a single ``alpha``
meaning (see ``.kilo/plans/deploy-tuning-params.md`` section 2):

  leg_ema_alpha      : t = a * prev + (1 - a) * new      (a = retain previous)
  leg_smooth_alpha   : s = a * raw  + (1 - a) * prev_s   (a = weight on new)
  arm_smooth_alpha   : s = a * raw  + (1 - a) * prev_s   (a = weight on new)
  smooth_body        : s = a * new  + (1 - a) * prev_s   (a = weight on new)
  smooth (window)    : arithmetic mean over the last N frames

All filters are pure numpy and default to disabled, so importing them never
changes existing behaviour.  The shared argparse wiring and validators keep the
sim, hardware, and policy nodes in sync and reject out-of-range tuning values.
"""

from __future__ import annotations

import argparse
from collections import deque

import numpy as np

MIMIC_DIM = 35

# 35D mimic layout: [0:6] root (vx, vy, z, roll, pitch, yaw_vel), [6:35] joints.
# Leg joints are joints 0..11 -> mimic[6:18]; arms are joints 15..28 -> [21:35].
_LEG_MASK = np.zeros(MIMIC_DIM, dtype=bool)
_LEG_MASK[0:6] = True    # root section
_LEG_MASK[6:18] = True   # 12 leg joints
_ARM_SLICE = slice(21, 35)
_ARM_DIM = 14
_LEG_DIM = int(_LEG_MASK.sum())  # 18


class _BaseEMA:
    def __init__(self, alpha: float, size: int, initial: np.ndarray | None):
        self.alpha = float(alpha)
        if initial is not None:
            self._prev = np.asarray(initial, dtype=np.float32).copy()
            self._initialized = True
        else:
            self._prev = np.zeros(size, dtype=np.float32)
            self._initialized = False

    @property
    def enabled(self) -> bool:
        return self.alpha > 0.0

    def reset(self) -> None:
        self._prev = np.zeros_like(self._prev)
        self._initialized = False


class EMAFilter(_BaseEMA):
    """EMA with the ``s = a * new + (1 - a) * prev`` convention."""

    def __init__(self, alpha: float, size: int, initial: np.ndarray | None = None):
        super().__init__(alpha, size, initial)

    def apply(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return x
        x = np.asarray(x, dtype=np.float32)
        if not self._initialized:
            self._prev = x.copy()
            self._initialized = True
            return self._prev
        a = self.alpha
        self._prev = (a * x + (1.0 - a) * self._prev).astype(np.float32)
        return self._prev


class EMARetainFilter(_BaseEMA):
    """EMA where the previous value is retained: ``t = a * prev + (1 - a) * new``.

    This matches the original ``leg_ema_alpha`` PD-target low-pass, where a
    larger alpha means *more* smoothing.
    """

    def __init__(self, alpha: float, size: int, initial: np.ndarray | None = None):
        super().__init__(alpha, size, initial)

    def apply(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return x
        x = np.asarray(x, dtype=np.float32)
        if not self._initialized:
            self._prev = x.copy()
            self._initialized = True
            return self._prev
        a = self.alpha
        self._prev = (a * self._prev + (1.0 - a) * x).astype(np.float32)
        return self._prev


class SlidingWindowMean:
    """Arithmetic mean over the most recent ``window_size`` samples."""

    def __init__(self, window_size: int = 1):
        self.window_size = int(window_size)
        self._buf: deque[np.ndarray] = deque(maxlen=max(1, self.window_size))

    @property
    def enabled(self) -> bool:
        return self.window_size > 1

    def reset(self) -> None:
        self._buf.clear()

    def apply(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return x
        obs = np.asarray(x, dtype=np.float32).copy()
        self._buf.append(obs)
        if len(self._buf) < 2:
            return obs
        return np.mean(np.stack(self._buf, axis=0), axis=0).astype(np.float32)


class MimicSmoother:
    """35D mimic smoothing pipeline matching the original TWIST2 ordering.

    Order: arm EMA -> leg EMA -> sliding window -> body EMA.  This mirrors the
    original teleop section (arm/leg/window) followed by the low-level server
    body smoothing, and is applied to the mimic observation *before* the policy.
    """

    def __init__(
        self,
        arm_alpha: float = 0.0,
        leg_alpha: float = 0.0,
        body_alpha: float = 0.0,
        window_size: int = 1,
    ):
        self._arm = EMAFilter(arm_alpha, _ARM_DIM)
        self._leg = EMAFilter(leg_alpha, _LEG_DIM)
        self._body = EMAFilter(body_alpha, MIMIC_DIM)
        self._window = SlidingWindowMean(window_size)

    @property
    def enabled(self) -> bool:
        return (self._arm.enabled or self._leg.enabled
                or self._body.enabled or self._window.enabled)

    def reset(self) -> None:
        self._arm.reset()
        self._leg.reset()
        self._body.reset()
        self._window.reset()

    def apply(self, mimic: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return mimic
        m = np.asarray(mimic, dtype=np.float32).copy()
        if self._arm.enabled:
            m[_ARM_SLICE] = self._arm.apply(m[_ARM_SLICE])
        if self._leg.enabled:
            m[_LEG_MASK] = self._leg.apply(m[_LEG_MASK])
        m = self._window.apply(m)
        m = self._body.apply(m)
        return m


# ---------------------------------------------------------------------------
# Shared tuning validation / argparse wiring
# ---------------------------------------------------------------------------
MAX_PD_GAIN = 2.0


def validate_alpha(name: str, value: float) -> float:
    """Validate a smoothing alpha is within the stable range [0, 1]."""
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def validate_pd_gain(name: str, value: float) -> float:
    """Validate a PD-gain scale factor is positive and bounded."""
    value = float(value)
    if not 0.0 < value <= MAX_PD_GAIN:
        raise ValueError(f"{name} must be in (0, {MAX_PD_GAIN}], got {value}")
    return value


def add_smoothing_args(parser: argparse.ArgumentParser) -> None:
    """Register the shared 35D mimic smoothing options on ``parser``."""
    parser.add_argument("--leg_smooth_alpha", type=float, default=0.0,
                        help="EMA on mimic root+leg section [0:6]+[6:18] (0=off, ~0.8)")
    parser.add_argument("--arm_smooth_alpha", type=float, default=0.0,
                        help="EMA on mimic arm joints [21:35] (0=off)")
    parser.add_argument("--smooth_body", type=float, default=0.0,
                        help="EMA on the full 35D mimic before the policy (0=off)")
    parser.add_argument("--smooth_window_size", type=int, default=1,
                        help="Sliding-window mean length on the 35D mimic (1=off)")


def build_smoother(args) -> MimicSmoother:
    """Build a validated ``MimicSmoother`` from parsed smoothing arguments."""
    window_size = int(args.smooth_window_size)
    if window_size < 1:
        raise ValueError(f"--smooth_window_size must be >= 1, got {window_size}")
    return MimicSmoother(
        arm_alpha=validate_alpha("--arm_smooth_alpha", args.arm_smooth_alpha),
        leg_alpha=validate_alpha("--leg_smooth_alpha", args.leg_smooth_alpha),
        body_alpha=validate_alpha("--smooth_body", args.smooth_body),
        window_size=window_size,
    )
