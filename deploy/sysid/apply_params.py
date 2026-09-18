"""Write identified params back for the mjlab port.

Outputs:
  * ``<out>/g1_sysid_fitted.xml``   sysid MJCF with joint theta + effort limits
  * ``<out>/dr_patch.md``           review-only DR narrowing for src/twist2_mjlab/config.py

Use the fitted XML in sim2sim with:
    TWIST2_MJLAB_G1_XML=<out>/g1_sysid_fitted.xml ./deploy/play_sim_twist2.sh
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .constants import JOINT_NAMES, NUM_JOINTS
from .space import load_identified


def _set_field(obj, name: str, value) -> None:
    try:
        setattr(obj, name, float(value))
    except TypeError:
        getattr(obj, name)[:] = float(value)


def _set_range(obj, name: str, values) -> None:
    try:
        setattr(obj, name, list(values))
    except TypeError:
        getattr(obj, name)[:] = list(values)


def write_fitted_xml(xml_in: str, theta: dict, out_path: str) -> str:
    import mujoco

    spec = mujoco.MjSpec.from_file(str(Path(xml_in).expanduser().resolve()))
    armature = np.asarray(theta.get("armature", np.zeros(NUM_JOINTS)), dtype=float)
    friction = np.asarray(theta.get("frictionloss", np.zeros(NUM_JOINTS)), dtype=float)
    damping = np.asarray(theta.get("damping", np.zeros(NUM_JOINTS)), dtype=float)
    effort = np.asarray(theta.get("effort_limit", np.zeros(NUM_JOINTS)), dtype=float)

    for i, name in enumerate(JOINT_NAMES):
        joint = spec.joint(name)
        if joint is None:
            continue
        if "armature" in theta:
            _set_field(joint, "armature", armature[i])
        if "frictionloss" in theta:
            _set_field(joint, "frictionloss", friction[i])
        if "damping" in theta:
            _set_field(joint, "damping", damping[i])

    if "effort_limit" in theta:
        for act in spec.actuators:
            if act.name in JOINT_NAMES:
                i = JOINT_NAMES.index(act.name)
                _set_range(act, "forcerange", [-float(effort[i]), float(effort[i])])

    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(spec.to_xml())
    return str(out)


def build_dr_patch(theta: dict) -> str:
    lines = [
        "# Suggested mjlab DR changes (review only)",
        "",
        "File: `src/twist2_mjlab/config.py`",
        "",
        "The privileged critic observes these DR quantities"
        " (`src/twist2_mjlab/observations.py:233-275`), so narrowing only affects",
        "future training runs -- it does not change an already-trained policy.",
        "",
        "| identified | location | suggested change |",
        "|---|---|---|",
    ]
    if "d_ctrl" in theta:
        d = float(np.asarray(theta["d_ctrl"]).reshape(-1)[0])
        lines.append(
            f"| d_ctrl = {d:.2f} control steps | `_apply_twist2_domain_rand` "
            f"`delay_min_lag`/`delay_max_lag` (config.py ~277-294) | centre on `{int(round(d))}` |"
        )
    if "effort_limit" in theta:
        ratio = np.asarray(theta["effort_limit"], dtype=float) / 88.0
        lo = float(np.clip(np.median(ratio) * 0.9, 0.8, 1.2))
        hi = float(np.clip(np.median(ratio) * 1.1, 0.8, 1.2))
        lines.append(f"| effort ratio {np.median(ratio):.3f} | `_TWIST2_MOTOR_STRENGTH_RANGE` (config.py ~39) | `[{lo:.2f}, {hi:.2f}]` |")
    if "payload_mass" in theta:
        m = float(np.asarray(theta["payload_mass"]).reshape(-1)[0])
        lines.append(f"| payload {m:+.2f} kg | `_TWIST2_BASE_MASS_RANGE` (config.py ~38) | centre on `{m:+.2f}` |")
    lines.append("| foot friction | upstream `foot_friction` event | **not identified in v1 -> leave unchanged** |")
    lines.append("| joint frictionloss/damping | no DR field | written into the sysid XML only |")
    lines.append("")
    lines.append("No file is edited automatically.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--identified", required=True)
    parser.add_argument("--xml", default="deploy/sysid/g1_sysid.xml")
    parser.add_argument("--out", default="deploy/sysid/generated")
    args = parser.parse_args()

    data = load_identified(args.identified)
    theta = data.get("theta", {})
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    xml_out = write_fitted_xml(args.xml, theta, out_dir / "g1_sysid_fitted.xml")
    print(f"[apply] fitted sysid MJCF: {xml_out}")
    patch = out_dir / "dr_patch.md"
    patch.write_text(build_dr_patch(theta))
    print(f"[apply] DR patch:          {patch}")
    print(f"[apply] sim2sim with: TWIST2_MJLAB_G1_XML={xml_out}")


if __name__ == "__main__":
    main()
