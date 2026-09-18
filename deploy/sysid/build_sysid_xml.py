"""Build a standalone sysid MJCF for the mjlab G1.

mjlab's G1 asset lives in the installed `mjlab` package with actuators added in
Python; `deploy/sim/sim_node.py` can override it with `TWIST2_MJLAB_G1_XML`, but
that path drops the articulation config. This script serialises the *full*
entity spec (joints + armature + position actuators) to XML and injects the
joint `frictionloss` / `damping` fields the sysid model identifies.

Run inside the mjlab uv environment:
    uv run python deploy/sysid/build_sysid_xml.py --out deploy/sysid/g1_sysid.xml
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build(out: str = "deploy/sysid/g1_sysid.xml", with_floor: bool = False) -> str:
    import mujoco
    from mjlab.asset_zoo.robots import get_g1_robot_cfg
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
    from mjlab.entity import Entity

    spec = Entity(get_g1_robot_cfg()).spec
    spec.option.timestep = 0.001
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.iterations = 50

    if with_floor:
        floor = spec.worldbody.add_geom(name="sysid_floor")
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [0.0, 0.0, 0.05]
        floor.friction = [0.6, 0.005, 0.0001]

    mesh_dir = (G1_XML.parent / "assets").resolve()
    xml_text = spec.to_xml()
    xml_text = xml_text.replace('meshdir="assets/"', f'meshdir="{mesh_dir}/"')
    xml_text = xml_text.replace('meshdir="assets"', f'meshdir="{mesh_dir}"')

    out_path = Path(out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(xml_text)
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="deploy/sysid/g1_sysid.xml")
    parser.add_argument("--with-floor", action="store_true")
    args = parser.parse_args()
    path = build(args.out, args.with_floor)
    print(f"[build] wrote {path}")


if __name__ == "__main__":
    main()
