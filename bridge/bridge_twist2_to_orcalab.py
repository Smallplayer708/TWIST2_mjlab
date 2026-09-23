"""
TWIST2 → OrcaLab RL Policy Bridge

Connects TWIST2 arm-only teleoperation (mimic_obs from Redis) to OrcaLab's
MuJoCo simulation running the TWIST2 ONNX policy. Uses position actuators
directly (pd_target = raw_action * 0.5 + default_dof_pos), adapting from
TWIST2's torque PD control.

Hands: grip=张开 / index_trig=闭合 (teleop key interpolation semantics),
consumed from action_hand_left/right Redis keys, per-joint remapped from
TWIST2 limits to OrcaLab limits, written to the 14 hand position actuators.

Two modes:
  gRPC mode (default): connects to OrcaStudio for XML + assets + rendering
  Local mode (--local_xml): uses pre-cached XML, no OrcaStudio needed

B1 twin mode (--replay_target): instead of running the ONNX policy locally,
replay the REAL robot's policy target (29 absolute joint positions) that
sim2real publishes to Redis. Both plants then receive the SAME joint command,
so sim-vs-real trajectory differences isolate the plant/contact gap. Requires
the real side to run with --publish_target (see setup notes) and, for a stable
scene, --fix_feet.

Architecture:
  (policy mode)  Redis mimic_obs(35) + OrcaLab proprio(92) → obs_buf(1524) → ONNX(29) → PD → position actuators
  (twin mode)    Redis real target(29) → position actuators

Usage:
  conda activate orca
  cd OrcaManipulation/src/examples/dataCollection

  # Local XML mode:
  LATEST_XML=$(ls -t ~/.orcagym/tmp/*.xml | head -1)
  python bridge_twist2_to_orcalab.py \
      --policy /path/to/twist2_1017_20k.onnx \
      --local_xml "$LATEST_XML" \
      --no_render --fix_feet --device cpu

  # gRPC mode (OrcaStudio must be running):
  python bridge_twist2_to_orcalab.py \
      --policy /path/to/twist2_1017_20k.onnx \
      --fix_feet --device cpu

  # Headless self-test (synthetic sine-arm mimic, no Redis, 5s, Ctrl+C-safe):
  python bridge_twist2_to_orcalab.py \
      --policy /path/to/twist2_1017_20k.onnx \
      --local_xml "$LATEST_XML" \
      --no_render --fix_feet --device cpu --selftest

  # B1 twin (replay the real robot's policy target; sim2real runs with
  # --publish_target). No policy needed:
  python bridge_twist2_to_orcalab.py \
      --replay_target --fix_feet --device cpu
  # headless twin self-test:
  python bridge_twist2_to_orcalab.py \
      --replay_target --local_xml "$LATEST_XML" \
      --no_render --fix_feet --device cpu --selftest
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
import traceback
from collections import deque

import numpy as np
import redis
import torch

from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv

try:
    import onnxruntime as ort
except ImportError:
    ort = None


# =============================================================================
# DataCollectionEnv (inlined from OrcaManipulation envs/dataCollection so the
# bridge only depends on orca-gym + OrcaLab)
# =============================================================================
class DataCollectionEnv(OrcaGymLocalEnv):
    def __init__(
        self,
        frame_skip: int,
        orcagym_addr: str,
        agent_names: list,
        time_step: float,
        default_joint_values: dict,
        obs_callback,
        **kwargs
    ):
        self.obs_callback = obs_callback
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs)

        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv

        self.ctrl = np.zeros(self.nu, dtype=np.float32)
        self._set_obs_space()
        self._set_action_space()

        self.default_joint_values = None
        self.set_default_joint_values(default_joint_values)

    def step(self, action):
        self.ctrl = action
        self.do_simulation(self.ctrl, self.frame_skip)
        obs = self._get_obs().copy()
        terminated = False
        truncated = False
        reward = 0.0
        return obs, reward, terminated, truncated, {}

    def reset_model(self):
        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv

        self.set_default_joint_values(self.default_joint_values)
        self.mj_forward()
        self._render_time_step = 0
        obs = self._get_obs().copy()
        return obs, {}

    def init_env(self):
        self.model, self.data = self.initialize_simulation()
        self.reset()

    def set_default_joint_values(self, default_joint_values: dict):
        self.default_joint_values = default_joint_values
        self._default_joint_qpos = {
            self.joint(joint_name): np.float32(value)
            for joint_name, value in default_joint_values.items()
        }
        self.set_joint_qpos(self._default_joint_qpos)

    def _set_obs_space(self):
        self.observation_space = self.generate_observation_space(self._get_obs().copy())

    def _set_action_space(self):
        low_bounds = -np.ones(self.nu, dtype=np.float32)
        high_bounds = np.ones(self.nu, dtype=np.float32)
        bound = np.array([[lo, hi] for lo, hi in zip(low_bounds, high_bounds)])
        self.action_space = self.generate_action_space(bound)

    def _get_obs(self):
        return self.obs_callback(self)


# =============================================================================
# quatToEuler (inlined from TWIST2 deploy_real/data_utils/rot_utils.py)
# =============================================================================
def quatToEuler(quat):
    """Convert quaternion (w,x,y,z) to Euler angles (roll, pitch, yaw)."""
    qw, qx, qy, qz = quat
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])


# =============================================================================
# ONNX Policy Wrapper (mirrors TWIST2 server_low_level_g1_sim.py:22-39)
# =============================================================================
class OnnxPolicyWrapper:
    def __init__(self, session, input_name):
        self.session = session
        self.input_name = input_name

    def __call__(self, obs_tensor):
        if isinstance(obs_tensor, torch.Tensor):
            obs_np = obs_tensor.detach().cpu().numpy()
        else:
            obs_np = np.asarray(obs_tensor, dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: obs_np})
        result = outputs[0]
        if not isinstance(result, np.ndarray):
            result = np.asarray(result, dtype=np.float32)
        return torch.from_numpy(result.astype(np.float32))


def load_onnx_policy(policy_path, device="cuda"):
    if ort is None:
        raise ImportError("onnxruntime is not installed.")
    providers = []
    available = ort.get_available_providers()
    if device.startswith("cuda") and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    else:
        if device.startswith("cuda"):
            print("CUDAExecutionProvider not available; falling back to CPU.")
    providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(policy_path, providers=providers)
    input_name = session.get_inputs()[0].name
    print(f"ONNX policy loaded from {policy_path} (providers: {session.get_providers()})")
    return OnnxPolicyWrapper(session, input_name)


# =============================================================================
# TWIST2 Constants (mirrors server_low_level_g1_sim.py:122-128)
# =============================================================================
DEFAULT_DOF_POS = np.array([
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,     # left leg (6)
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,     # right leg (6)
     0.0, 0.0, 0.0,                       # torso (3)
     0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0,  # left arm (7)
     0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0, # right arm (7)
], dtype=np.float32)

ANKLE_IDX = [4, 5, 10, 11]
ACTION_SCALE = 0.5

# TWIST2 训练/部署 PD 增益（stiffness/damping，与 DEFAULT_DOF_POS 同序：
# 腿6+腿6+腰3+左臂7+右臂7，来源 deploy_real/server_low_level_g1_sim.py）。
# OrcaLab XML position 执行器增益（肩肘 kp≈14、腕 kp≈14-17）与训练值
# （肩肘 kp=40/kv=5、腕 kp=4.0/kv=0.2）不匹配 → 双臂到位后极限环振荡。
TWIST2_STIFFNESS = np.array([
    100, 100, 100, 150, 40, 40,
    100, 100, 100, 150, 40, 40,
    150, 150, 150,
    40, 40, 40, 40, 4.0, 4.0, 4.0,
    40, 40, 40, 40, 4.0, 4.0, 4.0,
], dtype=np.float32)
TWIST2_DAMPING = np.array([
    2, 2, 2, 4, 2, 2,
    2, 2, 2, 4, 2, 2,
    4, 4, 4,
    5, 5, 5, 5, 0.2, 0.2, 0.2,
    5, 5, 5, 5, 0.2, 0.2, 0.2,
], dtype=np.float32)

# Real-robot downlink PD gains (deploy_real/robot_control/configs/g1.yaml kps/kds),
# same joint order. The real controller pushes kp/kd to the motor's internal loop
# while OrcaLab simulates a <position> servo, so replaying the real target with
# the REAL gains (not the training gains) is what makes the twin comparison fair.
REAL_STIFFNESS = np.array([
    100, 100, 100, 150, 40, 40,
    100, 100, 100, 150, 40, 40,
    150, 150, 150,
    40, 40, 40, 40, 20, 20, 20,
    40, 40, 40, 40, 20, 20, 20,
], dtype=np.float32)
REAL_DAMPING = np.array([
    2, 2, 2, 4, 2, 2,
    2, 2, 2, 4, 2, 2,
    4, 4, 4,
    5, 5, 5, 5, 1, 1, 1,
    5, 5, 5, 5, 1, 1, 1,
], dtype=np.float32)

# Default 35-dim mimic obs (mirrors TWIST2 data_utils/params.py DEFAULT_MIMIC_OBS_G1):
#   [0:2] xy vel, [2] z pos, [3:5] roll/pitch, [5] yaw ang vel, [6:35] = 29 dof
# Arms live at mimic idx 21-34 (left 21-27, right 28-34).
DEFAULT_MIMIC_OBS = np.concatenate([
    np.array([0.0, 0.0, 0.8, 0.0, 0.0, 0.0], dtype=np.float32),
    DEFAULT_DOF_POS,
]).astype(np.float32)
HISTORY_LEN = 11        # mjlab actor_history keeps 11 frames *including* current
N_MIMIC_OBS = 35
N_OBS_SINGLE = 127      # 35 (mimic) + 92 (proprio)
N_PROPRIO = 92           # 3 + 2 + 29 + 29 + 29 = 92
# mjlab actor obs = current(127) + 11-frame history(127 each) = 1524, and has no
# separate future-mimic block (the old TWIST2 student_future policy did).
TOTAL_OBS_SIZE = N_OBS_SINGLE * (HISTORY_LEN + 1)  # 1524

# TWIST2 29-joint names in order, matching DEFAULT_DOF_POS indices.
# OrcaLab XML actuator/joint names are g1_pick_{name}.
TWIST2_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# ── Hand joints (14 = 7 left + 7 right), order matches TWIST2 XML and
#    OrcaLab conf (g1_pick_conf.py l_hand/r_hand): thumb_0..2, middle_0/1, index_0/1 ──
HAND_JOINT_NAMES = [
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
]

# TWIST2-side hand poses (mirrors deploy_real/data_utils/params.py
# DEFAULT_HAND_POSE["unitree_g1_with_hands"]): open = all zero, close per side.
HAND_OPEN_TWIST2 = np.zeros(14, dtype=np.float32)
HAND_CLOSE_TWIST2 = np.concatenate([
    np.array([0.0, 1.0, 1.74, -1.57, -1.74, -1.57, -1.74], dtype=np.float32),   # left close
    np.array([0.0, -1.0, -1.74, 1.57, 1.74, 1.57, 1.74], dtype=np.float32),     # right close
]).astype(np.float32)


def _load_orca_hand_poses():
    """Load OrcaLab-side hand open/close poses (14 each) from conf/g1_pick_conf.py.

    Open = positions_init_ctrl (all zero). Close uses the same rule as the
    reference script g1_pick_collection_tele_lerobot.py HandController:
    np.where(|r1| > |r0|, r1, r0) over positions_ranges. Falls back to
    hardcoded values (source: conf/g1_pick_conf.py positions_ranges) if the
    module cannot be imported.
    """
    try:
        from conf.g1_pick_conf import l_hand, r_hand
    except ImportError:
        print("[Bridge] WARNING: conf.g1_pick_conf import failed — using hardcoded OrcaLab hand poses")
        l_hand = r_hand = None

    def _compute(hand_conf, fallback_close):
        open_pose = np.zeros(7, dtype=np.float32)
        if hand_conf is None:
            return open_pose, np.array(fallback_close, dtype=np.float32)
        ranges = np.array(hand_conf["positions_ranges"], dtype=np.float32)
        close_pose = np.where(
            np.abs(ranges[:, 1]) > np.abs(ranges[:, 0]),
            ranges[:, 1], ranges[:, 0],
        ).astype(np.float32)
        return open_pose, close_pose

    # Hardcoded fallbacks derived from g1_pick_conf.py positions_ranges
    # (thumb_2 rows corrected to match the XML assets: L [0, +1.745], R [-1.745, 0]):
    l_open, l_close = _compute(l_hand, [-1.047, 1.047, 1.745, -1.571, -1.745, -1.571, -1.745])
    r_open, r_close = _compute(r_hand, [-1.047, -1.047, -1.745, 1.571, 1.745, 1.571, 1.745])
    return (np.concatenate([l_open, r_open]).astype(np.float32),
            np.concatenate([l_close, r_close]).astype(np.float32))


# =============================================================================
# XML patching: weld pelvis for arm-only simulation (local mode)
# =============================================================================
def _patch_xml_with_feet_weld(xml_path, inplace=False):
    """Inject pelvis weld into XML. Returns path to patched XML.

    inplace=True: writes welds directly into xml_path (backs up to .bak first).
    inplace=False: writes to a temp file next to the original.
    """
    xml_dir = os.path.dirname(os.path.abspath(xml_path))
    with open(xml_path, "r") as f:
        xml = f.read()

    import re

    # Dynamically find the pelvis body and weld it to world.
    # Pelvis-only (verified: welding feet doesn't cause launch, but pelvis-only
    # is simplest; root is additionally hard-locked every timestep anyway).
    # Body names follow the pattern: {agent}_pelvis, etc.
    body_suffixes = [
        ("pelvis", "pelvis_weld"),
    ]

    # Find body names in XML that match these suffixes
    body_matches = {}
    for suffix, weld_name in body_suffixes:
        pattern = rf'name="([^"]*{suffix})"'
        for match in re.finditer(pattern, xml):
            body_name = match.group(1)
            # Only match robot bodies (not scene objects)
            if any(kw in body_name.lower() for kw in ["pelvis", "torso", "ankle", "hip", "waist", "knee", "elbow", "shoulder", "wrist", "hand"]):
                if weld_name not in body_matches:
                    body_matches[weld_name] = body_name
                    break

    # Build weld block
    weld_lines = []
    for weld_name, body_name in body_matches.items():
        weld_lines.append(
            f'        <weld active="true" name="{weld_name}" '
            f'body1="{body_name}" body2="world" '
            f'solref="0.02 1" solimp="0.9 0.95 0.001"/>'
        )

    if not weld_lines:
        print("[Fix Feet] WARNING: Could not find pelvis/torso/foot bodies — no welds injected")
        return xml_path

    welds_xml = '\n'.join(weld_lines)
    # Insert welds into <equality> block, or create one after </actuator>
    import re as _re
    eq_match = _re.search(r'<equality>(.*?)</equality>', xml, _re.DOTALL)
    if eq_match:
        insert_pos = eq_match.end() - len('</equality>')
        xml = xml[:insert_pos] + welds_xml + '\n' + xml[insert_pos:]
    else:
        act_match = _re.search(r'</actuator>', xml)
        if act_match:
            insert_pos = act_match.end()
            xml = (xml[:insert_pos] +
                   '\n    <equality>\n' + welds_xml + '\n    </equality>' +
                   xml[insert_pos:])

    welded_bodies = ", ".join(body_matches.values())
    print(f"[Fix Feet] Injected welds for: {welded_bodies}")

    if inplace:
        bak_path = xml_path + ".bak"
        if not os.path.exists(bak_path):
            os.rename(xml_path, bak_path)
        with open(xml_path, "w") as f:
            f.write(xml)
        print(f"[Fix Feet] Patched XML in place: {xml_path} (backup: {bak_path})")
        return xml_path

    fd, tmp_path = tempfile.mkstemp(suffix=".xml", prefix="bridge_", dir=xml_dir)
    with os.fdopen(fd, "w") as f:
        f.write(xml)
    return tmp_path


def _inject_xml_weld_text(xml_text):
    """Inject pelvis weld into XML text, using dynamic body names."""
    import re

    body_suffixes = [
        ("pelvis", "pelvis_weld"),
    ]

    body_matches = {}
    for suffix, weld_name in body_suffixes:
        # Idempotency: skip if this weld already exists in the XML
        if f'name="{weld_name}"' in xml_text:
            continue
        pattern = rf'name="([^"]*{suffix})"'
        for match in re.finditer(pattern, xml_text):
            body_name = match.group(1)
            if any(kw in body_name.lower() for kw in ["pelvis", "torso", "ankle", "hip", "waist", "knee", "elbow", "shoulder", "wrist", "hand"]):
                if weld_name not in body_matches:
                    body_matches[weld_name] = body_name
                    break

    if not body_matches:
        return xml_text

    weld_lines = []
    for weld_name, body_name in body_matches.items():
        weld_lines.append(
            f'        <weld active="true" name="{weld_name}" '
            f'body1="{body_name}" body2="world" '
            f'solref="0.02 1" solimp="0.9 0.95 0.001"/>'
        )
    welds_xml = '\n'.join(weld_lines)

    import re as _re
    eq_match = _re.search(r'<equality>(.*?)</equality>', xml_text, _re.DOTALL)
    if eq_match:
        insert_pos = eq_match.end() - len('</equality>')
        xml_text = xml_text[:insert_pos] + welds_xml + '\n' + xml_text[insert_pos:]
    else:
        act_match = _re.search(r'</actuator>', xml_text)
        if act_match:
            insert_pos = act_match.end()
            xml_text = (xml_text[:insert_pos] +
                        '\n    <equality>\n' + welds_xml + '\n    </equality>' +
                        xml_text[insert_pos:])
    return xml_text


# =============================================================================
# Hand grasp patch: stronger finger servos + screwdriver grip friction
# =============================================================================
# MuJoCo 3.x <position> actuators are pure PD servos (no gravity compensation):
# grip force = kp * (close_target - blocked_angle), so kp 1.5~2.0 yields too
# little force to hold props (measured: screwdriver 0.44 kg needs ~11 N/contact
# at its μ=0.2 handle, but typical grasps only produce 1~7 N). Scale kp/kv ×10
# (keeps the damping ratio) and raise the screwdriver handle friction to 1.0
# (the only prop with μ<1.0 — it was the guaranteed slip case).
_HAND_KP_THUMB0 = 10.0
_HAND_KP_DEFAULT = 8.0
_HAND_KV = 0.1

_HAND_ACT_TAG_RE = re.compile(r'<position\s+joint="[^"]*_hand_[a-z0-9_]+_joint"[^>]*>')
_SCREWDRIVER_BODY_RE = re.compile(
    r'(<body\s+[^>]*name="Group_Interactive_Screwdriver_task_screwdriver_joint">)(.*?)(</body>)',
    re.DOTALL,
)
_FRICTION_02_RE = re.compile(r'friction="0\.2\d*\s+0\.005\d*\s+0\.0001\d*"')
_FRICTION_1_REPL = 'friction="1.000000000 0.005000000 0.000100000"'


def _inject_hand_grasp_patch(xml_text):
    """Strengthen finger position servos (kp/kv) and fix screwdriver grip friction.

    Applied at load time next to the pelvis weld injection. Only touches the 14
    hand position actuators and the screwdriver handle geom; returns the
    original text unchanged (plus a WARNING) if nothing matched.
    """
    n_act = 0

    def _repl_act_tag(m):
        nonlocal n_act
        tag = m.group(0)
        kp = _HAND_KP_THUMB0 if 'thumb_0_joint"' in tag else _HAND_KP_DEFAULT
        tag = re.sub(r'\bkp="[0-9.]+"', f'kp="{kp}"', tag)
        tag = re.sub(r'\bkv="[0-9.]+"', f'kv="{_HAND_KV}"', tag)
        n_act += 1
        return tag

    patched = _HAND_ACT_TAG_RE.sub(_repl_act_tag, xml_text)

    n_fric = 0

    def _repl_screwdriver_body(m):
        nonlocal n_fric
        body = _FRICTION_02_RE.sub(lambda mm: _FRICTION_1_REPL, m.group(2))
        if body != m.group(2):
            n_fric += 1
        return m.group(1) + body + m.group(3)

    patched = _SCREWDRIVER_BODY_RE.sub(_repl_screwdriver_body, patched)

    if n_act or n_fric:
        print(f"[Hand Grasp] patched {n_act} hand actuators "
              f"(thumb_0 kp 2.0→{_HAND_KP_THUMB0}, others 1.5→{_HAND_KP_DEFAULT}, "
              f"kv 0.1→{_HAND_KV}), {n_fric} screwdriver geom(s) friction 0.2→1.0")
    else:
        print("[Hand Grasp] WARNING: no hand actuators / screwdriver geom matched — XML format changed?")
    return patched


# =============================================================================
# Bridge Controller
# =============================================================================
class Twist2OrcaLabBridge:
    def __init__(
        self,
        policy_path=None,
        device="cuda",
        fix_feet=False,
        pd_override=True,
        frame_skip=5,
        time_step=0.001,
        redis_host="localhost",
        redis_port=6379,
        enable_render=True,
        agent_name=None,
        # Default to RightController.key_two (B): key_one (A) is teleop's own
        # state-cycle button — sharing it desyncs the two state machines
        # (bridge misses a rising edge → stuck one state behind teleop).
        start_button="RightController.key_two",
        reset_button="LeftController.key_two",
        auto_teleop=False,
        patch_xml_inplace=False,
        hand_grasp=True,
        stale_ms=200,
        verbose=False,
        selftest=False,
        # Record/replay
        record_button="RightController.axis_click",
        replay_button="LeftController.axis_click",
        record_file="logs/bridge_record.json",
        # Local mode args
        local_xml=None,
        # gRPC mode args
        orcagym_addr="localhost:50051",
        # Leg PD tuning (A + C): gain multiplier on the 12 leg joints and EMA
        # low-pass on leg PD targets (0=off). Defaults keep prior behavior.
        leg_pd_gain=1.0,
        leg_ema_alpha=0.0,
        # B1 twin: replay the real robot's policy target instead of running the
        # ONNX policy locally. See module docstring.
        replay_target=False,
        replay_gains="real",
        target_key="action_low_level_unitree_g1_with_hands",
        target_ts_key="t_action_low_level",
    ):
        self.fix_feet = fix_feet
        self.pd_override = pd_override
        self.leg_pd_gain = leg_pd_gain
        self.leg_ema_alpha = leg_ema_alpha
        self.replay_target = replay_target
        self.replay_gains = replay_gains
        self.target_key = target_key
        self.target_ts_key = target_ts_key
        self.frame_skip = frame_skip
        self.time_step_val = time_step
        self.enable_render = enable_render
        self.local_mode = local_xml is not None
        self._agent_name_override = agent_name
        self.start_button = start_button
        self.reset_button = reset_button
        self.auto_teleop = auto_teleop
        self.patch_xml_inplace = patch_xml_inplace
        self.hand_grasp = hand_grasp
        self.stale_ms = stale_ms
        self.verbose = verbose
        self.selftest = selftest
        self.record_button = record_button
        self.replay_button = replay_button
        self.record_file = record_file

        # Record/replay state (independent of the 4-state teleop machine)
        self._recording = False
        self._record_buf = []          # list of (t_rel_s, mimic_35, hand_l7, hand_r7)
        self._record_t0 = 0.0
        self._replaying = False
        self._replay_buf = []          # loaded/recorded frames
        self._replay_idx = 0
        self._replay_t0 = 0.0

        # Load ONNX policy (skipped in replay-target mode: the real controller
        # already produced the joint targets; the bridge only replays them).
        self.device = device
        if replay_target:
            self.policy = None
            print(f"[Bridge] REPLAY-TARGET (B1 twin) mode: ONNX policy disabled; "
                  f"replaying '{target_key}' (ts '{target_ts_key}') with "
                  f"{replay_gains} actuator gains")
        else:
            self.policy = load_onnx_policy(policy_path, device)

        # Redis
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
        try:
            self.redis_client.ping()
            print(f"Redis connected: {redis_host}:{redis_port}")
        except Exception as e:
            print(f"WARNING: Redis connection failed ({e}). Will retry in run loop.")

        # Build env kwargs
        env_kwargs = {"xml_assets_dir": os.path.expanduser("~/.orcagym/tmp")}

        if self.local_mode:
            xml_path = local_xml
            # NOTE: no pre-patching here — welds are injected at load time
            # by patching gym.load_model_xml (mirroring g1_pick_collection_tele_lerobot.py)
            self._temp_xml_path = None
            print(f"Creating DataCollectionEnv in LOCAL mode: {xml_path}")
            if self.enable_render:
                # Hybrid: gRPC channel for OrcaStudio rendering, local XML for model data
                print("  (gRPC render enabled, skipping gRPC XML download)")
                env_kwargs["skip_grpc_load"] = False
                env_kwargs["local_xml_path"] = xml_path
            else:
                # Pure local: no gRPC at all
                env_kwargs["skip_grpc_load"] = True
                env_kwargs["local_xml_path"] = xml_path
        else:
            self._temp_xml_path = None
            print(f"Creating DataCollectionEnv in gRPC mode: {orcagym_addr}")

        self.env = DataCollectionEnv(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=["g1_pick"],  # placeholder, will be auto-detected below
            time_step=time_step,
            default_joint_values={},
            obs_callback=self._dummy_obs_callback,
            **env_kwargs,
        )

        # ── Patch gym.load_model_xml to inject XML edits at load time ──
        # Mirrors g1_pick_collection_tele_lerobot.py: env is created once (1st load,
        # no weld), the loader is patched, then init_env() re-loads (2nd load) with
        # welds (+ hand grasp patch). Hand grasp patch is applied whenever enabled,
        # independently of --fix_feet.
        if self.fix_feet or self.hand_grasp:
            self._wrap_xml_loader_with_patch()

        # ── Local mode: render fails without gRPC ──
        if self.local_mode and not self.enable_render:
            self.env.unwrapped.render = lambda: None

        self.env.init_env()
        print(f"[Bridge] Scene ready. nu={self.env.model.nu} nq={self.env.model.nq} nv={self.env.model.nv}")

        # Auto-detect robot agent name from model joints (robust to different robot assets)
        self._detect_agent_name()

        # Clean up temp welded XML created by the load-time weld injection
        for tmp in getattr(self, "_loader_temp_files", []):
            try:
                os.unlink(tmp)
                print(f"[Bridge] Temp welded XML cleaned up: {os.path.basename(tmp)}")
            except OSError:
                pass

        # Set robot to standing pose (TWIST2 DEFAULT_DOF_POS)
        self._init_robot_pose()
        self._resolve_actuators()
        self._override_pd_gains()
        self._resolve_joint_offsets()
        self._resolve_hand_actuators()

        # OrcaLab hand poses (open/close 14 each) + last raw hand pose for
        # graceful degradation in _read_hand_poses
        self.hand_open_orca, self.hand_close_orca = _load_orca_hand_poses()
        self._last_hand_raw = np.zeros(14, dtype=np.float32)
        self._validate_hand_poses()
        print(f"[Bridge] OrcaLab hand poses: open=zeros(14) "
              f"L close={np.round(self.hand_close_orca[:7], 3).tolist()} "
              f"R close={np.round(self.hand_close_orca[7:], 3).tolist()}")

        # Build lower body lock and apply immediately
        if self.fix_feet:
            self._build_lower_body_lock()
            self._apply_lower_body_lock()

        # Init history buffer with 10 zero frames
        self.history_buf = deque(maxlen=HISTORY_LEN)
        for _ in range(HISTORY_LEN):
            self.history_buf.append(np.zeros(N_OBS_SINGLE, dtype=np.float32))

        self.last_action = np.zeros(29, dtype=np.float32)
        self.step_count = 0

        # ── Time-driven policy beat (~100Hz, aligned with sim2sim sim_decimation=10) ──
        # Loop iterates at 1/(time_step*frame_skip) Hz; run policy every N iterations
        # even when mimic is unchanged, so proprio feedback keeps driving the arms.
        self.policy_interval = max(
            1,
            int(round(1000.0 / (self.time_step_val * 1000.0 * self.frame_skip) / 100.0)),
        )
        loop_hz = 1000.0 / (self.time_step_val * 1000.0 * self.frame_skip)
        print(f"[Bridge] Policy beat: every {self.policy_interval} loop iterations "
              f"(loop ~{loop_hz:.0f}Hz → policy ~{loop_hz / self.policy_interval:.0f}Hz)")

        # ── Diagnostics state (--verbose) ──
        # Initialized to standing pose so the leg EMA starts from the correct target.
        self.last_pd_target = DEFAULT_DOF_POS.copy().astype(np.float32)
        self.policy_calls = 0
        self._bridge_state = None
        self._diag_arm_last = None
        self._diag_t_last = 0.0
        self._mimic_unchanged_beats = 0

        # ── Selftest: synthetic button presses (s) and duration (s) ──
        self._selftest_press_times = [0.5, 2.5, 3.5]
        self._selftest_press_idx = 0
        self._selftest_duration = 5.0

    def _wrap_xml_loader_with_patch(self):
        """Patch env.gym.load_model_xml to inject XML edits at load time.

        Mirrors g1_pick_collection_tele_lerobot.py: the original loader is called
        first (gets the XML path), the file is read, edits (pelvis weld for
        --fix_feet, hand grasp kp/kv + screwdriver friction) are injected as
        plain strings, and the patched XML is written out before returning the
        path. The env is created once (no edits), then init_env() re-loads with
        this patched loader (2nd load), so MuJoCo parses the edited XML.
        """
        _orig_load = self.env.gym.load_model_xml
        self._loader_temp_files = []

        async def _patched_load_model_xml():
            path = await _orig_load()
            with open(path, "r") as f:
                xml = f.read()

            patched = xml
            if self.fix_feet:
                patched = _inject_xml_weld_text(patched)
            if self.hand_grasp:
                patched = _inject_hand_grasp_patch(patched)
            if patched == xml:
                print(f"[Patch] no edits applied, skip: {os.path.basename(path)}")
                return path

            if self.patch_xml_inplace:
                # Backup + write back to the original cached XML (like the teleop script)
                bak_path = path + ".bak"
                if not os.path.exists(bak_path):
                    os.rename(path, bak_path)
                with open(path, "w") as f:
                    f.write(patched)
                print(f"[Fix Feet] Injected welds in place: {os.path.basename(path)}")
                return path

            # Temp file next to the original so mesh relative paths still resolve
            xml_dir = os.path.dirname(os.path.abspath(path))
            fd, tmp_path = tempfile.mkstemp(suffix=".xml", prefix="bridge_", dir=xml_dir)
            with os.fdopen(fd, "w") as f:
                f.write(patched)
            self._loader_temp_files.append(tmp_path)
            print(f"[Fix Feet] Injected welds (load-time): {os.path.basename(tmp_path)}")
            return tmp_path

        self.env.gym.load_model_xml = _patched_load_model_xml
        print("[Fix Feet] Patched gym.load_model_xml — welds will be injected on next model load")

    def _detect_agent_name(self):
        """Auto-detect the robot agent prefix from model joint names."""
        if self._agent_name_override:
            self.env._agent_names = [self._agent_name_override]
            print(f"[Bridge] Using agent name override: '{self._agent_name_override}'")
            return

        import mujoco
        m = self.env.gym._mjModel
        all_joints = set()
        for i in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                all_joints.add(name)

        candidates = []
        for short_name in TWIST2_JOINT_NAMES:
            for full_name in all_joints:
                if full_name.endswith(short_name):
                    prefix = full_name[: -len(short_name)].rstrip("_")
                    if prefix:
                        candidates.append(prefix)
                    break

        if candidates:
            from collections import Counter
            agent_name = Counter(candidates).most_common(1)[0][0]
            self.env._agent_names = [agent_name]
            print(f"[Bridge] Auto-detected robot agent: '{agent_name}' "
                  f"(matched {len(candidates)}/{len(TWIST2_JOINT_NAMES)} joints)")
        else:
            print("[Bridge] WARNING: Could not auto-detect robot agent name; "
                  "assuming 'g1_pick'. Set via --agent_name if wrong.")
            self.env._agent_names = ["g1_pick"]

    def _init_robot_pose(self):
        init_joint_values = {}
        for i, name in enumerate(TWIST2_JOINT_NAMES):
            init_joint_values[name] = float(DEFAULT_DOF_POS[i])
        self.env.set_default_joint_values(init_joint_values)
        self.env.mj_forward()
        self.env.update_data()
        print("[Bridge] Robot set to TWIST2 standing pose (shoulder_roll=0.4)")

    def _resolve_actuators(self):
        self.body_act_ids = []
        for name in TWIST2_JOINT_NAMES:
            full_name = self.env.actuator(name)
            act_id = self.env.model.actuator_name2id(full_name)
            self.body_act_ids.append(act_id)
        print(f"[Bridge] Resolved {len(self.body_act_ids)} body actuator IDs")

    def _override_pd_gains(self):
        """Override the 29 body position-actuator PD gains.

        Policy mode uses the TWIST2 training gains (arm kp=40/kv=5, wrist
        kp=4.0/kv=0.2): the OrcaLab XML gains (arm kp≈14, wrist kp≈14-17)
        otherwise cause limit-cycle arm oscillation. Replay-target (B1 twin)
        mode defaults to the real robot's downlink gains (g1.yaml) so the twin
        plant matches the real plant; --replay_gains train forces training
        gains. The 14 hand actuators are untouched (no policy drives them).
        """
        if not self.pd_override:
            return
        if self.replay_target and self.replay_gains == "real":
            kp_src, kv_src = REAL_STIFFNESS, REAL_DAMPING
            gain_label = "real g1.yaml downlink values"
        else:
            kp_src, kv_src = TWIST2_STIFFNESS, TWIST2_DAMPING
            gain_label = "TWIST2 training values"
        m = self.env.gym._mjModel
        for i, act_id in enumerate(self.body_act_ids):
            kp = float(kp_src[i])
            kv = float(kv_src[i])
            if i < 12:
                kp *= self.leg_pd_gain
                kv *= self.leg_pd_gain
            m.actuator_gainprm[act_id, 0] = kp
            m.actuator_biasprm[act_id, 1] = -kp
            m.actuator_biasprm[act_id, 2] = -kv
        print(f"[Bridge] Overrode {len(self.body_act_ids)} body actuator PD gains "
              f"to {gain_label}; leg gain x{self.leg_pd_gain:.2f}")

    def _resolve_hand_actuators(self):
        self.hand_act_ids = []
        self._hand_joint_full_names = [self.env.joint(n) for n in HAND_JOINT_NAMES]
        for name in HAND_JOINT_NAMES:
            full_name = self.env.actuator(name)
            act_id = self.env.model.actuator_name2id(full_name)
            self.hand_act_ids.append(act_id)
        print(f"[Bridge] Resolved {len(self.hand_act_ids)} hand actuator IDs")

    def _validate_hand_poses(self):
        """Warn loudly if the OrcaLab hand open/close poses fall outside the
        model's actual joint ranges (guards against conf/XML drift, e.g. the
        thumb_2 sign error that once silently inverted the grip direction)."""
        import mujoco
        m = self.env.gym._mjModel
        tol = 1e-3
        for i, full_name in enumerate(self._hand_joint_full_names):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, full_name)
            if jid < 0:
                continue
            lo, hi = m.jnt_range[jid]
            for label, val in (("open", self.hand_open_orca[i]),
                               ("close", self.hand_close_orca[i])):
                if not (lo - tol <= val <= hi + tol):
                    print(f"[Bridge] WARNING: hand {label} pose [{i}] {full_name} = {val:.4f} "
                          f"outside model range [{lo:.4f}, {hi:.4f}]")

    def _resolve_joint_offsets(self):
        body_joint_names = [self.env.joint(n) for n in TWIST2_JOINT_NAMES]
        body_qpos_off, body_qvel_off, _ = self.env.query_joint_offsets(body_joint_names)
        self.body_qpos_offsets = np.array(body_qpos_off, dtype=int)
        self.body_qvel_offsets = np.array(body_qvel_off, dtype=int)

        hand_joint_names = [self.env.joint(n) for n in HAND_JOINT_NAMES]
        hand_qpos_off, _, _ = self.env.query_joint_offsets(hand_joint_names)
        self.hand_qpos_offsets = np.array(hand_qpos_off, dtype=int)

        free_joint_name = self.env.joint("floating_base_joint")
        free_qpos_off, free_qvel_off, _ = self.env.query_joint_offsets([free_joint_name])
        self.free_qpos_adr = int(free_qpos_off[0])
        self.free_qvel_adr = int(free_qvel_off[0])
        print(f"[Bridge] Freejoint qpos_adr={self.free_qpos_adr}, qvel_adr={self.free_qvel_adr}")
        print(f"[Bridge] Body joint qpos range: [{self.body_qpos_offsets[0]}, {self.body_qpos_offsets[-1]}]")
        print(f"[Bridge] Hand joint qpos range: [{self.hand_qpos_offsets[0]}, {self.hand_qpos_offsets[-1]}]")

    @staticmethod
    def _dummy_obs_callback(_env):
        return {}

    def extract_proprio(self):
        dof_pos = np.array([self.env.data.qpos[off] for off in self.body_qpos_offsets], dtype=np.float32)
        dof_vel = np.array([self.env.data.qvel[off] for off in self.body_qvel_offsets], dtype=np.float32)
        quat = self.env.data.qpos[self.free_qpos_adr + 3:self.free_qpos_adr + 7].copy()
        ang_vel = self.env.data.qvel[self.free_qvel_adr + 3:self.free_qvel_adr + 6].copy()
        return dof_pos, dof_vel, quat, ang_vel

    def build_obs_buf(self, mimic_obs, dof_pos, dof_vel, quat, ang_vel):
        rpy = quatToEuler(quat)
        obs_dof_vel = dof_vel.copy()
        obs_dof_vel[ANKLE_IDX] = 0.0

        obs_proprio = np.concatenate([
            ang_vel * 0.25,
            rpy[:2],
            dof_pos - DEFAULT_DOF_POS,
            obs_dof_vel * 0.05,
            self.last_action,
        ]).astype(np.float32)

        obs_full = np.concatenate([mimic_obs, obs_proprio])
        # mjlab builds the history *including* the current frame, and its actor
        # obs is [current(127), history(11 x 127)] with no future-mimic block.
        self.history_buf.append(obs_full)
        obs_hist = np.array(self.history_buf).flatten()
        obs_buf = np.concatenate([obs_full, obs_hist])
        return obs_buf

    def _remap_hand_pose(self, pose_twist):
        """Remap a TWIST2-convention 14-dim hand pose to OrcaLab joint limits.

        Per-joint linear remap: t_j = (p - open_t) / (close_t - open_t), then
        pose_orca = open_o + t_j * (close_o - open_o). Denominator 0 (TWIST2
        open == close, e.g. thumb_0 which TWIST2 never moves) → t = 0 (open).
        Returns float32(14) in OrcaLab convention.
        """
        pose_twist = np.asarray(pose_twist, dtype=np.float32).reshape(-1)
        denom = HAND_CLOSE_TWIST2 - HAND_OPEN_TWIST2
        t = np.zeros(14, dtype=np.float32)
        nz = denom != 0.0
        t[nz] = (pose_twist[nz] - HAND_OPEN_TWIST2[nz]) / denom[nz]
        return (self.hand_open_orca + t * (self.hand_close_orca - self.hand_open_orca)).astype(np.float32)

    def _read_hand_poses(self):
        """Read action_hand_left/right Redis keys (7-dim each), concat to 14, remap.

        Graceful degradation: a missing/unparseable key keeps that side at its
        previous pose (initial: all open = zeros).
        """
        pose = self._last_hand_raw.copy()
        for side, sl in (("left", slice(0, 7)), ("right", slice(7, 14))):
            raw = self.redis_client.get(f"action_hand_{side}_unitree_g1_with_hands")
            if raw is None:
                continue
            try:
                vals = np.array(json.loads(raw), dtype=np.float32).reshape(-1)
                if vals.size != 7:
                    continue
                pose[sl] = vals
            except (ValueError, TypeError):
                continue
        self._last_hand_raw = pose
        return self._remap_hand_pose(pose)

    def _save_record_file(self):
        """Persist the recorded session to JSON (wall-clock timed frames)."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.record_file)), exist_ok=True)
            payload = {"frames": [{"t_s": t, "mimic": m.tolist(), "hand_l": hl.tolist(), "hand_r": hr.tolist()}
                                  for (t, m, hl, hr) in self._record_buf]}
            n_frames = len(self._record_buf)
            with open(self.record_file, "w") as f:
                json.dump(payload, f)
            print(f"[Bridge] RECORD saved: {n_frames} frames → {self.record_file}")
            # Make the just-recorded session the replay source immediately,
            # then drop the duplicate in-memory copy (replay_buf is the live one).
            self._replay_buf = self._record_buf
            self._record_buf = []
        except Exception as e:
            print(f"[Bridge] RECORD save failed: {e}")

    def _load_record_file(self):
        """Load a recorded session for replay (returns True on success)."""
        try:
            with open(self.record_file) as f:
                payload = json.load(f)
            self._replay_buf = [
                (float(fr["t_s"]),
                 np.array(fr["mimic"], dtype=np.float32),
                 np.array(fr["hand_l"], dtype=np.float32),
                 np.array(fr["hand_r"], dtype=np.float32))
                for fr in payload["frames"]
            ]
            print(f"[Bridge] REPLAY loaded {len(self._replay_buf)} frames from {self.record_file}")
            return len(self._replay_buf) > 0
        except FileNotFoundError:
            print(f"[Bridge] REPLAY file not found: {self.record_file}")
            return False
        except Exception as e:
            print(f"[Bridge] REPLAY load failed: {e}")
            return False

    def run(self):
        print("[Bridge] Starting main control loop...")
        print(f"[Bridge] Press '{self.start_button}' to cycle IDLE → TELEOP → PAUSE → TELEOP ...")
        print(f"[Bridge] Press '{self.reset_button}' to refresh the scene layout (reset to standing + IDLE)")
        print(f"[Bridge] Press '{self.record_button}' to toggle RECORD (teleop input capture)")
        print(f"[Bridge] Press '{self.replay_button}' to toggle REPLAY (plays the recorded session)")

        if not self.selftest and not self.replay_target:
            # Send initial state to Redis. Skipped in twin mode: sim2real owns
            # state_* there, and zeroing them at startup would corrupt its
            # telemetry until the next real frame.
            initial_state = np.zeros(34, dtype=np.float32)
            self.redis_client.set("state_body_unitree_g1_with_hands", json.dumps(initial_state.tolist()))
            self.redis_client.set("state_hand_left_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
            self.redis_client.set("state_hand_right_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
            self.redis_client.set("state_neck_unitree_g1_with_hands", json.dumps(np.zeros(2).tolist()))
            self.redis_client.set("t_state", int(time.time() * 1000))

        # ── 4-state machine, aligned with teleop StateMachine.update() ──
        # idle → (key) → teleop → (key) → pause → (key) → teleop → (key) → pause ...
        state = "teleop" if self.auto_teleop else "idle"
        self._bridge_state = state
        if state == "teleop":
            print("[Bridge] STATE: teleop (--auto_teleop)")
        else:
            print("[Bridge] STATE: idle (waiting for button)")

        last_mimic = None
        last_hand = np.zeros(14, dtype=np.float32)  # initial: hands open
        last_ctrl = self._make_standing_ctrl()  # idle: standing pose; pause: frozen pose
        button_was_pressed = False
        reset_was_pressed = False
        record_was_pressed = False
        replay_was_pressed = False
        policy_counter = 0
        sim_dt = self.time_step_val * self.frame_skip  # sim time per loop iteration (s)
        iter_count = 0

        if self.selftest:
            print("[Bridge] SELFTEST mode: synthetic mimic (sine arms) + synthetic button presses "
                  f"(t={self._selftest_press_times[0]}s→teleop, {self._selftest_press_times[1]}s→pause, "
                  f"{self._selftest_press_times[2]}s→teleop), auto-exit at {self._selftest_duration}s")

        selftest_start = time.time()

        try:
            while True:
                t_start = time.time()
                loop_t = time.time() - selftest_start
                iter_count += 1

                # ── Joystick button gating (4-state cycle, mirrors teleop) ──
                pressed = False
                reset_pressed = False
                record_pressed = False
                replay_pressed = False
                if self.selftest:
                    while (self._selftest_press_idx < len(self._selftest_press_times)
                           and loop_t >= self._selftest_press_times[self._selftest_press_idx]):
                        pressed = True
                        self._selftest_press_idx += 1
                else:
                    raw_btn = self.redis_client.get("controller_data")
                    if raw_btn is not None:
                        try:
                            ctrl_data = json.loads(raw_btn)
                            pressed = self._read_button(ctrl_data)
                            reset_pressed = self._read_button_path(ctrl_data, self.reset_button)
                            record_pressed = self._read_button_path(ctrl_data, self.record_button)
                            replay_pressed = self._read_button_path(ctrl_data, self.replay_button)
                        except Exception:
                            pass

                if pressed and not button_was_pressed:
                    if state == "idle":
                        state = "teleop"
                        last_mimic = None
                    elif state == "teleop":
                        state = "pause"
                    elif state == "pause":
                        state = "teleop"
                        last_mimic = None
                    self._bridge_state = state
                    print(f"[Bridge] STATE: {state}")
                button_was_pressed = pressed

                # ── Layout refresh button: reset scene + back to IDLE (physics pauses) ──
                if reset_pressed and not reset_was_pressed:
                    self._reset_scene()
                    last_mimic = None
                    last_hand = np.zeros(14, dtype=np.float32)
                    last_ctrl = self._make_standing_ctrl()
                    policy_counter = 0
                    state = "idle"
                    self._bridge_state = state
                    print("[Bridge] STATE: idle (layout refreshed)")
                reset_was_pressed = reset_pressed

                # ── Record toggle: capture (mimic, hand) into memory ──
                if record_pressed and not record_was_pressed:
                    if self._recording:
                        self._recording = False
                        self._save_record_file()
                    else:
                        self._recording = True
                        self._record_buf = []
                        self._record_t0 = time.time()
                        print("[Bridge] RECORD started (press again to stop & save)")
                record_was_pressed = record_pressed

                # ── Replay toggle: play the recorded session instead of Redis ──
                if replay_pressed and not replay_was_pressed:
                    if self._replaying:
                        self._replaying = False
                        print("[Bridge] REPLAY stopped")
                    else:
                        # Use the in-memory latest recording when present
                        # (_save_record_file keeps _replay_buf fresh); fall back
                        # to disk only when nothing was recorded this session.
                        if not self._replay_buf and not self._load_record_file():
                            print("[Bridge] REPLAY: nothing to play")
                        else:
                            self._replaying = True
                            self._replay_idx = 0
                            self._replay_t0 = time.time()
                            state = "teleop"
                            self._bridge_state = state
                            last_mimic = None
                            print(f"[Bridge] REPLAY started ({len(self._replay_buf)} frames)")
                replay_was_pressed = replay_pressed

                # ── B1 twin: replay the real robot's policy target ──
                # No local policy, no mimic obs/history: sim2real already
                # computed the motor target from the real state, so both
                # plants receive the identical joint command.
                if state == "teleop" and self.replay_target:
                    if self.selftest:
                        target = self._selftest_target(loop_t)
                        hand = self._remap_hand_pose(self._selftest_hand(loop_t))
                    else:
                        target, _ = self._read_replay_target()
                        hand = None
                    if target is not None:
                        last_ctrl = self._apply_replay_target(target, hand)

                # ── Policy control (teleop state only) ──
                # Event-driven: mimic change → immediate policy run (reset beat counter).
                # Time-driven: beat every `policy_interval` iterations → policy runs even
                # when mimic is frozen, so proprio feedback keeps driving the arms.
                # Stale data: skip policy entirely (no event, no beat); physics still steps.
                elif state == "teleop":
                    run_policy = False
                    mimic_changed = False
                    hand = None
                    if self.selftest:
                        fresh = True
                        mimic = self._selftest_mimic(loop_t)
                        hand = self._remap_hand_pose(self._selftest_hand(loop_t))
                    elif self._replaying and self._replay_buf:
                        # Replay: feed the recorded stream by wall-clock timing
                        fresh = True
                        mimic = None
                        hand = None
                        t_rel = time.time() - self._replay_t0
                        while (self._replay_idx < len(self._replay_buf)
                               and self._replay_buf[self._replay_idx][0] <= t_rel):
                            _, mimic, hand_l, hand_r = self._replay_buf[self._replay_idx]
                            self._replay_idx += 1
                        if self._replay_idx >= len(self._replay_buf):
                            # end of recording: hold last frame, then loop
                            print("[Bridge] REPLAY finished (looping)")
                            self._replay_idx = 0
                            self._replay_t0 = time.time()
                        if mimic is not None:
                            hand = np.concatenate([hand_l, hand_r]).astype(np.float32)
                        else:
                            fresh = False
                    else:
                        fresh = False
                        mimic = None
                        raw = self.redis_client.get("action_body_unitree_g1_with_hands")
                        if raw is not None:
                            ta_raw = self.redis_client.get("t_action")
                            now_ms = int(time.time() * 1000)
                            fresh = True
                            if ta_raw is not None:
                                try:
                                    if now_ms - int(ta_raw) > self.stale_ms:
                                        fresh = False
                                except (ValueError, TypeError):
                                    pass
                            if fresh:
                                try:
                                    mimic_arr = np.array(json.loads(raw), dtype=np.float32)
                                except (ValueError, TypeError):
                                    print("[Bridge] WARNING: failed to parse action_body; skipping this frame")
                                    mimic_arr = None
                                if mimic_arr is not None and mimic_arr.size != N_MIMIC_OBS:
                                    print(f"[Bridge] WARNING: action_body has {mimic_arr.size} values "
                                          f"(expected {N_MIMIC_OBS}); skipping this frame")
                                    mimic_arr = None
                                if mimic_arr is not None:
                                    mimic = mimic_arr
                                    hand = self._read_hand_poses()

                        # Record: capture fresh (mimic, hand) into the buffer.
                        # Dedup: only append when the stream actually changed —
                        # the bridge loop (200Hz) re-reads the same Redis frame
                        # (100Hz) twice, which would double memory usage.
                        if self._recording and mimic is not None and hand is not None:
                            dup = False
                            if self._record_buf:
                                last_t, last_m, last_hl, last_hr = self._record_buf[-1]
                                dup = (np.array_equal(last_m, mimic)
                                       and np.array_equal(last_hl, hand[:7])
                                       and np.array_equal(last_hr, hand[7:]))
                            if not dup:
                                self._record_buf.append(
                                    (time.time() - self._record_t0,
                                     mimic.copy(),
                                     hand[:7].copy(), hand[7:].copy()))

                    if fresh and mimic is not None:
                        if last_mimic is None or not np.array_equal(mimic, last_mimic):
                            last_mimic = mimic.copy()
                            mimic_changed = True
                            policy_counter = 0
                            run_policy = True
                        else:
                            policy_counter += 1
                            if policy_counter >= self.policy_interval:
                                policy_counter = 0
                                run_policy = True

                        # Hand-only change (body frozen) → immediate policy run,
                        # so finger motion is never starved by the beat counter.
                        hand_changed = last_hand is None or not np.array_equal(hand, last_hand)
                        if hand_changed:
                            last_hand = hand.copy()
                            run_policy = True
                            policy_counter = 0

                    if run_policy and hand is not None:
                        last_ctrl = self._run_policy(mimic, mimic_changed, hand)

                # ── Physics step: paused in IDLE until the teleop button is pressed ──
                # (robot holds its initial pose instead of falling before control engages)
                if state != "idle":
                    self.env.do_simulation(last_ctrl, self.frame_skip)
                    if self.fix_feet:
                        self._apply_lower_body_lock()
                if self.enable_render:
                    try:
                        self.env.render()
                    except Exception:
                        pass

                # ── Selftest: print arm/leg/root state every 50 iterations ──
                if self.selftest and iter_count % 50 == 0:
                    self._selftest_print_state(loop_t, state)

                # ── Selftest: auto-exit ──
                if self.selftest and loop_t >= self._selftest_duration:
                    print(f"[Bridge] Selftest finished ({self._selftest_duration}s). Exiting.")
                    break

                # ── Real-time pacing: keep sim at real time (5ms per iteration) ──
                elapsed = time.time() - t_start
                if elapsed < sim_dt:
                    time.sleep(sim_dt - elapsed)

        except KeyboardInterrupt:
            print("\n[Bridge] Interrupted by user.")
        except Exception as e:
            print(f"[Bridge] Error in run loop: {e}")
            traceback.print_exc()
        finally:
            if self._recording and self._record_buf:
                self._save_record_file()
            self.env.close()
            print("[Bridge] Environment closed.")

    def _read_replay_target(self):
        """Read the real robot's policy target (29 absolute joint positions).

        Published by sim2real (server_low_level_g1_real.py --publish_target) to
        self.target_key with its own timestamp self.target_ts_key (ms).
        Returns (target, fresh); target is None when missing/unparseable/stale.
        """
        raw = self.redis_client.get(self.target_key)
        if raw is None:
            return None, False

        fresh = True
        ts_raw = self.redis_client.get(self.target_ts_key)
        if ts_raw is not None:
            try:
                if int(time.time() * 1000) - int(ts_raw) > self.stale_ms:
                    fresh = False
            except (ValueError, TypeError):
                pass
        if not fresh:
            return None, False

        try:
            arr = np.array(json.loads(raw), dtype=np.float32)
        except (ValueError, TypeError):
            print(f"[Bridge] WARNING: failed to parse {self.target_key}; skipping this frame")
            return None, False
        if arr.size != len(TWIST2_JOINT_NAMES):
            print(f"[Bridge] WARNING: {self.target_key} has {arr.size} values "
                  f"(expected {len(TWIST2_JOINT_NAMES)}); skipping this frame")
            return None, False
        return arr.reshape(-1), True

    def _apply_replay_target(self, target, hand=None):
        """Drive the 29 body actuators with the replayed real target + Redis hands.

        The target is already the absolute motor position from the real
        controller (default_dof_pos + raw_action*ACTION_SCALE, leg EMA applied
        there), so it is fed straight in: no ACTION_SCALE / DEFAULT_DOF_POS
        transform and no second leg EMA. `hand` (14-dim, OrcaLab convention)
        overrides the Redis hand read (used by --selftest).
        """
        self.last_action = ((target - DEFAULT_DOF_POS) / ACTION_SCALE).astype(np.float32)
        self.last_pd_target = target.copy()

        if hand is None:
            hand = self._read_hand_poses()

        ctrl = np.zeros(self.env.model.nu, dtype=np.float32)
        for i, act_id in enumerate(self.body_act_ids):
            ctrl[act_id] = target[i]
        for i, hid in enumerate(self.hand_act_ids):
            ctrl[hid] = hand[i]

        self.step_count += 1
        self.policy_calls += 1

        # NOTE: intentionally do NOT publish state_* in twin mode — sim2real is
        # the authoritative state publisher for the real robot, and overwriting
        # state_body_*/t_state from here would corrupt its telemetry.
        return ctrl

    def _selftest_target(self, t):
        """Synthetic 29-dim real target: arms (idx 15-28) sine, rest DEFAULT."""
        target = DEFAULT_DOF_POS.copy()
        target[15:29] = target[15:29] + 0.5 * np.sin(2.0 * np.pi * 0.5 * (t - 0.5))
        return target.astype(np.float32)

    def _run_policy(self, mimic, mimic_changed, hand=None):
        """Run ONNX policy on current sim state + mimic; update last_ctrl. Returns new ctrl."""
        dof_pos, dof_vel, quat, ang_vel = self.extract_proprio()

        obs_buf = self.build_obs_buf(mimic, dof_pos, dof_vel, quat, ang_vel)
        assert obs_buf.shape[0] == TOTAL_OBS_SIZE, \
            f"Expected {TOTAL_OBS_SIZE} obs, got {obs_buf.shape[0]}"

        obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()

        raw_action = np.clip(raw_action, -10.0, 10.0)
        if self.fix_feet:
            raw_action[0:15] = 0.0

        self.last_action = raw_action
        pd_target = raw_action * ACTION_SCALE + DEFAULT_DOF_POS

        # Leg PD target low-pass (EMA): attenuate high-freq jitter on the 12 leg
        # joints while leaving arms/waist untouched.
        if self.leg_ema_alpha > 0.0:
            pd_target[0:12] = (self.leg_ema_alpha * self.last_pd_target[0:12]
                               + (1.0 - self.leg_ema_alpha) * pd_target[0:12])
        self.last_pd_target = pd_target

        ctrl = np.zeros(self.env.model.nu, dtype=np.float32)
        for i, act_id in enumerate(self.body_act_ids):
            ctrl[act_id] = pd_target[i]
        if hand is not None:
            for i, hid in enumerate(self.hand_act_ids):
                ctrl[hid] = hand[i]

        self.step_count += 1
        self.policy_calls += 1

        if not self.selftest:
            # Publish state to Redis
            rpy = quatToEuler(quat)
            state_body = np.concatenate([ang_vel, rpy[:2], dof_pos])
            self.redis_client.set("state_body_unitree_g1_with_hands", json.dumps(state_body.tolist()))
            self.redis_client.set("t_state", int(time.time() * 1000))

        if self.verbose:
            self._diag_update(mimic, mimic_changed, hand)

        return ctrl

    def _diag_update(self, mimic, mimic_changed, hand=None):
        """Track arm-link diagnostics; print a summary every 500 policy calls (--verbose).

        Inferred teleop-side state:
          - mimic == DEFAULT_MIMIC_OBS        → teleop side idle
          - arms unchanged for many beats      → pause / retarget stalled
          - arms changing                      → teleop
        """
        arm_idx = [21, 22, 29, 30]
        arm_vals = mimic[arm_idx]

        rate = 0.0
        if self._diag_arm_last is not None and mimic_changed:
            dt = time.time() - self._diag_t_last
            if dt > 1e-3:
                rate = float(np.linalg.norm(arm_vals - self._diag_arm_last) / dt)
        self._diag_arm_last = arm_vals.copy()
        self._diag_t_last = time.time()

        if mimic_changed:
            self._mimic_unchanged_beats = 0
        else:
            self._mimic_unchanged_beats += 1

        if np.allclose(mimic, DEFAULT_MIMIC_OBS, atol=1e-2):
            inferred = "idle"
        elif self._mimic_unchanged_beats >= 20:
            inferred = "pause/停滞"
        else:
            inferred = "teleop"

        if self.policy_calls % 500 == 0:
            pd_arm = self.last_pd_target[15:29]
            hand_str = ""
            if hand is not None:
                hand_str = f" hand_ctrl[0:3]={np.round(hand[0:3], 3).tolist()}"
            print(f"[Bridge] Diag step={self.step_count} policy_calls={self.policy_calls} "
                  f"bridge_state={self._bridge_state} inferred_teleop={inferred} "
                  f"mimic_arm[21,22,29,30]={np.round(arm_vals, 4).tolist()} "
                  f"arm_change_rate={rate:.3f} rad/s "
                  f"pd_arm[15:28] range=[{pd_arm.min():.3f}, {pd_arm.max():.3f}]"
                  f"{hand_str}")

    def _selftest_mimic(self, t):
        """Synthetic mimic_obs: both arms (idx 21-34) swing as 0.5*sin(2π*0.5*t), rest DEFAULT."""
        mimic = DEFAULT_MIMIC_OBS.copy()
        mimic[21:35] = mimic[21:35] + 0.5 * np.sin(2.0 * np.pi * 0.5 * (t - 0.5))
        return mimic.astype(np.float32)

    def _selftest_hand(self, t):
        """Synthetic TWIST2-convention hand pose: open↔close at 0.25 Hz.

        g(t) = 0.5*(1 - cos(2π*0.25*(t-0.5))): g=0 at t=0.5 (open, synced with
        teleop entry), g=1 at t=2.5 (full close), g=0 at t=4.5. Must pass
        through _remap_hand_pose in the run loop (full chain verification).
        """
        g = 0.5 * (1.0 - np.cos(2.0 * np.pi * 0.25 * (t - 0.5)))
        return (HAND_OPEN_TWIST2 + g * (HAND_CLOSE_TWIST2 - HAND_OPEN_TWIST2)).astype(np.float32)

    def _selftest_print_state(self, t, state):
        dof_pos, _, _, _ = self.extract_proprio()
        root_xyz = self.env.data.qpos[self.free_qpos_adr:self.free_qpos_adr + 3]
        hand_dof = np.array([self.env.data.qpos[off] for off in self.hand_qpos_offsets], dtype=np.float32)
        hand_target = self._remap_hand_pose(self._selftest_hand(t))
        hand_idx = [0, 1, 2, 5, 7, 8, 9, 12]  # L/R thumb_0..2 + index_0
        print(f"[Selftest] t={t:.2f}s state={state} "
              f"arm_dof={np.round(dof_pos[15:29], 3).tolist()} "
              f"legs_dof={np.round(dof_pos[0:15], 3).tolist()} "
              f"root_xyz={np.round(root_xyz, 3).tolist()} "
              f"hand_dof={np.round(hand_dof[hand_idx], 3).tolist()} "
              f"hand_target={np.round(hand_target[0:8], 3).tolist()}")

    def _read_button_path(self, controller_data, path):
        """Read a button state from controller_data using a dotted path
        (e.g. RightController.key_two)."""
        node = controller_data
        for part in path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return False
        return bool(node)

    def _read_button(self, controller_data):
        """Read the state-cycle button (self.start_button)."""
        return self._read_button_path(controller_data, self.start_button)

    def _build_lower_body_lock(self):
        """Build dicts to force-lock the root (freejoint) and lower body (legs + waist)
        at every timestep. This hard lock prevents the robot from flying/drifting even
        if the XML weld is soft or missing."""
        # Freejoint: lock to the model's initial root position/orientation
        self._lock_freejoint_name = self.env.joint("floating_base_joint")
        free_off, free_vel_off, _ = self.env.query_joint_offsets([self._lock_freejoint_name])
        self._lock_free_qpos = self.env.data.qpos[int(free_off[0]):int(free_off[0]) + 7].copy().astype(np.float32)
        print(f"[Bridge] Freejoint lock target qpos: {self._lock_free_qpos}")

        # Legs + waist (indices 0-14 in TWIST2 order): force to DEFAULT_DOF_POS + zero velocity
        self._lock_lower_qpos = {}
        self._lock_lower_qvel = {}
        for i in range(15):
            full_name = self.env.joint(TWIST2_JOINT_NAMES[i])
            self._lock_lower_qpos[full_name] = np.float32(DEFAULT_DOF_POS[i])
            self._lock_lower_qvel[full_name] = np.float32(0.0)

        print(f"[Bridge] Built lower-body lock: freejoint + 15 joints (legs+waist) forced at every timestep")

    def _apply_lower_body_lock(self):
        """Force freejoint + lower body joints to their locked poses at every sim step.
        Calls mj_forward + update_data so the next proprio extraction sees the locked positions."""
        self.env.set_joint_qpos({self._lock_freejoint_name: self._lock_free_qpos})
        self.env.set_joint_qvel({self._lock_freejoint_name: np.zeros(6, dtype=np.float32)})
        self.env.set_joint_qpos(self._lock_lower_qpos)
        self.env.set_joint_qvel(self._lock_lower_qvel)
        self.env.mj_forward()
        self.env.update_data()

    def _make_standing_ctrl(self):
        """Build a ctrl array that holds the robot in the TWIST2 standing pose."""
        ctrl = np.zeros(self.env.model.nu, dtype=np.float32)
        for i, act_id in enumerate(self.body_act_ids):
            ctrl[act_id] = DEFAULT_DOF_POS[i]
        for i, hid in enumerate(self.hand_act_ids):
            ctrl[hid] = self.hand_open_orca[i]
        return ctrl

    def _reset_scene(self):
        """Refresh the scene layout: reset simulation data to the initial frame,
        put the robot back to the TWIST2 standing pose, and rebuild the lower-body
        lock (if --fix_feet) so the welded root target matches the reset state."""
        self.env.reset_simulation()
        self.env.set_default_joint_values(self.env.default_joint_values)
        self.env.mj_forward()
        self.env.update_data()
        if self.fix_feet:
            self._build_lower_body_lock()
            self._apply_lower_body_lock()
        print("[Bridge] Scene layout refreshed (robot reset to standing pose)")


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="TWIST2 → OrcaLab RL Policy Bridge")
    parser.add_argument("--policy", default=None,
                        help="Path to TWIST2 ONNX policy file (not needed with --replay_target)")
    parser.add_argument("--device", default="cpu",
                        help="Device for policy inference (cuda/cpu)")
    parser.add_argument("--fix_feet", action="store_true",
                        help="Weld pelvis+feet to world and zero leg+torso actions (arm-only)")

    grp = parser.add_argument_group("Local XML mode")
    grp.add_argument("--local_xml", default=None,
                     help="Path to pre-cached XML (enables local mode, skips OrcaStudio)")
    grp.add_argument("--agent_name", default=None,
                     help="Robot agent name/prefix (auto-detected if not set, e.g. g1_pick)")

    grp2 = parser.add_argument_group("gRPC mode (OrcaStudio)")
    grp2.add_argument("--orcagym_addr", default="localhost:50051",
                      help="OrcaStudio gRPC address (default: localhost:50051)")

    grp3 = parser.add_argument_group("Simulation/Rendering")
    grp3.add_argument("--frame_skip", type=int, default=5,
                      help="Simulation sub-steps per control update (default: 5)")
    grp3.add_argument("--time_step", type=float, default=0.001,
                      help="MuJoCo timestep in seconds (default: 0.001)")
    grp3.add_argument("--no_render", action="store_true",
                      help="Disable rendering (always applied in local mode unless OrcaStudio is reachable)")
    grp3.add_argument("--no_pd_override", action="store_true",
                      help="Keep OrcaLab XML actuator PD gains (default: override "
                           "29 body actuators to TWIST2 training gains to kill "
                           "arm oscillation)")
    grp3.add_argument("--leg_pd_gain", type=float, default=1.0,
                      help="Scale leg (12 joints) kp/kv by this factor "
                           "(OrcaLab soft-contact compensation; sweep 1.0~2.0)")
    grp3.add_argument("--leg_ema_alpha", type=float, default=0.0,
                      help="EMA smoothing factor for leg (12 joints) PD targets "
                           "(0=off, 0.5~0.7 recommended)")

    grp4 = parser.add_argument_group("Redis")
    grp4.add_argument("--redis_host", default="localhost")
    grp4.add_argument("--redis_port", type=int, default=6379)

    grp5 = parser.add_argument_group("Teleop gating")
    grp5.add_argument("--start_button", default="RightController.key_two",
                      help="Joystick button (dotted path in controller_data) to cycle "
                           "IDLE→TELEOP→PAUSE→TELEOP (default: RightController.key_two; "
                           "key_one is teleop's own state button and must not be shared)")
    grp5.add_argument("--reset_button", default="LeftController.key_two",
                      help="Joystick button (dotted path in controller_data) to refresh the "
                           "scene layout: reset simulation + robot back to standing pose + "
                           "return to IDLE (default: LeftController.key_two)")
    grp5.add_argument("--record_button", default="RightController.axis_click",
                      help="Joystick button to toggle RECORD of the teleop input "
                           "stream (default: right stick press, unused by teleop)")
    grp5.add_argument("--replay_button", default="LeftController.axis_click",
                      help="Joystick button to toggle REPLAY of the recorded "
                           "stream (default: left stick press). NOTE: teleop also "
                           "uses left stick press as its emergency stop (pkill "
                           "sim2real.sh) — harmless when only bridge runs.")
    grp5.add_argument("--record_file", default="logs/bridge_record.json",
                      help="JSON path for the recorded session (save on record "
                           "stop, load on replay)")
    grp5.add_argument("--auto_teleop", action="store_true",
                      help="Start in TELEOP state immediately (skip button gating)")

    grp6 = parser.add_argument_group("XML welding / data filtering / diagnostics")
    grp6.add_argument("--patch_xml_inplace", action="store_true",
                      help="Write welds directly into the XML file (backup .bak) instead of a temp copy")
    grp6.add_argument("--no_hand_grasp", action="store_true",
                      help="Disable load-time hand grasp patch (finger kp/kv ×10, "
                           "screwdriver handle friction 0.2→1.0)")
    grp6.add_argument("--stale_ms", type=int, default=200,
                      help="Drop mimic_obs older than this many ms (accumulated/stale teleop frames)")
    grp6.add_argument("--verbose", action="store_true",
                      help="Print arm-link diagnostics every 500 policy calls "
                           "(mimic arm values/rate, pd_target range, inferred teleop state)")
    grp6.add_argument("--selftest", action="store_true",
                      help="Headless self-test: synthetic sine-arm mimic + synthetic button presses, "
                           "no Redis reads, auto-exit after 5s")

    grp7 = parser.add_argument_group("B1 twin (replay real policy target)")
    grp7.add_argument("--replay_target", action="store_true",
                      help="Twin mode: do NOT run the ONNX policy; drive OrcaLab with the "
                           "real robot's 29-dim policy target read from Redis "
                           "(published by sim2real --publish_target). Use --fix_feet for "
                           "a stable scene.")
    grp7.add_argument("--replay_gains", choices=["real", "train"], default="real",
                      help="Actuator PD gains in twin mode: 'real' = g1.yaml downlink "
                           "kp/kd (default, fair comparison) or 'train' = TWIST2 training gains")
    grp7.add_argument("--target_key", default="action_low_level_unitree_g1_with_hands",
                      help="Redis key holding the 29-dim real policy target")
    grp7.add_argument("--target_ts_key", default="t_action_low_level",
                      help="Redis key holding that target's timestamp (ms)")

    args = parser.parse_args()

    if args.replay_target:
        if args.policy:
            print("[Bridge] NOTE: --policy is ignored in --replay_target mode.")
    else:
        if not args.policy:
            print("Error: --policy is required unless --replay_target is set.")
            sys.exit(1)
        if not os.path.exists(args.policy):
            print(f"Error: Policy file not found: {args.policy}")
            sys.exit(1)

    if args.local_xml:
        args.local_xml = os.path.abspath(args.local_xml)
        if not os.path.exists(args.local_xml):
            print(f"Error: XML file not found: {args.local_xml}")
            sys.exit(1)

    bridge = Twist2OrcaLabBridge(
        policy_path=args.policy,
        device=args.device,
         fix_feet=args.fix_feet,
         pd_override=not args.no_pd_override,
         leg_pd_gain=args.leg_pd_gain,
         leg_ema_alpha=args.leg_ema_alpha,
         frame_skip=args.frame_skip,
         time_step=args.time_step,
         redis_host=args.redis_host,
         redis_port=args.redis_port,
         enable_render=not args.no_render,
         agent_name=args.agent_name,
         start_button=args.start_button,
         reset_button=args.reset_button,
         record_button=args.record_button,
         replay_button=args.replay_button,
         record_file=args.record_file,
         auto_teleop=args.auto_teleop,
         patch_xml_inplace=args.patch_xml_inplace,
         hand_grasp=not args.no_hand_grasp,
         stale_ms=args.stale_ms,
         verbose=args.verbose,
         selftest=args.selftest,
         local_xml=args.local_xml,
         orcagym_addr=args.orcagym_addr,
         replay_target=args.replay_target,
         replay_gains=args.replay_gains,
         target_key=args.target_key,
         target_ts_key=args.target_ts_key,
     )
    bridge.run()


if __name__ == "__main__":
    main()
