"""TWIST2 environment configuration for MJLab."""

from __future__ import annotations

import copy
import os
from dataclasses import replace

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp as tracking_mdp
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

import twist2_mjlab.observations as twist2_obs
import twist2_mjlab.rewards as twist2_rewards
import twist2_mjlab.terminations as twist2_terms

from twist2_mjlab.commands import PklMotionCommandCfg

_BASE_ANG_VEL_SCALE = 0.25
_JOINT_POS_SCALE = 1.0
_JOINT_VEL_SCALE = 0.05
_ANKLE_DOF_INDICES = (4, 5, 10, 11)
_JOINT_VEL_SCALE_WITH_ANKLE_MASK = tuple(
  0.0 if i in _ANKLE_DOF_INDICES else _JOINT_VEL_SCALE for i in range(29)
)

_TWIST2_BASE_MASS_RANGE = (-3.0, 3.0)
_TWIST2_MOTOR_STRENGTH_RANGE = (0.8, 1.2)

# Reward preset. ``upstream`` reproduces the original ZhaoLong0808/TWIST2_mjlab
# reward set (tracking weights 2.0/0.2/1.0/... and no stability rewards);
# ``tuned`` (default) is this fork's tuning. Resolved once at import time from
# ``TWIST2_REWARD_PRESET``.
_REWARD_PRESET = os.environ.get("TWIST2_REWARD_PRESET", "tuned").strip().lower()
_UPSTREAM_REWARDS = _REWARD_PRESET == "upstream"

# AMP (Adversarial Motion Priors) is off by default. When enabled it adds a
# buffer-only ``amp_style`` observation group and an ``amp_style`` reward term;
# the discriminator itself is built by the runner (see ``rl/runner.py``).
_AMP_ENABLED = os.environ.get("TWIST2_ENABLE_AMP", "").strip().lower() in (
  "1",
  "true",
  "yes",
  "on",
)
_AMP_WEIGHT = float(os.environ.get("TWIST2_AMP_WEIGHT", "0.3"))
_TWIST2_DEFAULT_NUM_ENVS = 4096


def _feet_ground_sensor_cfg(*, track_air_time: bool = False) -> ContactSensorCfg:
  return ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=track_air_time,
  )


def _twist2_tracking_reward_cfg() -> dict[str, RewardTermCfg]:
  joint_dof_weight = 2.0 if _UPSTREAM_REWARDS else 6.0
  joint_vel_weight = 0.2 if _UPSTREAM_REWARDS else 0.6
  root_weight = 1.0 if _UPSTREAM_REWARDS else 2.0
  keybody_weight = 2.0 if _UPSTREAM_REWARDS else 6.0
  return {
    "tracking_joint_dof": RewardTermCfg(
      func=twist2_rewards.tracking_joint_dof,
      weight=joint_dof_weight,
      params={"command_name": "motion"},
    ),
    "tracking_joint_vel": RewardTermCfg(
      func=twist2_rewards.tracking_joint_vel,
      weight=joint_vel_weight,
      params={"command_name": "motion"},
    ),
    "tracking_root_translation_z": RewardTermCfg(
      func=twist2_rewards.tracking_root_translation_z,
      weight=root_weight,
      params={"command_name": "motion"},
    ),
    "tracking_root_rotation": RewardTermCfg(
      func=twist2_rewards.tracking_root_rotation,
      weight=root_weight,
      params={"command_name": "motion"},
    ),
    "tracking_root_linear_vel": RewardTermCfg(
      func=twist2_rewards.tracking_root_linear_vel,
      weight=root_weight,
      params={"command_name": "motion"},
    ),
    "tracking_root_angular_vel": RewardTermCfg(
      func=twist2_rewards.tracking_root_angular_vel,
      weight=root_weight,
      params={"command_name": "motion"},
    ),
    "tracking_keybody_pos": RewardTermCfg(
      func=twist2_rewards.tracking_keybody_pos,
      weight=keybody_weight,
      params={"command_name": "motion"},
    ),
    "tracking_keybody_pos_global": RewardTermCfg(
      func=twist2_rewards.tracking_keybody_pos_global,
      weight=keybody_weight,
      params={"command_name": "motion"},
    ),
  }


def _twist2_regularization_reward_cfg() -> dict[str, RewardTermCfg]:
  all_joints = SceneEntityCfg("robot", joint_names=(".*",))
  rewards = {
    "alive": RewardTermCfg(func=env_mdp.is_alive, weight=0.5),
    "feet_slip": RewardTermCfg(
      func=twist2_rewards.feet_slip,
      weight=-0.1,
      params={"sensor_name": "feet_ground_contact", "command_name": "motion"},
    ),
    "feet_contact_forces": RewardTermCfg(
      func=twist2_rewards.feet_contact_forces,
      weight=-5e-4,
      params={"sensor_name": "feet_ground_contact", "max_contact_force": 500.0},
    ),
    "feet_stumble": RewardTermCfg(
      func=twist2_rewards.feet_stumble,
      weight=-1.25,
      params={"sensor_name": "feet_ground_contact", "command_name": "motion"},
    ),
    "dof_pos_limits": RewardTermCfg(
      func=env_mdp.joint_pos_limits,
      weight=-5.0,
      params={"asset_cfg": all_joints},
    ),
    "dof_torque_limits": RewardTermCfg(
      func=twist2_rewards.dof_torque_limits,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", actuator_names=[".*"]), "soft_torque_limit": 0.95},
    ),
    "dof_vel": RewardTermCfg(
      func=env_mdp.joint_vel_l2,
      weight=-1e-4,
      params={"asset_cfg": all_joints},
    ),
    "dof_acc": RewardTermCfg(
      func=env_mdp.joint_acc_l2,
      weight=-5e-8,
      params={"asset_cfg": all_joints},
    ),
    "action_rate_l2": RewardTermCfg(func=env_mdp.action_rate_l2, weight=-1e-1),
    "joint_limit": RewardTermCfg(
      func=env_mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": all_joints},
    ),
    "self_collisions": RewardTermCfg(
      func=tracking_mdp.self_collision_cost,
      weight=-10.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
    "feet_air_time": RewardTermCfg(
      func=twist2_rewards.feet_air_time,
      weight=5.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "motion",
        "feet_air_time_target": 0.5,
      },
    ),
    "ang_vel_xy": RewardTermCfg(
      func=twist2_rewards.ang_vel_xy,
      weight=-0.01,
    ),
    "ankle_dof_acc": RewardTermCfg(
      func=twist2_rewards.ankle_dof_acc,
      weight=-1e-7,
      params={"asset_cfg": all_joints},
    ),
    "ankle_dof_vel": RewardTermCfg(
      func=twist2_rewards.ankle_dof_vel,
      weight=-2e-4,
      params={"asset_cfg": all_joints},
    ),
  }

  if not _UPSTREAM_REWARDS:
    # ------------------------------------------------------------------
    # Stability rewards (adapted from IHMC IsaacLab). Not part of the
    # original ZhaoLong0808/TWIST2_mjlab reward set.
    # ------------------------------------------------------------------
    rewards.update(
      {
        "com_in_support_polygon": RewardTermCfg(
          func=twist2_rewards.simplified_com_in_support_polygon,
          weight=0.1,
          params={"feet_sensor_cfg": SceneEntityCfg("feet_ground_contact")},
        ),
        "capture_point_in_support_polygon": RewardTermCfg(
          func=twist2_rewards.simplified_capture_point_in_support_polygon,
          weight=0.1,
          params={"feet_sensor_cfg": SceneEntityCfg("feet_ground_contact")},
        ),
        "ankle_hip_step": RewardTermCfg(
          func=twist2_rewards.AnkleHipStepReward,
          weight=0.05,
          params={"feet_sensor_cfg": SceneEntityCfg("feet_ground_contact")},
        ),
        "linear_momentum_change": RewardTermCfg(
          func=twist2_rewards.LinearMomentumChangePenalty,
          weight=1e-6,
          params={},
        ),
        "angular_momentum_change": RewardTermCfg(
          func=twist2_rewards.AngularMomentumChangePenalty,
          weight=1e-5,
          params={},
        ),
      }
    )

  if _AMP_ENABLED:
    rewards["amp_style"] = RewardTermCfg(
      func=twist2_rewards.amp_reward,
      weight=_AMP_WEIGHT,
      params={"command_name": "motion"},
    )

  return rewards


def _twist2_reward_cfg() -> dict[str, RewardTermCfg]:
  rewards = _twist2_tracking_reward_cfg()
  rewards.update(_twist2_regularization_reward_cfg())
  return rewards


def _twist2_termination_cfg(*, play: bool = False) -> dict[str, TerminationTermCfg]:
  return {
    "time_out": TerminationTermCfg(func=env_mdp.time_out, time_out=True),
    "motion_end": TerminationTermCfg(
      func=twist2_terms.twist2_motion_end,
      params={"command_name": "motion"},
      time_out=True,
    ),
    "root_height_diff": TerminationTermCfg(
      func=twist2_terms.twist2_root_height_diff,
      params={"command_name": "motion", "threshold": 0.3},
    ),
    "roll_limit": TerminationTermCfg(
      func=twist2_terms.twist2_roll_limit,
      params={"threshold": 4.0},
    ),
    "pitch_limit": TerminationTermCfg(
      func=twist2_terms.twist2_pitch_limit,
      params={"threshold": 4.0},
    ),
    "velocity_too_large": TerminationTermCfg(
      func=twist2_terms.twist2_velocity_too_large,
      params={"threshold": 5.0},
    ),
    "pose_fail": TerminationTermCfg(
      func=twist2_terms.twist2_pose_fail,
      params={
        "command_name": "motion",
        "threshold": 0.7,
        "track_root": False,
        "root_tracking_threshold": 2.0,
      },
    ),
  }


def _apply_twist2_domain_rand(cfg: ManagerBasedRlEnvCfg, *, play: bool = False) -> None:
  if play:
    return

  robot_cfg = copy.deepcopy(cfg.scene.entities["robot"])
  assert robot_cfg.articulation is not None

  delayed_actuators = []
  for actuator_cfg in robot_cfg.articulation.actuators:
    if isinstance(actuator_cfg, BuiltinPositionActuatorCfg):
      delayed_actuators.append(
        replace(
          actuator_cfg,
          delay_min_lag=0,
          delay_max_lag=cfg.decimation,
          delay_hold_prob=0.0,
          delay_update_period=cfg.decimation,
          delay_per_env_phase=True,
        )
      )
    else:
      delayed_actuators.append(copy.deepcopy(actuator_cfg))

  robot_cfg.articulation.actuators = tuple(delayed_actuators)
  cfg.scene.entities["robot"] = robot_cfg

  cfg.events["twist2_base_mass"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      "operation": "add",
      "ranges": _TWIST2_BASE_MASS_RANGE,
    },
  )
  cfg.events["twist2_motor_strength"] = EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot", actuator_names=[".*"]),
      "kp_range": _TWIST2_MOTOR_STRENGTH_RANGE,
      "kd_range": _TWIST2_MOTOR_STRENGTH_RANGE,
      "operation": "scale",
    },
  )


def _disable_play_randomization(cfg: ManagerBasedRlEnvCfg) -> None:
  for event_name in ("base_com", "encoder_bias", "foot_friction", "push_robot"):
    cfg.events.pop(event_name, None)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.joint_position_range = (0.0, 0.0)


def _set_default_num_envs(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
  if not play:
    cfg.scene.num_envs = _TWIST2_DEFAULT_NUM_ENVS


def _make_twist2_actor_terms(*, enable_noise: bool) -> dict[str, ObservationTermCfg]:
  return {
    "motion_root_vel_xy_b": ObservationTermCfg(
      func=twist2_obs.motion_root_vel_xy_b,
      params={"command_name": "motion"},
    ),
    "motion_root_z": ObservationTermCfg(
      func=twist2_obs.motion_root_z,
      params={"command_name": "motion"},
    ),
    "motion_root_roll_pitch": ObservationTermCfg(
      func=twist2_obs.motion_root_roll_pitch,
      params={"command_name": "motion"},
    ),
    "motion_root_yaw_ang_vel_b": ObservationTermCfg(
      func=twist2_obs.motion_root_yaw_ang_vel_b,
      params={"command_name": "motion"},
    ),
    "motion_joint_pos": ObservationTermCfg(
      func=twist2_obs.motion_joint_pos,
      params={"command_name": "motion"},
    ),
    "base_ang_vel": ObservationTermCfg(
      func=env_mdp.base_ang_vel,
      scale=_BASE_ANG_VEL_SCALE,
      noise=Unoise(n_min=-0.1, n_max=0.1) if enable_noise else None,
    ),
    "imu_roll_pitch": ObservationTermCfg(
      func=twist2_obs.imu_roll_pitch,
      noise=Unoise(n_min=-0.1, n_max=0.1) if enable_noise else None,
    ),
    "joint_pos": ObservationTermCfg(
      func=env_mdp.joint_pos_rel,
      params={"biased": True},
      scale=_JOINT_POS_SCALE,
      noise=Unoise(n_min=-0.01, n_max=0.01) if enable_noise else None,
    ),
    "joint_vel": ObservationTermCfg(
      func=env_mdp.joint_vel_rel,
      scale=_JOINT_VEL_SCALE_WITH_ANKLE_MASK,
      noise=Unoise(n_min=-0.1, n_max=0.1) if enable_noise else None,
    ),
    "actions": ObservationTermCfg(func=env_mdp.last_action),
  }


def _make_twist2_critic_current_terms() -> dict[str, ObservationTermCfg]:
  return {
    "base_ang_vel": ObservationTermCfg(
      func=env_mdp.base_ang_vel,
      scale=_BASE_ANG_VEL_SCALE,
    ),
    "imu_roll_pitch": ObservationTermCfg(func=twist2_obs.imu_roll_pitch),
    "joint_pos": ObservationTermCfg(
      func=env_mdp.joint_pos_rel,
      scale=_JOINT_POS_SCALE,
    ),
    "joint_vel": ObservationTermCfg(
      func=env_mdp.joint_vel_rel,
      scale=_JOINT_VEL_SCALE_WITH_ANKLE_MASK,
    ),
    "actions": ObservationTermCfg(func=env_mdp.last_action),
  }


def _make_twist2_critic_extras_terms() -> dict[str, ObservationTermCfg]:
  return {
    "base_lin_vel": ObservationTermCfg(func=env_mdp.base_lin_vel),
    "root_pos_w": ObservationTermCfg(
      func=twist2_obs.critic_root_pos_w,
      params={"command_name": "motion"},
    ),
    "root_quat_w": ObservationTermCfg(
      func=twist2_obs.critic_root_quat_w,
      params={"command_name": "motion"},
    ),
    "key_body_pos_b": ObservationTermCfg(
      func=twist2_obs.critic_key_body_pos_b,
      params={"command_name": "motion"},
    ),
    "foot_contact": ObservationTermCfg(
      func=twist2_obs.critic_foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "base_com_offset": ObservationTermCfg(
      func=twist2_obs.critic_base_com_offset,
      params={"command_name": "motion"},
    ),
    "foot_friction": ObservationTermCfg(
      func=twist2_obs.critic_foot_friction,
      params={"command_name": "motion"},
    ),
    "added_mass": ObservationTermCfg(
      func=twist2_obs.critic_added_mass,
      params={"command_name": "motion"},
    ),
    "motor_scales": ObservationTermCfg(
      func=twist2_obs.critic_motor_scales,
      params={"command_name": "motion"},
    ),
    "encoder_bias": ObservationTermCfg(
      func=twist2_obs.critic_encoder_bias,
      params={"command_name": "motion"},
    ),
  }


def unitree_g1_pkl_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)

  old_cmd = cfg.commands["motion"]
  assert isinstance(old_cmd, MotionCommandCfg)

  cfg.commands["motion"] = PklMotionCommandCfg(
    entity_name=old_cmd.entity_name,
    resampling_time_range=old_cmd.resampling_time_range,
    debug_vis=old_cmd.debug_vis,
    pose_range=old_cmd.pose_range,
    velocity_range=old_cmd.velocity_range,
    joint_position_range=old_cmd.joint_position_range,
    motion_file=old_cmd.motion_file,
    anchor_body_name=old_cmd.anchor_body_name,
    body_names=old_cmd.body_names,
    adaptive_kernel_size=old_cmd.adaptive_kernel_size,
    adaptive_lambda=old_cmd.adaptive_lambda,
    adaptive_uniform_ratio=old_cmd.adaptive_uniform_ratio,
    adaptive_alpha=old_cmd.adaptive_alpha,
    sampling_mode=old_cmd.sampling_mode,
    offload_to_cpu=False,
  )

  if play:
    _disable_play_randomization(cfg)

  _set_default_num_envs(cfg, play=play)
  return cfg


def _aux_observations_enabled(enable_aux: bool | None) -> bool:
  """Resolve whether the auxiliary observation groups are built.

  Off by default so baseline training keeps its original per-step cost and
  rollout-buffer memory. Opt in with ``enable_aux=True`` or the
  ``TWIST2_ENABLE_AUX=1`` environment variable (mirroring the ``TWIST2_*``
  launcher convention). The groups are only needed when
  ``--agent.algorithm.aux-coef`` is greater than zero.
  """
  if enable_aux is not None:
    return enable_aux
  return os.environ.get("TWIST2_ENABLE_AUX", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
  )


def unitree_g1_pkl_tracking_custom_ppo_env_cfg(
  play: bool = False,
  enable_aux: bool | None = None,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_pkl_tracking_env_cfg(play=play)

  feet_ground_cfg = _feet_ground_sensor_cfg()
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg,)

  cfg.observations = {
    "actor_current": ObservationGroupCfg(
      terms=_make_twist2_actor_terms(enable_noise=not play),
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "actor_history": ObservationGroupCfg(
      terms=_make_twist2_actor_terms(enable_noise=not play),
      concatenate_terms=True,
      enable_corruption=not play,
      history_length=twist2_obs.ACTOR_HISTORY_LENGTH,
      flatten_history_dim=False,
    ),
    "critic_priv_future_sequence": ObservationGroupCfg(
      terms={
        "privileged_future": ObservationTermCfg(
          func=twist2_obs.privileged_future_sequence,
          params={
            "command_name": "motion",
            "step_offsets": twist2_obs.PRIV_FUTURE_STEP_OFFSETS,
          },
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "critic_current": ObservationGroupCfg(
      terms=_make_twist2_critic_current_terms(),
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "critic_extras": ObservationGroupCfg(
      terms=_make_twist2_critic_extras_terms(),
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  if _aux_observations_enabled(enable_aux):
    # Stored in the rollout buffer but never consumed by actor/critic, so these
    # groups do not change policy input dimensions. They feed the differentiable
    # closed-loop auxiliary objective in `Twist2PPO`.
    cfg.observations["aux_state"] = ObservationGroupCfg(
      terms={
        "state": ObservationTermCfg(
          func=twist2_obs.aux_privileged_state,
          params={"command_name": "motion"},
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
    cfg.observations["aux_ref_future"] = ObservationGroupCfg(
      terms={
        "future": ObservationTermCfg(
          func=twist2_obs.aux_reference_future,
          params={
            "command_name": "motion",
            "step_offsets": twist2_obs.AUX_REF_STEP_OFFSETS,
          },
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )

  if _AMP_ENABLED:
    # Buffer-only style feature for the AMP discriminator; never fed to
    # actor/critic, so policy input dimensions are unchanged.
    cfg.observations["amp_style"] = ObservationGroupCfg(
      terms={
        "state": ObservationTermCfg(
          func=twist2_obs.amp_style_state,
          params={"command_name": "motion"},
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )

  _set_default_num_envs(cfg, play=play)
  return cfg


def unitree_g1_twist2_flat_env_cfg(
  play: bool = False,
  enable_aux: bool | None = None,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_pkl_tracking_custom_ppo_env_cfg(play=play, enable_aux=enable_aux)

  cfg.sim.njmax = 400

  sensors = []
  for sensor_cfg in cfg.scene.sensors or ():
    if sensor_cfg.name == "feet_ground_contact":
      sensors.append(_feet_ground_sensor_cfg(track_air_time=True))
    else:
      sensors.append(sensor_cfg)
  cfg.scene.sensors = tuple(sensors)

  _apply_twist2_domain_rand(cfg, play=play)
  cfg.rewards = _twist2_reward_cfg()
  cfg.terminations = _twist2_termination_cfg(play=play)
  _set_default_num_envs(cfg, play=play)
  return cfg


