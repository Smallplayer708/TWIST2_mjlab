"""TWIST2 runner shim.

The base MJLab runner already provides the PPO training and export flow.
This subclass preserves the optional ``registry_name`` argument that MJLab's
training script passes for tracking tasks, and — when ``TWIST2_ENABLE_AMP=1`` —
builds the AMP discriminator/expert buffer and attaches it to both the env
(so ``rewards.amp_reward`` can query ``D(s, s')``) and the PPO algorithm (so
the discriminator is updated once per PPO iteration).
"""

from __future__ import annotations

from mjlab.rl import MjlabOnPolicyRunner

from twist2_mjlab.observations import get_motion_command
from twist2_mjlab.rl.amp import AmpState, amp_enabled


class Twist2OnPolicyRunner(MjlabOnPolicyRunner):
  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ) -> None:
    super().__init__(env, train_cfg, log_dir, device)
    self.registry_name = registry_name

    if amp_enabled():
      raw_env = self.env.unwrapped
      motion_lib = get_motion_command(raw_env, "motion").motion_lib
      self.amp = AmpState(motion_lib, device, float(raw_env.step_dt))
      raw_env.amp_discriminator = self.amp.disc
      raw_env.amp_normalizer = self.amp.normalizer
      raw_env._prev_amp_obs = None
      # Attach to the algorithm so ``Twist2PPO.update()`` trains the disc once
      # per PPO iteration from the rollout buffer (with reset masking).
      self.alg.amp = self.amp
