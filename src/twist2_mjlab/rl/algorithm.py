"""TWIST2 PPO with a differentiable closed-loop auxiliary objective.

This is a drop-in replacement for ``rsl_rl.algorithms.PPO`` that keeps the PPO
update intact and additionally optimizes a differentiable auxiliary objective
``aux_coef * L_aux``. ``L_aux`` closes the loop over a short horizon: the
current policy's (mean) action is rolled through a differentiable model and
penalized against future reference-motion targets. Gradients therefore flow
``action -> future state -> future tracking error`` into the actor, giving
temporal credit assignment that PPO's one-step surrogate lacks.

Two differentiable substrates are supported (see the plan):

* ``aux_mode="world_model"``: a learned ``PrivilegedDynamicsModel`` MLP trained
  online by supervised MSE. It never touches contact forces, so the auxiliary
  gradient path is smooth by construction.
* ``aux_mode="analytic"``: a first-order position-actuator surrogate using the
  action scale inferred from the environment. Cheapest MVP / fallback.

The auxiliary observations (``aux_state``, ``aux_ref_future``) are stored in the
rollout buffer but are never consumed by the actor or critic, so policy input
dimensions and checkpoint compatibility are unchanged when ``aux_coef=0``.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.algorithms import PPO

from twist2_mjlab.observations import (
  AUX_HORIZON,
  AUX_JOINT_POS,
  AUX_JOINT_VEL,
  AUX_KEY_BODY,
  AUX_REF_JOINT_POS,
  AUX_REF_JOINT_VEL,
  AUX_REF_KEY_BODY,
  AUX_REF_ROOT_POS,
  AUX_REF_ROOT_RPY,
  AUX_REF_STEP_DIM,
  AUX_ROOT_POS,
  AUX_ROOT_RPY,
  AUX_STATE_KEY,
  AUX_REF_KEY,
)
from twist2_mjlab.rl.world_model import PrivilegedDynamicsModel


def _infer_action_scale(env) -> torch.Tensor | float | None:
  """Best-effort extraction of the joint position action scale from the env."""
  try:
    action_manager = env.unwrapped.action_manager
    term = action_manager.get_term("joint_pos")
    scale = term.scale
  except Exception:  # pragma: no cover - defensive; env layout may vary.
    return None
  if isinstance(scale, torch.Tensor):
    return scale[0].detach().clone().float()
  return float(scale)


class Twist2PPO(PPO):
  """PPO + differentiable closed-loop auxiliary objective."""

  def __init__(
    self,
    actor,
    critic,
    storage,
    *,
    aux_mode: str = "world_model",
    aux_coef: float = 0.0,
    aux_start_iter: int = 0,
    aux_coef_warmup_iters: int = 0,
    aux_horizon: int = AUX_HORIZON,
    aux_state_key: str = AUX_STATE_KEY,
    aux_ref_key: str = AUX_REF_KEY,
    aux_model_hidden: tuple[int, ...] | list[int] = (256, 256),
    aux_model_activation: str = "elu",
    aux_model_lr: float = 3e-4,
    aux_world_model_batch: int = 16384,
    aux_joint_pos_weight: float = 1.0,
    aux_joint_vel_weight: float = 0.1,
    aux_root_pos_weight: float = 1.0,
    aux_root_rpy_weight: float = 1.0,
    aux_key_body_weight: float = 1.0,
    aux_action_scale: float | torch.Tensor = 0.0,
    aux_first_order_alpha: float = 0.5,
    aux_step_dt: float = 0.0,
    **kwargs,
  ) -> None:
    super().__init__(actor, critic, storage, **kwargs)

    self.aux_mode = aux_mode
    self.aux_coef = float(aux_coef)
    self.aux_start_iter = int(aux_start_iter)
    self.aux_coef_warmup_iters = int(aux_coef_warmup_iters)
    self.aux_horizon = int(aux_horizon)
    self.aux_state_key = aux_state_key
    self.aux_ref_key = aux_ref_key
    self.aux_world_model_batch = int(aux_world_model_batch)
    self.aux_first_order_alpha = float(aux_first_order_alpha)
    self.aux_step_dt = float(aux_step_dt)
    self.aux_loss_weights = {
      "joint_pos": float(aux_joint_pos_weight),
      "joint_vel": float(aux_joint_vel_weight),
      "root_pos": float(aux_root_pos_weight),
      "root_rpy": float(aux_root_rpy_weight),
      "key_body": float(aux_key_body_weight),
    }

    obs = self.storage.observations
    missing_aux_groups = (
      self.aux_state_key not in obs or self.aux_ref_key not in obs
    )
    if missing_aux_groups:
      # Only complain when the auxiliary objective was actually requested;
      # otherwise stay silently inactive so default runs are unaffected.
      if self.aux_coef > 0.0 and self.aux_mode != "none":
        warnings.warn(
          f"Auxiliary observation groups '{self.aux_state_key}' / "
          f"'{self.aux_ref_key}' are missing from the environment observations; "
          "disabling the auxiliary objective. Enable them with "
          "`TWIST2_ENABLE_AUX=1` (or `enable_aux=True` in the env config) when "
          "using `--agent.algorithm.aux-coef > 0`.",
          stacklevel=2,
        )
      self.aux_mode = "none"

    self.aux_state_dim = int(obs[self.aux_state_key].shape[-1]) if self.aux_mode != "none" else 0
    self.aux_ref_step_dim = (
      int(obs[self.aux_ref_key].shape[-1]) if self.aux_mode != "none" else AUX_REF_STEP_DIM
    )
    self.aux_action_dim = int(self.storage.actions.shape[-1])

    if isinstance(aux_action_scale, torch.Tensor):
      # Twist2PPO is not an nn.Module, so keep the per-joint scale as a plain
      # tensor attribute (moved to the algorithm device explicitly).
      self._aux_action_scale_t = (
        aux_action_scale.detach().to(self.device).float().reshape(1, -1)
      )
      self.aux_action_scale = 0.0
    else:
      self._aux_action_scale_t = None
      self.aux_action_scale = float(aux_action_scale)

    self.world_model: PrivilegedDynamicsModel | None = None
    self.wm_optimizer: torch.optim.Optimizer | None = None
    if self.aux_mode == "world_model":
      self.world_model = PrivilegedDynamicsModel(
        state_dim=self.aux_state_dim,
        action_dim=self.aux_action_dim,
        ref_dim=self.aux_ref_step_dim,
        hidden_dims=aux_model_hidden,
        activation=aux_model_activation,
      ).to(self.device)
      self.wm_optimizer = torch.optim.Adam(
        self.world_model.parameters(), lr=aux_model_lr
      )

    self._update_count = 0

  # ------------------------------------------------------------------
  # Auxiliary objective helpers.
  # ------------------------------------------------------------------

  def _aux_active(self) -> bool:
    return self.aux_mode != "none" and self.aux_coef > 0.0

  def _current_aux_coef(self) -> float:
    if self._update_count < self.aux_start_iter:
      return 0.0
    if self.aux_coef_warmup_iters <= 0:
      return self.aux_coef
    progress = (self._update_count - self.aux_start_iter + 1) / float(
      self.aux_coef_warmup_iters
    )
    return self.aux_coef * min(1.0, max(0.0, progress))

  def _action_scale(self) -> torch.Tensor | float:
    if self._aux_action_scale_t is not None:
      return self._aux_action_scale_t
    return self.aux_action_scale

  def _tracking_loss(
    self, pred_state: torch.Tensor, ref_state: torch.Tensor
  ) -> torch.Tensor:
    w = self.aux_loss_weights
    loss = (
      w["joint_pos"]
      * F.mse_loss(pred_state[:, AUX_JOINT_POS], ref_state[:, AUX_REF_JOINT_POS])
      + w["joint_vel"]
      * F.mse_loss(pred_state[:, AUX_JOINT_VEL], ref_state[:, AUX_REF_JOINT_VEL])
      + w["root_pos"]
      * F.mse_loss(pred_state[:, AUX_ROOT_POS], ref_state[:, AUX_REF_ROOT_POS])
      + w["root_rpy"]
      * F.mse_loss(pred_state[:, AUX_ROOT_RPY], ref_state[:, AUX_REF_ROOT_RPY])
      + w["key_body"]
      * F.mse_loss(pred_state[:, AUX_KEY_BODY], ref_state[:, AUX_REF_KEY_BODY])
    )
    return loss

  def _analytic_aux_loss(
    self,
    state: torch.Tensor,
    mean_action: torch.Tensor,
    ref: torch.Tensor,
  ) -> torch.Tensor:
    scale = self._action_scale()
    alpha = self.aux_first_order_alpha
    # `aux_step_dt <= 0` means the control step is unknown; drop the velocity
    # term rather than dividing by a near-zero step and exploding the loss.
    dt = self.aux_step_dt
    has_dt = dt > 0.0

    q = state[:, AUX_JOINT_POS]
    w = self.aux_loss_weights
    losses: list[torch.Tensor] = []
    for h in range(1, self.aux_horizon + 1):
      delta = alpha * scale * mean_action
      q = q + delta
      target = ref[:, h, :]
      step_loss = w["joint_pos"] * F.mse_loss(q, target[:, AUX_REF_JOINT_POS])
      if has_dt:
        v = delta / dt
        step_loss = step_loss + w["joint_vel"] * F.mse_loss(
          v, target[:, AUX_REF_JOINT_VEL]
        )
      losses.append(step_loss)
    return torch.stack(losses).mean()

  def _world_model_aux_loss(
    self,
    state: torch.Tensor,
    mean_action: torch.Tensor,
    ref: torch.Tensor,
  ) -> torch.Tensor:
    assert self.world_model is not None
    # Freeze the world model for the actor backward pass: only the action path
    # should carry gradient. The graph is built while requires_grad is False, so
    # restoring afterwards is safe.
    for param in self.world_model.parameters():
      param.requires_grad_(False)
    try:
      s = state
      losses: list[torch.Tensor] = []
      for h in range(1, self.aux_horizon + 1):
        ref_in = ref[:, h - 1, :]
        s = self.world_model(s, mean_action, ref_in)
        losses.append(self._tracking_loss(s, ref[:, h, :]))
      return torch.stack(losses).mean()
    finally:
      for param in self.world_model.parameters():
        param.requires_grad_(True)

  def _compute_aux_loss(self, batch) -> torch.Tensor | None:
    obs = batch.observations
    state = obs[self.aux_state_key].detach()
    ref = obs[self.aux_ref_key].detach()
    if self.actor.distribution is None:
      return None
    mean_action = self.actor.output_mean
    if self.aux_mode == "analytic":
      return self._analytic_aux_loss(state, mean_action, ref)
    if self.aux_mode == "world_model":
      return self._world_model_aux_loss(state, mean_action, ref)
    return None

  def _train_world_model(self) -> float | None:
    if self.world_model is None or self.wm_optimizer is None or not self._aux_active():
      return None
    obs = self.storage.observations
    state = obs[self.aux_state_key]
    ref = obs[self.aux_ref_key]
    actions = self.storage.actions
    dones = self.storage.dones
    num_steps = state.shape[0]
    if num_steps < 2:
      return None

    s_t = state[:-1].reshape(-1, self.aux_state_dim)
    s_next = state[1:].reshape(-1, self.aux_state_dim)
    a_t = actions[:-1].reshape(-1, self.aux_action_dim)
    ref_t = ref[:-1, :, 0, :].reshape(-1, self.aux_ref_step_dim)
    valid = (dones[:-1].reshape(-1) < 0.5).nonzero(as_tuple=False).squeeze(-1)
    if valid.numel() == 0:
      return None
    if valid.numel() > self.aux_world_model_batch:
      sample = torch.randint(
        0, valid.numel(), (self.aux_world_model_batch,), device=valid.device
      )
      valid = valid[sample]

    pred = self.world_model(s_t[valid], a_t[valid], ref_t[valid])
    loss = F.mse_loss(pred, s_next[valid])
    self.wm_optimizer.zero_grad()
    loss.backward()
    self.wm_optimizer.step()
    return float(loss.item())

  # ------------------------------------------------------------------
  # PPO update (mirrors rsl_rl 5.x PPO.update with auxiliary injection).
  # ------------------------------------------------------------------

  def update(self) -> dict[str, float]:
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_aux_loss = 0.0
    aux_updates = 0
    mean_rnd_loss = 0 if self.rnd else None
    mean_symmetry_loss = 0 if self.symmetry else None
    wm_loss = None

    if self._aux_active() and self.aux_mode == "world_model":
      wm_loss = self._train_world_model()

    if self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )

    for batch in generator:
      original_batch_size = batch.observations.batch_size[0]

      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (batch.advantages - batch.advantages.mean()) / (
            batch.advantages.std() + 1e-8
          )

      if self.symmetry and self.symmetry["use_data_augmentation"]:
        data_augmentation_func = self.symmetry["data_augmentation_func"]
        batch.observations, batch.actions = data_augmentation_func(
          env=self.symmetry["_env"],
          obs=batch.observations,
          actions=batch.actions,
        )
        num_aug = int(batch.observations.batch_size[0] / original_batch_size)
        batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
        batch.values = batch.values.repeat(num_aug, 1)
        batch.advantages = batch.advantages.repeat(num_aug, 1)
        batch.returns = batch.returns.repeat(num_aug, 1)

      self.actor(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
      )
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[1],
      )
      distribution_params = tuple(
        p[:original_batch_size] for p in self.actor.output_distribution_params
      )
      entropy = self.actor.output_entropy[:original_batch_size]

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = self.actor.get_kl_divergence(
            batch.old_distribution_params, distribution_params
          )
          kl_mean = torch.mean(kl)
          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size
          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()
          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
      surrogate = -torch.squeeze(batch.advantages) * ratio
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (values - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy.mean()
      )

      if self._aux_active():
        aux_loss = self._compute_aux_loss(batch)
        if aux_loss is not None:
          loss = loss + self._current_aux_coef() * aux_loss
          mean_aux_loss += float(aux_loss.detach().item())
          aux_updates += 1

      if self.symmetry:
        if not self.symmetry["use_data_augmentation"]:
          data_augmentation_func = self.symmetry["data_augmentation_func"]
          batch.observations, _ = data_augmentation_func(
            obs=batch.observations, actions=None, env=self.symmetry["_env"]
          )
        mean_actions = self.actor(batch.observations.detach().clone())
        action_mean_orig = mean_actions[:original_batch_size]
        _, actions_mean_symm = data_augmentation_func(
          obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
        )
        mse_loss = nn.MSELoss()
        symmetry_loss = mse_loss(
          mean_actions[original_batch_size:],
          actions_mean_symm.detach()[original_batch_size:],
        )
        if self.symmetry["use_mirror_loss"]:
          loss = loss + self.symmetry["mirror_loss_coeff"] * symmetry_loss
        else:
          symmetry_loss = symmetry_loss.detach()

      if self.rnd:
        with torch.no_grad():
          rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])
          rnd_state = self.rnd.state_normalizer(rnd_state)
        predicted_embedding = self.rnd.predictor(rnd_state)
        target_embedding = self.rnd.target(rnd_state).detach()
        mse_loss = nn.MSELoss()
        rnd_loss = mse_loss(predicted_embedding, target_embedding)

      self.optimizer.zero_grad()
      loss.backward()
      if self.rnd:
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()

      if self.is_multi_gpu:
        self.reduce_parameters()

      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()
      if self.rnd_optimizer:
        self.rnd_optimizer.step()

      mean_value_loss += float(value_loss.item())
      mean_surrogate_loss += float(surrogate_loss.item())
      mean_entropy += float(entropy.mean().item())
      if mean_rnd_loss is not None:
        mean_rnd_loss += float(rnd_loss.item())
      if mean_symmetry_loss is not None:
        mean_symmetry_loss += float(symmetry_loss.item())

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    if mean_rnd_loss is not None:
      mean_rnd_loss /= num_updates
    if mean_symmetry_loss is not None:
      mean_symmetry_loss /= num_updates
    if aux_updates > 0:
      mean_aux_loss /= aux_updates

    # AMP: update the discriminator from this rollout's policy transitions
    # (reset-crossing transitions are masked inside ``AmpState.update``).
    amp_stats: dict[str, float] = {}
    if getattr(self, "amp", None) is not None:
      amp_stats = self.amp.update(self.storage)

    self.storage.clear()
    self._update_count += 1

    loss_dict = {
      "value": mean_value_loss,
      "surrogate": mean_surrogate_loss,
      "entropy": mean_entropy,
    }
    if self.rnd:
      loss_dict["rnd"] = mean_rnd_loss
    if self.symmetry:
      loss_dict["symmetry"] = mean_symmetry_loss
    if aux_updates > 0:
      loss_dict["aux"] = mean_aux_loss
      loss_dict["aux_coef"] = self._current_aux_coef()
    if wm_loss is not None:
      loss_dict["world_model"] = wm_loss
    loss_dict.update(amp_stats)
    return loss_dict

  def train_mode(self) -> None:
    super().train_mode()
    if self.world_model is not None:
      self.world_model.train()

  def eval_mode(self) -> None:
    super().eval_mode()
    if self.world_model is not None:
      self.world_model.eval()

  def save(self) -> dict:
    saved = super().save()
    if self.world_model is not None and self.wm_optimizer is not None:
      saved["world_model_state_dict"] = self.world_model.state_dict()
      saved["wm_optimizer_state_dict"] = self.wm_optimizer.state_dict()
    amp = getattr(self, "amp", None)
    if amp is not None:
      saved["amp_disc_state_dict"] = amp.disc.state_dict()
      saved["amp_normalizer_state"] = {
        "mean": amp.normalizer.mean,
        "var": amp.normalizer.var,
        "count": amp.normalizer.count,
      }
    return saved

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    load_iteration = super().load(loaded_dict, load_cfg, strict)
    if self.world_model is not None and "world_model_state_dict" in loaded_dict:
      self.world_model.load_state_dict(
        loaded_dict["world_model_state_dict"], strict=strict
      )
      if self.wm_optimizer is not None and "wm_optimizer_state_dict" in loaded_dict:
        self.wm_optimizer.load_state_dict(loaded_dict["wm_optimizer_state_dict"])
    amp = getattr(self, "amp", None)
    if amp is not None and "amp_disc_state_dict" in loaded_dict:
      amp.disc.load_state_dict(loaded_dict["amp_disc_state_dict"], strict=strict)
      if "amp_normalizer_state" in loaded_dict:
        state = loaded_dict["amp_normalizer_state"]
        amp.normalizer.mean = state["mean"].to(self.device)
        amp.normalizer.var = state["var"].to(self.device)
        amp.normalizer.count = float(state["count"])
    return load_iteration

  @staticmethod
  def construct_algorithm(obs, env, cfg, device):
    alg_cfg = cfg["algorithm"]
    scale_cfg = alg_cfg.get("aux_action_scale", 0.0)
    if not isinstance(scale_cfg, torch.Tensor) and float(scale_cfg or 0.0) <= 0.0:
      scale = _infer_action_scale(env)
      if scale is not None:
        alg_cfg["aux_action_scale"] = scale
    if not alg_cfg.get("aux_step_dt"):
      step_dt = getattr(getattr(env, "unwrapped", None), "step_dt", None)
      if step_dt:
        alg_cfg["aux_step_dt"] = float(step_dt)
    return PPO.construct_algorithm(obs, env, cfg, device)


__all__ = ["Twist2PPO"]
