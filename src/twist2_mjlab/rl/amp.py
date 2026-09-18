"""Adversarial Motion Priors (AMP) for TWIST2 — improved implementation.

Compared with the earlier integration that collapsed, this version fixes:

* **dt alignment**: expert transitions are sampled from the environment motion
  library at the *control* dt (``env.step_dt``), not from raw 30 fps PKL frames.
  This removes the trivial ``||dq||`` magnitude shortcut that let the
  discriminator separate expert from policy in a few hundred steps.
* **richer style feature**: ``joint_pos(29) + joint_vel(29) + root_z(1)`` for
  both expert and policy, using the same extractor semantics (the expert path
  goes through the same motion library the policy is trained against).
* **shared normalizer**: a single ``RunningMeanStd`` updated on expert and
  policy states, so the discriminator cannot key on distributional offsets.
* **stable discriminator objective**: lower LR, logit regularization and a
  self-consistent LSGAN + R1 penalty (the old code used LSGAN targets with a
  WGAN-style gradient penalty).
* **reset masking**: policy transitions are read from the rollout buffer and
  transitions crossing an episode reset (``dones``) are dropped.
* **observability**: discriminator accuracy is logged so collapse is visible.

Enable with ``TWIST2_ENABLE_AMP=1``. All knobs are environment overridable.
"""

from __future__ import annotations

import os

import torch
from torch import nn

from twist2_mjlab.observations import NUM_G1_JOINTS

STYLE_DIM = 2 * NUM_G1_JOINTS + 1  # joint_pos(29) + joint_vel(29) + root_z(1)

DISC_LR = float(os.environ.get("TWIST2_AMP_LR", "3e-5"))
DISC_HIDDEN = (512, 256)
R1_COEF = float(os.environ.get("TWIST2_AMP_R1", "5.0"))
DISC_LOGIT_REG = float(os.environ.get("TWIST2_AMP_LOGIT_REG", "0.05"))
DISC_BATCH = int(os.environ.get("TWIST2_AMP_BATCH", "2048"))
EXPERT_BATCH = DISC_BATCH
AMP_WEIGHT = float(os.environ.get("TWIST2_AMP_WEIGHT", "0.3"))
REWARD_COEF = 0.25
EXPERT_NUM_MOTIONS = int(os.environ.get("TWIST2_AMP_EXPERT_MOTIONS", "200"))
EXPERT_HORIZON_S = float(os.environ.get("TWIST2_AMP_EXPERT_HORIZON_S", "4.0"))
AMP_GRAD_CLIP = float(os.environ.get("TWIST2_AMP_GRAD_CLIP", "1.0"))
# Guard rails: stop training the discriminator once it gets too strong, and
# smooth the "real" target so it cannot become a hard ±1 classifier. This keeps
# the style reward informative instead of collapsing to zero.
AMP_ACC_TARGET = float(os.environ.get("TWIST2_AMP_ACC_TARGET", "0.85"))
DISC_LABEL_SMOOTH = float(os.environ.get("TWIST2_AMP_LABEL_SMOOTH", "0.1"))


def amp_enabled() -> bool:
  return os.environ.get("TWIST2_ENABLE_AMP", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
  )


class RunningMeanStd:
  """Minimal running mean/variance (shared by expert and policy states)."""

  def __init__(self, shape: int, device: str, eps: float = 1e-4) -> None:
    self.mean = torch.zeros(shape, device=device)
    self.var = torch.ones(shape, device=device)
    self.count = eps

  @torch.no_grad()
  def update(self, x: torch.Tensor) -> None:
    if x.numel() == 0:
      return
    n = x.shape[0]
    batch_mean = x.mean(dim=0)
    batch_var = x.var(dim=0, unbiased=False)
    delta = batch_mean - self.mean
    total = self.count + n
    self.mean = self.mean + delta * n / total
    m_a = self.var * self.count
    m_b = batch_var * n
    m2 = m_a + m_b + delta.pow(2) * self.count * n / total
    self.var = m2 / total
    self.count = total

  def normalize(self, x: torch.Tensor) -> torch.Tensor:
    return (x - self.mean) / torch.sqrt(self.var + 1e-8)


class AMPDiscriminator(nn.Module):
  def __init__(
    self, obs_dim: int = STYLE_DIM, hidden: tuple[int, ...] = DISC_HIDDEN
  ) -> None:
    super().__init__()
    layers: list[nn.Module] = []
    in_dim = 2 * obs_dim
    for h in hidden:
      layers.append(nn.Linear(in_dim, h))
      layers.append(nn.LeakyReLU(0.2))
      in_dim = h
    layers.append(nn.Linear(in_dim, 1))
    self.net = nn.Sequential(*layers)

  def forward(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
    return self.net(torch.cat([s, s_next], dim=-1))


@torch.no_grad()
def build_expert_transitions(
  motion_lib,
  device: str,
  step_dt: float,
  num_motions: int = EXPERT_NUM_MOTIONS,
  horizon_s: float = EXPERT_HORIZON_S,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Sample expert ``(s, s_next)`` pairs from the env motion library at control dt."""
  lengths = motion_lib._motion_lengths  # [M]
  num_available = int(lengths.shape[0])
  num_motions = min(num_motions, num_available)
  order = torch.randperm(num_available, device=lengths.device)[:num_motions]

  kmax = max(2, int(horizon_s / step_dt))
  states: list[torch.Tensor] = []
  for motion_id in order.tolist():
    length = float(lengths[motion_id].item())
    k = max(2, min(kmax, int(length / step_dt)))
    if k < 2:
      continue
    times = torch.arange(k, device=lengths.device, dtype=torch.float32) * step_dt
    ids = torch.full((k,), motion_id, device=lengths.device, dtype=torch.long)
    frame = motion_lib.get_frame(ids, times)
    state = torch.cat(
      (frame.joint_pos, frame.joint_vel, frame.body_pos_w[:, 0, 2:3]), dim=-1
    )
    states.append(state.to(device).float())

  all_states = torch.cat(states, dim=0)
  expert_s = all_states[:-1].contiguous()
  expert_sn = all_states[1:].contiguous()
  print(
    f"[AMP] expert transitions: {expert_s.shape[0]} from {num_motions} motions "
    f"(control dt={step_dt:.4f}s, dim={STYLE_DIM})"
  )
  return expert_s, expert_sn


def _disc_accuracy(
  disc: AMPDiscriminator,
  normalizer: RunningMeanStd,
  expert_s: torch.Tensor,
  expert_sn: torch.Tensor,
  policy_s: torch.Tensor,
  policy_sn: torch.Tensor,
) -> float:
  with torch.no_grad():
    logits_e = disc(normalizer.normalize(expert_s), normalizer.normalize(expert_sn))
    logits_p = disc(normalizer.normalize(policy_s), normalizer.normalize(policy_sn))
    return float(
      (0.5 * ((logits_e > 0).float().mean() + (logits_p < 0).float().mean())).item()
    )


def update_discriminator(
  disc: AMPDiscriminator,
  optim: torch.optim.Optimizer,
  normalizer: RunningMeanStd,
  expert_s: torch.Tensor,
  expert_sn: torch.Tensor,
  policy_s: torch.Tensor,
  policy_sn: torch.Tensor,
) -> tuple[float, float, float]:
  disc.train()
  normalizer.update(expert_s)
  normalizer.update(policy_s)

  es = normalizer.normalize(expert_s)
  esn = normalizer.normalize(expert_sn)
  ps = normalizer.normalize(policy_s)
  psn = normalizer.normalize(policy_sn)
  es.requires_grad_(True)
  esn.requires_grad_(True)

  logits_e = disc(es, esn)
  logits_p = disc(ps, psn)

  target_e = 1.0 - DISC_LABEL_SMOOTH
  lsgan = (
    0.5 * (logits_e - target_e).pow(2).mean()
    + 0.5 * (logits_p + 1.0).pow(2).mean()
  )
  logit_reg = DISC_LOGIT_REG * (
    logits_e.pow(2).mean() + logits_p.pow(2).mean()
  )
  grads = torch.autograd.grad(
    logits_e.sum(), (es, esn), create_graph=True, retain_graph=True
  )
  r1 = 0.5 * R1_COEF * sum(g.pow(2).sum(dim=-1).mean() for g in grads)
  loss = lsgan + logit_reg + r1

  optim.zero_grad()
  loss.backward()
  if AMP_GRAD_CLIP > 0:
    nn.utils.clip_grad_norm_(disc.parameters(), AMP_GRAD_CLIP)
  optim.step()

  with torch.no_grad():
    accuracy = 0.5 * (
      (logits_e > 0).float().mean() + (logits_p < 0).float().mean()
    )
  disc.eval()
  return float(loss.item()), float(r1.item()), float(accuracy.item())


class AmpState:
  """Holds the discriminator, its optimizer, the shared normalizer and the
  expert transition buffer. Attached to the PPO algorithm by the runner."""

  def __init__(
    self,
    motion_lib,
    device: str,
    step_dt: float,
    num_motions: int = EXPERT_NUM_MOTIONS,
    horizon_s: float = EXPERT_HORIZON_S,
  ) -> None:
    self.device = device
    self.disc = AMPDiscriminator().to(device)
    self.optim = torch.optim.Adam(self.disc.parameters(), lr=DISC_LR)
    self.normalizer = RunningMeanStd(STYLE_DIM, device)
    self.expert_s, self.expert_sn = build_expert_transitions(
      motion_lib, device, step_dt, num_motions, horizon_s
    )
    self.last_stats: dict[str, float] = {}

  def update(self, storage) -> dict[str, float]:
    obs_dict = storage.observations
    if "amp_style" not in obs_dict.keys():
      return {}
    style = obs_dict["amp_style"]  # [T, N, STYLE_DIM]
    # Policy transitions: consecutive stored time steps, dropping the ones that
    # cross an episode reset (dones[t] == 1).
    s = style[:-1].reshape(-1, STYLE_DIM)
    sn = style[1:].reshape(-1, STYLE_DIM)
    dones = storage.dones[:-1].reshape(-1)
    valid = (dones < 0.5).nonzero(as_tuple=False).squeeze(-1)
    if valid.numel() < DISC_BATCH:
      return {}
    perm = torch.randperm(valid.numel(), device=valid.device)[:DISC_BATCH]
    idx = valid[perm]
    policy_s = s[idx]
    policy_sn = sn[idx]

    ei = torch.randint(
      0, self.expert_s.shape[0], (EXPERT_BATCH,), device=self.device
    )
    es = self.expert_s[ei]
    esn = self.expert_sn[ei]

    # Guard rail: if the discriminator is stronger than the target, skip this
    # optimizer step but keep evaluating. Once the policy catches up and the
    # accuracy falls back below the target, training resumes automatically.
    acc = _disc_accuracy(
      self.disc, self.normalizer, es, esn, policy_s, policy_sn
    )
    if acc > AMP_ACC_TARGET:
      self.last_stats = {"amp_disc_acc": acc}
      return dict(self.last_stats)

    loss, r1, accuracy = update_discriminator(
      self.disc,
      self.optim,
      self.normalizer,
      es,
      esn,
      policy_s,
      policy_sn,
    )
    self.last_stats = {
      "amp_disc_loss": loss,
      "amp_r1": r1,
      "amp_disc_acc": accuracy,
    }
    return dict(self.last_stats)


__all__ = [
  "AMP_WEIGHT",
  "REWARD_COEF",
  "STYLE_DIM",
  "AMPDiscriminator",
  "AmpState",
  "RunningMeanStd",
  "amp_enabled",
  "build_expert_transitions",
  "update_discriminator",
]
