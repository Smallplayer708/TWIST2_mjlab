"""Differentiable privileged dynamics model for TWIST2 auxiliary training.

The model predicts the *delta* of a privileged robot state over one control step
from ``(state, action, reference)``. It is a plain MLP trained by supervised MSE
on on-policy transitions. Because it never touches contact forces or the physics
solver, the auxiliary gradient path is smooth by construction and avoids the
contact non-smoothness / gradient-explosion problem of true differentiable
physics.
"""

from __future__ import annotations

import torch
from torch import nn

from rsl_rl.utils import resolve_nn_activation


class PrivilegedDynamicsModel(nn.Module):
  """MLP predicting the one-step delta of a privileged robot state."""

  def __init__(
    self,
    state_dim: int,
    action_dim: int,
    ref_dim: int = 0,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256),
    activation: str = "elu",
  ) -> None:
    super().__init__()
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.ref_dim = ref_dim

    layers: list[nn.Module] = []
    in_dim = state_dim + action_dim + ref_dim
    for hidden_dim in hidden_dims:
      layers.append(nn.Linear(in_dim, hidden_dim))
      layers.append(resolve_nn_activation(activation))
      in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, state_dim))
    self.net = nn.Sequential(*layers)

    # Start as an identity (zero-delta) model so the auxiliary rollout is stable
    # at the beginning of training and cannot inject a large early bias.
    final = self.net[-1]
    assert isinstance(final, nn.Linear)
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)

  def forward(
    self,
    state: torch.Tensor,
    action: torch.Tensor,
    ref: torch.Tensor | None = None,
  ) -> torch.Tensor:
    parts = [state, action]
    if ref is not None and self.ref_dim > 0:
      parts.append(ref)
    return state + self.net(torch.cat(parts, dim=-1))


__all__ = ["PrivilegedDynamicsModel"]
