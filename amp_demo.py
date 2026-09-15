"""Standalone AMP (Adversarial Motion Priors) demo.

Demonstrates the CORE AMP mechanism in isolation (no RL env, no physics):
  1. Load expert motion from an enriched TWIST2 pkl.
  2. Build "style state" transitions (s_t, s_{t+1}) from joint positions + velocities.
  3. Train a discriminator D(s, s') to tell apart:
       - expert transitions (consecutive frames -> natural dynamics)
       - fake transitions   (random next frame   -> broken dynamics)
     using the LSGAN objective + gradient penalty.
  4. Show the AMP reward  r = max(0, 1 - 0.25*(D - 1)^2)  separating the two.

Run:
    cd /home/user/twist2_mjlab
    uv run python amp_demo.py /home/user/twist2_data/enriched/OMOMO_g1_GMR/sub13_tripod_010.pkl

Tune the knobs under "==== TUNABLE PARAMETERS ====" and watch the effect.
"""

import argparse
import pickle

import numpy as np
import torch
import torch.nn as nn

# ============================================================================
# TUNABLE PARAMETERS  (this is what you would tune in a real AMP setup)
# ============================================================================
LAMBDA_GP = 10.0        # gradient penalty coefficient (WGAN-GP). 5~10 typical.
                        #  too small -> discriminator overfits / training unstable
                        #  too big   -> discriminator underfits, can't tell apart
W_AMP = 1.0             # style-reward weight vs task reward (task = 1.0).
                        #  this demo has no task reward, so it just scales r_AMP
DISC_HIDDEN = (512, 256)  # discriminator MLP capacity
DISC_LR = 1e-3          # discriminator learning rate
BATCH = 512             # transitions sampled per update
ITERATIONS = 2000       # discriminator training steps
REWARD_COEF = 0.25      # the 0.25 in r = max(0, 1 - 0.25*(D-1)^2)
# ============================================================================


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def euler_roll_pitch_from_quat(q):
    """Extract roll, pitch from a wxyz quaternion (numpy)."""
    w, x, y, z = q
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return np.array([roll, pitch], dtype=np.float32)


def build_style_state(data: dict) -> torch.Tensor:
    """Build per-frame style state s(t) from an enriched pkl.

    Style state = [joint_pos_rel (29) | joint_vel (29) | root_height (1)
                   | root_roll_pitch (2) | root_lin_vel_xy (2)]  = 63 dims.

    Note: AMP feeds the discriminator STATE ONLY (no actions), specifically the
    parts that capture "style" -- joint pose/dynamics + root motion.
    """
    joint_pos = torch.tensor(np.asarray(data["dof_pos"]), dtype=torch.float32)   # (T,29)
    fps = data["fps"]
    dt = 1.0 / fps

    joint_vel = torch.gradient(joint_pos, spacing=(dt,), dim=0)[0]               # (T,29)
    joint_pos_rel = joint_pos - joint_pos[0]                                     # relative to frame 0

    body_pos_w = torch.tensor(np.asarray(data["body_pos_w"]), dtype=torch.float32)  # (T,B,3)
    body_quat_w = torch.tensor(np.asarray(data["body_quat_w"]), dtype=torch.float32)  # (T,B,4)

    root_pos = body_pos_w[:, 0]                                                  # (T,3)
    root_height = root_pos[:, 2:3]                                               # (T,1)
    root_vel = torch.gradient(root_pos, spacing=(dt,), dim=0)[0]                 # (T,3)
    root_lin_vel_xy = root_vel[:, :2]                                            # (T,2)

    root_quat = body_quat_w[:, 0]                                                # (T,4) wxyz
    rps = np.array([euler_roll_pitch_from_quat(q) for q in root_quat.numpy()],
                   dtype=np.float32)                                              # (T,2)
    roll_pitch = torch.tensor(rps)

    s = torch.cat([joint_pos_rel, joint_vel, root_height, roll_pitch, root_lin_vel_xy], dim=-1)
    return s


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------
class AMPDiscriminator(nn.Module):
    """D(s, s') -> scalar logit. Input is the CONCATENATED transition [s, s']."""

    def __init__(self, obs_dim: int, hidden: tuple[int, ...] = DISC_HIDDEN):
        super().__init__()
        layers, d = [], obs_dim * 2
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))  # scalar logit (no sigmoid)
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, s_next], dim=-1))


# ---------------------------------------------------------------------------
# Losses / reward
# ---------------------------------------------------------------------------
def lsgan_disc_loss(disc, se, sne, sp, spn):
    """LSGAN: pull expert toward +1, policy toward -1."""
    logits_e = disc(se, sne)
    logits_p = disc(sp, spn)
    loss_e = (logits_e - 1.0).pow(2).mean()
    loss_p = (logits_p + 1.0).pow(2).mean()
    return 0.5 * loss_e + 0.5 * loss_p, logits_e.detach(), logits_p.detach()


def gradient_penalty(disc, se, sne, sp, spn):
    """WGAN-GP: penalize ||grad D||^2 deviating from 1, on interpolated samples."""
    alpha = torch.rand(se.shape[0], 1, device=se.device)
    s = alpha * se + (1.0 - alpha) * sp
    s_next = alpha * sne + (1.0 - alpha) * spn
    s.requires_grad_(True)
    s_next.requires_grad_(True)
    logits = disc(s, s_next)
    grad = torch.autograd.grad(logits.sum(), [s, s_next], create_graph=True)[0]
    return (grad.norm(2, dim=-1) - 1.0).pow(2).mean()


def amp_reward(logits: torch.Tensor) -> torch.Tensor:
    """Bounded AMP reward: r = max(0, 1 - 0.25*(D - 1)^2)."""
    return torch.clamp(1.0 - REWARD_COEF * (logits - 1.0).pow(2), min=0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl", help="Path to an enriched TWIST2 pkl")
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    s = build_style_state(data)                     # (T, obs_dim)
    obs_dim = s.shape[1]
    T = s.shape[0]
    print(f"Loaded motion: {T} frames, style state dim = {obs_dim}")

    # ---- Expert transitions: consecutive frames (natural dynamics) ----
    expert_s = s[:-1]                               # (T-1, D)
    expert_sn = s[1:]                               # (T-1, D)

    # ---- Fake transitions: random next frame (broken dynamics) ----
    perm = torch.randperm(T - 1)
    fake_s = s[:-1]
    fake_sn = s[1:][perm]

    disc = AMPDiscriminator(obs_dim)
    optim = torch.optim.Adam(disc.parameters(), lr=DISC_LR)

    print(f"\nDiscriminator: {obs_dim*2} -> {DISC_HIDDEN} -> 1")
    print(f"lambda_gp={LAMBDA_GP}, lr={DISC_LR}, batch={BATCH}\n")
    print(f"{'iter':>6} {'disc_loss':>10} {'gp':>8} {'r_expert':>9} {'r_fake':>8}")

    for it in range(1, ITERATIONS + 1):
        # sample mini-batches
        ie = torch.randint(0, expert_s.shape[0], (BATCH,))
        ip = torch.randint(0, fake_s.shape[0], (BATCH,))
        se, sne = expert_s[ie], expert_sn[ie]
        sp, spn = fake_s[ip], fake_sn[ip]

        loss, le, lp = lsgan_disc_loss(disc, se, sne, sp, spn)
        gp = gradient_penalty(disc, se, sne, sp, spn)
        total = loss + LAMBDA_GP * gp

        optim.zero_grad()
        total.backward()
        optim.step()

        if it % 200 == 0 or it == 1:
            re = amp_reward(le).mean().item()
            rf = amp_reward(lp).mean().item()
            print(f"{it:6d} {loss.item():10.4f} {gp.item():8.4f} {re:9.4f} {rf:8.4f}")

    # ---- Final summary ----
    with torch.no_grad():
        le = disc(expert_s, expert_sn)
        lp = disc(fake_s, fake_sn)
        re = amp_reward(le).mean().item()
        rf = amp_reward(lp).mean().item()
    print("\n===== RESULT =====")
    print(f"expert reward: {re:.4f}  (should approach 1.0)")
    print(f"fake   reward: {rf:.4f}  (should approach 0.0)")
    print("\nInterpretation: the discriminator has learned to recognize 'natural'")
    print("consecutive motion transitions. In a real AMP setup this reward would")
    print("be added to the RL task reward to push the policy toward natural motion.")


if __name__ == "__main__":
    main()
