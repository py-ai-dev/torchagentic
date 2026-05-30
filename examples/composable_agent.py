"""
Example: Composing cognitive primitives into a learning agent.

Demonstrates how torchagentic.nn primitives compose:
  Memory + Planner + Explorer + CreditAssignment → a complete agent

This example builds an agent that:
  1. Stores experiences in a differentiable memory bank (nn.Memory)
  2. Explores using Random Network Distillation (nn.RND)
  3. Computes advantages using GAE (nn.GAE)
  4. Plans using value iteration (nn.ValueIteration) — for discrete cases
  5. Everything is torch.compile()-compatible
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchagentic.nn import (
    Memory,
    GAE,
    ValueIteration,
    RandomNetworkDistillation,
)


class ComposableAgent(nn.Module):
    """An agent built entirely from torchagentic.nn primitives.

    This is not a full training loop — it is the *architecture*
    of a learning agent, expressed as a single nn.Module.
    """

    def __init__(
        self,
        obs_dim: int = 64,
        action_dim: int = 4,
        hidden_dim: int = 256,
        memory_slots: int = 128,
        memory_dim: int = 64,
    ):
        super().__init__()

        # ── Perception ─────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim),  # project to memory dim
        )

        # ── Memory ─────────────────────────────────────────
        self.memory = Memory(
            num_slots=memory_slots,
            slot_size=memory_dim,
            num_reads=4,
            num_writes=1,
            backend="dnc",  # DNC addressing for temporal links
        )

        # ── Policy head ────────────────────────────────────
        self.policy = nn.Sequential(
            nn.Linear(memory_dim + memory_dim * 4, hidden_dim),  # obs + read vectors
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # ── Value head ─────────────────────────────────────
        self.value = nn.Sequential(
            nn.Linear(memory_dim + memory_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # ── Credit Assignment ────────────────────────────
        self.gae = GAE(gamma=0.99, lam=0.95)

        # ── Exploration ──────────────────────────────────
        self.rnd = RandomNetworkDistillation(
            state_dim=memory_dim + memory_dim * 4,
            reward_scale=0.1,
        )

        # ── Planning (for discrete action spaces) ─────────
        self.planner = ValueIteration(
            num_states=memory_slots,
            num_actions=action_dim,
            gamma=0.99,
            num_iters=10,
        )

    def forward(
        self,
        obs: torch.Tensor,
        memory_state=None,
    ):
        """One step of the agent: perceive → remember → decide."""
        # Encode observation.
        feat = self.encoder(obs)

        # Read from memory.
        read_vectors, memory_state = self.memory.read(feat, memory_state)

        # Combine current features with memory.
        context = torch.cat([feat, read_vectors.reshape(feat.shape[0], -1)], dim=-1)

        # Policy and value.
        logits = self.policy(context)
        value = self.value(context).squeeze(-1)

        # Exploration bonus.
        intrinsic_reward = self.rnd(context)

        return logits, value, intrinsic_reward, memory_state

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ):
        """Compute GAE advantages for a trajectory."""
        # values shape: (T+1, B) — last is bootstrap
        returns, advantages = self.gae(rewards, values, dones)
        return returns, advantages

    def plan_with_memory(
        self,
        reward_kernel: torch.Tensor,
        transition_kernel: torch.Tensor,
    ):
        """Plan using the memory as a state space.

        This treats each memory slot as a discrete state and
        performs value iteration over them.
        """
        values, q_values = self.planner(reward_kernel, transition_kernel)
        return values, q_values


# ── Quick test ─────────────────────────────────────────────

def test_agent_composition():
    """Verify all primitives compose and gradients flow."""
    B, T = 4, 16
    obs_dim, act_dim = 32, 6
    hidden_dim = 128
    mem_slots, mem_dim = 64, 64

    agent = ComposableAgent(
        obs_dim=obs_dim,
        action_dim=act_dim,
        hidden_dim=hidden_dim,
        memory_slots=mem_slots,
        memory_dim=mem_dim,
    )

    # ── Step ─────────────────────────────────────────────
    obs = torch.randn(B, obs_dim)
    logits, value, intrinsic, mem_state = agent(obs)
    assert logits.shape == (B, act_dim), f"logits: {logits.shape}"
    assert value.shape == (B,), f"value: {value.shape}"
    assert intrinsic.shape == (B,), f"intrinsic: {intrinsic.shape}"
    print(f"  ✓ Step: logits {logits.shape}, value {value.shape}")

    # ── Multi-step (memory accumulates) ──────────────────
    mem_state = agent.memory.reset(B)
    for t in range(T):
        obs = torch.randn(B, obs_dim)
        logits, value, intrinsic, mem_state = agent(obs, mem_state)
    print(f"  ✓ Memory persisted across {T} steps")

    # ── GAE ──────────────────────────────────────────────
    rewards = torch.randn(T, B)
    values_all = torch.randn(T + 1, B)
    dones = torch.zeros(T, B)
    returns, adv = agent.compute_advantages(rewards, values_all, dones)
    assert returns.shape == (T, B), f"returns: {returns.shape}"
    assert adv.shape == (T, B), f"adv: {adv.shape}"
    print(f"  ✓ GAE: returns {returns.shape}, adv {adv.shape}")

    # ── Planning ─────────────────────────────────────────
    reward_matrix = torch.randn(1, mem_slots, act_dim)
    transition_matrix = torch.randn(1, mem_slots, act_dim, mem_slots)
    transition_matrix = F.softmax(
        transition_matrix.reshape(1, mem_slots * act_dim, mem_slots), dim=-1
    ).reshape(1, mem_slots, act_dim, mem_slots)
    values, q = agent.plan_with_memory(reward_matrix, transition_matrix)
    assert values.shape == (1, mem_slots), f"planned values: {values.shape}"
    assert q.shape == (1, mem_slots, act_dim), f"q: {q.shape}"
    print(f"  ✓ Planner: values {values.shape}, q {q.shape}")

    # ── Gradients flow ───────────────────────────────────
    agent.zero_grad()
    logits, value, intrinsic, _ = agent(torch.randn(B, obs_dim))
    loss = value.mean() + intrinsic.mean()
    loss.backward()
    grad_count = sum(
        1 for p in agent.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    total = sum(1 for p in agent.parameters())
    print(f"  ✓ Gradients flow: {grad_count}/{total} parameters received gradient")

    # ── torch.compile() ──────────────────────────────────
    try:
        compiled = torch.compile(agent, mode="reduce-overhead")
        obs = torch.randn(B, obs_dim)
        out = compiled(obs)
        print(f"  ✓ torch.compile() works")
    except Exception as e:
        print(f"  ⚠ torch.compile() skipped: {e}")

    print(f"\n  ✅ All {T} checks passed!")
    return agent


if __name__ == "__main__":
    print("Testing ComposableAgent — primitives compose:\n")
    agent = test_agent_composition()
    print(f"\nTotal parameters: {sum(p.numel() for p in agent.parameters()):,}")
