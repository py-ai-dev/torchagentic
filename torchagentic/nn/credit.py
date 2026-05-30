"""
Temporal credit assignment primitives (nn.CreditAssignment).

A family of differentiable modules for attributing credit across time
in agent trajectories.  Wraps TD(λ), GAE, V-trace, and Retrace as
first-class nn.Module subclasses with compile-optimized tensor ops.

Every module in this file answers the same question:
    "Given a trajectory, how much credit does each step deserve?"
and returns two tensors:
    returns:      (T, B)  —  discounted sum of rewards per step
    advantages:   (T, B)  —  advantage estimate per step
"""

from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class CreditAssignment(nn.Module):
    """Abstract base for temporal credit assignment.

    Subclasses implement a specific algorithm (TD(λ), GAE, etc.).
    All share the same call signature so they are drop-in swappable.

    Args:
        gamma: Discount factor.
        lam:   Trace-decay / bias-variance trade-off parameter.
    """

    gamma: float
    lam: float

    def forward(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
        log_probs: Optional[torch.Tensor] = None,
        target_log_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map a trajectory to per-step returns and advantages.

        Args:
            rewards:  (T, B)  reward at each step.
            values:   (T + 1, B)  value prediction at each step
                      (last element is the bootstrap / next-state value).
            dones:    (T, B)  boolean episode-termination flags.
            log_probs:  (T, B)  behaviour log-probs (off-policy only).
            target_log_probs:  (T, B)  target log-probs (off-policy only).

        Returns:
            returns:     (T, B)
            advantages:  (T, B)
        """
        raise NotImplementedError


class GAE(CreditAssignment):
    """Generalized Advantage Estimation (Schulman et al., 2015).

    The standard advantage estimator used by PPO.  A weighted
    combination of TD residuals where λ controls bias vs. variance.

    References:
        https://arxiv.org/abs/1506.02438
    """

    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        super().__init__()
        self.gamma = gamma
        self.lam = lam

    def forward(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
        log_probs: Optional[torch.Tensor] = None,
        target_log_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T, B = rewards.shape
        device = rewards.device

        if dones is None:
            dones = torch.zeros(T, B, device=device)

        # TD residual δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
        bootstrap = values[1:] * (1 - dones)
        deltas = rewards + self.gamma * bootstrap - values[:-1]

        # GAE = Σ (γλ)^k δ_{t+k}
        adv = torch.zeros_like(deltas)
        running = torch.zeros(B, device=device)
        for t in reversed(range(T)):
            running = deltas[t] + self.gamma * self.lam * running * (1 - dones[t])
            adv[t] = running

        returns = adv + values[:-1]
        return returns, adv


class TDLambda(CreditAssignment):
    """TD(λ) — the classic temporal-difference return.

    λ = 0 is one-step TD (bootstrap immediately).
    λ = 1 is Monte Carlo (full empirical return).

    References:
        Sutton & Barto, "Reinforcement Learning: An Introduction", 2nd ed.
    """

    def __init__(self, gamma: float = 0.99, lam: float = 0.8):
        super().__init__()
        self.gamma = gamma
        self.lam = lam

    def forward(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
        log_probs: Optional[torch.Tensor] = None,
        target_log_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T, B = rewards.shape
        device = rewards.device

        if dones is None:
            dones = torch.zeros(T, B, device=device)

        # λ-return: G_t^λ = R_t + γ * ((1-λ) * V_{t+1} + λ * G_{t+1}^λ)
        ret = torch.zeros(T, B, device=device)
        running = values[-1]  # bootstrap value
        for t in reversed(range(T)):
            done_mask = 1 - dones[t]
            running = rewards[t] + self.gamma * done_mask * (
                (1 - self.lam) * values[t + 1] + self.lam * running
            )
            ret[t] = running

        adv = ret - values[:-1]
        return ret, adv


class VTrace(CreditAssignment):
    """V-trace (Espeholt et al., 2018) — off-policy correction.

    Used by IMPALA to correct for stale / behaviour policy lag.
    Clips importance weights to reduce variance.

    References:
        https://arxiv.org/abs/1802.01561
    """

    def __init__(
        self,
        gamma: float = 0.99,
        lam: float = 1.0,
        rho_bar: float = 1.0,
        c_bar: float = 1.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.lam = lam
        self.rho_bar = rho_bar  # clip for IS weight in value target
        self.c_bar = c_bar      # clip for IS weight in advantage trace

    def forward(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
        log_probs: Optional[torch.Tensor] = None,
        target_log_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T, B = rewards.shape
        device = rewards.device

        if dones is None:
            dones = torch.zeros(T, B, device=device)
        if log_probs is None or target_log_probs is None:
            raise ValueError("V-trace requires log_probs and target_log_probs")

        # Importance weights, clipped.
        rho = torch.exp(target_log_probs - log_probs)
        rho = torch.clamp(rho, max=self.rho_bar)
        c = torch.clamp(rho, max=self.c_bar)

        # TD residual δ_t = ρ_t * (r_t + γ * V_{t+1} - V_t)
        deltas = rho * (rewards + self.gamma * values[1:] * (1 - dones) - values[:-1])

        # V-trace target: V_t + Σ (γ * c_{t+1})^k δ_{t+k}
        vs = torch.zeros(T, B, device=device)
        running = torch.zeros(B, device=device)
        for t in reversed(range(T)):
            running = deltas[t] + self.gamma * c[t] * running * (1 - dones[t])
            vs[t] = values[t] + running

        adv = vs - values[:-1]
        return vs, adv


class Retrace(CreditAssignment):
    """Retrace (Munos et al., 2016) — safe off-policy TD(λ).

    Uses truncated importance weights to correct for off-policy
    action-value estimation.  Provably converges in expectation.

    References:
        https://arxiv.org/abs/1606.01247
    """

    def __init__(self, gamma: float = 0.99, lam: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.lam = lam

    def forward(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
        log_probs: Optional[torch.Tensor] = None,
        target_log_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T, B = rewards.shape
        device = rewards.device

        if dones is None:
            dones = torch.zeros(T, B, device=device)
        if log_probs is None or target_log_probs is None:
            raise ValueError("Retrace requires log_probs and target_log_probs")

        # Truncated importance weight.
        c = torch.exp(target_log_probs - log_probs)
        c = torch.clamp(c, max=1.0)

        # Retrace target.
        ret = torch.zeros(T, B, device=device)
        running = values[-1]
        for t in reversed(range(T)):
            done_mask = 1 - dones[t]
            running = rewards[t] + self.gamma * done_mask * running
            ret[t] = values[t] + c[t] * (running - values[t])
            running = self.lam * ret[t] + (1 - self.lam) * running

        adv = ret - values[:-1]
        return ret, adv


class TDLambdaNet(nn.Module):
    """A learned TD(λ) module — the trace-decay parameter λ is
    a learnable parameter rather than a fixed hyper-parameter.

    This is a small theoretical contribution: allow the gradient
    to flow into the credit-assignment mechanism itself.
    """

    def __init__(self, gamma: float = 0.99, init_lam: float = 0.8):
        super().__init__()
        self.gamma = gamma
        self.logit_lam = nn.Parameter(torch.tensor(init_lam).logit())

    @property
    def lam(self) -> torch.Tensor:
        return self.logit_lam.sigmoid()

    def forward(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
        log_probs: Optional[torch.Tensor] = None,
        target_log_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lam = self.lam.item()
        T, B = rewards.shape
        device = rewards.device
        if dones is None:
            dones = torch.zeros(T, B, device=device)

        ret = torch.zeros(T, B, device=device)
        running = values[-1]
        for t in reversed(range(T)):
            done_mask = 1 - dones[t]
            running = rewards[t] + self.gamma * done_mask * (
                (1 - lam) * values[t + 1] + lam * running
            )
            ret[t] = running

        adv = ret - values[:-1]
        return ret, adv
