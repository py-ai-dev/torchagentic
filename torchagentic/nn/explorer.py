"""
Exploration primitives (nn.Explorer).

Exploration is a first-class cognitive operation, not a heuristic tacked
onto a policy.  These modules provide differentiable exploration
strategies that can be composed with any policy network.

Includes: intrinsic curiosity, Random Network Distillation (RND),
count-based exploration, and uncertainty-based exploration.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Explorer(nn.Module):
    """Abstract base for exploration bonuses.

    All explorers produce an *intrinsic reward* that is added
    to the environment reward during training.  The intrinsic
    reward is high for novel states and decays with visitation.

    Args:
        reward_scale: How much to scale the intrinsic reward.
    """

    reward_scale: float

    def forward(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        next_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute intrinsic reward for a batch of transitions.

        Returns:
            intrinsic_reward: (batch,) — added to extrinsic reward.
        """
        raise NotImplementedError


class RandomNetworkDistillation(Explorer):
    """Random Network Distillation (Burda et al., 2018).

    A fixed random network (target) and a trainable predictor.
    The predictor is trained to match the target's output on seen
    states.  Error = novelty = intrinsic reward.

    References:
        https://arxiv.org/abs/1810.12894
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        reward_scale: float = 1.0,
    ):
        super().__init__()
        self.reward_scale = reward_scale

        # Fixed random target.
        self.target = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        for p in self.target.parameters():
            p.requires_grad_(False)

        # Trainable predictor.
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        next_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            target_feat = self.target(states)
        pred_feat = self.predictor(states)
        error = (pred_feat - target_feat).pow(2).mean(dim=-1)
        return error * self.reward_scale


class ICM(Explorer):
    """Intrinsic Curiosity Module (Pathak et al., 2017).

    Curiosity = error in predicting the next state given (state, action).
    High error = novel dynamics = explore more.

    References:
        https://arxiv.org/abs/1705.05363
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embed_dim: int = 128,
        reward_scale: float = 0.01,
    ):
        super().__init__()
        self.reward_scale = reward_scale

        # State embedding.
        self.embed = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Forward dynamics: (state_embed, action) → next_state_embed.
        self.dynamics = nn.Sequential(
            nn.Linear(embed_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        next_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if actions is None or next_states is None:
            raise ValueError("ICM requires actions and next_states")

        s_embed = self.embed(states)
        ns_embed = self.embed(next_states).detach()
        pred_ns = self.dynamics(torch.cat([s_embed, actions], dim=-1))
        error = (pred_ns - ns_embed).pow(2).mean(dim=-1)
        return error * self.reward_scale

    def forward_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        s_embed = self.embed(states)
        ns_embed_target = self.embed(next_states).detach()
        pred_ns = self.dynamics(torch.cat([s_embed, actions], dim=-1))
        return (pred_ns - ns_embed_target).pow(2).mean()


class CountBonus(Explorer):
    """Count-based exploration bonus — log(1/N(s)).

    Tracks visitation counts via a learned density model
    (a small MLP that outputs a pseudo-count).

    Args:
        state_dim: State dimensionality.
        reward_scale: Scale factor for the bonus.
    """

    def __init__(self, state_dim: int, reward_scale: float = 0.1):
        super().__init__()
        self.reward_scale = reward_scale
        self.register_buffer("total_count", torch.tensor(0.0))
        self.density = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        next_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            rho = self.density(states).squeeze(-1)
        bonus = (1.0 / (rho + 1e-8)).sqrt() * self.reward_scale
        return bonus


class DisagreementEnsemble(Explorer):
    """Ensemble disagreement exploration.

    An ensemble of forward dynamics models.  Disagreement in their
    predictions = epistemic uncertainty = exploration bonus.

    References:
        Pathak et al., "Curiosity-driven Exploration by Self-Supervised Prediction", 2017
        (ensemble variant popularized by exploration literature).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        ensemble_size: int = 5,
        reward_scale: float = 0.05,
    ):
        super().__init__()
        self.reward_scale = reward_scale
        self.ensemble_size = ensemble_size

        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim),
            )
            for _ in range(ensemble_size)
        ])

    def forward(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        next_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if actions is None:
            raise ValueError("DisagreementEnsemble requires actions")

        sa = torch.cat([states, actions], dim=-1)
        preds = torch.stack([m(sa) for m in self.models], dim=0)  # (E, B, D)
        disagreement = preds.var(dim=0).mean(dim=-1)  # (B,)
        return disagreement * self.reward_scale
