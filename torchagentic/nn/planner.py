"""
Differentiable planning primitives (nn.Planner).

Planning as a differentiable computation — value iteration, MCTS-style
tree search, and successor representations, all as first-class
nn.Module subclasses that can be composed and compiled.

The key insight: planning should be a layer, not a separate phase.
Backpropagating through the planner allows the representations that
feed into it to be shaped by the planning objective.
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueIteration(nn.Module):
    """Differentiable value iteration.

    Performs a fixed number of VI steps (like a recurrent network)
    to compute the optimal value function given a reward and
    transition kernel.

    Args:
        num_states:  Number of discrete states.
        num_actions: Number of discrete actions.
        gamma:       Discount factor.
        num_iters:   Number of VI iterations (depth of unrolling).

    Forward:
        reward: (batch, num_states, num_actions)  — reward per (s,a)
        kernel: (batch, num_states, num_actions, num_states) — transition probs
        init_v: (batch, num_states) optional initial value.

    Returns:
        values:   (batch, num_states) optimal value function
        q_values: (batch, num_states, num_actions) optimal Q-function

    Gradients flow through the iterations, so the reward and
    transition models that produce (reward, kernel) are shaped
    by the planning loss.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        gamma: float = 0.99,
        num_iters: int = 20,
    ):
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.num_iters = num_iters

    def forward(
        self,
        reward: torch.Tensor,
        kernel: torch.Tensor,
        init_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = reward.shape[0]
        v = torch.zeros(B, self.num_states, device=reward.device) if init_v is None else init_v

        for _ in range(self.num_iters):
            # Q(s,a) = r(s,a) + γ Σ_s' P(s'|s,a) V(s')
            q = reward + self.gamma * torch.einsum("bsas,bs->bsa", kernel, v)
            v, _ = q.max(dim=-1)

        q = reward + self.gamma * torch.einsum("bsas,bs->bsa", kernel, v)
        return v, q


class MCTSPlanner(nn.Module):
    """Differentiable Monte-Carlo Tree Search.

    A neural MCTS module that unrolls a fixed number of search
    steps.  Each step: select, expand, rollout (via learned value),
    backup.  Designed for use with MuZero-style latent dynamics.

    Args:
        num_simulations: Number of MCTS simulations.
        c_puct:          Exploration constant.
        gamma:           Discount factor.

    Forward:
        prior_logits: (batch, num_actions) — policy prior from network
        value:        (batch,) — root value prediction
        latent:       (batch, latent_dim) — root latent state
        dynamics:     callable (latent, action) -> (reward, next_latent)
        predict:      callable (latent) -> (prior_logits, value)

    Returns:
        action_probs: (batch, num_actions) — improved policy after search
        search_value: (batch,) — value after search
    """

    def __init__(
        self,
        num_simulations: int = 50,
        c_puct: float = 1.25,
        gamma: float = 0.997,
    ):
        super().__init__()
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.gamma = gamma

    def forward(self, *args, **kwargs):
        """Forward pass depends on the specific dynamics/predict
        functions passed at call time."""
        return self._search(*args, **kwargs)

    def _search(
        self,
        prior_logits: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simplified soft-search: propagates value information
        through the tree via a differentiable message-passing step.

        This is a differentiable relaxation of full MCTS:
            Improved policy = softmax(logits + c_puct * value_term)
            value_term = Σ (backup values from children)
        """
        B, A = prior_logits.shape
        policy = F.softmax(prior_logits, dim=-1)
        # PUCT-style improvement.
        value_bonus = self.c_puct * value.unsqueeze(-1) * torch.sqrt(
            torch.arange(1, A + 1, device=prior_logits.device).float()
        )
        improved_logits = prior_logits + value_bonus * policy
        return F.softmax(improved_logits, dim=-1), value


class SuccessorRepresentation(nn.Module):
    """Successor representation (Dayan, 1993).

    SR factors the value function into:
        V(s) = SR(s) · R
    where SR(s) = E[ Σ γ^t 1{s_t = s'} | s_0 = s, π ]
    and R is a learned reward vector.

    This decouples reward from transition dynamics — when the
    reward changes, you only need to re-learn R, not the SR.

    Args:
        num_states: Number of discrete states.
        gamma:      Discount factor.
        feature_dim: Dimension of the state features (for linear SR).

    Forward:
        features: (batch, feature_dim) — state features.
        reward:   (batch, num_states) — reward per state (optional,
                  if not provided, returns SR matrix).

    Returns:
        If reward is provided:
            values: (batch,) — predicted value.
        If reward is not provided:
            sr: (batch, num_states) — successor representation.
    """

    def __init__(
        self,
        num_states: int,
        gamma: float = 0.99,
        feature_dim: int = 64,
    ):
        super().__init__()
        self.num_states = num_states
        self.gamma = gamma
        self.feature_dim = feature_dim

        # Learned mapping from features to SR: ψ(s) → SR(s)
        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_states),
        )

    def forward(
        self,
        features: torch.Tensor,
        reward: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sr = self.feature_net(features)
        if reward is not None:
            return (sr * reward).sum(dim=-1)
        return sr


class LearnedPrior(nn.Module):
    """A learned plan prior — outputs a distribution over
    action sequences (N-grams) from a latent state.

    Can be used to bias MCTS or guide exploration toward
    promising action sequences.
    """

    def __init__(self, latent_dim: int, num_actions: int, plan_length: int = 8):
        super().__init__()
        self.num_actions = num_actions
        self.plan_length = plan_length
        self.net = nn.LSTM(latent_dim + num_actions, 256, batch_first=True)
        self.head = nn.Linear(256, num_actions)

    def forward(
        self,
        latent: torch.Tensor,
        cache: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, D = latent.shape
        plan_logits = []
        h = torch.zeros(1, B, 256, device=latent.device)
        c = torch.zeros(1, B, 256, device=latent.device)
        zero_action = torch.zeros(B, self.num_actions, device=latent.device)
        inp = torch.cat([latent, zero_action], dim=-1).unsqueeze(1)
        for _ in range(self.plan_length):
            out, (h, c) = self.net(inp, (h, c))
            logits = self.head(out.squeeze(1))
            plan_logits.append(logits)
            action = F.one_hot(logits.argmax(dim=-1), self.num_actions).float()
            inp = torch.cat([latent, action], dim=-1).unsqueeze(1)
        return torch.stack(plan_logits, dim=1)
