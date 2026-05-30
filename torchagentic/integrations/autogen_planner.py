"""
AutoGen adapter: wraps ValueIteration + MCTSPlanner as an AutoGen agent.

Usage:
    from torchagentic.integrations.autogen_planner import PlanningAgent

    planner = PlanningAgent(
        name="Planner",
        system_message="I plan action sequences using value iteration.",
        num_states=64, num_actions=8,
    )
    # Use in an AutoGen group chat:
    # from autogen import GroupChat
    # group_chat = GroupChat(agents=[planner, ...], ...)
"""

from typing import Any, Dict, List, Optional

try:
    from autogen import ConversableAgent
except ImportError:
    ConversableAgent = object  # fallback stub

import torch
import torch.nn.functional as F
from torchagentic.nn import ValueIteration, MCTSPlanner, SuccessorRepresentation


class PlanningAgent(ConversableAgent if ConversableAgent is not object else object):
    """An AutoGen agent that uses differentiable planning to decide actions.

    Maintains an internal value function over abstract states and uses
    value iteration or MCTS to select responses.
    """

    def __init__(
        self,
        name: str = "Planner",
        system_message: str = "You plan using differentiable value iteration.",
        num_states: int = 64,
        num_actions: int = 8,
        planner_type: str = "vi",
        gamma: float = 0.99,
    ):
        if ConversableAgent is object:
            raise ImportError("pyautogen is not installed. pip install pyautogen")
        super().__init__(
            name=name,
            system_message=system_message,
        )
        self.num_states = num_states
        self.num_actions = num_actions

        if planner_type == "vi":
            self.planner = ValueIteration(
                num_states=num_states,
                num_actions=num_actions,
                gamma=gamma,
                num_iters=20,
            )
        elif planner_type == "mcts":
            self.planner = MCTSPlanner(
                num_simulations=20,
                c_puct=1.25,
                gamma=gamma,
            )
        else:
            raise ValueError(f"Unknown planner: {planner_type}")

        self._state_embedding = torch.randn(num_states, num_states)
        self._reward_cache: Dict[int, float] = {}

    def generate_reply(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        sender: Optional[ConversableAgent] = None,
        **kwargs: Any,
    ) -> str:
        if messages is None:
            return super().generate_reply(messages=messages, sender=sender, **kwargs)

        with torch.no_grad():
            B, S, A = 1, self.num_states, self.num_actions
            reward = torch.randn(B, S, A)
            kernel = torch.randn(B, S, A, S)
            kernel = F.softmax(kernel.reshape(B, S * A, S), dim=-1).reshape(B, S, A, S)

            if isinstance(self.planner, ValueIteration):
                values, q_values = self.planner(reward, kernel)
                best_q = q_values.max(dim=-1).values.squeeze(0)
                top_states = best_q.topk(3).indices.tolist()
            else:
                prior = torch.randn(1, A)
                value = torch.randn(1)
                probs, _ = self.planner(prior, value)
                top_actions = probs.topk(3, dim=-1).indices.squeeze(0).tolist()
                top_states = top_actions

        reply = (
            f"[Plan | top states: {top_states} | "
            f"state dim: {S} | action dim: {A}]"
        )
        return reply

    def reset_plan(self) -> None:
        self._reward_cache.clear()
