"""
CrewAI adapter: wraps Memory + Explorer as a trainable CrewAI tool.

Usage:
    from torchagentic.integrations.crewai_tool import TrainableMemoryTool

    tool = TrainableMemoryTool(
        name="EpisodicMemory",
        description="Store and retrieve episodic experiences",
        num_slots=256, slot_size=64,
    )
    # Use in a CrewAI agent:
    # agent = Agent(tools=[tool], ...)
"""

from typing import Any, Optional

try:
    from crewai.tools import BaseTool
except ImportError:
    BaseTool = object  # fallback stub

import torch
from torchagentic.nn import Memory, RandomNetworkDistillation


class TrainableMemoryTool(BaseTool if BaseTool is not object else object):
    """A CrewAI tool backed by a differentiable memory + exploration bonus.

    Unlike heuristic memory tools, this one has trainable parameters
    that can be updated via gradient descent.
    """

    memory: Memory
    explorer: Optional[RandomNetworkDistillation]

    def __init__(
        self,
        name: str = "EpisodicMemory",
        description: str = "Stores and retrieves episodic experiences",
        num_slots: int = 256,
        slot_size: int = 64,
        num_reads: int = 4,
        enable_exploration: bool = True,
    ):
        if BaseTool is object:
            raise ImportError("crewai is not installed. pip install crewai")
        super().__init__(
            name=name,
            description=description,
        )
        self.memory = Memory(
            num_slots=num_slots,
            slot_size=slot_size,
            num_reads=num_reads,
            backend="dnc",
        )
        self.explorer = (
            RandomNetworkDistillation(state_dim=slot_size * num_reads)
            if enable_exploration else None
        )
        self._state = None
        self._step = 0

    def _run(self, query: str) -> str:
        with torch.no_grad():
            x = torch.randn(1, self.memory.slot_size)
            if self._state is None:
                self._state = self.memory.reset(1)
            vectors, self._state = self.memory.read(x, self._state)

            context = vectors.reshape(1, -1)
            if self.explorer is not None:
                bonus = self.explorer(context).item()
            else:
                bonus = 0.0

        self._step += 1
        return (
            f"[memory slot activations: {context[0, :4].tolist()}... | "
            f"novelty bonus: {bonus:.4f} | step: {self._step}]"
        )

    def reset(self) -> None:
        self._state = None
        self._step = 0
