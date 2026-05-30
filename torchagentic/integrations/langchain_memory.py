"""
LangChain adapter: wraps nn.Memory as a LangChain BaseMemory.

Usage:
    from torchagentic.integrations.langchain_memory import TorchMemory

    memory = TorchMemory(
        num_slots=128, slot_size=64, num_reads=4, backend="dnc"
    )
    # Use like any LangChain memory:
    memory.save_context({"input": "hello"}, {"output": "world"})
    vars = memory.load_memory_variables({})
"""

from typing import Any, Dict, List, Optional

try:
    from langchain.schema import BaseMemory
except ImportError:
    BaseMemory = object  # fallback stub

import torch
from torchagentic.nn import Memory, MemoryState


class TorchMemory(BaseMemory if BaseMemory is not object else object):
    """LangChain memory backed by a differentiable nn.Memory primitive.

    Stores conversation history as tensors that can be used for
    gradient-based training (e.g. RL fine-tuning of the agent's
    representation network).
    """

    memory: Memory
    state: Optional[MemoryState]
    history: List[str]

    def __init__(
        self,
        num_slots: int = 128,
        slot_size: int = 64,
        num_reads: int = 4,
        backend: str = "dnc",
        device: Optional[torch.device] = None,
    ):
        if BaseMemory is object:
            raise ImportError("langchain is not installed. pip install langchain")
        super().__init__()
        self.memory = Memory(
            num_slots=num_slots,
            slot_size=slot_size,
            num_reads=num_reads,
            backend=backend,
        )
        self.device = device or torch.device("cpu")
        self.memory.to(self.device)
        self.state = None
        self.history = []

    @property
    def memory_variables(self) -> List[str]:
        return ["memory_context"]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.state is None:
            return {"memory_context": None}
        return {"memory_context": self.state.memory.detach().cpu().numpy()}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        inp_str = str(inputs.get("input", ""))
        out_str = str(outputs.get("output", ""))
        self.history.append(f"{inp_str} | {out_str}")

        with torch.no_grad():
            query = torch.randn(1, self.memory.slot_size, device=self.device)
            if self.state is None:
                self.state = self.memory.reset(1)
            _, self.state = self.memory.read(query, self.state)

    def clear(self) -> None:
        self.state = None
        self.history.clear()
