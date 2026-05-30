"""
torchagentic.nn — Differentiable Cognitive Primitives for Agentic AI.

A library of composable, compile-optimized PyTorch primitives for building
agents that learn. Each primitive fills a gap in torch.nn that is specific
to agentic AI: memory, credit assignment, planning, exploration, abstraction,
multi-agent communication, world models, and reward learning.
"""

from torchagentic.nn.memory import (
    Memory,
    MemoryBackend,
    MemoryState,
    NTMBank,
    DNCBank,
    SlidingWindowBank,
)

from torchagentic.nn.credit import (
    CreditAssignment,
    GAE,
    TDLambda,
    VTrace,
    Retrace,
    TDLambdaNet,
)

from torchagentic.nn.planner import (
    ValueIteration,
    MCTSPlanner,
    SuccessorRepresentation,
    LearnedPrior,
)

from torchagentic.nn.explorer import (
    Explorer,
    RandomNetworkDistillation,
    ICM,
    CountBonus,
    DisagreementEnsemble,
)

__all__ = [
    "Memory",
    "MemoryBackend",
    "MemoryState",
    "NTMBank",
    "DNCBank",
    "SlidingWindowBank",
    "CreditAssignment",
    "GAE",
    "TDLambda",
    "VTrace",
    "Retrace",
    "TDLambdaNet",
    "ValueIteration",
    "MCTSPlanner",
    "SuccessorRepresentation",
    "LearnedPrior",
    "Explorer",
    "RandomNetworkDistillation",
    "ICM",
    "CountBonus",
    "DisagreementEnsemble",
]
