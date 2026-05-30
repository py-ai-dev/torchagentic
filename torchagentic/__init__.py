"""
TorchAgentic — Differentiable Cognitive Primitives for Agentic AI.

A library of composable, compile-optimized PyTorch primitives for building
agents that learn.  Provides the missing primitives for agentic AI:
differentiable memory, temporal credit assignment, planning, exploration,
and hierarchical abstraction — all as first-class nn.Module subclasses.
"""

__version__ = "0.2.0"
__author__ = "Liodon AI"

# ── Cognitive Primitives ──────────────────────────────────
from torchagentic import nn as nn

# ── Memory ────────────────────────────────────────────────
from torchagentic.nn.memory import (
    Memory, MemoryState, NTMBank, DNCBank, SlidingWindowBank,
)

# ── Credit Assignment ─────────────────────────────────────
from torchagentic.nn.credit import (
    GAE, TDLambda, VTrace, Retrace, TDLambdaNet,
)

# ── Planning ──────────────────────────────────────────────
from torchagentic.nn.planner import (
    ValueIteration, MCTSPlanner, SuccessorRepresentation, LearnedPrior,
)

# ── Exploration ───────────────────────────────────────────
from torchagentic.nn.explorer import (
    RandomNetworkDistillation, ICM, CountBonus, DisagreementEnsemble,
)

# ── Integration Adapters ──────────────────────────────────
from torchagentic import integrations as integrations

# ── Legacy models (RL, transformers, full agents) ─────────
from torchagentic.models.base import BaseAgentModel, ModelConfig
from torchagentic.models.mlp import MLPNetwork
from torchagentic.models.cnn import CNNNetwork, NatureCNN, ResNetNetwork
from torchagentic.models.rnn import RNNNetwork, LSTMAgent, GRUAgent

from torchagentic.rl.dqn import DQN, DuelingDQN, NoisyDQN
from torchagentic.rl.ppo import PPOActor, PPOCritic, PPOActorCritic
from torchagentic.rl.a3c import A3CNetwork
from torchagentic.rl.sac import SACActor, SACCritic, SACValue
from torchagentic.rl.td3 import TD3Actor, TD3Critic

from torchagentic.transformers.agent import TransformerAgent, DecisionTransformer
from torchagentic.transformers.perceiver import PerceiverAgent
from torchagentic.transformers.attention import SelfAttention, MultiHeadAttention

from torchagentic.memory.ntm import NeuralTuringMachine
from torchagentic.memory.dnc import DifferentiableNeuralComputer

from torchagentic.multiagent.base import MultiAgentBase
from torchagentic.multiagent.maddpg import MADDPGAgent
from torchagentic.multiagent.qmix import QMIXNetwork, VDNNetwork

from torchagentic.utils.initialization import orthogonal_init_, xavier_init_
from torchagentic.utils.normalization import RunningNorm, LayerNorm2D

try:
    from torchagentic.compile import (
        CompileConfig, compile_model, compile_function,
        optimize_for_inference, optimize_for_training,
        optimize_speed, optimize_memory, is_compiled,
    )
    COMPILE_SUPPORT = True
except ImportError:
    COMPILE_SUPPORT = False

__all__ = [
    # Primitives
    "nn", "integrations",
    "Memory", "MemoryState", "NTMBank", "DNCBank", "SlidingWindowBank",
    "GAE", "TDLambda", "VTrace", "Retrace", "TDLambdaNet",
    "ValueIteration", "MCTSPlanner", "SuccessorRepresentation", "LearnedPrior",
    "RandomNetworkDistillation", "ICM", "CountBonus", "DisagreementEnsemble",
    # Base
    "BaseAgentModel", "ModelConfig",
    "MLPNetwork", "CNNNetwork", "NatureCNN", "ResNetNetwork",
    "RNNNetwork", "LSTMAgent", "GRUAgent",
    # RL
    "DQN", "DuelingDQN", "NoisyDQN",
    "PPOActor", "PPOCritic", "PPOActorCritic",
    "A3CNetwork", "SACActor", "SACCritic", "SACValue",
    "TD3Actor", "TD3Critic",
    # Transformers
    "TransformerAgent", "DecisionTransformer", "PerceiverAgent",
    "SelfAttention", "MultiHeadAttention",
    # Memory (full)
    "NeuralTuringMachine", "DifferentiableNeuralComputer",
    # Multi-agent
    "MultiAgentBase", "MADDPGAgent", "QMIXNetwork", "VDNNetwork",
    # Utils
    "orthogonal_init_", "xavier_init_", "RunningNorm", "LayerNorm2D",
    # Compile
    "CompileConfig", "compile_model", "compile_function",
    "optimize_for_inference", "optimize_for_training",
    "optimize_speed", "optimize_memory", "is_compiled",
    "COMPILE_SUPPORT",
]
