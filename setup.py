"""
TorchAgentic - PyTorch Model Definitions for AI Agents.

A library of neural network architectures for building trainable AI agents,
including RL agents, transformer-based models, memory-augmented networks,
and multi-agent systems.
"""

from setuptools import setup, find_packages
from pathlib import Path

readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="torchagentic",
    version="0.2.0",
    author="Py-AI-Dev",
    author_email="contact@liodon.ai",
    description="Differentiable cognitive primitives for agentic AI - Memory, Credit Assignment, Planning, Exploration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/liodon-ai/torchagentic",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
            "gymnasium>=0.29.0",
        ],
        "transformers": [
            "einops>=0.7.0",
        ],
        "full": [
            "einops>=0.7.0",
            "gymnasium>=0.29.0",
            "tensorboard>=2.15.0",
        ],
    },
    keywords=[
        "pytorch",
        "reinforcement-learning",
        "deep-learning",
        "agents",
        "neural-networks",
        "transformers",
        "multi-agent",
        "rl",
        "dqn",
        "ppo",
        "sac",
        "memory-networks",
    ],
    include_package_data=True,
    package_data={"torchagentic": ["py.typed"]},
    zip_safe=False,
)
