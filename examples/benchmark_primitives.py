"""
Benchmark: torch.compile() speedup for nn primitives.

Measures eager vs compiled throughput for Memory, GAE, ValueIteration,
and RND. Run with:
    python examples/benchmark_primitives.py
"""

import time
import torch
from torchagentic.nn import (
    Memory, GAE, ValueIteration, RandomNetworkDistillation,
)


def benchmark(fn, num_warmup=10, num_runs=50, description=""):
    for _ in range(num_warmup):
        fn()

    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_runs):
            fn()
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / num_runs
    else:
        start = time.perf_counter()
        for _ in range(num_runs):
            fn()
        elapsed = (time.perf_counter() - start) * 1000 / num_runs

    return elapsed


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  PyTorch {torch.__version__}")
    print(f"{'Primitive':<25} {'Eager (ms)':<12} {'Compiled (ms)':<14} {'Speedup':<8}")
    print("-" * 60)

    has_compile = hasattr(torch, "compile")

    # ── Memory (NTM) ────────────────────────────────────
    B, S, D = 16, 128, 64
    mem = Memory(num_slots=S, slot_size=D, num_reads=4, backend="ntm").to(device)
    x = torch.randn(B, D, device=device)

    def run_memory():
        mem(x)
        mem.reset(B)

    eager = benchmark(run_memory)
    if has_compile:
        mem_compiled = torch.compile(mem, mode="reduce-overhead")
        compiled = benchmark(lambda: (mem_compiled(x), mem_compiled.reset(B)))
        speedup = eager / compiled
    else:
        compiled = speedup = 0.0
    print(f"{'Memory (NTM)':<25} {eager:<12.3f} {compiled:<14.3f} {speedup:<8.2f}x")

    # ── GAE ─────────────────────────────────────────────
    T, B = 128, 32
    gae = GAE(gamma=0.99, lam=0.95).to(device)

    def run_gae():
        rewards = torch.randn(T, B, device=device)
        values = torch.randn(T + 1, B, device=device)
        dones = torch.zeros(T, B, device=device)
        gae(rewards, values, dones)

    eager = benchmark(run_gae)
    if has_compile:
        gae_compiled = torch.compile(gae, mode="reduce-overhead")
        compiled = benchmark(lambda: (
            gae_compiled(
                torch.randn(T, B, device=device),
                torch.randn(T + 1, B, device=device),
            )
        ))
        speedup = eager / compiled
    else:
        compiled = speedup = 0.0
    print(f"{'GAE':<25} {eager:<12.3f} {compiled:<14.3f} {speedup:<8.2f}x")

    # ── ValueIteration ──────────────────────────────────
    VI_S, VI_A = 32, 8
    vi = ValueIteration(num_states=VI_S, num_actions=VI_A, num_iters=20).to(device)

    def run_vi():
        B = 8
        reward = torch.randn(B, VI_S, VI_A, device=device)
        kernel = torch.randn(B, VI_S, VI_A, VI_S, device=device)
        kernel = torch.nn.functional.softmax(
            kernel.reshape(B, VI_S * VI_A, VI_S), dim=-1
        ).reshape(B, VI_S, VI_A, VI_S)
        vi(reward, kernel)

    eager = benchmark(run_vi)
    if has_compile:
        vi_compiled = torch.compile(vi, mode="reduce-overhead")
        compiled = benchmark(lambda: (
            vi_compiled(
                torch.randn(8, VI_S, VI_A, device=device),
                torch.nn.functional.softmax(
                    torch.randn(8, VI_S, VI_A, VI_S, device=device)
                    .reshape(8, VI_S * VI_A, VI_S), dim=-1
                ).reshape(8, VI_S, VI_A, VI_S),
            )
        ))
        speedup = eager / compiled
    else:
        compiled = speedup = 0.0
    print(f"{'ValueIteration':<25} {eager:<12.3f} {compiled:<14.3f} {speedup:<8.2f}x")

    # ── RND ─────────────────────────────────────────────
    rnd = RandomNetworkDistillation(state_dim=256, hidden_dim=512).to(device)

    def run_rnd():
        states = torch.randn(64, 256, device=device)
        rnd(states)

    eager = benchmark(run_rnd)
    if has_compile:
        rnd_compiled = torch.compile(rnd, mode="reduce-overhead")
        compiled = benchmark(lambda: (
            rnd_compiled(torch.randn(64, 256, device=device))
        ))
        speedup = eager / compiled
    else:
        compiled = speedup = 0.0
    print(f"{'RND':<25} {eager:<12.3f} {compiled:<14.3f} {speedup:<8.2f}x")

    print("-" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
