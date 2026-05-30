"""
Compilation Benchmark Suite

Benchmarks torch.compile() performance improvements across different models.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import torch

from torchagentic import (
    DQN,
    DuelingDQN,
    PPOActorCritic,
    MLPNetwork,
    NatureCNN,
    ModelConfig,
    compile_model,
    CompileConfig,
    COMPILE_SUPPORT,
)
from torchagentic.nn import (
    Memory, GAE, ValueIteration, RandomNetworkDistillation,
)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    model_name: str
    batch_size: int
    device: str
    uncompiled_ms: float
    compiled_ms: float
    speedup: float
    compile_time_s: float
    config: dict


def run_benchmark(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    num_warmup: int = 50,
    num_runs: int = 200,
) -> tuple[float, float]:
    """Run benchmark and return (uncompiled_time, compiled_time) in ms."""
    
    device = inputs.device
    model.eval()
    
    # Uncompiled benchmark
    for _ in range(num_warmup):
        with torch.inference_mode():
            _ = model(inputs)
    
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        with torch.inference_mode():
            for _ in range(num_runs):
                _ = model(inputs)
        end.record()
        torch.cuda.synchronize()
        uncompiled_ms = start.elapsed_time(end) / num_runs
    else:
        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(num_runs):
                _ = model(inputs)
        uncompiled_ms = (time.perf_counter() - start) * 1000 / num_runs
    
    # Compile model
    start_compile = time.perf_counter()
    compiled_model = compile_model(
        model,
        config=CompileConfig(mode="reduce-overhead"),
        warmup=True,
        example_inputs=(inputs,),
    )
    compile_time_s = time.perf_counter() - start_compile
    
    # Compiled benchmark
    if torch.cuda.is_available():
        start.record()
        with torch.inference_mode():
            for _ in range(num_runs):
                _ = compiled_model(inputs)
        end.record()
        torch.cuda.synchronize()
        compiled_ms = start.elapsed_time(end) / num_runs
    else:
        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(num_runs):
                _ = compiled_model(inputs)
        compiled_ms = (time.perf_counter() - start) * 1000 / num_runs
    
    return uncompiled_ms, compiled_ms, compile_time_s


def benchmark_all_models(
    output_path: Optional[str] = None,
) -> list[BenchmarkResult]:
    """Run benchmarks on all model types."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmarks on {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Compile support: {COMPILE_SUPPORT}")
    
    if not COMPILE_SUPPORT:
        print("torch.compile() not available. Skipping benchmarks.")
        return []
    
    results = []
    
    # MLP benchmarks
    print("\n--- MLP Networks ---")
    for hidden_dims in [[128], [256, 256], [512, 512, 512]]:
        for batch_size in [1, 32, 256]:
            config = ModelConfig(input_dim=64, action_dim=4, hidden_dims=hidden_dims)
            model = MLPNetwork(config).to(device)
            inputs = torch.randn(batch_size, 64, device=device)
            
            uncompiled, compiled, compile_time = run_benchmark(model, inputs)
            speedup = uncompiled / compiled if compiled > 0 else 1.0
            
            result = BenchmarkResult(
                model_name=f"MLP_{hidden_dims}",
                batch_size=batch_size,
                device=device,
                uncompiled_ms=uncompiled,
                compiled_ms=compiled,
                speedup=speedup,
                compile_time_s=compile_time,
                config=config.to_dict(),
            )
            results.append(result)
            
            print(f"  MLP {hidden_dims} (bs={batch_size}): "
                  f"{uncompiled:.3f} -> {compiled:.3f} ms ({speedup:.2f}x)")
    
    # CNN benchmarks
    print("\n--- CNN Networks ---")
    for batch_size in [1, 8, 32]:
        config = ModelConfig(input_dim=4, action_dim=6)
        model = NatureCNN(config, image_shape=(4, 84, 84)).to(device)
        inputs = torch.randn(batch_size, 4, 84, 84, device=device)
        
        uncompiled, compiled, compile_time = run_benchmark(model, inputs)
        speedup = uncompiled / compiled if compiled > 0 else 1.0
        
        result = BenchmarkResult(
            model_name="NatureCNN",
            batch_size=batch_size,
            device=device,
            uncompiled_ms=uncompiled,
            compiled_ms=compiled,
            speedup=speedup,
            compile_time_s=compile_time,
            config=config.to_dict(),
        )
        results.append(result)
        
        print(f"  NatureCNN (bs={batch_size}): "
              f"{uncompiled:.3f} -> {compiled:.3f} ms ({speedup:.2f}x)")
    
    # DQN benchmarks
    print("\n--- DQN Models ---")
    for model_class, name in [(DQN, "DQN"), (DuelingDQN, "DuelingDQN")]:
        for batch_size in [1, 32, 128]:
            config = ModelConfig(input_dim=64, action_dim=4, hidden_dims=[256, 256])
            model = model_class(config, image_input=False).to(device)
            inputs = torch.randn(batch_size, 64, device=device)
            
            uncompiled, compiled, compile_time = run_benchmark(model, inputs)
            speedup = uncompiled / compiled if compiled > 0 else 1.0
            
            result = BenchmarkResult(
                model_name=name,
                batch_size=batch_size,
                device=device,
                uncompiled_ms=uncompiled,
                compiled_ms=compiled,
                speedup=speedup,
                compile_time_s=compile_time,
                config=config.to_dict(),
            )
            results.append(result)
            
            print(f"  {name} (bs={batch_size}): "
                  f"{uncompiled:.3f} -> {compiled:.3f} ms ({speedup:.2f}x)")
    
    # PPO benchmarks
    print("\n--- PPO Models ---")
    for continuous in [False, True]:
        for batch_size in [1, 64, 256]:
            config = ModelConfig(input_dim=32, action_dim=4, hidden_dims=[256, 256])
            model = PPOActorCritic(config, continuous=continuous).to(device)
            inputs = torch.randn(batch_size, 32, device=device)
            
            uncompiled, compiled, compile_time = run_benchmark(model, inputs)
            speedup = uncompiled / compiled if compiled > 0 else 1.0
            
            result = BenchmarkResult(
                model_name=f"PPO_{'continuous' if continuous else 'discrete'}",
                batch_size=batch_size,
                device=device,
                uncompiled_ms=uncompiled,
                compiled_ms=compiled,
                speedup=speedup,
                compile_time_s=compile_time,
                config=config.to_dict(),
            )
            results.append(result)
            
            print(f"  PPO {'cont' if continuous else 'disc'} (bs={batch_size}): "
                  f"{uncompiled:.3f} -> {compiled:.3f} ms ({speedup:.2f}x)")
    
    # ── nn Primitive benchmarks ──────────────────────────────
    print("\n--- nn Primitives ---")

    # Memory (NTM)
    B, S, D = 16, 128, 64
    mem = Memory(num_slots=S, slot_size=D, num_reads=4, backend="ntm").to(device)
    mem_inputs = torch.randn(B, D, device=device)
    uncompiled, compiled, compile_time = run_benchmark(mem, mem_inputs)
    speedup = uncompiled / compiled if compiled > 0 else 1.0
    results.append(BenchmarkResult("nn.Memory(NTM)", B, device, uncompiled, compiled, speedup, compile_time, {}))
    print(f"  nn.Memory(NTM): {uncompiled:.3f} -> {compiled:.3f} ms ({speedup:.2f}x)")

    # GAE
    T, B_gae = 128, 32
    gae = GAE(gamma=0.99, lam=0.95).to(device)
    gae_inputs = (torch.randn(T, B_gae, device=device), torch.randn(T + 1, B_gae, device=device))
    uncompiled, compiled, compile_time = run_benchmark(gae, gae_inputs[0])
    speedup = uncompiled / compiled if compiled > 0 else 1.0
    results.append(BenchmarkResult("nn.GAE", B_gae, device, uncompiled, compiled, speedup, compile_time, {}))
    print(f"  nn.GAE: {uncompiled:.3f} -> {compiled:.3f} ms ({speedup:.2f}x)")

    # ValueIteration
    VI_S, VI_A = 32, 8
    vi = ValueIteration(num_states=VI_S, num_actions=VI_A, num_iters=20).to(device)
    vi_inputs = torch.randn(8, VI_S, VI_A, device=device)
    uncompiled, compiled, compile_time = run_benchmark(vi, vi_inputs)
    speedup = uncompiled / compiled if compiled > 0 else 1.0
    results.append(BenchmarkResult("nn.ValueIteration", 8, device, uncompiled, compiled, speedup, compile_time, {}))
    print(f"  nn.ValueIteration: {uncompiled:.3f} -> {compiled:.3f} ms ({speedup:.2f}x)")

    # Save results
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": device,
            "results": [
                {
                    "model_name": r.model_name,
                    "batch_size": r.batch_size,
                    "uncompiled_ms": round(r.uncompiled_ms, 3),
                    "compiled_ms": round(r.compiled_ms, 3),
                    "speedup": round(r.speedup, 2),
                    "compile_time_s": round(r.compile_time_s, 2),
                }
                for r in results
            ],
        }
        
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
    
    return results


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print benchmark summary."""
    if not results:
        return
    
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    # Group by model type
    by_model: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        if r.model_name not in by_model:
            by_model[r.model_name] = []
        by_model[r.model_name].append(r)
    
    for model_name, model_results in by_model.items():
        avg_speedup = sum(r.speedup for r in model_results) / len(model_results)
        max_speedup = max(r.speedup for r in model_results)
        print(f"\n{model_name}:")
        print(f"  Average speedup: {avg_speedup:.2f}x")
        print(f"  Max speedup: {max_speedup:.2f}x")
    
    # Overall stats
    all_speedups = [r.speedup for r in results]
    print(f"\nOverall:")
    print(f"  Mean speedup: {sum(all_speedups) / len(all_speedups):.2f}x")
    print(f"  Median speedup: {sorted(all_speedups)[len(all_speedups) // 2]:.2f}x")
    print(f"  Max speedup: {max(all_speedups):.2f}x")


def main():
    """Run benchmarks and print summary."""
    print("=" * 60)
    print("TorchAgentic Compilation Benchmark Suite")
    print("=" * 60)
    
    results = benchmark_all_models(output_path="benchmark_results.json")
    print_summary(results)


if __name__ == "__main__":
    main()
