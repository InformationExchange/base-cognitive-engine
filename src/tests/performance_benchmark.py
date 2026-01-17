"""
BAIS Cognitive Governance Engine v48.0
Performance Benchmark Suite

Measures latency, throughput, and scalability.
"""

import asyncio
import time
import statistics
from datetime import datetime
from typing import Dict, List, Any
import sys


class BenchmarkResults:
    def __init__(self):
        self.latencies: Dict[str, List[float]] = {}
        self.throughput: Dict[str, float] = {}
        self.memory_usage: Dict[str, float] = {}
    
    def add_latency(self, component: str, latency_ms: float):
        if component not in self.latencies:
            self.latencies[component] = []
        self.latencies[component].append(latency_ms)
    
    def get_stats(self, component: str) -> Dict[str, float]:
        if component not in self.latencies:
            return {}
        data = self.latencies[component]
        return {
            "min": min(data),
            "max": max(data),
            "mean": statistics.mean(data),
            "median": statistics.median(data),
            "p95": sorted(data)[int(len(data) * 0.95)] if len(data) >= 20 else max(data),
            "count": len(data)
        }


def benchmark_component(results: BenchmarkResults, name: str, func, iterations: int = 10):
    """Benchmark a component."""
    for _ in range(iterations):
        start = time.time()
        try:
            func()
        except Exception:
            pass
        latency = (time.time() - start) * 1000
        results.add_latency(name, latency)


async def benchmark_async_component(results: BenchmarkResults, name: str, func, iterations: int = 10):
    """Benchmark an async component."""
    for _ in range(iterations):
        start = time.time()
        try:
            await func()
        except Exception:
            pass
        latency = (time.time() - start) * 1000
        results.add_latency(name, latency)


def run_benchmarks():
    """Run all performance benchmarks."""
    print("=" * 70)
    print("BAIS v48.0 PERFORMANCE BENCHMARK SUITE")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    
    results = BenchmarkResults()
    
    # Initialize engine
    print("\n[1] Initializing Engine")
    print("-" * 60)
    
    init_start = time.time()
    from core.integrated_engine import IntegratedGovernanceEngine
    engine = IntegratedGovernanceEngine()
    init_time = (time.time() - init_start) * 1000
    print(f"  Engine initialization: {init_time:.0f}ms")
    results.add_latency("initialization", init_time)
    
    # Benchmark core operations
    print("\n[2] Core Operation Benchmarks (10 iterations each)")
    print("-" * 60)
    
    loop = asyncio.get_event_loop()
    
    # Simple evaluation
    async def simple_eval():
        return await engine.evaluate(
            query="What is 2+2?",
            response="2+2 equals 4.",
            documents=[]
        )
    
    loop.run_until_complete(benchmark_async_component(
        results, "simple_evaluate", simple_eval, 10
    ))
    stats = results.get_stats("simple_evaluate")
    print(f"  Simple evaluate: mean={stats['mean']:.0f}ms, p95={stats['p95']:.0f}ms")
    
    # Complex evaluation with documents
    async def complex_eval():
        return await engine.evaluate(
            query="What is the capital of France?",
            response="The capital of France is Paris, a city known for the Eiffel Tower.",
            documents=[
                {"content": "Paris is the capital and largest city of France."},
                {"content": "The Eiffel Tower is located in Paris."}
            ]
        )
    
    loop.run_until_complete(benchmark_async_component(
        results, "complex_evaluate", complex_eval, 10
    ))
    stats = results.get_stats("complex_evaluate")
    print(f"  Complex evaluate: mean={stats['mean']:.0f}ms, p95={stats['p95']:.0f}ms")
    
    # Evaluate and improve
    async def eval_improve():
        return await engine.evaluate_and_improve(
            query="Tell me about AI",
            response="AI is great.",
            documents=[]
        )
    
    loop.run_until_complete(benchmark_async_component(
        results, "evaluate_improve", eval_improve, 5
    ))
    stats = results.get_stats("evaluate_improve")
    print(f"  Evaluate & improve: mean={stats['mean']:.0f}ms, p95={stats['p95']:.0f}ms")
    
    # Benchmark operational components
    print("\n[3] Operational Component Benchmarks")
    print("-" * 60)
    
    # Monitoring
    if engine.monitoring_engine:
        def monitor_op():
            engine.monitoring_engine.record_metric("test_metric", 42.5)
            engine.monitoring_engine.check_alerts()
        benchmark_component(results, "monitoring", monitor_op, 100)
        stats = results.get_stats("monitoring")
        print(f"  Monitoring (100 ops): mean={stats['mean']:.2f}ms")
    
    # Configuration
    if engine.configuration_engine:
        from core.configuration_management import ConfigParameter, ConfigScope
        def config_op():
            param = ConfigParameter("test_threshold", 0.5, 0.5, ConfigScope.GLOBAL, "Test")
            engine.configuration_engine.set_parameter(param)
        benchmark_component(results, "configuration", config_op, 100)
        stats = results.get_stats("configuration")
        print(f"  Configuration (100 ops): mean={stats['mean']:.2f}ms")
    
    # Logging
    if engine.logging_engine:
        from core.logging_telemetry import LogLevel
        def log_op():
            engine.logging_engine.log(LogLevel.INFO, "Test message", "benchmark")
        benchmark_component(results, "logging", log_op, 100)
        stats = results.get_stats("logging")
        print(f"  Logging (100 ops): mean={stats['mean']:.2f}ms")
    
    # Workflow
    if engine.workflow_engine:
        def workflow_op():
            engine.workflow_engine.execute_workflow("governance_evaluation", {})
        benchmark_component(results, "workflow", workflow_op, 10)
        stats = results.get_stats("workflow")
        print(f"  Workflow (10 ops): mean={stats['mean']:.2f}ms")
    
    # API Gateway
    if engine.api_gateway:
        def gateway_op():
            engine.api_gateway.handle_request("/api/evaluate", "POST", "bench_client")
        benchmark_component(results, "api_gateway", gateway_op, 100)
        stats = results.get_stats("api_gateway")
        print(f"  API Gateway (100 ops): mean={stats['mean']:.2f}ms")
    
    # Throughput test
    print("\n[4] Throughput Test (100 sequential evaluations)")
    print("-" * 60)
    
    throughput_start = time.time()
    throughput_count = 0
    
    async def throughput_test():
        nonlocal throughput_count
        for _ in range(100):
            await engine.evaluate(
                query="Quick test",
                response="Quick response",
                documents=[]
            )
            throughput_count += 1
    
    loop.run_until_complete(throughput_test())
    throughput_time = time.time() - throughput_start
    throughput_rps = throughput_count / throughput_time
    
    print(f"  Completed: {throughput_count} evaluations in {throughput_time:.1f}s")
    print(f"  Throughput: {throughput_rps:.1f} evaluations/second")
    results.throughput["evaluations"] = throughput_rps
    
    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    print("\n[Latency Summary]")
    print(f"  {'Component':<25} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Count':<8}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*8}")
    
    for component in ["initialization", "simple_evaluate", "complex_evaluate", 
                      "evaluate_improve", "monitoring", "configuration", 
                      "logging", "workflow", "api_gateway"]:
        stats = results.get_stats(component)
        if stats:
            print(f"  {component:<25} {stats['mean']:<12.1f} {stats['p95']:<12.1f} {stats['count']:<8}")
    
    print(f"\n[Throughput]")
    print(f"  Evaluations: {results.throughput.get('evaluations', 0):.1f} req/sec")
    
    # Performance grade
    simple_mean = results.get_stats("simple_evaluate").get("mean", 1000)
    if simple_mean < 100:
        grade = "A (Excellent)"
    elif simple_mean < 500:
        grade = "B (Good)"
    elif simple_mean < 1000:
        grade = "C (Acceptable)"
    else:
        grade = "D (Needs Optimization)"
    
    print(f"\n[Performance Grade]: {grade}")
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_benchmarks()
