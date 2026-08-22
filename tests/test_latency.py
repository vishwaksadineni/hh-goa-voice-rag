import pytest
import asyncio
from rag.benchmark import LatencyBenchmarkSuite
from rag.config import settings

@pytest.mark.asyncio
async def test_sub_200ms_sla_and_percentiles():
    suite = LatencyBenchmarkSuite()
    # Run benchmark over test suite
    summary = await suite.run_benchmark(strategy="hierarchical", warmup_queries=2)
    
    # Assertions on Technical Requirements
    assert summary.total_queries >= 30
    assert summary.p50_latency_ms < settings.latency_target_ms, f"P50 {summary.p50_latency_ms}ms exceeds 200ms"
    assert summary.p70_latency_ms < settings.latency_target_ms, f"P70 {summary.p70_latency_ms}ms exceeds 200ms"
    assert summary.p100_max_latency_ms < settings.latency_target_ms, f"P100 {summary.p100_max_latency_ms}ms exceeds 200ms"
    assert summary.sub_200ms_compliance_pct == 100.0
