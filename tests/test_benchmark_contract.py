"""Tests for benchmark helpers that do not require Ray."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_ray_contract import benchmark_payload  # noqa: E402


def test_benchmark_payload_is_deterministic_and_sized() -> None:
    assert benchmark_payload(32) == benchmark_payload(32)
    assert len(benchmark_payload(32)) == 32


def test_benchmark_payload_rejects_non_positive_sizes() -> None:
    with pytest.raises(ValueError):
        benchmark_payload(0)
