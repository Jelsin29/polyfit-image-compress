"""Benchmark runner for polyfit-image-compress."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    model_name: str = "quadratic"
    block_size: int = 8
    seed: int = 42


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    config: BenchmarkConfig
    image_name: str
    psnr: float
    ssim: float
    mse: float
    compression_ratio: float
    encode_time: float
    decode_time: float


class BenchmarkSuite:
    """Benchmark suite for comparing compression methods."""

    def __init__(self, dataset_path: Path, output_dir: Path) -> None:
        raise NotImplementedError("Stub")

    def run_polynomial(self, configs: list[BenchmarkConfig]) -> list[BenchmarkResult]:
        raise NotImplementedError("Stub")

    def run_baselines(self, target_sizes: list[int]) -> list[BenchmarkResult]:
        raise NotImplementedError("Stub")

    def generate_report(self) -> str:
        raise NotImplementedError("Stub")


if __name__ == "__main__":
    raise NotImplementedError("CLI not yet implemented")
