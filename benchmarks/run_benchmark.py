"""Benchmark runner for polyfit-image-compress."""

from __future__ import annotations

import csv
import io
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from polyfit_compress.compressor import LeastSquaresCompressor
from polyfit_compress.metrics import compression_ratio as calc_cr
from polyfit_compress.metrics import mse as calc_mse
from polyfit_compress.metrics import psnr as calc_psnr
from polyfit_compress.metrics import ssim as calc_ssim
from polyfit_compress.models import LinearModel, QuadraticModel

MODEL_REGISTRY: dict[str, type] = {
    "linear": LinearModel,
    "quadratic": QuadraticModel,
}

# Default polynomial configurations from the benchmarking strategy
DEFAULT_CONFIGS: list[BenchmarkConfig] = []  # populated after class definition


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
    image_path: str
    method: str
    psnr: float
    ssim: float
    mse: float
    compression_ratio: float
    compressed_size_bytes: int
    original_size_bytes: int
    encode_time: float
    decode_time: float


# Populate default configs after class is defined
DEFAULT_CONFIGS = [
    BenchmarkConfig(model_name="linear", block_size=4),
    BenchmarkConfig(model_name="linear", block_size=8),
    BenchmarkConfig(model_name="linear", block_size=16),
    BenchmarkConfig(model_name="quadratic", block_size=4),
    BenchmarkConfig(model_name="quadratic", block_size=8),
    BenchmarkConfig(model_name="quadratic", block_size=16),
]


def _load_grayscale(image_path: Path) -> NDArray[np.uint8]:
    """Load an image as grayscale uint8 array."""
    img = Image.open(image_path).convert("L")
    return np.array(img, dtype=np.uint8)


def _find_images(dataset_path: Path) -> list[Path]:
    """Find all PNG/JPG/BMP images in a directory."""
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    images = sorted(
        p for p in dataset_path.iterdir() if p.is_file() and p.suffix.lower() in extensions
    )
    return images


def _jpeg_at_target_size(
    image: NDArray[np.uint8], target_bytes: int
) -> tuple[NDArray[np.uint8], int, float]:
    """Binary search for JPEG quality that produces ~target_bytes.

    Returns (reconstructed_image, actual_size, encode_time).
    """
    pil_img = Image.fromarray(image, mode="L")
    lo, hi = 1, 100
    best_quality = 50
    best_diff = float("inf")

    # Binary search for closest quality
    for _ in range(15):
        mid = (lo + hi) // 2
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=mid)
        size = buf.tell()
        diff = abs(size - target_bytes)

        if diff < best_diff:
            best_diff = diff
            best_quality = mid

        if size < target_bytes:
            lo = mid + 1
        elif size > target_bytes:
            hi = mid - 1
        else:
            break

        if lo > hi:
            break

    # Final encode with best quality, timed
    buf = io.BytesIO()
    t0 = time.perf_counter()
    pil_img.save(buf, format="JPEG", quality=best_quality)
    encode_time = time.perf_counter() - t0

    actual_size = buf.tell()
    buf.seek(0)
    reconstructed = np.array(Image.open(buf).convert("L"), dtype=np.uint8)
    return reconstructed, actual_size, encode_time


def _svd_at_target_size(
    image: NDArray[np.uint8], target_bytes: int
) -> tuple[NDArray[np.uint8], int, float]:
    """Find SVD rank that produces ~target_bytes of coefficients.

    SVD storage: rank * (rows + cols + 1) * 4 bytes (float32).

    Returns (reconstructed_image, actual_size, encode_time).
    """
    img_float = image.astype(np.float64)
    rows, cols = img_float.shape

    t0 = time.perf_counter()
    u, s, vt = np.linalg.svd(img_float, full_matrices=False)
    svd_time = time.perf_counter() - t0

    max_rank = min(rows, cols)

    # Find rank that gives closest to target size
    # Storage per rank: rank * (rows + cols + 1) * 4 bytes
    best_rank = 1
    best_diff = float("inf")

    for rank in range(1, max_rank + 1):
        size = rank * (rows + cols + 1) * 4
        diff = abs(size - target_bytes)
        if diff < best_diff:
            best_diff = diff
            best_rank = rank
        if size > target_bytes:
            break

    # Reconstruct with best rank
    t0 = time.perf_counter()
    reconstructed = (u[:, :best_rank] * s[:best_rank]) @ vt[:best_rank, :]
    encode_time = svd_time + (time.perf_counter() - t0)

    actual_size = best_rank * (rows + cols + 1) * 4
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    return reconstructed, actual_size, encode_time


class BenchmarkSuite:
    """Benchmark suite for comparing compression methods."""

    def __init__(self, dataset_path: Path, output_dir: Path) -> None:
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._results: list[BenchmarkResult] = []

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset path does not exist: {self.dataset_path}\n"
                "Run 'python benchmarks/download_datasets.py' first."
            )

    def run_polynomial(self, configs: list[BenchmarkConfig]) -> list[BenchmarkResult]:
        """Run polynomial compression benchmarks over all images in the dataset."""
        images = _find_images(self.dataset_path)
        if not images:
            raise FileNotFoundError(
                f"No images found in {self.dataset_path}. "
                "Run 'python benchmarks/download_datasets.py' first."
            )

        results: list[BenchmarkResult] = []

        for config in configs:
            model_cls = MODEL_REGISTRY.get(config.model_name)
            if model_cls is None:
                print(f"  [WARN] Unknown model '{config.model_name}', skipping.")
                continue

            model = model_cls()
            compressor = LeastSquaresCompressor(model=model, block_size=config.block_size)
            method_label = f"poly_{config.model_name}_{config.block_size}x{config.block_size}"

            print(f"  Running {method_label}...")

            for img_path in images:
                image = _load_grayscale(img_path)
                original_size = image.shape[0] * image.shape[1]

                # Time compression
                t0 = time.perf_counter()
                result = compressor.compress(image)
                encode_time = time.perf_counter() - t0

                reconstructed = result.reconstructed

                # Time decompression
                t0 = time.perf_counter()
                _ = compressor.decompress(
                    result.coefficients, result.original_shape, result.padded_shape
                )
                decode_time = time.perf_counter() - t0

                # Compute metrics
                psnr_val = calc_psnr(image, reconstructed)
                ssim_val = calc_ssim(image, reconstructed)
                mse_val = calc_mse(image, reconstructed)
                cr_val = calc_cr(original_size, result.compressed_size_bytes)

                bench_result = BenchmarkResult(
                    config=config,
                    image_name=img_path.name,
                    image_path=str(img_path),
                    method=method_label,
                    psnr=psnr_val,
                    ssim=ssim_val,
                    mse=mse_val,
                    compression_ratio=cr_val,
                    compressed_size_bytes=result.compressed_size_bytes,
                    original_size_bytes=original_size,
                    encode_time=encode_time,
                    decode_time=decode_time,
                )
                results.append(bench_result)

        self._results.extend(results)
        return results

    def run_baselines(self, target_sizes: list[int] | None = None) -> list[BenchmarkResult]:
        """Compare against JPEG and SVD at matched file sizes.

        If target_sizes is None, uses compressed sizes from previous polynomial runs.
        """
        images = _find_images(self.dataset_path)
        if not images:
            raise FileNotFoundError(f"No images found in {self.dataset_path}.")

        # Derive target sizes from polynomial results if not provided
        if target_sizes is None:
            if not self._results:
                print("  [WARN] No polynomial results yet. Using default target sizes.")
                target_sizes = [10000, 50000, 100000]
            else:
                # Use unique compressed sizes from polynomial runs
                seen: set[int] = set()
                target_sizes = []
                for r in self._results:
                    if r.compressed_size_bytes not in seen:
                        seen.add(r.compressed_size_bytes)
                        target_sizes.append(r.compressed_size_bytes)

        results: list[BenchmarkResult] = []

        for target_size in target_sizes:
            for img_path in images:
                image = _load_grayscale(img_path)
                original_size = image.shape[0] * image.shape[1]

                # --- JPEG baseline ---
                jpeg_config = BenchmarkConfig(model_name="jpeg", block_size=0)
                try:
                    jpeg_recon, jpeg_size, jpeg_enc_time = _jpeg_at_target_size(image, target_size)
                    results.append(
                        BenchmarkResult(
                            config=jpeg_config,
                            image_name=img_path.name,
                            image_path=str(img_path),
                            method=f"jpeg_target{target_size}",
                            psnr=calc_psnr(image, jpeg_recon),
                            ssim=calc_ssim(image, jpeg_recon),
                            mse=calc_mse(image, jpeg_recon),
                            compression_ratio=calc_cr(original_size, jpeg_size),
                            compressed_size_bytes=jpeg_size,
                            original_size_bytes=original_size,
                            encode_time=jpeg_enc_time,
                            decode_time=0.0,
                        )
                    )
                except Exception as e:
                    print(f"  [WARN] JPEG baseline failed for {img_path.name}: {e}")

                # --- SVD baseline ---
                svd_config = BenchmarkConfig(model_name="svd", block_size=0)
                try:
                    svd_recon, svd_size, svd_enc_time = _svd_at_target_size(image, target_size)
                    results.append(
                        BenchmarkResult(
                            config=svd_config,
                            image_name=img_path.name,
                            image_path=str(img_path),
                            method=f"svd_target{target_size}",
                            psnr=calc_psnr(image, svd_recon),
                            ssim=calc_ssim(image, svd_recon),
                            mse=calc_mse(image, svd_recon),
                            compression_ratio=calc_cr(original_size, svd_size),
                            compressed_size_bytes=svd_size,
                            original_size_bytes=original_size,
                            encode_time=svd_enc_time,
                            decode_time=0.0,
                        )
                    )
                except Exception as e:
                    print(f"  [WARN] SVD baseline failed for {img_path.name}: {e}")

        self._results.extend(results)
        return results

    def generate_report(self) -> str:
        """Produce markdown table and save CSV of all results."""
        if not self._results:
            return "No results to report."

        timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

        # --- Save CSV ---
        csv_path = self.output_dir / f"{timestamp}_raw_data.csv"
        fieldnames = [
            "method",
            "image_name",
            "model_name",
            "block_size",
            "psnr",
            "ssim",
            "mse",
            "compression_ratio",
            "compressed_size_bytes",
            "original_size_bytes",
            "encode_time",
            "decode_time",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self._results:
                writer.writerow(
                    {
                        "method": r.method,
                        "image_name": r.image_name,
                        "model_name": r.config.model_name,
                        "block_size": r.config.block_size,
                        "psnr": f"{r.psnr:.2f}",
                        "ssim": f"{r.ssim:.4f}",
                        "mse": f"{r.mse:.2f}",
                        "compression_ratio": f"{r.compression_ratio:.2f}",
                        "compressed_size_bytes": r.compressed_size_bytes,
                        "original_size_bytes": r.original_size_bytes,
                        "encode_time": f"{r.encode_time:.4f}",
                        "decode_time": f"{r.decode_time:.4f}",
                    }
                )

        # --- Build markdown report ---
        # Aggregate per method: mean +/- std
        from collections import defaultdict

        method_metrics: dict[str, list[BenchmarkResult]] = defaultdict(list)
        for r in self._results:
            method_metrics[r.method].append(r)

        lines: list[str] = []
        lines.append(f"# Benchmark Report — {timestamp}")
        lines.append("")
        lines.append("| Method | CR | PSNR (dB) | SSIM | MSE | Encode (s) | Decode (s) | Images |")
        lines.append("| ------ | -- | --------- | ---- | --- | ---------- | ---------- | ------ |")

        for method, method_results in sorted(method_metrics.items()):
            psnrs = [r.psnr for r in method_results if r.psnr != float("inf")]
            ssims = [r.ssim for r in method_results]
            mses = [r.mse for r in method_results]
            crs = [r.compression_ratio for r in method_results]
            enc_times = [r.encode_time for r in method_results]
            dec_times = [r.decode_time for r in method_results]
            n = len(method_results)

            def _fmt(values: list[float]) -> str:
                if not values:
                    return "N/A"
                mean = np.mean(values)
                std = np.std(values)
                return f"{mean:.2f} +/- {std:.2f}"

            lines.append(
                f"| {method} "
                f"| {_fmt(crs)} "
                f"| {_fmt(psnrs)} "
                f"| {_fmt(ssims)} "
                f"| {_fmt(mses)} "
                f"| {_fmt(enc_times)} "
                f"| {_fmt(dec_times)} "
                f"| {n} |"
            )

        lines.append("")
        lines.append(f"Raw data: `{csv_path.name}`")

        report = "\n".join(lines)

        # Save markdown report
        report_path = self.output_dir / f"{timestamp}_report.md"
        report_path.write_text(report)

        print(f"  Report saved to {report_path}")
        print(f"  CSV saved to {csv_path}")

        return report


def _resolve_configs(config_filter: str) -> list[BenchmarkConfig]:
    """Resolve config filter string to list of BenchmarkConfig."""
    if config_filter == "all":
        return DEFAULT_CONFIGS

    configs = []
    for name in config_filter.split(","):
        name = name.strip().lower()
        if name in MODEL_REGISTRY:
            configs.extend(c for c in DEFAULT_CONFIGS if c.model_name == name)
        else:
            print(f"  [WARN] Unknown config filter '{name}', skipping.")
    return configs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run compression benchmarks")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--output", default="benchmarks/results", help="Output directory")
    parser.add_argument(
        "--configs",
        default="all",
        help="Configs to run: all, linear, quadratic (comma-separated)",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip baseline comparisons (JPEG, SVD)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output)
    configs = _resolve_configs(args.configs)

    if not configs:
        print("No valid configs selected. Use --configs all|linear|quadratic")
        raise SystemExit(1)

    print(f"Dataset: {dataset_path}")
    print(f"Output:  {output_dir}")
    print(f"Configs: {len(configs)} configurations")
    print()

    suite = BenchmarkSuite(dataset_path=dataset_path, output_dir=output_dir)

    print("[1/3] Running polynomial benchmarks...")
    poly_results = suite.run_polynomial(configs)
    print(f"  {len(poly_results)} results collected.")
    print()

    if not args.skip_baselines:
        print("[2/3] Running baseline comparisons (JPEG, SVD)...")
        baseline_results = suite.run_baselines()
        print(f"  {len(baseline_results)} baseline results collected.")
        print()
    else:
        print("[2/3] Baselines skipped.")
        print()

    print("[3/3] Generating report...")
    report = suite.generate_report()
    print()
    print(report)
