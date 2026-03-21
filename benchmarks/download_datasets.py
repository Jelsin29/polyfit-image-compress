"""Download standard benchmark datasets."""

from __future__ import annotations

import urllib.request
from pathlib import Path

KODAK_BASE_URL = "https://r0k.us/graphics/kodak/kodak"
KODAK_COUNT = 24
# Minimum expected file size in bytes for a valid Kodak image (~100KB)
KODAK_MIN_SIZE = 100_000


def download_kodak(output_dir: Path) -> None:
    """Download the Kodak dataset (24 images, 768x512).

    Images are downloaded from r0k.us/graphics/kodak/ as kodim01.png through kodim24.png.
    Files that already exist and meet the minimum size requirement are skipped.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Kodak dataset to {output_dir}/")
    downloaded = 0
    skipped = 0

    for i in range(1, KODAK_COUNT + 1):
        filename = f"kodim{i:02d}.png"
        filepath = output_dir / filename
        url = f"{KODAK_BASE_URL}/{filename}"

        if filepath.exists() and filepath.stat().st_size >= KODAK_MIN_SIZE:
            skipped += 1
            continue

        print(f"  [{i:2d}/{KODAK_COUNT}] Downloading {filename}...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, filepath)
            size = filepath.stat().st_size
            if size < KODAK_MIN_SIZE:
                print(
                    f"WARNING: {filename} is only {size} bytes "
                    f"(expected >= {KODAK_MIN_SIZE}). File may be corrupt."
                )
            else:
                print(f"OK ({size:,} bytes)")
            downloaded += 1
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nDone. Downloaded: {downloaded}, Skipped (already exist): {skipped}")
    total = len(list(output_dir.glob("kodim*.png")))
    print(f"Total Kodak images in {output_dir}: {total}/{KODAK_COUNT}")


def download_set14(output_dir: Path) -> None:
    """Download the Set14 dataset.

    Note: Set14 does not have a single canonical download URL.
    This function provides instructions for manual download.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(output_dir.glob("*.png")) + list(output_dir.glob("*.bmp"))
    if len(existing) >= 14:
        print(f"Set14 dataset already present ({len(existing)} images in {output_dir})")
        return

    print("Set14 dataset requires manual download.")
    print("Please download from one of these sources:")
    print("  - https://github.com/jbhuang0604/SelfExSR/tree/master/data/Set14")
    print(f"  - Place images in: {output_dir}/")
    print()
    print("After downloading, convert to PNG if needed and ensure 14 images are present.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument(
        "--dataset",
        default="kodak",
        choices=["kodak", "set14", "all"],
        help="Which dataset to download (default: kodak)",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/datasets",
        help="Base output directory (default: benchmarks/datasets)",
    )
    args = parser.parse_args()

    output_base = Path(args.output)

    if args.dataset in ("kodak", "all"):
        download_kodak(output_base / "kodak")

    if args.dataset in ("set14", "all"):
        download_set14(output_base / "set14")
