"""Download standard benchmark datasets."""

from __future__ import annotations

from pathlib import Path


def download_kodak(output_dir: Path) -> None:
    """Download the Kodak dataset (24 images, 768x512)."""
    raise NotImplementedError("Stub")


def download_set14(output_dir: Path) -> None:
    """Download the Set14 dataset."""
    raise NotImplementedError("Stub")


if __name__ == "__main__":
    raise NotImplementedError("CLI not yet implemented")
