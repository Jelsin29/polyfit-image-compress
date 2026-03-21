"""File I/O for .pfic (PolyFit Image Compressed) format."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from polyfit_compress.compressor import CompressionResult


MAGIC_BYTES = b"PFIC"
FORMAT_VERSION = 1


def save(path: str | Path, result: CompressionResult) -> None:
    """Save compression result to .pfic file.

    Not yet implemented — coming in Phase 2.
    """
    raise NotImplementedError("save() is not yet implemented — coming in Phase 2")


def load(path: str | Path) -> dict:
    """Load a .pfic file and return metadata + coefficients.

    Not yet implemented — coming in Phase 2.
    """
    raise NotImplementedError("load() is not yet implemented — coming in Phase 2")
