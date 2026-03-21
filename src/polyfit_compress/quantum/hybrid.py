"""Hybrid classical-quantum compression pipeline."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from polyfit_compress.quantum import require_quantum


class HybridCompressor:
    """Compressor that uses quantum solver for coefficient computation."""

    def __init__(self, block_size: int = 4, model_type: str = "quadratic") -> None:
        require_quantum()
        raise NotImplementedError

    def compress(self, image: NDArray) -> dict:
        """Compress using hybrid classical-quantum pipeline."""
        raise NotImplementedError
