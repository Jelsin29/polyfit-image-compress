"""Hybrid classical-quantum compression pipeline."""

from __future__ import annotations

from numpy.typing import NDArray

from polyfit_compress.quantum import require_quantum


class HybridCompressor:
    """Compressor that uses quantum solver for coefficient computation."""

    def __init__(self, block_size: int = 4, model_type: str = "quadratic") -> None:
        """Not yet implemented — coming in Phase 3."""
        require_quantum()
        raise NotImplementedError("HybridCompressor not yet implemented — coming in Phase 3")

    def compress(self, image: NDArray) -> dict:
        """Compress using hybrid classical-quantum pipeline.

        Not yet implemented — coming in Phase 3.
        """
        raise NotImplementedError(
            "HybridCompressor.compress not yet implemented — coming in Phase 3"
        )
