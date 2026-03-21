"""I/O module for .pfic file format serialization and deserialization."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from numpy.typing import NDArray

from polyfit_compress.models import LinearModel, QuadraticModel

# Constants
MAGIC_BYTES = b"PFIC"
FORMAT_VERSION = 1
HEADER_FORMAT = "<4sBIIBHBHH12s"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# Model type registry
MODEL_TYPE_MAP: dict[int, type] = {
    0: LinearModel,
    1: QuadraticModel,
}
MODEL_TYPE_REVERSE: dict[type, int] = {v: k for k, v in MODEL_TYPE_MAP.items()}


@dataclass(frozen=True)
class PficHeader:
    """Header for .pfic file format."""

    magic: bytes
    version: int
    height: int
    width: int
    channels: int
    block_size: int
    model_type: int
    padding_height: int
    padding_width: int

    def pack(self) -> bytes:
        """Pack header into bytes."""
        raise NotImplementedError("Stub — implement in phase2/io")

    @classmethod
    def unpack(cls, data: bytes) -> PficHeader:
        """Unpack header from bytes."""
        raise NotImplementedError("Stub — implement in phase2/io")


def save_pfic(path: Path | str, result: object) -> None:
    """Save compression result to .pfic file.

    Parameters
    ----------
    path : Path or str
        Output file path.
    result : CompressionResult
        Compression result containing coefficients and metadata.
    """
    raise NotImplementedError("Stub — implement in phase2/io")


def load_pfic(path: Path | str) -> tuple[PficHeader, NDArray]:
    """Load .pfic file and return header and coefficients.

    Parameters
    ----------
    path : Path or str
        Input .pfic file path.

    Returns
    -------
    tuple[PficHeader, NDArray]
        Header metadata and coefficient array.
    """
    raise NotImplementedError("Stub — implement in phase2/io")
