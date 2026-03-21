"""I/O module for .pfic file format serialization and deserialization."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from polyfit_compress.compressor import CompressionResult
from polyfit_compress.exceptions import CorruptedFileError, IncompatibleVersionError
from polyfit_compress.models import LinearModel, QuadraticModel

# Constants
MAGIC_BYTES = b"PFIC"
FORMAT_VERSION = 1
HEADER_FORMAT = "<4sBIIBHBII6s"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# Model type registry
MODEL_TYPE_MAP: dict[int, type] = {
    0: LinearModel,
    1: QuadraticModel,
}
MODEL_TYPE_REVERSE: dict[type, int] = {v: k for k, v in MODEL_TYPE_MAP.items()}


@dataclass(frozen=True)
class PficHeader:
    """Header for .pfic file format.

    Attributes
    ----------
    magic : bytes
        Magic bytes identifying the file format (``b'PFIC'``).
    version : int
        Format version number.
    height : int
        Original image height.
    width : int
        Original image width.
    channels : int
        Number of color channels (1 for grayscale, 3 for RGB).
    block_size : int
        Block size used during compression.
    model_type : int
        Polynomial model type (0=linear, 1=quadratic).
    padding_height : int
        Padded image height.
    padding_width : int
        Padded image width.
    """

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
        """Pack header into bytes.

        Returns
        -------
        bytes
            Packed header of exactly ``HEADER_SIZE`` bytes.
        """
        return struct.pack(
            HEADER_FORMAT,
            self.magic,
            self.version,
            self.height,
            self.width,
            self.channels,
            self.block_size,
            self.model_type,
            self.padding_height,
            self.padding_width,
            b"\x00" * 6,  # reserved
        )

    @classmethod
    def unpack(cls, data: bytes) -> PficHeader:
        """Unpack header from bytes.

        Parameters
        ----------
        data : bytes
            Raw bytes of at least ``HEADER_SIZE`` length.

        Returns
        -------
        PficHeader
            Parsed header.

        Raises
        ------
        CorruptedFileError
            If magic bytes are not ``b'PFIC'``.
        IncompatibleVersionError
            If version is not 1.
        """
        if len(data) < HEADER_SIZE:
            raise CorruptedFileError(
                f"Header too short: expected {HEADER_SIZE} bytes, got {len(data)}"
            )

        fields = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
        magic = fields[0]
        version = fields[1]

        if magic != MAGIC_BYTES:
            raise CorruptedFileError(
                f"Invalid magic bytes: expected {MAGIC_BYTES!r}, got {magic!r}"
            )
        if version != FORMAT_VERSION:
            raise IncompatibleVersionError(version)

        return cls(
            magic=magic,
            version=version,
            height=fields[2],
            width=fields[3],
            channels=fields[4],
            block_size=fields[5],
            model_type=fields[6],
            padding_height=fields[7],
            padding_width=fields[8],
        )


def save_pfic(path: Path | str, result: CompressionResult) -> None:
    """Save compression result to .pfic file.

    Parameters
    ----------
    path : Path or str
        Output file path.
    result : CompressionResult
        Compression result containing coefficients and metadata.
    """
    path = Path(path)

    channels = 1 if result.coefficients.ndim == 2 else result.coefficients.shape[0]

    # Look up model type from name (derived from MODEL_TYPE_MAP)
    model_name_to_type = {cls().name: code for code, cls in MODEL_TYPE_MAP.items()}
    model_type = model_name_to_type.get(result.model_name)
    if model_type is None:
        raise CorruptedFileError(f"Unknown model name: {result.model_name!r}")

    header = PficHeader(
        magic=MAGIC_BYTES,
        version=FORMAT_VERSION,
        height=result.original_shape[0],
        width=result.original_shape[1],
        channels=channels,
        block_size=result.block_size,
        model_type=model_type,
        padding_height=result.padded_shape[0],
        padding_width=result.padded_shape[1],
    )

    coefficients = np.ascontiguousarray(result.coefficients, dtype=np.float32)

    with path.open("wb") as f:
        f.write(header.pack())
        f.write(coefficients.tobytes())


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

    Raises
    ------
    CorruptedFileError
        If the file is too short, has wrong magic bytes, or payload size
        doesn't match the expected coefficient dimensions.
    IncompatibleVersionError
        If the file version is not supported.
    """
    path = Path(path)
    raw = path.read_bytes()

    if len(raw) < HEADER_SIZE:
        raise CorruptedFileError(
            f"File too short: expected at least {HEADER_SIZE} bytes, got {len(raw)}"
        )

    header = PficHeader.unpack(raw[:HEADER_SIZE])

    payload = raw[HEADER_SIZE:]

    # Determine expected coefficient shape
    model_cls = MODEL_TYPE_MAP.get(header.model_type)
    if model_cls is None:
        raise CorruptedFileError(f"Unknown model type: {header.model_type}")

    # Note: assumes all registered models have zero-argument constructors
    model = model_cls()
    num_coeffs = model.num_coefficients
    n_block_rows = header.padding_height // header.block_size
    n_block_cols = header.padding_width // header.block_size
    num_blocks = n_block_rows * n_block_cols

    if header.channels == 1:
        expected_shape = (num_coeffs, num_blocks)
    else:
        expected_shape = (header.channels, num_coeffs, num_blocks)

    expected_bytes = int(np.prod(expected_shape)) * 4  # float32

    if len(payload) != expected_bytes:
        raise CorruptedFileError(
            f"Payload size mismatch: expected {expected_bytes} bytes "
            f"for shape {expected_shape}, got {len(payload)} bytes"
        )

    coefficients = np.frombuffer(payload, dtype=np.float32).reshape(expected_shape).copy()

    return header, coefficients
