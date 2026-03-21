"""Tests for io module — PficHeader, save_pfic, and load_pfic."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from polyfit_compress.compressor import LeastSquaresCompressor
from polyfit_compress.exceptions import CorruptedFileError, IncompatibleVersionError
from polyfit_compress.io import (
    HEADER_FORMAT,
    HEADER_SIZE,
    MAGIC_BYTES,
    PficHeader,
    load_pfic,
    save_pfic,
)
from polyfit_compress.models import LinearModel, QuadraticModel


class TestPficHeader:
    """Tests for PficHeader pack/unpack."""

    def _make_header(self, **overrides: object) -> PficHeader:
        defaults = dict(
            magic=MAGIC_BYTES,
            version=1,
            height=64,
            width=64,
            channels=1,
            block_size=8,
            model_type=0,
            padding_height=64,
            padding_width=64,
        )
        defaults.update(overrides)
        return PficHeader(**defaults)

    def test_pack_returns_correct_size(self) -> None:
        header = self._make_header()
        packed = header.pack()
        assert len(packed) == HEADER_SIZE

    def test_pack_unpack_roundtrip(self) -> None:
        header = self._make_header(
            height=50,
            width=70,
            channels=3,
            block_size=16,
            model_type=1,
            padding_height=64,
            padding_width=80,
        )
        packed = header.pack()
        restored = PficHeader.unpack(packed)
        assert restored.magic == header.magic
        assert restored.version == header.version
        assert restored.height == header.height
        assert restored.width == header.width
        assert restored.channels == header.channels
        assert restored.block_size == header.block_size
        assert restored.model_type == header.model_type
        assert restored.padding_height == header.padding_height
        assert restored.padding_width == header.padding_width

    def test_unpack_wrong_magic_raises_corrupted(self) -> None:
        header = self._make_header()
        packed = bytearray(header.pack())
        packed[0:4] = b"BAAD"
        with pytest.raises(CorruptedFileError, match="Invalid magic bytes"):
            PficHeader.unpack(bytes(packed))

    def test_unpack_wrong_version_raises_incompatible(self) -> None:
        header = self._make_header()
        packed = bytearray(header.pack())
        # Version byte is at offset 4 (after 4-byte magic)
        packed[4] = 99
        with pytest.raises(IncompatibleVersionError, match="99"):
            PficHeader.unpack(bytes(packed))

    def test_unpack_truncated_data_raises_corrupted(self) -> None:
        with pytest.raises(CorruptedFileError, match="Header too short"):
            PficHeader.unpack(b"PF")

    def test_pack_has_correct_magic(self) -> None:
        header = self._make_header()
        packed = header.pack()
        assert packed[:4] == MAGIC_BYTES

    def test_header_size_is_33(self) -> None:
        assert HEADER_SIZE == struct.calcsize(HEADER_FORMAT)


class TestSaveLoadPfic:
    """Tests for save_pfic and load_pfic roundtrips."""

    def test_roundtrip_grayscale(self, sample_grayscale, tmp_path) -> None:
        compressor = LeastSquaresCompressor(model=LinearModel(), block_size=8)
        result = compressor.compress(sample_grayscale)

        path = tmp_path / "test.pfic"
        save_pfic(path, result)
        header, coefficients = load_pfic(path)

        np.testing.assert_array_almost_equal(coefficients, result.coefficients)
        assert header.height == sample_grayscale.shape[0]
        assert header.width == sample_grayscale.shape[1]
        assert header.channels == 1
        assert header.block_size == 8
        assert header.model_type == 0  # LinearModel

    def test_roundtrip_rgb(self, sample_rgb, tmp_path) -> None:
        compressor = LeastSquaresCompressor(model=QuadraticModel(), block_size=8)
        result = compressor.compress(sample_rgb)

        path = tmp_path / "test_rgb.pfic"
        save_pfic(path, result)
        header, coefficients = load_pfic(path)

        np.testing.assert_array_almost_equal(coefficients, result.coefficients)
        assert header.height == sample_rgb.shape[0]
        assert header.width == sample_rgb.shape[1]
        assert header.channels == 3
        assert header.model_type == 1  # QuadraticModel

    def test_roundtrip_non_square_image(self, tmp_path) -> None:
        img = np.random.default_rng(42).integers(0, 256, size=(50, 70), dtype=np.uint8)
        compressor = LeastSquaresCompressor(model=LinearModel(), block_size=8)
        result = compressor.compress(img)

        path = tmp_path / "test_nonsquare.pfic"
        save_pfic(path, result)
        header, coefficients = load_pfic(path)

        np.testing.assert_array_almost_equal(coefficients, result.coefficients)
        assert header.height == 50
        assert header.width == 70

    def test_load_wrong_magic_raises_corrupted(self, tmp_path) -> None:
        path = tmp_path / "bad_magic.pfic"
        # Write a file with bad magic
        data = b"BAAD" + b"\x00" * (HEADER_SIZE - 4 + 100)
        path.write_bytes(data)
        with pytest.raises(CorruptedFileError, match="Invalid magic bytes"):
            load_pfic(path)

    def test_load_truncated_file_raises_corrupted(self, tmp_path) -> None:
        path = tmp_path / "truncated.pfic"
        path.write_bytes(b"PFIC\x01")
        with pytest.raises(CorruptedFileError, match="File too short"):
            load_pfic(path)

    def test_load_wrong_version_raises_incompatible(self, tmp_path) -> None:
        path = tmp_path / "wrong_version.pfic"
        # Build a valid header but with version = 5
        header_bytes = struct.pack(
            HEADER_FORMAT,
            MAGIC_BYTES,
            5,  # wrong version
            64,
            64,
            1,
            8,
            0,
            64,
            64,
            b"\x00" * 12,
        )
        path.write_bytes(header_bytes + b"\x00" * 100)
        with pytest.raises(IncompatibleVersionError, match="5"):
            load_pfic(path)

    def test_load_payload_size_mismatch_raises_corrupted(self, sample_grayscale, tmp_path) -> None:
        compressor = LeastSquaresCompressor(model=LinearModel(), block_size=8)
        result = compressor.compress(sample_grayscale)

        path = tmp_path / "bad_payload.pfic"
        save_pfic(path, result)

        # Truncate the payload
        raw = path.read_bytes()
        path.write_bytes(raw[: HEADER_SIZE + 4])  # only 4 bytes of payload

        with pytest.raises(CorruptedFileError, match="Payload size mismatch"):
            load_pfic(path)

    def test_coefficients_dtype_is_float32(self, sample_grayscale, tmp_path) -> None:
        compressor = LeastSquaresCompressor(model=LinearModel(), block_size=8)
        result = compressor.compress(sample_grayscale)

        path = tmp_path / "dtype.pfic"
        save_pfic(path, result)
        _, coefficients = load_pfic(path)
        assert coefficients.dtype == np.float32
