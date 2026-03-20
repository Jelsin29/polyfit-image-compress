"""Custom exceptions for polyfit-image-compress."""


class PolyfitError(Exception):
    """Base exception for polyfit-image-compress."""


class InvalidBlockSizeError(PolyfitError):
    """Raised when block_size is invalid."""


class UnsupportedImageFormatError(PolyfitError):
    """Raised when image format is not supported."""


class CorruptedFileError(PolyfitError):
    """Raised when a .pfic file is corrupted."""
