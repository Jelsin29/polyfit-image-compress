"""Custom exceptions for polyfit-image-compress."""


class PolyfitError(Exception):
    """Base exception for polyfit-image-compress."""


class InvalidBlockSizeError(PolyfitError):
    """Raised when block_size is invalid."""


class UnsupportedImageFormatError(PolyfitError):
    """Raised when image format is not supported."""


class CorruptedFileError(PolyfitError):
    """Raised when a .pfic file is invalid or corrupted."""

    def __init__(self, detail: str = "File is corrupted or invalid") -> None:
        super().__init__(detail)


class IncompatibleVersionError(PolyfitError):
    """Raised when a .pfic file version is unsupported."""

    def __init__(self, version: int) -> None:
        super().__init__(f"Incompatible .pfic version: found {version}, expected 1")
        self.version = version
