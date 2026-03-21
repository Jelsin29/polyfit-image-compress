"""polyfit-image-compress: Image compression via polynomial surface fitting."""

from polyfit_compress.compressor import CompressionResult, LeastSquaresCompressor
from polyfit_compress.exceptions import CorruptedFileError, IncompatibleVersionError
from polyfit_compress.io import PficHeader, load_pfic, save_pfic
from polyfit_compress.models import LinearModel, QuadraticModel

__version__ = "0.1.0"
__all__ = [
    "CompressionResult",
    "CorruptedFileError",
    "IncompatibleVersionError",
    "LeastSquaresCompressor",
    "LinearModel",
    "PficHeader",
    "QuadraticModel",
    "load_pfic",
    "save_pfic",
]
