"""polyfit-image-compress: Image compression via polynomial surface fitting."""

from polyfit_compress.compressor import CompressionResult, LeastSquaresCompressor
from polyfit_compress.models import LinearModel, QuadraticModel

__version__ = "0.1.0"
__all__ = ["CompressionResult", "LeastSquaresCompressor", "LinearModel", "QuadraticModel"]
