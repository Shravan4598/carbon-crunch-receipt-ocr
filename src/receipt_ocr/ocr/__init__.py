"""OCR backends and standardized OCR data structures."""

from .models import OCRLine, OCRResult, OCRWord
from .paddle_engine import PaddleOCREngine
from .tesseract_engine import TesseractOCREngine

__all__ = [
    "OCRLine",
    "OCRResult",
    "OCRWord",
    "PaddleOCREngine",
    "TesseractOCREngine",
]