"""Abstract interface for OCR engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .models import OCRResult


class BaseOCREngine(ABC):
    """Common interface implemented by all OCR backends."""

    @abstractmethod
    def extract(self, image_path: Path) -> OCRResult:
        """
        Extract text, bounding boxes, and confidence from an image.

        Args:
            image_path: Path to the input image.

        Returns:
            Standardized OCRResult.
        """
        raise NotImplementedError