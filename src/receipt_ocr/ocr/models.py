"""Data models used by the OCR layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class OCRWord:
    """Represents a single OCR-recognized text region."""

    text: str
    confidence: float
    bbox: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert the OCR word to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class OCRLine:
    """Represents a logically grouped OCR line."""

    text: str
    confidence: float
    bbox: tuple[float, float, float, float]
    words: tuple[OCRWord, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Convert the OCR line to a JSON-serializable dictionary."""
        return {
            "text": self.text,
            "confidence": round(self.confidence, 6),
            "bbox": list(self.bbox),
            "words": [word.to_dict() for word in self.words],
        }


@dataclass(frozen=True)
class OCRResult:
    """Standardized OCR result independent of the OCR backend."""

    image_path: str
    lines: tuple[OCRLine, ...]
    engine: str
    engine_version: str
    processing_time_seconds: float
    image_width: int
    image_height: int

    @property
    def full_text(self) -> str:
        """Return all recognized lines as a single text block."""
        return "\n".join(line.text for line in self.lines)

    @property
    def average_confidence(self) -> float:
        """Return the mean confidence across recognized lines."""
        if not self.lines:
            return 0.0

        return sum(line.confidence for line in self.lines) / len(self.lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert the complete OCR result to a JSON-serializable dictionary."""
        return {
            "image_path": self.image_path,
            "engine": self.engine,
            "engine_version": self.engine_version,
            "processing_time_seconds": round(
                self.processing_time_seconds,
                4,
            ),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "line_count": len(self.lines),
            "average_confidence": round(
                self.average_confidence,
                6,
            ),
            "full_text": self.full_text,
            "lines": [line.to_dict() for line in self.lines],
        }