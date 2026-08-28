"""Tesseract OCR backend."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import pytesseract
from pytesseract import Output

from .models import OCRLine, OCRResult, OCRWord

logger = logging.getLogger(__name__)


class TesseractOCREngine:
    """OCR engine backed by Tesseract."""

    def __init__(
        self,
        language: str = "eng",
        psm: int = 6,
        min_confidence: float = 0.0,
        tesseract_cmd: str | None = None,
    ) -> None:
        """Initialize the Tesseract OCR engine.

        Args:
            language: Tesseract language code.
            psm: Tesseract page segmentation mode.
            min_confidence: Minimum confidence for retaining OCR words.
            tesseract_cmd: Optional explicit path to tesseract.exe.
        """
        self.language = language
        self.psm = psm
        self.min_confidence = min_confidence

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        self._verify_tesseract()

    def _verify_tesseract(self) -> None:
        """Verify that Tesseract is available."""
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(
                "Tesseract initialized successfully: %s",
                version,
            )
        except Exception as exc:
            raise RuntimeError(
                "Tesseract executable could not be initialized. "
                "Make sure Tesseract is installed and available on PATH."
            ) from exc

    def extract(self, image_path: str | Path) -> OCRResult:
        """Run OCR on an image.

        Args:
            image_path: Path to the input image.

        Returns:
            Standardized OCRResult.
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image does not exist: {image_path}"
            )

        if not image_path.is_file():
            raise ValueError(
                f"Path is not a file: {image_path}"
            )

        start_time = time.perf_counter()

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"Unable to read image: {image_path}"
            )

        image_height, image_width = image.shape[:2]

        logger.info(
            "Running Tesseract OCR | image=%s | psm=%d | language=%s",
            image_path.name,
            self.psm,
            self.language,
        )

        config = f"--psm {self.psm}"

        data = pytesseract.image_to_data(
            image,
            lang=self.language,
            config=config,
            output_type=Output.DICT,
        )

        lines = self._build_lines(data)

        processing_time = time.perf_counter() - start_time

        result = OCRResult(
            image_path=str(image_path),
            lines=tuple(lines),
            engine="tesseract",
            engine_version=self._get_version(),
            processing_time_seconds=processing_time,
            image_width=image_width,
            image_height=image_height,
        )

        logger.info(
            "OCR completed | image=%s | lines=%d | words=%d "
            "| avg_confidence=%.3f | processing_time=%.3fs",
            image_path.name,
            len(result.lines),
            sum(len(line.words) for line in result.lines),
            result.average_confidence,
            processing_time,
        )

        return result

    def _build_lines(self, data: dict) -> list[OCRLine]:
        """Group Tesseract word-level output into logical lines."""
        grouped: dict[tuple[int, int, int], list[OCRWord]] = {}

        texts: list[str] = data.get("text", [])

        for index, raw_text in enumerate(texts):
            text = str(raw_text).strip()

            if not text:
                continue

            confidence = self._parse_confidence(
                data.get("conf", []),
                index,
            )

            if confidence < self.min_confidence:
                continue

            bbox = self._parse_bbox(data, index)

            word = OCRWord(
                text=text,
                confidence=confidence,
                bbox=bbox,
            )

            block_num = self._safe_int(
                data.get("block_num", []),
                index,
            )

            paragraph_num = self._safe_int(
                data.get("par_num", []),
                index,
            )

            line_num = self._safe_int(
                data.get("line_num", []),
                index,
            )

            key = (
                block_num,
                paragraph_num,
                line_num,
            )

            grouped.setdefault(key, []).append(word)

        lines: list[OCRLine] = []

        for words in grouped.values():
            if not words:
                continue

            ordered_words = sorted(
                words,
                key=lambda word: (
                    word.bbox[1],
                    word.bbox[0],
                ),
            )

            line_text = " ".join(
                word.text for word in ordered_words
            )

            line_confidence = sum(
                word.confidence
                for word in ordered_words
            ) / len(ordered_words)

            line_bbox = self._line_bbox(ordered_words)

            lines.append(
                OCRLine(
                    text=line_text,
                    confidence=line_confidence,
                    bbox=line_bbox,
                    words=tuple(ordered_words),
                )
            )

        lines.sort(
            key=lambda line: (
                line.bbox[1],
                line.bbox[0],
            )
        )

        return lines

    @staticmethod
    def _parse_confidence(
        confidence_values: list,
        index: int,
    ) -> float:
        """Convert Tesseract confidence from 0-100 to 0-1."""
        try:
            value = float(confidence_values[index])
        except (ValueError, TypeError, IndexError):
            return 0.0

        if value < 0:
            return 0.0

        return max(0.0, min(1.0, value / 100.0))

    @staticmethod
    def _parse_bbox(
        data: dict,
        index: int,
    ) -> tuple[float, float, float, float]:
        """Extract an OCR bounding box."""
        try:
            left = float(data["left"][index])
            top = float(data["top"][index])
            width = float(data["width"][index])
            height = float(data["height"][index])
        except (KeyError, IndexError, TypeError, ValueError):
            return (0.0, 0.0, 0.0, 0.0)

        return (left, top, width, height)

    @staticmethod
    def _safe_int(
        values: list,
        index: int,
    ) -> int:
        """Safely read an integer metadata value."""
        try:
            return int(values[index])
        except (IndexError, TypeError, ValueError):
            return 0

    @staticmethod
    def _line_bbox(
        words: list[OCRWord],
    ) -> tuple[float, float, float, float]:
        """Calculate the enclosing bounding box for a line."""
        if not words:
            return (0.0, 0.0, 0.0, 0.0)

        left = min(word.bbox[0] for word in words)
        top = min(word.bbox[1] for word in words)

        right = max(
            word.bbox[0] + word.bbox[2]
            for word in words
        )

        bottom = max(
            word.bbox[1] + word.bbox[3]
            for word in words
        )

        return (
            left,
            top,
            right - left,
            bottom - top,
        )

    @staticmethod
    def _get_version() -> str:
        """Return the installed Tesseract version."""
        try:
            version = pytesseract.get_tesseract_version()
            return str(version).splitlines()[0].strip()
        except Exception:
            return "unknown"