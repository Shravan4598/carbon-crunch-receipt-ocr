"""PaddleOCR backend."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import paddleocr

from .base import BaseOCREngine
from .models import OCRLine, OCRResult, OCRWord


LOGGER = logging.getLogger(__name__)


class PaddleOCREngine(BaseOCREngine):
    """
    PaddleOCR-based OCR implementation.

    The implementation is intentionally isolated from the rest of the
    pipeline so that the OCR backend can be replaced later.
    """

    def __init__(
        self,
        language: str = "en",
        use_doc_orientation_classify: bool = True,
        use_doc_unwarping: bool = False,
        use_textline_orientation: bool = True,
    ) -> None:
        """
        Initialize PaddleOCR.

        Args:
            language: OCR language.
            use_doc_orientation_classify: Enable document orientation
                classification when supported.
            use_doc_unwarping: Enable document unwarping when supported.
            use_textline_orientation: Enable text-line orientation
                classification when supported.
        """
        self.language = language

        kwargs: dict[str, Any] = {
            "lang": language,
            "use_doc_orientation_classify": use_doc_orientation_classify,
            "use_doc_unwarping": use_doc_unwarping,
            "use_textline_orientation": use_textline_orientation,
        }

        try:
            self._ocr = paddleocr.PaddleOCR(**kwargs)
        except TypeError:
            # Compatibility fallback for older PaddleOCR releases.
            LOGGER.warning(
                "Modern PaddleOCR arguments were not accepted. "
                "Trying legacy initialization."
            )

            self._ocr = paddleocr.PaddleOCR(
                lang=language,
                use_angle_cls=True,
            )

    @property
    def engine_version(self) -> str:
        """Return the installed PaddleOCR version."""
        return getattr(
            paddleocr,
            "__version__",
            "unknown",
        )

    def extract(self, image_path: Path) -> OCRResult:
        """
        Run OCR on one image.

        Args:
            image_path: Path to an image.

        Returns:
            Standardized OCRResult.

        Raises:
            FileNotFoundError: If the image does not exist.
            ValueError: If the image cannot be decoded.
            RuntimeError: If OCR fails.
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Receipt image does not exist: {image_path}"
            )

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"OpenCV could not decode image: {image_path}"
            )

        image_height, image_width = image.shape[:2]

        start_time = time.perf_counter()

        try:
            raw_results = self._ocr.predict(
                str(image_path),
            )
        except AttributeError:
            # Compatibility with older PaddleOCR versions.
            raw_results = self._ocr.ocr(
                str(image_path),
                cls=True,
            )

        elapsed = time.perf_counter() - start_time

        lines = self._parse_results(raw_results)

        LOGGER.info(
            "OCR completed | image=%s | lines=%d | time=%.3fs",
            image_path.name,
            len(lines),
            elapsed,
        )

        return OCRResult(
            image_path=str(image_path),
            lines=tuple(lines),
            engine="PaddleOCR",
            engine_version=self.engine_version,
            processing_time_seconds=elapsed,
            image_width=image_width,
            image_height=image_height,
        )

    def _parse_results(
        self,
        raw_results: Any,
    ) -> list[OCRLine]:
        """
        Convert PaddleOCR output into our standardized representation.

        PaddleOCR has changed result formats between major releases.
        This parser intentionally supports the common modern result
        structures while keeping backend-specific logic isolated here.
        """
        parsed_lines: list[OCRLine] = []

        for result in self._iterate_results(raw_results):
            payload = self._extract_payload(result)

            if payload is None:
                continue

            texts = payload.get("rec_texts", [])
            scores = payload.get("rec_scores", [])
            boxes = payload.get("rec_boxes", [])

            if not texts:
                # Some versions expose OCR data using nested structures.
                nested = payload.get("ocr_res")

                if nested is not None:
                    nested_lines = self._parse_legacy_nested(nested)

                    if nested_lines:
                        parsed_lines.extend(nested_lines)

                continue

            for index, text in enumerate(texts):
                text = str(text).strip()

                if not text:
                    continue

                confidence = self._safe_confidence(
                    scores[index] if index < len(scores) else 0.0
                )

                bbox = self._parse_bbox(
                    boxes[index] if index < len(boxes) else None
                )

                parsed_lines.append(
                    OCRLine(
                        text=text,
                        confidence=confidence,
                        bbox=bbox,
                    )
                )

        return self._sort_lines(parsed_lines)

    @staticmethod
    def _iterate_results(raw_results: Any) -> list[Any]:
        """Normalize PaddleOCR's result container."""
        if raw_results is None:
            return []

        if isinstance(raw_results, list):
            return raw_results

        try:
            return list(raw_results)
        except TypeError:
            return [raw_results]

    @staticmethod
    def _extract_payload(result: Any) -> dict[str, Any] | None:
        """
        Extract dictionary payload from a PaddleOCR result object.
        """
        if isinstance(result, dict):
            return result

        # Modern PaddleOCR result objects expose JSON data.
        json_data = getattr(result, "json", None)

        if callable(json_data):
            try:
                json_data = json_data()
            except Exception:
                json_data = None

        if isinstance(json_data, str):
            try:
                json_data = json.loads(json_data)
            except json.JSONDecodeError:
                json_data = None

        if isinstance(json_data, dict):
            # Modern output may wrap OCR fields inside "res".
            nested = json_data.get("res")

            if isinstance(nested, dict):
                return nested

            return json_data

        # Some versions expose a direct dictionary-like attribute.
        for attribute in ("res", "result"):
            candidate = getattr(result, attribute, None)

            if isinstance(candidate, dict):
                return candidate

        return None

    @classmethod
    def _parse_legacy_nested(
        cls,
        nested: Any,
    ) -> list[OCRLine]:
        """Parse legacy PaddleOCR nested output."""
        lines: list[OCRLine] = []

        if not isinstance(nested, list):
            return lines

        # Legacy format is approximately:
        #
        # [
        #   [
        #       [box_points, ("text", confidence)],
        #       ...
        #   ]
        # ]
        for page in nested:
            if not isinstance(page, list):
                continue

            for entry in page:
                if not isinstance(entry, (list, tuple)):
                    continue

                if len(entry) < 2:
                    continue

                box_data = entry[0]
                text_data = entry[1]

                if not isinstance(text_data, (list, tuple)):
                    continue

                if len(text_data) < 2:
                    continue

                text = str(text_data[0]).strip()
                confidence = cls._safe_confidence(text_data[1])
                bbox = cls._parse_polygon_bbox(box_data)

                if text:
                    lines.append(
                        OCRLine(
                            text=text,
                            confidence=confidence,
                            bbox=bbox,
                        )
                    )

        return lines

    @staticmethod
    def _safe_confidence(value: Any) -> float:
        """Convert OCR confidence to a bounded float in [0, 1]."""
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0

        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _parse_bbox(
        box: Any,
    ) -> tuple[float, float, float, float]:
        """Convert common PaddleOCR box formats to xyxy."""
        if box is None:
            return (0.0, 0.0, 0.0, 0.0)

        try:
            values = [float(value) for value in box]
        except (TypeError, ValueError):
            return (0.0, 0.0, 0.0, 0.0)

        if len(values) == 4:
            x1, y1, x2, y2 = values

            return (
                min(x1, x2),
                min(y1, y2),
                max(x1, x2),
                max(y1, y2),
            )

        if len(values) == 8:
            points = [
                (values[index], values[index + 1])
                for index in range(0, 8, 2)
            ]

            return PaddleOCREngine._points_to_bbox(points)

        return (0.0, 0.0, 0.0, 0.0)

    @staticmethod
    def _parse_polygon_bbox(
        polygon: Any,
    ) -> tuple[float, float, float, float]:
        """Convert a polygon into an xyxy bounding box."""
        if not polygon:
            return (0.0, 0.0, 0.0, 0.0)

        try:
            points = [
                (float(point[0]), float(point[1]))
                for point in polygon
            ]
        except (TypeError, ValueError, IndexError):
            return (0.0, 0.0, 0.0, 0.0)

        return PaddleOCREngine._points_to_bbox(points)

    @staticmethod
    def _points_to_bbox(
        points: list[tuple[float, float]],
    ) -> tuple[float, float, float, float]:
        """Convert polygon points into xyxy bounding box."""
        if not points:
            return (0.0, 0.0, 0.0, 0.0)

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]

        return (
            min(xs),
            min(ys),
            max(xs),
            max(ys),
        )

    @staticmethod
    def _sort_lines(
        lines: list[OCRLine],
    ) -> list[OCRLine]:
        """Sort OCR lines top-to-bottom and left-to-right."""
        return sorted(
            lines,
            key=lambda line: (
                line.bbox[1],
                line.bbox[0],
            ),
        )