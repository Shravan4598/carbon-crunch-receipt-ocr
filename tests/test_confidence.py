"""Confidence scoring smoke test."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from receipt_ocr.confidence import ConfidenceScorer
from receipt_ocr.ocr.tesseract_engine import TesseractOCREngine
from receipt_ocr.parser import ReceiptParser
from receipt_ocr.validation import (
    ConflictDetector,
    ReceiptValidator,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test OCR + parser + validation + "
            "conflict detection + confidence scoring."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to receipt image.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    image_path = Path(args.image)

    print("=" * 70)
    print("Receipt Confidence Scoring Smoke Test")
    print("=" * 70)
    print(f"Image: {image_path}")

    # ---------------------------------------------------------
    # 1. OCR
    # ---------------------------------------------------------

    ocr_engine = TesseractOCREngine(
        language="eng",
        psm=6,
        min_confidence=0.0,
    )

    ocr_result = ocr_engine.extract(image_path)

    # ---------------------------------------------------------
    # 2. Receipt parsing
    # ---------------------------------------------------------

    receipt_parser = ReceiptParser()

    receipt = receipt_parser.parse(ocr_result)

    # ---------------------------------------------------------
    # 3. Standard validation
    # ---------------------------------------------------------

    validator = ReceiptValidator()

    validation_result = validator.validate(receipt)

    # ---------------------------------------------------------
    # 4. Conflict detection
    # ---------------------------------------------------------

    conflict_detector = ConflictDetector()

    conflicts = conflict_detector.detect(receipt)

    # ---------------------------------------------------------
    # 5. Confidence scoring
    # ---------------------------------------------------------

    scorer = ConfidenceScorer()

    confidence_result = scorer.score(
        receipt=receipt,
        ocr_confidence=ocr_result.average_confidence,
        conflicts=conflicts,
    )

    # ---------------------------------------------------------
    # 6. Display confidence result
    # ---------------------------------------------------------

    print()
    print("-" * 70)
    print("CONFIDENCE RESULT")
    print("-" * 70)

    print(
        json.dumps(
            confidence_result,
            indent=2,
            ensure_ascii=False,
        )
    )

    # ---------------------------------------------------------
    # 7. Display validation result
    # ---------------------------------------------------------

    print()
    print("-" * 70)
    print("VALIDATION")
    print("-" * 70)

    print(
        json.dumps(
            validation_result.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
    )

    # ---------------------------------------------------------
    # 8. Display conflicts
    # ---------------------------------------------------------

    print()
    print("-" * 70)
    print("CONFLICTS")
    print("-" * 70)

    print(
        json.dumps(
            conflicts,
            indent=2,
        )
    )

    # ---------------------------------------------------------
    # 9. Final status
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("CONFIDENCE SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()