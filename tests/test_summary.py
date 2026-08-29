"""Financial summary smoke test."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from receipt_ocr.ocr.tesseract_engine import TesseractOCREngine
from receipt_ocr.parser import ReceiptParser
from receipt_ocr.summary import FinancialSummaryGenerator
from receipt_ocr.validation import ConflictDetector, ReceiptValidator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test financial summary generation."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to receipt image.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    image_path = Path(args.image)

    print("=" * 70)
    print("Financial Summary Smoke Test")
    print("=" * 70)
    print(f"Image: {image_path}")

    # ---------------------------------------------------------
    # OCR
    # ---------------------------------------------------------

    ocr_engine = TesseractOCREngine(
        language="eng",
        psm=6,
        min_confidence=0.0,
    )

    ocr_result = ocr_engine.extract(image_path)

    # ---------------------------------------------------------
    # Parsing
    # ---------------------------------------------------------

    receipt_parser = ReceiptParser()
    receipt = receipt_parser.parse(ocr_result)

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    validator = ReceiptValidator()
    validation = validator.validate(receipt)

    # ---------------------------------------------------------
    # Conflict detection
    # ---------------------------------------------------------

    conflict_detector = ConflictDetector()
    conflicts = conflict_detector.detect(receipt)

    # ---------------------------------------------------------
    # Financial summary
    # ---------------------------------------------------------

    generator = FinancialSummaryGenerator()

    summary = generator.generate(
        receipts=[receipt],
        ocr_confidences=[
            ocr_result.average_confidence
        ],
        conflict_counts=[
            len(conflicts)
        ],
        validation_warning_counts=[
            len(validation.warnings)
        ],
    )

    print()
    print("-" * 70)
    print("FINANCIAL SUMMARY")
    print("-" * 70)

    print(
        json.dumps(
            summary.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print("-" * 70)
    print("VALIDATION")
    print("-" * 70)

    print(
        json.dumps(
            validation.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print("-" * 70)
    print("CONFLICTS")
    print("-" * 70)

    print(
        json.dumps(
            conflicts,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print("=" * 70)
    print("FINANCIAL SUMMARY SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()