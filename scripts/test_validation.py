"""Receipt validation smoke test."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from receipt_ocr.ocr.tesseract_engine import TesseractOCREngine
from receipt_ocr.parser import ReceiptParser
from receipt_ocr.validation import (
    ConflictDetector,
    ReceiptValidator,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test receipt validation."
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
    print("Receipt Validation Smoke Test")
    print("=" * 70)
    print(f"Image: {image_path}")

    ocr_engine = TesseractOCREngine(
        language="eng",
        psm=6,
        min_confidence=0.0,
    )

    ocr_result = ocr_engine.extract(image_path)

    receipt_parser = ReceiptParser()
    receipt = receipt_parser.parse(ocr_result)

    validator = ReceiptValidator()
    validation_result = validator.validate(receipt)

    detector = ConflictDetector()
    conflicts = detector.detect(receipt)

    print()
    print("-" * 70)
    print("VALIDATION RESULT")
    print("-" * 70)

    print(
        json.dumps(
            validation_result.to_dict(),
            indent=2,
        )
    )

    print()
    print("-" * 70)
    print("CONFLICT DETECTION")
    print("-" * 70)

    print(
        json.dumps(
            {
                "conflict_count": len(conflicts),
                "conflicts": conflicts,
            },
            indent=2,
        )
    )

    print()
    print("=" * 70)
    print("VALIDATION SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()