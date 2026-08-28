"""Receipt parser smoke test."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from receipt_ocr.ocr.tesseract_engine import TesseractOCREngine
from receipt_ocr.parser import ReceiptParser


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test OCR + receipt parser."
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
    print("Receipt Parser Smoke Test")
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

    print()
    print("-" * 70)
    print("STRUCTURED RECEIPT")
    print("-" * 70)

    print(
        json.dumps(
            receipt.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print("=" * 70)
    print("PARSER SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()