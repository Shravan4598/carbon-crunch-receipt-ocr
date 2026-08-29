"""Smoke test for the Tesseract OCR backend."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from receipt_ocr.ocr.tesseract_engine import TesseractOCREngine


def configure_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a Tesseract OCR smoke test on a receipt image."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the receipt image.",
    )

    parser.add_argument(
        "--language",
        default="eng",
        help="Tesseract language code. Default: eng",
    )

    parser.add_argument(
        "--psm",
        type=int,
        default=6,
        help="Tesseract page segmentation mode. Default: 6",
    )

    return parser.parse_args()


def print_separator(title: str) -> None:
    """Print a formatted section separator."""
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


def main() -> int:
    """Run the Tesseract OCR smoke test."""
    configure_logging()

    args = parse_args()
    image_path = Path(args.image)

    print("=" * 70)
    print("Tesseract OCR Smoke Test")
    print("=" * 70)
    print(f"Image: {image_path}")

    if not image_path.exists():
        print(f"ERROR: Image does not exist: {image_path}")
        return 1

    try:
        engine = TesseractOCREngine(
            language=args.language,
            psm=args.psm,
        )

        result = engine.extract(image_path)

        print_separator("OCR RESULTS")

        print(f"Engine: {result.engine}")
        print(f"Engine version: {result.engine_version}")
        print(f"Image size: {result.image_width} x {result.image_height}")
        print(f"Lines detected: {len(result.lines)}")

        total_words = sum(
            len(line.words)
            for line in result.lines
        )

        print(f"Words detected: {total_words}")
        print(
            f"Average confidence: "
            f"{result.average_confidence:.3f}"
        )
        print(
            f"Processing time: "
            f"{result.processing_time_seconds:.3f}s"
        )

        print_separator("RECOGNIZED TEXT")

        if result.full_text.strip():
            print(result.full_text)
        else:
            print("[No text detected]")

        print_separator("LINE-LEVEL OCR")

        for index, line in enumerate(result.lines, start=1):
            print(
                f"{index:03d} | "
                f"confidence={line.confidence:.3f} | "
                f"bbox={line.bbox} | "
                f"text={line.text}"
            )

        print_separator("WORD-LEVEL OCR")

        word_index = 1

        for line in result.lines:
            for word in line.words:
                print(
                    f"{word_index:03d} | "
                    f"confidence={word.confidence:.3f} | "
                    f"bbox={word.bbox} | "
                    f"text={word.text}"
                )

                word_index += 1

        print()
        print("=" * 70)
        print("OCR SMOKE TEST PASSED")
        print("=" * 70)

        return 0

    except Exception:
        logging.exception("Tesseract OCR smoke test failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())