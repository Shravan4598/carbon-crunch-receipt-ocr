"""Run a single-image OCR smoke test."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running this script directly from the repository root.
ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from receipt_ocr.ocr import PaddleOCREngine  # noqa: E402


def configure_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )


def main() -> int:
    """Execute the OCR smoke test."""
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR on a single receipt image."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the receipt image.",
    )

    parser.add_argument(
        "--output",
        default="outputs/ocr/smoke_test.json",
        help="Output JSON path.",
    )

    args = parser.parse_args()

    configure_logging()

    image_path = Path(args.image)

    if not image_path.exists():
        print(f"ERROR: Image does not exist: {image_path}")
        return 1

    print()
    print("=" * 70)
    print("Receipt OCR Smoke Test")
    print("=" * 70)
    print(f"Image: {image_path}")
    print()

    try:
        engine = PaddleOCREngine(language="en")

        result = engine.extract(image_path)

    except Exception as exc:
        logging.exception(
            "OCR smoke test failed."
        )

        print()
        print(f"ERROR: {exc}")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result.to_dict(),
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"OCR engine: {result.engine}")
    print(f"OCR version: {result.engine_version}")
    print(f"Image size: {result.image_width} x {result.image_height}")
    print(f"Detected lines: {len(result.lines)}")
    print(
        "Average confidence: "
        f"{result.average_confidence:.3f}"
    )
    print(
        "Processing time: "
        f"{result.processing_time_seconds:.3f}s"
    )

    print()
    print("-" * 70)
    print("RECOGNIZED TEXT")
    print("-" * 70)

    for index, line in enumerate(result.lines, start=1):
        print(
            f"{index:03d} | "
            f"{line.confidence:.3f} | "
            f"{line.text}"
        )

    print()
    print("-" * 70)
    print(f"JSON saved to: {output_path}")
    print("-" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())