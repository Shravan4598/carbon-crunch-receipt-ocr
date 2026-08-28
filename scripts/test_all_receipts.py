from __future__ import annotations

import csv
import logging
from pathlib import Path

from receipt_ocr.ocr.paddle_engine import PaddleOCREngine
from receipt_ocr.parser.receipt_parser import ReceiptParser


logging.basicConfig(
    level=logging.WARNING,
)

RAW_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/parser_results.csv")


def main() -> None:
    images = sorted(
        [
            *RAW_DIR.glob("*.jpg"),
            *RAW_DIR.glob("*.JPG"),
            *RAW_DIR.glob("*.jpeg"),
            *RAW_DIR.glob("*.JPEG"),
            *RAW_DIR.glob("*.png"),
            *RAW_DIR.glob("*.PNG"),
        ]
    )

    images = [
        image
        for image in images
        if image.is_file()
    ]

    print(f"Found {len(images)} receipt images.")

    if not images:
        print("No images found.")
        return

    engine = PaddleOCREngine(
        language="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    parser = ReceiptParser()

    results = []

    for index, image_path in enumerate(images, start=1):
        print(
            f"[{index}/{len(images)}] "
            f"Processing {image_path.name}..."
        )

        try:
            ocr_result = engine.extract(
                str(image_path)
            )

            receipt = parser.parse(
                ocr_result
            )

            results.append(
                {
                    "file": image_path.name,
                    "merchant": receipt.merchant or "",
                    "date": receipt.receipt_date or "",
                    "receipt_number": (
                        receipt.receipt_number or ""
                    ),
                    "item_count": len(receipt.items),
                    "subtotal": receipt.subtotal,
                    "discount": receipt.discount,
                    "tax": receipt.tax,
                    "total": receipt.total,
                    "payment_method": (
                        receipt.payment_method or ""
                    ),
                    "confidence": receipt.extraction_confidence,
                    "warning_count": len(
                        receipt.warnings
                    ),
                    "warnings": " | ".join(
                        receipt.warnings
                    ),
                }
            )

        except Exception as exc:
            print(
                f"ERROR: {image_path.name}: {exc}"
            )

            results.append(
                {
                    "file": image_path.name,
                    "merchant": "",
                    "date": "",
                    "receipt_number": "",
                    "item_count": 0,
                    "subtotal": None,
                    "discount": None,
                    "tax": None,
                    "total": None,
                    "payment_method": "",
                    "confidence": 0.0,
                    "warning_count": 1,
                    "warnings": (
                        f"PROCESSING ERROR: {exc}"
                    ),
                }
            )

    fieldnames = [
        "file",
        "merchant",
        "date",
        "receipt_number",
        "item_count",
        "subtotal",
        "discount",
        "tax",
        "total",
        "payment_method",
        "confidence",
        "warning_count",
        "warnings",
    ]

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 60)
    print("DATASET TEST COMPLETE")
    print("=" * 60)
    print(f"Images processed: {len(images)}")
    print(f"Results saved to: {OUTPUT_FILE}")

    successful = [
        result
        for result in results
        if result["total"] is not None
    ]

    with_items = [
        result
        for result in results
        if result["item_count"] > 0
    ]

    warning_free = [
        result
        for result in results
        if result["warning_count"] == 0
    ]

    print(
        f"Receipts with total: "
        f"{len(successful)}/{len(results)}"
    )

    print(
        f"Receipts with items: "
        f"{len(with_items)}/{len(results)}"
    )

    print(
        f"Receipts without warnings: "
        f"{len(warning_free)}/{len(results)}"
    )

    if results:
        average_confidence = (
            sum(
                float(r["confidence"])
                for r in results
            )
            / len(results)
        )

        print(
            f"Average confidence: "
            f"{average_confidence:.3f}"
        )


if __name__ == "__main__":
    main()