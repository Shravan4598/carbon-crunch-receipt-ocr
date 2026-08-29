from __future__ import annotations

import csv
import logging
from pathlib import Path

from receipt_ocr.ocr.paddle_engine import PaddleOCREngine
from receipt_ocr.parser.receipt_parser import ReceiptParser


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.WARNING,
)


# ============================================================
# PATHS
# ============================================================

RAW_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/parser_results.csv")


# ============================================================
# SUPPORTED IMAGE EXTENSIONS
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


# ============================================================
# FIND UNIQUE RECEIPT IMAGES
# ============================================================

def get_image_files() -> list[Path]:
    """
    Find all unique receipt images inside data/raw.

    Handles:
        .jpg
        .JPG
        .jpeg
        .JPEG
        .png
        .PNG

    Returns:
        Sorted list of unique image paths.
    """

    if not RAW_DIR.exists():
        print(f"ERROR: Raw directory does not exist: {RAW_DIR}")
        return []

    images = {
        image.resolve()
        for image in RAW_DIR.iterdir()
        if (
            image.is_file()
            and image.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    }

    return sorted(
        images,
        key=lambda path: path.name.lower(),
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    images = get_image_files()

    print(
        f"Found {len(images)} unique receipt images."
    )

    if not images:
        print("No images found.")
        return

    # --------------------------------------------------------
    # Initialize OCR engine
    # --------------------------------------------------------

    print("Initializing OCR engine...")

    engine = PaddleOCREngine(
        language="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    # --------------------------------------------------------
    # Initialize parser
    # --------------------------------------------------------

    print("Initializing receipt parser...")

    parser = ReceiptParser()

    # --------------------------------------------------------
    # Results container
    # --------------------------------------------------------

    results: list[dict] = []

    # --------------------------------------------------------
    # Process every receipt
    # --------------------------------------------------------

    for index, image_path in enumerate(
        images,
        start=1,
    ):

        print(
            f"[{index}/{len(images)}] "
            f"Processing {image_path.name}..."
        )

        try:

            # ------------------------------------------------
            # OCR
            # ------------------------------------------------

            ocr_result = engine.extract(
                str(image_path)
            )

            # ------------------------------------------------
            # Parse receipt
            # ------------------------------------------------

            receipt = parser.parse(
                ocr_result
            )

            # ------------------------------------------------
            # Calculate extracted item total
            # ------------------------------------------------

            items_total = round(
                sum(
                    item.total_price
                    for item in receipt.items
                    if item.total_price is not None
                ),
                2,
            )

            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            results.append(
                {
                    "file": image_path.name,

                    "merchant": (
                        receipt.merchant or ""
                    ),

                    "receipt_date": (
                        receipt.receipt_date or ""
                    ),

                    "receipt_number": (
                        receipt.receipt_number or ""
                    ),

                    "item_count": len(
                        receipt.items
                    ),

                    "items_total": (
                        items_total
                        if receipt.items
                        else None
                    ),

                    "subtotal": (
                        receipt.subtotal
                    ),

                    "discount": (
                        receipt.discount
                    ),

                    "tax": (
                        receipt.tax
                    ),

                    "total": (
                        receipt.total
                    ),

                    "payment_method": (
                        receipt.payment_method or ""
                    ),

                    "extraction_confidence": (
                        receipt.extraction_confidence
                    ),

                    "warning_count": len(
                        receipt.warnings
                    ),

                    "warnings": (
                        " | ".join(
                            receipt.warnings
                        )
                    ),
                }
            )

        except Exception as exc:

            # ------------------------------------------------
            # Handle processing errors without stopping
            # ------------------------------------------------

            print(
                f"ERROR: {image_path.name}: {exc}"
            )

            results.append(
                {
                    "file": image_path.name,

                    "merchant": "",

                    "receipt_date": "",

                    "receipt_number": "",

                    "item_count": 0,

                    "items_total": None,

                    "subtotal": None,

                    "discount": None,

                    "tax": None,

                    "total": None,

                    "payment_method": "",

                    "extraction_confidence": 0.0,

                    "warning_count": 1,

                    "warnings": (
                        f"PROCESSING ERROR: {exc}"
                    ),
                }
            )

    # ========================================================
    # CSV FIELD NAMES
    # ========================================================

    fieldnames = [
        "file",
        "merchant",
        "receipt_date",
        "receipt_number",
        "item_count",
        "items_total",
        "subtotal",
        "discount",
        "tax",
        "total",
        "payment_method",
        "extraction_confidence",
        "warning_count",
        "warnings",
    ]

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # WRITE CSV
    # ========================================================

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

    # ========================================================
    # CALCULATE SUMMARY
    # ========================================================

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

    processing_errors = [
        result
        for result in results
        if str(
            result["warnings"]
        ).startswith(
            "PROCESSING ERROR:"
        )
    ]

    # ========================================================
    # AVERAGE CONFIDENCE
    # ========================================================

    average_confidence = 0.0

    if results:

        average_confidence = (
            sum(
                float(
                    result[
                        "extraction_confidence"
                    ]
                )
                for result in results
            )
            / len(results)
        )

    # ========================================================
    # CHECK DUPLICATES
    # ========================================================

    result_files = [
        result["file"]
        for result in results
    ]

    unique_result_files = set(
        result_files
    )

    duplicate_count = (
        len(result_files)
        - len(unique_result_files)
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("DATASET TEST COMPLETE")
    print("=" * 70)

    print(
        f"Images found       : "
        f"{len(images)}"
    )

    print(
        f"Images processed   : "
        f"{len(results)}"
    )

    print(
        f"Unique CSV files   : "
        f"{len(unique_result_files)}"
    )

    print(
        f"Duplicate CSV rows : "
        f"{duplicate_count}"
    )

    print(
        f"Results saved to   : "
        f"{OUTPUT_FILE}"
    )

    print()

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

    print(
        f"Processing errors : "
        f"{len(processing_errors)}/{len(results)}"
    )

    print(
        f"Average confidence: "
        f"{average_confidence:.3f}"
    )

    print("=" * 70)

    # ========================================================
    # VALIDATION
    # ========================================================

    if len(images) == len(results):
        print(
            "VALIDATION: PASS - "
            "Every image produced exactly one result."
        )
    else:
        print(
            "VALIDATION: WARNING - "
            "Image count and result count differ."
        )

    if duplicate_count == 0:
        print(
            "DUPLICATE CHECK: PASS - "
            "No duplicate CSV rows by filename."
        )
    else:
        print(
            "DUPLICATE CHECK: FAIL - "
            f"{duplicate_count} duplicate rows found."
        )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()