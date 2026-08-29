"""Run the complete receipt OCR pipeline on a directory."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from receipt_ocr.ocr.tesseract_engine import TesseractOCREngine
from receipt_ocr.parser import ReceiptParser
from receipt_ocr.validation import (
    ConflictDetector,
    ReceiptValidator,
)
from receipt_ocr.confidence import ConfidenceScorer


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def setup_logging() -> None:
    """Configure application logging."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )


def process_receipt(
    image_path: Path,
    output_dir: Path,
    ocr_engine: TesseractOCREngine,
    parser: ReceiptParser,
    validator: ReceiptValidator,
    conflict_detector: ConflictDetector,
    confidence_scorer: ConfidenceScorer,
) -> dict[str, Any]:
    """Process one receipt image through the complete pipeline."""

    logger = logging.getLogger(__name__)

    started = time.perf_counter()

    result: dict[str, Any] = {
        "image": image_path.name,
        "image_path": str(image_path),
        "status": "failed",
    }

    try:
        # --------------------------------------------------
        # OCR
        # --------------------------------------------------

        ocr_result = ocr_engine.extract(image_path)

        # --------------------------------------------------
        # Parsing
        # --------------------------------------------------

        receipt = parser.parse(ocr_result)

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        validation = validator.validate(receipt)

        # --------------------------------------------------
        # Conflict detection
        # --------------------------------------------------

        conflicts = conflict_detector.detect(receipt)

        # --------------------------------------------------
        # Confidence scoring
        # --------------------------------------------------

        confidence = confidence_scorer.score(
            receipt=receipt,
            ocr_confidence=ocr_result.average_confidence,
            conflicts=conflicts,
        )

        # --------------------------------------------------
        # Build output
        # --------------------------------------------------

        elapsed = time.perf_counter() - started

        result = {
            "image": image_path.name,
            "image_path": str(image_path),
            "status": "success",

            "ocr": {
                "engine": ocr_result.engine,
                "engine_version": ocr_result.metadata.get(
                    "engine_version"
                ),
                "image_width": ocr_result.metadata.get(
                    "image_width"
                ),
                "image_height": ocr_result.metadata.get(
                    "image_height"
                ),
                "line_count": len(
                    ocr_result.text.splitlines()
                ),
                "word_count": len(ocr_result.words),
                "average_confidence": ocr_result.average_confidence,
                "processing_time_seconds": (
                    ocr_result.processing_time
                ),
                "full_text": ocr_result.text,
            },

            "receipt": receipt.to_dict(),

            "validation": validation.to_dict(),

            "conflicts": {
                "count": len(conflicts),
                "items": conflicts,
            },

            "confidence": confidence,

            "pipeline": {
                "processing_time_seconds": round(
                    elapsed,
                    4,
                )
            },
        }

        # --------------------------------------------------
        # Save individual receipt JSON
        # --------------------------------------------------

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = output_dir / (
            f"{image_path.stem}.json"
        )

        output_file.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        logger.info(
            "Processed %s | confidence=%.3f | conflicts=%d",
            image_path.name,
            confidence["overall_confidence"],
            len(conflicts),
        )

        return result

    except Exception as exc:
        elapsed = time.perf_counter() - started

        logger.exception(
            "Failed to process %s",
            image_path.name,
        )

        result.update(
            {
                "error": str(exc),
                "pipeline": {
                    "processing_time_seconds": round(
                        elapsed,
                        4,
                    )
                },
            }
        )

        return result


def generate_batch_summary(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate aggregate batch statistics."""

    successful = [
        result
        for result in results
        if result.get("status") == "success"
    ]

    failed = [
        result
        for result in results
        if result.get("status") == "failed"
    ]

    total_spend = 0.0
    total_items = 0

    receipts_with_totals = 0
    receipts_with_conflicts = 0
    receipts_with_warnings = 0

    ocr_confidences: list[float] = []
    extraction_confidences: list[float] = []
    processing_times: list[float] = []

    for result in successful:

        receipt = result["receipt"]

        total = receipt.get("total")

        if total is not None:
            total_spend += float(total)
            receipts_with_totals += 1

        total_items += len(
            receipt.get("items", [])
        )

        if result["conflicts"]["count"] > 0:
            receipts_with_conflicts += 1

        if result["validation"]["warning_count"] > 0:
            receipts_with_warnings += 1

        ocr_confidences.append(
            float(
                result["ocr"]["average_confidence"]
            )
        )

        extraction_confidences.append(
            float(
                receipt["extraction_confidence"]
            )
        )

        processing_times.append(
            float(
                result["pipeline"][
                    "processing_time_seconds"
                ]
            )
        )

    receipt_count = len(successful)

    return {
        "dataset": {
            "total_images": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": (
                len(successful) / len(results)
                if results
                else 0.0
            ),
        },

        "financial": {
            "total_spend": round(
                total_spend,
                2,
            ),
            "receipts_with_totals": (
                receipts_with_totals
            ),
            "receipts_without_totals": (
                receipt_count
                - receipts_with_totals
            ),
        },

        "extraction": {
            "total_items": total_items,
            "average_items_per_receipt": (
                total_items / receipt_count
                if receipt_count
                else 0.0
            ),
            "receipts_with_conflicts": (
                receipts_with_conflicts
            ),
            "receipts_with_warnings": (
                receipts_with_warnings
            ),
        },

        "confidence": {
            "average_ocr_confidence": (
                sum(ocr_confidences)
                / len(ocr_confidences)
                if ocr_confidences
                else 0.0
            ),
            "average_extraction_confidence": (
                sum(extraction_confidences)
                / len(extraction_confidences)
                if extraction_confidences
                else 0.0
            ),
        },

        "performance": {
            "total_processing_time_seconds": round(
                sum(processing_times),
                3,
            ),
            "average_processing_time_seconds": (
                sum(processing_times)
                / len(processing_times)
                if processing_times
                else 0.0
            ),
        },

        "failed_images": [
            {
                "image": result.get("image"),
                "error": result.get("error"),
            }
            for result in failed
        ],
    }


def main() -> None:
    """Run batch receipt processing."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete receipt OCR pipeline."
        )
    )

    parser.add_argument(
        "--input",
        default="data/raw",
        help="Directory containing receipt images.",
    )

    parser.add_argument(
        "--output",
        default="outputs/receipts",
        help="Directory for individual receipt JSON files.",
    )

    args = parser.parse_args()

    setup_logging()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    images = sorted(
        path
        for path in input_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    print("=" * 70)
    print("RECEIPT OCR BATCH PIPELINE")
    print("=" * 70)
    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Images found    : {len(images)}")
    print()

    if not images:
        print("No receipt images found.")
        return

    # ------------------------------------------------------
    # Initialize pipeline components once
    # ------------------------------------------------------

    print("Initializing OCR pipeline...")

    ocr_engine = TesseractOCREngine(
        language="eng",
        psm=6,
        min_confidence=0.0,
    )

    receipt_parser = ReceiptParser()
    validator = ReceiptValidator()
    conflict_detector = ConflictDetector()
    confidence_scorer = ConfidenceScorer()

    print("Pipeline initialized.")
    print()

    # ------------------------------------------------------
    # Process images
    # ------------------------------------------------------

    results: list[dict[str, Any]] = []

    batch_started = time.perf_counter()

    for index, image_path in enumerate(
        images,
        start=1,
    ):

        print(
            f"[{index}/{len(images)}] "
            f"Processing {image_path.name}"
        )

        result = process_receipt(
            image_path=image_path,
            output_dir=output_dir,
            ocr_engine=ocr_engine,
            parser=receipt_parser,
            validator=validator,
            conflict_detector=conflict_detector,
            confidence_scorer=confidence_scorer,
        )

        results.append(result)

    batch_time = (
        time.perf_counter()
        - batch_started
    )

    # ------------------------------------------------------
    # Generate batch summary
    # ------------------------------------------------------

    summary = generate_batch_summary(results)

    summary["performance"][
        "wall_clock_time_seconds"
    ] = round(
        batch_time,
        3,
    )

    # ------------------------------------------------------
    # Save batch results
    # ------------------------------------------------------

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    batch_results_file = (
        outputs_dir / "batch_results.json"
    )

    batch_results_file.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary_file = (
        outputs_dir / "batch_summary.json"
    )

    summary_file.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------
    # Display summary
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("BATCH PIPELINE SUMMARY")
    print("=" * 70)

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print("=" * 70)
    print("PIPELINE COMPLETED")
    print("=" * 70)

    print(
        f"Batch time: {batch_time:.2f} seconds"
    )

    print(
        f"Results saved to: {output_dir}"
    )

    print(
        f"Batch results: {batch_results_file}"
    )

    print(
        f"Batch summary: {summary_file}"
    )


if __name__ == "__main__":
    main()