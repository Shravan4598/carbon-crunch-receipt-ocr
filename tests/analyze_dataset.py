"""
Dataset analysis utility for the Carbon Crunch receipt OCR assignment.

This script inspects the receipt dataset without modifying the original data.

It reports:
- Number of files
- Image formats
- Image dimensions
- Aspect ratios
- Color modes
- Corrupted/unreadable images
- Directory distribution
- Basic image-quality statistics

Usage:
    python tests/analyze_dataset.py
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def collect_image_files(dataset_dir: Path) -> list[Path]:
    """Recursively collect supported image files."""
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {dataset_dir}"
        )

    image_files = [
        path
        for path in dataset_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(image_files)


def calculate_image_quality(
    image: np.ndarray,
) -> dict[str, float]:
    """
    Calculate simple image-quality indicators.

    Sharpness:
        Variance of the Laplacian. Lower values generally indicate blur.

    Brightness:
        Mean grayscale intensity.

    Contrast:
        Standard deviation of grayscale intensity.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    return {
        "sharpness": sharpness,
        "brightness": brightness,
        "contrast": contrast,
    }


def analyze_image(path: Path) -> dict[str, Any]:
    """Analyze one image and return metadata."""
    relative_path = path.relative_to(PROJECT_ROOT)

    result: dict[str, Any] = {
        "path": str(relative_path),
        "filename": path.name,
        "extension": path.suffix.lower(),
        "directory": str(path.parent.relative_to(PROJECT_ROOT)),
        "readable": False,
        "width": None,
        "height": None,
        "channels": None,
        "aspect_ratio": None,
        "file_size_bytes": path.stat().st_size,
        "pil_mode": None,
        "sharpness": None,
        "brightness": None,
        "contrast": None,
        "error": None,
    }

    try:
        # PIL gives us reliable metadata such as image mode.
        with Image.open(path) as pil_image:
            result["pil_mode"] = pil_image.mode
            result["width"], result["height"] = pil_image.size

        # OpenCV is used for quality statistics.
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if image is None:
            raise ValueError("OpenCV could not decode the image.")

        result["readable"] = True

        if image.ndim == 2:
            result["channels"] = 1
        else:
            result["channels"] = image.shape[2]

        width = result["width"]
        height = result["height"]

        if width and height:
            result["aspect_ratio"] = round(
                float(width) / float(height),
                4,
            )

        quality = calculate_image_quality(image)

        result["sharpness"] = round(quality["sharpness"], 4)
        result["brightness"] = round(quality["brightness"], 4)
        result["contrast"] = round(quality["contrast"], 4)

    except Exception as exc:
        result["error"] = str(exc)

    return result


def summarize_numeric_values(
    records: list[dict[str, Any]],
    key: str,
) -> dict[str, float | None]:
    """Calculate basic statistics for a numeric field."""
    values = [
        float(record[key])
        for record in records
        if record.get("readable")
        and record.get(key) is not None
    ]

    if not values:
        return {
            "min": None,
            "mean": None,
            "median": None,
            "max": None,
        }

    return {
        "min": round(float(np.min(values)), 4),
        "mean": round(float(np.mean(values)), 4),
        "median": round(float(np.median(values)), 4),
        "max": round(float(np.max(values)), 4),
    }


def build_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a dataset-level summary report."""
    readable_records = [
        record for record in records
        if record["readable"]
    ]

    failed_records = [
        record for record in records
        if not record["readable"]
    ]

    extension_counts = Counter(
        record["extension"]
        for record in records
    )

    mode_counts = Counter(
        record["pil_mode"]
        for record in readable_records
    )

    directory_counts = Counter(
        record["directory"]
        for record in records
    )

    dimension_counts = Counter(
        (
            record["width"],
            record["height"],
        )
        for record in readable_records
    )

    report = {
        "dataset": {
            "path": str(DATASET_DIR),
            "total_files": len(records),
            "readable_images": len(readable_records),
            "unreadable_images": len(failed_records),
        },
        "formats": dict(extension_counts),
        "image_modes": dict(mode_counts),
        "directories": dict(directory_counts),
        "most_common_dimensions": [
            {
                "width": dimensions[0],
                "height": dimensions[1],
                "count": count,
            }
            for dimensions, count in dimension_counts.most_common(20)
        ],
        "quality_statistics": {
            "width": summarize_numeric_values(records, "width"),
            "height": summarize_numeric_values(records, "height"),
            "aspect_ratio": summarize_numeric_values(
                records,
                "aspect_ratio",
            ),
            "sharpness": summarize_numeric_values(
                records,
                "sharpness",
            ),
            "brightness": summarize_numeric_values(
                records,
                "brightness",
            ),
            "contrast": summarize_numeric_values(
                records,
                "contrast",
            ),
        },
        "unreadable_images": failed_records,
    }

    return report


def print_report(report: dict[str, Any]) -> None:
    """Print a concise human-readable report."""
    dataset = report["dataset"]

    print("\n" + "=" * 70)
    print("CARBON CRUNCH RECEIPT DATASET ANALYSIS")
    print("=" * 70)

    print(f"\nDataset path      : {dataset['path']}")
    print(f"Total files       : {dataset['total_files']}")
    print(f"Readable images   : {dataset['readable_images']}")
    print(f"Unreadable images : {dataset['unreadable_images']}")

    print("\nImage formats:")
    for extension, count in report["formats"].items():
        print(f"  {extension:<8} {count}")

    print("\nImage modes:")
    for mode, count in report["image_modes"].items():
        print(f"  {str(mode):<8} {count}")

    print("\nDirectory distribution:")
    for directory, count in report["directories"].items():
        print(f"  {directory:<40} {count}")

    print("\nMost common dimensions:")
    for item in report["most_common_dimensions"][:10]:
        print(
            f"  {item['width']} x {item['height']}"
            f" -> {item['count']} images"
        )

    print("\nQuality statistics:")

    for metric, stats in report["quality_statistics"].items():
        print(f"\n  {metric}:")
        print(f"    min    : {stats['min']}")
        print(f"    mean   : {stats['mean']}")
        print(f"    median : {stats['median']}")
        print(f"    max    : {stats['max']}")

    if report["unreadable_images"]:
        print("\nUnreadable images:")
        for image in report["unreadable_images"][:20]:
            print(f"  {image['path']}")
            print(f"    Error: {image['error']}")

    print("\n" + "=" * 70)


def main() -> None:
    """Run dataset analysis."""
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Scanning dataset: %s",
        DATASET_DIR,
    )

    image_files = collect_image_files(DATASET_DIR)

    logger.info(
        "Found %d image files.",
        len(image_files),
    )

    records: list[dict[str, Any]] = []

    for index, image_path in enumerate(image_files, start=1):
        record = analyze_image(image_path)
        records.append(record)

        if index % 100 == 0:
            logger.info(
                "Analyzed %d/%d images.",
                index,
                len(image_files),
            )

    report = build_report(records)

    report_path = REPORT_DIR / "dataset_analysis.json"

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    csv_path = REPORT_DIR / "image_inventory.csv"

    try:
        import pandas as pd

        dataframe = pd.DataFrame(records)

        dataframe.to_csv(
            csv_path,
            index=False,
        )

        logger.info(
            "Image inventory saved to: %s",
            csv_path,
        )

    except ImportError:
        logger.warning(
            "Pandas not available; CSV inventory was not generated."
        )

    print_report(report)

    logger.info(
        "JSON report saved to: %s",
        report_path,
    )


if __name__ == "__main__":
    main()