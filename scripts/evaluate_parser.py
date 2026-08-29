"""Generate a quality report for receipt parser results."""

from __future__ import annotations

import csv
from pathlib import Path


RESULTS_FILE = Path("data/parser_results.csv")


def is_present(value: str | None) -> bool:
    """Return True when a CSV value contains usable data."""
    if value is None:
        return False

    value = value.strip()

    return value not in {
        "",
        "None",
        "null",
        "NULL",
        "nan",
    }


def to_float(value: str | None) -> float | None:
    """Convert a CSV value to float."""
    if not is_present(value):
        return None

    try:
        return float(value)
    except ValueError:
        return None


def main() -> None:
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_FILE}"
        )

    with RESULTS_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    total = len(rows)

    if total == 0:
        print("No parser results found.")
        return

    merchant_count = sum(
        is_present(row.get("merchant"))
        for row in rows
    )

    date_count = sum(
        is_present(row.get("receipt_date"))
        for row in rows
    )

    receipt_number_count = sum(
        is_present(row.get("receipt_number"))
        for row in rows
    )

    item_count = sum(
        int(float(row.get("item_count", 0) or 0)) > 0
        for row in rows
    )

    subtotal_count = sum(
        is_present(row.get("subtotal"))
        for row in rows
    )

    discount_count = sum(
        is_present(row.get("discount"))
        for row in rows
    )

    tax_count = sum(
        is_present(row.get("tax"))
        for row in rows
    )

    total_count = sum(
        is_present(row.get("total"))
        for row in rows
    )

    payment_count = sum(
        is_present(row.get("payment_method"))
        for row in rows
    )

    warning_count = sum(
        int(float(row.get("warning_count", 0) or 0)) > 0
        for row in rows
    )

    no_warning_count = total - warning_count

    confidences = []

    for row in rows:
        confidence = to_float(
            row.get("extraction_confidence")
        )

        if confidence is not None:
            confidences.append(confidence)

    average_confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else 0.0
    )

    # ----------------------------------------------------------
    # Reconciliation statistics
    # ----------------------------------------------------------

    item_reconciled = 0
    item_reconciliation_candidates = 0

    financial_reconciled = 0
    financial_reconciliation_candidates = 0

    for row in rows:
        subtotal = to_float(row.get("subtotal"))
        total_value = to_float(row.get("total"))

        items_total = to_float(
            row.get("items_total")
        )

        discount = to_float(
            row.get("discount")
        )

        tax = to_float(
            row.get("tax")
        )

        # Items -> subtotal
        if items_total is not None and subtotal is not None:
            item_reconciliation_candidates += 1

            if abs(items_total - subtotal) <= 0.05:
                item_reconciled += 1

        # Subtotal -> total
        if subtotal is not None and total_value is not None:
            financial_reconciliation_candidates += 1

            expected = subtotal

            if discount is not None:
                expected -= discount

            if tax is not None:
                expected += tax

            if abs(expected - total_value) <= 0.10:
                financial_reconciled += 1

    item_reconciliation_rate = (
        item_reconciled
        / item_reconciliation_candidates
        * 100
        if item_reconciliation_candidates
        else 0.0
    )

    financial_reconciliation_rate = (
        financial_reconciled
        / financial_reconciliation_candidates
        * 100
        if financial_reconciliation_candidates
        else 0.0
    )

    # ----------------------------------------------------------
    # Report
    # ----------------------------------------------------------

    print()
    print("=" * 64)
    print("RECEIPT OCR QUALITY REPORT")
    print("=" * 64)

    print()
    print("Dataset")
    print("-" * 64)
    print(f"Images processed          : {total}")

    print()
    print("Field Extraction")
    print("-" * 64)

    print(
        f"Merchant extracted        : "
        f"{merchant_count}/{total} "
        f"({merchant_count / total * 100:.1f}%)"
    )

    print(
        f"Date extracted            : "
        f"{date_count}/{total} "
        f"({date_count / total * 100:.1f}%)"
    )

    print(
        f"Receipt number extracted  : "
        f"{receipt_number_count}/{total} "
        f"({receipt_number_count / total * 100:.1f}%)"
    )

    print(
        f"Items extracted           : "
        f"{item_count}/{total} "
        f"({item_count / total * 100:.1f}%)"
    )

    print(
        f"Subtotal extracted        : "
        f"{subtotal_count}/{total} "
        f"({subtotal_count / total * 100:.1f}%)"
    )

    print(
        f"Discount extracted        : "
        f"{discount_count}/{total} "
        f"({discount_count / total * 100:.1f}%)"
    )

    print(
        f"Tax extracted             : "
        f"{tax_count}/{total} "
        f"({tax_count / total * 100:.1f}%)"
    )

    print(
        f"Total extracted           : "
        f"{total_count}/{total} "
        f"({total_count / total * 100:.1f}%)"
    )

    print(
        f"Payment method extracted  : "
        f"{payment_count}/{total} "
        f"({payment_count / total * 100:.1f}%)"
    )

    print()
    print("Warnings & Confidence")
    print("-" * 64)

    print(
        f"Receipts without warnings : "
        f"{no_warning_count}/{total} "
        f"({no_warning_count / total * 100:.1f}%)"
    )

    print(
        f"Receipts with warnings    : "
        f"{warning_count}/{total} "
        f"({warning_count / total * 100:.1f}%)"
    )

    print(
        f"Average confidence        : "
        f"{average_confidence:.3f}"
    )

    print()
    print("Reconciliation")
    print("-" * 64)

    print(
        f"Items ↔ subtotal          : "
        f"{item_reconciled}/"
        f"{item_reconciliation_candidates} "
        f"({item_reconciliation_rate:.1f}%)"
    )

    print(
        f"Subtotal ↔ total         : "
        f"{financial_reconciled}/"
        f"{financial_reconciliation_candidates} "
        f"({financial_reconciliation_rate:.1f}%)"
    )

    print()
    print("=" * 64)


if __name__ == "__main__":
    main()