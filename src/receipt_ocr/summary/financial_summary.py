"""Financial summary generation for receipt OCR results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from receipt_ocr.schemas import ReceiptData


@dataclass
class FinancialSummary:
    """Aggregated financial and pipeline statistics."""

    receipt_count: int = 0

    total_spend: float = 0.0
    average_receipt_amount: float = 0.0

    total_subtotal: float = 0.0
    total_tax: float = 0.0
    total_discount: float = 0.0

    total_items: int = 0
    average_items_per_receipt: float = 0.0

    receipts_with_totals: int = 0
    receipts_without_totals: int = 0

    receipts_with_conflicts: int = 0
    receipts_with_warnings: int = 0

    average_ocr_confidence: float = 0.0
    average_extraction_confidence: float = 0.0

    currency: str = "USD"

    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert summary into a JSON-serializable dictionary."""

        return {
            "receipt_count": self.receipt_count,
            "total_spend": round(self.total_spend, 2),
            "average_receipt_amount": round(
                self.average_receipt_amount,
                2,
            ),
            "total_subtotal": round(self.total_subtotal, 2),
            "total_tax": round(self.total_tax, 2),
            "total_discount": round(self.total_discount, 2),
            "total_items": self.total_items,
            "average_items_per_receipt": round(
                self.average_items_per_receipt,
                2,
            ),
            "receipts_with_totals": self.receipts_with_totals,
            "receipts_without_totals": self.receipts_without_totals,
            "receipts_with_conflicts": self.receipts_with_conflicts,
            "receipts_with_warnings": self.receipts_with_warnings,
            "average_ocr_confidence": round(
                self.average_ocr_confidence,
                6,
            ),
            "average_extraction_confidence": round(
                self.average_extraction_confidence,
                6,
            ),
            "currency": self.currency,
            "warnings": self.warnings,
        }


class FinancialSummaryGenerator:
    """Generate financial summaries from structured receipts."""

    def generate(
        self,
        receipts: list[ReceiptData],
        ocr_confidences: list[float] | None = None,
        conflict_counts: list[int] | None = None,
        validation_warning_counts: list[int] | None = None,
    ) -> FinancialSummary:
        """Generate an aggregate financial summary."""

        summary = FinancialSummary()

        if not receipts:
            summary.warnings.append(
                "No receipts were provided for financial summary."
            )
            return summary

        summary.receipt_count = len(receipts)

        ocr_confidences = ocr_confidences or []
        conflict_counts = conflict_counts or []
        validation_warning_counts = (
            validation_warning_counts or []
        )

        total_values: list[float] = []
        extraction_confidences: list[float] = []

        for index, receipt in enumerate(receipts):

            # -------------------------------------------------
            # Total
            # -------------------------------------------------

            if receipt.total is not None:
                total = float(receipt.total)

                summary.total_spend += total
                total_values.append(total)
                summary.receipts_with_totals += 1

            else:
                summary.receipts_without_totals += 1

            # -------------------------------------------------
            # Financial fields
            # -------------------------------------------------

            if receipt.subtotal is not None:
                summary.total_subtotal += float(
                    receipt.subtotal
                )

            if receipt.tax is not None:
                summary.total_tax += float(receipt.tax)

            if receipt.discount is not None:
                summary.total_discount += float(
                    receipt.discount
                )

            # -------------------------------------------------
            # Items
            # -------------------------------------------------

            summary.total_items += len(receipt.items)

            # -------------------------------------------------
            # Confidence
            # -------------------------------------------------

            extraction_confidences.append(
                float(receipt.extraction_confidence)
            )

            # -------------------------------------------------
            # Conflicts
            # -------------------------------------------------

            if index < len(conflict_counts):
                if conflict_counts[index] > 0:
                    summary.receipts_with_conflicts += 1

            # -------------------------------------------------
            # Validation warnings
            # -------------------------------------------------

            if index < len(validation_warning_counts):
                if validation_warning_counts[index] > 0:
                    summary.receipts_with_warnings += 1

            # -------------------------------------------------
            # Currency
            # -------------------------------------------------

            if receipt.currency:
                summary.currency = receipt.currency

        # -----------------------------------------------------
        # Averages
        # -----------------------------------------------------

        if total_values:
            summary.average_receipt_amount = (
                sum(total_values) / len(total_values)
            )

        if summary.receipt_count:
            summary.average_items_per_receipt = (
                summary.total_items
                / summary.receipt_count
            )

        if ocr_confidences:
            summary.average_ocr_confidence = (
                sum(ocr_confidences)
                / len(ocr_confidences)
            )

        if extraction_confidences:
            summary.average_extraction_confidence = (
                sum(extraction_confidences)
                / len(extraction_confidences)
            )

        # -----------------------------------------------------
        # Summary warnings
        # -----------------------------------------------------

        if summary.receipts_without_totals > 0:
            summary.warnings.append(
                f"{summary.receipts_without_totals} receipt(s) "
                "have no confidently extracted total."
            )

        if summary.receipts_with_conflicts > 0:
            summary.warnings.append(
                f"{summary.receipts_with_conflicts} receipt(s) "
                "contain detected conflicts."
            )

        return summary