"""Confidence scoring for structured receipt extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from receipt_ocr.schemas import ReceiptData


@dataclass
class FieldConfidence:
    """Confidence information for a single extracted field."""

    field: str
    confidence: float
    status: str
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "confidence": round(self.confidence, 6),
            "status": self.status,
            "reasons": self.reasons,
        }


class ConfidenceScorer:
    """Calculate field-level and overall extraction confidence."""

    HIGH_THRESHOLD = 0.85
    MEDIUM_THRESHOLD = 0.60

    def score_field(
        self,
        field: str,
        value: Any,
        base_confidence: float,
    ) -> FieldConfidence:
        """Score an individual extracted field."""

        confidence = max(0.0, min(1.0, base_confidence))
        reasons: list[str] = []

        if value is None or value == "":
            confidence = 0.0
            status = "missing"
            reasons.append("Field was not confidently extracted.")
        elif confidence >= self.HIGH_THRESHOLD:
            status = "high"
            reasons.append("Field extraction confidence is high.")
        elif confidence >= self.MEDIUM_THRESHOLD:
            status = "medium"
            reasons.append("Field extraction confidence is moderate.")
        else:
            status = "low"
            reasons.append("Field extraction confidence is low.")

        return FieldConfidence(
            field=field,
            confidence=confidence,
            status=status,
            reasons=reasons,
        )

    def score(
        self,
        receipt: ReceiptData,
        ocr_confidence: float = 0.0,
        conflicts: list[str] | None = None,
    ) -> dict[str, Any]:
        """Calculate confidence for all important receipt fields."""

        conflicts = conflicts or []

        fields: dict[str, FieldConfidence] = {}

        fields["merchant"] = self.score_field(
            "merchant",
            receipt.merchant,
            self._field_confidence(
                receipt.merchant,
                receipt.extraction_confidence,
            ),
        )

        fields["receipt_date"] = self.score_field(
            "receipt_date",
            receipt.receipt_date,
            0.0 if receipt.receipt_date is None else receipt.extraction_confidence,
        )

        fields["receipt_number"] = self.score_field(
            "receipt_number",
            receipt.receipt_number,
            receipt.extraction_confidence,
        )

        fields["subtotal"] = self.score_field(
            "subtotal",
            receipt.subtotal,
            receipt.extraction_confidence,
        )

        fields["discount"] = self.score_field(
            "discount",
            receipt.discount,
            receipt.extraction_confidence,
        )

        fields["tax"] = self.score_field(
            "tax",
            receipt.tax,
            receipt.extraction_confidence,
        )

        fields["total"] = self.score_field(
            "total",
            receipt.total,
            receipt.extraction_confidence,
        )

        fields["payment_method"] = self.score_field(
            "payment_method",
            receipt.payment_method,
            receipt.extraction_confidence,
        )

        item_confidences = [
            item.confidence
            for item in receipt.items
            if item.confidence > 0
        ]

        if item_confidences:
            item_confidence = sum(item_confidences) / len(item_confidences)
        else:
            item_confidence = 0.0

        fields["items"] = self.score_field(
            "items",
            receipt.items,
            item_confidence,
        )

        # Penalize confidence when the validator reports conflicts.
        conflict_penalty = min(0.20, len(conflicts) * 0.05)

        if conflict_penalty:
            for field_result in fields.values():
                field_result.confidence = max(
                    0.0,
                    field_result.confidence - conflict_penalty,
                )

                field_result.reasons.append(
                    f"Confidence reduced by conflict penalty "
                    f"({conflict_penalty:.2f})."
                )

        available_confidences = [
            field.confidence
            for field in fields.values()
            if field.status != "missing"
        ]

        if available_confidences:
            overall = sum(available_confidences) / len(
                available_confidences
            )
        else:
            overall = 0.0

        overall = max(0.0, min(1.0, overall))

        return {
            "overall_confidence": round(overall, 6),
            "ocr_confidence": round(ocr_confidence, 6),
            "conflict_count": len(conflicts),
            "conflict_penalty": round(conflict_penalty, 6),
            "fields": {
                name: result.to_dict()
                for name, result in fields.items()
            },
        }

    @staticmethod
    def _field_confidence(
        value: Any,
        extraction_confidence: float,
    ) -> float:
        """Return a bounded confidence value for an extracted field."""

        if value is None or value == "":
            return 0.0

        return max(
            0.0,
            min(1.0, extraction_confidence),
        )