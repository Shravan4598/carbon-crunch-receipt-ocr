"""Business-level data schemas for the receipt OCR pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ReceiptItem:
    """Represents a single purchased item."""

    name: str
    quantity: float | None = None
    unit_price: float | None = None
    total_price: float | None = None
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert item to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class ReceiptData:
    """Structured receipt information extracted from OCR."""

    merchant: str | None = None
    receipt_date: str | None = None
    receipt_number: str | None = None

    items: list[ReceiptItem] = field(default_factory=list)

    subtotal: float | None = None
    discount: float | None = None
    tax: float | None = None
    total: float | None = None

    payment_method: str | None = None
    currency: str = "USD"

    extraction_confidence: float = 0.0

    raw_text: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert receipt data to a JSON-serializable dictionary."""
        return {
            "merchant": self.merchant,
            "receipt_date": self.receipt_date,
            "receipt_number": self.receipt_number,
            "items": [item.to_dict() for item in self.items],
            "subtotal": self.subtotal,
            "discount": self.discount,
            "tax": self.tax,
            "total": self.total,
            "payment_method": self.payment_method,
            "currency": self.currency,
            "extraction_confidence": round(
                self.extraction_confidence,
                6,
            ),
            "raw_text": self.raw_text,
            "warnings": self.warnings,
        }
