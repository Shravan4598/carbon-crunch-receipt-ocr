"""Conflict detection for receipt extraction."""

from __future__ import annotations

from receipt_ocr.schemas import ReceiptData


class ConflictDetector:
    """Detect suspicious or conflicting extracted values."""

    def detect(self, receipt: ReceiptData) -> list[str]:
        """Return a list of detected conflicts."""

        conflicts: list[str] = []

        self._check_item_count(receipt, conflicts)
        self._check_item_prices(receipt, conflicts)
        self._check_discount(receipt, conflicts)
        self._check_total(receipt, conflicts)

        return conflicts

    def _check_item_count(
        self,
        receipt: ReceiptData,
        conflicts: list[str],
    ) -> None:
        """Check whether OCR detected an expected item count."""

        raw_text = receipt.raw_text.upper()

        marker = "ITEMS # SOLD"

        if marker not in raw_text:
            return

        after_marker = raw_text.split(marker, 1)[1]

        numbers = []

        for token in after_marker.split():
            cleaned = token.strip(":#;,.")

            if cleaned.isdigit():
                numbers.append(int(cleaned))

        if not numbers:
            return

        expected_count = numbers[0]
        actual_count = len(receipt.items)

        if expected_count != actual_count:
            conflicts.append(
                f"Receipt indicates {expected_count} items sold, "
                f"but {actual_count} items were extracted."
            )

    def _check_item_prices(
        self,
        receipt: ReceiptData,
        conflicts: list[str],
    ) -> None:
        """Check item price consistency."""

        for index, item in enumerate(receipt.items, start=1):

            if (
                item.quantity is not None
                and item.unit_price is not None
                and item.total_price is not None
            ):
                expected = item.quantity * item.unit_price

                if abs(expected - item.total_price) > 0.10:
                    conflicts.append(
                        f"Item {index} price mismatch: "
                        f"quantity × unit price = {expected:.2f}, "
                        f"but total price = {item.total_price:.2f}."
                    )

    def _check_discount(
        self,
        receipt: ReceiptData,
        conflicts: list[str],
    ) -> None:
        """Check discount consistency."""

        if receipt.discount is None:
            return

        if receipt.subtotal is not None:
            if receipt.discount > receipt.subtotal:
                conflicts.append(
                    "Discount is greater than subtotal."
                )

    def _check_total(
        self,
        receipt: ReceiptData,
        conflicts: list[str],
    ) -> None:
        """Check total against item values."""

        if receipt.total is None:
            return

        item_total = sum(
            item.total_price
            for item in receipt.items
            if item.total_price is not None
        )

        if item_total <= 0:
            return

        discount = receipt.discount or 0.0
        tax = receipt.tax or 0.0

        expected = item_total - discount + tax

        if abs(expected - receipt.total) > 0.10:
            conflicts.append(
                f"Receipt total conflict: expected approximately "
                f"{expected:.2f}, extracted {receipt.total:.2f}."
            )