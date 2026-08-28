"""Rule-based receipt information parser."""

from __future__ import annotations

import logging
import re
from typing import Iterable

from receipt_ocr.ocr.models import OCRLine, OCRResult
from receipt_ocr.schemas import ReceiptData, ReceiptItem

logger = logging.getLogger(__name__)


class ReceiptParser:
    """Extract structured receipt information from OCR output."""

    TOTAL_PATTERNS = (
        r"\bgrand\s*total\b",
        r"\btotal\s*(?:due|amount)?\b",
        r"\bamount\s*due\b",
        r"\bbalance\s*due\b",
        r"\bnet\s*total\b",
    )

    SUBTOTAL_PATTERNS = (
        r"\bsubtotal\b",
        r"\bsub\s*total\b",
    )

    DISCOUNT_PATTERNS = (
        r"\bdiscount\b",
        r"\bsavings\b",
        r"\bdiscount\s*given\b",
    )

    TAX_PATTERNS = (
        r"\btax\b",
        r"\bsales\s*tax\b",
        r"\bgst\b",
        r"\bvat\b",
    )

    PAYMENT_PATTERNS = (
        "cash",
        "credit",
        "debit",
        "visa",
        "mastercard",
        "amex",
        "card",
        "upi",
        "paytm",
        "phonepe",
        "google pay",
    )

    DATE_PATTERN = re.compile(
        r"\b("
        r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"
        r"|"
        r"\d{1,2}\s+"
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
        r"(?:\s+\d{2,4})?"
        r")\b",
        re.IGNORECASE,
    )

    # Supports:
    # 12.34
    # 1,234.56
    # $12.34
    # €12.34
    # £12.34
    # ₹12.34
    MONEY_PATTERN = re.compile(
        r"(?<![\w.])"
        r"(?:[$€£₹]\s*)?"
        r"\d{1,3}(?:,\d{3})*(?:\.\d{2})"
        r"|"
        r"(?:[$€£₹]\s*)?"
        r"\d+\.\d{2}"
        r"(?!\w)",
        re.UNICODE,
    )

    WEIGHT_PATTERN = re.compile(
        r"\b"
        r"(\d+(?:\.\d+)?)"
        r"\s*"
        r"(lb|lbs|kg|g|oz)"
        r"\b",
        re.IGNORECASE,
    )

    QUANTITY_PATTERN = re.compile(
        r"\b"
        r"(?:qty|quantity)"
        r"\s*[:=]?\s*"
        r"(\d+(?:\.\d+)?)"
        r"\b",
        re.IGNORECASE,
    )

    MULTIPLICATION_PATTERN = re.compile(
        r"\b"
        r"(\d+(?:\.\d+)?)"
        r"\s*[x×]\s*"
        r"(?:[$€£₹]\s*)?"
        r"(\d+(?:\.\d+)?)"
        r"\b",
        re.IGNORECASE,
    )

    def parse(self, ocr_result: OCRResult) -> ReceiptData:
        """Parse OCR output into structured receipt information."""

        lines = list(ocr_result.lines)

        if not lines:
            return ReceiptData(
                raw_text=ocr_result.full_text,
                warnings=["No OCR lines were detected."],
            )

        logger.info(
            "Parsing receipt | lines=%d | OCR confidence=%.3f",
            len(lines),
            ocr_result.average_confidence,
        )

        merchant = self._extract_merchant(lines)
        receipt_date = self._extract_date(lines)
        receipt_number = self._extract_receipt_number(lines)

        subtotal = self._extract_amount_by_patterns(
            lines,
            self.SUBTOTAL_PATTERNS,
        )

        discount = self._extract_amount_by_patterns(
            lines,
            self.DISCOUNT_PATTERNS,
        )

        tax = self._extract_amount_by_patterns(
            lines,
            self.TAX_PATTERNS,
        )

        total = self._extract_total(lines)

        payment_method = self._extract_payment_method(lines)

        items = self._extract_items(
            lines,
            subtotal=subtotal,
            discount=discount,
            tax=tax,
            total=total,
        )

        warnings = self._build_warnings(
            merchant=merchant,
            total=total,
            items=items,
            ocr_confidence=ocr_result.average_confidence,
        )

        warnings.extend(
            self._build_reconciliation_warnings(
                items=items,
                subtotal=subtotal,
                discount=discount,
                tax=tax,
                total=total,
            )
        )

        confidence = self._calculate_confidence(
            ocr_confidence=ocr_result.average_confidence,
            merchant=merchant,
            total=total,
            items=items,
            warnings=warnings,
        )

        result = ReceiptData(
            merchant=merchant,
            receipt_date=receipt_date,
            receipt_number=receipt_number,
            items=items,
            subtotal=subtotal,
            discount=discount,
            tax=tax,
            total=total,
            payment_method=payment_method,
            extraction_confidence=confidence,
            raw_text=ocr_result.full_text,
            warnings=warnings,
        )

        logger.info(
            "Receipt parsed | merchant=%s | items=%d | total=%s "
            "| confidence=%.3f",
            merchant,
            len(items),
            total,
            confidence,
        )

        return result

    # ------------------------------------------------------------------
    # Merchant
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_merchant(
        lines: list[OCRLine],
    ) -> str | None:
        """Extract likely merchant from the upper receipt area."""

        candidates = lines[:10]

        ignored_exact = {
            "low prices always",
            "supercenter",
            "hours open",
            "hours open 24",
            "thank you",
            "thank you for shopping",
        }

        ignored_contains = (
            "hours",
            "manager",
            "phone",
            "tel",
            "address",
            "store #",
            "store no",
        )

        for line in candidates:
            text = line.text.strip()

            if not text:
                continue

            normalized = re.sub(
                r"[^a-zA-Z0-9\s*&'./-]",
                "",
                text,
            )

            normalized = re.sub(
                r"\s+",
                " ",
                normalized,
            ).strip()

            if not normalized:
                continue

            lower = normalized.lower()

            if lower in ignored_exact:
                continue

            if any(
                keyword in lower
                for keyword in ignored_contains
            ):
                continue

            # Don't select lines that are primarily numbers.
            alpha_count = sum(
                char.isalpha()
                for char in normalized
            )

            digit_count = sum(
                char.isdigit()
                for char in normalized
            )

            if alpha_count < 3:
                continue

            if digit_count > alpha_count:
                continue

            # Avoid selecting long slogans.
            if len(normalized) > 35:
                continue

            return normalized

        return None

    # ------------------------------------------------------------------
    # Date
    # ------------------------------------------------------------------

    def _extract_date(
        self,
        lines: Iterable[OCRLine],
    ) -> str | None:
        """Extract the first date-like value."""

        for line in lines:
            match = self.DATE_PATTERN.search(line.text)

            if match:
                return match.group(1)

        return None

    # ------------------------------------------------------------------
    # Receipt number
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_receipt_number(
        lines: Iterable[OCRLine],
    ) -> str | None:
        """Extract a likely receipt or transaction identifier."""

        patterns = (
            r"\b(?:receipt|transaction|trans|txn|order)"
            r"\s*#?\s*[:\-]?\s*([A-Z0-9-]+)",

            r"\bop#\s*([A-Z0-9-]+)",

            r"\b(?:tr#|st#|te#)\s*([A-Z0-9-]+)",
        )

        for line in lines:
            text = line.text

            for pattern in patterns:
                match = re.search(
                    pattern,
                    text,
                    re.IGNORECASE,
                )

                if match:
                    return match.group(1)

        return None

    # ------------------------------------------------------------------
    # Financial fields
    # ------------------------------------------------------------------

    def _extract_total(
        self,
        lines: list[OCRLine],
    ) -> float | None:
        """Extract the most likely final receipt total."""

        candidates: list[tuple[int, float]] = []

        for index, line in enumerate(lines):
            lower = line.text.lower()

            if not any(
                re.search(pattern, lower)
                for pattern in self.TOTAL_PATTERNS
            ):
                continue

            amount = self._last_money_value(line.text)

            if amount is None:
                continue

            candidates.append(
                (index, amount)
            )

        if candidates:
            # Prefer the lowest occurrence on the receipt.
            candidates.sort(
                key=lambda value: value[0],
                reverse=True,
            )

            return candidates[0][1]

        return None

    def _extract_amount_by_patterns(
        self,
        lines: Iterable[OCRLine],
        patterns: tuple[str, ...],
    ) -> float | None:
        """Extract an amount from a matching financial line."""

        for line in lines:
            lower = line.text.lower()

            if any(
                re.search(pattern, lower)
                for pattern in patterns
            ):
                amount = self._last_money_value(
                    line.text
                )

                if amount is not None:
                    return amount

        return None

    # ------------------------------------------------------------------
    # Payment
    # ------------------------------------------------------------------

    def _extract_payment_method(
        self,
        lines: Iterable[OCRLine],
    ) -> str | None:
        """Detect likely payment method."""

        # More specific methods should win over generic "card".
        priority = (
            "google pay",
            "phonepe",
            "paytm",
            "mastercard",
            "visa",
            "amex",
            "credit",
            "debit",
            "upi",
            "cash",
            "card",
        )

        text = "\n".join(
            line.text.lower()
            for line in lines
        )

        for payment in priority:
            if re.search(
                rf"\b{re.escape(payment)}\b",
                text,
            ):
                return payment.upper()

        return None

    # ------------------------------------------------------------------
    # Items
    # ------------------------------------------------------------------

    def _extract_items(
        self,
        lines: list[OCRLine],
        *,
        subtotal: float | None,
        discount: float | None,
        tax: float | None,
        total: float | None,
    ) -> list[ReceiptItem]:
        """Extract likely purchased items."""

        items: list[ReceiptItem] = []

        excluded_keywords = (
            "subtotal",
            "sub total",
            "discount",
            "savings",
            "tax",
            "sales tax",
            "gst",
            "vat",
            "total",
            "amount due",
            "balance due",
            "cash",
            "change",
            "items",
            "manager",
            "hours",
            "supercenter",
            "receipt",
            "transaction",
            "order",
            "payment",
            "tend",
        )

        # Item extraction usually happens in the middle section.
        for index, line in enumerate(lines):
            text = line.text.strip()

            if not text:
                continue

            lower = text.lower()

            if any(
                keyword in lower
                for keyword in excluded_keywords
            ):
                continue

            # Header/footer protection.
            if index < 5 and len(text) > 30:
                continue

            if self._looks_like_identifier(text):
                continue

            amounts = self.MONEY_PATTERN.findall(text)

            if not amounts:
                continue

            price = self._last_money_value(text)

            if price is None:
                continue

            item_info = self._parse_item_line(
                text,
                line.confidence,
            )

            if item_info is None:
                continue

            items.append(item_info)

        return items

    def _parse_item_line(
        self,
        text: str,
        line_confidence: float,
    ) -> ReceiptItem | None:
        """Parse one potential item line."""

        # --------------------------------------------------------------
        # Weighted item
        # Example:
        # 0.41 lb @ 1 lb / 0.49 0.20
        # --------------------------------------------------------------

        weight_match = self.WEIGHT_PATTERN.search(text)

        if weight_match:
            quantity = float(
                weight_match.group(1)
            )

            money_values = self._money_values(text)

            if not money_values:
                return None

            total_price = money_values[-1]

            unit_price = None

            # Common weighted format:
            # quantity lb @ 1 lb / 0.49
            at_match = re.search(
                r"@\s*(?:1\s*)?(?:lb|lbs|kg|g)?"
                r"\s*/?\s*"
                r"(?:[$€£₹]\s*)?"
                r"(\d+(?:\.\d{2}))",
                text,
                re.IGNORECASE,
            )

            if at_match:
                unit_price = float(
                    at_match.group(1)
                )

            # If no explicit unit price was found,
            # infer it from total / weight.
            if unit_price is None and quantity > 0:
                inferred = total_price / quantity

                if 0 < inferred < 10000:
                    unit_price = round(
                        inferred,
                        2,
                    )

            name = self._clean_item_name(
                text,
                remove_weight=True,
            )

            if not name:
                return None

            confidence = self._item_confidence(
                line_confidence=line_confidence,
                name=name,
                quantity=quantity,
                unit_price=unit_price,
                total_price=total_price,
            )

            return ReceiptItem(
                name=name,
                quantity=quantity,
                unit_price=unit_price,
                total_price=total_price,
                confidence=confidence,
            )

        # --------------------------------------------------------------
        # Explicit quantity
        # --------------------------------------------------------------

        quantity = None

        quantity_match = (
            self.QUANTITY_PATTERN.search(text)
        )

        if quantity_match:
            quantity = float(
                quantity_match.group(1)
            )

        # --------------------------------------------------------------
        # Multiplication format
        # Example:
        # 2 x 5.99 11.98
        # --------------------------------------------------------------

        multiplication = (
            self.MULTIPLICATION_PATTERN.search(text)
        )

        money_values = self._money_values(text)

        if not money_values:
            return None

        total_price = money_values[-1]

        unit_price = None

        if multiplication:
            quantity = float(
                multiplication.group(1)
            )

            unit_price = float(
                multiplication.group(2)
            )

        # --------------------------------------------------------------
        # Generic item
        # --------------------------------------------------------------

        name = self._clean_item_name(
            text,
            remove_weight=False,
        )

        if not name:
            return None

        # Don't accept lines whose name is almost entirely numeric.
        alpha_count = sum(
            char.isalpha()
            for char in name
        )

        if alpha_count < 2:
            return None

        if quantity is not None and unit_price is None:
            if quantity > 0:
                inferred = total_price / quantity

                if inferred < 10000:
                    unit_price = round(
                        inferred,
                        2,
                    )

        # If quantity isn't explicitly available,
        # assume one item only when this looks like a normal
        # product-price line.
        if quantity is None:
            quantity = 1.0

        confidence = self._item_confidence(
            line_confidence=line_confidence,
            name=name,
            quantity=quantity,
            unit_price=unit_price,
            total_price=total_price,
        )

        return ReceiptItem(
            name=name,
            quantity=quantity,
            unit_price=unit_price,
            total_price=total_price,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Item helpers
    # ------------------------------------------------------------------

    def _clean_item_name(
        self,
        text: str,
        *,
        remove_weight: bool,
    ) -> str:
        """Remove prices and receipt-specific noise from item text."""

        cleaned = text

        if remove_weight:
            cleaned = self.WEIGHT_PATTERN.sub(
                "",
                cleaned,
            )

        # Remove multiplication expressions.
        cleaned = self.MULTIPLICATION_PATTERN.sub(
            "",
            cleaned,
        )

        # Remove quantity markers.
        cleaned = self.QUANTITY_PATTERN.sub(
            "",
            cleaned,
        )

        # Remove "@" pricing fragments.
        cleaned = re.sub(
            r"@\s*"
            r"(?:1\s*)?"
            r"(?:lb|lbs|kg|g)?"
            r"\s*/?\s*"
            r"(?:[$€£₹]\s*)?"
            r"\d+(?:\.\d{2})?",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        # Remove all money-like values.
        cleaned = self.MONEY_PATTERN.sub(
            "",
            cleaned,
        )

        # Remove common OCR markers at the end.
        cleaned = re.sub(
            r"\b[FN]\b",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\b\d{6,}\b",
            "",
            cleaned,
        )

        cleaned = re.sub(
            r"\s+",
            " ",
            cleaned,
        ).strip()

        cleaned = cleaned.strip(
            "-:./,;@"
        )

        return cleaned

    @staticmethod
    def _looks_like_identifier(
        text: str,
    ) -> bool:
        """Determine whether a line is probably an ID/barcode."""

        digits = sum(
            char.isdigit()
            for char in text
        )

        letters = sum(
            char.isalpha()
            for char in text
        )

        # Long numeric strings are generally barcodes,
        # receipt IDs or transaction identifiers.
        if digits >= 10 and digits > letters * 2:
            return True

        return False

    @classmethod
    def _money_values(
        cls,
        text: str,
    ) -> list[float]:
        """Return all money-like values from a string."""

        values: list[float] = []

        for raw in cls.MONEY_PATTERN.findall(text):
            value = cls._parse_money(raw)

            if value is not None:
                values.append(value)

        return values

    @classmethod
    def _last_money_value(
        cls,
        text: str,
    ) -> float | None:
        """Return the final money-like value."""

        values = cls._money_values(text)

        if not values:
            return None

        return values[-1]

    @staticmethod
    def _parse_money(
        raw: str,
    ) -> float | None:
        """Convert a money string into float."""

        raw = raw.strip()

        raw = re.sub(
            r"[$€£₹\s]",
            "",
            raw,
        )

        try:
            if "," in raw and "." in raw:
                raw = raw.replace(",", "")
            elif "," in raw:
                raw = raw.replace(",", "")

            return float(raw)

        except ValueError:
            return None

    @staticmethod
    def _item_confidence(
        *,
        line_confidence: float,
        name: str,
        quantity: float | None,
        unit_price: float | None,
        total_price: float | None,
    ) -> float:
        """Calculate confidence for one extracted item."""

        score = line_confidence

        if len(name) >= 3:
            score += 0.05

        if quantity is not None:
            score += 0.03

        if unit_price is not None:
            score += 0.04

        if total_price is not None:
            score += 0.04

        return max(
            0.0,
            min(1.0, score),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _build_reconciliation_warnings(
        self,
        *,
        items: list[ReceiptItem],
        subtotal: float | None,
        discount: float | None,
        tax: float | None,
        total: float | None,
    ) -> list[str]:
        """Check whether extracted financial fields reconcile."""

        warnings: list[str] = []

        if not items:
            return warnings

        item_sum = sum(
            item.total_price
            for item in items
            if item.total_price is not None
        )

        if subtotal is not None:
            if abs(item_sum - subtotal) > 0.05:
                warnings.append(
                    "Extracted item totals do not reconcile "
                    "with the detected subtotal."
                )

        if total is not None and subtotal is not None:
            expected = subtotal

            if discount is not None:
                expected -= discount

            if tax is not None:
                expected += tax

            if abs(expected - total) > 0.10:
                warnings.append(
                    "Subtotal, discount, tax and total "
                    "do not reconcile."
                )

        return warnings

    # ------------------------------------------------------------------
    # Warnings
    # ------------------------------------------------------------------

    def _build_warnings(
        self,
        *,
        merchant: str | None,
        total: float | None,
        items: list[ReceiptItem],
        ocr_confidence: float,
    ) -> list[str]:
        """Generate warnings for incomplete extraction."""

        warnings: list[str] = []

        if merchant is None:
            warnings.append(
                "Merchant could not be confidently identified."
            )

        if total is None:
            warnings.append(
                "Receipt total could not be confidently identified."
            )

        if not items:
            warnings.append(
                "No purchasable items could be confidently extracted."
            )

        if ocr_confidence < 0.50:
            warnings.append(
                "Overall OCR confidence is low; "
                "manual review recommended."
            )

        elif ocr_confidence < 0.70:
            warnings.append(
                "OCR confidence is moderate; "
                "extracted fields should be reviewed."
            )

        return warnings

    # ------------------------------------------------------------------
    # Overall confidence
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_confidence(
        *,
        ocr_confidence: float,
        merchant: str | None,
        total: float | None,
        items: list[ReceiptItem],
        warnings: list[str],
    ) -> float:
        """Calculate overall extraction confidence."""

        score = ocr_confidence

        if merchant is not None:
            score += 0.08

        if total is not None:
            score += 0.12

        if items:
            score += 0.08

        # Penalize serious extraction problems.
        score -= 0.03 * len(warnings)

        return max(
            0.0,
            min(1.0, score),
        )