"""Rule-based receipt information parser.

The parser is designed for OCR engines that may split a single receipt
item across multiple OCR lines.
"""

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
        r"\bdiscount\s*given\b",
        r"\bdiscount\b",
        r"\bsavings\b",
    )

    TAX_PATTERNS = (
        r"\bsales\s*tax\b",
        r"\btax\b",
        r"\bgst\b",
        r"\bvat\b",
    )

    PAYMENT_PATTERNS = (
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

    MONEY_PATTERN = re.compile(
        r"(?<![\w.])"
        r"(?:[$€£₹]\s*)?"
        r"\d{1,3}(?:,\d{3})*\.\d{2}"
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
        r"\s*[x×]"
        r"\s*"
        r"(?:[$€£₹]\s*)?"
        r"(\d+(?:\.\d+)?)"
        r"\b",
        re.IGNORECASE,
    )

    ITEM_SECTION_END_MARKERS = (
        "subtotal",
        "sub total",
        "discount",
        "discount given",
        "savings",
        "tax",
        "sales tax",
        "gst",
        "vat",
        "total",
        "amount due",
        "balance due",
        "cash tend",
        "change due",
        "items sold",
        "thank you",
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

        return ReceiptData(
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

    # ------------------------------------------------------------------
    # Merchant
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_merchant(
        lines: list[OCRLine],
    ) -> str | None:
        """Extract likely merchant from the upper receipt area."""

        for line in lines[:10]:
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

            lower = normalized.lower()

            if not normalized:
                continue

            if lower in {
                "low prices always",
                "always low prices",
                "supercenter",
                "hours open",
                "hours open 24",
                "thank you",
            }:
                continue

            if any(
                word in lower
                for word in (
                    "hours",
                    "manager",
                    "phone",
                    "tel",
                    "address",
                )
            ):
                continue

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
        """Extract first date-like value."""

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
        """Extract receipt / transaction identifier."""

        patterns = (
            r"\b(?:receipt|transaction|trans|txn|order)"
            r"\s*#?\s*[:\-]?\s*([A-Z0-9-]+)",

            r"\bop#\s*([A-Z0-9-]+)",

            r"\b(?:tr#|st#|te#)\s*([A-Z0-9-]+)",
        )

        for line in lines:
            for pattern in patterns:
                match = re.search(
                    pattern,
                    line.text,
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
        """Extract the final receipt total."""

        for line in reversed(lines):
            lower = line.text.lower()

            if "cash tend" in lower:
                continue

            if "change due" in lower:
                continue

            if any(
                re.search(pattern, lower)
                for pattern in self.TOTAL_PATTERNS
            ):
                amount = self._last_money_value(line.text)

                if amount is not None:
                    return amount

        # Handle OCR splitting:
        #
        # TOTAL
        # 5.11
        #
        for index, line in enumerate(lines):
            if line.text.strip().lower() == "total":
                amount = self._find_amount_nearby(
                    lines,
                    index,
                )

                if amount is not None:
                    return amount

        return None

    def _extract_amount_by_patterns(
        self,
        lines: Iterable[OCRLine],
        patterns: tuple[str, ...],
    ) -> float | None:
        """Extract an amount from a matching financial line."""

        lines_list = list(lines)

        for index, line in enumerate(lines_list):
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

                amount = self._find_amount_nearby(
                    lines_list,
                    index,
                )

                if amount is not None:
                    return amount

        return None

    @staticmethod
    def _find_amount_nearby(
        lines: list[OCRLine],
        index: int,
        window: int = 2,
    ) -> float | None:
        """Find a money value immediately after a label."""

        for offset in range(1, window + 1):
            next_index = index + offset

            if next_index >= len(lines):
                break

            text = lines[next_index].text.strip()

            values = ReceiptParser._money_values(text)

            if values:
                return values[-1]

            lower = text.lower()

            if any(
                marker in lower
                for marker in ReceiptParser.ITEM_SECTION_END_MARKERS
            ):
                break

        return None

    # ------------------------------------------------------------------
    # Payment
    # ------------------------------------------------------------------

    def _extract_payment_method(
        self,
        lines: Iterable[OCRLine],
    ) -> str | None:
        """Detect payment method."""

        text = "\n".join(
            line.text.lower()
            for line in lines
        )

        for payment in self.PAYMENT_PATTERNS:
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
        """Extract items from multi-line OCR receipt blocks."""

        items = self._extract_multiline_items(lines)

        if items:
            return items

        return self._extract_single_line_items(lines)

    def _extract_multiline_items(
        self,
        lines: list[OCRLine],
    ) -> list[ReceiptItem]:
        """Parse receipt products whose information spans multiple lines."""

        items: list[ReceiptItem] = []

        start_index = self._find_item_section_start(lines)

        if start_index is None:
            return []

        block: list[OCRLine] = []

        for line in lines[start_index:]:
            text = line.text.strip()

            if not text:
                continue

            lower = text.lower()

            if self._is_item_section_end(lower):
                if block:
                    item = self._parse_item_block(block)

                    if item is not None:
                        items.append(item)

                    block = []

                break

            if self._looks_like_item_name(text):
                if block:
                    item = self._parse_item_block(block)

                    if item is not None:
                        items.append(item)

                block = [line]

            else:
                if block:
                    block.append(line)

        if block:
            item = self._parse_item_block(block)

            if item is not None:
                items.append(item)

        return self._deduplicate_items(items)

    def _find_item_section_start(
        self,
        lines: list[OCRLine],
    ) -> int | None:
        """Find the first plausible product line."""

        for index, line in enumerate(lines):
            text = line.text.strip()

            if self._looks_like_item_name(text):
                if index >= 4:
                    return index

        return None

    def _is_item_section_end(
        self,
        text: str,
    ) -> bool:
        """Return True when receipt product section has ended."""

        normalized = text.lower().strip()

        return any(
            normalized.startswith(marker)
            for marker in self.ITEM_SECTION_END_MARKERS
        )

    def _looks_like_item_name(
        self,
        text: str,
    ) -> bool:
        """Determine whether an OCR line resembles a product name."""

        normalized = text.strip()

        if not normalized:
            return False

        lower = normalized.lower()

        if self._is_item_section_end(lower):
            return False

        if self._looks_like_identifier(normalized):
            return False

        if self.MONEY_PATTERN.fullmatch(normalized):
            return False

        if self.WEIGHT_PATTERN.fullmatch(normalized):
            return False

        if re.fullmatch(
            r"[\d\s./#*_-]+",
            normalized,
        ):
            return False

        if re.fullmatch(
            r"[A-Z0-9-]{8,}",
            normalized,
            re.IGNORECASE,
        ):
            return False

        alpha_count = sum(
            char.isalpha()
            for char in normalized
        )

        digit_count = sum(
            char.isdigit()
            for char in normalized
        )

        if alpha_count < 2:
            return False

        if digit_count > alpha_count * 2:
            return False

        metadata_words = (
            "wal-mart",
            "walmart",
            "supercenter",
            "manager",
            "hours",
            "open",
            "phone",
            "thank you",
            "shop at",
            "backtoschool",
        )

        if any(
            word in lower
            for word in metadata_words
        ):
            return False

        if len(normalized) > 35:
            return False

        return True

    # ------------------------------------------------------------------
    # Item parsing
    # ------------------------------------------------------------------

    def _parse_item_block(
        self,
        block: list[OCRLine],
    ) -> ReceiptItem | None:
        """Parse one multi-line receipt item."""

        if not block:
            return None

        texts = [
            line.text.strip()
            for line in block
            if line.text.strip()
        ]

        if not texts:
            return None

        # The first textual line is normally the product name.
        name = self._clean_product_name(texts[0])

        if not name:
            return None

        # Keep OCR lines joined because quantity, unit price,
        # and total may appear on separate lines.
        full_block = " ".join(texts)

        # ----------------------------------------------------------
        # Quantity / weight
        # ----------------------------------------------------------

        quantity: float | None = None

        weight_match = self.WEIGHT_PATTERN.search(
            full_block
        )

        if weight_match:
            try:
                quantity = float(
                    weight_match.group(1)
                )
            except ValueError:
                quantity = None

        if quantity is None:
            quantity_match = self.QUANTITY_PATTERN.search(
                full_block
            )

            if quantity_match:
                try:
                    quantity = float(
                        quantity_match.group(1)
                    )
                except ValueError:
                    quantity = None

        # ----------------------------------------------------------
        # Money values
        # ----------------------------------------------------------

        money_values = self._money_values(
            full_block
        )

        money_values = [
            value
            for value in money_values
            if 0 <= value < 100000
        ]

        if not money_values:
            return None

        unit_price: float | None = None
        total_price: float | None = None

        # ----------------------------------------------------------
        # Weighted item
        # ----------------------------------------------------------

        if weight_match:

            # Example:
            #
            # BANANAS
            # 0.41 lb
            # 0.20 N
            # 1 lb / 0.49
            #
            # quantity   = 0.41
            # unit_price = 0.49
            # total      = 0.20

            unit_price = self._extract_weight_unit_price(
                full_block
            )

            if unit_price is not None:

                # Prefer the money value that occurs BEFORE
                # the "1 lb / 0.49" unit-price expression.
                unit_price_match = self._find_weight_price_match(
                    full_block
                )

                if unit_price_match:
                    before_unit_price = full_block[
                        :unit_price_match.start()
                    ]

                    previous_values = self._money_values(
                        before_unit_price
                    )

                    if previous_values:
                        total_price = previous_values[-1]

                # If the layout places the actual charge after
                # the unit price, use the nearest non-unit amount.
                if total_price is None:
                    candidates = [
                        value
                        for value in money_values
                        if abs(value - unit_price) > 0.001
                    ]

                    if candidates:
                        total_price = candidates[-1]

            # No explicit price-per-unit was found.
            if total_price is None:
                total_price = money_values[-1]

            # Infer price-per-unit when necessary.
            if (
                unit_price is None
                and quantity is not None
                and quantity > 0
            ):
                unit_price = round(
                    total_price / quantity,
                    2,
                )

        else:

            # ------------------------------------------------------
            # Normal product
            # ------------------------------------------------------

            multiplication = (
                self.MULTIPLICATION_PATTERN.search(
                    full_block
                )
            )

            if multiplication:

                quantity = float(
                    multiplication.group(1)
                )

                unit_price = float(
                    multiplication.group(2)
                )

                total_price = money_values[-1]

            else:

                total_price = money_values[-1]

                if quantity is None:
                    quantity = 1.0

                if quantity > 0:
                    unit_price = round(
                        total_price / quantity,
                        2,
                    )

        # ----------------------------------------------------------
        # Defaults
        # ----------------------------------------------------------

        if quantity is None:
            quantity = 1.0

        if unit_price is None and total_price is not None:
            unit_price = total_price

        if total_price is None:
            return None

        confidence = self._block_confidence(
            block=block,
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

    def _find_weight_price_match(
        self,
        text: str,
    ) -> re.Match[str] | None:
        """Find the price-per-unit expression in a weighted item."""

        patterns = (
            r"(?:1\s*)?"
            r"(?:lb|lbs|kg|g|oz)"
            r"\s*/\s*"
            r"(?:[$€£₹]\s*)?"
            r"\d+(?:,\d{3})*\.\d{2}",

            r"@\s*"
            r"(?:1\s*)?"
            r"(?:lb|lbs|kg|g|oz)?"
            r"\s*/?\s*"
            r"(?:[$€£₹]\s*)?"
            r"\d+(?:,\d{3})*\.\d{2}",
        )

        for pattern in patterns:
            match = re.search(
                pattern,
                text,
                re.IGNORECASE,
            )

            if match:
                return match

        return None

    def _extract_weight_unit_price(
        self,
        text: str,
    ) -> float | None:
        """Extract price-per-unit from a weighted item."""

        patterns = (
            # 1 lb / 0.49
            r"(?:1\s*)?"
            r"(?:lb|lbs|kg|g|oz)"
            r"\s*/\s*"
            r"(?:[$€£₹]\s*)?"
            r"(\d+(?:,\d{3})*\.\d{2})",

            # @ 1 lb / 0.49
            r"@\s*"
            r"(?:1\s*)?"
            r"(?:lb|lbs|kg|g|oz)?"
            r"\s*/?\s*"
            r"(?:[$€£₹]\s*)?"
            r"(\d+(?:,\d{3})*\.\d{2})",
        )

        for pattern in patterns:
            match = re.search(
                pattern,
                text,
                re.IGNORECASE,
            )

            if match:
                try:
                    return self._parse_money(
                        match.group(1)
                    )
                except (TypeError, ValueError):
                    return None

        return None

    def _extract_single_line_items(
        self,
        lines: list[OCRLine],
    ) -> list[ReceiptItem]:
        """Fallback parser for conventional one-line item receipts."""

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
            "payment",
            "tend",
        )

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

            if index < 5:
                continue

            if self._looks_like_identifier(text):
                continue

            money_values = self._money_values(text)

            if not money_values:
                continue

            name = self._clean_product_name(text)

            if not name:
                continue

            total_price = money_values[-1]

            quantity = self._extract_quantity(text)

            if quantity is None:
                quantity = 1.0

            unit_price = total_price

            if quantity > 0:
                unit_price = round(
                    total_price / quantity,
                    2,
                )

            items.append(
                ReceiptItem(
                    name=name,
                    quantity=quantity,
                    unit_price=unit_price,
                    total_price=total_price,
                    confidence=line.confidence,
                )
            )

        return self._deduplicate_items(items)

    # ------------------------------------------------------------------
    # Item helpers
    # ------------------------------------------------------------------

    def _clean_product_name(
        self,
        text: str,
    ) -> str:
        """Clean OCR product name."""

        cleaned = text.strip()

        cleaned = self.WEIGHT_PATTERN.sub(
            "",
            cleaned,
        )

        cleaned = self.QUANTITY_PATTERN.sub(
            "",
            cleaned,
        )

        cleaned = self.MULTIPLICATION_PATTERN.sub(
            "",
            cleaned,
        )

        cleaned = self.MONEY_PATTERN.sub(
            "",
            cleaned,
        )

        cleaned = re.sub(
            r"\b[FN]\b",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s+",
            " ",
            cleaned,
        ).strip()

        cleaned = cleaned.strip(
            "-:;,./"
        )

        if self._looks_like_identifier(cleaned):
            return ""

        alpha_count = sum(
            char.isalpha()
            for char in cleaned
        )

        if alpha_count < 2:
            return ""

        return cleaned

    @staticmethod
    def _extract_quantity(
        text: str,
    ) -> float | None:
        """Extract quantity from an item line."""

        patterns = (
            r"\bqty\s*[:=]?\s*(\d+(?:\.\d+)?)",
            r"\bquantity\s*[:=]?\s*(\d+(?:\.\d+)?)",
        )

        for pattern in patterns:
            match = re.search(
                pattern,
                text,
                re.IGNORECASE,
            )

            if match:
                try:
                    return float(
                        match.group(1)
                    )
                except ValueError:
                    return None

        weight_match = re.search(
            r"\b(\d+(?:\.\d+)?)"
            r"\s*(?:lb|lbs|kg|g|oz)\b",
            text,
            re.IGNORECASE,
        )

        if weight_match:
            try:
                return float(
                    weight_match.group(1)
                )
            except ValueError:
                return None

        return None

    @staticmethod
    def _looks_like_identifier(
        text: str,
    ) -> bool:
        """Detect barcode/SKU/receipt identifier strings."""

        normalized = re.sub(
            r"\s+",
            "",
            text,
        )

        if len(normalized) >= 8 and re.fullmatch(
            r"[A-Z0-9-]+",
            normalized,
            re.IGNORECASE,
        ):
            digit_count = sum(
                char.isdigit()
                for char in normalized
            )

            if digit_count >= 6:
                return True

        if re.fullmatch(
            r"\d{8,}",
            normalized,
        ):
            return True

        return False

    @classmethod
    def _money_values(
        cls,
        text: str,
    ) -> list[float]:
        """Extract all money values from text."""

        values: list[float] = []

        for match in cls.MONEY_PATTERN.finditer(text):
            value = cls._parse_money(
                match.group(0)
            )

            if value is not None:
                values.append(value)

        return values

    @classmethod
    def _last_money_value(
        cls,
        text: str,
    ) -> float | None:
        """Return the last money value in text."""

        values = cls._money_values(text)

        if not values:
            return None

        return values[-1]

    @staticmethod
    def _parse_money(
        raw: str,
    ) -> float | None:
        """Convert money text into float."""

        cleaned = raw.strip()

        cleaned = re.sub(
            r"[$€£₹\s]",
            "",
            cleaned,
        )

        try:
            if "," in cleaned:
                cleaned = cleaned.replace(
                    ",",
                    "",
                )

            return float(cleaned)

        except ValueError:
            return None

    @staticmethod
    def _block_confidence(
        *,
        block: list[OCRLine],
        name: str,
        quantity: float | None,
        unit_price: float | None,
        total_price: float | None,
    ) -> float:
        """Calculate confidence for a multi-line item."""

        if not block:
            return 0.0

        line_confidence = sum(
            line.confidence
            for line in block
        ) / len(block)

        score = line_confidence

        if len(name) >= 3:
            score += 0.04

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

    @staticmethod
    def _deduplicate_items(
        items: list[ReceiptItem],
    ) -> list[ReceiptItem]:
        """Remove duplicate item detections."""

        result: list[ReceiptItem] = []

        seen: set[
            tuple[
                str,
                float | None,
                float | None,
            ]
        ] = set()

        for item in items:
            key = (
                item.name.lower().strip(),
                item.quantity,
                item.total_price,
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_reconciliation_warnings(
        *,
        items: list[ReceiptItem],
        subtotal: float | None,
        discount: float | None,
        tax: float | None,
        total: float | None,
    ) -> list[str]:
        """Check financial reconciliation."""

        warnings: list[str] = []

        if not items:
            return warnings

        item_sum = round(
            sum(
                item.total_price
                for item in items
                if item.total_price is not None
            ),
            2,
        )

        # ----------------------------------------------------------
        # Items -> subtotal
        # ----------------------------------------------------------
        #
        # A receipt may apply a discount between item total
        # and subtotal:
        #
        # item_sum - discount = subtotal
        #
        # Example:
        # 5.68 - 0.57 = 5.11
        #

        if subtotal is not None:

            direct_difference = abs(
                item_sum - subtotal
            )

            discounted_difference = float("inf")

            if discount is not None:
                discounted_difference = abs(
                    (item_sum - discount) - subtotal
                )

            if min(
                direct_difference,
                discounted_difference,
            ) > 0.05:

                warnings.append(
                    "Extracted item totals do not reconcile "
                    "with the detected subtotal."
                )

        # ----------------------------------------------------------
        # Subtotal -> total
        # ----------------------------------------------------------

        if total is not None and subtotal is not None:

            candidates = [
                subtotal,
            ]

            if tax is not None:
                candidates.append(
                    subtotal + tax
                )

            if discount is not None:
                candidates.append(
                    subtotal - discount
                )

                if tax is not None:
                    candidates.append(
                        subtotal - discount + tax
                    )

            closest = min(
                candidates,
                key=lambda value: abs(
                    value - total
                ),
            )

            if abs(closest - total) > 0.10:
                warnings.append(
                    "Subtotal, discount, tax and total "
                    "do not reconcile."
                )

        return warnings

    # ------------------------------------------------------------------
    # Warnings
    # ------------------------------------------------------------------

    @staticmethod
    def _build_warnings(
        *,
        merchant: str | None,
        total: float | None,
        items: list[ReceiptItem],
        ocr_confidence: float,
    ) -> list[str]:
        """Generate extraction warnings."""

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

        score -= 0.03 * len(warnings)

        return max(
            0.0,
            min(1.0, score),
        )