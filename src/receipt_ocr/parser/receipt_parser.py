"""Robust rule-based receipt parser for PaddleOCR output."""

from __future__ import annotations

import logging
import re
from typing import Iterable

from receipt_ocr.ocr.models import OCRLine, OCRResult
from receipt_ocr.schemas import ReceiptData, ReceiptItem

logger = logging.getLogger(__name__)


class ReceiptParser:
    """Extract structured receipt information from OCR output."""

    # ------------------------------------------------------------------
    # Patterns
    # ------------------------------------------------------------------

    TOTAL_LABEL_RE = re.compile(
        r"\b(?:"
        r"grand\s*total|"
        r"total\s*(?:due|amount|purchase|sale)?|"
        r"amount\s*due|"
        r"balance\s*due|"
        r"net\s*total|"
        r"order\s*total|"
        r"invoice\s*total|"
        r"payable|"
        r"amount\s*payable"
        r")\b",
        re.IGNORECASE,
    )

    SUBTOTAL_LABEL_RE = re.compile(
        r"\b(?:sub\s*total|subtotal|"
        r"merchandise\s*subtotal|"
        r"items?\s*subtotal)\b",
        re.IGNORECASE,
    )

    DISCOUNT_LABEL_RE = re.compile(
        r"\b(?:discount|discounts|"
        r"savings|save|coupon|promotion|promo)\b",
        re.IGNORECASE,
    )

    TAX_LABEL_RE = re.compile(
        r"\b(?:sales\s*tax|tax|gst|cgst|sgst|igst|vat|"
        r"service\s*tax|taxes)\b",
        re.IGNORECASE,
    )

    RECEIPT_NUMBER_PATTERNS = (
        r"\b(?:receipt|receipt\s*no|receipt\s*number)"
        r"\s*(?:no|number|#)?\s*[:#\-]?\s*([A-Z0-9][A-Z0-9\-\/]{2,})",

        r"\b(?:transaction|transaction\s*no|transaction\s*number|"
        r"trans|txn|txn\s*no)\s*(?:#|no|number)?\s*[:\-]?\s*"
        r"([A-Z0-9][A-Z0-9\-\/]{2,})",

        r"\b(?:order|order\s*no|order\s*number)"
        r"\s*(?:#|no|number)?\s*[:\-]?\s*"
        r"([A-Z0-9][A-Z0-9\-\/]{2,})",

        r"\b(?:invoice|invoice\s*no|invoice\s*number)"
        r"\s*(?:#|no|number)?\s*[:\-]?\s*"
        r"([A-Z0-9][A-Z0-9\-\/]{2,})",

        r"\b(?:ref|reference|reference\s*no)"
        r"\s*(?:#|no|number)?\s*[:\-]?\s*"
        r"([A-Z0-9][A-Z0-9\-\/]{2,})",

        r"\b(?:op|op#)\s*[:#\-]?\s*"
        r"([A-Z0-9][A-Z0-9\-\/]{2,})",

        r"\b(?:tr|st|te)\s*#?\s*[:\-]?\s*"
        r"([A-Z0-9][A-Z0-9\-\/]{2,})",
    )

    PAYMENT_PATTERNS = (
        ("GOOGLE PAY", r"\bgoogle\s*pay\b"),
        ("PHONEPE", r"\bphone\s*pe\b"),
        ("PAYTM", r"\bpaytm\b"),
        ("BHIM UPI", r"\bbhim\b"),
        ("UPI", r"\bupi\b"),
        ("MASTERCARD", r"\bmaster\s*card\b|\bmastercard\b"),
        ("VISA", r"\bvisa\b"),
        ("AMEX", r"\bamerican\s*express\b|\bamex\b"),
        ("CREDIT CARD", r"\bcredit\s*card\b|\bcredit\b"),
        ("DEBIT CARD", r"\bdebit\s*card\b|\bdebit\b"),
        ("CASH", r"\bcash\b"),
        ("CARD", r"\bcard\b"),
    )

    DATE_PATTERN = re.compile(
        r"\b("
        r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"
        r"|"
        r"\d{1,2}\s+"
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
        r"(?:\s+\d{2,4})?"
        r"|"
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
        r"\s+\d{1,2}(?:\s+\d{2,4})?"
        r")\b",
        re.IGNORECASE,
    )

    MONEY_PATTERN = re.compile(
        r"(?<![\w.])"
        r"(?:[$€£₹]\s*)?"
        r"\d{1,3}(?:,\d{3})+\.\d{2}"
        r"(?!\w)"
        r"|"
        r"(?<![\w.])"
        r"(?:[$€£₹]\s*)?"
        r"\d+\.\d{2}"
        r"(?!\w)",
        re.UNICODE,
    )

    WEIGHT_PATTERN = re.compile(
        r"\b(\d+(?:\.\d+)?)\s*(lb|lbs|kg|g|oz)\b",
        re.IGNORECASE,
    )

    QUANTITY_PATTERN = re.compile(
        r"\b(?:qty|quantity)\s*[:=]?\s*(\d+(?:\.\d+)?)\b",
        re.IGNORECASE,
    )

    MULTIPLICATION_PATTERN = re.compile(
        r"\b"
        r"(\d+(?:\.\d+)?)"
        r"\s*[x×]"
        r"\s*"
        r"(?:[$€£₹]\s*)?"
        r"(\d+(?:,\d{3})*\.\d{2})"
        r"\b",
        re.IGNORECASE,
    )

    ITEM_END_RE = re.compile(
        r"^(?:"
        r"sub\s*total|subtotal|"
        r"discount|discounts|savings|coupon|"
        r"tax|sales\s*tax|gst|cgst|sgst|igst|vat|"
        r"grand\s*total|total|amount\s*due|balance\s*due|"
        r"cash\s*tend|cash\s*given|change\s*due|"
        r"payment|tender|"
        r"items?\s*sold|"
        r"thank\s*you|"
        r"visit\s*again"
        r")\b",
        re.IGNORECASE,
    )

    METADATA_WORDS = (
        "wal-mart",
        "walmart",
        "supercenter",
        "manager",
        "hours",
        "phone",
        "telephone",
        "address",
        "thank you",
        "shop at",
        "customer",
        "www.",
        "http",
    )

    # ------------------------------------------------------------------
    # Main parse
    # ------------------------------------------------------------------

    def parse(self, ocr_result: OCRResult) -> ReceiptData:
        lines = list(ocr_result.lines)

        if not lines:
            return ReceiptData(
                raw_text=ocr_result.full_text,
                warnings=["No OCR lines were detected."],
            )

        merchant = self._extract_merchant(lines)
        receipt_date = self._extract_date(lines)
        receipt_number = self._extract_receipt_number(lines)

        subtotal = self._extract_financial_field(
            lines,
            self.SUBTOTAL_LABEL_RE,
        )

        discount = self._extract_financial_field(
            lines,
            self.DISCOUNT_LABEL_RE,
            prefer_positive=True,
        )

        tax = self._extract_financial_field(
            lines,
            self.TAX_LABEL_RE,
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
            receipt_date=receipt_date,
            receipt_number=receipt_number,
            subtotal=subtotal,
            tax=tax,
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
            receipt_date=receipt_date,
            receipt_number=receipt_number,
            subtotal=subtotal,
            tax=tax,
            total=total,
            payment_method=payment_method,
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

    def _extract_merchant(self, lines: list[OCRLine]) -> str | None:
        """Find the most likely merchant from the receipt header."""

        candidates: list[tuple[float, str]] = []

        for index, line in enumerate(lines[:15]):
            text = self._normalize_text(line.text)

            if not text:
                continue

            lower = text.lower()

            if self._is_metadata_line(lower):
                continue

            if self.DATE_PATTERN.search(text):
                continue

            if self._looks_like_identifier(text):
                continue

            if self.MONEY_PATTERN.search(text):
                continue

            alpha_count = sum(c.isalpha() for c in text)
            digit_count = sum(c.isdigit() for c in text)

            if alpha_count < 3:
                continue

            if digit_count > alpha_count:
                continue

            if len(text) > 60:
                continue

            score = float(line.confidence)

            # Strong preference for upper receipt lines.
            score += max(0.0, 0.20 - index * 0.015)

            # Merchant names are often uppercase.
            if line.text.strip().isupper():
                score += 0.08

            if len(text.split()) >= 2:
                score += 0.04

            candidates.append((score, text))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    # ------------------------------------------------------------------
    # Date
    # ------------------------------------------------------------------

    def _extract_date(self, lines: Iterable[OCRLine]) -> str | None:
        """Extract the most likely receipt date."""

        candidates: list[tuple[int, float, str]] = []

        for index, line in enumerate(lines):
            match = self.DATE_PATTERN.search(line.text)

            if not match:
                continue

            value = match.group(1).strip()

            # Prefer dates near the top/middle receipt metadata area.
            priority = 0

            if index < 15:
                priority += 3

            lower = line.text.lower()

            if any(
                word in lower
                for word in ("date", "invoice", "receipt", "transaction")
            ):
                priority += 2

            candidates.append(
                (
                    priority,
                    line.confidence,
                    value,
                )
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )

        return candidates[0][2]

    # ------------------------------------------------------------------
    # Receipt number
    # ------------------------------------------------------------------

    def _extract_receipt_number(
        self,
        lines: Iterable[OCRLine],
    ) -> str | None:
        """Extract receipt/transaction/order/invoice identifier."""

        for line in lines:
            text = line.text.strip()

            for pattern in self.RECEIPT_NUMBER_PATTERNS:
                match = re.search(
                    pattern,
                    text,
                    re.IGNORECASE,
                )

                if not match:
                    continue

                value = match.group(1).strip(" :-#")

                if self._valid_identifier(value):
                    return value

        # Handle split OCR:
        #
        # Receipt No
        # 123456
        #
        lines_list = list(lines)

        for index, line in enumerate(lines_list[:-1]):
            label = line.text.strip().lower()

            if not re.search(
                r"\b(receipt|transaction|txn|order|invoice|reference|ref)\b",
                label,
            ):
                continue

            next_text = lines_list[index + 1].text.strip()

            cleaned = next_text.strip(" :#-")

            if self._valid_identifier(cleaned):
                return cleaned

        return None

    @staticmethod
    def _valid_identifier(value: str) -> bool:
        if not value:
            return False

        if len(value) < 3 or len(value) > 40:
            return False

        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9\-\/]*",
            value,
        ):
            return False

        return True

    # ------------------------------------------------------------------
    # Financial fields
    # ------------------------------------------------------------------

    def _extract_total(
        self,
        lines: list[OCRLine],
    ) -> float | None:
        """Extract the most likely final total."""

        candidates: list[tuple[int, int, float]] = []

        for index, line in enumerate(lines):
            text = line.text.strip()
            lower = text.lower()

            if "cash tend" in lower:
                continue

            if "change due" in lower:
                continue

            if not self.TOTAL_LABEL_RE.search(lower):
                continue

            values = self._money_values(text)

            if values:
                candidates.append(
                    (
                        3,
                        index,
                        values[-1],
                    )
                )
                continue

            nearby = self._find_amount_nearby(
                lines,
                index,
                window=2,
            )

            if nearby is not None:
                candidates.append(
                    (
                        2,
                        index,
                        nearby,
                    )
                )

        if candidates:
            # Prefer the candidate closest to the bottom of the receipt.
            candidates.sort(
                key=lambda item: (item[0], item[1]),
                reverse=True,
            )

            return candidates[0][2]

        # Final fallback: a line containing only "TOTAL".
        for index, line in enumerate(lines):
            if re.fullmatch(
                r"\s*(?:grand\s*)?total\s*:?\s*",
                line.text,
                re.IGNORECASE,
            ):
                amount = self._find_amount_nearby(
                    lines,
                    index,
                    window=3,
                )

                if amount is not None:
                    return amount

        return None

    def _extract_financial_field(
        self,
        lines: Iterable[OCRLine],
        label_pattern: re.Pattern[str],
        *,
        prefer_positive: bool = False,
    ) -> float | None:
        """Extract amount associated with a financial label."""

        lines_list = list(lines)

        candidates: list[tuple[int, int, float]] = []

        for index, line in enumerate(lines_list):
            text = line.text.strip()

            if not label_pattern.search(text):
                continue

            values = self._money_values(text)

            if values:
                value = values[-1]

                if prefer_positive:
                    value = abs(value)

                candidates.append(
                    (
                        3,
                        index,
                        value,
                    )
                )

                continue

            # Search up to two lines after label.
            for offset in (1, 2):
                next_index = index + offset

                if next_index >= len(lines_list):
                    break

                next_text = lines_list[next_index].text.strip()

                if self._is_financial_label(next_text):
                    break

                next_values = self._money_values(next_text)

                if next_values:
                    value = next_values[-1]

                    if prefer_positive:
                        value = abs(value)

                    candidates.append(
                        (
                            2 if offset == 1 else 1,
                            index,
                            value,
                        )
                    )

                    break

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (item[0], -item[1]),
            reverse=True,
        )

        return candidates[0][2]

    def _find_amount_nearby(
        self,
        lines: list[OCRLine],
        index: int,
        window: int = 2,
    ) -> float | None:
        for offset in range(1, window + 1):
            next_index = index + offset

            if next_index >= len(lines):
                break

            text = lines[next_index].text.strip()

            values = self._money_values(text)

            if values:
                return values[-1]

            if self._is_financial_label(text):
                break

        return None

    def _is_financial_label(self, text: str) -> bool:
        return bool(
            self.SUBTOTAL_LABEL_RE.search(text)
            or self.DISCOUNT_LABEL_RE.search(text)
            or self.TAX_LABEL_RE.search(text)
            or self.TOTAL_LABEL_RE.search(text)
        )

    # ------------------------------------------------------------------
    # Payment
    # ------------------------------------------------------------------

    def _extract_payment_method(
        self,
        lines: Iterable[OCRLine],
    ) -> str | None:
        text = "\n".join(
            line.text.lower()
            for line in lines
        )

        for name, pattern in self.PAYMENT_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return name

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

        # First attempt: structured/multiline parsing.
        items = self._extract_multiline_items(lines)

        if items:
            return items

        # Second attempt: conventional single-line receipts.
        return self._extract_single_line_items(lines)

    def _extract_multiline_items(
        self,
        lines: list[OCRLine],
    ) -> list[ReceiptItem]:

        start_index = self._find_item_section_start(lines)

        if start_index is None:
            return []

        items: list[ReceiptItem] = []
        block: list[OCRLine] = []

        for line in lines[start_index:]:
            text = line.text.strip()

            if not text:
                continue

            lower = text.lower()

            if self._is_item_section_end(lower):
                if block:
                    item = self._parse_item_block(block)

                    if item:
                        items.append(item)

                    block = []

                break

            if self._looks_like_item_name(text):
                if block:
                    item = self._parse_item_block(block)

                    if item:
                        items.append(item)

                block = [line]
            else:
                if block:
                    block.append(line)

        if block:
            item = self._parse_item_block(block)

            if item:
                items.append(item)

        return self._deduplicate_items(items)

    def _find_item_section_start(
        self,
        lines: list[OCRLine],
    ) -> int | None:

        # Look for explicit item headers first.
        for index, line in enumerate(lines):
            lower = line.text.lower().strip()

            if re.search(
                r"\b(item|items|description|product|products|"
                r"qty|quantity|price)\b",
                lower,
            ):
                # Start after header.
                if index + 1 < len(lines):
                    return index + 1

        # General fallback.
        for index, line in enumerate(lines):
            if index < 3:
                continue

            if self._looks_like_item_name(line.text):
                return index

        return None

    def _is_item_section_end(
        self,
        text: str,
    ) -> bool:
        return bool(
            self.ITEM_END_RE.search(
                text.strip()
            )
        )

    def _looks_like_item_name(
        self,
        text: str,
    ) -> bool:

        normalized = self._normalize_text(text)

        if not normalized:
            return False

        lower = normalized.lower()

        if self._is_item_section_end(lower):
            return False

        if self._looks_like_identifier(normalized):
            return False

        if self.DATE_PATTERN.search(normalized):
            return False

        # Pure monetary line.
        if self.MONEY_PATTERN.fullmatch(normalized):
            return False

        if self.WEIGHT_PATTERN.fullmatch(normalized):
            return False

        if re.fullmatch(
            r"[\d\s./#*_\-:+]+",
            normalized,
        ):
            return False

        if any(
            word in lower
            for word in self.METADATA_WORDS
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

        if digit_count > alpha_count * 3:
            return False

        if len(normalized) > 70:
            return False

        # Don't treat obvious financial lines as products.
        if self._is_financial_label(normalized):
            return False

        return True

    # ------------------------------------------------------------------
    # Item parsing
    # ------------------------------------------------------------------

    def _parse_item_block(
        self,
        block: list[OCRLine],
    ) -> ReceiptItem | None:

        if not block:
            return None

        texts = [
            line.text.strip()
            for line in block
            if line.text.strip()
        ]

        if not texts:
            return None

        # Find the best textual line for product name.
        name = ""

        for text in texts:
            candidate = self._clean_product_name(text)

            if candidate:
                name = candidate
                break

        if not name:
            return None

        full_block = " ".join(texts)

        quantity = self._extract_quantity(full_block)

        weight_match = self.WEIGHT_PATTERN.search(
            full_block
        )

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

        multiplication = self.MULTIPLICATION_PATTERN.search(
            full_block
        )

        if multiplication:
            try:
                quantity = float(
                    multiplication.group(1)
                )

                unit_price = self._parse_money(
                    multiplication.group(2)
                )

            except (TypeError, ValueError):
                pass

            total_price = money_values[-1]

        elif weight_match:
            quantity = float(
                weight_match.group(1)
            )

            unit_price = self._extract_weight_unit_price(
                full_block
            )

            if unit_price is not None:
                # Usually final monetary value is extended total.
                different_values = [
                    value
                    for value in money_values
                    if abs(value - unit_price) > 0.001
                ]

                if different_values:
                    total_price = different_values[-1]

            if total_price is None:
                total_price = money_values[-1]

        else:
            # Most conventional receipt layouts:
            #
            # PRODUCT       12.99
            # PRODUCT   2 x 4.99
            #
            total_price = money_values[-1]

        if total_price is None:
            return None

        if quantity is None or quantity <= 0:
            quantity = 1.0

        if unit_price is None:
            unit_price = round(
                total_price / quantity,
                2,
            )

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

    def _extract_single_line_items(
        self,
        lines: list[OCRLine],
    ) -> list[ReceiptItem]:

        items: list[ReceiptItem] = []

        for index, line in enumerate(lines):
            text = line.text.strip()

            if not text:
                continue

            if index < 3:
                continue

            if self._is_item_section_end(text.lower()):
                continue

            if self._is_financial_label(text):
                continue

            if self._looks_like_identifier(text):
                continue

            if not self.MONEY_PATTERN.search(text):
                continue

            name = self._clean_product_name(text)

            if not name:
                continue

            values = self._money_values(text)

            if not values:
                continue

            total_price = values[-1]

            quantity = self._extract_quantity(text)

            if quantity is None or quantity <= 0:
                quantity = 1.0

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

        cleaned = text.strip()

        # Remove quantity/multiplication expressions.
        cleaned = self.MULTIPLICATION_PATTERN.sub(
            "",
            cleaned,
        )

        cleaned = self.QUANTITY_PATTERN.sub(
            "",
            cleaned,
        )

        # Remove weight.
        cleaned = self.WEIGHT_PATTERN.sub(
            "",
            cleaned,
        )

        # Remove money.
        cleaned = self.MONEY_PATTERN.sub(
            "",
            cleaned,
        )

        # Common receipt markers.
        cleaned = re.sub(
            r"\b(?:sku|item|upc)\s*[:#-]?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
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

        if not cleaned:
            return ""

        if self._looks_like_identifier(cleaned):
            return ""

        alpha_count = sum(
            char.isalpha()
            for char in cleaned
        )

        if alpha_count < 2:
            return ""

        return cleaned

    def _extract_quantity(
        self,
        text: str,
    ) -> float | None:

        match = self.QUANTITY_PATTERN.search(text)

        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

        match = self.WEIGHT_PATTERN.search(text)

        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

        match = self.MULTIPLICATION_PATTERN.search(text)

        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

        return None

    @staticmethod
    def _looks_like_identifier(
        text: str,
    ) -> bool:

        normalized = re.sub(
            r"\s+",
            "",
            text,
        )

        if len(normalized) >= 8 and re.fullmatch(
            r"[A-Z0-9\-\/]+",
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

    # ------------------------------------------------------------------
    # Money
    # ------------------------------------------------------------------

    @classmethod
    def _money_values(
        cls,
        text: str,
    ) -> list[float]:

        values: list[float] = []

        for match in cls.MONEY_PATTERN.finditer(text):
            value = cls._parse_money(
                match.group(0)
            )

            if value is not None:
                values.append(value)

        return values

    @classmethod
    def _parse_money(
        cls,
        raw: str,
    ) -> float | None:

        cleaned = raw.strip()

        cleaned = re.sub(
            r"[$€£₹\s]",
            "",
            cleaned,
        )

        cleaned = cleaned.replace(
            ",",
            "",
        )

        try:
            return round(
                float(cleaned),
                2,
            )
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Weighted products
    # ------------------------------------------------------------------

    def _extract_weight_unit_price(
        self,
        text: str,
    ) -> float | None:

        patterns = (
            r"(?:1\s*)?(?:lb|lbs|kg|g|oz)"
            r"\s*/\s*"
            r"(?:[$€£₹]\s*)?"
            r"(\d+(?:,\d{3})*\.\d{2})",

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
                return self._parse_money(
                    match.group(1)
                )

        return None

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    @staticmethod
    def _block_confidence(
        *,
        block: list[OCRLine],
        name: str,
        quantity: float | None,
        unit_price: float | None,
        total_price: float | None,
    ) -> float:

        if not block:
            return 0.0

        line_confidence = (
            sum(
                max(
                    0.0,
                    min(1.0, line.confidence),
                )
                for line in block
            )
            / len(block)
        )

        score = (
            line_confidence * 0.70
        )

        if len(name) >= 3:
            score += 0.08

        if quantity is not None:
            score += 0.06

        if unit_price is not None:
            score += 0.05

        if total_price is not None:
            score += 0.06

        return max(
            0.0,
            min(1.0, score),
        )

    @staticmethod
    def _calculate_confidence(
        *,
        ocr_confidence: float,
        merchant: str | None,
        receipt_date: str | None,
        receipt_number: str | None,
        subtotal: float | None,
        tax: float | None,
        total: float | None,
        payment_method: str | None,
        items: list[ReceiptItem],
        warnings: list[str],
    ) -> float:

        # Do NOT allow the score to automatically become 1.0.
        score = (
            max(
                0.0,
                min(1.0, ocr_confidence),
            )
            * 0.45
        )

        if merchant:
            score += 0.10

        if receipt_date:
            score += 0.08

        if receipt_number:
            score += 0.05

        if subtotal is not None:
            score += 0.07

        if tax is not None:
            score += 0.05

        if total is not None:
            score += 0.10

        if payment_method:
            score += 0.04

        if items:
            score += 0.06

        # Penalize warnings, but don't destroy confidence.
        score -= min(
            0.25,
            len(warnings) * 0.025,
        )

        return round(
            max(
                0.0,
                min(1.0, score),
            ),
            3,
        )

    # ------------------------------------------------------------------
    # Warnings
    # ------------------------------------------------------------------

    @staticmethod
    def _build_warnings(
        *,
        merchant: str | None,
        receipt_date: str | None,
        receipt_number: str | None,
        subtotal: float | None,
        tax: float | None,
        total: float | None,
        items: list[ReceiptItem],
        ocr_confidence: float,
    ) -> list[str]:

        warnings: list[str] = []

        if merchant is None:
            warnings.append(
                "Merchant could not be confidently identified."
            )

        if receipt_date is None:
            warnings.append(
                "Receipt date could not be identified."
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
                "Overall OCR confidence is low; manual review recommended."
            )
        elif ocr_confidence < 0.70:
            warnings.append(
                "OCR confidence is moderate; extracted fields should be reviewed."
            )

        return warnings

    @staticmethod
    def _build_reconciliation_warnings(
        *,
        items: list[ReceiptItem],
        subtotal: float | None,
        discount: float | None,
        tax: float | None,
        total: float | None,
    ) -> list[str]:

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

        if subtotal is not None:

            possibilities = [
                abs(item_sum - subtotal)
            ]

            if discount is not None:
                possibilities.append(
                    abs(
                        item_sum
                        - abs(discount)
                        - subtotal
                    )
                )

            if min(possibilities) > 0.10:
                warnings.append(
                    "Extracted item totals do not reconcile with the detected subtotal."
                )

        if subtotal is not None and total is not None:

            candidates = [
                subtotal
            ]

            if tax is not None:
                candidates.append(
                    subtotal + tax
                )

            if discount is not None:
                candidates.append(
                    subtotal - abs(discount)
                )

                if tax is not None:
                    candidates.append(
                        subtotal
                        - abs(discount)
                        + tax
                    )

            closest = min(
                candidates,
                key=lambda value: abs(
                    value - total
                ),
            )

            if abs(closest - total) > 0.15:
                warnings.append(
                    "Subtotal, discount, tax and total do not reconcile."
                )

        return warnings

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:

        normalized = re.sub(
            r"[^a-zA-Z0-9\s&'./#:@$€£₹x×\-]",
            "",
            text,
        )

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        )

        return normalized.strip()

    @classmethod
    def _is_metadata_line(
        cls,
        lower: str,
    ) -> bool:

        return any(
            word in lower
            for word in cls.METADATA_WORDS
        )

    @staticmethod
    def _deduplicate_items(
        items: list[ReceiptItem],
    ) -> list[ReceiptItem]:

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