"""Validation utilities for structured receipt data."""

from __future__ import annotations

from dataclasses import dataclass, field

from receipt_ocr.schemas import ReceiptData


@dataclass
class ValidationIssue:
    """Represents a validation problem."""

    field: str
    message: str
    severity: str = "warning"

    def to_dict(self) -> dict[str, str]:
        return {
            "field": self.field,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass
class ValidationResult:
    """Result of receipt validation."""

    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [
            issue
            for issue in self.issues
            if issue.severity == "error"
        ]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [
            issue
            for issue in self.issues
            if issue.severity == "warning"
        ]

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],
        }


class ReceiptValidator:
    """Validate extracted receipt information."""

    VALID_PAYMENT_METHODS = {
        "CASH",
        "CARD",
        "CREDIT",
        "DEBIT",
        "UPI",
        "ONLINE",
        "PAYPAL",
    }

    def validate(self, receipt: ReceiptData) -> ValidationResult:
        """Validate a structured receipt."""

        issues: list[ValidationIssue] = []

        self._validate_items(receipt, issues)
        self._validate_financial_fields(receipt, issues)
        self._validate_payment_method(receipt, issues)
        self._validate_required_fields(receipt, issues)
        self._validate_totals(receipt, issues)

        has_errors = any(
            issue.severity == "error"
            for issue in issues
        )

        return ValidationResult(
            is_valid=not has_errors,
            issues=issues,
        )

    def _validate_items(
        self,
        receipt: ReceiptData,
        issues: list[ValidationIssue],
    ) -> None:
        """Validate receipt items."""

        if not receipt.items:
            issues.append(
                ValidationIssue(
                    field="items",
                    message="No receipt items were extracted.",
                    severity="error",
                )
            )
            return

        for index, item in enumerate(receipt.items, start=1):

            if not item.name.strip():
                issues.append(
                    ValidationIssue(
                        field=f"items[{index}].name",
                        message="Item name is empty.",
                        severity="error",
                    )
                )

            if item.quantity is not None and item.quantity <= 0:
                issues.append(
                    ValidationIssue(
                        field=f"items[{index}].quantity",
                        message="Quantity must be greater than zero.",
                        severity="error",
                    )
                )

            if item.unit_price is not None and item.unit_price < 0:
                issues.append(
                    ValidationIssue(
                        field=f"items[{index}].unit_price",
                        message="Unit price cannot be negative.",
                        severity="error",
                    )
                )

            if item.total_price is not None and item.total_price < 0:
                issues.append(
                    ValidationIssue(
                        field=f"items[{index}].total_price",
                        message="Total price cannot be negative.",
                        severity="error",
                    )
                )

            if item.confidence < 0 or item.confidence > 1:
                issues.append(
                    ValidationIssue(
                        field=f"items[{index}].confidence",
                        message="Confidence must be between 0 and 1.",
                        severity="error",
                    )
                )

    def _validate_financial_fields(
        self,
        receipt: ReceiptData,
        issues: list[ValidationIssue],
    ) -> None:
        """Validate financial values."""

        fields = {
            "subtotal": receipt.subtotal,
            "discount": receipt.discount,
            "tax": receipt.tax,
            "total": receipt.total,
        }

        for field_name, value in fields.items():
            if value is not None and value < 0:
                issues.append(
                    ValidationIssue(
                        field=field_name,
                        message="Financial value cannot be negative.",
                        severity="error",
                    )
                )

    def _validate_payment_method(
        self,
        receipt: ReceiptData,
        issues: list[ValidationIssue],
    ) -> None:
        """Validate payment method."""

        if not receipt.payment_method:
            issues.append(
                ValidationIssue(
                    field="payment_method",
                    message="Payment method could not be identified.",
                    severity="warning",
                )
            )
            return

        method = receipt.payment_method.upper().strip()

        if method not in self.VALID_PAYMENT_METHODS:
            issues.append(
                ValidationIssue(
                    field="payment_method",
                    message=f"Unknown payment method: {receipt.payment_method}",
                    severity="warning",
                )
            )

    def _validate_required_fields(
        self,
        receipt: ReceiptData,
        issues: list[ValidationIssue],
    ) -> None:
        """Check important missing fields."""

        if not receipt.merchant:
            issues.append(
                ValidationIssue(
                    field="merchant",
                    message="Merchant could not be identified.",
                    severity="warning",
                )
            )

        if not receipt.receipt_date:
            issues.append(
                ValidationIssue(
                    field="receipt_date",
                    message="Receipt date could not be identified.",
                    severity="warning",
                )
            )

        if receipt.total is None:
            issues.append(
                ValidationIssue(
                    field="total",
                    message="Receipt total could not be identified.",
                    severity="warning",
                )
            )

    def _validate_totals(
        self,
        receipt: ReceiptData,
        issues: list[ValidationIssue],
    ) -> None:
        """Check consistency between item totals and receipt total."""

        item_total = sum(
            item.total_price
            for item in receipt.items
            if item.total_price is not None
        )

        if receipt.total is not None and item_total > 0:

            discount = receipt.discount or 0.0
            tax = receipt.tax or 0.0

            expected_total = item_total - discount + tax

            difference = abs(
                expected_total - receipt.total
            )

            if difference > 0.10:
                issues.append(
                    ValidationIssue(
                        field="total",
                        message=(
                            f"Total mismatch. "
                            f"Expected approximately {expected_total:.2f}, "
                            f"but extracted {receipt.total:.2f}."
                        ),
                        severity="warning",
                    )
                )