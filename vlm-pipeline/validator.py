"""Business-rule validation for a normalized Invoice.

Duplicate detection is intentionally NOT performed here: it must happen
atomically with the Excel write to avoid a race between concurrent workers
(see excel_writer.ExcelWriter.append_invoice), so it lives there instead.
This module only checks that field formats and numeric relationships are
sane.
"""

import hashlib
import logging
from datetime import datetime
from typing import Optional, Tuple

from config import VALIDATION_ABS_TOLERANCE, VALIDATION_REL_TOLERANCE
from parser import Invoice

logger = logging.getLogger(__name__)


def validate_date(date_str: str) -> bool:
    """Require a canonical ISO 'YYYY-MM-DD' date (parser.normalize_date output)."""
    if not date_str:
        return False
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def validate_time(time_str: str) -> bool:
    """Require a canonical 24-hour 'HH:MM:SS' time (parser.normalize_time output)."""
    if not time_str:
        return False
    try:
        datetime.strptime(time_str, "%H:%M:%S")
        return True
    except ValueError:
        return False


def validate_number(value: float) -> bool:
    """Numeric fields must be non-negative (zero is allowed)."""
    return value >= 0


def validate_invoice_consistency(inv: Invoice) -> bool:
    """Check subtotal - discount + vat == grand_total within tolerance.

    Tolerance combines a small absolute floor with a proportion of the
    grand total, so minor OCR/rounding noise on larger bills does not
    trigger a false failure while still catching real mismatches.
    """
    expected = inv.subtotal - inv.discount + inv.vat
    diff = abs(expected - inv.grand_total)
    tolerance = max(VALIDATION_ABS_TOLERANCE, VALIDATION_REL_TOLERANCE * inv.grand_total)
    if diff > tolerance:
        logger.warning(
            f"Total mismatch: subtotal({inv.subtotal}) - discount({inv.discount}) "
            f"+ vat({inv.vat}) = {expected}, grand_total={inv.grand_total}, "
            f"diff={diff:.2f} > tolerance={tolerance:.2f}"
        )
        return False
    return True


def compute_invoice_hash(inv: Invoice) -> str:
    """SHA256 fallback key (date + time + grand_total) for duplicate detection
    when no invoice_number is present or two invoices share one."""
    raw = f"{inv.date}|{inv.time}|{inv.grand_total:.2f}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_invoice(inv: Invoice) -> Tuple[bool, Optional[str]]:
    """Full field/consistency validation (no duplicate check).

    Returns (is_valid, error_message_or_None).
    """
    if not validate_date(inv.date):
        return False, f"Invalid date format: {inv.date!r}"
    if not validate_time(inv.time):
        return False, f"Invalid time format: {inv.time!r}"

    for name, value in (
        ("subtotal", inv.subtotal),
        ("vat", inv.vat),
        ("discount", inv.discount),
        ("grand_total", inv.grand_total),
    ):
        if not validate_number(value):
            return False, f"Negative {name}: {value}"

    if inv.grand_total <= 0:
        return False, "grand_total must be greater than zero"

    for idx, item in enumerate(inv.items):
        if item.qty < 0 or item.unit_price < 0 or item.total < 0:
            return False, f"Negative values in item {idx} ({item.name!r})"
        expected_total = item.qty * item.unit_price
        if abs(item.total - expected_total) > max(0.5, 0.02 * max(item.total, expected_total)):
            logger.warning(
                f"Item total mismatch: {item.name!r} qty={item.qty} * "
                f"unit_price={item.unit_price} = {expected_total}, but total={item.total}"
            )

    if not validate_invoice_consistency(inv):
        return False, "Total amount inconsistency (subtotal - discount + vat != grand_total)"

    return True, None
