"""Parse raw model JSON into typed, normalized Invoice/Item dataclasses.

Normalization happens here (not in validator.py) so that validator.py can
stay a pure pass/fail checker against canonical formats: ISO date, 24-hour
time, and plain floats.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

_DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y")


def _normalize_digits(text: str) -> str:
    return text.translate(_ARABIC_INDIC_DIGITS)


def normalize_number(value: Any) -> float:
    """Coerce a raw model value (number or messy string) into a float.

    Strips currency symbols/commas, converts Arabic-Indic digits, and
    defaults to 0.0 for anything unparseable.
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = _normalize_digits(str(value)).strip()
    text = re.sub(r"[^0-9.\-]", "", text)
    if not text or text in ("-", "."):
        return 0.0
    try:
        return float(text)
    except ValueError:
        logger.warning(f"Could not parse numeric value: {value!r}")
        return 0.0


def normalize_date(value: Any) -> str:
    """Best-effort conversion to ISO 'YYYY-MM-DD'. Returns the cleaned
    original string if no known format matches (validator will then flag
    it as invalid, which is the correct outcome for unparseable data).
    """
    text = _normalize_digits(str(value or "")).strip()
    if not text:
        return ""
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return text


def normalize_time(value: Any) -> str:
    """Best-effort conversion to 24-hour 'HH:MM:SS', tolerating missing
    leading zeros (e.g. '21:1:52') and AM/PM suffixes.
    """
    text = _normalize_digits(str(value or "")).strip()
    if not text:
        return ""

    am_pm_match = re.search(r"\b(am|pm)\b", text, re.IGNORECASE)
    am_pm = am_pm_match.group(1).lower() if am_pm_match else None
    text_no_suffix = re.sub(r"\b(am|pm)\b", "", text, flags=re.IGNORECASE).strip()

    parts = text_no_suffix.split(":")
    if len(parts) not in (2, 3) or not all(p.strip().isdigit() for p in parts):
        return text

    hour, minute = int(parts[0]), int(parts[1])
    second = int(parts[2]) if len(parts) == 3 else 0

    if am_pm == "pm" and hour < 12:
        hour += 12
    elif am_pm == "am" and hour == 12:
        hour = 0

    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        return text

    return f"{hour:02d}:{minute:02d}:{second:02d}"


@dataclass
class Item:
    name: str
    qty: float
    unit_price: float
    total: float


@dataclass
class Invoice:
    invoice_number: str
    date: str
    time: str
    table: str
    cashier: str
    subtotal: float
    vat: float
    discount: float
    grand_total: float
    items: List[Item] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Invoice":
        """Build a normalized Invoice from the raw model JSON dict."""
        items: List[Item] = []
        for raw_item in data.get("items", []) or []:
            items.append(
                Item(
                    name=str(raw_item.get("name", "")).strip(),
                    qty=normalize_number(raw_item.get("qty", 0)),
                    unit_price=normalize_number(raw_item.get("unit_price", 0)),
                    total=normalize_number(raw_item.get("total", 0)),
                )
            )

        subtotal = normalize_number(data.get("subtotal", 0))
        if subtotal == 0.0 and items:
            # This POS frequently prints only a net total ("الصافي") with no
            # separate subtotal line. Fall back to the sum of item totals so
            # the subtotal - discount + vat == grand_total check still holds.
            subtotal = round(sum(i.total for i in items), 2)

        return cls(
            invoice_number=str(data.get("invoice_number", "")).strip(),
            date=normalize_date(data.get("date", "")),
            time=normalize_time(data.get("time", "")),
            table=str(data.get("table", "")).strip(),
            cashier=str(data.get("cashier", "")).strip(),
            subtotal=subtotal,
            vat=normalize_number(data.get("vat", 0)),
            discount=normalize_number(data.get("discount", 0)),
            grand_total=normalize_number(data.get("grand_total", 0)),
            items=items,
        )
