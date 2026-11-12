"""Template-based invoice parser.

This module deliberately does *not* try to understand invoice text with
regex alone. Instead it uses the OCR bounding boxes to decide which text
belongs to which section of the receipt (header, items table, totals),
exactly mirroring the physical layout produced by the POS system. Regex is
only ever used as a small cleanup step on text that coordinates have
already isolated to a single field or a single table cell -- never as the
primary segmentation mechanism.

All coordinates/thresholds are read from ``config.py``; nothing here is
hardcoded, so retargeting a new (but still fixed-layout) POS template only
requires editing region constants in one place.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import config
from extractor import OCRWord

logger = logging.getLogger("restaurant_ocr.parser")

Number = Union[int, float]

_NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*\.?\d*")
_DATE_PATTERN = re.compile(r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}")
_TIME_PATTERN = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?")
_DIGITS_PATTERN = re.compile(r"\d+")


@dataclass
class InvoiceItem:
    """A single line item extracted from the items table."""

    item: str
    qty: Number
    unit_price: Number
    total: Number

    def to_dict(self) -> dict:
        return {
            "item": self.item,
            "qty": _clean_number(self.qty),
            "unit_price": _clean_number(self.unit_price),
            "total": _clean_number(self.total),
        }


@dataclass
class Invoice:
    """The fully parsed invoice, matching the pipeline's required JSON
    output shape."""

    invoice_number: str = ""
    date: str = ""
    time: str = ""
    table: str = ""
    cashier: str = ""
    subtotal: Number = 0
    vat: Number = 0
    discount: Number = 0
    grand_total: Number = 0
    items: List[InvoiceItem] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "invoice_number": self.invoice_number,
            "date": self.date,
            "time": self.time,
            "table": self.table,
            "cashier": self.cashier,
            "subtotal": _clean_number(self.subtotal),
            "vat": _clean_number(self.vat),
            "discount": _clean_number(self.discount),
            "grand_total": _clean_number(self.grand_total),
            "items": [item.to_dict() for item in self.items],
        }


def _clean_number(value: Number) -> Number:
    """Return an int when a float carries no fractional part, otherwise a
    2-decimal-rounded float -- keeps JSON/Excel output tidy."""
    rounded = round(float(value), 2)
    if rounded == int(rounded):
        return int(rounded)
    return rounded


def _parse_number(text: str) -> Optional[float]:
    """Extract the last numeric token in ``text`` (receipts print
    "Label .... 123.45", so the rightmost number is the value) and parse it
    as a float. Returns None if no number is present."""
    matches = _NUMBER_PATTERN.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def _line_text(line: Sequence[OCRWord]) -> str:
    return " ".join(word.text for word in line)


def group_lines(
    words: Sequence[OCRWord],
    region: config.Region,
    width: int,
    height: int,
    y_tolerance: int = config.ROW_Y_TOLERANCE_PX,
) -> List[List[OCRWord]]:
    """Select the words whose center falls inside ``region`` and group them
    into text lines/rows by proximity in the y axis, then sort each line
    left-to-right. This is the coordinate-based building block every
    section parser is built on."""
    in_region = [w for w in words if region.contains(w.cx, w.cy, width, height)]
    in_region.sort(key=lambda w: w.cy)

    lines: List[List[OCRWord]] = []
    for word in in_region:
        if lines and abs(word.cy - lines[-1][-1].cy) <= y_tolerance:
            lines[-1].append(word)
        else:
            lines.append([word])

    for line in lines:
        line.sort(key=lambda w: w.cx)

    return lines


def _strip_labels(text: str, keywords: Sequence[str]) -> str:
    """Remove a known field label (e.g. "Invoice No:") from the start of a
    line so what remains is just the value."""
    lowered = text.lower()
    for keyword in sorted(keywords, key=len, reverse=True):
        idx = lowered.find(keyword)
        if idx != -1:
            remainder = text[idx + len(keyword):]
            return remainder.lstrip(" :#-\t").strip()
    return text.strip()


def _extract_field(
    lines: Sequence[Sequence[OCRWord]],
    keywords: Sequence[str],
    value_pattern: Optional[re.Pattern] = None,
) -> str:
    """Find the first line matching one of ``keywords`` and return its
    cleaned value, optionally constrained to ``value_pattern``."""
    for line in lines:
        text = _line_text(line)
        lowered = text.lower()
        if any(keyword in lowered for keyword in keywords):
            remainder = _strip_labels(text, keywords)
            if value_pattern is not None:
                match = value_pattern.search(remainder) or value_pattern.search(text)
                if match:
                    return match.group().strip()
            if remainder:
                return remainder
    return ""


def _parse_header(
    words: Sequence[OCRWord], width: int, height: int
) -> Dict[str, str]:
    """Parse the header block using dedicated sub-regions first (primary,
    coordinate-based), falling back to keyword search across the whole
    header if a sub-region comes up empty (e.g. the printer shifted a field
    slightly on a given receipt)."""
    header_lines = group_lines(words, config.HEADER_REGION, width, height)

    def field_value(region: config.Region, keywords: Sequence[str], pattern: Optional[re.Pattern]) -> str:
        region_lines = group_lines(words, region, width, height)
        value = _extract_field(region_lines, keywords, pattern)
        if value:
            return value
        # Sub-region empty/unmatched: fall back to scanning the full header.
        return _extract_field(header_lines, keywords, pattern)

    invoice_number = field_value(
        config.INVOICE_NUMBER_REGION, config.HEADER_KEYWORDS["invoice_number"], _DIGITS_PATTERN
    )
    date = field_value(config.DATE_REGION, config.HEADER_KEYWORDS["date"], _DATE_PATTERN)
    time_value = field_value(config.TIME_REGION, config.HEADER_KEYWORDS["time"], _TIME_PATTERN)
    table = field_value(config.TABLE_REGION, config.HEADER_KEYWORDS["table"], _DIGITS_PATTERN)
    cashier = field_value(config.CASHIER_REGION, config.HEADER_KEYWORDS["cashier"], None)

    return {
        "invoice_number": invoice_number,
        "date": date,
        "time": time_value,
        "table": table,
        "cashier": cashier,
    }


def _parse_totals(words: Sequence[OCRWord], width: int, height: int) -> Dict[str, float]:
    """Parse the totals block: each line is matched against
    ``config.TOTALS_KEYWORDS`` (checked in the configured, most-specific-
    first order) and the rightmost number on that line becomes the value."""
    lines = group_lines(words, config.TOTALS_REGION, width, height)
    totals: Dict[str, float] = {"subtotal": 0.0, "vat": 0.0, "discount": 0.0, "grand_total": 0.0}
    matched_lines: set = set()

    for key, keywords in config.TOTALS_KEYWORDS.items():
        for index, line in enumerate(lines):
            if index in matched_lines:
                continue
            text = _line_text(line).lower()
            if any(keyword in text for keyword in keywords):
                value = _parse_number(_line_text(line))
                if value is not None:
                    totals[key] = value
                    matched_lines.add(index)
                    break

    return totals


_HEADER_ROW_MARKERS = ("item", "qty", "quantity", "price", "unit", "total", "amount")


def _looks_like_column_header(line: Sequence[OCRWord]) -> bool:
    """Detect the items table's own header row (e.g. "Item Qty Price
    Total") so it isn't parsed as a line item."""
    text = _line_text(line).lower()
    hits = sum(1 for marker in _HEADER_ROW_MARKERS if marker in text)
    return hits >= 2


def _words_in_column(line: Sequence[OCRWord], column: Tuple[float, float], width: int) -> List[OCRWord]:
    x_start = column[0] * width
    x_end = column[1] * width
    return [w for w in line if x_start <= w.cx < x_end]


def _parse_items(words: Sequence[OCRWord], width: int, height: int) -> List[InvoiceItem]:
    """Parse the items table: rows are formed by clustering words with
    similar y-coordinates, then each row is split into its name/qty/price/
    total cells using the column boundaries from config.py."""
    lines = group_lines(words, config.ITEMS_REGION, width, height)
    items: List[InvoiceItem] = []

    for line in lines:
        if _looks_like_column_header(line):
            continue

        name_words = _words_in_column(line, config.ITEM_NAME_COLUMN, width)
        qty_words = _words_in_column(line, config.ITEM_QTY_COLUMN, width)
        price_words = _words_in_column(line, config.ITEM_UNIT_PRICE_COLUMN, width)
        total_words = _words_in_column(line, config.ITEM_TOTAL_COLUMN, width)

        item_name = " ".join(w.text for w in name_words).strip()
        qty = _parse_number(_line_text(qty_words)) if qty_words else None
        unit_price = _parse_number(_line_text(price_words)) if price_words else None
        total = _parse_number(_line_text(total_words)) if total_words else None

        if not item_name or (qty is None and unit_price is None and total is None):
            continue

        if unit_price is not None and total is not None and qty is None:
            qty = round(total / unit_price, 2) if unit_price else 1
        elif qty is not None and unit_price is not None and total is None:
            total = round(qty * unit_price, 2)
        elif qty is not None and total is not None and unit_price is None:
            unit_price = round(total / qty, 2) if qty else total

        items.append(
            InvoiceItem(
                item=item_name,
                qty=qty if qty is not None else 1,
                unit_price=unit_price if unit_price is not None else 0.0,
                total=total if total is not None else 0.0,
            )
        )

    return items


def parse_invoice(words: Sequence[OCRWord], width: int, height: int) -> Invoice:
    """Parse a full invoice from its OCR words and the image dimensions
    they were detected in (needed to resolve fraction-based region
    boundaries)."""
    header = _parse_header(words, width, height)
    totals = _parse_totals(words, width, height)
    items = _parse_items(words, width, height)

    if not totals["subtotal"] and items:
        totals["subtotal"] = round(sum(item.total for item in items), 2)
    if not totals["grand_total"]:
        totals["grand_total"] = round(
            totals["subtotal"] + totals["vat"] - totals["discount"], 2
        )

    logger.debug(
        "Parsed invoice: header=%s totals=%s item_count=%d", header, totals, len(items)
    )

    return Invoice(
        invoice_number=header["invoice_number"],
        date=header["date"],
        time=header["time"],
        table=header["table"],
        cashier=header["cashier"],
        subtotal=totals["subtotal"],
        vat=totals["vat"],
        discount=totals["discount"],
        grand_total=totals["grand_total"],
        items=items,
    )
