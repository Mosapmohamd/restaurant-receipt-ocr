"""Excel output writer with duplicate-invoice protection.

Every parsed invoice becomes one row per line item in ``output/invoices.xlsx``
(columns defined in ``config.EXCEL_COLUMNS``). Before writing, the invoice's
identity is checked against every row already in the sheet so the same
receipt is never recorded twice, even across separate runs of the CLI or the
folder watcher.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Set, Tuple

from openpyxl import Workbook, load_workbook
from openpyxl.worksheet.worksheet import Worksheet

import config
from parser import Invoice

logger = logging.getLogger("restaurant_ocr.excel_writer")


def _hash_key(date: str, time: str, grand_total) -> str:
    raw = f"{date or ''}|{time or ''}|{grand_total or 0}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"hash:{digest}"


def _invoice_key(invoice: Invoice) -> str:
    """A stable identifier used for duplicate detection: the invoice number
    when the parser found one, otherwise a hash of date + time + grand
    total (stable across runs since those three fields together reliably
    identify a distinct receipt for this POS system)."""
    if invoice.invoice_number:
        return f"num:{invoice.invoice_number.strip().lower()}"
    return _hash_key(invoice.date, invoice.time, invoice.grand_total)


def _load_existing_keys(sheet: Worksheet) -> Set[str]:
    """Reconstruct the set of invoice keys already present in the sheet by
    re-deriving them from each stored row, so duplicate detection works
    correctly across separate process runs (nothing is cached outside the
    spreadsheet itself)."""
    keys: Set[str] = set()
    header = [cell.value for cell in next(sheet.iter_rows(min_row=1, max_row=1))]

    try:
        invoice_no_idx = header.index("Invoice No")
        date_idx = header.index("Date")
        time_idx = header.index("Time")
        grand_total_idx = header.index("Grand Total")
    except ValueError:
        logger.warning("Existing workbook has an unexpected header; duplicate detection disabled for it")
        return keys

    for row in sheet.iter_rows(min_row=2, values_only=True):
        invoice_no = row[invoice_no_idx]
        if invoice_no:
            keys.add(f"num:{str(invoice_no).strip().lower()}")
        else:
            keys.add(_hash_key(row[date_idx], row[time_idx], row[grand_total_idx]))

    return keys


def _open_or_create_workbook(path: Path) -> Tuple[Workbook, Worksheet]:
    if path.exists():
        workbook = load_workbook(path)
        if config.EXCEL_SHEET_NAME in workbook.sheetnames:
            sheet = workbook[config.EXCEL_SHEET_NAME]
        else:
            sheet = workbook.create_sheet(config.EXCEL_SHEET_NAME)
            sheet.append(config.EXCEL_COLUMNS)
    else:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = config.EXCEL_SHEET_NAME
        sheet.append(config.EXCEL_COLUMNS)

    return workbook, sheet


def save_invoice(invoice: Invoice, output_file: Path = config.OUTPUT_FILE) -> bool:
    """Append one row per line item of ``invoice`` to ``output_file``.

    Returns True if the invoice was written, False if it was skipped as a
    duplicate of an invoice already present in the sheet.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    workbook, sheet = _open_or_create_workbook(output_file)
    existing_keys = _load_existing_keys(sheet)

    key = _invoice_key(invoice)
    if key in existing_keys:
        logger.warning(
            "Duplicate invoice skipped (key=%s, invoice_number=%r)", key, invoice.invoice_number
        )
        return False

    rows = invoice.items if invoice.items else [None]
    for item in rows:
        sheet.append(
            [
                invoice.invoice_number,
                invoice.date,
                invoice.time,
                invoice.table,
                invoice.cashier,
                item.item if item else "",
                item.qty if item else "",
                item.unit_price if item else "",
                item.total if item else "",
                invoice.subtotal,
                invoice.vat,
                invoice.discount,
                invoice.grand_total,
            ]
        )

    workbook.save(output_file)
    logger.info(
        "Saved invoice %s to %s (%d item rows)",
        invoice.invoice_number or key,
        output_file,
        len(rows),
    )
    return True
