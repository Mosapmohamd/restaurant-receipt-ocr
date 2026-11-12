"""Thread-safe, incremental Excel writer with atomic duplicate detection.

Duplicate-check + append is performed under a single lock so two workers
processing different invoices concurrently cannot both pass the duplicate
check for what turns out to be the same invoice before either has written
it (a classic check-then-act race). The workbook is kept open in memory for
the life of the writer instead of being fully re-read from disk on every
call, which keeps batch/concurrent processing fast.
"""

import logging
import threading
from pathlib import Path
from typing import Optional

from openpyxl import Workbook, load_workbook
from openpyxl.worksheet.worksheet import Worksheet

from config import OUTPUT_FILE
from parser import Invoice
from validator import compute_invoice_hash

logger = logging.getLogger(__name__)

SHEET_NAME = "Invoices"
COLUMNS = [
    "Invoice Number", "Date", "Time", "Table", "Cashier",
    "Item", "Quantity", "Unit Price", "Item Total",
    "Subtotal", "VAT", "Discount", "Grand Total", "Hash",
]
_INVOICE_NUMBER_COL = COLUMNS.index("Invoice Number")
_HASH_COL = COLUMNS.index("Hash")


class ExcelWriter:
    """Owns the single output workbook for a whole run."""

    def __init__(self, file_path: Path = OUTPUT_FILE) -> None:
        self.file_path = Path(file_path)
        self._lock = threading.Lock()
        self._invoice_numbers: set[str] = set()
        self._hashes: set[str] = set()
        self._wb: Workbook
        self._ws: Worksheet
        self._load_or_create()

    def _load_or_create(self) -> None:
        if self.file_path.exists():
            try:
                self._wb = load_workbook(self.file_path)
                self._ws = (
                    self._wb[SHEET_NAME] if SHEET_NAME in self._wb.sheetnames else self._wb.active
                )
                self._rebuild_dedup_index()
                return
            except Exception as e:
                logger.error(f"Could not load existing workbook {self.file_path}, starting fresh: {e}")

        self._wb = Workbook()
        self._ws = self._wb.active
        self._ws.title = SHEET_NAME
        self._ws.append(COLUMNS)

    def _rebuild_dedup_index(self) -> None:
        for row in self._ws.iter_rows(min_row=2, values_only=True):
            if row is None or len(row) <= _HASH_COL:
                continue
            inv_num, row_hash = row[_INVOICE_NUMBER_COL], row[_HASH_COL]
            if inv_num:
                self._invoice_numbers.add(str(inv_num))
            if row_hash:
                self._hashes.add(str(row_hash))

    def append_invoice(self, inv: Invoice) -> str:
        """Atomically check for a duplicate and append if new.

        Returns "appended" or "duplicate". Never raises for expected
        conditions; unexpected I/O errors propagate to the caller so they
        can be logged and the source file routed to failed/.
        """
        with self._lock:
            inv_hash = compute_invoice_hash(inv)
            if inv.invoice_number and inv.invoice_number in self._invoice_numbers:
                return "duplicate"
            if inv_hash in self._hashes:
                return "duplicate"

            rows = [
                [
                    inv.invoice_number, inv.date, inv.time, inv.table, inv.cashier,
                    item.name, item.qty, item.unit_price, item.total,
                    inv.subtotal, inv.vat, inv.discount, inv.grand_total, inv_hash,
                ]
                for item in inv.items
            ] or [[
                inv.invoice_number, inv.date, inv.time, inv.table, inv.cashier,
                "", 0, 0, 0,
                inv.subtotal, inv.vat, inv.discount, inv.grand_total, inv_hash,
            ]]

            for row in rows:
                self._ws.append(row)

            self._wb.save(self.file_path)

            if inv.invoice_number:
                self._invoice_numbers.add(inv.invoice_number)
            self._hashes.add(inv_hash)

            logger.info(
                f"Appended invoice {inv.invoice_number or inv_hash[:8]} "
                f"({len(inv.items)} items) to {self.file_path}"
            )
            return "appended"


_writer_lock = threading.Lock()
_writer_instance: Optional[ExcelWriter] = None


def get_excel_writer(file_path: Path = OUTPUT_FILE) -> ExcelWriter:
    """Process-wide singleton writer, created lazily on first use."""
    global _writer_instance
    with _writer_lock:
        if _writer_instance is None:
            _writer_instance = ExcelWriter(file_path)
        return _writer_instance
