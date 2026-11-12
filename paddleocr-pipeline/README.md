# Restaurant/Cafe Invoice OCR

Extracts structured data from restaurant/cafe POS receipt images and appends
it to a single Excel workbook (`output/invoices.xlsx`), one row per line
item. Built for a **single, fixed POS receipt layout** — this is a
template-based coordinate parser, not a generic invoice-understanding model.

## Installation

Requires Python 3.9+ (validated on Python 3.13 with PaddlePaddle 3.x / PaddleOCR 3.x).

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Requirements installed:

- `paddlepaddle` / `paddleocr` — text detection + recognition
- `opencv-python` — image preprocessing (grayscale, contrast, denoise, deskew, resize)
- `pandas` — light data shaping
- `openpyxl` — Excel read/write
- `numpy` — array plumbing between OpenCV and PaddleOCR
- `watchdog` — folder watching for `watcher.py`

First run downloads the PaddleOCR detection/recognition/angle-classification
models (a few hundred MB) into `~/.paddleocr`; this requires an internet
connection once.

## Running

```bash
python main.py            # process every image currently in invoices/
python main.py --debug    # also save annotated bounding-box overlays to debug/

python watcher.py         # keep running, auto-process new files dropped into invoices/
python watcher.py --debug
```

Every invoice image placed in `invoices/` is processed and moved into
`processed/` (renamed if a name collision occurs). Results are appended to
`output/invoices.xlsx`. Logs are written to `logs/application.log` (pipeline
events, errors, per-invoice timing) and `logs/ocr.log` (OCR confidence and
orientation-detection detail).

## Project architecture

```
config.py         Paths, OCR settings, and every template Region (single source of truth)
image_utils.py    Deterministic preprocessing: resize, grayscale, contrast, denoise, deskew
                  + debug-overlay rendering
extractor.py      PaddleOCR wrapper -> OCRWord(text, score, x, y, x2, y2, cx, cy)
                  + OCR-confidence-based auto page-rotation (0/90/180/270)
parser.py         Region-based template parser -> Invoice / InvoiceItem dataclasses
excel_writer.py   Appends one row per item to output/invoices.xlsx, with duplicate detection
main.py           CLI: batch-process invoices/, move to processed/, optional --debug
watcher.py        watchdog-based folder watcher, reuses main.py's pipeline
```

Data flows one way through the pipeline:

```
image file -> image_utils.preprocess_image -> extractor.extract_words
           -> parser.parse_invoice -> excel_writer.save_invoice
```

`main.py` and `watcher.py` are both thin orchestrators around
`process_invoice_file()`; the watcher does not duplicate any pipeline logic.

## How OCR works (`extractor.py`)

1. `image_utils.preprocess_image` standardizes the image to a fixed width
   (`config.TARGET_WIDTH`), converts to grayscale, applies CLAHE contrast
   enhancement, denoises, and corrects small skew (a few degrees) using the
   minimum-area-rectangle of foreground pixels.
2. `InvoiceOCREngine` hands the image to PaddleOCR's pipeline with
   `use_doc_orientation_classify=True`: a trained classifier detects gross
   0°/90°/180°/270° page rotation (e.g. a receipt photographed sideways)
   and corrects it *before* text detection runs, so every bounding box
   PaddleOCR returns is already relative to the corrected image.
   `use_textline_orientation=True` additionally corrects individual
   upside-down text lines. `extractor.extract_words` returns this corrected
   image (`page["doc_preprocessor_res"]["output_img"]`) alongside the
   words, so parser/debug code stay in the same coordinate space.
3. Every recognized text box is converted into an `OCRWord`:

   ```python
   {
       "text": "120.00",
       "score": 0.998,
       "x": 351, "y": 621, "x2": 401, "y2": 648,
       "cx": 376, "cy": 634,
   }
   ```

   `cx`/`cy` (box center) are what the parser uses to decide which region
   and which row/column a word belongs to. Words below
   `config.MIN_OCR_CONFIDENCE` are dropped before they ever reach the
   parser.

## How the parser works (`parser.py`)

The parser never tries to understand the whole receipt as free text. It
uses OCR **coordinates** to decide which section a word belongs to, mirroring
the physical layout the POS system always prints:

1. **Region assignment.** `config.py` defines `HEADER_REGION`,
   `ITEMS_REGION` and `TOTALS_REGION` as pixel bands anchored from the top
   (header — fixed height) or from the bottom (totals — fixed height); the
   items table fills whatever space remains between them, so receipts with
   more or fewer line items are handled correctly without any region
   re-tuning.
2. **Header fields** (`invoice_number`, `date`, `time`, `table`, `cashier`)
   each have their own sub-region (`DATE_REGION`, `TABLE_REGION`, etc.).
   The parser first looks for a value in that dedicated sub-region; if
   nothing usable is found there it falls back to a keyword search
   (`config.HEADER_KEYWORDS`) across the whole header. Regex is only used
   at the very end, to pull a date/time/number pattern out of the text a
   region has already isolated — never to scan the whole receipt.
3. **Items table.** Words inside `ITEMS_REGION` are grouped into rows by
   clustering on `cy` (`config.ROW_Y_TOLERANCE_PX`), then each row is split
   into name/qty/unit-price/total cells using the column boundaries in
   `config.ITEM_NAME_COLUMN` / `ITEM_QTY_COLUMN` / `ITEM_UNIT_PRICE_COLUMN`
   / `ITEM_TOTAL_COLUMN` (fractions of image width). A missing cell (e.g. a
   printer that omits unit price when qty is 1) is back-filled from the
   other two when possible (`total = qty * unit_price`, etc.).
4. **Totals.** Lines inside `TOTALS_REGION` are matched against
   `config.TOTALS_KEYWORDS`, checked in a deliberate order (`subtotal`
   before the generic `total`, so a "Subtotal" line is never mistaken for
   the grand total). The rightmost number on the matched line is taken as
   the value.

Output shape (`Invoice.to_dict()` / one row per item in Excel):

```json
{
  "invoice_number": "",
  "date": "",
  "time": "",
  "table": "",
  "cashier": "",
  "subtotal": 0,
  "vat": 0,
  "discount": 0,
  "grand_total": 0,
  "items": [
    {"item": "Avocado Honey", "qty": 1, "unit_price": 120.0, "total": 120.0}
  ]
}
```

## Excel output

`output/invoices.xlsx`, sheet `Invoices`, columns:

```
Invoice No | Date | Time | Table | Cashier | Item | Quantity | Unit Price | Item Total | Subtotal | VAT | Discount | Grand Total
```

One row is written per item; invoice-level fields (date, totals, etc.) are
repeated on every row belonging to that invoice.

**Duplicate detection**: before writing, every existing row in the sheet is
used to reconstruct an identity key — `Invoice No` when present, otherwise a
hash of `Date + Time + Grand Total`. If the invoice being processed matches
an existing key, the whole invoice is skipped (logged as a warning) instead
of being appended again. This works across separate runs of `main.py` and
`watcher.py` since the check is against the spreadsheet itself, not an
in-memory cache.

## Debug mode

`--debug` renders, for every processed invoice, a PNG in `debug/` showing:

- every detected OCR box, labeled with its recognized text and confidence
  (green if `score >= 0.8`, orange otherwise)
- every configured template region, outlined and labeled (`HEADER`,
  `ITEMS`, `TOTALS`, `DATE`, `TABLE`, `CASHIER`, ...)

This is the primary tool for calibrating regions (see below) — it makes it
immediately obvious whether a field's sub-region is too narrow, too wide, or
misplaced.

## How to add a new invoice template / retune regions

This project targets **one** POS layout at a time via the constants in
`config.py`. To adapt it to a different (but still fixed-layout) receipt:

1. Drop a handful of representative images into `invoices/` and run
   `python main.py --debug`.
2. Open the corresponding file in `debug/`. Check:
   - Does `HEADER` fully contain the shop name/invoice number/date/time/
     table/cashier block, and nothing from the items table? Adjust
     `HEADER_HEIGHT_PX`.
   - Does `TOTALS` fully contain subtotal/VAT/discount/grand total, and
     nothing from the last item row? Adjust `TOTALS_HEIGHT_PX`.
   - Do `DATE_REGION` / `TIME_REGION` / `TABLE_REGION` / `CASHIER_REGION` /
     `INVOICE_NUMBER_REGION` each tightly bound their field? Adjust their
     `y_start`/`y_end` (and `x_start_frac`/`x_end_frac` if two fields sit
     side by side on the same line) in `config.py`.
   - Do `ITEM_NAME_COLUMN` / `ITEM_QTY_COLUMN` / `ITEM_UNIT_PRICE_COLUMN` /
     `ITEM_TOTAL_COLUMN` align with where those columns actually print?
     Adjust the fraction boundaries.
3. Re-run with `--debug` and repeat until the overlay matches the real
   layout. No code changes are needed — `parser.py` only ever reads these
   constants.
4. If the new template's line spacing is tighter/looser than the default,
   adjust `ROW_Y_TOLERANCE_PX` (too small: one row gets split into two; too
   large: two rows get merged into one).
5. If receipts are in a different language, change `OCR_LANGUAGE` in
   `config.py` to the appropriate PaddleOCR language code (e.g. `"en"`,
   `"ar"`, `"ch"`).

## Notes and assumptions

- The pipeline assumes every invoice comes from the same POS system and
  therefore the same physical layout — this is a deliberate simplification
  requested for this project, not a limitation of the OCR itself.
- `TARGET_WIDTH` standardization is what makes pixel-based regions portable
  across photos taken at different resolutions/distances; if your capture
  setup varies wildly in aspect ratio (not just resolution), recalibrate
  `HEADER_HEIGHT_PX`/`TOTALS_HEIGHT_PX` against your own sample images.
- `MIN_OCR_CONFIDENCE` (default `0.35`) filters obviously wrong detections
  before parsing; lower it if faint thermal-print text is being dropped,
  raise it if noise is leaking into rows as phantom words.
