"""Central configuration for the restaurant/cafe invoice OCR pipeline.

This is the *only* place where filesystem paths, OCR engine options and
template regions live. ``parser.py`` never hardcodes a coordinate: it always
reads a :class:`Region` from this module. To adapt the pipeline to a new
receipt layout, recalibrate the constants below (see README.md, "How to add
a new invoice template").

All region coordinates are expressed relative to the *standardized* image
produced by ``image_utils.preprocess_image`` (fixed width ``TARGET_WIDTH``,
deskewed, upright). Because receipts vary in length depending on the number
of items, vertical anchors are expressed either as an absolute pixel offset
from the top of the image (header, which has fixed height) or as an absolute
pixel offset from the bottom of the image (totals block, which also has a
fixed height). The items table fills whatever vertical space remains between
the two. Horizontal anchors are expressed as fractions of the image width,
since width is standardized for every invoice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# --------------------------------------------------------------------------
# Filesystem layout
# --------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INVOICE_FOLDER = BASE_DIR / "invoices"
PROCESSED_FOLDER = BASE_DIR / "processed"
OUTPUT_FOLDER = BASE_DIR / "output"
OUTPUT_FILE = OUTPUT_FOLDER / "invoices.xlsx"
LOG_FOLDER = BASE_DIR / "logs"
DEBUG_FOLDER = BASE_DIR / "debug"

for _folder in (
    INVOICE_FOLDER,
    PROCESSED_FOLDER,
    OUTPUT_FOLDER,
    LOG_FOLDER,
    DEBUG_FOLDER,
):
    _folder.mkdir(parents=True, exist_ok=True)

APPLICATION_LOG_FILE = LOG_FOLDER / "application.log"
OCR_LOG_FILE = LOG_FOLDER / "ocr.log"

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_LEVEL = logging.INFO

# --------------------------------------------------------------------------
# OCR engine configuration
# --------------------------------------------------------------------------

OCR_LANGUAGE = "en"
# PaddleOCR 3.x's own document-orientation classifier: detects and corrects
# gross 0/90/180/270 degree page rotation (e.g. a receipt photographed
# sideways) using a trained classifier, before text detection ever runs.
# This is what satisfies the "auto rotate if needed" preprocessing step.
OCR_USE_DOC_ORIENTATION_CLASSIFY = True
# Corrects perspective warp/curvature (e.g. a receipt photographed at an
# angle so it isn't flat). Off by default: it is comparatively expensive and
# most POS receipts are scanned/photographed flat; image_utils.deskew
# already corrects simple in-plane skew. Turn on for a camera-capture
# workflow with visibly curved/warped receipts.
OCR_USE_DOC_UNWARPING = False
# Per-textline 0/180 degree flip correction (distinct from whole-page
# rotation above); cheap and worth always leaving on.
OCR_USE_TEXTLINE_ORIENTATION = True
# "cpu" or "gpu:0" (requires a CUDA-enabled paddlepaddle build).
OCR_DEVICE = "cpu"
# Some paddlepaddle/oneDNN CPU builds (observed with paddlepaddle 3.x on
# Windows) raise "ConvertPirAttribute2RuntimeAttribute not support" from the
# PIR executor when MKLDNN is enabled for the text detection model. Disabled
# by default for reliability; set True if your build doesn't hit this and
# you want the CPU speedup.
OCR_ENABLE_MKLDNN = False
# Minimum recognition confidence to keep a word; anything below is dropped
# before it ever reaches the parser.
MIN_OCR_CONFIDENCE = 0.35

# --------------------------------------------------------------------------
# Image preprocessing
# --------------------------------------------------------------------------

# All invoice images are normalized to this width (aspect ratio preserved)
# before OCR so that every pixel-based region below means the same thing
# regardless of the source photo/scan resolution.
TARGET_WIDTH = 1000
# If a source image is narrower than this, it is considered low resolution
# and is upscaled (in addition to the standardization above).
MIN_ACCEPTABLE_WIDTH = 600

CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILE_GRID_SIZE = (8, 8)
DENOISE_STRENGTH = 7
# Maximum absolute skew angle (degrees) that deskewing will attempt to
# correct. Larger deviations are assumed to be gross rotations handled by
# auto-rotation instead.
MAX_DESKEW_ANGLE = 15.0

# --------------------------------------------------------------------------
# Template regions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Region:
    """A rectangular region of interest on the standardized invoice image.

    Horizontal bounds are fractions of image width (0.0-1.0). Vertical
    bounds may be anchored from the top (``y_start``/``y_end``) or from the
    bottom (``y_start_from_bottom``/``y_end_from_bottom``) of the image,
    which lets a fixed-height header/totals block be located correctly on
    receipts of any length.
    """

    x_start_frac: float
    x_end_frac: float
    y_start: Optional[int] = None
    y_end: Optional[int] = None
    y_start_from_bottom: Optional[int] = None
    y_end_from_bottom: Optional[int] = None

    def bounds(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Resolve this region to absolute pixel coordinates for an image
        of the given size."""
        x1 = int(round(self.x_start_frac * width))
        x2 = int(round(self.x_end_frac * width))

        if self.y_start is not None:
            y1 = self.y_start
        elif self.y_start_from_bottom is not None:
            y1 = height - self.y_start_from_bottom
        else:
            y1 = 0

        if self.y_end is not None:
            y2 = self.y_end
        elif self.y_end_from_bottom is not None:
            y2 = height - self.y_end_from_bottom
        else:
            y2 = height

        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))
        return x1, y1, x2, y2

    def contains(self, cx: float, cy: float, width: int, height: int) -> bool:
        """Return True if the point (cx, cy) falls inside this region when
        resolved against an image of the given size."""
        x1, y1, x2, y2 = self.bounds(width, height)
        return x1 <= cx <= x2 and y1 <= cy <= y2


# Fixed-height block at the very top of the receipt: shop name, invoice
# number, date/time, table, cashier.
HEADER_HEIGHT_PX = 260
# Fixed-height block at the very bottom of the receipt: subtotal, VAT,
# discount, grand total (and any footer text below it is ignored).
TOTALS_HEIGHT_PX = 220

HEADER_REGION = Region(x_start_frac=0.0, x_end_frac=1.0, y_start=0, y_end=HEADER_HEIGHT_PX)

ITEMS_REGION = Region(
    x_start_frac=0.0,
    x_end_frac=1.0,
    y_start=HEADER_HEIGHT_PX,
    y_end_from_bottom=TOTALS_HEIGHT_PX,
)

TOTALS_REGION = Region(
    x_start_frac=0.0,
    x_end_frac=1.0,
    y_start_from_bottom=TOTALS_HEIGHT_PX,
    y_end_from_bottom=0,
)

# Sub-regions inside HEADER_REGION. These are the calibration knobs to tune
# when pointing the parser at a new (but still POS-generated, fixed-layout)
# receipt template. Use debug mode to see where OCR boxes actually land and
# adjust these pixel bands accordingly.
INVOICE_NUMBER_REGION = Region(x_start_frac=0.0, x_end_frac=1.0, y_start=0, y_end=65)
DATE_REGION = Region(x_start_frac=0.0, x_end_frac=0.55, y_start=65, y_end=130)
TIME_REGION = Region(x_start_frac=0.55, x_end_frac=1.0, y_start=65, y_end=130)
TABLE_REGION = Region(x_start_frac=0.0, x_end_frac=0.5, y_start=130, y_end=195)
CASHIER_REGION = Region(x_start_frac=0.5, x_end_frac=1.0, y_start=130, y_end=195)

# Column boundaries inside ITEMS_REGION, as fractions of image width. Words
# whose center-x falls in a column are assigned to that column of the
# current row.
ITEM_NAME_COLUMN = (0.0, 0.45)
ITEM_QTY_COLUMN = (0.45, 0.60)
ITEM_UNIT_PRICE_COLUMN = (0.60, 0.80)
ITEM_TOTAL_COLUMN = (0.80, 1.0)

# Maximum difference (pixels) between two words' center-y for them to be
# considered part of the same item row.
ROW_Y_TOLERANCE_PX = 14

# Labels used to identify each line inside TOTALS_REGION. Order matters:
# more specific labels (e.g. "subtotal") must be checked before the generic
# "total" so a subtotal line is never misread as the grand total.
TOTALS_KEYWORDS = {
    "subtotal": ("subtotal", "sub total", "sub-total"),
    "vat": ("vat", "tax"),
    "discount": ("discount", "disc"),
    "grand_total": ("grand total", "total due", "amount due", "net total", "total"),
}

HEADER_KEYWORDS = {
    "invoice_number": ("invoice no", "invoice #", "invoice number", "inv no", "invoice", "receipt no"),
    "date": ("date",),
    "time": ("time",),
    "table": ("table", "table no", "table number"),
    "cashier": ("cashier", "server", "waiter"),
}

# --------------------------------------------------------------------------
# Excel output
# --------------------------------------------------------------------------

EXCEL_COLUMNS = [
    "Invoice No",
    "Date",
    "Time",
    "Table",
    "Cashier",
    "Item",
    "Quantity",
    "Unit Price",
    "Item Total",
    "Subtotal",
    "VAT",
    "Discount",
    "Grand Total",
]

EXCEL_SHEET_NAME = "Invoices"
