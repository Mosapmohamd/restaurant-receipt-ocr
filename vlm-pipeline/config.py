"""Central configuration for the restaurant invoice extraction system.

All tunable settings live here so behaviour can be changed without touching
application code. Values can be overridden with environment variables for
deployment flexibility.
"""

import logging
import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent

INVOICES_DIR = BASE_DIR / "invoices"
PROCESSED_DIR = BASE_DIR / "processed"
FAILED_DIR = BASE_DIR / "failed"
DEBUG_DIR = BASE_DIR / "debug"
OUTPUT_DIR = BASE_DIR / "output"
LOGS_DIR = BASE_DIR / "logs"

# Spec-named aliases (kept identical to the directories above).
WATCH_FOLDER = INVOICES_DIR
PROCESSED_FOLDER = PROCESSED_DIR
DEBUG_FOLDER = DEBUG_DIR

for _d in (INVOICES_DIR, PROCESSED_DIR, FAILED_DIR, DEBUG_DIR, OUTPUT_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL_NAME = os.getenv("MODEL_NAME", "MedAIBase/PaddleOCR-VL:0.9b")

# Minimum fraction of key fields that must be non-empty/non-zero for an
# extraction to be trusted. The model does not return per-field confidence
# scores, so this is used as a heuristic completeness gate (see
# extractor.py: _completeness_score). Extractions below this threshold are
# treated as failed attempts and retried.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))

# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
OUTPUT_FILE = OUTPUT_DIR / "invoices.xlsx"

# --------------------------------------------------------------------------
# Processing
# --------------------------------------------------------------------------
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "3"))
RETRY_BACKOFF_SECONDS = float(os.getenv("RETRY_BACKOFF_SECONDS", "1.0"))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Consistency check tolerance for subtotal - discount + vat == grand_total.
# Combines an absolute floor with a small relative allowance so minor OCR
# rounding does not fail otherwise-correct invoices.
VALIDATION_ABS_TOLERANCE = float(os.getenv("VALIDATION_ABS_TOLERANCE", "0.5"))
VALIDATION_REL_TOLERANCE = float(os.getenv("VALIDATION_REL_TOLERANCE", "0.01"))

# --------------------------------------------------------------------------
# Image preprocessing
# --------------------------------------------------------------------------
MAX_IMAGE_WIDTH = int(os.getenv("MAX_IMAGE_WIDTH", "2000"))
MAX_IMAGE_HEIGHT = int(os.getenv("MAX_IMAGE_HEIGHT", "2000"))
SAVE_DEBUG_ARTIFACTS = os.getenv("SAVE_DEBUG_ARTIFACTS", "1") not in ("0", "false", "False")

# --------------------------------------------------------------------------
# Watcher
# --------------------------------------------------------------------------
WATCHER_STABLE_CHECK_INTERVAL = float(os.getenv("WATCHER_STABLE_CHECK_INTERVAL", "0.5"))
WATCHER_STABLE_CHECKS = int(os.getenv("WATCHER_STABLE_CHECKS", "3"))

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
LOG_FILE = LOGS_DIR / "application.log"
ERROR_LOG_FILE = LOGS_DIR / "errors.log"

_LOGGING_CONFIGURED = False


def configure_logging() -> None:
    """Attach application/error/console handlers to the root logger.

    Idempotent: safe to call from main.py, watcher.py, or any entry point
    without producing duplicate log lines.
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    app_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(fmt)

    # Every ERROR-and-above record from any logger in the process (JSON
    # parse failures, validation failures, duplicate invoices, exceptions)
    # propagates to the root logger and lands here.
    error_handler = logging.FileHandler(ERROR_LOG_FILE, encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(app_handler)
    root.addHandler(error_handler)
    root.addHandler(console_handler)

    _LOGGING_CONFIGURED = True
