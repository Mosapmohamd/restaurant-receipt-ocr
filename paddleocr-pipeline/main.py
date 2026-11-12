"""CLI entry point: batch-process every invoice image currently in
invoices/ into output/invoices.xlsx, then move each into processed/.

Usage:
    python main.py             # process all invoices once
    python main.py --debug     # also save annotated overlays into debug/
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

import cv2

import config
from excel_writer import save_invoice
from extractor import extract_words
from image_utils import DebugWord, draw_debug_overlay, draw_region_overlay, preprocess_image
from parser import Invoice, parse_invoice

logger = logging.getLogger("restaurant_ocr.main")


def configure_logging() -> None:
    """Set up application-wide logging: a general application.log for
    pipeline events/errors (plus console output), and a dedicated ocr.log
    capturing OCR confidence/timing detail emitted by extractor.py."""
    root = logging.getLogger("restaurant_ocr")
    if root.handlers:
        return  # already configured, e.g. watcher.py imported after main.py

    root.setLevel(config.LOG_LEVEL)
    formatter = logging.Formatter(config.LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    app_handler = logging.FileHandler(config.APPLICATION_LOG_FILE, encoding="utf-8")
    app_handler.setFormatter(formatter)
    root.addHandler(app_handler)

    ocr_logger = logging.getLogger("restaurant_ocr.ocr")
    ocr_handler = logging.FileHandler(config.OCR_LOG_FILE, encoding="utf-8")
    ocr_handler.setFormatter(formatter)
    ocr_logger.addHandler(ocr_handler)


def save_debug_artifacts(image_path: Path, image, words) -> None:
    """Render labeled OCR bounding boxes and the configured template
    regions on top of the processed image and save the result into
    debug/, for calibrating regions against a new invoice template."""
    debug_words = [
        DebugWord(text=w.text, score=w.score, x=w.x, y=w.y, x2=w.x2, y2=w.y2) for w in words
    ]
    overlay = draw_debug_overlay(image, debug_words)

    regions = {
        "HEADER": config.HEADER_REGION,
        "ITEMS": config.ITEMS_REGION,
        "TOTALS": config.TOTALS_REGION,
        "INVOICE_NO": config.INVOICE_NUMBER_REGION,
        "DATE": config.DATE_REGION,
        "TIME": config.TIME_REGION,
        "TABLE": config.TABLE_REGION,
        "CASHIER": config.CASHIER_REGION,
    }
    overlay = draw_region_overlay(overlay, regions)

    debug_path = config.DEBUG_FOLDER / f"{image_path.stem}_debug.png"
    cv2.imwrite(str(debug_path), overlay)
    logger.info("Saved debug overlay to %s", debug_path)


def process_invoice_file(image_path: Path, debug: bool = False) -> Optional[Invoice]:
    """Run the full pipeline (preprocess -> OCR -> parse -> save to Excel)
    for a single invoice image.

    Returns the parsed Invoice, or None if the image could not be read or
    processing raised an error (the error is logged with a full traceback
    either way).
    """
    start = time.perf_counter()
    logger.info("Processing %s", image_path.name)

    raw_image = cv2.imread(str(image_path))
    if raw_image is None:
        logger.error("Could not read image: %s", image_path)
        return None

    try:
        preprocessed = preprocess_image(raw_image)
        oriented_image, words = extract_words(preprocessed)

        if not words:
            logger.warning("No text detected in %s", image_path.name)

        height, width = oriented_image.shape[:2]
        invoice = parse_invoice(words, width, height)

        if debug:
            save_debug_artifacts(image_path, oriented_image, words)

        written = save_invoice(invoice, config.OUTPUT_FILE)
        elapsed = time.perf_counter() - start
        mean_confidence = sum(w.score for w in words) / len(words) if words else 0.0

        logger.info(
            "Finished %s in %.2fs (words=%d, mean_confidence=%.3f, items=%d, written=%s)",
            image_path.name, elapsed, len(words), mean_confidence, len(invoice.items), written,
        )
        return invoice

    except Exception:
        logger.exception("Failed to process %s", image_path.name)
        return None


def move_to_processed(image_path: Path) -> None:
    """Move a handled invoice image into processed/, disambiguating the
    filename if one with the same name is already there."""
    destination = config.PROCESSED_FOLDER / image_path.name
    if destination.exists():
        destination = (
            config.PROCESSED_FOLDER / f"{image_path.stem}_{int(time.time())}{image_path.suffix}"
        )
    shutil.move(str(image_path), str(destination))
    logger.info("Moved %s to %s", image_path.name, destination)


def run_batch(debug: bool = False) -> None:
    """Process every supported image currently in invoices/, in filename
    order, moving each into processed/ once handled."""
    images = sorted(
        p
        for p in config.INVOICE_FOLDER.iterdir()
        if p.is_file() and p.suffix.lower() in config.SUPPORTED_IMAGE_EXTENSIONS
    )

    if not images:
        logger.info("No invoice images found in %s", config.INVOICE_FOLDER)
        return

    processed_count = 0
    failed_count = 0

    for image_path in images:
        invoice = process_invoice_file(image_path, debug=debug)
        if invoice is not None:
            processed_count += 1
        else:
            failed_count += 1
        move_to_processed(image_path)

    logger.info(
        "Batch complete: %d processed, %d failed, %d total",
        processed_count, failed_count, len(images),
    )


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Extract restaurant/cafe invoices into Excel")
    arg_parser.add_argument(
        "--debug",
        action="store_true",
        help="Save annotated OCR/region overlays for each invoice into debug/",
    )
    args = arg_parser.parse_args()

    configure_logging()
    run_batch(debug=args.debug)


if __name__ == "__main__":
    main()
