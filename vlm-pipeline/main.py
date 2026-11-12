"""Batch-process every invoice image currently in invoices/.

Usage:
    python main.py
"""

import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from config import (
    FAILED_DIR,
    IMAGE_EXTENSIONS,
    INVOICES_DIR,
    MAX_WORKERS,
    PROCESSED_DIR,
    configure_logging,
)
from excel_writer import ExcelWriter, get_excel_writer
from extractor import extract_invoice_data
from ollama_client import OllamaClient
from parser import Invoice
from validator import validate_invoice

configure_logging()
logger = logging.getLogger(__name__)


def move_file_safe(src: Path, dest_dir: Path) -> Path:
    """Move src into dest_dir, disambiguating the filename if one already
    exists there (Windows raises rather than silently overwriting)."""
    dest = dest_dir / src.name
    if dest.exists():
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        dest = dest_dir / f"{src.stem}_{timestamp}{src.suffix}"
    shutil.move(str(src), str(dest))
    return dest


def process_single_image(image_path: Path, client: OllamaClient, writer: ExcelWriter) -> bool:
    """Extract, validate, append, and move one image. Returns True on success."""
    start = time.perf_counter()
    logger.info(f"Processing {image_path.name}")

    data = extract_invoice_data(str(image_path), client)
    if data is None:
        logger.error(f"Extraction failed after retries for {image_path.name}")
        move_file_safe(image_path, FAILED_DIR)
        return False

    try:
        inv = Invoice.from_dict(data)
    except Exception as e:
        logger.error(f"Parsing failed for {image_path.name}: {e}")
        move_file_safe(image_path, FAILED_DIR)
        return False

    is_valid, error_msg = validate_invoice(inv)
    if not is_valid:
        logger.error(f"Validation failed for {image_path.name}: {error_msg}")
        move_file_safe(image_path, FAILED_DIR)
        return False

    try:
        outcome = writer.append_invoice(inv)
    except Exception as e:
        logger.error(f"Excel write failed for {image_path.name}: {e}")
        move_file_safe(image_path, FAILED_DIR)
        return False

    if outcome == "duplicate":
        logger.error(
            f"Duplicate invoice skipped for {image_path.name}: "
            f"invoice_number={inv.invoice_number!r}, date={inv.date}, time={inv.time}"
        )
        move_file_safe(image_path, PROCESSED_DIR)
        return False

    move_file_safe(image_path, PROCESSED_DIR)
    elapsed = time.perf_counter() - start
    logger.info(f"Successfully processed {image_path.name} in {elapsed:.2f}s")
    return True


def main() -> None:
    """Process all images currently sitting in invoices/."""
    image_files = sorted(f for f in INVOICES_DIR.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_files:
        logger.info("No images found in invoices/.")
        return

    logger.info(f"Found {len(image_files)} images to process.")

    client = OllamaClient()
    writer = get_excel_writer()
    success_count = 0
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_single_image, img, client, writer): img
                for img in image_files
            }
            with tqdm(total=len(futures), desc="Processing invoices") as pbar:
                for future in as_completed(futures):
                    img = futures[future]
                    try:
                        if future.result():
                            success_count += 1
                    except Exception as e:
                        logger.error(f"Unexpected error processing {img.name}: {e}")
                    pbar.update(1)
    finally:
        client.close()

    logger.info(
        f"Batch processing complete. Success: {success_count}, "
        f"Failed/duplicate: {len(image_files) - success_count}"
    )


if __name__ == "__main__":
    main()
