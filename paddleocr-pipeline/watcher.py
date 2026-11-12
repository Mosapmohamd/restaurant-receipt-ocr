"""Folder watcher: automatically process every new invoice image dropped
into invoices/, without needing to manually re-run the CLI batch job.

Usage:
    python watcher.py             # watch invoices/ forever
    python watcher.py --debug     # also save annotated overlays into debug/
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

import config
from main import configure_logging, move_to_processed, process_invoice_file

logger = logging.getLogger("restaurant_ocr.watcher")

# A file being copied/synced into invoices/ can trigger a "created" event
# before it is fully written. Wait for its size to stop changing across a
# few consecutive checks before handing it to the OCR pipeline.
STABLE_CHECK_INTERVAL_SECONDS = 0.5
STABLE_CHECK_ROUNDS = 3


def _wait_until_stable(path: Path) -> bool:
    """Block until ``path``'s size stops changing across consecutive
    checks. Returns False if the file disappears before stabilizing."""
    previous_size = -1
    stable_rounds = 0

    while stable_rounds < STABLE_CHECK_ROUNDS:
        if not path.exists():
            return False
        current_size = path.stat().st_size
        if current_size == previous_size:
            stable_rounds += 1
        else:
            stable_rounds = 0
            previous_size = current_size
        time.sleep(STABLE_CHECK_INTERVAL_SECONDS)

    return True


class InvoiceHandler(FileSystemEventHandler):
    """Reacts to new files appearing in invoices/ by running them through
    the same pipeline used by the CLI batch job (main.process_invoice_file)."""

    def __init__(self, debug: bool = False) -> None:
        self._debug = debug

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._handle(Path(event.src_path))

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._handle(Path(event.dest_path))

    def _handle(self, path: Path) -> None:
        if path.suffix.lower() not in config.SUPPORTED_IMAGE_EXTENSIONS:
            return

        logger.info("Detected new invoice: %s", path.name)
        if not _wait_until_stable(path):
            logger.warning("File disappeared before it finished writing: %s", path.name)
            return

        process_invoice_file(path, debug=self._debug)
        move_to_processed(path)


def watch(debug: bool = False) -> None:
    """Start watching invoices/ forever, processing every new image as it
    arrives. Blocks until interrupted with Ctrl+C."""
    configure_logging()
    handler = InvoiceHandler(debug=debug)
    observer = Observer()
    observer.schedule(handler, str(config.INVOICE_FOLDER), recursive=False)
    observer.start()

    logger.info("Watching %s for new invoices (Ctrl+C to stop)...", config.INVOICE_FOLDER)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping watcher...")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Watch invoices/ and auto-process new invoices")
    arg_parser.add_argument(
        "--debug",
        action="store_true",
        help="Save annotated OCR/region overlays for each invoice into debug/",
    )
    args = arg_parser.parse_args()

    watch(debug=args.debug)
