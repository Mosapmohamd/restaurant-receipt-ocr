"""Continuously watch invoices/ and process every new receipt image as it lands.

Usage:
    python watcher.py
"""

import logging
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from config import INVOICES_DIR, IMAGE_EXTENSIONS, WATCHER_STABLE_CHECKS, WATCHER_STABLE_CHECK_INTERVAL, configure_logging
from excel_writer import get_excel_writer
from main import process_single_image
from ollama_client import OllamaClient

configure_logging()
logger = logging.getLogger(__name__)


def _wait_until_stable(path: Path) -> bool:
    """Poll file size until it stops changing, so we don't read a
    partially-copied image. Returns False if the file disappeared."""
    last_size = -1
    stable_count = 0
    while stable_count < WATCHER_STABLE_CHECKS:
        if not path.exists():
            return False
        try:
            size = path.stat().st_size
        except OSError:
            return False
        if size == last_size and size > 0:
            stable_count += 1
        else:
            stable_count = 0
        last_size = size
        time.sleep(WATCHER_STABLE_CHECK_INTERVAL)
    return True


class InvoiceHandler(FileSystemEventHandler):
    """Handles new-file events in invoices/."""

    def __init__(self, client: OllamaClient) -> None:
        self.client = client
        self.writer = get_excel_writer()
        self._processing: set[str] = set()

    def _handle(self, path: Path) -> None:
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            return
        key = str(path)
        if key in self._processing:
            return
        self._processing.add(key)
        try:
            if not _wait_until_stable(path):
                logger.warning(f"File disappeared before it stabilized: {path.name}")
                return
            logger.info(f"Watcher detected new file: {path.name}")
            process_single_image(path, self.client, self.writer)
        except Exception as e:
            logger.error(f"Error processing {path.name}: {e}")
        finally:
            self._processing.discard(key)

    def on_created(self, event) -> None:
        if event.is_directory:
            return
        self._handle(Path(event.src_path))

    def on_moved(self, event) -> None:
        if event.is_directory:
            return
        self._handle(Path(event.dest_path))


def main() -> None:
    """Run the watcher until interrupted."""
    client = OllamaClient()
    event_handler = InvoiceHandler(client)
    observer = Observer()
    observer.schedule(event_handler, str(INVOICES_DIR), recursive=False)
    observer.start()
    logger.info(f"Watcher started. Monitoring {INVOICES_DIR} for new images.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Watcher stopping (KeyboardInterrupt).")
        observer.stop()
    observer.join()
    client.close()


if __name__ == "__main__":
    main()
