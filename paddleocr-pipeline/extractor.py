"""OCR extraction layer.

Wraps PaddleOCR (the PaddleX-based 3.x pipeline API) to produce structured,
word-level results -- text, confidence, bounding box and center coordinates
-- for every line of text on an invoice.

Gross page misorientation (a receipt photographed sideways or upside down)
is corrected by PaddleOCR's own document-orientation classifier
(``use_doc_orientation_classify``) rather than a hand-rolled heuristic: it
runs before text detection, and every bounding box returned is already
relative to the corrected image, which this module returns alongside the
words so callers (parser, debug renderer) stay in the same coordinate space.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import (
    MIN_OCR_CONFIDENCE,
    OCR_DEVICE,
    OCR_ENABLE_MKLDNN,
    OCR_LANGUAGE,
    OCR_USE_DOC_ORIENTATION_CLASSIFY,
    OCR_USE_DOC_UNWARPING,
    OCR_USE_TEXTLINE_ORIENTATION,
)

logger = logging.getLogger("restaurant_ocr.extractor")
ocr_logger = logging.getLogger("restaurant_ocr.ocr")

logging.getLogger("paddleocr").setLevel(logging.WARNING)
logging.getLogger("paddlex").setLevel(logging.WARNING)


@dataclass(frozen=True)
class OCRWord:
    """A single recognized word/text line with its bounding box, in the
    exact shape required by the pipeline's OCR contract."""

    text: str
    score: float
    x: int
    y: int
    x2: int
    y2: int

    @property
    def cx(self) -> int:
        return (self.x + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y + self.y2) // 2

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "score": round(self.score, 4),
            "x": self.x,
            "y": self.y,
            "x2": self.x2,
            "y2": self.y2,
            "cx": self.cx,
            "cy": self.cy,
        }


class InvoiceOCREngine:
    """Reusable wrapper around a single PaddleOCR pipeline instance.

    Loading the underlying models is expensive (multiple seconds), so this
    class should be instantiated once per process and reused for every
    invoice via :func:`get_engine`.
    """

    def __init__(self) -> None:
        from paddleocr import PaddleOCR

        logger.info(
            "Loading PaddleOCR pipeline (lang=%s, device=%s, doc_orientation_classify=%s)",
            OCR_LANGUAGE, OCR_DEVICE, OCR_USE_DOC_ORIENTATION_CLASSIFY,
        )
        self._ocr = PaddleOCR(
            lang=OCR_LANGUAGE,
            device=OCR_DEVICE,
            use_doc_orientation_classify=OCR_USE_DOC_ORIENTATION_CLASSIFY,
            use_doc_unwarping=OCR_USE_DOC_UNWARPING,
            use_textline_orientation=OCR_USE_TEXTLINE_ORIENTATION,
            enable_mkldnn=OCR_ENABLE_MKLDNN,
        )

    def run(self, image: np.ndarray) -> Tuple[np.ndarray, List[OCRWord]]:
        """Run the full OCR pipeline on a single image array.

        Returns the image PaddleOCR actually detected text on (i.e. after
        its own orientation correction, if enabled) together with the
        structured words detected on it, filtering out anything below
        :data:`config.MIN_OCR_CONFIDENCE`.
        """
        if image.ndim == 2:
            # The recognition/orientation models expect a 3-channel image;
            # image_utils hands back grayscale, so restore the channel dim.
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        pages = self._ocr.predict(image)
        if not pages:
            return image, []

        page = pages[0]
        oriented_image = page["doc_preprocessor_res"]["output_img"]

        words: List[OCRWord] = []
        for text, score, box in zip(page["rec_texts"], page["rec_scores"], page["rec_boxes"]):
            score = float(score)
            if score < MIN_OCR_CONFIDENCE:
                continue

            x1, y1, x2, y2 = (int(round(v)) for v in box)
            words.append(
                OCRWord(text=str(text).strip(), score=score, x=x1, y=y1, x2=x2, y2=y2)
            )

        return oriented_image, words


_engine: Optional[InvoiceOCREngine] = None


def get_engine() -> InvoiceOCREngine:
    """Return the process-wide singleton OCR engine, creating it lazily on
    first use so importing this module never triggers a model load."""
    global _engine
    if _engine is None:
        _engine = InvoiceOCREngine()
    return _engine


def extract_words(image: np.ndarray) -> Tuple[np.ndarray, List[OCRWord]]:
    """Extract OCR words from a preprocessed invoice image.

    Returns the (possibly re-oriented) image alongside its OCR words so
    that callers such as the parser and the debug renderer all agree on
    the same coordinate space.
    """
    engine = get_engine()
    start = time.perf_counter()

    oriented_image, words = engine.run(image)

    elapsed = time.perf_counter() - start
    mean_score = sum(w.score for w in words) / len(words) if words else 0.0
    ocr_logger.info(
        "OCR complete: words=%d mean_confidence=%.3f elapsed=%.2fs",
        len(words), mean_score, elapsed,
    )

    return oriented_image, words
