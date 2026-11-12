"""Image preprocessing utilities for the invoice OCR pipeline.

Every invoice image goes through :func:`preprocess_image` before it is
handed to the OCR engine. The pipeline is deliberately deterministic and
geometry-based (no OCR calls here) so it stays fast and side-effect free:

    1. standardize resolution (resize to a fixed width, upscale low-res
       photos)
    2. grayscale conversion
    3. contrast enhancement (CLAHE)
    4. denoising
    5. deskewing (small-angle rotation correction)

Gross 90/180/270 degree rotation correction is handled in ``extractor.py``
instead, by PaddleOCR's own trained document-orientation classifier, which
runs as part of the OCR pipeline itself -- ``image_utils`` intentionally has
no OCR dependency and only ever fixes small in-plane skew.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence

import cv2
import numpy as np

from config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    DENOISE_STRENGTH,
    MAX_DESKEW_ANGLE,
    MIN_ACCEPTABLE_WIDTH,
    TARGET_WIDTH,
)

logger = logging.getLogger("restaurant_ocr.image_utils")


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate ``image`` clockwise by ``angle`` degrees, expanding the canvas
    so no content is cropped."""
    if angle % 360 == 0:
        return image

    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    return cv2.warpAffine(
        image,
        rotation_matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def standardize_resolution(image: np.ndarray) -> np.ndarray:
    """Resize ``image`` to :data:`config.TARGET_WIDTH`, preserving aspect
    ratio, so that every downstream pixel-based template region means the
    same thing regardless of source resolution. Upscaling uses cubic
    interpolation (helps low-resolution photos); downscaling uses area
    interpolation (reduces aliasing)."""
    height, width = image.shape[:2]
    if width == 0:
        raise ValueError("Cannot resize an image with zero width")

    if width == TARGET_WIDTH:
        return image

    scale = TARGET_WIDTH / width
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA

    if width < MIN_ACCEPTABLE_WIDTH:
        logger.info("Upscaling low-resolution image from width=%d to width=%d", width, TARGET_WIDTH)

    return cv2.resize(image, (TARGET_WIDTH, new_height), interpolation=interpolation)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR or already-grayscale image to single-channel
    grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) so
    faint thermal-printer text becomes easier to binarize and recognize."""
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    return clahe.apply(gray)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Remove speckle/salt-and-pepper noise typical of scanned or
    photographed thermal receipts, without blurring text edges too much."""
    return cv2.fastNlMeansDenoising(gray, h=DENOISE_STRENGTH)


def _estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate the small rotation angle (degrees) needed to make text
    lines horizontal, using the minimum-area bounding rectangle of
    foreground pixels."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    binary = cv2.dilate(binary, np.ones((3, 25), np.uint8), iterations=1)

    coords = cv2.findNonZero(binary)
    if coords is None:
        return 0.0

    angle = cv2.minAreaRect(coords)[-1]

    # cv2.minAreaRect returns angles in (-90, 0]; normalize to a signed
    # deviation from horizontal in the range (-45, 45].
    if angle < -45:
        angle = 90 + angle
    return angle


def deskew(gray: np.ndarray) -> np.ndarray:
    """Correct small skew (a few degrees, from an imperfectly aligned
    scan/photo). Large deviations are left untouched here; those are
    handled as gross rotation in the OCR-confidence-based step."""
    angle = _estimate_skew_angle(gray)

    if abs(angle) < 0.1 or abs(angle) > MAX_DESKEW_ANGLE:
        return gray

    logger.debug("Deskewing image by %.2f degrees", angle)
    return rotate_image(gray, angle)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Run the full deterministic preprocessing pipeline on a raw BGR/gray
    image loaded from disk and return a clean grayscale image ready for
    OCR (gross rotation correction happens separately, inside PaddleOCR's
    own pipeline in ``extractor.InvoiceOCREngine``)."""
    resized = standardize_resolution(image)
    gray = to_grayscale(resized)
    contrasted = enhance_contrast(gray)
    denoised = denoise(contrasted)
    deskewed = deskew(denoised)
    return deskewed


@dataclass(frozen=True)
class DebugWord:
    """Minimal shape needed to render a debug overlay; kept decoupled from
    ``extractor.OCRWord`` so this module has no OCR dependency."""

    text: str
    score: float
    x: int
    y: int
    x2: int
    y2: int


def draw_debug_overlay(image: np.ndarray, words: Sequence[DebugWord]) -> np.ndarray:
    """Draw a bounding box and label for every detected word on top of
    ``image`` and return the annotated copy (used by debug mode)."""
    if len(image.shape) == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()

    for word in words:
        color = (0, 200, 0) if word.score >= 0.8 else (0, 140, 255)
        cv2.rectangle(overlay, (word.x, word.y), (word.x2, word.y2), color, 1)

        label = f"{word.text} ({word.score:.2f})"
        text_y = max(0, word.y - 4)
        cv2.putText(
            overlay,
            label,
            (word.x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return overlay


def draw_region_overlay(image: np.ndarray, regions: dict) -> np.ndarray:
    """Draw named template regions (from config.py) on top of ``image``,
    useful when calibrating a new invoice template."""
    if len(image.shape) == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()

    height, width = overlay.shape[:2]
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 255), (0, 128, 255),
    ]

    for index, (name, region) in enumerate(regions.items()):
        x1, y1, x2, y2 = region.bounds(width, height)
        color = palette[index % len(palette)]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            overlay,
            name,
            (x1 + 2, max(12, y1 + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return overlay
