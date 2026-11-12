"""Image preprocessing for OCR: auto-rotate, deskew, denoise, contrast, resize."""

import base64
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import ExifTags, Image

from config import MAX_IMAGE_HEIGHT, MAX_IMAGE_WIDTH

logger = logging.getLogger(__name__)

_ORIENTATION_TAG = next(
    (tag for tag, name in ExifTags.TAGS.items() if name == "Orientation"), None
)


def load_image(image_path: str) -> np.ndarray:
    """Load an image with OpenCV, preserving colour."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to load image: {image_path}")
    return img


def auto_rotate(img: np.ndarray, image_path: str) -> np.ndarray:
    """Rotate image based on EXIF orientation metadata, if present."""
    if _ORIENTATION_TAG is None:
        return img
    try:
        with Image.open(image_path) as pil_img:
            exif = pil_img.getexif()
            orientation = exif.get(_ORIENTATION_TAG) if exif else None
    except Exception:
        return img

    if orientation == 3:
        return cv2.rotate(img, cv2.ROTATE_180)
    if orientation == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 8:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def deskew(img: np.ndarray) -> np.ndarray:
    """Deskew image using the minimum-area rectangle of foreground pixels."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 50:
        return img

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5 or abs(angle) > 20:
        # Skip tiny (noise) or implausibly large (misdetection) corrections.
        return img

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def denoise(img: np.ndarray) -> np.ndarray:
    """Apply Non-local Means Denoising."""
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """Improve contrast using CLAHE on the L channel of LAB colour space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def resize_if_needed(
    img: np.ndarray, max_width: int = MAX_IMAGE_WIDTH, max_height: int = MAX_IMAGE_HEIGHT
) -> np.ndarray:
    """Downscale large images, preserving aspect ratio. Never upscales."""
    h, w = img.shape[:2]
    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def preprocess_image(image_path: str) -> np.ndarray:
    """Full preprocessing pipeline: rotate, deskew, denoise, contrast, resize."""
    img = load_image(image_path)
    img = auto_rotate(img, image_path)
    img = deskew(img)
    img = denoise(img)
    img = enhance_contrast(img)
    img = resize_if_needed(img)
    return img


def image_to_base64(img: np.ndarray) -> str:
    """Encode an OpenCV image (BGR ndarray) as base64 PNG for Ollama."""
    ok, buffer = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Failed to encode image to PNG")
    return base64.b64encode(buffer).decode("utf-8")


def save_debug_image(
    img: np.ndarray, debug_dir: Path, source_path: str, suffix: str
) -> Optional[Path]:
    """Save an intermediate image to debug/ for troubleshooting. Never raises."""
    try:
        stem = Path(source_path).stem
        dest = debug_dir / f"{stem}_{suffix}.png"
        cv2.imwrite(str(dest), img)
        return dest
    except Exception as e:
        logger.warning(f"Could not write debug image for {source_path}: {e}")
        return None


def save_debug_text(text: str, debug_dir: Path, source_path: str, suffix: str) -> Optional[Path]:
    """Save raw model output to debug/ for troubleshooting. Never raises."""
    try:
        stem = Path(source_path).stem
        dest = debug_dir / f"{stem}_{suffix}.txt"
        dest.write_text(text, encoding="utf-8")
        return dest
    except Exception as e:
        logger.warning(f"Could not write debug text for {source_path}: {e}")
        return None
