"""Extract structured invoice data from a receipt image using Ollama.

The prompt below is tailored to the fixed layout produced by this
restaurant's POS: an Arabic thermal receipt with a boxed order number, a
table line, a date/time line, a cashier line, an items table, and a summary
that frequently prints only a net total. See README.md ("How to support new
invoice templates") for guidance on adapting this prompt to a different POS.
"""

import json
import logging
import time
from typing import Any, Dict, Optional

import requests

from config import CONFIDENCE_THRESHOLD, DEBUG_DIR, RETRY_ATTEMPTS, RETRY_BACKOFF_SECONDS, SAVE_DEBUG_ARTIFACTS
from image_utils import image_to_base64, preprocess_image, save_debug_image, save_debug_text
from ollama_client import OllamaClient

logger = logging.getLogger(__name__)

REQUIRED_KEYS = {
    "invoice_number", "date", "time", "table", "cashier",
    "subtotal", "vat", "discount", "grand_total", "items",
}

EXTRACTION_PROMPT = """You are a precision OCR data-extraction engine reading a restaurant/cafe point-of-sale (POS) thermal receipt. Every receipt from this POS uses the SAME fixed layout, printed in Arabic, structured top to bottom like this:

1. A sales header labeled "المبيعات" (Sales) and "طاولة" (Table), with a table number next to "رقم ط" (Table No.).
2. An order number shown inside a boxed field, labeled "رقم الطلب" (Order/Invoice No.).
3. A date line formatted like "2026/07/23" (YYYY/MM/DD) and a time line formatted like "21:1:52" (H:M:S, 24-hour; a component may be missing its leading zero).
4. A cashier line labeled "كاشير" (Cashier), which may contain a number and/or a shift word such as "مسائي" (evening) or "صباحي" (morning) — capture the text exactly as printed.
5. An items table with right-to-left Arabic column headers: "اسم الصنف" (Item Name), "الكمية" (Quantity), "السعر" (Unit Price), "الاجمالي" (Line Total). Each row is one item.
6. A summary section that may show "الاجمالي" (Subtotal) and/or "الصافي" (Net / Grand Total). Many receipts on this POS print ONLY "الصافي" with no separate subtotal/VAT/discount line. In that case set "subtotal" to the sum of the item line totals, and set "vat" and "discount" to 0.
7. A footer print timestamp (when the paper was printed). Do NOT use this as the order date/time — always use the date/time from step 3.

The photo may be blurry, held by fingers, tilted, or show more than one overlapping receipt. Extract only the single receipt that is the clear, in-focus, primary subject of the image. Ignore skin, fingers, background clutter, and any other receipts partially visible behind or beside it.

Return STRICT JSON ONLY. No markdown, no code fences, no explanations, no text before or after the JSON object. Output exactly this shape:

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
    {"name": "", "qty": 0, "unit_price": 0, "total": 0}
  ]
}

Field rules:
- "invoice_number": the order number from the boxed "رقم الطلب" field, exactly as printed (digits/text). Use "" only if truly absent.
- "date": normalized to "YYYY-MM-DD" (zero-padded).
- "time": normalized to 24-hour "HH:MM:SS" (zero-padded). Convert AM/PM if the receipt shows it.
- "table": the table number/label from "رقم ط" (e.g. "5").
- "cashier": the full cashier field text, original language, exactly as printed.
- "subtotal", "vat", "discount", "grand_total": plain numbers, no currency symbols, no thousands separators, "." as the decimal point. "grand_total" is the "الصافي" (net) value.
- "items": one entry per row of the items table, in the printed order. "qty", "unit_price", "total" are numbers, and "total" should equal "qty" * "unit_price" for that row.
- Never invent data. If a field is not visible or legible, use "" for text fields and 0 for numeric fields.
- Return ONLY the JSON object described above. Nothing else."""


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` / ``` ... ``` wrapping if the model added it anyway."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _extract_json_object(text: str) -> str:
    """Return the first balanced {...} substring, tolerating leading/trailing noise."""
    text = _strip_code_fences(text)
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return text[start:]


def _completeness_score(data: Dict[str, Any]) -> float:
    """Heuristic fraction of key fields that are non-empty/non-zero.

    The model does not expose per-token confidence via the Ollama chat API,
    so CONFIDENCE_THRESHOLD is applied against this structural completeness
    signal instead: an extraction with too many blank/zero fields is treated
    as a failed attempt and retried.
    """
    text_fields = ["invoice_number", "date", "time", "cashier"]
    filled = sum(1 for f in text_fields if str(data.get(f, "")).strip())
    filled += 1 if float(data.get("grand_total", 0) or 0) > 0 else 0
    filled += 1 if data.get("items") else 0
    return filled / (len(text_fields) + 2)


def extract_invoice_data(image_path: str, client: OllamaClient) -> Optional[Dict[str, Any]]:
    """Preprocess an image, call Ollama, and return the parsed field dict.

    Retries up to RETRY_ATTEMPTS times on transport errors, invalid JSON,
    missing required keys, or low structural completeness. Returns None if
    every attempt fails.
    """
    logger.info(f"Processing image: {image_path}")
    try:
        img = preprocess_image(image_path)
        if SAVE_DEBUG_ARTIFACTS:
            save_debug_image(img, DEBUG_DIR, image_path, suffix="preprocessed")
        base64_img = image_to_base64(img)
    except Exception as e:
        logger.error(f"Image preprocessing failed for {image_path}: {e}")
        return None

    content = ""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            start = time.perf_counter()
            raw = client.chat(base64_img, EXTRACTION_PROMPT)
            latency = time.perf_counter() - start
            logger.info(f"Model latency for {image_path}: {latency:.2f}s (attempt {attempt})")

            content = (raw.get("message") or {}).get("content", "").strip()
            if SAVE_DEBUG_ARTIFACTS:
                save_debug_text(content, DEBUG_DIR, image_path, suffix=f"response_attempt{attempt}")

            json_text = _extract_json_object(content)
            data = json.loads(json_text)

            if not isinstance(data, dict) or not REQUIRED_KEYS.issubset(data.keys()):
                missing = REQUIRED_KEYS - (data.keys() if isinstance(data, dict) else set())
                logger.warning(f"Missing keys {missing} for {image_path} (attempt {attempt})")
                continue

            score = _completeness_score(data)
            if score < CONFIDENCE_THRESHOLD:
                logger.warning(
                    f"Low-completeness extraction for {image_path} "
                    f"(score={score:.2f} < {CONFIDENCE_THRESHOLD}), attempt {attempt}"
                )
                if attempt == RETRY_ATTEMPTS:
                    return data  # best effort; let downstream validation decide
                continue

            return data

        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error for {image_path} (attempt {attempt}): {e}")
            if attempt == RETRY_ATTEMPTS:
                logger.error(f"Final raw content for {image_path}: {content[:500]}")
        except requests.RequestException as e:
            logger.error(f"Ollama request failed for {image_path} (attempt {attempt}): {e}")
        except Exception as e:
            logger.error(f"Unexpected extraction error for {image_path} (attempt {attempt}): {e}")

        if attempt < RETRY_ATTEMPTS:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    return None
