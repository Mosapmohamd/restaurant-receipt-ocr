# Restaurant Invoice Extraction System

Extracts structured data (order number, date/time, table, cashier, items,
totals) from restaurant/cafe POS receipt photos and appends it to an Excel
workbook — fully offline, using a locally-running [Ollama](https://ollama.com)
vision-language model. No cloud APIs, no PaddleOCR/Tesseract/EasyOCR
packages are used anywhere in this project; OCR is performed entirely by the
VLM through Ollama's chat API.

## ⚠️ Current status: model needs a vision-capable pull

This codebase is complete and tested end-to-end **except for the live model
call**, because the locally pulled `MedAIBase/PaddleOCR-VL:0.9b` currently
has no vision projector wired in. Every image request fails with:

```
500: "image input is not supported - hint: if this is unexpected, you may need to provide the mmproj"
```

This was confirmed by inspecting the model's Ollama manifest
(`~/.ollama/models/manifests/registry.ollama.ai/MedAIBase/PaddleOCR-VL/0.9b`):
it has only `model` / `system` / `params` layers, no `projector` layer, and
`ollama show MedAIBase/PaddleOCR-VL:0.9b` reports `"capabilities":["completion"]`
instead of `["completion","vision"]`. This is a model/Ollama-side gap, not a
bug in this code — see **Troubleshooting** below for how to fix it. Every
other stage of the pipeline (preprocessing, JSON parsing, normalization,
validation, duplicate detection, Excel writing, retries, logging, the
watcher) has been exercised directly against real sample receipts from
`invoices/` and works correctly.

## Installation

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

Requires **Python 3.12+** and a running local Ollama instance with the model
pulled:

```bash
ollama pull MedAIBase/PaddleOCR-VL:0.9b
ollama serve   # if not already running as a service
```

## Usage

```bash
# Process every image currently in invoices/, then exit
python main.py

# Watch invoices/ forever and process new files as they arrive
python watcher.py
```

Drop receipt photos into `invoices/`. Each one is preprocessed, sent to the
model, validated, and:

- appended to `output/invoices.xlsx` and moved to `processed/` on success,
- moved to `processed/` (but **not** appended again) if it's a duplicate,
- moved to `failed/` if extraction or validation ultimately fails.

## Architecture

```
invoices/  ──▶  image_utils.py   ──▶  extractor.py  ──▶  parser.py  ──▶  validator.py  ──▶  excel_writer.py
(raw photos)   (OpenCV cleanup)     (Ollama call,        (dataclass +    (business rules,    (thread-safe,
                                      JSON parsing,        normalization)  format checks)      atomic dup-check
                                      retries)                                                 + append)
                     │
                     ▼
              debug/ (intermediate images + raw model responses, when SAVE_DEBUG_ARTIFACTS=True)

main.py    – one-shot batch runner (ThreadPoolExecutor, tqdm progress bar)
watcher.py – long-running watchdog-based folder monitor, reuses main.process_single_image
config.py  – all settings + centralized logging setup (logs/application.log, logs/errors.log)
```

Each module has one job:

| Module | Responsibility |
|---|---|
| `config.py` | Paths, Ollama settings, tuning knobs, `configure_logging()` |
| `image_utils.py` | OpenCV preprocessing (rotate/deskew/denoise/contrast/resize), base64 encoding, debug artifact saving |
| `ollama_client.py` | Thin `requests.Session` wrapper around `POST /api/chat` |
| `extractor.py` | The extraction prompt, retry loop, JSON-object recovery, completeness heuristic |
| `parser.py` | Raw JSON → `Invoice`/`Item` dataclasses, with normalization (Arabic-Indic digits, date/time formats, subtotal fallback) |
| `validator.py` | Format checks (date/time/numbers) and the `subtotal - discount + vat == grand_total` consistency check |
| `excel_writer.py` | `ExcelWriter` — in-memory workbook, thread-safe atomic duplicate-check-then-append |
| `main.py` | Batch entry point, safe file moves, thread pool orchestration |
| `watcher.py` | Continuous folder monitor with file-stability detection |

## How Ollama works (in this project)

Ollama runs models locally and exposes an HTTP API on `localhost:11434`.
This project talks to exactly one endpoint:

```
POST http://localhost:11434/api/chat
{
  "model": "MedAIBase/PaddleOCR-VL:0.9b",
  "messages": [{"role": "user", "content": "<prompt>", "images": ["<base64 png>"]}],
  "stream": false,
  "format": "json",
  "options": {"temperature": 0, "num_predict": 2048}
}
```

- `"stream": false` — a single complete JSON response body, no chunked
  streaming to parse.
- `"format": "json"` — asks Ollama/llama.cpp to constrain generation to
  valid JSON syntax (on top of the prompt's own JSON-only instructions),
  which meaningfully cuts down on parse failures.
- `"images"` — a list of base64-encoded image bytes with **no** data-URI
  prefix, attached to the user message, per the Ollama chat API's
  multimodal contract.
- `"temperature": 0` — deterministic, repeatable extractions for a fixed
  document layout.

`ollama_client.OllamaClient` wraps this in a reusable `requests.Session`
(connection pooling across every call in a run) with a small connect/read
retry policy at the transport layer. Higher-level retries (bad JSON, missing
fields, HTTP 5xx from the model) are handled in `extractor.py`, with full
logging at each attempt — retrying blindly at the transport layer too would
double the retry count without adding useful signal.

## How PaddleOCR-VL is used

`MedAIBase/PaddleOCR-VL:0.9b` is a vision-language OCR model exposed through
Ollama like any chat model, **not** through the `paddleocr` Python package
(which is explicitly excluded from this project). We never call a
recognition API with bounding boxes; instead we send the whole preprocessed
receipt image plus a single detailed prompt, and the model returns the
already-structured JSON object directly — it does layout understanding and
field extraction in one pass, rather than us gluing together raw OCR tokens.

This only works if the specific Ollama model pull includes the model's
vision projector (`mmproj`) layer — see **Troubleshooting** if you hit the
`"image input is not supported"` error described above.

## Prompt design (why it's tailored, not generic)

The prompt in `extractor.py::EXTRACTION_PROMPT` was written after examining
real sample receipts from this POS system (found in this project's
`invoices/`/`failed/` folders during development), not from a generic
"extract this receipt" template. It encodes the exact fixed layout:

- Arabic field labels and what they mean: `رقم الطلب` (order number, in a
  boxed field), `طاولة`/`رقم ط` (table), `كاشير` (cashier, may include a
  shift word like `مسائي`/`صباحي`), item table headers `اسم الصنف` / `الكمية`
  / `السعر` / `الاجمالي`.
- The POS's native date/time formats (`YYYY/MM/DD`, `H:M:S` with possibly
  missing leading zeros) and instructs the model to normalize them.
- The fact that many receipts on this POS print **only** a net total
  (`الصافي`) with no separate subtotal/VAT/discount line — the model is told
  to fall back to summing item totals for `subtotal` in that case (and
  `parser.py` applies the same fallback defensively if the model doesn't).
- An explicit instruction to distinguish the order date/time from the
  footer print timestamp, and to ignore fingers/background clutter/other
  overlapping receipts in the photo and focus on the single primary receipt
  — real sample photos in this dataset are hand-held, tilted, and sometimes
  show a stack of receipts.
- A hard requirement for strict JSON output with no markdown/fences/prose,
  reinforced by `"format": "json"` at the API level and a fence-stripping
  fallback parser (`extractor._extract_json_object`) in case the model adds
  them anyway.

## Configuration

Everything tunable lives in `config.py` (override via environment variables):

| Setting | Default | Purpose |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434/api/chat` | Ollama chat endpoint |
| `MODEL_NAME` | `MedAIBase/PaddleOCR-VL:0.9b` | Model tag to call |
| `CONFIDENCE_THRESHOLD` | `0.6` | Minimum structural-completeness score (see below) before an extraction is retried instead of accepted |
| `OUTPUT_FILE` | `output/invoices.xlsx` | Excel destination |
| `WATCH_FOLDER` / `INVOICES_DIR` | `invoices/` | Input folder |
| `PROCESSED_FOLDER` / `PROCESSED_DIR` | `processed/` | Successful + duplicate images land here |
| `DEBUG_FOLDER` / `DEBUG_DIR` | `debug/` | Preprocessed images + raw model responses |
| `FAILED_DIR` | `failed/` | Images that never produced a valid, passing invoice |
| `MAX_WORKERS` | `4` | Batch-mode thread pool size |
| `RETRY_ATTEMPTS` | `3` | Extraction retries per image |
| `SAVE_DEBUG_ARTIFACTS` | `True` | Toggle debug image/text dumps |
| `MAX_IMAGE_WIDTH` / `MAX_IMAGE_HEIGHT` | `2000` | Preprocessing resize ceiling |
| `VALIDATION_ABS_TOLERANCE` / `VALIDATION_REL_TOLERANCE` | `0.5` / `0.01` | Tolerance for the totals consistency check |

### `CONFIDENCE_THRESHOLD` — how it's actually used

The Ollama chat API does not return per-field or per-token confidence
scores for this model, so `CONFIDENCE_THRESHOLD` is applied against a
structural completeness heuristic instead
(`extractor._completeness_score`): the fraction of key fields
(`invoice_number`, `date`, `time`, `cashier`, a positive `grand_total`, a
non-empty `items` list) that came back filled in. An extraction below the
threshold is treated as a failed attempt and retried like a JSON parse
error, up to `RETRY_ATTEMPTS`.

## Validation rules

`validator.py` checks, in order:

1. `date` matches canonical `YYYY-MM-DD` (parser normalizes to this first).
2. `time` matches canonical 24-hour `HH:MM:SS`.
3. `subtotal`, `vat`, `discount`, `grand_total` are all non-negative, and
   `grand_total > 0`.
4. Each item's `qty * unit_price ≈ total` (logged as a warning, not a hard
   failure — OCR rounding on the printed line total is common and not worth
   discarding an otherwise-good invoice over).
5. `subtotal - discount + vat == grand_total`, within
   `max(VALIDATION_ABS_TOLERANCE, VALIDATION_REL_TOLERANCE * grand_total)`.

Anything that fails validation is logged to `logs/errors.log` and the source
image is moved to `failed/`.

## Duplicate detection

Handled atomically inside `excel_writer.ExcelWriter.append_invoice`, under a
lock, so two images finishing concurrently can never both pass the check
before either is written (a check-then-act race that a naive
"check separately, then write" split would have):

1. If `invoice_number` is non-empty and already seen, it's a duplicate.
2. Otherwise, a SHA-256 hash of `date|time|grand_total` is computed and
   checked against previously seen hashes.

Duplicates are logged (at ERROR level, so they land in `logs/errors.log` per
spec) and the image is moved to `processed/` without a second Excel row.

## Excel output

`output/invoices.xlsx`, one row per item, columns:

```
Invoice Number | Date | Time | Table | Cashier | Item | Quantity | Unit Price |
Item Total | Subtotal | VAT | Discount | Grand Total | Hash
```

(`Hash` is an internal column used to rebuild the duplicate index when the
workbook is reloaded — it does not need to be treated as invoice data.)

The workbook is kept open in memory for the life of a run (`main.py`
process or `watcher.py` process) instead of being fully re-read from disk on
every single append, which keeps batch/concurrent throughput reasonable —
`.xlsx` has no true streaming-append primitive, so a full read-once /
save-on-each-write strategy is the practical tradeoff between durability
(saved after every invoice, so a crash loses at most the in-flight one) and
speed.

## Logging

- `logs/application.log` — every INFO+ event: images picked up, model
  latency per attempt, successes, retries.
- `logs/errors.log` — every ERROR+ event only: JSON parse failures (final
  attempt), validation failures, duplicate invoices, Excel write failures,
  unexpected exceptions. This works because `config.configure_logging()`
  attaches both handlers to the **root** logger once, so any module calling
  `logger.error(...)` anywhere automatically reaches both files — no need
  to log to two places manually.

## Error recovery

- Invalid/incomplete JSON, low-completeness extractions, and transient
  Ollama errors are retried up to `RETRY_ATTEMPTS` times (default 3) with a
  short backoff, all logged per attempt.
- If every attempt fails, the source image is moved to `failed/` and
  processing continues with the next image — one bad receipt never stops a
  batch run or the watcher.
- File moves use `main.move_file_safe`, which disambiguates with a
  timestamp suffix if a same-named file already exists at the destination
  (plain `Path.rename`/`shutil.move` would otherwise raise on Windows
  instead of overwriting, unlike POSIX).

## Performance tuning

- **HTTP session reuse**: `OllamaClient` holds one `requests.Session` for
  its whole lifetime (connection pooling), shared across all workers in a
  batch run.
- **Batch processing**: `python main.py` processes every image currently in
  `invoices/` with a `ThreadPoolExecutor(max_workers=MAX_WORKERS)` and a
  `tqdm` progress bar.
- **Concurrency**: raise `MAX_WORKERS` for more parallel Ollama calls (bound
  by how many concurrent requests your Ollama instance/GPU can actually
  serve — check `ollama ps` for VRAM headroom). The Excel write path is
  serialized via a lock regardless of `MAX_WORKERS`, since it's a shared
  single-file resource; extraction (the slow, model-bound part) is where
  concurrency actually helps.
- **Image size**: `MAX_IMAGE_WIDTH`/`MAX_IMAGE_HEIGHT` (default 2000px)
  trade off legibility of small printed text against payload size/latency.
  Increase for very dense receipts, decrease if latency is the bottleneck.
- **Debug artifacts**: set `SAVE_DEBUG_ARTIFACTS=0` in the environment to
  skip writing preprocessed images and raw responses to `debug/` once a
  template is stable — it's pure I/O overhead once you no longer need it
  for prompt troubleshooting.

## Troubleshooting

**`"image input is not supported - hint: ... provide the mmproj"` /
every extraction fails with HTTP 500** — the pulled model doesn't have a
vision projector registered in Ollama. Check:

```bash
ollama show MedAIBase/PaddleOCR-VL:0.9b
# capabilities should include "vision"; if it only shows "completion", the
# projector is missing.
```

Fixes to try, roughly in order:
1. Update Ollama to the latest version (`ollama --version`) — support for
   newer/less common vision architectures like PaddleOCR-VL is still
   evolving in Ollama's backend.
2. Remove and re-pull the model (`ollama rm MedAIBase/PaddleOCR-VL:0.9b &&
   ollama pull MedAIBase/PaddleOCR-VL:0.9b`) in case the original pull was
   incomplete.
3. Check the Ollama library page for this model for a tag that explicitly
   bundles the projector, or an updated version from the publisher.
4. As a temporary stand-in while sorting the above out, point `MODEL_NAME`
   in `config.py` (or the `MODEL_NAME` environment variable) at another
   locally-pulled vision-capable Ollama model to validate the rest of the
   pipeline end-to-end.

**JSON parse errors in `logs/errors.log`** — check the corresponding
`debug/<image>_response_attempt*.txt` file (requires
`SAVE_DEBUG_ARTIFACTS=1`, the default) to see exactly what the model
returned; usually either truncation (raise `num_predict` in
`ollama_client.py`) or the model wrapping the JSON in prose despite
instructions (the fence-stripping fallback in `extractor._extract_json_object`
already handles simple cases, but wildly different wrapping may need a
prompt tweak).

**Validation failures on otherwise-correct receipts** — check
`debug/<image>_preprocessed.png` to confirm the image is actually legible
after preprocessing, then check the corresponding raw model response. If
the model is consistently off by a fixed small amount on totals, widen
`VALIDATION_ABS_TOLERANCE`/`VALIDATION_REL_TOLERANCE` in `config.py` rather
than disabling the check.

**Watcher processes a half-copied file** — `watcher.py` already waits for a
file's size to stabilize (`WATCHER_STABLE_CHECKS` consecutive unchanged
reads, `WATCHER_STABLE_CHECK_INTERVAL` seconds apart) before handing it to
the pipeline. If you're copying very large files over a slow network share,
increase `WATCHER_STABLE_CHECKS`.

## How to support a new invoice template

This system is currently tuned for one fixed Arabic POS layout. To adapt it
to a different POS template:

1. Collect a handful of real sample receipts from the new POS into a scratch
   folder and inspect them (field labels, language, date/time formats,
   whether subtotal/VAT/discount are printed separately or only a net
   total).
2. Update `extractor.EXTRACTION_PROMPT` with the new layout's structure and
   field-label glossary, following the same pattern as the current prompt:
   describe the fields top-to-bottom, name the literal label text for each
   field, and call out any layout quirks (multiple totals, footer
   timestamps that aren't the order time, etc.).
3. If the new POS's date/time formats aren't already covered, extend
   `parser._DATE_FORMATS` and/or `parser.normalize_time`.
4. If numeric fields ever contain a currency symbol or thousands-separator
   style not already stripped, extend `parser.normalize_number`.
5. Re-run the sanity checks: preprocess a real sample with
   `image_utils.preprocess_image` and inspect the `debug/` output, then run
   a few images through `main.py` and check `debug/*_response_attempt*.txt`
   against `logs/errors.log` to see where extraction or validation disagrees
   with the receipt.
6. If several distinct POS templates need to coexist, the cleanest
   extension is a small template-detection step before calling
   `extract_invoice_data` that selects between multiple prompts — the
   current single-prompt design intentionally keeps things simple for the
   fixed-template case described in the project brief.

## Project structure

```
vlm-pipeline/  (this directory)
    invoices/       # drop new receipt images here
    processed/      # successfully processed (or duplicate) images
    failed/         # images that never produced a valid invoice
    output/         # invoices.xlsx
    debug/          # preprocessed images + raw model responses
    logs/           # application.log, errors.log

    config.py
    ollama_client.py
    extractor.py
    parser.py
    validator.py
    excel_writer.py
    image_utils.py
    main.py
    watcher.py

    requirements.txt
    README.md
```
