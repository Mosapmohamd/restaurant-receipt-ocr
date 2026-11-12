"""Thin, reusable client for the local Ollama chat API.

Communicates exclusively with ``POST {OLLAMA_URL}`` (default
``http://localhost:11434/api/chat``) using ``stream: false`` and a JSON
response body, per the Ollama API contract. No other network calls are made
anywhere in this project.
"""

import logging
from typing import Any, Dict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import MODEL_NAME, OLLAMA_TEMPERATURE, OLLAMA_TIMEOUT_SECONDS, OLLAMA_URL

logger = logging.getLogger(__name__)


class OllamaClient:
    """Reusable HTTP session for talking to a local Ollama server."""

    def __init__(
        self,
        url: str = OLLAMA_URL,
        model: str = MODEL_NAME,
        timeout: int = OLLAMA_TIMEOUT_SECONDS,
    ) -> None:
        self.url = url
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        # Only retry at the connection/read level (server unreachable, reset,
        # etc.). Retries for bad model output (invalid JSON, HTTP 500 from the
        # model) are handled with full logging in extractor.py, so we do not
        # double-retry on status codes here.
        adapter = HTTPAdapter(
            max_retries=Retry(total=2, connect=2, read=0, status=0, backoff_factor=0.5)
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def chat(self, image_base64: str, prompt: str) -> Dict[str, Any]:
        """Send one image + prompt to Ollama and return the parsed JSON body.

        Raises ``requests.RequestException`` on transport/HTTP errors.
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_base64],
                }
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": OLLAMA_TEMPERATURE,
                "num_predict": 2048,
            },
        }
        try:
            resp = self.session.post(self.url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            body = e.response.text if e.response is not None else ""
            logger.error(f"Ollama HTTP {e.response.status_code if e.response is not None else '?'}: {body}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            raise

    def close(self) -> None:
        self.session.close()
