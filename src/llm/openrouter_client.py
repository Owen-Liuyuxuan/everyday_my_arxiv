"""
OpenRouter API client using direct HTTP (requests), matching the chat/completions JSON API.

No OpenAI SDK: same pattern as a raw POST to
https://openrouter.ai/api/v1/chat/completions with Bearer auth and JSON body.
"""
import base64
import json
import os
from typing import Any, Dict, List, Optional

import requests

from src.llm.base import BaseLLMClient
from src.llm.openai_client import OpenAIClient


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(OpenAIClient):
    """
    LLM client for OpenRouter via HTTP POST (see try_openrouter.py).

    API key resolution order:
    1. OPENROUTER_API_KEY
    2. OPEN_ROUTE (compat with local scripts)
    3. config openrouter_api_key
    4. config api_key

    Optional config keys:
    - openrouter_http_referer -> HTTP-Referer header
    - openrouter_x_title -> X-Title header
    - openrouter_pdf_plugins: JSON array (list) passed as top-level "plugins" for PDF calls
    - openrouter_pdf_engine: if set, builds file-parser plugin
      [{"id": "file-parser", "pdf": {"engine": "<value>"}}]
    """

    def __init__(self, config_path: str = "config/config.json"):
        BaseLLMClient.__init__(self, config_path)

        self._api_key = (
            os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPEN_ROUTE")
            or self.config.get("openrouter_api_key")
            or self.config.get("api_key")
        )
        if not self._api_key:
            raise ValueError(
                "Set OPENROUTER_API_KEY or OPEN_ROUTE, or set 'openrouter_api_key' / "
                "'api_key' in config llm section"
            )

        self._base_url = self.config.get("base_url", DEFAULT_OPENROUTER_BASE_URL).rstrip("/")
        self._chat_url = f"{self._base_url}/chat/completions"

        self._extra_headers: Dict[str, str] = {}
        referer = self.config.get("openrouter_http_referer")
        if referer:
            self._extra_headers["HTTP-Referer"] = referer
        title = self.config.get("openrouter_x_title")
        if title:
            self._extra_headers["X-Title"] = title

        self.model_name = self.config.get("model", "openai/gpt-4o-mini")

    def _request_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self._extra_headers)
        return headers

    def _post_chat_completions(self, payload: Dict[str, Any]) -> str:
        response = requests.post(
            self._chat_url,
            headers=self._request_headers(),
            json=payload,
            timeout=300,
        )
        try:
            data = response.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"OpenRouter returned non-JSON (HTTP {response.status_code}): {response.text[:500]}"
            ) from e

        if response.status_code >= 400:
            err = data.get("error") if isinstance(data, dict) else None
            detail = err if err is not None else data
            raise RuntimeError(
                f"OpenRouter API error HTTP {response.status_code}: {detail}"
            )

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenRouter response missing choices: {data!r}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        # Some models return a list of content parts
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text") or "")
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)

    def _call_api(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        elif self.temperature is not None:
            payload["temperature"] = self.temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        elif self.max_output_tokens:
            payload["max_tokens"] = self.max_output_tokens
        return self._post_chat_completions(payload)

    def _openrouter_pdf_plugins(self) -> Optional[List[Dict[str, Any]]]:
        raw = self.config.get("openrouter_pdf_plugins")
        if raw is not None:
            if isinstance(raw, str):
                return json.loads(raw)
            if isinstance(raw, list):
                return raw
        engine = self.config.get("openrouter_pdf_engine")
        if engine:
            return [{"id": "file-parser", "pdf": {"engine": engine}}]
        return None

    def analyze_paper_from_pdf(
        self,
        pdf_data: bytes,
        paper_metadata: Dict,
        prompt_type: str = "summary",
    ) -> str:
        prompt_template = self._load_prompt_template(prompt_type)
        prompt = prompt_template

        b64 = base64.standard_b64encode(pdf_data).decode("ascii")
        data_url = f"data:application/pdf;base64,{b64}"
        filename = "paper.pdf"
        title = paper_metadata.get("title")
        if isinstance(title, str) and title.strip():
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in title[:80])
            if safe:
                filename = f"{safe}.pdf"

        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "file",
                        "file": {
                            "filename": filename,
                            "file_data": data_url,
                        },
                    },
                ],
            }
        ]

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_output_tokens:
            payload["max_tokens"] = self.max_output_tokens

        plugins = self._openrouter_pdf_plugins()
        if plugins:
            payload["plugins"] = plugins

        return self._post_chat_completions(payload)
