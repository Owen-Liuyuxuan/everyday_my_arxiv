"""
OpenRouter API client using direct HTTP (requests), matching the chat/completions JSON API.

Uses requests only (no OpenAI SDK). POST to e.g.
https://openrouter.ai/api/v1/chat/completions with Bearer auth and JSON body.

Inherits BaseLLMClient only so importing this module does not load the OpenAI package.
"""
import base64
import json
import os
from typing import Any, Dict, List, Optional

import requests

from src.llm.base import BaseLLMClient


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(BaseLLMClient):
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
        super().__init__(config_path)

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

    def _format_pdf_prompt(self, prompt_template: str, paper_metadata: Dict) -> str:
        """Fill template placeholders like Ark PDF flow (title, authors, abstract, length)."""
        authors = paper_metadata.get("authors") or []
        if isinstance(authors, str):
            authors_str = authors
        else:
            authors_str = ", ".join(authors)
        return prompt_template.format(
            title=paper_metadata.get("title", ""),
            authors=authors_str,
            abstract=paper_metadata.get("abstract", ""),
            summary_length=self.summary_length,
        )

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
        prompt = self._format_pdf_prompt(prompt_template, paper_metadata)

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

    def analyze_paper_from_abstract(
        self, paper: Dict, prompt_type: str = "abstract_analysis"
    ) -> str:
        prompt_template = self._load_prompt_template(prompt_type)
        prompt = prompt_template.format(
            title=paper["title"],
            authors=", ".join(paper["authors"]),
            abstract=paper["abstract"],
            categories=", ".join(paper["categories"]),
            published_date=paper["published_date"],
        )
        return self._call_api(prompt)

    def generate_report_summary(self, papers: List[Dict], report_type: str = "daily") -> str:
        prompt_template = self._load_prompt_template("report_summary")
        paper_info = []
        for i, paper in enumerate(papers, 1):
            paper_info.append(f"{i}. \"{paper['title']}\" by {paper['formatted_authors']}")
        paper_list = "\n".join(paper_info)
        prompt = prompt_template.format(
            report_type=report_type,
            paper_count=len(papers),
            paper_list=paper_list,
            date=papers[0]["published_date"] if papers else "today",
        )
        return self._call_api(prompt)

    def translate_content(self, content: str, target_language: str) -> str:
        prompt_template = self._load_prompt_template("translate")
        prompt = prompt_template.format(
            content=content,
            target_language=target_language,
        )
        return self._call_api(prompt, temperature=0.1)

    def _format_author_preferences(self, author_preferences: Optional[Dict]) -> str:
        if not author_preferences:
            return "No specific author preferences"
        formatted = []
        for category, values in author_preferences.items():
            if values:
                formatted.append(f"{category}: {', '.join(values)}")
        return "; ".join(formatted) if formatted else "No specific author preferences"

    def _score_single_paper(
        self,
        paper: Dict,
        keywords: List[str],
        negative_keywords: Optional[List[str]] = None,
        author_preferences: Optional[Dict] = None,
    ) -> Dict:
        prompt_template = self._load_prompt_template("relevance_scoring")
        prompt = prompt_template.format(
            title=paper["title"],
            authors=", ".join(paper["authors"]),
            abstract=paper["abstract"],
            categories=", ".join(paper.get("categories", [])),
            published_date=paper.get("published_date", "N/A"),
            venue=paper.get("venue", "N/A"),
            code_url=paper.get("code_url", "N/A"),
            keywords=", ".join(keywords),
            negative_keywords=", ".join(negative_keywords or []),
            author_preferences=self._format_author_preferences(author_preferences),
        )
        response_text = self._call_api(prompt, temperature=0.05, max_tokens=1024)
        return self._parse_json_response(response_text)
