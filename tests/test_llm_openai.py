"""Tests for the official OpenAI PDF-input integration."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.llm.openai_client import OpenAIClient
from src.llm.factory import create_pdf_client


@pytest.fixture
def openai_config(tmp_path):
    path = tmp_path / "config_openai.json"
    path.write_text(
        json.dumps(
            {
                "llm": {
                    "model": "gpt-test",
                    "max_output_tokens": 321,
                    "pdf_detail": "low",
                    "delete_uploaded_files": True,
                }
            }
        )
    )
    return path


def test_missing_api_key_is_rejected(openai_config, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIClient(str(openai_config))


def test_pdf_uses_files_and_responses_apis(openai_config, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    sdk_client = MagicMock()
    sdk_client.files.create.return_value = SimpleNamespace(id="file-test")
    sdk_client.responses.create.return_value = SimpleNamespace(
        output_text="grounded PDF result"
    )

    with patch("openai.OpenAI", return_value=sdk_client):
        client = OpenAIClient(str(openai_config))
        with patch.object(client, "_load_prompt_template", return_value="Read it"):
            result = client.analyze_paper_from_pdf(
                b"%PDF-1.4\nfixture",
                {"title": "Test paper: PDF/API"},
                prompt_type="pdf_smoke_test",
            )

    assert result == "grounded PDF result"
    sdk_client.files.create.assert_called_once_with(
        file=("Test_paper_PDF_API.pdf", b"%PDF-1.4\nfixture", "application/pdf"),
        purpose="user_data",
    )
    sdk_client.responses.create.assert_called_once_with(
        model="gpt-test",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": "file-test", "detail": "low"},
                    {"type": "input_text", "text": "Read it"},
                ],
            }
        ],
        max_output_tokens=321,
    )
    sdk_client.files.delete.assert_called_once_with("file-test")


def test_invalid_pdf_is_rejected_before_upload(openai_config, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    sdk_client = MagicMock()

    with patch("openai.OpenAI", return_value=sdk_client):
        client = OpenAIClient(str(openai_config))
        with pytest.raises(ValueError, match="missing %PDF header"):
            client.analyze_paper_from_pdf(b"plain text", {})

    sdk_client.files.create.assert_not_called()


def test_pdf_factory_accepts_openai_provider(tmp_path, monkeypatch):
    config_path = tmp_path / "provider_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llm": {
                    "pdf_provider": "openai",
                    "pdf_model": "gpt-pdf-test",
                }
            }
        )
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with patch("openai.OpenAI", return_value=MagicMock()):
        client = create_pdf_client(str(config_path))

    assert isinstance(client, OpenAIClient)
    assert client.model_name == "gpt-pdf-test"
