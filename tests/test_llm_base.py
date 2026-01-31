"""
Tests for the BaseLLMClient abstract class.
"""
import json
import os
import pytest
import tempfile

from src.llm.base import BaseLLMClient


class MockLLMClient(BaseLLMClient):
    """Mock implementation of BaseLLMClient for testing."""
    
    def analyze_paper_from_pdf(self, pdf_data, paper_metadata, prompt_type="summary"):
        return f"Mock PDF analysis for {paper_metadata['title']}"
    
    def analyze_paper_from_abstract(self, paper, prompt_type="abstract_analysis"):
        return f"Mock abstract analysis for {paper['title']}"
    
    def generate_report_summary(self, papers, report_type="daily"):
        return f"Mock summary for {len(papers)} papers"
    
    def translate_content(self, content, target_language):
        return f"Mock translation to {target_language}"
    
    def _score_single_paper(self, paper, keywords, negative_keywords=None):
        return {
            "relevance_score": 2,
            "significance_score": 2,
            "combined_score": 4
        }


@pytest.fixture
def temp_config():
    """Create a temporary config file for testing."""
    config = {
        "llm": {
            "temperature": 0.3,
            "max_output_tokens": 2048,
            "summary_length": "short",
            "batch_size": 8
        }
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def mock_client(temp_config):
    """Create a mock LLM client instance."""
    return MockLLMClient(temp_config)


class TestBaseLLMClient:
    """Tests for BaseLLMClient base functionality."""
    
    def test_config_loading(self, mock_client):
        """Test that configuration is loaded correctly."""
        assert mock_client.temperature == 0.3
        assert mock_client.max_output_tokens == 2048
        assert mock_client.summary_length == "short"
        assert mock_client.batch_size == 8
    
    def test_config_defaults(self, temp_config):
        """Test that default values are used when not specified."""
        # Create config without some values
        config = {"llm": {}}
        with open(temp_config, 'w') as f:
            json.dump(config, f)
        
        client = MockLLMClient(temp_config)
        assert client.temperature == 0.2  # default
        assert client.max_output_tokens == 4096  # default
        assert client.batch_size == 16  # default
    
    def test_parse_json_response_valid(self, mock_client):
        """Test JSON parsing from LLM response."""
        response = 'Some text {"relevance_score": 3, "significance_score": 2, "combined_score": 5} more text'
        result = mock_client._parse_json_response(response)
        
        assert result["relevance_score"] == 3
        assert result["significance_score"] == 2
        assert result["combined_score"] == 5
    
    def test_parse_json_response_invalid(self, mock_client):
        """Test JSON parsing fallback on invalid response."""
        # Use malformed JSON that will cause JSONDecodeError
        response = "This has braces {but invalid json: syntax}"
        result = mock_client._parse_json_response(response)
        
        # Should return default scores
        assert result["relevance_score"] == 1
        assert result["significance_score"] == 1
        assert result["combined_score"] == 2
    
    def test_parse_json_response_no_json(self, mock_client):
        """Test JSON parsing when no JSON object is found."""
        response = "Just plain text without any braces"
        result = mock_client._parse_json_response(response)
        
        assert "error" in result
    
    def test_batch_score_papers(self, mock_client):
        """Test batch scoring of papers."""
        papers = [
            {"title": f"Paper {i}", "authors": ["Author"], "abstract": "Abstract",
             "categories": ["cs.CV"], "published_date": "2024-01-01"}
            for i in range(5)
        ]
        
        scored = mock_client.batch_score_papers(papers, keywords=["test"])
        
        assert len(scored) == 5
        for paper in scored:
            assert "relevance_score" in paper
            assert "significance_score" in paper
            assert "combined_score" in paper
            assert paper["combined_score"] == 4
    
    def test_batch_score_papers_batching(self, mock_client):
        """Test that batch scoring respects batch_size."""
        # Create more papers than batch_size
        papers = [
            {"title": f"Paper {i}", "authors": ["Author"], "abstract": "Abstract",
             "categories": ["cs.CV"], "published_date": "2024-01-01"}
            for i in range(20)  # batch_size is 8, so 3 batches
        ]
        
        scored = mock_client.batch_score_papers(papers, keywords=["test"])
        assert len(scored) == 20


class TestPromptTemplateLoading:
    """Tests for prompt template loading."""
    
    def test_load_prompt_template(self, mock_client):
        """Test loading a prompt template file."""
        # This test requires the actual prompt files to exist
        # Skip if they don't exist
        prompt_path = "src/llm/prompts/relevance_scoring.txt"
        if not os.path.exists(prompt_path):
            pytest.skip("Prompt template file not found")
        
        template = mock_client._load_prompt_template("relevance_scoring")
        assert len(template) > 0
        assert "{title}" in template or "{abstract}" in template

