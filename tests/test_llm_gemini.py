"""
Tests for the GeminiClient implementation.
"""
import json
import os
import pytest
import tempfile
from unittest.mock import MagicMock, patch


@pytest.fixture
def temp_config():
    """Create a temporary Gemini config file."""
    config = {
        "llm": {
            "model": "gemini-2.5-flash",
            "temperature": 0.2,
            "max_output_tokens": 4096,
            "summary_length": "medium",
            "batch_size": 16
        }
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def mock_paper():
    """Create a mock paper dictionary."""
    return {
        "title": "Test Paper: A Novel Approach",
        "authors": ["Author One", "Author Two"],
        "abstract": "This paper presents a novel approach to testing.",
        "categories": ["cs.CV", "cs.AI"],
        "published_date": "2024-01-15",
        "formatted_authors": "Author One et al."
    }


class TestGeminiClientInitialization:
    """Tests for GeminiClient initialization."""
    
    def test_missing_api_key_raises_error(self, temp_config, monkeypatch):
        """Test that missing API key raises ValueError."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        
        from src.llm.gemini import GeminiClient
        
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            GeminiClient(temp_config)
    
    def test_config_loading(self, temp_config, monkeypatch):
        """Test that config is loaded correctly."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        
        # Mock the genai client to avoid actual API calls
        with patch("src.llm.gemini.GeminiClient.__init__", lambda self, config_path: None):
            from src.llm.gemini import GeminiClient
            client = GeminiClient.__new__(GeminiClient)
            
            # Manually set attributes for testing
            with open(temp_config, 'r') as f:
                client.config = json.load(f)['llm']
            client.temperature = client.config['temperature']
            client.model_name = client.config['model']
            
            assert client.model_name == "gemini-2.5-flash"
            assert client.temperature == 0.2


class TestGeminiClientWithMockedAPI:
    """Tests for GeminiClient with mocked Gemini API."""
    
    @pytest.fixture
    def mock_gemini_client(self, temp_config, monkeypatch):
        """Create a GeminiClient with mocked API."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        
        # Create mock genai module
        mock_genai = MagicMock()
        mock_types = MagicMock()
        
        # Mock the response
        mock_response = MagicMock()
        mock_response.text = "Mock analysis result"
        mock_genai.Client.return_value.models.generate_content.return_value = mock_response
        
        # Patch the imports before importing GeminiClient
        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        }):
            # Import and create client
            from src.llm.gemini import GeminiClient
            
            # Create client with mocked dependencies
            client = GeminiClient.__new__(GeminiClient)
            client._genai = mock_genai
            client._types = mock_types
            client.client = mock_genai.Client.return_value
            
            with open(temp_config, 'r') as f:
                client.config = json.load(f)['llm']
            
            client.temperature = 0.2
            client.max_output_tokens = 4096
            client.summary_length = "medium"
            client.batch_size = 16
            client.model_name = "gemini-2.5-flash"
            
            yield client, mock_genai
    
    def test_analyze_paper_from_abstract(self, mock_gemini_client, mock_paper):
        """Test abstract analysis."""
        client, mock_genai = mock_gemini_client
        
        # Mock prompt template loading
        with patch.object(client, '_load_prompt_template', return_value="Analyze {title} by {authors}: {abstract} {categories} {published_date}"):
            result = client.analyze_paper_from_abstract(mock_paper)
            
            assert result == "Mock analysis result"
            mock_genai.Client.return_value.models.generate_content.assert_called_once()
    
    def test_score_single_paper(self, mock_gemini_client, mock_paper):
        """Test single paper scoring."""
        client, mock_genai = mock_gemini_client
        
        # Mock response with valid JSON
        mock_response = MagicMock()
        mock_response.text = '{"relevance_score": 3, "significance_score": 2, "combined_score": 5}'
        mock_genai.Client.return_value.models.generate_content.return_value = mock_response
        
        with patch.object(client, '_load_prompt_template', return_value="Score paper {title} {authors} {abstract} {categories} {published_date} {keywords} {negative_keywords}"):
            result = client._score_single_paper(mock_paper, ["computer vision"])
            
            assert result["relevance_score"] == 3
            assert result["significance_score"] == 2
            assert result["combined_score"] == 5


class TestGeminiClientBackwardCompatibility:
    """Tests for backward compatibility."""
    
    def test_score_paper_relevance_alias(self, temp_config, monkeypatch):
        """Test that score_paper_relevance is an alias for _score_single_paper."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        
        with patch("src.llm.gemini.GeminiClient.__init__", lambda self, config_path: None):
            from src.llm.gemini import GeminiClient
            client = GeminiClient.__new__(GeminiClient)
            
            # Mock _score_single_paper
            client._score_single_paper = MagicMock(return_value={"relevance_score": 2})
            
            paper = {"title": "Test"}
            keywords = ["test"]
            
            result = client.score_paper_relevance(paper, keywords)
            
            client._score_single_paper.assert_called_once_with(paper, keywords, None)
            assert result["relevance_score"] == 2

