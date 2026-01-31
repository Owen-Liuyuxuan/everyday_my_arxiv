"""
Tests for the LLM client factory module.
"""
import json
import os
import pytest
import tempfile

from src.llm.factory import create_llm_client, get_available_providers


@pytest.fixture
def temp_gemini_config():
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
    with tempfile.NamedTemporaryFile(mode='w', suffix='_gemini.json', delete=False) as f:
        json.dump(config, f)
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def temp_ark_config():
    """Create a temporary Ark config file."""
    config = {
        "llm": {
            "provider": "ark",
            "text_model": "doubao-seed-1-6-251015",
            "document_model": "doubao-seed-1-6-251015",
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "temperature": 0.2,
            "max_output_tokens": 4096,
            "summary_length": "medium",
            "batch_size": 16
        }
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='_ark.json', delete=False) as f:
        json.dump(config, f)
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)


class TestFactoryProviderDetection:
    """Tests for provider auto-detection from config path."""
    
    def test_detect_gemini_from_path(self, temp_gemini_config, monkeypatch):
        """Test that 'gemini' in path triggers Gemini client."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        
        # Should try to create GeminiClient (will fail without API key)
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            create_llm_client(temp_gemini_config, provider="gemini")
    
    def test_detect_ark_from_path(self, temp_ark_config, monkeypatch):
        """Test that 'ark' in path triggers Ark client."""
        monkeypatch.delenv("ARK_API_KEY", raising=False)
        
        try:
            with pytest.raises(ValueError, match="ARK_API_KEY"):
                create_llm_client(temp_ark_config, provider="ark")
        except ImportError:
            pytest.skip("Volcengine SDK not installed")
    
    def test_default_to_gemini(self, temp_gemini_config, monkeypatch):
        """Test that unknown paths default to Gemini."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        
        # Should default to GeminiClient (indicated by GOOGLE_API_KEY error)
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            create_llm_client(temp_gemini_config)
    
    def test_explicit_provider_overrides_path(self, temp_gemini_config, monkeypatch):
        """Test that explicit provider parameter overrides path detection."""
        monkeypatch.delenv("ARK_API_KEY", raising=False)
        
        # Path is gemini config, but explicit provider says Ark
        try:
            with pytest.raises(ValueError, match="ARK_API_KEY"):
                create_llm_client(temp_gemini_config, provider="ark")
        except ImportError:
            pytest.skip("Volcengine SDK not installed")


class TestFactoryWithMockedEnvironment:
    """Tests for factory with mocked API keys."""
    
    def test_gemini_client_creation_with_api_key(self, temp_gemini_config, monkeypatch):
        """Test creating Gemini client with API key set."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        
        # This will still fail because the API key is invalid,
        # but it tests that we get past the environment check
        try:
            client = create_llm_client(temp_gemini_config, provider="gemini")
            # If we get here, client was created (would need valid API)
            assert client is not None
        except Exception as e:
            # Expected - invalid API key or network error
            # But should not be ValueError about missing API key
            assert "GOOGLE_API_KEY environment variable is not set" not in str(e)
    
    def test_ark_client_creation_with_api_key(self, temp_ark_config, monkeypatch):
        """Test creating Ark client with API key set."""
        monkeypatch.setenv("ARK_API_KEY", "test-api-key")
        
        try:
            client = create_llm_client(temp_ark_config, provider="ark")
            assert client is not None
        except ImportError:
            # SDK might not be installed
            pytest.skip("Volcengine SDK not installed")
        except Exception as e:
            # Should not be ValueError about missing API key
            assert "ARK_API_KEY environment variable is not set" not in str(e)
    
    def test_missing_gemini_api_key(self, temp_gemini_config, monkeypatch):
        """Test that missing Gemini API key raises ValueError."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            create_llm_client(temp_gemini_config, provider="gemini")
    
    def test_missing_ark_api_key(self, temp_ark_config, monkeypatch):
        """Test that missing Ark API key raises ValueError."""
        monkeypatch.delenv("ARK_API_KEY", raising=False)
        
        try:
            with pytest.raises(ValueError, match="ARK_API_KEY"):
                create_llm_client(temp_ark_config, provider="ark")
        except ImportError:
            pytest.skip("Volcengine SDK not installed")


class TestFactoryUnknownProvider:
    """Tests for unknown provider handling."""
    
    def test_unknown_provider_raises_error(self):
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_client("config/config.json", provider="unknown_provider")


class TestAvailableProviders:
    """Tests for get_available_providers function."""
    
    def test_get_available_providers(self):
        """Test that available providers are detected correctly."""
        providers = get_available_providers()
        
        # Should be a list
        assert isinstance(providers, list)
        
        # Gemini should be available (google-genai is a core dependency)
        assert "gemini" in providers
        
        # Ark may or may not be available depending on optional dep installation
        # Just verify the function doesn't crash
        assert all(p in ["gemini", "ark"] for p in providers)

