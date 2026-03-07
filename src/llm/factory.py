"""
Factory module for creating LLM clients with lazy imports.

This module provides factory functions that instantiate the appropriate
LLM client based on configuration, with lazy imports to avoid requiring
all provider SDKs to be installed.

Supports two types of clients:
1. Scoring clients: Text-only, used for relevance/significance scoring
2. PDF clients: Multimodal, used for PDF analysis (Gemini, Ark with vision)
"""
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.llm.base import BaseLLMClient


def create_llm_client(config_path: str = "config/config.json",
                      provider: str = None) -> tuple["BaseLLMClient", str]:
    """
    Factory function to create an LLM client with lazy imports.

    Provider detection order:
    1. Explicit provider parameter
    2. Auto-detect from config filename (e.g., config_ark.json -> ark)
    3. Default to Gemini for backward compatibility

    Args:
        config_path: Path to the configuration JSON file
        provider: Explicit provider name ('gemini', 'ark', or 'openai').
                  If None, auto-detects from config_path.

    Returns:
        A tuple of (client_instance, provider_name) where:
        - client_instance: An instance of BaseLLMClient
        - provider_name: The actual provider being used (e.g., 'gemini', 'ark', 'openai')

    Raises:
        ValueError: If provider is unknown
        ImportError: If required SDK is not installed

    Example:
        # Auto-detect from config path
        client, provider = create_llm_client("config/config.json")      # -> (GeminiClient, 'gemini')
        client, provider = create_llm_client("config/config_ark.json")  # -> (ArkClient, 'ark')

        # Explicit provider
        client, provider = create_llm_client("config/custom.json", provider="ark")  # -> (ArkClient, 'ark')
    """
    # Determine provider
    if provider is None:
        # Auto-detect from config filename
        if "ark" in config_path.lower():
            provider = "ark"
        elif "gemini" in config_path.lower():
            provider = "gemini"
        elif "openai" in config_path.lower():
            provider = "openai"
        else:
            # Default to Gemini for backward compatibility
            provider = "gemini"

    provider = provider.lower()

    if provider == "gemini":
        try:
            from src.llm.gemini import GeminiClient
            return GeminiClient(config_path), "gemini"
        except ImportError as e:
            raise ImportError(
                "Google GenAI SDK not installed. "
                "Install with: pip install google-genai"
            ) from e

    elif provider == "ark":
        try:
            from src.llm.ark import ArkClient
            return ArkClient(config_path), "ark"
        except ImportError as e:
            raise ImportError(
                "Volcengine Ark SDK not installed. "
                "Install with: pip install 'volcengine-python-sdk[ark]'"
            ) from e

    elif provider == "openai":
        try:
            from src.llm.openai_client import OpenAIClient
            return OpenAIClient(config_path), "openai"
        except ImportError as e:
            raise ImportError(
                "OpenAI SDK not installed. "
                "Install with: pip install openai"
            ) from e

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Supported providers: 'gemini', 'ark', 'openai'"
        )


def create_scoring_client(config_path: str = "config/config.json") -> "BaseLLMClient":
    """
    Create a client for text-only scoring operations.

    Reads from config to determine which provider to use for scoring:
    - config['llm']['scoring_provider']: Provider to use (gemini, ark, openai)
    - config['llm']['scoring_model']: Model name for the provider
    - config['llm']['scoring_base_url']: Base URL for OpenAI-compatible endpoints

    If scoring_provider is not specified, falls back to the main provider.

    Args:
        config_path: Path to the configuration JSON file

    Returns:
        An instance of BaseLLMClient configured for scoring

    Example config:
        {
            "llm": {
                "scoring_provider": "openai",
                "scoring_model": "deepseek-chat",
                "scoring_base_url": "https://api.deepseek.com"
            }
        }
    """
    import json

    with open(config_path, 'r') as f:
        config = json.load(f)

    llm_config = config.get('llm', {})

    # Check for scoring-specific provider
    scoring_provider = llm_config.get('scoring_provider')

    if scoring_provider:
        # Create a temporary config with scoring-specific settings
        scoring_config = {**llm_config}

        if 'scoring_model' in llm_config:
            scoring_config['model'] = llm_config['scoring_model']
        if 'scoring_base_url' in llm_config:
            scoring_config['base_url'] = llm_config['scoring_base_url']

        # Write temporary config if needed
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump({**config, 'llm': scoring_config}, tmp)
            tmp_path = tmp.name

        try:
            client, _ = create_llm_client(tmp_path, provider=scoring_provider)
            return client
        finally:
            os.unlink(tmp_path)
    else:
        # Fall back to main provider
        return create_llm_client(config_path)[0]


def create_pdf_client(config_path: str = "config/config.json") -> "BaseLLMClient":
    """
    Create a client for multimodal PDF analysis.

    Reads from config to determine which provider to use for PDF analysis:
    - config['llm']['pdf_provider']: Provider to use (gemini, ark)
    - config['llm']['pdf_model']: Model name for the provider

    Only Gemini and Ark support PDF analysis (multimodal).

    If pdf_provider is not specified, falls back to the main provider.

    Args:
        config_path: Path to the configuration JSON file

    Returns:
        An instance of BaseLLMClient configured for PDF analysis

    Raises:
        ValueError: If pdf_provider is set to 'openai' (not supported)

    Example config:
        {
            "llm": {
                "pdf_provider": "gemini",
                "pdf_model": "gemini-2.5-flash-lite"
            }
        }
    """
    import json

    with open(config_path, 'r') as f:
        config = json.load(f)

    llm_config = config.get('llm', {})

    # Check for PDF-specific provider
    pdf_provider = llm_config.get('pdf_provider')

    if pdf_provider:
        if pdf_provider.lower() == "openai":
            raise ValueError(
                "PDF analysis is not supported by OpenAI-compatible client. "
                "Use 'gemini' or 'ark' for pdf_provider."
            )

        # Create a temporary config with PDF-specific settings
        pdf_config = {**llm_config}

        if 'pdf_model' in llm_config:
            pdf_config['model'] = llm_config['pdf_model']

        # Write temporary config if needed
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump({**config, 'llm': pdf_config}, tmp)
            tmp_path = tmp.name

        try:
            client, _ = create_llm_client(tmp_path, provider=pdf_provider)
            return client
        finally:
            os.unlink(tmp_path)
    else:
        # Fall back to main provider
        return create_llm_client(config_path)[0]


def get_available_providers() -> list:
    """
    Get list of available LLM providers based on installed SDKs.
    
    Returns:
        List of provider names that can be instantiated
    """
    available = []
    
    try:
        from google import genai  # noqa: F401
        available.append("gemini")
    except ImportError:
        pass
    
    try:
        from volcenginesdkarkruntime import Ark  # noqa: F401
        available.append("ark")
    except ImportError:
        pass
    
    try:
        from openai import OpenAI  # noqa: F401
        available.append("openai")
    except ImportError:
        pass
    
    return available

