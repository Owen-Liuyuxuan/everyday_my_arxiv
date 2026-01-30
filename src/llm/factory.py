"""
Factory module for creating LLM clients with lazy imports.

This module provides a factory function that instantiates the appropriate
LLM client based on configuration, with lazy imports to avoid requiring
all provider SDKs to be installed.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm.base import BaseLLMClient


def create_llm_client(config_path: str = "config/config.json", 
                      provider: str = None) -> "BaseLLMClient":
    """
    Factory function to create an LLM client with lazy imports.
    
    Provider detection order:
    1. Explicit provider parameter
    2. Auto-detect from config filename (e.g., config_ark.json -> ark)
    3. Default to Gemini for backward compatibility
    
    Args:
        config_path: Path to the configuration JSON file
        provider: Explicit provider name ('gemini' or 'ark'). 
                  If None, auto-detects from config_path.
    
    Returns:
        An instance of BaseLLMClient (either GeminiClient or ArkClient)
    
    Raises:
        ValueError: If provider is unknown
        ImportError: If required SDK is not installed
    
    Example:
        # Auto-detect from config path
        client = create_llm_client("config/config.json")      # -> GeminiClient
        client = create_llm_client("config/config_ark.json")  # -> ArkClient
        
        # Explicit provider
        client = create_llm_client("config/custom.json", provider="ark")
    """
    # Determine provider
    if provider is None:
        # Auto-detect from config filename
        if "ark" in config_path.lower():
            provider = "ark"
        elif "gemini" in config_path.lower():
            provider = "gemini"
        else:
            # Default to Gemini for backward compatibility
            provider = "gemini"
    
    provider = provider.lower()
    
    if provider == "gemini":
        try:
            from src.llm.gemini import GeminiClient
            return GeminiClient(config_path)
        except ImportError as e:
            raise ImportError(
                "Google GenAI SDK not installed. "
                "Install with: pip install google-genai"
            ) from e
    
    elif provider == "ark":
        try:
            from src.llm.ark import ArkClient
            return ArkClient(config_path)
        except ImportError as e:
            raise ImportError(
                "Volcengine Ark SDK not installed. "
                "Install with: pip install 'volcengine-python-sdk[ark]'"
            ) from e
    
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Supported providers: 'gemini', 'ark'"
        )


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
    
    return available

