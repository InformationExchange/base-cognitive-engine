"""
BASE Centralized Model & API Key Provider

SINGLE SOURCE OF TRUTH for all LLM configurations across the codebase.
- Auto-discovers latest models from provider APIs
- Manages API keys centrally (env vars or explicit)
- Tracks which providers are active based on available keys
- Supports all major LLM providers

Usage:
    from core.model_provider import (
        get_model, get_api_key, get_active_providers,
        is_provider_active, get_provider_config
    )
    
    # Get model and key for any provider
    model = get_model("grok")
    api_key = get_api_key("grok")
    
    # Check which providers are available
    active = get_active_providers()  # ["grok", "openai"] if keys exist
    
    # Configure a provider at runtime
    set_api_key("openai", "sk-...")
"""

import os
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ProviderStatus(Enum):
    """Status of an LLM provider."""
    ACTIVE = "active"        # API key present and valid
    INACTIVE = "inactive"    # No API key configured
    ERROR = "error"          # API key present but invalid
    RATE_LIMITED = "rate_limited"  # Temporarily unavailable


@dataclass
class ProviderConfig:
    """Configuration for a model provider."""
    provider: str
    display_name: str
    api_base_url: str
    models_endpoint: str
    env_var: str
    alt_env_vars: List[str]  # Alternative env var names
    default_model: str
    reasoning_model: str
    fallback_models: List[str]
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer"
    extra_headers: Dict[str, str] = field(default_factory=dict)


# ============================================================================
# PROVIDER CONFIGURATIONS - All Major LLM Providers
# ============================================================================

PROVIDER_CONFIGS: Dict[str, ProviderConfig] = {
    "grok": ProviderConfig(
        provider="grok",
        display_name="Grok (xAI)",
        api_base_url="https://api.x.ai/v1",
        models_endpoint="/models",
        env_var="GROK_API_KEY",
        alt_env_vars=["XAI_API_KEY"],
        default_model="grok-3",
        reasoning_model="grok-4-1-fast-reasoning",
        fallback_models=["grok-3", "grok-3-mini", "grok-2-1212"]
    ),
    "openai": ProviderConfig(
        provider="openai",
        display_name="OpenAI",
        api_base_url="https://api.openai.com/v1",
        models_endpoint="/models",
        env_var="OPENAI_API_KEY",
        alt_env_vars=[],
        default_model="gpt-4o",
        reasoning_model="gpt-4o",
        fallback_models=["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
    ),
    "anthropic": ProviderConfig(
        provider="anthropic",
        display_name="Anthropic (Claude)",
        api_base_url="https://api.anthropic.com/v1",
        models_endpoint="/models",
        env_var="ANTHROPIC_API_KEY",
        alt_env_vars=["CLAUDE_API_KEY"],
        default_model="claude-sonnet-4-20250514",
        reasoning_model="claude-sonnet-4-20250514",
        fallback_models=["claude-3-5-sonnet-20241022", "claude-3-opus", "claude-3-haiku"],
        auth_header="x-api-key",
        auth_prefix="",
        extra_headers={"anthropic-version": "2023-06-01"}
    ),
    "google": ProviderConfig(
        provider="google",
        display_name="Google (Gemini)",
        api_base_url="https://generativelanguage.googleapis.com/v1beta",
        models_endpoint="/models",
        env_var="GOOGLE_API_KEY",
        alt_env_vars=["GEMINI_API_KEY"],
        default_model="gemini-3-flash-preview",
        reasoning_model="gemini-3-pro-preview",
        fallback_models=["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"]
    ),
    "mistral": ProviderConfig(
        provider="mistral",
        display_name="Mistral AI",
        api_base_url="https://api.mistral.ai/v1",
        models_endpoint="/models",
        env_var="MISTRAL_API_KEY",
        alt_env_vars=[],
        default_model="mistral-large-latest",
        reasoning_model="mistral-large-latest",
        fallback_models=["mistral-medium", "mistral-small"]
    ),
    "cohere": ProviderConfig(
        provider="cohere",
        display_name="Cohere",
        api_base_url="https://api.cohere.ai/v1",
        models_endpoint="/models",
        env_var="COHERE_API_KEY",
        alt_env_vars=["CO_API_KEY"],
        default_model="command-r-plus",
        reasoning_model="command-r-plus",
        fallback_models=["command-r", "command"]
    ),
    "groq": ProviderConfig(
        provider="groq",
        display_name="Groq",
        api_base_url="https://api.groq.com/openai/v1",
        models_endpoint="/models",
        env_var="GROQ_API_KEY",
        alt_env_vars=[],
        default_model="llama-3.3-70b-versatile",
        reasoning_model="llama-3.3-70b-versatile",
        fallback_models=["llama-3.1-70b-versatile", "mixtral-8x7b-32768"]
    ),
    "together": ProviderConfig(
        provider="together",
        display_name="Together AI",
        api_base_url="https://api.together.xyz/v1",
        models_endpoint="/models",
        env_var="TOGETHER_API_KEY",
        alt_env_vars=[],
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        reasoning_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        fallback_models=["mistralai/Mixtral-8x7B-Instruct-v0.1"]
    ),
    "deepseek": ProviderConfig(
        provider="deepseek",
        display_name="DeepSeek",
        api_base_url="https://api.deepseek.com/v1",
        models_endpoint="/models",
        env_var="DEEPSEEK_API_KEY",
        alt_env_vars=[],
        default_model="deepseek-chat",
        reasoning_model="deepseek-reasoner",
        fallback_models=["deepseek-chat", "deepseek-coder"]
    ),
    # Vertex AI (Google Cloud) - separate from public Gemini API
    # Uses service account or ADC for authentication
    "vertex": ProviderConfig(
        provider="vertex",
        display_name="Google Vertex AI",
        api_base_url="https://us-central1-aiplatform.googleapis.com/v1",
        models_endpoint="/models",
        env_var="GOOGLE_APPLICATION_CREDENTIALS",
        alt_env_vars=["VERTEX_API_KEY", "GCP_PROJECT_ID"],
        default_model="gemini-3-flash-preview",
        reasoning_model="gemini-3-pro-preview",
        fallback_models=["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"]
    ),
}


# ============================================================================
# RUNTIME STATE
# ============================================================================

# Cache for discovered models
_model_cache: Dict[str, List[str]] = {}
_cache_timestamp: Optional[datetime] = None
_cache_ttl = timedelta(hours=1)

# Runtime API key storage (overrides env vars)
_runtime_api_keys: Dict[str, str] = {}

# Provider status tracking
_provider_status: Dict[str, ProviderStatus] = {}


# ============================================================================
# API KEY MANAGEMENT
# ============================================================================

def get_api_key(provider: str) -> Optional[str]:
    """
    Get API key for a provider.
    
    Priority:
    1. Runtime-configured key (via set_api_key)
    2. Primary environment variable
    3. Alternative environment variables
    
    Returns None if no key is available.
    """
    # Check runtime keys first
    if provider in _runtime_api_keys:
        return _runtime_api_keys[provider]
    
    config = PROVIDER_CONFIGS.get(provider)
    if not config:
        return None
    
    # Check primary env var
    key = os.environ.get(config.env_var)
    if key:
        return key
    
    # Check alternative env vars
    for alt_var in config.alt_env_vars:
        key = os.environ.get(alt_var)
        if key:
            return key
    
    return None


def set_api_key(provider: str, api_key: str) -> bool:
    """
    Set API key for a provider at runtime.
    
    This allows clients to configure providers dynamically
    without environment variables.
    
    Returns True if provider exists and key was set.
    """
    if provider not in PROVIDER_CONFIGS:
        logger.warning(f"Unknown provider: {provider}")
        return False
    
    _runtime_api_keys[provider] = api_key
    _provider_status[provider] = ProviderStatus.ACTIVE
    logger.info(f"API key configured for {provider}")
    return True


def clear_api_key(provider: str) -> None:
    """Remove runtime API key for a provider."""
    if provider in _runtime_api_keys:
        del _runtime_api_keys[provider]
        _provider_status[provider] = ProviderStatus.INACTIVE


def is_provider_active(provider: str) -> bool:
    """Check if a provider has an API key configured."""
    return get_api_key(provider) is not None


def get_active_providers() -> List[str]:
    """
    Get list of providers that have API keys configured.
    
    These are the providers available for use.
    """
    return [p for p in PROVIDER_CONFIGS if is_provider_active(p)]


def get_all_providers() -> List[str]:
    """Get list of all supported providers."""
    return list(PROVIDER_CONFIGS.keys())


# Alias for backward compatibility
list_available_providers = get_active_providers


def get_provider_config(provider: str) -> Optional[ProviderConfig]:
    """Get full configuration for a provider."""
    return PROVIDER_CONFIGS.get(provider)


def get_provider_status(provider: str) -> ProviderStatus:
    """Get current status of a provider."""
    if provider in _provider_status:
        return _provider_status[provider]
    
    if is_provider_active(provider):
        return ProviderStatus.ACTIVE
    
    return ProviderStatus.INACTIVE


# ============================================================================
# MODEL DISCOVERY & SELECTION
# ============================================================================

def _discover_models(provider: str) -> List[str]:
    """
    Query provider API to discover available models.
    
    Returns list sorted by recency (newest first).
    """
    config = PROVIDER_CONFIGS.get(provider)
    if not config:
        return []
    
    api_key = get_api_key(provider)
    if not api_key:
        return []
    
    try:
        # Build headers
        headers = {}
        url = f"{config.api_base_url}{config.models_endpoint}"
        
        # Google uses API key as URL parameter, not header
        if provider == "google":
            url = f"{url}?key={api_key}"
        else:
            if config.auth_prefix:
                headers[config.auth_header] = f"{config.auth_prefix} {api_key}"
            else:
                headers[config.auth_header] = api_key
        
        headers.update(config.extra_headers)
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if "data" in data:
                models = sorted(
                    data["data"],
                    key=lambda x: x.get("created", 0),
                    reverse=True
                )
                _provider_status[provider] = ProviderStatus.ACTIVE
                return [m["id"] for m in models]
            
            # Some providers use different response formats
            if "models" in data:
                return [m.get("name", m.get("id", "")) for m in data["models"]]
        
        elif response.status_code == 401:
            _provider_status[provider] = ProviderStatus.ERROR
            logger.warning(f"Invalid API key for {provider}")
        
        elif response.status_code == 429:
            _provider_status[provider] = ProviderStatus.RATE_LIMITED
            logger.warning(f"Rate limited by {provider}")
            
    except Exception as e:
        logger.warning(f"Error discovering {provider} models: {e}")
    
    return []


def _refresh_cache() -> None:
    """Refresh the model cache if expired."""
    global _model_cache, _cache_timestamp
    
    if _cache_timestamp and datetime.now() - _cache_timestamp < _cache_ttl:
        return
    
    for provider in PROVIDER_CONFIGS:
        if is_provider_active(provider):
            discovered = _discover_models(provider)
            if discovered:
                _model_cache[provider] = discovered
                logger.info(f"Discovered {len(discovered)} {provider} models")
    
    _cache_timestamp = datetime.now()


def get_available_models(provider: str) -> List[str]:
    """
    Get list of available models for a provider.
    
    Auto-discovers from API if cache is empty/expired.
    """
    _refresh_cache()
    
    if provider in _model_cache and _model_cache[provider]:
        return _model_cache[provider]
    
    config = PROVIDER_CONFIGS.get(provider)
    if config:
        return [config.default_model] + config.fallback_models
    
    return []


def get_model(provider: str) -> str:
    """
    Get the recommended model for a provider.
    
    Returns the latest discovered model, or default if unavailable.
    """
    models = get_available_models(provider)
    
    if models:
        return models[0]
    
    config = PROVIDER_CONFIGS.get(provider)
    return config.default_model if config else "unknown"


def get_reasoning_model(provider: str) -> str:
    """
    Get the best reasoning-capable model for a provider.
    
    Prioritizes models with 'reasoning' or 'thinking' in name.
    """
    models = get_available_models(provider)
    
    # Filter for reasoning models
    reasoning = [
        m for m in models 
        if any(kw in m.lower() for kw in ["reasoning", "thinking", "o1", "o3"])
        and "non-reasoning" not in m.lower()
    ]
    
    if reasoning:
        return reasoning[0]
    
    # Fall back to configured reasoning model
    config = PROVIDER_CONFIGS.get(provider)
    if config:
        return config.reasoning_model
    
    # Fall back to any latest model
    return models[0] if models else "unknown"


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

# Primary provider for BASE (Vertex AI as main, matching ai.invitas.com)
PRIMARY_PROVIDER = "vertex"
PROVIDER_PRIORITY = ["vertex", "google", "openai", "grok"]  # Preferred order


def get_default_model() -> str:
    """Get the default model from the primary provider (Vertex AI)."""
    # Try primary provider first
    if is_provider_active(PRIMARY_PROVIDER):
        return get_model(PRIMARY_PROVIDER)
    
    # Fall back to priority list
    for provider in PROVIDER_PRIORITY:
        if is_provider_active(provider):
            return get_model(provider)
    
    # Last resort
    active = get_active_providers()
    if active:
        return get_model(active[0])
    return PROVIDER_CONFIGS["vertex"].default_model


def get_default_provider() -> str:
    """Get the primary provider (Vertex AI), or first active provider."""
    # Try primary provider first
    if is_provider_active(PRIMARY_PROVIDER):
        return PRIMARY_PROVIDER
    
    # Fall back to priority list
    for provider in PROVIDER_PRIORITY:
        if is_provider_active(provider):
            return provider
    
    # Last resort
    active = get_active_providers()
    return active[0] if active else "vertex"


def force_refresh() -> Dict[str, List[str]]:
    """Force refresh the model cache and return discovered models."""
    global _cache_timestamp
    _cache_timestamp = None
    _refresh_cache()
    return _model_cache.copy()


# ============================================================================
# PROVIDER SUMMARY FOR UI/REPORTING
# ============================================================================

def get_provider_summary() -> List[Dict]:
    """
    Get summary of all providers for UI display.
    
    Returns list of dicts with provider info and status.
    """
    summary = []
    
    for provider, config in PROVIDER_CONFIGS.items():
        status = get_provider_status(provider)
        models = get_available_models(provider) if status == ProviderStatus.ACTIVE else []
        
        summary.append({
            "provider": provider,
            "display_name": config.display_name,
            "status": status.value,
            "active": status == ProviderStatus.ACTIVE,
            "default_model": config.default_model,
            "reasoning_model": config.reasoning_model,
            "available_models": models[:5],  # Top 5
            "model_count": len(models),
            "env_var": config.env_var
        })
    
    return summary


# ============================================================================
# BACKWARD COMPATIBILITY - Legacy function names
# ============================================================================

def get_best_reasoning_model(provider: str) -> str:
    """Alias for get_reasoning_model (backward compatibility)."""
    return get_reasoning_model(provider)


def get_default_grok_model() -> str:
    """Convenience function for Grok model."""
    return get_reasoning_model("grok")


def get_default_openai_model() -> str:
    """Convenience function for OpenAI model."""
    return get_model("openai")


def get_default_anthropic_model() -> str:
    """Convenience function for Anthropic model."""
    return get_model("anthropic")


# Pre-computed defaults (static fallbacks)
DEFAULT_GROK = "grok-3"
DEFAULT_OPENAI = "gpt-4o"
DEFAULT_ANTHROPIC = "claude-sonnet-4-20250514"
DEFAULT_GOOGLE = "gemini-3-flash-preview"
DEFAULT_MISTRAL = "mistral-large-latest"


# ============================================================================
# AUTO-INITIALIZATION
# ============================================================================

def _auto_initialize():
    """
    Auto-initialize API keys from config_keys module.
    
    This ensures default keys are available without explicit initialization.
    """
    try:
        from core.config_keys import DEFAULT_API_KEYS, VERTEX_AI_PROJECT_ID
        
        # Initialize standard API key providers
        for provider, key in DEFAULT_API_KEYS.items():
            if key and provider not in _runtime_api_keys:
                _runtime_api_keys[provider] = key
                _provider_status[provider] = ProviderStatus.ACTIVE
        
        # Initialize Vertex AI with project ID (uses service account auth)
        if VERTEX_AI_PROJECT_ID and "vertex" not in _runtime_api_keys:
            _runtime_api_keys["vertex"] = VERTEX_AI_PROJECT_ID
            _provider_status["vertex"] = ProviderStatus.ACTIVE
            
    except ImportError:
        # config_keys not available - rely on env vars only
        pass

# Auto-initialize on module import
_auto_initialize()


# ============================================================================
# TEST/DEBUG
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BASE CENTRALIZED MODEL PROVIDER")
    print("=" * 70)
    
    print("\n[Supported Providers]")
    for provider, config in PROVIDER_CONFIGS.items():
        status = "ACTIVE" if is_provider_active(provider) else "INACTIVE"
        print(f"  {config.display_name:25} [{status:8}] env: {config.env_var}")
    
    print(f"\n[Active Providers]: {get_active_providers()}")
    
    # Force refresh for active providers
    if get_active_providers():
        print("\n[Discovering Models...]")
        discovered = force_refresh()
        
        for provider, models in discovered.items():
            print(f"\n  {provider.upper()}:")
            for i, model in enumerate(models[:5]):
                marker = "→" if i == 0 else " "
                print(f"    {marker} {model}")
    
    print("\n[Recommended Models]")
    for provider in get_active_providers():
        print(f"  {provider}: {get_reasoning_model(provider)}")
    
    print("\n" + "=" * 70)
