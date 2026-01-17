"""
BAIS API Key Configuration

Configure API keys via environment variables:
- GROK_API_KEY or XAI_API_KEY
- OPENAI_API_KEY
- GEMINI_API_KEY or GOOGLE_API_KEY
- VERTEX_AI_CREDENTIALS (JSON string or file path)
- VERTEX_AI_PROJECT_ID
- VERTEX_AI_LOCATION

See env.example for template.
"""

import os
import json

# API Keys from environment variables only
DEFAULT_API_KEYS = {
    "grok": os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY") or "",
    "openai": os.environ.get("OPENAI_API_KEY") or "",
    "google": os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "",
}

# Vertex AI credentials from environment
def _load_vertex_credentials():
    """Load Vertex AI credentials from environment."""
    creds_json = os.environ.get("VERTEX_AI_CREDENTIALS")
    if creds_json:
        try:
            if os.path.isfile(creds_json):
                with open(creds_json) as f:
                    return json.load(f)
            return json.loads(creds_json)
        except (json.JSONDecodeError, IOError):
            pass
    return {}

VERTEX_AI_CREDENTIALS = _load_vertex_credentials()
VERTEX_AI_PROJECT_ID = os.environ.get("VERTEX_AI_PROJECT_ID") or VERTEX_AI_CREDENTIALS.get("project_id", "")
VERTEX_AI_LOCATION = os.environ.get("VERTEX_AI_LOCATION", "global")


def initialize_default_keys():
    """
    Initialize model_provider with default API keys.
    
    Call this at application startup to ensure keys are available.
    """
    from core.model_provider import set_api_key, is_provider_active
    
    for provider, key in DEFAULT_API_KEYS.items():
        if key and not is_provider_active(provider):
            set_api_key(provider, key)
    
    # Initialize Vertex AI with service account credentials
    if VERTEX_AI_CREDENTIALS and not is_provider_active("vertex"):
        # Set project ID as the "key" for Vertex (it uses service account auth)
        set_api_key("vertex", VERTEX_AI_PROJECT_ID)


def get_default_key(provider: str) -> str:
    """Get default key for a provider."""
    return DEFAULT_API_KEYS.get(provider, "")


def get_vertex_credentials() -> dict:
    """Get Vertex AI service account credentials."""
    return VERTEX_AI_CREDENTIALS


def get_vertex_credentials_json() -> str:
    """Get Vertex AI credentials as JSON string."""
    return json.dumps(VERTEX_AI_CREDENTIALS)

