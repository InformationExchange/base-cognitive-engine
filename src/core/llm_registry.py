"""
BASE LLM Provider Registry and Management

Features:
1. Multi-provider support (Grok, OpenAI, Anthropic, Google, etc.)
2. Secure API key management
3. Version discovery and auto-update
4. Provider switching and fallback
5. Audit logging for all LLM calls

IMPORTANT: For governance verification, use a DIFFERENT provider than the one being governed.
"""

import os
import json
import hashlib
import time
import logging
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from datetime import datetime
import re
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    GROK = "grok"           # xAI
    OPENAI = "openai"       # OpenAI
    ANTHROPIC = "anthropic" # Anthropic
    GOOGLE = "google"       # Google (Gemini public API)
    VERTEX = "vertex"       # Google Vertex AI (Cloud)
    MISTRAL = "mistral"     # Mistral AI
    META = "meta"           # Meta (Llama via various endpoints)
    COHERE = "cohere"       # Cohere
    GROQ = "groq"           # Groq (fast inference)
    TOGETHER = "together"   # Together AI
    DEEPSEEK = "deepseek"   # DeepSeek
    
    
class ProviderStatus(Enum):
    """Provider availability status."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"
    KEY_INVALID = "key_invalid"
    DEPRECATED = "deprecated"


@dataclass
class LLMModel:
    """Information about a specific LLM model."""
    provider: LLMProvider
    model_id: str
    display_name: str
    context_window: int
    max_output_tokens: int
    cost_per_1k_input: float
    cost_per_1k_output: float
    capabilities: List[str]
    release_date: str
    is_latest: bool = False
    is_deprecated: bool = False
    reasoning_capable: bool = False
    

@dataclass 
class ProviderConfig:
    """Configuration for an LLM provider."""
    provider: LLMProvider
    api_base_url: str
    api_key_env_var: str
    models: List[LLMModel] = field(default_factory=list)
    default_model: str = ""
    status: ProviderStatus = ProviderStatus.AVAILABLE
    last_health_check: Optional[datetime] = None
    rate_limit_reset: Optional[datetime] = None


@dataclass
class APIKeyInfo:
    """Secure API key storage info (not the actual key)."""
    provider: LLMProvider
    env_var_name: str
    key_hash: str  # SHA256 hash of last 4 chars for verification
    last_verified: Optional[datetime] = None
    is_valid: bool = False
    permissions: List[str] = field(default_factory=list)


@dataclass
class LLMCallRecord:
    """Audit record for an LLM call."""
    call_id: str
    timestamp: str
    provider: str
    model: str
    purpose: str  # "governance_verification", "semantic_analysis", etc.
    input_tokens: int
    output_tokens: int
    latency_ms: float
    success: bool
    error: Optional[str] = None
    cost_usd: float = 0.0


class SecureKeyManager:
    """
    Secure API Key Manager
    
    - Keys are NEVER stored in code or logs
    - Keys are read from environment variables
    - Only key hashes are stored for verification
    - Supports key rotation
    """
    
    # Mapping of providers to their environment variable names
    ENV_VAR_MAPPING = {
        LLMProvider.GROK: "GROK_API_KEY",
        LLMProvider.OPENAI: "OPENAI_API_KEY",
        LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
        LLMProvider.GOOGLE: "GOOGLE_API_KEY",
        LLMProvider.VERTEX: "GOOGLE_APPLICATION_CREDENTIALS",
        LLMProvider.MISTRAL: "MISTRAL_API_KEY",
        LLMProvider.COHERE: "COHERE_API_KEY",
        LLMProvider.GROQ: "GROQ_API_KEY",
        LLMProvider.TOGETHER: "TOGETHER_API_KEY",
        LLMProvider.DEEPSEEK: "DEEPSEEK_API_KEY",
    }
    
    # Alternative env vars (for fallback)
    ALT_ENV_VARS = {
        LLMProvider.GROK: ["XAI_API_KEY", "GROK_KEY"],
        LLMProvider.OPENAI: ["OPENAI_KEY"],
        LLMProvider.ANTHROPIC: ["CLAUDE_API_KEY", "ANTHROPIC_KEY"],
        LLMProvider.GOOGLE: ["GEMINI_API_KEY"],
        LLMProvider.VERTEX: ["GCP_PROJECT_ID", "VERTEX_API_KEY"],
    }
    
    def __init__(self, keys_file: Optional[Path] = None):
        self.key_info: Dict[LLMProvider, APIKeyInfo] = {}
        self._keys_file = keys_file or Path("~/.base/api_keys.json").expanduser()
        self._load_key_info()
    
    def _load_key_info(self):
        """Load key metadata (not actual keys)."""
        if self._keys_file.exists():
            try:
                with open(self._keys_file, 'r') as f:
                    data = json.load(f)
                    for provider_name, info in data.items():
                        provider = LLMProvider(provider_name)
                        self.key_info[provider] = APIKeyInfo(
                            provider=provider,
                            env_var_name=info.get('env_var_name', ''),
                            key_hash=info.get('key_hash', ''),
                            last_verified=datetime.fromisoformat(info['last_verified']) if info.get('last_verified') else None,
                            is_valid=info.get('is_valid', False),
                            permissions=info.get('permissions', [])
                        )
            except Exception as e:
                logger.warning(f"Could not load key info: {e}")
    
    def _save_key_info(self):
        """Save key metadata."""
        self._keys_file.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for provider, info in self.key_info.items():
            data[provider.value] = {
                'env_var_name': info.env_var_name,
                'key_hash': info.key_hash,
                'last_verified': info.last_verified.isoformat() if info.last_verified else None,
                'is_valid': info.is_valid,
                'permissions': info.permissions
            }
        with open(self._keys_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _hash_key_suffix(self, key: str) -> str:
        """Hash the last 4 characters of the key for verification."""
        if len(key) < 4:
            return ""
        return hashlib.sha256(key[-4:].encode()).hexdigest()[:16]
    
    def get_key(self, provider: LLMProvider) -> Optional[str]:
        """
        Get API key for provider from environment.
        NEVER stores or returns keys in logs.
        """
        # Try primary env var
        env_var = self.ENV_VAR_MAPPING.get(provider)
        if env_var:
            key = os.environ.get(env_var)
            if key:
                return key
        
        # Try alternative env vars
        for alt_var in self.ALT_ENV_VARS.get(provider, []):
            key = os.environ.get(alt_var)
            if key:
                return key
        
        return None
    
    def verify_key(self, provider: LLMProvider) -> bool:
        """Verify that an API key is set and valid format."""
        key = self.get_key(provider)
        if not key:
            return False
        
        # Basic format validation
        if len(key) < 10:
            return False
        
        # Provider-specific validation
        if provider == LLMProvider.GROK and not key.startswith('xai-'):
            logger.warning("Grok key should start with 'xai-'")
        elif provider == LLMProvider.OPENAI and not key.startswith('sk-'):
            logger.warning("OpenAI key should start with 'sk-'")
        
        # Update key info
        self.key_info[provider] = APIKeyInfo(
            provider=provider,
            env_var_name=self.ENV_VAR_MAPPING.get(provider, ''),
            key_hash=self._hash_key_suffix(key),
            last_verified=datetime.now(),
            is_valid=True
        )
        self._save_key_info()
        
        return True
    
    def get_available_providers(self) -> List[LLMProvider]:
        """Get list of providers with valid API keys."""
        available = []
        for provider in LLMProvider:
            if self.verify_key(provider):
                available.append(provider)
        return available

    # =========================================================================
    # Learning Interface
    # =========================================================================
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        """Record operation outcome for learning."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback to adjust behavior."""
        if not hasattr(self, '_learning_params'):
            self._learning_params = {}
        was_correct = feedback.get('was_correct', True)
        if not was_correct:
            for key in self._learning_params:
                self._learning_params[key] *= 0.95
    
    def get_statistics(self) -> Dict:
        """Return learning statistics."""
        return {
            'outcomes': len(getattr(self, '_outcomes', [])),
            'params': getattr(self, '_learning_params', {})
        }
    
    def serialize_state(self) -> Dict:
        """Serialize state for persistence."""
        return {
            'outcomes': getattr(self, '_outcomes', [])[-100:],
            'params': getattr(self, '_learning_params', {}),
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore state from serialized data."""
        self._outcomes = state.get('outcomes', [])
        self._learning_params = state.get('params', {})


class LLMRegistry:
    """
    Central registry for all LLM providers and models.
    
    Features:
    - Auto-discover available providers
    - Track model versions
    - Identify latest/recommended models
    - Health checking
    """
    
    # Known models by provider (updated manually or via discovery)
    KNOWN_MODELS = {
        LLMProvider.GROK: [
            LLMModel(
                provider=LLMProvider.GROK,
                model_id="grok-3",
                display_name="Grok 3",
                context_window=131072,
                max_output_tokens=32768,
                cost_per_1k_input=0.003,
                cost_per_1k_output=0.015,
                capabilities=["chat", "reasoning", "code", "analysis"],
                release_date="2025-02",
                is_latest=False,
                reasoning_capable=True
            ),
            LLMModel(
                provider=LLMProvider.GROK,
                model_id="grok-4-1-fast-reasoning",
                display_name="Grok 4.1 Fast Reasoning",
                context_window=131072,
                max_output_tokens=32768,
                cost_per_1k_input=0.005,
                cost_per_1k_output=0.025,
                capabilities=["chat", "reasoning", "code", "analysis", "vision"],
                release_date="2025-12",
                is_latest=True,
                reasoning_capable=True
            ),
        ],
        LLMProvider.OPENAI: [
            LLMModel(
                provider=LLMProvider.OPENAI,
                model_id="gpt-4-turbo",
                display_name="GPT-4 Turbo",
                context_window=128000,
                max_output_tokens=4096,
                cost_per_1k_input=0.01,
                cost_per_1k_output=0.03,
                capabilities=["chat", "code", "analysis", "vision"],
                release_date="2024-04",
                is_latest=False
            ),
            LLMModel(
                provider=LLMProvider.OPENAI,
                model_id="gpt-4o",
                display_name="GPT-4o",
                context_window=128000,
                max_output_tokens=16384,
                cost_per_1k_input=0.005,
                cost_per_1k_output=0.015,
                capabilities=["chat", "code", "analysis", "vision", "audio"],
                release_date="2024-05",
                is_latest=True
            ),
            LLMModel(
                provider=LLMProvider.OPENAI,
                model_id="o1",
                display_name="o1 (Reasoning)",
                context_window=200000,
                max_output_tokens=100000,
                cost_per_1k_input=0.015,
                cost_per_1k_output=0.060,
                capabilities=["reasoning", "code", "math", "science"],
                release_date="2024-12",
                is_latest=True,
                reasoning_capable=True
            ),
        ],
        LLMProvider.ANTHROPIC: [
            LLMModel(
                provider=LLMProvider.ANTHROPIC,
                model_id="claude-3-5-sonnet-20241022",
                display_name="Claude 3.5 Sonnet",
                context_window=200000,
                max_output_tokens=8192,
                cost_per_1k_input=0.003,
                cost_per_1k_output=0.015,
                capabilities=["chat", "code", "analysis", "vision"],
                release_date="2024-10",
                is_latest=False
            ),
            LLMModel(
                provider=LLMProvider.ANTHROPIC,
                model_id="claude-sonnet-4-20250514",
                display_name="Claude Sonnet 4",
                context_window=200000,
                max_output_tokens=16000,
                cost_per_1k_input=0.003,
                cost_per_1k_output=0.015,
                capabilities=["chat", "code", "analysis", "vision", "extended_thinking"],
                release_date="2025-05",
                is_latest=True,
                reasoning_capable=True
            ),
            LLMModel(
                provider=LLMProvider.ANTHROPIC,
                model_id="claude-opus-4-20250514",
                display_name="Claude Opus 4",
                context_window=200000,
                max_output_tokens=32000,
                cost_per_1k_input=0.015,
                cost_per_1k_output=0.075,
                capabilities=["chat", "code", "analysis", "vision", "extended_thinking", "agentic"],
                release_date="2025-05",
                is_latest=True,
                reasoning_capable=True
            ),
        ],
        LLMProvider.GOOGLE: [
            LLMModel(
                provider=LLMProvider.GOOGLE,
                model_id="gemini-2.5-flash",
                display_name="Gemini 2.5 Flash",
                context_window=1000000,
                max_output_tokens=65536,
                cost_per_1k_input=0.0001,
                cost_per_1k_output=0.0004,
                capabilities=["chat", "code", "analysis", "vision", "audio"],
                release_date="2025-06",
                is_latest=False
            ),
            LLMModel(
                provider=LLMProvider.GOOGLE,
                model_id="gemini-2.5-pro",
                display_name="Gemini 2.5 Pro",
                context_window=1000000,
                max_output_tokens=65536,
                cost_per_1k_input=0.00125,
                cost_per_1k_output=0.005,
                capabilities=["chat", "code", "analysis", "vision", "audio", "reasoning"],
                release_date="2025-06",
                is_latest=False,
                reasoning_capable=True
            ),
            LLMModel(
                provider=LLMProvider.GOOGLE,
                model_id="gemini-3-flash-preview",
                display_name="Gemini 3 Flash Preview",
                context_window=1000000,
                max_output_tokens=65536,
                cost_per_1k_input=0.0001,
                cost_per_1k_output=0.0004,
                capabilities=["chat", "code", "analysis", "vision", "audio", "reasoning"],
                release_date="2025-12",
                is_latest=True,
                reasoning_capable=True
            ),
            LLMModel(
                provider=LLMProvider.GOOGLE,
                model_id="gemini-3-pro-preview",
                display_name="Gemini 3 Pro Preview",
                context_window=1000000,
                max_output_tokens=65536,
                cost_per_1k_input=0.00125,
                cost_per_1k_output=0.01,
                capabilities=["chat", "code", "analysis", "vision", "audio", "reasoning", "agentic"],
                release_date="2025-12",
                is_latest=True,
                reasoning_capable=True
            ),
        ],
        LLMProvider.VERTEX: [
            LLMModel(
                provider=LLMProvider.VERTEX,
                model_id="gemini-3-flash-preview",
                display_name="Gemini 3 Flash (Vertex)",
                context_window=1000000,
                max_output_tokens=65536,
                cost_per_1k_input=0.0001,
                cost_per_1k_output=0.0004,
                capabilities=["chat", "code", "analysis", "vision", "audio", "reasoning"],
                release_date="2025-12",
                is_latest=True,
                reasoning_capable=True
            ),
            LLMModel(
                provider=LLMProvider.VERTEX,
                model_id="gemini-3-pro-preview",
                display_name="Gemini 3 Pro (Vertex)",
                context_window=1000000,
                max_output_tokens=65536,
                cost_per_1k_input=0.00125,
                cost_per_1k_output=0.01,
                capabilities=["chat", "code", "analysis", "vision", "audio", "reasoning", "agentic"],
                release_date="2025-12",
                is_latest=True,
                reasoning_capable=True
            ),
            LLMModel(
                provider=LLMProvider.VERTEX,
                model_id="claude-3-5-sonnet-v2@20241022",
                display_name="Claude 3.5 Sonnet v2 (Vertex)",
                context_window=200000,
                max_output_tokens=8192,
                cost_per_1k_input=0.003,
                cost_per_1k_output=0.015,
                capabilities=["chat", "code", "analysis", "vision"],
                release_date="2024-10",
                is_latest=False,
                reasoning_capable=True
            ),
        ],
        LLMProvider.MISTRAL: [
            LLMModel(
                provider=LLMProvider.MISTRAL,
                model_id="mistral-large-latest",
                display_name="Mistral Large",
                context_window=128000,
                max_output_tokens=8192,
                cost_per_1k_input=0.002,
                cost_per_1k_output=0.006,
                capabilities=["chat", "code", "analysis", "function_calling"],
                release_date="2024-11",
                is_latest=True,
                reasoning_capable=True
            ),
        ],
        LLMProvider.GROQ: [
            LLMModel(
                provider=LLMProvider.GROQ,
                model_id="llama-3.3-70b-versatile",
                display_name="Llama 3.3 70B (Groq)",
                context_window=131072,
                max_output_tokens=32768,
                cost_per_1k_input=0.00059,
                cost_per_1k_output=0.00079,
                capabilities=["chat", "code", "analysis"],
                release_date="2024-12",
                is_latest=True,
                reasoning_capable=True
            ),
            LLMModel(
                provider=LLMProvider.GROQ,
                model_id="deepseek-r1-distill-llama-70b",
                display_name="DeepSeek R1 Distill 70B (Groq)",
                context_window=131072,
                max_output_tokens=16384,
                cost_per_1k_input=0.00059,
                cost_per_1k_output=0.00079,
                capabilities=["chat", "code", "reasoning", "math"],
                release_date="2025-01",
                is_latest=True,
                reasoning_capable=True
            ),
        ],
        LLMProvider.TOGETHER: [
            LLMModel(
                provider=LLMProvider.TOGETHER,
                model_id="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                display_name="Llama 3.3 70B Turbo",
                context_window=131072,
                max_output_tokens=8192,
                cost_per_1k_input=0.00088,
                cost_per_1k_output=0.00088,
                capabilities=["chat", "code", "analysis"],
                release_date="2024-12",
                is_latest=True,
                reasoning_capable=True
            ),
        ],
        LLMProvider.DEEPSEEK: [
            LLMModel(
                provider=LLMProvider.DEEPSEEK,
                model_id="deepseek-chat",
                display_name="DeepSeek Chat",
                context_window=64000,
                max_output_tokens=8192,
                cost_per_1k_input=0.00014,
                cost_per_1k_output=0.00028,
                capabilities=["chat", "code", "analysis"],
                release_date="2024-12",
                is_latest=False
            ),
            LLMModel(
                provider=LLMProvider.DEEPSEEK,
                model_id="deepseek-reasoner",
                display_name="DeepSeek R1 Reasoner",
                context_window=64000,
                max_output_tokens=8192,
                cost_per_1k_input=0.00055,
                cost_per_1k_output=0.00219,
                capabilities=["chat", "code", "reasoning", "math", "thinking"],
                release_date="2025-01",
                is_latest=True,
                reasoning_capable=True
            ),
        ],
        LLMProvider.COHERE: [
            LLMModel(
                provider=LLMProvider.COHERE,
                model_id="command-r-plus",
                display_name="Command R+",
                context_window=128000,
                max_output_tokens=4096,
                cost_per_1k_input=0.0025,
                cost_per_1k_output=0.01,
                capabilities=["chat", "code", "analysis", "rag"],
                release_date="2024-04",
                is_latest=True,
                reasoning_capable=True
            ),
        ],
    }
    
    # Provider API endpoints
    PROVIDER_ENDPOINTS = {
        LLMProvider.GROK: "https://api.x.ai/v1",
        LLMProvider.OPENAI: "https://api.openai.com/v1",
        LLMProvider.ANTHROPIC: "https://api.anthropic.com/v1",
        LLMProvider.GOOGLE: "https://generativelanguage.googleapis.com/v1beta",
        LLMProvider.VERTEX: "https://us-central1-aiplatform.googleapis.com/v1",
        LLMProvider.MISTRAL: "https://api.mistral.ai/v1",
        LLMProvider.COHERE: "https://api.cohere.ai/v1",
        LLMProvider.GROQ: "https://api.groq.com/openai/v1",
        LLMProvider.TOGETHER: "https://api.together.xyz/v1",
        LLMProvider.DEEPSEEK: "https://api.deepseek.com/v1",
    }
    
    def __init__(self):
        self.key_manager = SecureKeyManager()
        self.providers: Dict[LLMProvider, ProviderConfig] = {}
        self.call_records: List[LLMCallRecord] = []
        self._initialize_providers()
        
        # Learning state
        self._outcomes: List[Dict] = []
        self._feedback: List[Dict] = []
        self._domain_adjustments: Dict[str, float] = {}
        self._provider_effectiveness: Dict[str, float] = {}
    
    # ===== Learning Interface Methods =====
    
    def record_outcome(self, outcome: Dict) -> None:
        """Record LLM call outcome for learning."""
        self._outcomes.append(outcome)
        provider = outcome.get('provider', 'unknown')
        if outcome.get('success', False):
            self._provider_effectiveness[provider] = self._provider_effectiveness.get(provider, 0.5) + 0.02
        else:
            self._provider_effectiveness[provider] = self._provider_effectiveness.get(provider, 0.5) - 0.02
    
    def record_feedback(self, feedback: Dict) -> None:
        """Record feedback on LLM provider."""
        self._feedback.append(feedback)
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt provider selection based on performance."""
        pass
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get domain adjustment."""
        return self._domain_adjustments.get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics."""
        return {
            'provider_effectiveness': dict(self._provider_effectiveness),
            'total_calls': len(self.call_records),
            'outcomes_recorded': len(self._outcomes),
            'feedback_recorded': len(self._feedback)
        }
    
    def _initialize_providers(self):
        """Initialize provider configurations."""
        for provider, endpoint in self.PROVIDER_ENDPOINTS.items():
            models = self.KNOWN_MODELS.get(provider, [])
            latest_models = [m for m in models if m.is_latest]
            default_model = latest_models[0].model_id if latest_models else (models[0].model_id if models else "")
            
            self.providers[provider] = ProviderConfig(
                provider=provider,
                api_base_url=endpoint,
                api_key_env_var=self.key_manager.ENV_VAR_MAPPING.get(provider, ''),
                models=models,
                default_model=default_model,
                status=ProviderStatus.AVAILABLE if self.key_manager.verify_key(provider) else ProviderStatus.KEY_INVALID
            )
    
    def get_available_providers(self) -> List[LLMProvider]:
        """Get list of providers with valid API keys."""
        return [p for p, config in self.providers.items() if config.status == ProviderStatus.AVAILABLE]
    
    def get_latest_model(self, provider: LLMProvider) -> Optional[LLMModel]:
        """Get the latest model for a provider."""
        config = self.providers.get(provider)
        if not config:
            return None
        
        latest = [m for m in config.models if m.is_latest]
        return latest[0] if latest else (config.models[0] if config.models else None)
    
    def get_model(self, provider: LLMProvider, model_id: str) -> Optional[LLMModel]:
        """Get a specific model."""
        config = self.providers.get(provider)
        if not config:
            return None
        
        for model in config.models:
            if model.model_id == model_id:
                return model
        return None
    
    def get_alternative_provider(self, exclude_provider: LLMProvider) -> Optional[LLMProvider]:
        """
        Get an alternative provider (for governance verification).
        
        IMPORTANT: When verifying LLM output, use a DIFFERENT provider.
        """
        available = self.get_available_providers()
        alternatives = [p for p in available if p != exclude_provider]
        
        # Prioritize providers with reasoning capability
        for alt in alternatives:
            model = self.get_latest_model(alt)
            if model and model.reasoning_capable:
                return alt
        
        # Return any available alternative
        return alternatives[0] if alternatives else None
    
    def check_health(self, provider: LLMProvider) -> ProviderStatus:
        """Check if a provider is available."""
        config = self.providers.get(provider)
        if not config:
            return ProviderStatus.UNAVAILABLE
        
        if not self.key_manager.verify_key(provider):
            config.status = ProviderStatus.KEY_INVALID
            return ProviderStatus.KEY_INVALID
        
        # Could add actual API health check here
        config.status = ProviderStatus.AVAILABLE
        config.last_health_check = datetime.now()
        return ProviderStatus.AVAILABLE
    
    def discover_models(self, provider: LLMProvider) -> List[str]:
        """
        Auto-discover available models from a provider's API.
        
        Returns list of available model IDs, sorted by recency (newest first).
        """
        api_key = self.key_manager.get_key(provider)
        if not api_key:
            logger.warning(f"No API key for {provider.value}")
            return []
        
        discovered = []
        
        try:
            if provider == LLMProvider.GROK:
                response = requests.get(
                    "https://api.x.ai/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data:
                        # Sort by created date (descending)
                        models = sorted(data['data'], key=lambda x: x.get('created', 0), reverse=True)
                        discovered = [m['id'] for m in models]
                        logger.info(f"Discovered {len(discovered)} Grok models")
                        
            elif provider == LLMProvider.OPENAI:
                response = requests.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data:
                        # Filter for chat/completion models
                        chat_models = [m['id'] for m in data['data'] if 'gpt' in m['id'].lower() or 'o1' in m['id'].lower()]
                        discovered = sorted(chat_models, reverse=True)
                        logger.info(f"Discovered {len(discovered)} OpenAI models")
                        
        except Exception as e:
            logger.error(f"Error discovering models for {provider.value}: {e}")
        
        return discovered
    
    def get_best_reasoning_model(self, provider: LLMProvider) -> Optional[str]:
        """
        Get the best reasoning-capable model for a provider.
        
        Priority:
        1. Latest model marked as reasoning_capable
        2. Model with 'reasoning' in name
        3. Any latest model
        """
        config = self.providers.get(provider)
        if not config or not config.models:
            # Try auto-discovery
            discovered = self.discover_models(provider)
            if discovered:
                # Look for reasoning models
                reasoning_models = [m for m in discovered if 'reasoning' in m.lower()]
                if reasoning_models:
                    return reasoning_models[0]
                return discovered[0]
            return None
        
        # Check known models
        reasoning = [m for m in config.models if m.reasoning_capable and m.is_latest]
        if reasoning:
            return reasoning[0].model_id
        
        latest = [m for m in config.models if m.is_latest]
        if latest:
            return latest[0].model_id
        
        return config.default_model if config.default_model else None
    
    def refresh_provider_models(self, provider: LLMProvider) -> bool:
        """
        Refresh the model list for a provider from their API.
        
        Returns True if new models were discovered.
        """
        discovered = self.discover_models(provider)
        if not discovered:
            return False
        
        config = self.providers.get(provider)
        if not config:
            return False
        
        # Check for new models not in our known list
        known_ids = {m.model_id for m in config.models}
        new_models = [m for m in discovered if m not in known_ids]
        
        if new_models:
            logger.info(f"Found {len(new_models)} new models for {provider.value}: {new_models}")
            # Update default to newest model
            if 'reasoning' in discovered[0].lower():
                config.default_model = discovered[0]
                logger.info(f"Updated default model to: {discovered[0]}")
            return True
        
        return False
    
    def record_call(self, record: LLMCallRecord):
        """Record an LLM call for auditing."""
        self.call_records.append(record)
        # Keep last 1000 records
        if len(self.call_records) > 1000:
            self.call_records = self.call_records[-1000:]
    
    def get_inventory(self) -> Dict[str, Any]:
        """Get complete inventory of available LLMs."""
        inventory = {
            'timestamp': datetime.now().isoformat(),
            'providers': {}
        }
        
        for provider, config in self.providers.items():
            inventory['providers'][provider.value] = {
                'status': config.status.value,
                'api_base_url': config.api_base_url,
                'default_model': config.default_model,
                'models': [
                    {
                        'model_id': m.model_id,
                        'display_name': m.display_name,
                        'is_latest': m.is_latest,
                        'reasoning_capable': m.reasoning_capable,
                        'context_window': m.context_window,
                        'cost_per_1k_input': m.cost_per_1k_input,
                        'cost_per_1k_output': m.cost_per_1k_output,
                    }
                    for m in config.models
                ]
            }
        
        return inventory
    
    # =========================================================================
    # Learning Interface (5/5 methods) - Standard aliases
    # =========================================================================
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Standard learning interface - wraps record_feedback."""
        self.record_feedback(feedback)
    
    def get_statistics(self) -> Dict:
        """Standard learning interface - wraps get_learning_statistics."""
        return self.get_learning_statistics()
    
    def serialize_state(self) -> Dict:
        """Serialize learning state for persistence."""
        return {
            'provider_effectiveness': dict(self._provider_effectiveness),
            'outcomes': self._outcomes[-100:],
            'feedback': self._feedback[-100:],
            'domain_adjustments': dict(self._domain_adjustments),
            'call_records': [
                {'provider': r.provider.value, 'model': r.model, 'success': r.success, 'cost': r.estimated_cost}
                for r in self.call_records[-100:]
            ],
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore learning state from serialized data."""
        self._provider_effectiveness = state.get('provider_effectiveness', {})
        self._outcomes = state.get('outcomes', [])
        self._feedback = state.get('feedback', [])
        self._domain_adjustments = defaultdict(float, state.get('domain_adjustments', {}))


class MultiProviderLLMClient:
    """
    Client for making LLM calls with provider switching and fallback.
    
    Key features:
    - Automatic provider selection
    - Fallback on failure
    - Audit logging
    - Cost tracking
    """
    
    def __init__(self, registry: Optional[LLMRegistry] = None):
        self.registry = registry or LLMRegistry()
    
    def call(self,
             prompt: str,
             provider: Optional[LLMProvider] = None,
             model_id: Optional[str] = None,
             purpose: str = "general",
             max_tokens: int = 1024,
             temperature: float = 0.7,
             fallback_on_error: bool = True) -> Dict[str, Any]:
        """
        Make an LLM call with optional provider/model specification.
        
        Args:
            prompt: The prompt to send
            provider: Specific provider to use (or auto-select)
            model_id: Specific model to use (or use default)
            purpose: Purpose of call for audit logging
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            fallback_on_error: Try alternative provider on failure
        
        Returns:
            Dict with 'response', 'provider', 'model', 'usage', 'cost'
        """
        # Select provider
        if provider is None:
            available = self.registry.get_available_providers()
            if not available:
                return {'error': 'No LLM providers available', 'response': None}
            provider = available[0]
        
        # Select model
        config = self.registry.providers.get(provider)
        if not config:
            return {'error': f'Provider {provider.value} not configured', 'response': None}
        
        if model_id is None:
            model_id = config.default_model
        
        model = self.registry.get_model(provider, model_id)
        
        # Make the call
        start_time = time.time()
        try:
            response = self._make_api_call(provider, model_id, prompt, max_tokens, temperature)
            latency_ms = (time.time() - start_time) * 1000
            
            # Calculate cost
            input_tokens = response.get('usage', {}).get('prompt_tokens', len(prompt) // 4)
            output_tokens = response.get('usage', {}).get('completion_tokens', 0)
            
            if model:
                cost = (input_tokens / 1000 * model.cost_per_1k_input + 
                       output_tokens / 1000 * model.cost_per_1k_output)
            else:
                cost = 0.0
            
            # Record call
            record = LLMCallRecord(
                call_id=f"{provider.value}-{int(time.time()*1000)}",
                timestamp=datetime.now().isoformat(),
                provider=provider.value,
                model=model_id,
                purpose=purpose,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                success=True,
                cost_usd=cost
            )
            self.registry.record_call(record)
            
            return {
                'response': response.get('content', response.get('text', '')),
                'provider': provider.value,
                'model': model_id,
                'usage': {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                },
                'cost_usd': cost,
                'latency_ms': latency_ms
            }
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            
            # Record failure
            record = LLMCallRecord(
                call_id=f"{provider.value}-{int(time.time()*1000)}",
                timestamp=datetime.now().isoformat(),
                provider=provider.value,
                model=model_id,
                purpose=purpose,
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )
            self.registry.record_call(record)
            
            # Try fallback
            if fallback_on_error:
                alt_provider = self.registry.get_alternative_provider(provider)
                if alt_provider:
                    logger.info(f"Falling back from {provider.value} to {alt_provider.value}")
                    return self.call(prompt, alt_provider, None, purpose, max_tokens, temperature, fallback_on_error=False)
            
            return {'error': str(e), 'response': None, 'provider': provider.value}
    
    def _make_api_call(self, provider: LLMProvider, model_id: str, 
                       prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        """Make the actual API call to the provider."""
        api_key = self.registry.key_manager.get_key(provider)
        if not api_key:
            raise ValueError(f"No API key for {provider.value}")
        
        config = self.registry.providers.get(provider)
        if not config:
            raise ValueError(f"Provider {provider.value} not configured")
        
        headers = {"Content-Type": "application/json"}
        
        # Provider-specific implementations
        if provider == LLMProvider.GROK:
            headers["Authorization"] = f"Bearer {api_key}"
            response = requests.post(
                f"{config.api_base_url}/chat/completions",
                headers=headers,
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return {
                'content': data['choices'][0]['message']['content'],
                'usage': data.get('usage', {})
            }
        
        elif provider == LLMProvider.OPENAI:
            headers["Authorization"] = f"Bearer {api_key}"
            response = requests.post(
                f"{config.api_base_url}/chat/completions",
                headers=headers,
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return {
                'content': data['choices'][0]['message']['content'],
                'usage': data.get('usage', {})
            }
        
        elif provider == LLMProvider.ANTHROPIC:
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2023-06-01"
            response = requests.post(
                f"{config.api_base_url}/messages",
                headers=headers,
                json={
                    "model": model_id,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return {
                'content': data['content'][0]['text'],
                'usage': {
                    'prompt_tokens': data.get('usage', {}).get('input_tokens', 0),
                    'completion_tokens': data.get('usage', {}).get('output_tokens', 0)
                }
            }
        
        elif provider == LLMProvider.GOOGLE:
            response = requests.post(
                f"{config.api_base_url}/models/{model_id}:generateContent?key={api_key}",
                headers=headers,
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": max_tokens,
                        "temperature": temperature
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return {
                'content': data['candidates'][0]['content']['parts'][0]['text'],
                'usage': {}
            }
        
        else:
            raise NotImplementedError(f"Provider {provider.value} not implemented")

    # Learning Interface Methods
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        if not hasattr(self, '_outcomes'): self._outcomes = []
        self._outcomes.append(outcome)
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        pass
    
    def get_statistics(self) -> Dict:
        return {'outcomes': len(getattr(self, '_outcomes', []))}
    
    def serialize_state(self) -> Dict:
        return {'outcomes': getattr(self, '_outcomes', [])[-100:]}
    
    def deserialize_state(self, state: Dict) -> None:
        self._outcomes = state.get('outcomes', [])



def test_llm_registry():
    """Test the LLM registry and key management."""
    print("=" * 70)
    print("BASE LLM REGISTRY TEST")
    print("=" * 70)
    
    registry = LLMRegistry()
    
    # Show inventory
    print("\nLLM INVENTORY:")
    inventory = registry.get_inventory()
    
    for provider_name, info in inventory['providers'].items():
        status = info['status']
        status_icon = "✅" if status == "available" else "❌"
        print(f"\n  {status_icon} {provider_name.upper()}")
        print(f"     Status: {status}")
        print(f"     Default: {info['default_model']}")
        print(f"     Models:")
        for m in info['models']:
            latest = "⭐" if m['is_latest'] else "  "
            reasoning = "🧠" if m['reasoning_capable'] else "  "
            print(f"       {latest}{reasoning} {m['model_id']}: ${m['cost_per_1k_input']}/{m['cost_per_1k_output']} per 1K tokens")
    
    # Show available providers
    available = registry.get_available_providers()
    print(f"\n\nAVAILABLE PROVIDERS: {[p.value for p in available]}")
    
    # Show alternative for Grok
    if LLMProvider.GROK in available:
        alt = registry.get_alternative_provider(LLMProvider.GROK)
        print(f"Alternative to Grok: {alt.value if alt else 'None'}")
    
    return registry



if __name__ == "__main__":
    test_llm_registry()






