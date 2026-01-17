"""
BASE Tenant Manager

Manages multi-tenant operations including:
- Tenant registration and authentication
- Per-tenant LLM configuration
- Usage tracking and rate limiting
- Tenant isolation
"""

import os
import hashlib
import secrets
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class TenantTier(Enum):
    """Tenant subscription tiers."""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass
class TenantLLMConfig:
    """LLM configuration for a tenant."""
    provider: str
    api_key_encrypted: str  # In production, encrypt this
    model_name: Optional[str] = None
    is_active: bool = True
    priority: int = 0
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TenantUsage:
    """Usage metrics for a tenant."""
    api_calls: int = 0
    audits_performed: int = 0
    llm_tokens_used: int = 0
    last_activity: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "api_calls": self.api_calls,
            "audits_performed": self.audits_performed,
            "llm_tokens_used": self.llm_tokens_used,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None
        }


@dataclass
class Tenant:
    """Tenant entity."""
    id: str
    name: str
    api_key: str
    api_key_hash: str
    tier: TenantTier = TenantTier.FREE
    status: str = "active"
    llm_configs: Dict[str, TenantLLMConfig] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    usage: TenantUsage = field(default_factory=TenantUsage)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "tier": self.tier.value,
            "status": self.status,
            "llm_providers": list(self.llm_configs.keys()),
            "settings": self.settings,
            "usage": self.usage.to_dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


# Rate limits by tier
TIER_LIMITS = {
    TenantTier.FREE: {
        "requests_per_minute": 10,
        "requests_per_day": 100,
        "max_tokens_per_request": 1000,
        "llm_providers": 1
    },
    TenantTier.STARTER: {
        "requests_per_minute": 60,
        "requests_per_day": 1000,
        "max_tokens_per_request": 4000,
        "llm_providers": 3
    },
    TenantTier.PROFESSIONAL: {
        "requests_per_minute": 300,
        "requests_per_day": 10000,
        "max_tokens_per_request": 16000,
        "llm_providers": 5
    },
    TenantTier.ENTERPRISE: {
        "requests_per_minute": 1000,
        "requests_per_day": 100000,
        "max_tokens_per_request": 100000,
        "llm_providers": 10
    }
}


class TenantManager:
    """
    Manages multi-tenant operations for BASE.
    
    In production, this would interface with PostgreSQL.
    For development, uses in-memory storage with file persistence.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize tenant manager.
        
        Args:
            storage_path: Path to persist tenant data (development only)
        """
        self.storage_path = storage_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "data", "tenants.json"
        )
        self._tenants: Dict[str, Tenant] = {}
        self._api_key_index: Dict[str, str] = {}  # api_key_hash -> tenant_id
        
        self._load_tenants()
        self._ensure_default_tenant()
    
    def _load_tenants(self) -> None:
        """Load tenants from storage."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    for tenant_data in data.get("tenants", []):
                        tenant = self._dict_to_tenant(tenant_data)
                        self._tenants[tenant.id] = tenant
                        self._api_key_index[tenant.api_key_hash] = tenant.id
                logger.info(f"Loaded {len(self._tenants)} tenants from storage")
        except Exception as e:
            logger.warning(f"Failed to load tenants: {e}")
    
    def _save_tenants(self) -> None:
        """Save tenants to storage."""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            data = {
                "tenants": [self._tenant_to_dict(t) for t in self._tenants.values()],
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save tenants: {e}")
    
    def _tenant_to_dict(self, tenant: Tenant) -> Dict[str, Any]:
        """Convert tenant to dictionary for storage."""
        return {
            "id": tenant.id,
            "name": tenant.name,
            "api_key": tenant.api_key,
            "api_key_hash": tenant.api_key_hash,
            "tier": tenant.tier.value,
            "status": tenant.status,
            "llm_configs": {
                k: {
                    "provider": v.provider,
                    "api_key_encrypted": v.api_key_encrypted,
                    "model_name": v.model_name,
                    "is_active": v.is_active,
                    "priority": v.priority,
                    "settings": v.settings
                } for k, v in tenant.llm_configs.items()
            },
            "settings": tenant.settings,
            "usage": tenant.usage.to_dict(),
            "created_at": tenant.created_at.isoformat(),
            "updated_at": tenant.updated_at.isoformat()
        }
    
    def _dict_to_tenant(self, data: Dict[str, Any]) -> Tenant:
        """Convert dictionary to tenant."""
        llm_configs = {}
        for k, v in data.get("llm_configs", {}).items():
            llm_configs[k] = TenantLLMConfig(
                provider=v["provider"],
                api_key_encrypted=v["api_key_encrypted"],
                model_name=v.get("model_name"),
                is_active=v.get("is_active", True),
                priority=v.get("priority", 0),
                settings=v.get("settings", {})
            )
        
        usage_data = data.get("usage", {})
        usage = TenantUsage(
            api_calls=usage_data.get("api_calls", 0),
            audits_performed=usage_data.get("audits_performed", 0),
            llm_tokens_used=usage_data.get("llm_tokens_used", 0),
            last_activity=datetime.fromisoformat(usage_data["last_activity"]) 
                if usage_data.get("last_activity") else None
        )
        
        return Tenant(
            id=data["id"],
            name=data["name"],
            api_key=data["api_key"],
            api_key_hash=data["api_key_hash"],
            tier=TenantTier(data.get("tier", "free")),
            status=data.get("status", "active"),
            llm_configs=llm_configs,
            settings=data.get("settings", {}),
            usage=usage,
            created_at=datetime.fromisoformat(data["created_at"]) 
                if isinstance(data.get("created_at"), str) else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"]) 
                if isinstance(data.get("updated_at"), str) else datetime.now(timezone.utc)
        )
    
    def _ensure_default_tenant(self) -> None:
        """Ensure a default development tenant exists."""
        if not self._tenants:
            self.create_tenant(
                name="Development",
                tier=TenantTier.ENTERPRISE,
                api_key="dev-api-key-for-testing-only"
            )
    
    def _hash_api_key(self, api_key: str) -> str:
        """Hash an API key."""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def _generate_api_key(self) -> str:
        """Generate a new API key."""
        return f"base_{secrets.token_hex(24)}"
    
    def _generate_tenant_id(self) -> str:
        """Generate a new tenant ID."""
        return f"tenant_{secrets.token_hex(8)}"
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    def create_tenant(
        self,
        name: str,
        tier: TenantTier = TenantTier.FREE,
        api_key: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> Tenant:
        """
        Create a new tenant.
        
        Args:
            name: Tenant name
            tier: Subscription tier
            api_key: Optional custom API key (for development)
            settings: Optional custom settings
        
        Returns:
            Created tenant
        """
        tenant_id = self._generate_tenant_id()
        api_key = api_key or self._generate_api_key()
        api_key_hash = self._hash_api_key(api_key)
        
        tenant = Tenant(
            id=tenant_id,
            name=name,
            api_key=api_key,
            api_key_hash=api_key_hash,
            tier=tier,
            settings=settings or {}
        )
        
        self._tenants[tenant_id] = tenant
        self._api_key_index[api_key_hash] = tenant_id
        self._save_tenants()
        
        logger.info(f"Created tenant: {name} (id={tenant_id}, tier={tier.value})")
        return tenant
    
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        return self._tenants.get(tenant_id)
    
    def get_tenant_by_api_key(self, api_key: str) -> Optional[Tenant]:
        """Get tenant by API key."""
        api_key_hash = self._hash_api_key(api_key)
        tenant_id = self._api_key_index.get(api_key_hash)
        if tenant_id:
            return self._tenants.get(tenant_id)
        return None
    
    def authenticate(self, api_key: str) -> Optional[Tenant]:
        """
        Authenticate a request by API key.
        
        Args:
            api_key: The API key to authenticate
        
        Returns:
            Tenant if authenticated, None otherwise
        """
        tenant = self.get_tenant_by_api_key(api_key)
        if tenant and tenant.status == "active":
            return tenant
        return None
    
    def list_tenants(self) -> List[Tenant]:
        """List all tenants."""
        return list(self._tenants.values())
    
    def update_tenant(
        self,
        tenant_id: str,
        name: Optional[str] = None,
        tier: Optional[TenantTier] = None,
        status: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> Optional[Tenant]:
        """Update tenant information."""
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return None
        
        if name:
            tenant.name = name
        if tier:
            tenant.tier = tier
        if status:
            tenant.status = status
        if settings:
            tenant.settings.update(settings)
        
        tenant.updated_at = datetime.now(timezone.utc)
        self._save_tenants()
        return tenant
    
    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant (soft delete)."""
        tenant = self._tenants.get(tenant_id)
        if tenant:
            tenant.status = "deleted"
            tenant.updated_at = datetime.now(timezone.utc)
            self._save_tenants()
            return True
        return False
    
    # =========================================================================
    # LLM Configuration
    # =========================================================================
    
    def add_llm_config(
        self,
        tenant_id: str,
        provider: str,
        api_key: str,
        model_name: Optional[str] = None,
        priority: int = 0,
        settings: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add LLM configuration for a tenant.
        
        Args:
            tenant_id: Tenant ID
            provider: LLM provider name
            api_key: Provider API key
            model_name: Optional model name
            priority: Priority for multi-track (lower = higher priority)
            settings: Optional provider settings
        
        Returns:
            True if successful
        """
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False
        
        # Check tier limits
        limits = TIER_LIMITS[tenant.tier]
        if len(tenant.llm_configs) >= limits["llm_providers"]:
            logger.warning(f"Tenant {tenant_id} has reached LLM provider limit")
            return False
        
        # In production, encrypt the API key
        config = TenantLLMConfig(
            provider=provider,
            api_key_encrypted=api_key,  # TODO: Encrypt in production
            model_name=model_name,
            priority=priority,
            settings=settings or {}
        )
        
        tenant.llm_configs[provider] = config
        tenant.updated_at = datetime.now(timezone.utc)
        self._save_tenants()
        
        logger.info(f"Added LLM config for tenant {tenant_id}: {provider}")
        return True
    
    def remove_llm_config(self, tenant_id: str, provider: str) -> bool:
        """Remove LLM configuration for a tenant."""
        tenant = self._tenants.get(tenant_id)
        if tenant and provider in tenant.llm_configs:
            del tenant.llm_configs[provider]
            tenant.updated_at = datetime.now(timezone.utc)
            self._save_tenants()
            return True
        return False
    
    def get_tenant_llm_key(self, tenant_id: str, provider: str) -> Optional[str]:
        """Get LLM API key for a tenant and provider."""
        tenant = self._tenants.get(tenant_id)
        if tenant and provider in tenant.llm_configs:
            config = tenant.llm_configs[provider]
            if config.is_active:
                return config.api_key_encrypted  # TODO: Decrypt in production
        return None
    
    # =========================================================================
    # Usage Tracking
    # =========================================================================
    
    def record_usage(
        self,
        tenant_id: str,
        api_calls: int = 1,
        audits: int = 0,
        tokens: int = 0
    ) -> None:
        """Record usage for a tenant."""
        tenant = self._tenants.get(tenant_id)
        if tenant:
            tenant.usage.api_calls += api_calls
            tenant.usage.audits_performed += audits
            tenant.usage.llm_tokens_used += tokens
            tenant.usage.last_activity = datetime.now(timezone.utc)
            # Save periodically, not on every call in production
    
    def check_rate_limit(self, tenant_id: str) -> bool:
        """
        Check if tenant is within rate limits.
        
        Returns:
            True if within limits, False if rate limited
        """
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False
        
        limits = TIER_LIMITS[tenant.tier]
        # In production, implement proper rate limiting with Redis
        # For now, just return True
        return True
    
    def get_usage_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Get usage statistics for a tenant."""
        tenant = self._tenants.get(tenant_id)
        if tenant:
            return {
                "tenant_id": tenant_id,
                "tier": tenant.tier.value,
                "limits": TIER_LIMITS[tenant.tier],
                "usage": tenant.usage.to_dict()
            }
        return {}

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


# Global instance
_tenant_manager: Optional[TenantManager] = None


def get_tenant_manager() -> TenantManager:
    """Get the global tenant manager instance."""
    global _tenant_manager
    if _tenant_manager is None:
        _tenant_manager = TenantManager()
    return _tenant_manager

