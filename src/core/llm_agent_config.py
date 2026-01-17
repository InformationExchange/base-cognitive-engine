"""
LLM Agent Configuration Module

Patent: NOVEL-28 Enhancement - User-Configurable LLM Agents

This module allows users to configure which LLM handles which function within
the BASE governance system. Users can assign different LLMs to different roles
based on their preferences, availability, and cost considerations.

Three Distinct LLM Roles in BASE:
1. LLM for Dimension Identification - Discovers relevant dimensions (complex queries only)
2. LLM for Task Execution - Performs the user's requested task (always)
3. LLM for Governance - Monitors, audits, and improves output (always)

This module manages role assignments for all three.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import os
import tempfile


class AgentRole(Enum):
    """
    Defines the functional roles that LLMs can be assigned to.
    
    Each role represents a distinct function in the BASE pipeline.
    Users can assign different LLMs to different roles.
    """
    # Dimension Identification Roles
    DIMENSIONAL_EXPANDER = "dimensional_expander"      # Identifies relevant dimensions
    DIMENSION_CORRELATOR = "dimension_correlator"      # Finds cross-dimension patterns
    
    # Governance Roles
    CHALLENGER = "challenger"                          # Multi-track adversarial analysis
    BIAS_DETECTOR = "bias_detector"                    # Detects bias in outputs
    FACT_VERIFIER = "fact_verifier"                    # Verifies factual claims
    COMPLETION_VERIFIER = "completion_verifier"        # Verifies completion claims
    
    # Enhancement Roles
    RESPONSE_IMPROVER = "response_improver"            # Improves responses
    HEDGING_GENERATOR = "hedging_generator"            # Adds appropriate hedging
    
    # Learning Roles
    PATTERN_LEARNER = "pattern_learner"                # Learns from outcomes
    THRESHOLD_ADAPTER = "threshold_adapter"            # Adapts detection thresholds


class LLMProvider(Enum):
    """
    Supported LLM providers.
    
    Users can select from these providers for each role.
    """
    CLAUDE = "claude"           # Anthropic Claude
    GPT4 = "gpt4"               # OpenAI GPT-4
    GROK = "grok"               # xAI Grok
    GEMINI = "gemini"           # Google Gemini
    MISTRAL = "mistral"         # Mistral AI
    LLAMA = "llama"             # Meta Llama
    COHERE = "cohere"           # Cohere
    LOCAL = "local"             # Local/self-hosted
    PATTERN_ONLY = "pattern"    # Pattern-based (no LLM) - for simple tasks


@dataclass
class AgentAssignment:
    """
    Maps a role to a specific LLM provider and model.
    
    Includes fallback configuration for resilience.
    """
    role: AgentRole
    primary_provider: LLMProvider
    primary_model: str                              # e.g., "claude-3-opus", "gpt-4-turbo"
    fallback_provider: Optional[LLMProvider] = None
    fallback_model: Optional[str] = None
    temperature: float = 0.3                        # Lower for governance tasks
    max_tokens: int = 2000
    timeout_seconds: int = 30
    enabled: bool = True
    
    # Usage tracking
    invocation_count: int = 0
    fallback_count: int = 0
    last_used: Optional[datetime] = None
    average_latency_ms: float = 0.0


@dataclass
class AgentConfigState:
    """
    Persistent state for agent configuration.
    """
    assignments: Dict[str, AgentAssignment] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"


class AgentConfigManager:
    """
    Manages LLM agent configurations for all BASE roles.
    
    Provides:
    - User-configurable role-to-LLM assignments
    - Fallback handling when primary unavailable
    - Persistent configuration storage
    - Usage tracking and statistics
    
    Patent: NOVEL-28 Enhancement
    """
    
    # Default assignments for each role
    DEFAULT_ASSIGNMENTS: Dict[AgentRole, tuple] = {
        # Dimension roles - pattern-based for simple, LLM for complex
        AgentRole.DIMENSIONAL_EXPANDER: (LLMProvider.CLAUDE, "claude-3-sonnet"),
        AgentRole.DIMENSION_CORRELATOR: (LLMProvider.CLAUDE, "claude-3-sonnet"),
        
        # Governance roles - require LLM
        AgentRole.CHALLENGER: (LLMProvider.GROK, "grok-4-1-fast-reasoning"),
        AgentRole.BIAS_DETECTOR: (LLMProvider.CLAUDE, "claude-3-haiku"),
        AgentRole.FACT_VERIFIER: (LLMProvider.GPT4, "gpt-4-turbo"),
        AgentRole.COMPLETION_VERIFIER: (LLMProvider.CLAUDE, "claude-3-sonnet"),
        
        # Enhancement roles
        AgentRole.RESPONSE_IMPROVER: (LLMProvider.CLAUDE, "claude-3-opus"),
        AgentRole.HEDGING_GENERATOR: (LLMProvider.CLAUDE, "claude-3-haiku"),
        
        # Learning roles
        AgentRole.PATTERN_LEARNER: (LLMProvider.PATTERN_ONLY, "local"),
        AgentRole.THRESHOLD_ADAPTER: (LLMProvider.PATTERN_ONLY, "local"),
    }
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the agent configuration manager.
        
        Args:
            storage_path: Path to persist configuration. Uses temp dir if None.
        """
        if storage_path:
            self.storage_path = storage_path
        else:
            self.storage_path = os.path.join(tempfile.gettempdir(), "base_agent_config.json")
        
        self.assignments: Dict[AgentRole, AgentAssignment] = {}
        self._initialize_defaults()
        self._load_state()
    
    def _initialize_defaults(self):
        """Initialize with default assignments for all roles."""
        for role, (provider, model) in self.DEFAULT_ASSIGNMENTS.items():
            self.assignments[role] = AgentAssignment(
                role=role,
                primary_provider=provider,
                primary_model=model,
                fallback_provider=LLMProvider.PATTERN_ONLY,
                fallback_model="local"
            )
    
    def configure_agent(
        self,
        role: AgentRole,
        provider: LLMProvider,
        model: str,
        fallback_provider: Optional[LLMProvider] = None,
        fallback_model: Optional[str] = None,
        temperature: float = 0.3,
        enabled: bool = True
    ) -> AgentAssignment:
        """
        Configure an LLM agent for a specific role.
        
        Args:
            role: The functional role to configure
            provider: Primary LLM provider
            model: Specific model to use
            fallback_provider: Backup provider if primary fails
            fallback_model: Backup model
            temperature: Generation temperature (lower for governance)
            enabled: Whether this role is enabled
        
        Returns:
            The configured AgentAssignment
        
        Example:
            manager.configure_agent(
                role=AgentRole.CHALLENGER,
                provider=LLMProvider.GROK,
                model="grok-4-1-fast-reasoning",
                fallback_provider=LLMProvider.CLAUDE,
                fallback_model="claude-3-haiku"
            )
        """
        assignment = AgentAssignment(
            role=role,
            primary_provider=provider,
            primary_model=model,
            fallback_provider=fallback_provider or LLMProvider.PATTERN_ONLY,
            fallback_model=fallback_model or "local",
            temperature=temperature,
            enabled=enabled
        )
        
        self.assignments[role] = assignment
        self._save_state()
        
        return assignment
    
    def configure_all(self, config: Dict[AgentRole, Dict[str, Any]]) -> Dict[AgentRole, AgentAssignment]:
        """
        Configure multiple agents at once.
        
        Args:
            config: Dictionary mapping roles to their configuration
        
        Returns:
            Dictionary of configured assignments
        
        Example:
            manager.configure_all({
                AgentRole.CHALLENGER: {
                    "provider": LLMProvider.GROK,
                    "model": "grok-4-1-fast-reasoning"
                },
                AgentRole.DIMENSIONAL_EXPANDER: {
                    "provider": LLMProvider.CLAUDE,
                    "model": "claude-3-opus"
                }
            })
        """
        results = {}
        for role, role_config in config.items():
            results[role] = self.configure_agent(
                role=role,
                provider=role_config.get("provider", LLMProvider.CLAUDE),
                model=role_config.get("model", "claude-3-sonnet"),
                fallback_provider=role_config.get("fallback_provider"),
                fallback_model=role_config.get("fallback_model"),
                temperature=role_config.get("temperature", 0.3),
                enabled=role_config.get("enabled", True)
            )
        return results
    
    def get_agent(self, role: AgentRole) -> Optional[AgentAssignment]:
        """
        Get the agent assignment for a role.
        
        Returns None if role is not configured or disabled.
        """
        assignment = self.assignments.get(role)
        if assignment and assignment.enabled:
            return assignment
        return None
    
    def get_provider_for_role(self, role: AgentRole) -> tuple:
        """
        Get the provider and model for a role.
        
        Returns:
            Tuple of (provider, model) or (PATTERN_ONLY, "local") if unavailable
        """
        assignment = self.get_agent(role)
        if assignment:
            return (assignment.primary_provider, assignment.primary_model)
        return (LLMProvider.PATTERN_ONLY, "local")
    
    def record_invocation(
        self,
        role: AgentRole,
        used_fallback: bool = False,
        latency_ms: float = 0.0
    ):
        """
        Record an invocation for tracking purposes.
        
        Args:
            role: The role that was invoked
            used_fallback: Whether fallback was used
            latency_ms: Latency of the invocation
        """
        if role in self.assignments:
            assignment = self.assignments[role]
            assignment.invocation_count += 1
            assignment.last_used = datetime.now()
            
            if used_fallback:
                assignment.fallback_count += 1
            
            # Rolling average for latency
            if assignment.average_latency_ms == 0:
                assignment.average_latency_ms = latency_ms
            else:
                assignment.average_latency_ms = (
                    assignment.average_latency_ms * 0.9 + latency_ms * 0.1
                )
            
            self._save_state()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get usage statistics for all configured agents.
        
        Returns:
            Dictionary with statistics per role
        """
        stats = {
            "total_roles_configured": len(self.assignments),
            "enabled_roles": sum(1 for a in self.assignments.values() if a.enabled),
            "total_invocations": sum(a.invocation_count for a in self.assignments.values()),
            "total_fallbacks": sum(a.fallback_count for a in self.assignments.values()),
            "roles": {}
        }
        
        for role, assignment in self.assignments.items():
            stats["roles"][role.value] = {
                "provider": assignment.primary_provider.value,
                "model": assignment.primary_model,
                "invocations": assignment.invocation_count,
                "fallbacks": assignment.fallback_count,
                "fallback_rate": (
                    assignment.fallback_count / assignment.invocation_count
                    if assignment.invocation_count > 0 else 0
                ),
                "avg_latency_ms": assignment.average_latency_ms,
                "enabled": assignment.enabled
            }
        
        return stats
    
    def should_use_llm(self, role: AgentRole, complexity: str) -> bool:
        """
        Determine if LLM should be invoked for this role and complexity.
        
        For dimension identification roles:
        - SIMPLE complexity: Use pattern-based (no LLM)
        - MODERATE complexity: Optional LLM
        - COMPLEX complexity: Require LLM
        
        For governance/enhancement roles:
        - Always use LLM if configured
        
        Args:
            role: The role to check
            complexity: Query complexity ("simple", "moderate", "complex")
        
        Returns:
            True if LLM should be invoked
        """
        assignment = self.get_agent(role)
        if not assignment:
            return False
        
        # Governance roles always use LLM
        governance_roles = [
            AgentRole.CHALLENGER,
            AgentRole.BIAS_DETECTOR,
            AgentRole.FACT_VERIFIER,
            AgentRole.COMPLETION_VERIFIER,
            AgentRole.RESPONSE_IMPROVER,
            AgentRole.HEDGING_GENERATOR
        ]
        
        if role in governance_roles:
            return assignment.primary_provider != LLMProvider.PATTERN_ONLY
        
        # Dimension roles depend on complexity
        if role in [AgentRole.DIMENSIONAL_EXPANDER, AgentRole.DIMENSION_CORRELATOR]:
            if complexity == "simple":
                return False  # Pattern-based is sufficient
            elif complexity == "moderate":
                return True  # LLM can help
            else:  # complex
                return True  # LLM required
        
        # Learning roles typically pattern-based
        return assignment.primary_provider != LLMProvider.PATTERN_ONLY
    
    def _load_state(self):
        """Load configuration state from disk."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for role_name, assignment_data in data.get("assignments", {}).items():
                        try:
                            role = AgentRole(role_name)
                            self.assignments[role] = AgentAssignment(
                                role=role,
                                primary_provider=LLMProvider(assignment_data["primary_provider"]),
                                primary_model=assignment_data["primary_model"],
                                fallback_provider=LLMProvider(assignment_data.get("fallback_provider", "pattern")),
                                fallback_model=assignment_data.get("fallback_model", "local"),
                                temperature=assignment_data.get("temperature", 0.3),
                                max_tokens=assignment_data.get("max_tokens", 2000),
                                timeout_seconds=assignment_data.get("timeout_seconds", 30),
                                enabled=assignment_data.get("enabled", True),
                                invocation_count=assignment_data.get("invocation_count", 0),
                                fallback_count=assignment_data.get("fallback_count", 0),
                                average_latency_ms=assignment_data.get("average_latency_ms", 0.0)
                            )
                        except (ValueError, KeyError):
                            continue
        except (json.JSONDecodeError, IOError):
            pass
    
    def _save_state(self):
        """Save configuration state to disk."""
        try:
            data = {
                "version": "1.0.0",
                "updated_at": datetime.now().isoformat(),
                "assignments": {}
            }
            
            for role, assignment in self.assignments.items():
                data["assignments"][role.value] = {
                    "primary_provider": assignment.primary_provider.value,
                    "primary_model": assignment.primary_model,
                    "fallback_provider": assignment.fallback_provider.value if assignment.fallback_provider else "pattern",
                    "fallback_model": assignment.fallback_model or "local",
                    "temperature": assignment.temperature,
                    "max_tokens": assignment.max_tokens,
                    "timeout_seconds": assignment.timeout_seconds,
                    "enabled": assignment.enabled,
                    "invocation_count": assignment.invocation_count,
                    "fallback_count": assignment.fallback_count,
                    "average_latency_ms": assignment.average_latency_ms
                }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError:
            pass


# Convenience function for quick configuration
def create_agent_config(
    challenger: str = "grok",
    expander: str = "claude",
    governance: str = "claude"
) -> AgentConfigManager:
    """
    Create an agent configuration with common presets.
    
    Args:
        challenger: Provider for challenger role ("grok", "claude", "gpt4")
        expander: Provider for dimension expansion ("claude", "gpt4")
        governance: Provider for governance roles ("claude", "gpt4")
    
    Returns:
        Configured AgentConfigManager
    
    Example:
        config = create_agent_config(
            challenger="grok",
            expander="claude",
            governance="claude"
        )
    """
    provider_map = {
        "grok": (LLMProvider.GROK, "grok-4-1-fast-reasoning"),
        "claude": (LLMProvider.CLAUDE, "claude-3-sonnet"),
        "gpt4": (LLMProvider.GPT4, "gpt-4-turbo"),
        "gemini": (LLMProvider.GEMINI, "gemini-pro"),
    }
    
    manager = AgentConfigManager()
    
    # Configure challenger
    if challenger in provider_map:
        provider, model = provider_map[challenger]
        manager.configure_agent(AgentRole.CHALLENGER, provider, model)
    
    # Configure expander
    if expander in provider_map:
        provider, model = provider_map[expander]
        manager.configure_agent(AgentRole.DIMENSIONAL_EXPANDER, provider, model)
        manager.configure_agent(AgentRole.DIMENSION_CORRELATOR, provider, model)
    
    # Configure governance roles
    if governance in provider_map:
        provider, model = provider_map[governance]
        manager.configure_agent(AgentRole.BIAS_DETECTOR, provider, model)
        manager.configure_agent(AgentRole.FACT_VERIFIER, provider, model)
        manager.configure_agent(AgentRole.COMPLETION_VERIFIER, provider, model)
    
    return manager

