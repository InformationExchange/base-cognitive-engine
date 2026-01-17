"""
BASE LLM-Aware Learning Persistence

Ensures learnings persist and transfer appropriately when switching between LLMs.

Key Features:
1. Per-LLM learning namespaces (learnings specific to each LLM)
2. Cross-LLM transferable learnings (common patterns that apply to all LLMs)
3. LLM-specific bias profiles (each LLM has unique failure patterns)
4. Learning portability scoring (how well learnings transfer between LLMs)

Patent Alignment:
- NOVEL-8: Cross-LLM Governance Orchestration
- PPA3-Inv2: Adaptive Thresholds (per-LLM adjustments)
- NOVEL-19: LLM Registry integration
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from enum import Enum
import json


class LLMProviderType(Enum):
    """Supported LLM providers."""
    GROK = "grok"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MISTRAL = "mistral"
    COHERE = "cohere"
    META = "meta"
    LOCAL = "local"
    UNKNOWN = "unknown"


class LearningType(Enum):
    """Types of learnings."""
    PATTERN = "pattern"              # Regex/keyword patterns
    THRESHOLD = "threshold"          # Detection thresholds
    DIMENSION = "dimension"          # Dimensional relevance
    BIAS_PROFILE = "bias_profile"    # LLM-specific biases
    EFFECTIVENESS = "effectiveness"  # Pattern effectiveness rates
    CIRCUMSTANCE = "circumstance"    # Circumstance-specific learnings


class TransferabilityLevel(Enum):
    """How transferable a learning is between LLMs."""
    UNIVERSAL = "universal"      # Applies to all LLMs (e.g., "100%" detection)
    SIMILAR_ARCH = "similar"     # Applies to similar architectures
    LLM_SPECIFIC = "specific"    # Only applies to one LLM
    UNKNOWN = "unknown"          # Not yet determined


@dataclass
class LearningRecord:
    """A single learning record with LLM context."""
    learning_id: str
    learning_type: LearningType
    content: Dict[str, Any]          # The actual learning data
    
    # LLM context
    source_llm: LLMProviderType
    source_model: str                # Specific model (e.g., "grok-4-1-fast-reasoning")
    applicable_llms: Set[LLMProviderType] = field(default_factory=set)
    
    # Effectiveness tracking per LLM
    effectiveness_per_llm: Dict[str, float] = field(default_factory=dict)
    
    # Transferability
    transferability: TransferabilityLevel = TransferabilityLevel.UNKNOWN
    transfer_success_rate: float = 0.0
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    total_applications: int = 0
    successful_applications: int = 0
    
    def get_effectiveness(self, llm: LLMProviderType) -> float:
        """Get effectiveness for a specific LLM."""
        llm_key = llm.value
        if llm_key in self.effectiveness_per_llm:
            return self.effectiveness_per_llm[llm_key]
        # Default to source LLM effectiveness or base rate
        if self.source_llm.value in self.effectiveness_per_llm:
            return self.effectiveness_per_llm[self.source_llm.value] * 0.8  # 20% penalty for untested
        return 0.5  # Unknown
    
    def record_application(self, llm: LLMProviderType, was_successful: bool):
        """Record an application of this learning."""
        self.total_applications += 1
        if was_successful:
            self.successful_applications += 1
        
        # Update per-LLM effectiveness
        llm_key = llm.value
        if llm_key not in self.effectiveness_per_llm:
            self.effectiveness_per_llm[llm_key] = 0.5
        
        # Exponential moving average
        alpha = 0.1
        self.effectiveness_per_llm[llm_key] = (
            alpha * (1.0 if was_successful else 0.0) + 
            (1 - alpha) * self.effectiveness_per_llm[llm_key]
        )
        
        # Update transferability assessment
        self._assess_transferability()
        self.last_updated = datetime.now().isoformat()
    
    def _assess_transferability(self):
        """Assess how transferable this learning is."""
        if len(self.effectiveness_per_llm) < 2:
            self.transferability = TransferabilityLevel.UNKNOWN
            return
        
        # Calculate variance in effectiveness across LLMs
        values = list(self.effectiveness_per_llm.values())
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        
        if variance < 0.05 and mean > 0.6:
            self.transferability = TransferabilityLevel.UNIVERSAL
            self.transfer_success_rate = mean
        elif variance < 0.15:
            self.transferability = TransferabilityLevel.SIMILAR_ARCH
            self.transfer_success_rate = mean
        else:
            self.transferability = TransferabilityLevel.LLM_SPECIFIC
            self.transfer_success_rate = max(values)
    
    def to_dict(self) -> Dict:
        return {
            "learning_id": self.learning_id,
            "learning_type": self.learning_type.value,
            "content": self.content,
            "source_llm": self.source_llm.value,
            "source_model": self.source_model,
            "applicable_llms": [l.value for l in self.applicable_llms],
            "effectiveness_per_llm": self.effectiveness_per_llm,
            "transferability": self.transferability.value,
            "transfer_success_rate": self.transfer_success_rate,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "total_applications": self.total_applications,
            "successful_applications": self.successful_applications
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LearningRecord':
        return cls(
            learning_id=data["learning_id"],
            learning_type=LearningType(data["learning_type"]),
            content=data["content"],
            source_llm=LLMProviderType(data["source_llm"]),
            source_model=data["source_model"],
            applicable_llms={LLMProviderType(l) for l in data.get("applicable_llms", [])},
            effectiveness_per_llm=data.get("effectiveness_per_llm", {}),
            transferability=TransferabilityLevel(data.get("transferability", "unknown")),
            transfer_success_rate=data.get("transfer_success_rate", 0.0),
            created_at=data.get("created_at", datetime.now().isoformat()),
            last_updated=data.get("last_updated", datetime.now().isoformat()),
            total_applications=data.get("total_applications", 0),
            successful_applications=data.get("successful_applications", 0)
        )


@dataclass
class LLMBiasProfile:
    """Known bias profile for a specific LLM."""
    llm_provider: LLMProviderType
    model_id: str
    
    # Known bias patterns
    common_biases: List[str] = field(default_factory=list)
    bias_severity: Dict[str, float] = field(default_factory=dict)  # bias_type -> severity
    
    # Detection adjustments needed
    threshold_adjustments: Dict[str, float] = field(default_factory=dict)
    
    # Performance profile
    avg_overconfidence: float = 0.0          # How often it's overconfident
    avg_hallucination_rate: float = 0.0       # Hallucination frequency
    avg_sycophancy_score: float = 0.0         # People-pleasing tendency
    
    # Learned patterns specific to this LLM
    specific_patterns: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "llm_provider": self.llm_provider.value,
            "model_id": self.model_id,
            "common_biases": self.common_biases,
            "bias_severity": self.bias_severity,
            "threshold_adjustments": self.threshold_adjustments,
            "avg_overconfidence": self.avg_overconfidence,
            "avg_hallucination_rate": self.avg_hallucination_rate,
            "avg_sycophancy_score": self.avg_sycophancy_score,
            "specific_patterns": self.specific_patterns
        }


class LLMAwareLearningManager:
    """
    Manages learning persistence across LLM switches.
    
    Architecture:
    - Universal learnings: Apply to all LLMs
    - Per-LLM learnings: Specific to each provider
    - Transfer learnings: Attempt to apply cross-LLM with scoring
    """
    
    # Known LLM characteristics (starting points)
    LLM_CHARACTERISTICS = {
        LLMProviderType.GROK: {
            "known_biases": ["sycophancy", "overconfidence"],
            "strength_areas": ["reasoning", "code"],
            "weakness_areas": ["medical_specifics", "legal_citations"]
        },
        LLMProviderType.OPENAI: {
            "known_biases": ["safety_overcorrection", "refusal_bias"],
            "strength_areas": ["general_knowledge", "instruction_following"],
            "weakness_areas": ["recent_events", "controversial_topics"]
        },
        LLMProviderType.ANTHROPIC: {
            "known_biases": ["hedging", "safety_conscious"],
            "strength_areas": ["nuanced_reasoning", "long_context"],
            "weakness_areas": ["assertive_claims", "code_execution"]
        },
        LLMProviderType.GOOGLE: {
            "known_biases": ["source_preference", "google_centric"],
            "strength_areas": ["multimodal", "search_integration"],
            "weakness_areas": ["controversial_opinions"]
        },
        LLMProviderType.MISTRAL: {
            "known_biases": ["european_perspective"],
            "strength_areas": ["multilingual", "code"],
            "weakness_areas": ["us_specific_knowledge"]
        }
    }
    
    def __init__(self, storage_path: Path = None):
        """Initialize the LLM-aware learning manager."""
        self.storage_path = storage_path or Path("llm_aware_learning")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Learning storage
        self.universal_learnings: Dict[str, LearningRecord] = {}
        self.per_llm_learnings: Dict[str, Dict[str, LearningRecord]] = {}  # llm -> {id -> learning}
        self.bias_profiles: Dict[str, LLMBiasProfile] = {}
        
        # Current LLM context
        self.current_llm: Optional[LLMProviderType] = None
        self.current_model: Optional[str] = None
        
        # Statistics
        self.transfer_attempts: int = 0
        self.successful_transfers: int = 0
        
        self._load_state()
    
    def set_current_llm(self, provider: LLMProviderType, model: str):
        """Set the current LLM being governed."""
        self.current_llm = provider
        self.current_model = model
        
        # Initialize per-LLM storage if needed
        if provider.value not in self.per_llm_learnings:
            self.per_llm_learnings[provider.value] = {}
        
        # Initialize bias profile if needed
        if provider.value not in self.bias_profiles:
            self._initialize_bias_profile(provider, model)
    
    def _initialize_bias_profile(self, provider: LLMProviderType, model: str):
        """Initialize a bias profile from known characteristics."""
        chars = self.LLM_CHARACTERISTICS.get(provider, {})
        
        profile = LLMBiasProfile(
            llm_provider=provider,
            model_id=model,
            common_biases=chars.get("known_biases", []),
            bias_severity={b: 0.5 for b in chars.get("known_biases", [])},
            threshold_adjustments={}
        )
        
        self.bias_profiles[provider.value] = profile
    
    def add_learning(
        self,
        learning_id: str,
        learning_type: LearningType,
        content: Dict[str, Any],
        is_universal: bool = False
    ) -> LearningRecord:
        """
        Add a new learning record.
        
        Args:
            learning_id: Unique identifier for this learning
            learning_type: Type of learning
            content: The learning data
            is_universal: If True, applies to all LLMs
        """
        if not self.current_llm:
            raise ValueError("Must set current LLM before adding learnings")
        
        record = LearningRecord(
            learning_id=learning_id,
            learning_type=learning_type,
            content=content,
            source_llm=self.current_llm,
            source_model=self.current_model or "unknown",
            applicable_llms={self.current_llm} if not is_universal else set(LLMProviderType),
            effectiveness_per_llm={self.current_llm.value: 0.7}  # Initial estimate
        )
        
        if is_universal:
            self.universal_learnings[learning_id] = record
        else:
            self.per_llm_learnings[self.current_llm.value][learning_id] = record
        
        self._save_state()
        return record
    
    def get_applicable_learnings(
        self,
        learning_type: Optional[LearningType] = None,
        min_effectiveness: float = 0.5
    ) -> List[LearningRecord]:
        """
        Get all learnings applicable to the current LLM.
        
        Returns learnings sorted by effectiveness for current LLM.
        """
        if not self.current_llm:
            return []
        
        applicable = []
        
        # Universal learnings
        for learning in self.universal_learnings.values():
            if learning_type and learning.learning_type != learning_type:
                continue
            effectiveness = learning.get_effectiveness(self.current_llm)
            if effectiveness >= min_effectiveness:
                applicable.append(learning)
        
        # Per-LLM learnings for current LLM
        llm_key = self.current_llm.value
        if llm_key in self.per_llm_learnings:
            for learning in self.per_llm_learnings[llm_key].values():
                if learning_type and learning.learning_type != learning_type:
                    continue
                effectiveness = learning.get_effectiveness(self.current_llm)
                if effectiveness >= min_effectiveness:
                    applicable.append(learning)
        
        # Attempt to transfer learnings from other LLMs
        for other_llm, learnings in self.per_llm_learnings.items():
            if other_llm == llm_key:
                continue
            
            for learning in learnings.values():
                if learning_type and learning.learning_type != learning_type:
                    continue
                
                # Only transfer if transferability is proven
                if learning.transferability in [TransferabilityLevel.UNIVERSAL, TransferabilityLevel.SIMILAR_ARCH]:
                    effectiveness = learning.get_effectiveness(self.current_llm)
                    if effectiveness >= min_effectiveness:
                        applicable.append(learning)
        
        # Sort by effectiveness
        applicable.sort(
            key=lambda l: l.get_effectiveness(self.current_llm),
            reverse=True
        )
        
        return applicable
    
    def record_learning_outcome(
        self,
        learning_id: str,
        was_successful: bool
    ):
        """Record the outcome of applying a learning."""
        if not self.current_llm:
            return
        
        # Find the learning
        learning = None
        if learning_id in self.universal_learnings:
            learning = self.universal_learnings[learning_id]
        else:
            for llm_learnings in self.per_llm_learnings.values():
                if learning_id in llm_learnings:
                    learning = llm_learnings[learning_id]
                    break
        
        if learning:
            learning.record_application(self.current_llm, was_successful)
            
            # Track transfer statistics
            if learning.source_llm != self.current_llm:
                self.transfer_attempts += 1
                if was_successful:
                    self.successful_transfers += 1
            
            self._save_state()
    
    def get_bias_profile(self, llm: Optional[LLMProviderType] = None) -> Optional[LLMBiasProfile]:
        """Get bias profile for an LLM."""
        llm = llm or self.current_llm
        if not llm:
            return None
        return self.bias_profiles.get(llm.value)
    
    def update_bias_profile(
        self,
        bias_type: str,
        severity: float,
        llm: Optional[LLMProviderType] = None
    ):
        """Update bias severity for an LLM."""
        llm = llm or self.current_llm
        if not llm:
            return
        
        profile = self.get_bias_profile(llm)
        if profile:
            profile.bias_severity[bias_type] = severity
            if bias_type not in profile.common_biases:
                profile.common_biases.append(bias_type)
            self._save_state()
    
    def get_transfer_statistics(self) -> Dict[str, Any]:
        """Get statistics on learning transfer between LLMs."""
        return {
            "transfer_attempts": self.transfer_attempts,
            "successful_transfers": self.successful_transfers,
            "transfer_success_rate": (
                self.successful_transfers / self.transfer_attempts 
                if self.transfer_attempts > 0 else 0.0
            ),
            "universal_learnings_count": len(self.universal_learnings),
            "per_llm_learnings": {
                llm: len(learnings) 
                for llm, learnings in self.per_llm_learnings.items()
            },
            "bias_profiles": list(self.bias_profiles.keys())
        }
    
    def transfer_learnings(
        self,
        from_llm: LLMProviderType,
        to_llm: LLMProviderType,
        transferability_threshold: float = 0.5,
        min_effectiveness: float = 0.6
    ) -> Dict[str, Any]:
        """
        Transfer learnings from one LLM to another.
        
        Only transfers learnings that are:
        1. UNIVERSAL transferability level
        2. SIMILAR_ARCH if LLMs share architecture characteristics
        3. Effectiveness above threshold on source LLM
        
        Args:
            from_llm: Source LLM provider
            to_llm: Target LLM provider
            transferability_threshold: Minimum transfer success rate to attempt
            min_effectiveness: Minimum effectiveness on source LLM
        
        Returns:
            Report on transfer results
        """
        from_key = from_llm.value
        to_key = to_llm.value
        
        if from_key not in self.per_llm_learnings:
            return {"transferred": 0, "skipped": 0, "reason": "No learnings from source LLM"}
        
        # Initialize target LLM storage if needed
        if to_key not in self.per_llm_learnings:
            self.per_llm_learnings[to_key] = {}
        
        transferred = []
        skipped = []
        
        # Get source learnings
        source_learnings = list(self.per_llm_learnings[from_key].values())
        
        # Also check universal learnings
        source_learnings.extend(self.universal_learnings.values())
        
        for learning in source_learnings:
            # Check transferability level
            if learning.transferability == TransferabilityLevel.LLM_SPECIFIC:
                skipped.append({
                    "id": learning.learning_id,
                    "reason": "LLM-specific (not transferable)"
                })
                continue
            
            # Check effectiveness on source
            source_effectiveness = learning.get_effectiveness(from_llm)
            if source_effectiveness < min_effectiveness:
                skipped.append({
                    "id": learning.learning_id,
                    "reason": f"Low effectiveness ({source_effectiveness:.2f} < {min_effectiveness})"
                })
                continue
            
            # Check transfer success rate
            if learning.transfer_success_rate < transferability_threshold:
                # If unknown, give it a chance for universal/similar learnings
                if learning.transferability not in [TransferabilityLevel.UNIVERSAL, TransferabilityLevel.SIMILAR_ARCH]:
                    skipped.append({
                        "id": learning.learning_id,
                        "reason": f"Low transfer rate ({learning.transfer_success_rate:.2f})"
                    })
                    continue
            
            # Check if already exists in target
            if learning.learning_id in self.per_llm_learnings[to_key]:
                skipped.append({
                    "id": learning.learning_id,
                    "reason": "Already exists in target"
                })
                continue
            
            # Transfer the learning
            # Clone with adjusted effectiveness estimate
            transferred_learning = LearningRecord(
                learning_id=learning.learning_id,
                learning_type=learning.learning_type,
                content=learning.content.copy(),
                source_llm=learning.source_llm,
                source_model=learning.source_model,
                applicable_llms=learning.applicable_llms | {to_llm},
                effectiveness_per_llm={
                    **learning.effectiveness_per_llm,
                    to_key: source_effectiveness * 0.7  # Conservative estimate
                },
                transferability=learning.transferability,
                transfer_success_rate=learning.transfer_success_rate
            )
            
            self.per_llm_learnings[to_key][learning.learning_id] = transferred_learning
            transferred.append({
                "id": learning.learning_id,
                "type": learning.learning_type.value,
                "estimated_effectiveness": source_effectiveness * 0.7
            })
        
        # Track statistics
        self.transfer_attempts += len(source_learnings)
        self.successful_transfers += len(transferred)
        
        self._save_state()
        
        return {
            "from_llm": from_key,
            "to_llm": to_key,
            "transferred": len(transferred),
            "skipped": len(skipped),
            "transferred_learnings": transferred,
            "skipped_learnings": skipped[:10],  # Limit to first 10
            "success_rate": len(transferred) / max(len(source_learnings), 1)
        }
    
    def get_portability_report(self) -> Dict[str, Any]:
        """Generate a report on learning portability."""
        report = {
            "total_learnings": (
                len(self.universal_learnings) + 
                sum(len(l) for l in self.per_llm_learnings.values())
            ),
            "by_transferability": {
                level.value: 0 for level in TransferabilityLevel
            },
            "by_llm": {},
            "recommendations": []
        }
        
        # Count by transferability
        for learning in self.universal_learnings.values():
            report["by_transferability"][learning.transferability.value] += 1
        
        for llm, learnings in self.per_llm_learnings.items():
            report["by_llm"][llm] = len(learnings)
            for learning in learnings.values():
                report["by_transferability"][learning.transferability.value] += 1
        
        # Recommendations
        universal_count = report["by_transferability"]["universal"]
        total = report["total_learnings"]
        
        if total > 0:
            universal_ratio = universal_count / total
            if universal_ratio < 0.3:
                report["recommendations"].append(
                    "Low universal learning ratio - consider testing patterns across multiple LLMs"
                )
            if universal_ratio > 0.7:
                report["recommendations"].append(
                    "High universal learning ratio - learnings should transfer well between LLMs"
                )
        
        return report
    
    def _save_state(self):
        """Save learning state to disk."""
        state = {
            "universal_learnings": {
                lid: l.to_dict() for lid, l in self.universal_learnings.items()
            },
            "per_llm_learnings": {
                llm: {lid: l.to_dict() for lid, l in learnings.items()}
                for llm, learnings in self.per_llm_learnings.items()
            },
            "bias_profiles": {
                llm: p.to_dict() for llm, p in self.bias_profiles.items()
            },
            "statistics": {
                "transfer_attempts": self.transfer_attempts,
                "successful_transfers": self.successful_transfers
            },
            "saved_at": datetime.now().isoformat()
        }
        
        state_file = self.storage_path / "llm_aware_learning.json"
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"[LLMAwareLearning] Save error: {e}")
    
    def _load_state(self):
        """Load learning state from disk."""
        state_file = self.storage_path / "llm_aware_learning.json"
        
        if not state_file.exists():
            return
        
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Load universal learnings
            for lid, ldata in state.get("universal_learnings", {}).items():
                self.universal_learnings[lid] = LearningRecord.from_dict(ldata)
            
            # Load per-LLM learnings
            for llm, learnings in state.get("per_llm_learnings", {}).items():
                self.per_llm_learnings[llm] = {}
                for lid, ldata in learnings.items():
                    self.per_llm_learnings[llm][lid] = LearningRecord.from_dict(ldata)
            
            # Load bias profiles
            for llm, pdata in state.get("bias_profiles", {}).items():
                self.bias_profiles[llm] = LLMBiasProfile(
                    llm_provider=LLMProviderType(pdata["llm_provider"]),
                    model_id=pdata["model_id"],
                    common_biases=pdata.get("common_biases", []),
                    bias_severity=pdata.get("bias_severity", {}),
                    threshold_adjustments=pdata.get("threshold_adjustments", {}),
                    avg_overconfidence=pdata.get("avg_overconfidence", 0.0),
                    avg_hallucination_rate=pdata.get("avg_hallucination_rate", 0.0),
                    avg_sycophancy_score=pdata.get("avg_sycophancy_score", 0.0),
                    specific_patterns=pdata.get("specific_patterns", [])
                )
            
            # Load statistics
            stats = state.get("statistics", {})
            self.transfer_attempts = stats.get("transfer_attempts", 0)
            self.successful_transfers = stats.get("successful_transfers", 0)
            
        except Exception as e:
            print(f"[LLMAwareLearning] Load error: {e}")

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


# Convenience function for integration
def get_llm_provider_type(provider_name: str) -> LLMProviderType:
    """Convert provider name string to enum."""
    name_map = {
        "grok": LLMProviderType.GROK,
        "xai": LLMProviderType.GROK,
        "openai": LLMProviderType.OPENAI,
        "gpt": LLMProviderType.OPENAI,
        "anthropic": LLMProviderType.ANTHROPIC,
        "claude": LLMProviderType.ANTHROPIC,
        "google": LLMProviderType.GOOGLE,
        "gemini": LLMProviderType.GOOGLE,
        "mistral": LLMProviderType.MISTRAL,
        "cohere": LLMProviderType.COHERE,
        "meta": LLMProviderType.META,
        "llama": LLMProviderType.META,
        "local": LLMProviderType.LOCAL,
        "ollama": LLMProviderType.LOCAL
    }
    return name_map.get(provider_name.lower(), LLMProviderType.UNKNOWN)


# Phase 15 Enhancement: Simplified interface for integrated_engine.py
# This wrapper class provides the interface expected by the integrated engine
class LLMAwareLearning:
    """
    Simplified interface for LLM-Aware Learning integration.
    
    This class wraps LLMAwareLearningManager to provide the exact interface
    expected by integrated_engine.py:
    - get_applicable_learnings(llm_provider: str, domain: Optional[str])
    - record_learning_outcome(learning_id: str, success: bool, llm_provider: str, domain: Optional[str])
    - get_bias_profile(llm_provider: str)
    - update_bias_profile(llm_provider: str, bias_type: str, severity: float)
    """
    
    def __init__(self, storage_path: Path = None):
        """Initialize the LLM-Aware Learning system."""
        self.manager = LLMAwareLearningManager(storage_path=storage_path)
    
    def get_applicable_learnings(
        self, 
        llm_provider: str, 
        domain: Optional[str] = None
    ) -> List[LearningRecord]:
        """
        Get learnings applicable to the specified LLM provider.
        
        Args:
            llm_provider: String identifier for LLM (e.g., "grok", "claude")
            domain: Optional domain filter (not used in current impl)
        
        Returns:
            List of applicable LearningRecord objects
        """
        # Set current LLM context
        provider_type = get_llm_provider_type(llm_provider)
        self.manager.set_current_llm(provider_type, llm_provider)
        
        # Get applicable learnings
        return self.manager.get_applicable_learnings()
    
    def record_learning_outcome(
        self,
        learning_id: str,
        success: bool,
        llm_provider: str,
        domain: Optional[str] = None
    ):
        """
        Record the outcome of applying a learning.
        
        Args:
            learning_id: ID of the learning that was applied
            success: Whether the application was successful
            llm_provider: The LLM provider that was used
            domain: Optional domain context
        """
        # Ensure current LLM is set
        provider_type = get_llm_provider_type(llm_provider)
        self.manager.set_current_llm(provider_type, llm_provider)
        
        # Record outcome
        self.manager.record_learning_outcome(learning_id, success)
    
    def get_bias_profile(self, llm_provider: str) -> Optional[LLMBiasProfile]:
        """
        Get the bias profile for an LLM provider.
        
        Args:
            llm_provider: String identifier for LLM
        
        Returns:
            LLMBiasProfile or None if not found
        """
        provider_type = get_llm_provider_type(llm_provider)
        return self.manager.get_bias_profile(provider_type)
    
    def update_bias_profile(
        self,
        llm_provider: str,
        bias_type: str,
        severity: float
    ):
        """
        Update the bias profile for an LLM.
        
        Args:
            llm_provider: String identifier for LLM
            bias_type: Type of bias detected
            severity: Severity score (0-1)
        """
        provider_type = get_llm_provider_type(llm_provider)
        self.manager.set_current_llm(provider_type, llm_provider)
        self.manager.update_bias_profile(bias_type, severity, provider_type)
    
    def add_learning(
        self,
        learning: 'LLMLearning'
    ):
        """
        Add a new learning (compatible with older interface).
        
        Args:
            learning: LLMLearning object to add
        """
        provider_type = get_llm_provider_type(learning.llm_provider or "grok")
        self.manager.set_current_llm(provider_type, learning.llm_provider or "grok")
        self.manager.add_learning(
            learning_id=learning.learning_id,
            learning_type=LearningType.PATTERN,
            content=learning.content if isinstance(learning.content, dict) else {"value": learning.content},
            is_universal=(learning.learning_type == LearningType2.UNIVERSAL if hasattr(learning, 'learning_type') else False)
        )
    
    def get_learning(self, learning_id: str) -> Optional[LearningRecord]:
        """Get a specific learning by ID."""
        if learning_id in self.manager.universal_learnings:
            return self.manager.universal_learnings[learning_id]
        for llm_learnings in self.manager.per_llm_learnings.values():
            if learning_id in llm_learnings:
                return llm_learnings[learning_id]
        return None
    
    def record_detection(
        self,
        llm_provider: str,
        model_name: str,
        pattern_type: str,
        was_correct: bool,
        severity: float = 0.5
    ):
        """
        Simple interface to record a detection outcome.
        
        Args:
            llm_provider: Provider name (e.g., 'grok', 'openai')
            model_name: Model name (e.g., 'grok-3')
            pattern_type: Type of pattern detected (e.g., 'false_completion')
            was_correct: Whether the detection was correct
            severity: Severity score (0-1)
        """
        from datetime import datetime
        learning_id = f"{pattern_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Record the detection as a learning
        provider_type = get_llm_provider_type(llm_provider)
        self.manager.set_current_llm(provider_type, llm_provider)
        
        self.manager.add_learning(
            learning_id=learning_id,
            learning_type=LearningType.PATTERN,
            content={
                "pattern_type": pattern_type,
                "model_name": model_name,
                "was_correct": was_correct,
                "severity": severity
            },
            is_universal=False
        )
        
        # Update bias profile
        if not was_correct:
            self.update_bias_profile(llm_provider, pattern_type, severity)
    
    def get_effectiveness(
        self,
        llm_provider: str,
        model_name: str,
        pattern_type: str
    ) -> float:
        """
        Get effectiveness of a pattern detection for an LLM.
        
        Returns effectiveness score between 0 and 1.
        """
        provider_type = get_llm_provider_type(llm_provider)
        
        # Check if we have learnings for this pattern
        learnings = self.manager.per_llm_learnings.get(provider_type, {})
        relevant = [
            l for l in learnings.values()
            if l.content.get("pattern_type") == pattern_type
        ]
        
        if not relevant:
            return 0.5  # Default effectiveness
        
        # Calculate effectiveness from outcomes
        correct = sum(1 for l in relevant if l.content.get("was_correct", False))
        return correct / len(relevant) if relevant else 0.5
    
    def get_transfer_statistics(self) -> Dict[str, Any]:
        """Get transfer statistics."""
        return self.manager.get_transfer_statistics()
    
    def get_portability_report(self) -> Dict[str, Any]:
        """Get portability report."""
        return self.manager.get_portability_report()

    # Learning Interface
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


# Dataclass for backward compatibility with test interface
@dataclass
class LLMLearning:
    """Simple learning object for test compatibility."""
    learning_id: str
    name: str
    content: Any
    llm_provider: Optional[str] = None
    learning_type: 'LearningType2' = None
    domain: Optional[str] = None
    total_applications: int = 0
    successful_applications: int = 0
    effectiveness_score: float = 0.7


class LearningType2(Enum):
    """Learning types for test compatibility."""
    UNIVERSAL = "universal"
    LLM_SPECIFIC = "llm_specific"
    DOMAIN_SPECIFIC = "domain_specific"


class Transferability(Enum):
    """Transferability levels for test compatibility."""
    UNIVERSAL = "universal"
    SIMILAR = "similar"
    SPECIFIC = "specific"
    UNKNOWN = "unknown"

