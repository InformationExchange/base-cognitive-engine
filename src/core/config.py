"""
BAIS Cognitive Governance Engine v16.2
Configuration and Capability Detection

Supports two deployment modes:
- FULL: Neural embeddings + NLI + ML classifiers (~90% accuracy)
- LITE: Statistical + Rule-based (~70% accuracy, lower cost)

Mode is determined by:
1. BAIS_MODE environment variable (explicit)
2. Auto-detection of installed packages (implicit)
"""

import os
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class DeploymentMode(Enum):
    """Deployment mode determines which detection methods are used."""
    FULL = "full"  # ML-based: embeddings, NLI, classifiers
    LITE = "lite"  # Statistical: TF-IDF, regex, rules
    AUTO = "auto"  # Detect based on installed packages


@dataclass
class Capabilities:
    """Detected capabilities based on installed packages."""
    has_sentence_transformers: bool = False
    has_torch: bool = False
    has_transformers: bool = False
    has_sklearn: bool = False
    has_spacy: bool = False
    
    @property
    def can_use_embeddings(self) -> bool:
        return self.has_sentence_transformers and self.has_torch
    
    @property
    def can_use_nli(self) -> bool:
        return self.has_sentence_transformers and self.has_torch
    
    @property
    def can_use_ml_classifiers(self) -> bool:
        return self.has_sklearn or (self.has_transformers and self.has_torch)
    
    @property
    def recommended_mode(self) -> DeploymentMode:
        if self.can_use_embeddings:
            return DeploymentMode.FULL
        return DeploymentMode.LITE


def detect_capabilities() -> Capabilities:
    """Detect which ML packages are available."""
    caps = Capabilities()
    
    # Check sentence-transformers
    try:
        import sentence_transformers
        caps.has_sentence_transformers = True
    except ImportError:
        pass
    
    # Check torch
    try:
        import torch
        caps.has_torch = True
    except ImportError:
        pass
    
    # Check transformers
    try:
        import transformers
        caps.has_transformers = True
    except ImportError:
        pass
    
    # Check sklearn
    try:
        import sklearn
        caps.has_sklearn = True
    except ImportError:
        pass
    
    # Check spacy
    try:
        import spacy
        caps.has_spacy = True
    except ImportError:
        pass
    
    return caps


@dataclass
class BAISConfig:
    """Configuration for BAIS engine."""
    
    # Mode
    mode: DeploymentMode = DeploymentMode.AUTO
    
    # Detection settings
    use_embeddings: bool = False
    use_nli: bool = False
    use_ml_classifiers: bool = False
    
    # Model names (for FULL mode)
    embedding_model: str = "all-MiniLM-L6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-small"  # Smaller model for efficiency
    
    # Thresholds
    grounding_threshold: float = 0.4
    behavioral_threshold: float = 0.35
    factual_threshold: float = 0.5
    
    # Performance
    max_claims_to_verify: int = 10
    max_similar_cases: int = 5
    
    # Paths (use temp directory by default to avoid read-only filesystem issues)
    data_dir: str = ""  # Will be set dynamically if empty
    
    # LLM
    llm_api_key: Optional[str] = None
    llm_model: str = "grok-4-1-fast-reasoning"
    
    # Learning
    learning_algorithm: str = "oco"
    
    @classmethod
    def from_environment(cls) -> 'BAISConfig':
        """Create config from environment variables."""
        config = cls()
        
        # Get mode from environment
        mode_str = os.environ.get('BAIS_MODE', 'auto').lower()
        if mode_str == 'full':
            config.mode = DeploymentMode.FULL
        elif mode_str == 'lite':
            config.mode = DeploymentMode.LITE
        else:
            config.mode = DeploymentMode.AUTO
        
        # Detect capabilities
        caps = detect_capabilities()
        
        # Set detection methods based on mode
        if config.mode == DeploymentMode.FULL:
            # Full mode: use ML if available, warn if not
            config.use_embeddings = caps.can_use_embeddings
            config.use_nli = caps.can_use_nli
            config.use_ml_classifiers = caps.can_use_ml_classifiers
            
            if not caps.can_use_embeddings:
                print("WARNING: FULL mode requested but ML packages not installed. Falling back to statistical methods.")
        
        elif config.mode == DeploymentMode.LITE:
            # Lite mode: never use ML even if available
            config.use_embeddings = False
            config.use_nli = False
            config.use_ml_classifiers = False
        
        else:  # AUTO
            # Auto mode: use whatever is available
            config.use_embeddings = caps.can_use_embeddings
            config.use_nli = caps.can_use_nli
            config.use_ml_classifiers = caps.can_use_ml_classifiers
        
        # Load other settings from environment
        # Use temp directory if not specified to avoid read-only filesystem issues
        default_data_dir = os.environ.get('BAIS_DATA_DIR', '')
        if not default_data_dir:
            import tempfile
            default_data_dir = tempfile.mkdtemp(prefix="bais_data_")
        config.data_dir = default_data_dir
        config.llm_api_key = os.environ.get('XAI_API_KEY', '')
        config.llm_model = os.environ.get('LLM_MODEL', 'grok-4-1-fast-reasoning')
        config.learning_algorithm = os.environ.get('LEARNING_ALGORITHM', 'oco')
        
        # Model overrides
        config.embedding_model = os.environ.get('EMBEDDING_MODEL', config.embedding_model)
        config.nli_model = os.environ.get('NLI_MODEL', config.nli_model)
        
        return config
    
    def get_mode_description(self) -> str:
        """Get human-readable description of current mode."""
        if self.use_embeddings and self.use_nli:
            return "FULL (Neural ML: embeddings + NLI)"
        elif self.use_embeddings:
            return "FULL (Neural embeddings, rule-based NLI)"
        else:
            return "LITE (Statistical: TF-IDF + rules)"
    
    def get_capabilities_summary(self) -> dict:
        """Get summary of capabilities."""
        caps = detect_capabilities()
        return {
            'mode': self.mode.value,
            'mode_description': self.get_mode_description(),
            'detection_methods': {
                'grounding': 'neural_embeddings' if self.use_embeddings else 'tfidf_statistical',
                'behavioral': 'ml_classifier' if self.use_ml_classifiers else 'regex_patterns',
                'factual': 'nli_entailment' if self.use_nli else 'word_overlap',
                'temporal': 'statistical_patterns'  # Always statistical
            },
            'packages_installed': {
                'sentence_transformers': caps.has_sentence_transformers,
                'torch': caps.has_torch,
                'transformers': caps.has_transformers,
                'sklearn': caps.has_sklearn,
                'spacy': caps.has_spacy
            },
            'estimated_accuracy': '~90%' if self.use_embeddings else '~70%',
            'resource_usage': 'high' if self.use_embeddings else 'low'
        }


# Global config instance (lazy loaded)
_config: Optional[BAISConfig] = None


def get_config() -> BAISConfig:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = BAISConfig.from_environment()
    return _config


def reset_config():
    """Reset config (for testing)."""
    global _config
    _config = None


