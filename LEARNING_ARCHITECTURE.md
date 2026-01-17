# BASE Learning Architecture

## Overview

The BASE Cognitive Governance Engine implements a **hybrid AI + statistical learning** architecture that connects all learning components to a centralized LLM registry, enabling users to switch LLM providers while maintaining consistent governance.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BASE LEARNING ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐      ┌─────────────────────────────────────────────┐ │
│  │   User Query     │─────>│        CentralizedLearningManager           │ │
│  └──────────────────┘      │   (Coordinates all learning components)     │ │
│                            └───────────────────┬─────────────────────────┘ │
│                                                │                           │
│                                                v                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              AIEnhancedLearningManager (NOVEL-36)                   │   │
│  │                                                                     │   │
│  │   Pattern-Based Analysis ──┬──> LLM-Based Analysis                  │   │
│  │   (Fast Path)              │    (Uncertain Cases)                   │   │
│  │                            │                                        │   │
│  │   - Domain familiarity     │    - Outcome reasoning                 │   │
│  │   - Module stability       │    - Cross-module applicability        │   │
│  │   - Error repetition       │    - Contradiction detection           │   │
│  │   - Low signal detection   │    - Weight suggestions                │   │
│  └────────────────────────────┴────────────────┬───────────────────────┘   │
│                                                │                           │
│                                                v                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    LLMRegistry (NOVEL-19)                           │   │
│  │                                                                     │   │
│  │   Supported Providers:                                              │   │
│  │   ┌─────────┬─────────┬───────────┬────────┬─────────┬─────────┐   │   │
│  │   │  Grok   │ OpenAI  │ Anthropic │ Google │ Mistral │ Cohere  │   │   │
│  │   └─────────┴─────────┴───────────┴────────┴─────────┴─────────┘   │   │
│  │                                                                     │   │
│  │   User Configuration:                                               │   │
│  │   - set_preferred_provider(provider)                                │   │
│  │   - get_provider(name)                                              │   │
│  │   - register_provider(name, provider)                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. AIEnhancedLearningManager (NOVEL-36)

**Location:** `src/core/ai_enhanced_learning.py`

The core AI-enhanced learning component that provides hybrid learning capabilities:

```python
from core.ai_enhanced_learning import AIEnhancedLearningManager
from core.llm_registry import LLMRegistry

# Create and connect
learning_manager = AIEnhancedLearningManager()
llm_registry = LLMRegistry()
learning_manager.set_llm_registry(llm_registry)

# Record outcome with AI analysis
result = learning_manager.record_with_analysis(
    module_name="bias_detector",
    input_data={"query": "Is this claim biased?"},
    output_data={"bias_score": 0.3},
    was_correct=True,
    domain="general"
)

print(f"AI Decision: {result.ai_analysis.decision}")
print(f"Confidence: {result.ai_analysis.confidence}")
print(f"Reasoning: {result.ai_analysis.reasoning}")
```

**Features:**
- Pattern-based analysis for fast decisions
- LLM-based analysis for uncertain cases
- Cross-module pattern sharing
- Contradiction detection
- Full learning interface (5 methods)

### 2. LLMRegistry (NOVEL-19)

**Location:** `src/core/llm_registry.py`

Centralized LLM provider management:

```python
from core.llm_registry import LLMRegistry, get_registry

# Get singleton registry
registry = get_registry()

# Switch provider centrally
registry.set_preferred_provider("grok")  # or "openai", "anthropic", etc.

# Get current provider
provider = registry.get_preferred_provider()
```

**Supported Providers:**
- Grok (xAI)
- OpenAI (GPT-4)
- Anthropic (Claude)
- Google (Gemini)
- Mistral
- Cohere

### 3. Learning Interface

All BASE modules implement the standard learning interface:

```python
class LearningCapable:
    def record_outcome(self, outcome: Dict) -> None:
        """Record learning outcome."""
        
    def record_feedback(self, feedback: Dict) -> None:
        """Record human feedback."""
        
    def adapt_thresholds(self, domain: str, performance_data: Dict) -> None:
        """Adapt thresholds based on performance."""
        
    def get_domain_adjustment(self, domain: str) -> float:
        """Get domain-specific adjustment."""
        
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics."""
```

## Learning Decisions

The AI analysis returns one of six decisions:

| Decision | Description | Action |
|----------|-------------|--------|
| `LEARN` | Standard learning | Record with normal weight |
| `SKIP` | Low signal | Do not record |
| `DEFER` | Uncertain | Escalate to human |
| `AMPLIFY` | Important pattern | Record with increased weight |
| `ATTENUATE` | Noise detected | Record with decreased weight |
| `CONTRADICT` | Conflicts with prior | Flag for review |

## Coverage Statistics

| Metric | Value |
|--------|-------|
| Total Inventions | 69 |
| Fully Implemented | 68 (98.6%) |
| Brain Layers Covered | 10/10 (100%) |
| Total Claims | 109 |
| Critical Claims | 23 |
| Modules with Learning | 69 |

## How Users Change LLMs

```python
from core.llm_registry import get_registry

# Get the centralized registry
registry = get_registry()

# Option 1: Switch to a different provider
registry.set_preferred_provider("openai")

# Option 2: Register a custom provider
class MyLLMProvider:
    name = "my_llm"
    def complete(self, prompt: str) -> str:
        # Custom implementation
        pass

registry.register_provider("my_llm", MyLLMProvider())
registry.set_preferred_provider("my_llm")

# All BASE components using learning will now use the new LLM
```

## Patent Claims Covered

| Patent ID | Name | Module |
|-----------|------|--------|
| NOVEL-36 | AI-Enhanced Learning Manager | `core.ai_enhanced_learning` |
| NOVEL-7 | Neuroplasticity Learning | `core.ai_enhanced_learning` |
| NOVEL-19 | LLM Registry | `core.llm_registry` |
| NOVEL-37 | Invention-Module Registry | `core.invention_module_mapping` |
| NOVEL-38 | Claims Verification Registry | `core.claims_verification` |
| PPA2-Inv27 | OCO Threshold Adapter | `learning.algorithms` |
| PPA1-Inv22 | Feedback Loop | `learning.feedback_loop` |

## Integration with Integrated Engine

```python
from core.integrated_engine import IntegratedGovernanceEngine
from core.ai_enhanced_learning import AIEnhancedLearningManager
from core.llm_registry import get_registry

# Create integrated engine
engine = IntegratedGovernanceEngine()

# Get learning manager
learning = AIEnhancedLearningManager()
learning.set_llm_registry(get_registry())

# Connect learning to engine
engine.learning_manager = learning

# Now all governance decisions flow through AI-enhanced learning
```

## Version History

- **v1.0** (2026-01-03): Initial AI-enhanced learning architecture
  - AIEnhancedLearningManager implemented
  - LLMRegistry integration
  - 69 inventions mapped
  - 109 claims tracked

