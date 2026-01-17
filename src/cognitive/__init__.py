"""
BAIS Cognitive Enhancement Module

This module implements NOVEL cognitive capabilities:
- Decision quality (improving judgment)
- Mission alignment (preventing drift)
- Uncertainty quantification (calibrated confidence)

NOTE: Causal reasoning, inference, and truth detection
use EXISTING modules (world_models, neurosymbolic, factual)
to avoid duplication.

The goal is to IMPROVE outputs, not just flag them.
"""

# Novel modules (no duplicates in codebase)
from .decision_quality import DecisionQualityEnhancer
from .mission_alignment import MissionAlignmentChecker
from .uncertainty_quantifier import UncertaintyQuantifier

__all__ = [
    'DecisionQualityEnhancer',
    'MissionAlignmentChecker',
    'UncertaintyQuantifier'
]

