"""
BAIS Bias History (Layer 8 - Hippocampus)

Alias module that provides documentation compatibility.
The actual implementation is in core/bias_evolution_tracker.py

Patent Alignment:
- PPA1-Inv1: Bias Evolution Tracking
- Brain Layer: 8 (Hippocampus - Memory)
"""

# Import from actual location
from core.bias_evolution_tracker import (
    BiasEvolutionTracker,
    BiasSnapshot
)

# Documentation compatibility alias
BiasScoreHistory = BiasEvolutionTracker

__all__ = [
    'BiasScoreHistory',
    'BiasEvolutionTracker',
    'BiasSnapshot'
]

