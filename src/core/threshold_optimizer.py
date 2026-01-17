"""
BASE Threshold Optimizer (Layer 6 - Basal Ganglia)

Alias module that provides documentation compatibility.
The actual implementation is in learning/threshold_optimizer.py

Patent Alignment:
- PPA2-Comp1: Online Convex Optimization
- Brain Layer: 6 (Basal Ganglia - Learning)
"""

# Import from actual location
from learning.threshold_optimizer import (
    AdaptiveThresholdOptimizer,
    ThresholdDecision
)

# Documentation compatibility alias
ThresholdOptimizer = AdaptiveThresholdOptimizer

__all__ = [
    'ThresholdOptimizer',
    'AdaptiveThresholdOptimizer',
    'ThresholdDecision'
]

