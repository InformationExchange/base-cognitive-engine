"""BASE Fusion Module - Signal fusion and clinical validation."""

from .signal_fusion import (
    SignalFusion, SignalVector, FusedSignal, 
    FusionMethod, BayesianWeightState
)

__all__ = [
    'SignalFusion', 'SignalVector', 'FusedSignal',
    'FusionMethod', 'BayesianWeightState'
]







