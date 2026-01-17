"""
BASE Detectors - Phase 2 Implementation

ML-based and pattern-learning detectors:
- GroundingDetector: Semantic grounding verification
- BehavioralDetector: All 4 bias types (PPA 3)
- TemporalDetector: Multi-timescale pattern learning
- FactualDetector: Entailment and contradiction detection
"""

from .grounding import GroundingDetector, GroundingResult
from .behavioral import BehavioralBiasDetector, ComprehensiveBiasResult
from .temporal import TemporalDetector, TemporalResult
from .factual import FactualDetector, FactualAnalysis
from .big5 import Big5Detector, Big5Result

__all__ = [
    'GroundingDetector', 'GroundingResult',
    'BehavioralBiasDetector', 'ComprehensiveBiasResult',
    'TemporalDetector', 'TemporalResult',
    'FactualDetector', 'FactualAnalysis',
    'Big5Detector', 'Big5Result'
]

