"""
BAIS Temporal Detector (Core Module)

Implements temporal pattern detection in text:
1. Time reference detection
2. Temporal bias identification
3. Recency bias detection
4. Historical reference analysis

Patent Alignment:
- Core detector module
- Brain Layer: 2 (Sensory Cortex)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import re


class TemporalReferenceType(Enum):
    """Types of temporal references."""
    ABSOLUTE = "absolute"      # "in 2024", "on January 1st"
    RELATIVE = "relative"      # "yesterday", "last week"
    VAGUE = "vague"           # "recently", "long ago"
    NONE = "none"             # No temporal reference


class TemporalBiasType(Enum):
    """Types of temporal biases."""
    RECENCY_BIAS = "recency_bias"           # Overweights recent events
    ANCHORING_BIAS = "anchoring_bias"       # Anchors to specific time
    NOSTALGIA_BIAS = "nostalgia_bias"       # Romanticizes the past
    FUTURE_DISCOUNTING = "future_discounting"  # Undervalues future


@dataclass
class TemporalReference:
    """A detected temporal reference."""
    text: str
    ref_type: TemporalReferenceType
    estimated_date: Optional[datetime]
    confidence: float


@dataclass
class TemporalResult:
    """Result of temporal analysis."""
    references: List[TemporalReference]
    detected_biases: List[TemporalBiasType]
    temporal_span: str  # e.g., "past", "present", "future", "mixed"
    bias_score: float
    confidence: float


@dataclass
class TemporalObservation:
    """
    Observation for temporal learning.
    Used to record decisions and their outcomes for pattern learning.
    """
    timestamp: datetime
    accuracy: float
    domain: str
    was_accepted: bool
    was_correct: bool = True
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class TemporalSignal:
    """
    Signal from temporal analysis.
    Used to communicate temporal bias detection results.
    """
    bias_type: str = "temporal"
    severity: float = 0.0
    description: str = ""
    evidence: List[str] = field(default_factory=list)
    recommendation: str = ""


class TemporalDetector:
    """
    Detects temporal patterns and biases in text.
    
    Brain Layer: 2 (Sensory Cortex)
    
    Capabilities:
    1. Extract temporal references
    2. Identify temporal biases
    3. Assess temporal context of statements
    """
    
    # Temporal indicator patterns
    TEMPORAL_PATTERNS = {
        'recent': ['recently', 'just now', 'latest', 'new', 'current', 'today'],
        'past': ['used to', 'back then', 'historically', 'traditional', 'old', 'previously'],
        'future': ['will', 'going to', 'soon', 'eventually', 'next', 'upcoming'],
        'absolute': [r'\b\d{4}\b', r'\bjanuary\b', r'\bfebruary\b', r'\bmarch\b', 
                    r'\bapril\b', r'\bmay\b', r'\bjune\b', r'\bjuly\b', r'\baugust\b',
                    r'\bseptember\b', r'\boctober\b', r'\bnovember\b', r'\bdecember\b'],
        'relative': ['yesterday', 'last week', 'last month', 'last year', 'tomorrow',
                    'next week', 'next month', 'next year', 'ago'],
    }
    
    BIAS_INDICATORS = {
        TemporalBiasType.RECENCY_BIAS: ['latest is best', 'newest', 'most recent', 'just released'],
        TemporalBiasType.ANCHORING_BIAS: ['since', 'from the beginning', 'originally'],
        TemporalBiasType.NOSTALGIA_BIAS: ['good old days', 'better before', 'used to be better', 'golden age'],
        TemporalBiasType.FUTURE_DISCOUNTING: ['not my problem', 'later', "won't matter"],
    }
    
    def __init__(self, storage_path: Any = None):
        """Initialize temporal detector."""
        # Storage for persistence
        self.storage_path = storage_path
        
        # Learning state
        self._outcomes: List[Dict] = []
        self._feedback: List[Dict] = []
        self._domain_adjustments: Dict[str, float] = {}
        self._total_detections: int = 0
        self._detection_accuracy: List[bool] = []
        self._observations: List[Dict] = []
    
    def detect(self, text: str, context: Dict[str, Any] = None) -> TemporalResult:
        """
        Detect temporal patterns and biases.
        
        Args:
            text: Text to analyze
            context: Optional context
            
        Returns:
            TemporalResult with detected patterns
        """
        self._total_detections += 1
        text_lower = text.lower()
        
        # Extract temporal references
        references = self._extract_references(text_lower)
        
        # Detect biases
        biases = self._detect_biases(text_lower)
        
        # Determine temporal span
        span = self._determine_span(references, text_lower)
        
        # Calculate bias score
        bias_score = min(1.0, len(biases) * 0.25)
        
        # Calculate confidence
        confidence = 0.5 + min(0.4, len(references) * 0.1)
        
        return TemporalResult(
            references=references,
            detected_biases=biases,
            temporal_span=span,
            bias_score=bias_score,
            confidence=confidence
        )
    
    def _extract_references(self, text: str) -> List[TemporalReference]:
        """Extract temporal references from text."""
        references = []
        
        # Check for recent references
        for term in self.TEMPORAL_PATTERNS['recent']:
            if term in text:
                references.append(TemporalReference(
                    text=term,
                    ref_type=TemporalReferenceType.RELATIVE,
                    estimated_date=None,
                    confidence=0.7
                ))
        
        # Check for past references
        for term in self.TEMPORAL_PATTERNS['past']:
            if term in text:
                references.append(TemporalReference(
                    text=term,
                    ref_type=TemporalReferenceType.VAGUE,
                    estimated_date=None,
                    confidence=0.6
                ))
        
        # Check for future references
        for term in self.TEMPORAL_PATTERNS['future']:
            if term in text:
                references.append(TemporalReference(
                    text=term,
                    ref_type=TemporalReferenceType.VAGUE,
                    estimated_date=None,
                    confidence=0.6
                ))
        
        # Check for absolute references (years)
        for pattern in self.TEMPORAL_PATTERNS['absolute']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                references.append(TemporalReference(
                    text=match,
                    ref_type=TemporalReferenceType.ABSOLUTE,
                    estimated_date=None,
                    confidence=0.9
                ))
        
        # Check for relative references
        for term in self.TEMPORAL_PATTERNS['relative']:
            if term in text:
                references.append(TemporalReference(
                    text=term,
                    ref_type=TemporalReferenceType.RELATIVE,
                    estimated_date=None,
                    confidence=0.8
                ))
        
        return references
    
    def _detect_biases(self, text: str) -> List[TemporalBiasType]:
        """Detect temporal biases in text."""
        biases = []
        
        for bias_type, indicators in self.BIAS_INDICATORS.items():
            for indicator in indicators:
                if indicator in text:
                    if bias_type not in biases:
                        biases.append(bias_type)
        
        return biases
    
    def _determine_span(self, references: List[TemporalReference], text: str) -> str:
        """Determine overall temporal span."""
        has_past = any(t in text for t in self.TEMPORAL_PATTERNS['past'])
        has_recent = any(t in text for t in self.TEMPORAL_PATTERNS['recent'])
        has_future = any(t in text for t in self.TEMPORAL_PATTERNS['future'])
        
        if has_past and has_future:
            return "mixed"
        elif has_past:
            return "past"
        elif has_future:
            return "future"
        elif has_recent:
            return "present"
        else:
            return "unspecified"
    
    # ===== Learning Interface Methods =====
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        """Record detection outcome for learning."""
        self._outcomes.append(outcome)
        if 'detection_correct' in outcome:
            self._detection_accuracy.append(outcome['detection_correct'])
    
    def record_feedback(self, feedback: Dict) -> None:
        """Record feedback on temporal detection."""
        self._feedback.append(feedback)
        domain = feedback.get('domain', 'general')
        if feedback.get('detection_inaccurate', False):
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) - 0.05
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt detection thresholds."""
        pass
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get domain adjustment."""
        return self._domain_adjustments.get(domain, 0.0)
    
    def record(self, observation: 'TemporalObservation') -> Dict:
        """
        Record a temporal observation for learning.
        
        Args:
            observation: TemporalObservation dataclass with timestamp, accuracy, etc.
            
        Returns:
            Dict with recorded observation info.
        """
        if not hasattr(self, '_observations'):
            self._observations = []
        
        obs_dict = {
            'timestamp': observation.timestamp.isoformat() if observation.timestamp else None,
            'accuracy': observation.accuracy,
            'domain': observation.domain,
            'was_accepted': observation.was_accepted,
            'was_correct': observation.was_correct,
            'features': observation.features
        }
        
        self._observations.append(obs_dict)
        self._outcomes.append(obs_dict)
        
        # Update domain adjustments based on observation
        if not observation.was_correct:
            self._domain_adjustments[observation.domain] = \
                self._domain_adjustments.get(observation.domain, 0.0) - 0.02
        
        return {'recorded': True, 'observation_count': len(self._observations)}
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics."""
        det_acc = sum(self._detection_accuracy) / max(len(self._detection_accuracy), 1)
        
        return {
            'total_detections': self._total_detections,
            'detection_accuracy': det_acc,
            'domain_adjustments': dict(self._domain_adjustments),
            'outcomes_recorded': len(self._outcomes),
            'feedback_recorded': len(self._feedback)
        }

    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback."""
        pass  # Implement specific learning logic


    def get_statistics(self) -> Dict:
        """Return statistics."""
        return {'outcomes': len(getattr(self, '_outcomes', []))}


    def serialize_state(self) -> Dict:
        """Serialize state."""
        return {'outcomes': getattr(self, '_outcomes', [])[-100:]}


    def deserialize_state(self, state: Dict) -> None:
        """Restore state."""
        self._outcomes = state.get('outcomes', [])


if __name__ == "__main__":
    detector = TemporalDetector()
    
    test_text = """
    The latest research from 2024 shows significant improvements.
    Back in the good old days, things were simpler. Recently, 
    we've seen a shift, but the future will bring even more changes.
    """
    
    result = detector.detect(test_text)
    
    print("=" * 60)
    print("TEMPORAL DETECTOR TEST")
    print("=" * 60)
    print(f"Temporal Span: {result.temporal_span}")
    print(f"References found: {len(result.references)}")
    print(f"Bias Score: {result.bias_score:.2f}")
    print(f"Detected Biases: {[b.value for b in result.detected_biases]}")
    print(f"\nReferences:")
    for ref in result.references[:5]:  # Show first 5
        print(f"  - '{ref.text}' ({ref.ref_type.value})")
    print(f"\nLearning stats: {detector.get_learning_statistics()}")
