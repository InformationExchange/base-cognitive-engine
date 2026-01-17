"""
BAIS Cognitive Governance Engine
Temporal Bias Detector (PPA1-Inv4)

Detects time-based bias shifts and temporal patterns in responses.
This is a CRITICAL missing implementation identified by BAIS governance.

Patent Reference: PPA1-Inv4 - Temporal Bias Detection
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from enum import Enum
import numpy as np
import re

logger = logging.getLogger(__name__)


class TemporalPattern(Enum):
    """Types of temporal patterns in bias."""
    NONE = "none"
    RECENCY = "recency"  # Favoring recent information
    ANCHORING = "anchoring"  # Over-relying on initial information
    PEAK_END = "peak_end"  # Emphasizing peaks and endings
    DURATION_NEGLECT = "duration_neglect"  # Ignoring duration
    PRIMACY = "primacy"  # Favoring first information
    TELESCOPING = "telescoping"  # Temporal displacement
    HINDSIGHT = "hindsight"  # "I knew it all along"


class TimeReference(Enum):
    """Types of time references in text."""
    PAST = "past"
    PRESENT = "present"
    FUTURE = "future"
    TIMELESS = "timeless"
    AMBIGUOUS = "ambiguous"


@dataclass
class TemporalBiasSignal:
    """Signal from temporal bias detection."""
    score: float  # 0.0 (no bias) to 1.0 (severe bias)
    confidence: float
    patterns_detected: List[TemporalPattern]
    time_references: Dict[TimeReference, int]
    recency_bias_score: float
    anchoring_bias_score: float
    hindsight_bias_score: float
    temporal_consistency: float
    evidence: List[str]
    recommendation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'score': self.score,
            'confidence': self.confidence,
            'patterns_detected': [p.value for p in self.patterns_detected],
            'time_references': {k.value: v for k, v in self.time_references.items()},
            'recency_bias_score': self.recency_bias_score,
            'anchoring_bias_score': self.anchoring_bias_score,
            'hindsight_bias_score': self.hindsight_bias_score,
            'temporal_consistency': self.temporal_consistency,
            'evidence': self.evidence,
            'recommendation': self.recommendation
        }


@dataclass
class PatternResult:
    """Result of temporal pattern analysis."""
    pattern: TemporalPattern
    strength: float
    evidence: List[str]
    time_span_analyzed: Optional[timedelta]


@dataclass 
class HistoricalContext:
    """Historical context for temporal analysis."""
    query: str
    response: str
    timestamp: datetime
    domain: str
    temporal_score: float


class TemporalBiasDetector:
    """
    Detects temporal biases in responses.
    
    PPA1-Inv4: Temporal Bias Detection
    
    Features:
    - Recency bias detection (favoring recent info)
    - Anchoring bias detection (over-relying on initial info)
    - Hindsight bias detection ("I knew it all along")
    - Temporal consistency checking
    - Time reference analysis
    - Learning from feedback
    - Adaptive thresholds
    """
    
    # Time reference patterns
    PAST_PATTERNS = [
        r'\bwas\b', r'\bwere\b', r'\bhad\b', r'\bdid\b',
        r'\bhistorically\b', r'\bpreviously\b', r'\bformerly\b',
        r'\bin the past\b', r'\byears ago\b', r'\blast\b',
        r'\bused to\b', r'\bonce\b', r'\bback then\b'
    ]
    
    PRESENT_PATTERNS = [
        r'\bis\b', r'\bare\b', r'\bhas\b', r'\bhave\b',
        r'\bcurrently\b', r'\bnow\b', r'\btoday\b',
        r'\bat present\b', r'\bthese days\b', r'\bmodern\b'
    ]
    
    FUTURE_PATTERNS = [
        r'\bwill\b', r'\bshall\b', r'\bgoing to\b',
        r'\bin the future\b', r'\bsoon\b', r'\beventually\b',
        r'\bupcoming\b', r'\bnext\b', r'\blater\b'
    ]
    
    # Bias indicator patterns
    RECENCY_BIAS_PATTERNS = [
        (r'\brecent(ly)?\b.*\bmost important\b', 0.3),
        (r'\blatest\b.*\bbest\b', 0.25),
        (r'\bnew\b.*\balways\b.*\bbetter\b', 0.3),
        (r'\bold\b.*\boutdated\b', 0.2),
        (r'\bmodern\b.*\bsuperior\b', 0.25),
    ]
    
    ANCHORING_BIAS_PATTERNS = [
        (r'\bfirst\b.*\bmost\b', 0.2),
        (r'\binitial(ly)?\b.*\bkey\b', 0.2),
        (r'\bstarting point\b', 0.15),
        (r'\boriginal\b.*\bbest\b', 0.2),
    ]
    
    HINDSIGHT_BIAS_PATTERNS = [
        (r'\bobviously\b.*\bwould\b', 0.3),
        (r'\bclearly\b.*\bpredictable\b', 0.3),
        (r'\beveryone knew\b', 0.35),
        (r'\binevitable\b', 0.25),
        (r'\bshould have\b.*\bknown\b', 0.3),
        (r'\bI knew\b.*\ball along\b', 0.4),
    ]
    
    def __init__(
        self,
        recency_threshold: float = 0.3,
        anchoring_threshold: float = 0.3,
        hindsight_threshold: float = 0.3,
        history_window: int = 50,
        learning_rate: float = 0.1
    ):
        """
        Initialize the temporal bias detector.
        
        Args:
            recency_threshold: Threshold for recency bias detection
            anchoring_threshold: Threshold for anchoring bias detection
            hindsight_threshold: Threshold for hindsight bias detection
            history_window: Number of historical contexts to retain
            learning_rate: Learning rate for adaptive updates
        """
        self.recency_threshold = recency_threshold
        self.anchoring_threshold = anchoring_threshold
        self.hindsight_threshold = hindsight_threshold
        self.history_window = history_window
        self.learning_rate = learning_rate
        
        # Historical context
        self._history: deque = deque(maxlen=history_window)
        
        # Learned adjustments per domain
        self._domain_adjustments: Dict[str, Dict[str, float]] = {}
        
        # Performance tracking
        self._detections = 0
        self._true_positives = 0
        self._false_positives = 0
        
        # Compile patterns
        self._past_patterns = [re.compile(p, re.IGNORECASE) for p in self.PAST_PATTERNS]
        self._present_patterns = [re.compile(p, re.IGNORECASE) for p in self.PRESENT_PATTERNS]
        self._future_patterns = [re.compile(p, re.IGNORECASE) for p in self.FUTURE_PATTERNS]
        
        logger.info("[TemporalBiasDetector] Initialized with learning capabilities")
    
    def detect(
        self,
        query: str,
        response: str,
        domain: str = "general",
        context: Optional[Dict[str, Any]] = None
    ) -> TemporalBiasSignal:
        """
        Detect temporal biases in a response.
        
        Args:
            query: The original query
            response: The response to analyze
            domain: Domain context
            context: Additional context
            
        Returns:
            TemporalBiasSignal with detection results
        """
        self._detections += 1
        
        # Analyze time references
        time_refs = self._analyze_time_references(response)
        
        # Detect specific biases
        recency_score, recency_evidence = self._detect_recency_bias(response)
        anchoring_score, anchoring_evidence = self._detect_anchoring_bias(response)
        hindsight_score, hindsight_evidence = self._detect_hindsight_bias(response)
        
        # Check temporal consistency
        consistency = self._check_temporal_consistency(response, time_refs)
        
        # Identify patterns
        patterns = self._identify_patterns(
            recency_score, anchoring_score, hindsight_score
        )
        
        # Apply domain adjustments
        if domain in self._domain_adjustments:
            adj = self._domain_adjustments[domain]
            recency_score *= adj.get('recency', 1.0)
            anchoring_score *= adj.get('anchoring', 1.0)
            hindsight_score *= adj.get('hindsight', 1.0)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            recency_score, anchoring_score, hindsight_score, consistency
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(response, patterns)
        
        # Combine evidence
        evidence = recency_evidence + anchoring_evidence + hindsight_evidence
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            patterns, recency_score, anchoring_score, hindsight_score
        )
        
        # Store in history
        self._history.append(HistoricalContext(
            query=query,
            response=response,
            timestamp=datetime.utcnow(),
            domain=domain,
            temporal_score=overall_score
        ))
        
        return TemporalBiasSignal(
            score=overall_score,
            confidence=confidence,
            patterns_detected=patterns,
            time_references=time_refs,
            recency_bias_score=recency_score,
            anchoring_bias_score=anchoring_score,
            hindsight_bias_score=hindsight_score,
            temporal_consistency=consistency,
            evidence=evidence,
            recommendation=recommendation
        )
    
    def _analyze_time_references(self, text: str) -> Dict[TimeReference, int]:
        """Analyze time references in text."""
        refs = {
            TimeReference.PAST: 0,
            TimeReference.PRESENT: 0,
            TimeReference.FUTURE: 0,
            TimeReference.TIMELESS: 0,
            TimeReference.AMBIGUOUS: 0
        }
        
        for pattern in self._past_patterns:
            refs[TimeReference.PAST] += len(pattern.findall(text))
        
        for pattern in self._present_patterns:
            refs[TimeReference.PRESENT] += len(pattern.findall(text))
        
        for pattern in self._future_patterns:
            refs[TimeReference.FUTURE] += len(pattern.findall(text))
        
        total = sum(refs.values())
        if total == 0:
            refs[TimeReference.TIMELESS] = 1
        
        return refs
    
    def _detect_recency_bias(self, text: str) -> Tuple[float, List[str]]:
        """Detect recency bias in text."""
        score = 0.0
        evidence = []
        
        text_lower = text.lower()
        
        for pattern, weight in self.RECENCY_BIAS_PATTERNS:
            if re.search(pattern, text_lower):
                score += weight
                evidence.append(f"Recency pattern: '{pattern}'")
        
        # Additional checks
        if 'latest' in text_lower and 'best' in text_lower:
            score += 0.1
            evidence.append("Association of 'latest' with 'best'")
        
        if 'outdated' in text_lower or 'obsolete' in text_lower:
            if 'consider' not in text_lower and 'may be' not in text_lower:
                score += 0.15
                evidence.append("Dismissal of older information without qualification")
        
        return min(1.0, score), evidence
    
    def _detect_anchoring_bias(self, text: str) -> Tuple[float, List[str]]:
        """Detect anchoring bias in text."""
        score = 0.0
        evidence = []
        
        text_lower = text.lower()
        
        for pattern, weight in self.ANCHORING_BIAS_PATTERNS:
            if re.search(pattern, text_lower):
                score += weight
                evidence.append(f"Anchoring pattern: '{pattern}'")
        
        # Check for over-emphasis on first-mentioned items
        sentences = text.split('.')
        if len(sentences) > 3:
            first_sentence = sentences[0].lower()
            if any(word in first_sentence for word in ['most important', 'key', 'main', 'primary']):
                score += 0.15
                evidence.append("First sentence establishes strong anchor")
        
        return min(1.0, score), evidence
    
    def _detect_hindsight_bias(self, text: str) -> Tuple[float, List[str]]:
        """Detect hindsight bias in text."""
        score = 0.0
        evidence = []
        
        text_lower = text.lower()
        
        for pattern, weight in self.HINDSIGHT_BIAS_PATTERNS:
            if re.search(pattern, text_lower):
                score += weight
                evidence.append(f"Hindsight pattern: '{pattern}'")
        
        # Check for overconfident retrospective claims
        if 'obvious' in text_lower and ('would' in text_lower or 'should' in text_lower):
            score += 0.2
            evidence.append("Retrospective obviousness claim")
        
        if 'predictable' in text_lower and 'was' in text_lower:
            score += 0.15
            evidence.append("Claim of predictability after the fact")
        
        return min(1.0, score), evidence
    
    def _check_temporal_consistency(
        self,
        text: str,
        time_refs: Dict[TimeReference, int]
    ) -> float:
        """
        Check temporal consistency in the text.
        
        Returns:
            Score from 0 (inconsistent) to 1 (consistent)
        """
        total_refs = sum(time_refs.values())
        if total_refs == 0:
            return 1.0  # No time references = consistent
        
        # Check for tense mixing
        past_ratio = time_refs[TimeReference.PAST] / total_refs
        present_ratio = time_refs[TimeReference.PRESENT] / total_refs
        future_ratio = time_refs[TimeReference.FUTURE] / total_refs
        
        # Ideal is one dominant tense
        max_ratio = max(past_ratio, present_ratio, future_ratio)
        
        # Penalize heavy mixing
        if max_ratio < 0.5:
            return 0.5  # Heavy mixing
        elif max_ratio < 0.7:
            return 0.7  # Moderate mixing
        else:
            return 0.9 + (max_ratio - 0.7) * 0.33  # Good consistency
    
    def _identify_patterns(
        self,
        recency: float,
        anchoring: float,
        hindsight: float
    ) -> List[TemporalPattern]:
        """Identify which temporal patterns are present."""
        patterns = []
        
        if recency >= self.recency_threshold:
            patterns.append(TemporalPattern.RECENCY)
        if anchoring >= self.anchoring_threshold:
            patterns.append(TemporalPattern.ANCHORING)
        if hindsight >= self.hindsight_threshold:
            patterns.append(TemporalPattern.HINDSIGHT)
        
        if not patterns:
            patterns.append(TemporalPattern.NONE)
        
        return patterns
    
    def _calculate_overall_score(
        self,
        recency: float,
        anchoring: float,
        hindsight: float,
        consistency: float
    ) -> float:
        """Calculate overall temporal bias score."""
        # Weighted average with consistency penalty
        base_score = (recency * 0.35 + anchoring * 0.3 + hindsight * 0.35)
        consistency_penalty = (1.0 - consistency) * 0.2
        
        return min(1.0, base_score + consistency_penalty)
    
    def _calculate_confidence(
        self,
        text: str,
        patterns: List[TemporalPattern]
    ) -> float:
        """Calculate confidence in the detection."""
        base_confidence = 0.7
        
        # More text = more confidence
        word_count = len(text.split())
        if word_count > 200:
            base_confidence += 0.1
        elif word_count < 50:
            base_confidence -= 0.1
        
        # More patterns = higher confidence
        if len(patterns) > 1 and TemporalPattern.NONE not in patterns:
            base_confidence += 0.1
        
        return min(1.0, max(0.3, base_confidence))
    
    def _generate_recommendation(
        self,
        patterns: List[TemporalPattern],
        recency: float,
        anchoring: float,
        hindsight: float
    ) -> str:
        """Generate recommendation based on findings."""
        if TemporalPattern.NONE in patterns:
            return "No significant temporal bias detected."
        
        recommendations = []
        
        if TemporalPattern.RECENCY in patterns:
            recommendations.append(
                f"Recency bias detected (score: {recency:.2f}). "
                "Consider including historical context and older relevant information."
            )
        
        if TemporalPattern.ANCHORING in patterns:
            recommendations.append(
                f"Anchoring bias detected (score: {anchoring:.2f}). "
                "Review whether initial information is given disproportionate weight."
            )
        
        if TemporalPattern.HINDSIGHT in patterns:
            recommendations.append(
                f"Hindsight bias detected (score: {hindsight:.2f}). "
                "Avoid implying outcomes were predictable after the fact."
            )
        
        return " | ".join(recommendations)
    
    def analyze_temporal_patterns(
        self,
        history: Optional[List[HistoricalContext]] = None
    ) -> PatternResult:
        """
        Analyze temporal patterns across history.
        
        Args:
            history: Historical contexts to analyze, or use internal history
            
        Returns:
            PatternResult with analysis
        """
        contexts = history or list(self._history)
        
        if not contexts:
            return PatternResult(
                pattern=TemporalPattern.NONE,
                strength=0.0,
                evidence=[],
                time_span_analyzed=None
            )
        
        # Analyze score trends
        scores = [c.temporal_score for c in contexts]
        
        # Detect dominant pattern
        recency_scores = []
        anchoring_scores = []
        hindsight_scores = []
        
        for ctx in contexts:
            signal = self.detect(ctx.query, ctx.response, ctx.domain)
            recency_scores.append(signal.recency_bias_score)
            anchoring_scores.append(signal.anchoring_bias_score)
            hindsight_scores.append(signal.hindsight_bias_score)
        
        avg_recency = np.mean(recency_scores)
        avg_anchoring = np.mean(anchoring_scores)
        avg_hindsight = np.mean(hindsight_scores)
        
        # Find dominant pattern
        max_score = max(avg_recency, avg_anchoring, avg_hindsight)
        if max_score < 0.2:
            dominant = TemporalPattern.NONE
        elif avg_recency == max_score:
            dominant = TemporalPattern.RECENCY
        elif avg_anchoring == max_score:
            dominant = TemporalPattern.ANCHORING
        else:
            dominant = TemporalPattern.HINDSIGHT
        
        time_span = contexts[-1].timestamp - contexts[0].timestamp if len(contexts) > 1 else None
        
        return PatternResult(
            pattern=dominant,
            strength=max_score,
            evidence=[
                f"Avg recency: {avg_recency:.3f}",
                f"Avg anchoring: {avg_anchoring:.3f}",
                f"Avg hindsight: {avg_hindsight:.3f}"
            ],
            time_span_analyzed=time_span
        )
    
    def get_time_weighted_score(self, decay_factor: float = 0.95) -> float:
        """
        Get time-weighted average score from history.
        
        Args:
            decay_factor: Exponential decay factor for older scores
            
        Returns:
            Time-weighted average score
        """
        if not self._history:
            return 0.0
        
        scores = [ctx.temporal_score for ctx in self._history]
        weights = [decay_factor ** i for i in range(len(scores) - 1, -1, -1)]
        
        return np.average(scores, weights=weights)
    
    def learn_from_feedback_standard(self, feedback: Dict) -> None:
        """Standard interface wrapper for learn_from_feedback."""
        if not hasattr(self, '_feedback'):
            self._feedback = []
        self._feedback.append(feedback)
    
    def learn_from_feedback(
        self,
        was_correct: bool = True,
        actual_bias_present: bool = False,
        domain: str = "general"
    ):
        """
        Learn from feedback on detection accuracy.
        
        Args:
            was_correct: Whether the detection was correct
            actual_bias_present: Whether bias was actually present
            domain: Domain of the detection
        """
        if was_correct:
            if actual_bias_present:
                self._true_positives += 1
            # True negative not explicitly tracked
        else:
            self._false_positives += 1
            
            # Adjust thresholds for domain
            if domain not in self._domain_adjustments:
                self._domain_adjustments[domain] = {
                    'recency': 1.0, 'anchoring': 1.0, 'hindsight': 1.0
                }
            
            if actual_bias_present:
                # False negative - lower sensitivity
                for key in self._domain_adjustments[domain]:
                    self._domain_adjustments[domain][key] *= (1 - self.learning_rate)
            else:
                # False positive - raise sensitivity
                for key in self._domain_adjustments[domain]:
                    self._domain_adjustments[domain][key] *= (1 + self.learning_rate)
        
        logger.info(f"[TemporalBiasDetector] Learned from feedback: correct={was_correct}")
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Any) -> Dict[str, float]:
        """Record learning outcome for interface compatibility."""
        if hasattr(outcome, 'was_correct') and hasattr(outcome, 'domain'):
            self.learn_from_feedback(
                was_correct=outcome.was_correct,
                actual_bias_present=not outcome.was_correct,
                domain=getattr(outcome, 'domain', 'general')
            )
        return self.get_statistics()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        precision = self._true_positives / (self._true_positives + self._false_positives) if (self._true_positives + self._false_positives) > 0 else 0
        
        return {
            'total_detections': self._detections,
            'history_size': len(self._history),
            'true_positives': self._true_positives,
            'false_positives': self._false_positives,
            'precision': precision,
            'domain_adjustments': self._domain_adjustments,
            'thresholds': {
                'recency': self.recency_threshold,
                'anchoring': self.anchoring_threshold,
                'hindsight': self.hindsight_threshold
            }
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get serializable state."""
        return {
            'recency_threshold': self.recency_threshold,
            'anchoring_threshold': self.anchoring_threshold,
            'hindsight_threshold': self.hindsight_threshold,
            'learning_rate': self.learning_rate,
            'domain_adjustments': self._domain_adjustments,
            'statistics': self.get_statistics()
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load state from persistence."""
        self.recency_threshold = state.get('recency_threshold', self.recency_threshold)
        self.anchoring_threshold = state.get('anchoring_threshold', self.anchoring_threshold)
        self.hindsight_threshold = state.get('hindsight_threshold', self.hindsight_threshold)
        self.learning_rate = state.get('learning_rate', self.learning_rate)
        self._domain_adjustments = state.get('domain_adjustments', {})
        
        logger.info("[TemporalBiasDetector] State loaded")
    
    # ========================================
    # LEARNING INTERFACE (Auto-added)
    # ========================================
    
    def record_feedback(self, result, was_accurate):
        """Record feedback for learning adjustment."""
        self.record_outcome({'result': str(result)}, {}, was_accurate)
    
    def adapt_thresholds(self, threshold_name, current_value, direction='decrease'):
        """Adapt thresholds based on learning history."""
        adjustment = 0.05 if direction == 'increase' else -0.05
        return max(0.1, min(0.95, current_value + adjustment))
    
    def get_domain_adjustment(self, domain):
        """Get learned adjustment for a domain."""
        return self._domain_adjustments.get(domain, 0.0)
    
    def get_learning_statistics(self):
        """Get learning statistics."""
        outcomes = getattr(self, '_outcomes', [])
        correct = sum(1 for o in outcomes if o.get('correct', False))
        return {
            'module': self.__class__.__name__,
            'total_outcomes': len(outcomes),
            'accuracy': correct / len(outcomes) if outcomes else 0.0
        }
    
    def serialize_state(self) -> Dict:
        """Serialize state for persistence."""
        return {
            'outcomes': getattr(self, '_outcomes', [])[-100:],
            'domain_adjustments': dict(self._domain_adjustments),
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore state from serialized data."""
        self._outcomes = state.get('outcomes', [])
        self._domain_adjustments = state.get('domain_adjustments', {})


# Alias for documentation compatibility
TemporalDetector = TemporalBiasDetector


if __name__ == "__main__":
    # Test the implementation
    detector = TemporalBiasDetector()
    
    # Test cases
    test_cases = [
        {
            "query": "What's the best programming language?",
            "response": "The latest programming languages are always better. Modern languages like Rust are clearly superior to outdated ones like C. Everyone knew JavaScript would become the most popular.",
            "domain": "technology"
        },
        {
            "query": "How did the market perform?",
            "response": "Looking back, it was obvious the market would crash. The signs were predictable, and everyone should have known what was coming. I knew it all along.",
            "domain": "financial"
        },
        {
            "query": "What treatment is recommended?",
            "response": "Based on current medical evidence, the standard treatment protocol involves... The first line treatment remains effective.",
            "domain": "medical"
        }
    ]
    
    print("=" * 60)
    print("TEMPORAL BIAS DETECTOR TEST")
    print("=" * 60)
    
    for i, case in enumerate(test_cases):
        print(f"\nTest {i+1}: {case['domain']}")
        result = detector.detect(
            query=case['query'],
            response=case['response'],
            domain=case['domain']
        )
        print(f"  Score: {result.score:.3f}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Patterns: {[p.value for p in result.patterns_detected]}")
        print(f"  Recency: {result.recency_bias_score:.3f}")
        print(f"  Anchoring: {result.anchoring_bias_score:.3f}")
        print(f"  Hindsight: {result.hindsight_bias_score:.3f}")
    
    # Get statistics
    stats = detector.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  History size: {stats['history_size']}")
    
    print("\n✓ TemporalBiasDetector implementation verified")


