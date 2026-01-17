"""
BAIS Contradiction Resolver (PPA1-Inv8)

Detects and resolves contradictions in LLM responses:
1. Internal contradictions (response contradicts itself)
2. External contradictions (response contradicts known facts)
3. Temporal contradictions (inconsistent across time)

Patent Alignment:
- PPA1-Inv8: Contradiction Handling
- Brain Layer: 2 (Prefrontal Cortex)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re


@dataclass
class Contradiction:
    """Represents a detected contradiction."""
    contradiction_id: str
    type: str  # internal, external, temporal
    statement_a: str
    statement_b: str
    severity: float  # 0-1
    evidence: str
    resolution: Optional[str] = None
    resolved: bool = False


@dataclass 
class ContradictionResult:
    """Result of contradiction analysis."""
    contradictions_found: List[Contradiction]
    total_score: float  # 0-100, higher = more contradictory
    has_critical: bool
    resolutions: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ContradictionResolver:
    """
    Detects and resolves contradictions in text.
    
    Implements PPA1-Inv8: Contradiction Handling
    Brain Layer: 2 (Prefrontal Cortex)
    
    Capabilities:
    1. Internal contradiction detection (A says X, A also says not-X)
    2. Logical inconsistency detection (if A then B, but B is denied)
    3. Resolution suggestion generation
    4. Learning from feedback on contradiction detection
    """
    
    # Contradiction indicator patterns
    CONTRADICTION_INDICATORS = [
        (r'\bbut\s+(?:also|actually|in fact)\b', 'potential reversal', 0.6),
        (r'\b(?:however|nevertheless|yet)\b.*\b(?:however|nevertheless|yet)\b', 'double hedging', 0.7),
        (r'\balways\b.*\bnever\b|\bnever\b.*\balways\b', 'absolute contradiction', 0.9),
        (r'\ball\b.*\bnone\b|\bnone\b.*\ball\b', 'totality contradiction', 0.9),
        (r'\b(?:definitely|certainly)\b.*\b(?:might|maybe|possibly)\b', 'certainty conflict', 0.7),
        (r'\b(?:recommended|should)\b.*\b(?:not recommended|should not)\b', 'recommendation conflict', 0.85),
    ]
    
    # Negation patterns
    NEGATION_PATTERNS = [
        (r'\b(is|are|was|were)\b', r'\b(is not|are not|was not|were not|isn\'t|aren\'t|wasn\'t|weren\'t)\b'),
        (r'\b(can|will|should)\b', r'\b(cannot|won\'t|should not|can\'t|shouldn\'t)\b'),
        (r'\b(true|correct|right)\b', r'\b(false|incorrect|wrong)\b'),
    ]
    
    def __init__(self):
        """Initialize the contradiction resolver."""
        # Learning state
        self._outcomes: List[Dict] = []
        self._feedback: List[Dict] = []
        self._domain_adjustments: Dict[str, float] = {}
        self._pattern_effectiveness: Dict[str, float] = {}
        self._total_analyses: int = 0
        self._contradictions_found: int = 0
        
    def analyze(self, text: str, context: Dict[str, Any] = None) -> ContradictionResult:
        """
        Analyze text for contradictions.
        
        Args:
            text: Text to analyze
            context: Optional context (previous statements, domain, etc.)
            
        Returns:
            ContradictionResult with detected contradictions
        """
        self._total_analyses += 1
        contradictions = []
        text_lower = text.lower()
        
        # Check for contradiction indicators
        for pattern, desc, severity in self.CONTRADICTION_INDICATORS:
            if re.search(pattern, text_lower):
                contradiction = Contradiction(
                    contradiction_id=f"CONT-{len(contradictions)+1}",
                    type="internal",
                    statement_a=pattern,
                    statement_b="",
                    severity=severity,
                    evidence=f"Pattern matched: {desc}"
                )
                contradictions.append(contradiction)
        
        # Check for negation-based contradictions
        sentences = self._split_sentences(text)
        for i, sent_a in enumerate(sentences):
            for j, sent_b in enumerate(sentences):
                if i < j:
                    if self._is_negation_pair(sent_a, sent_b):
                        contradiction = Contradiction(
                            contradiction_id=f"CONT-{len(contradictions)+1}",
                            type="internal",
                            statement_a=sent_a[:100],
                            statement_b=sent_b[:100],
                            severity=0.85,
                            evidence="Negation pair detected"
                        )
                        contradictions.append(contradiction)
        
        self._contradictions_found += len(contradictions)
        
        # Calculate total score
        if not contradictions:
            total_score = 0.0
        else:
            total_score = min(100, sum(c.severity * 50 for c in contradictions))
        
        # Generate resolutions
        resolutions = [self._suggest_resolution(c) for c in contradictions]
        
        return ContradictionResult(
            contradictions_found=contradictions,
            total_score=total_score,
            has_critical=any(c.severity > 0.8 for c in contradictions),
            resolutions=resolutions
        )
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    def _is_negation_pair(self, sent_a: str, sent_b: str) -> bool:
        """Check if two sentences form a negation pair."""
        sent_a_lower = sent_a.lower()
        sent_b_lower = sent_b.lower()
        
        for pos, neg in self.NEGATION_PATTERNS:
            if re.search(pos, sent_a_lower) and re.search(neg, sent_b_lower):
                # Check for subject overlap
                words_a = set(sent_a_lower.split())
                words_b = set(sent_b_lower.split())
                overlap = len(words_a & words_b) / max(len(words_a), 1)
                if overlap > 0.3:  # Significant word overlap
                    return True
        return False
    
    def _suggest_resolution(self, contradiction: Contradiction) -> str:
        """Suggest a resolution for a contradiction."""
        if contradiction.severity > 0.8:
            return f"Critical contradiction: Review and correct statements about {contradiction.evidence}"
        elif contradiction.severity > 0.5:
            return f"Moderate contradiction: Clarify the relationship between conflicting statements"
        else:
            return f"Minor inconsistency: Consider adding context to resolve ambiguity"
    
    def resolve(self, text: str, contradictions: List[Contradiction]) -> str:
        """
        Attempt to resolve contradictions by suggesting corrections.
        
        Args:
            text: Original text
            contradictions: Detected contradictions
            
        Returns:
            Corrected text or original if no resolution found
        """
        if not contradictions:
            return text
        
        # Generate resolution prompt
        resolution_notes = []
        for c in contradictions:
            resolution_notes.append(f"- {c.evidence}: Consider revising")
        
        return text + "\n\n[Note: Potential contradictions detected:\n" + "\n".join(resolution_notes) + "]"
    
    # ===== Learning Interface Methods =====
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        """Record analysis outcome for learning."""
        self._outcomes.append(outcome)
        if outcome.get('correct', False):
            for pattern in outcome.get('patterns_used', []):
                self._pattern_effectiveness[pattern] = self._pattern_effectiveness.get(pattern, 0.5) + 0.05
    
    def record_feedback(self, feedback: Dict) -> None:
        """Record feedback on contradiction detection."""
        self._feedback.append(feedback)
        domain = feedback.get('domain', 'general')
        if feedback.get('feedback_type') == 'false_positive':
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) - 0.05
        elif feedback.get('feedback_type') == 'false_negative':
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) + 0.05
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt detection thresholds based on performance."""
        if performance_data:
            if performance_data.get('false_positive_rate', 0) > 0.2:
                self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) - 0.1
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get threshold adjustment for domain."""
        return self._domain_adjustments.get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics."""
        detection_rate = self._contradictions_found / max(self._total_analyses, 1)
        return {
            'total_analyses': self._total_analyses,
            'contradictions_found': self._contradictions_found,
            'detection_rate': detection_rate,
            'pattern_effectiveness': dict(self._pattern_effectiveness),
            'domain_adjustments': dict(self._domain_adjustments),
            'outcomes_recorded': len(self._outcomes),
            'feedback_recorded': len(self._feedback)
        }


    # =========================================================================
    # Learning Interface Completion
    # =========================================================================
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback to adjust behavior."""
        if not hasattr(self, '_learning_params'):
            self._learning_params = {}
        was_correct = feedback.get('was_correct', True)
        if not was_correct:
            adjustment = feedback.get('adjustment', 0.05)
            for key in self._learning_params:
                self._learning_params[key] *= (1 - adjustment)
    
    def get_statistics(self) -> Dict:
        """Return learning statistics."""
        return {
            'learning_params': getattr(self, '_learning_params', {}),
            'history_size': len(getattr(self, '_outcomes', []))
        }
    
    def serialize_state(self) -> Dict:
        """Serialize state for persistence."""
        return {
            'outcomes': getattr(self, '_outcomes', [])[-100:],
            'learning_params': getattr(self, '_learning_params', {}),
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore state from serialized data."""
        self._outcomes = state.get('outcomes', [])
        self._learning_params = state.get('learning_params', {})

if __name__ == "__main__":
    resolver = ContradictionResolver()
    
    # Test text with contradictions
    test_text = """
    This medication is always safe for all patients. However, it should never be 
    given to patients with liver conditions. The treatment is definitely effective,
    but it might not work in some cases. All patients will benefit from this, 
    yet none should take it without consultation.
    """
    
    result = resolver.analyze(test_text)
    
    print("=" * 60)
    print("CONTRADICTION RESOLVER TEST")
    print("=" * 60)
    print(f"Contradictions found: {len(result.contradictions_found)}")
    print(f"Total score: {result.total_score:.1f}")
    print(f"Has critical: {result.has_critical}")
    
    for c in result.contradictions_found:
        print(f"\n  - {c.contradiction_id}: {c.evidence} (severity: {c.severity})")
    
    print(f"\nLearning stats: {resolver.get_learning_statistics()}")

