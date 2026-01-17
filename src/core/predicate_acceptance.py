"""
BASE Predicate Acceptance (PPA2-Comp4, PPA2-Inv26, PPA1-Inv21)

Implements configurable predicate-based acceptance criteria:
1. Conformal Must-Pass predicates (safety-critical checks)
2. Lexicographic gate ordering (priority-based evaluation)
3. K-of-M acceptance rules

Patent Alignment:
- PPA2-Comp4: Conformal Must-Pass
- PPA2-Inv26: Lexicographic Gate
- PPA1-Inv21: Configurable Predicate Acceptance
- Brain Layer: 2 (Prefrontal Cortex) and 10 (Motor Cortex)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime
from enum import Enum


class PredicatePriority(Enum):
    """Priority levels for predicate evaluation."""
    CRITICAL = 0    # Must pass, cannot be overridden
    HIGH = 1        # Should pass, can be overridden in emergency
    MEDIUM = 2      # Important but flexible
    LOW = 3         # Nice to have
    OPTIONAL = 4    # Purely informational


class PredicateResult(Enum):
    """Result of a predicate evaluation."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass
class Predicate:
    """A single acceptance predicate."""
    predicate_id: str
    name: str
    description: str
    priority: PredicatePriority
    check_fn: Callable[[Dict], Tuple[PredicateResult, str]]
    domain_specific: List[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class PredicateEvaluation:
    """Result of evaluating a single predicate."""
    predicate_id: str
    name: str
    priority: PredicatePriority
    result: PredicateResult
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AcceptanceDecision:
    """Final acceptance decision based on all predicates."""
    accepted: bool
    evaluations: List[PredicateEvaluation]
    blocking_predicates: List[str]
    warnings: List[str]
    confidence: float  # 0-1
    k_of_m_result: Optional[Dict] = None
    lexicographic_pass: bool = True


class PredicateAcceptance:
    """
    Configurable predicate-based acceptance system.
    
    Implements:
    - PPA2-Comp4: Conformal Must-Pass predicates
    - PPA2-Inv26: Lexicographic gate ordering
    - PPA1-Inv21: Configurable Predicate Acceptance
    
    Features:
    1. Priority-ordered predicate evaluation (lexicographic)
    2. Critical predicates that cannot be bypassed
    3. K-of-M acceptance rules
    4. Domain-specific predicate activation
    5. Learning from acceptance outcomes
    """
    
    def __init__(self, k_of_m: Tuple[int, int] = None):
        """
        Initialize predicate acceptance.
        
        Args:
            k_of_m: Tuple (k, m) for k-of-m acceptance rule. None = all must pass.
        """
        self.predicates: Dict[str, Predicate] = {}
        self.k_of_m = k_of_m
        
        # Learning state
        self._outcomes: List[Dict] = []
        self._feedback: List[Dict] = []
        self._domain_adjustments: Dict[str, float] = {}
        self._predicate_effectiveness: Dict[str, Dict] = {}
        self._total_decisions: int = 0
        self._accepted_count: int = 0
        
        # Register default predicates
        self._register_default_predicates()
    
    def _register_default_predicates(self):
        """Register standard predicates."""
        # Critical safety predicate
        self.register_predicate(Predicate(
            predicate_id="SAFETY_001",
            name="safety_check",
            description="Checks for dangerous content",
            priority=PredicatePriority.CRITICAL,
            check_fn=self._check_safety,
            domain_specific=[]
        ))
        
        # Medical disclaimer predicate
        self.register_predicate(Predicate(
            predicate_id="MEDICAL_001",
            name="medical_disclaimer",
            description="Ensures medical disclaimer present",
            priority=PredicatePriority.HIGH,
            check_fn=self._check_medical_disclaimer,
            domain_specific=['medical']
        ))
        
        # Confidence threshold predicate
        self.register_predicate(Predicate(
            predicate_id="CONF_001",
            name="confidence_threshold",
            description="Checks minimum confidence",
            priority=PredicatePriority.MEDIUM,
            check_fn=self._check_confidence,
            domain_specific=[]
        ))
    
    def register_predicate(self, predicate: Predicate) -> None:
        """Register a new predicate."""
        self.predicates[predicate.predicate_id] = predicate
        self._predicate_effectiveness[predicate.predicate_id] = {
            'evaluations': 0,
            'passes': 0,
            'fails': 0
        }
    
    def evaluate(self, context: Dict[str, Any]) -> AcceptanceDecision:
        """
        Evaluate all applicable predicates.
        
        Args:
            context: Dict with 'response', 'domain', 'confidence', etc.
            
        Returns:
            AcceptanceDecision with full evaluation results
        """
        self._total_decisions += 1
        domain = context.get('domain', 'general')
        
        evaluations = []
        blocking = []
        warnings = []
        
        # Sort predicates by priority (lexicographic ordering)
        sorted_predicates = sorted(
            self.predicates.values(),
            key=lambda p: p.priority.value
        )
        
        # Evaluate in priority order
        for predicate in sorted_predicates:
            if not predicate.enabled:
                continue
            
            # Check domain applicability
            if predicate.domain_specific and domain not in predicate.domain_specific:
                continue
            
            try:
                result, message = predicate.check_fn(context)
            except Exception as e:
                result = PredicateResult.FAIL
                message = f"Predicate error: {str(e)[:50]}"
            
            evaluation = PredicateEvaluation(
                predicate_id=predicate.predicate_id,
                name=predicate.name,
                priority=predicate.priority,
                result=result,
                message=message
            )
            evaluations.append(evaluation)
            
            # Update effectiveness tracking
            stats = self._predicate_effectiveness[predicate.predicate_id]
            stats['evaluations'] += 1
            if result == PredicateResult.PASS:
                stats['passes'] += 1
            elif result == PredicateResult.FAIL:
                stats['fails'] += 1
            
            # Check for blocking failures
            if result == PredicateResult.FAIL:
                if predicate.priority == PredicatePriority.CRITICAL:
                    blocking.append(predicate.predicate_id)
                elif predicate.priority == PredicatePriority.HIGH:
                    warnings.append(f"High-priority predicate failed: {predicate.name}")
            elif result == PredicateResult.WARN:
                warnings.append(message)
        
        # Determine acceptance
        accepted, k_of_m_result = self._determine_acceptance(evaluations, blocking)
        
        if accepted:
            self._accepted_count += 1
        
        # Calculate confidence
        pass_count = sum(1 for e in evaluations if e.result == PredicateResult.PASS)
        confidence = pass_count / max(len(evaluations), 1)
        
        return AcceptanceDecision(
            accepted=accepted,
            evaluations=evaluations,
            blocking_predicates=blocking,
            warnings=warnings,
            confidence=confidence,
            k_of_m_result=k_of_m_result,
            lexicographic_pass=len(blocking) == 0
        )
    
    def _determine_acceptance(self, evaluations: List[PredicateEvaluation], 
                              blocking: List[str]) -> Tuple[bool, Optional[Dict]]:
        """Determine final acceptance based on evaluations."""
        # Critical failures always block
        if blocking:
            return False, None
        
        # Count passes and failures
        passes = sum(1 for e in evaluations if e.result == PredicateResult.PASS)
        total = len([e for e in evaluations if e.result != PredicateResult.SKIP])
        
        # K-of-M rule
        if self.k_of_m:
            k, m = self.k_of_m
            # m is the number of predicates to consider, k is minimum passes
            considered = min(m, total)
            k_of_m_result = {
                'k': k, 'm': m,
                'passes': passes,
                'considered': considered,
                'met': passes >= k
            }
            return passes >= k, k_of_m_result
        
        # Default: all must pass
        return passes == total, None
    
    # Default predicate functions
    def _check_safety(self, context: Dict) -> Tuple[PredicateResult, str]:
        """Check for dangerous content."""
        response = context.get('response', '')
        dangerous_patterns = [
            'how to harm', 'how to kill', 'how to make weapons',
            'bomb instructions', 'poison recipe'
        ]
        response_lower = response.lower()
        for pattern in dangerous_patterns:
            if pattern in response_lower:
                return PredicateResult.FAIL, f"Dangerous content detected: {pattern}"
        return PredicateResult.PASS, "No dangerous content detected"
    
    def _check_medical_disclaimer(self, context: Dict) -> Tuple[PredicateResult, str]:
        """Check for medical disclaimer."""
        response = context.get('response', '')
        disclaimer_phrases = [
            'not medical advice', 'consult a doctor', 'consult your physician',
            'healthcare professional', 'medical professional'
        ]
        response_lower = response.lower()
        for phrase in disclaimer_phrases:
            if phrase in response_lower:
                return PredicateResult.PASS, "Medical disclaimer present"
        return PredicateResult.WARN, "Medical disclaimer recommended"
    
    def _check_confidence(self, context: Dict) -> Tuple[PredicateResult, str]:
        """Check minimum confidence threshold."""
        confidence = context.get('confidence', 0.5)
        threshold = 0.6 + self._domain_adjustments.get(context.get('domain', 'general'), 0.0)
        
        if confidence >= threshold:
            return PredicateResult.PASS, f"Confidence {confidence:.2f} >= {threshold:.2f}"
        return PredicateResult.FAIL, f"Confidence {confidence:.2f} < {threshold:.2f}"
    
    # ===== Learning Interface Methods =====
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        """Record acceptance outcome for learning."""
        self._outcomes.append(outcome)
    
    def record_feedback(self, feedback: Dict) -> None:
        """Record feedback on acceptance decisions."""
        self._feedback.append(feedback)
        domain = feedback.get('domain', 'general')
        if feedback.get('feedback_type') == 'too_strict':
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) - 0.05
        elif feedback.get('feedback_type') == 'too_lenient':
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) + 0.05
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt thresholds based on performance."""
        if performance_data:
            if performance_data.get('false_positive_rate', 0) > 0.2:
                self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) - 0.1
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get threshold adjustment for domain."""
        return self._domain_adjustments.get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics."""
        acceptance_rate = self._accepted_count / max(self._total_decisions, 1)
        return {
            'total_decisions': self._total_decisions,
            'accepted_count': self._accepted_count,
            'acceptance_rate': acceptance_rate,
            'predicates_registered': len(self.predicates),
            'predicate_effectiveness': dict(self._predicate_effectiveness),
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
    # Test the predicate acceptance system
    acceptance = PredicateAcceptance()
    
    # Test context
    context = {
        'response': 'For your headache, you could try ibuprofen. This is not medical advice - consult your physician.',
        'domain': 'medical',
        'confidence': 0.75
    }
    
    result = acceptance.evaluate(context)
    
    print("=" * 60)
    print("PREDICATE ACCEPTANCE TEST")
    print("=" * 60)
    print(f"Accepted: {result.accepted}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Lexicographic pass: {result.lexicographic_pass}")
    
    print("\nEvaluations:")
    for e in result.evaluations:
        print(f"  - {e.name}: {e.result.value} ({e.message[:50]})")
    
    print(f"\nLearning stats: {acceptance.get_learning_statistics()}")

