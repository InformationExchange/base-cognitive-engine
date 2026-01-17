"""
BAIS Cognitive Governance Engine - Skeptical Learning Manager
Phase 5: Learning with skepticism towards user labels

Key principle: User labels may be wrong. BAIS prioritizes:
1. Execution evidence (highest trust)
2. Cross-validation (high trust)
3. User feedback (discounted by skepticism factor)

Patent: NOVEL-44 (Skeptical Learning)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime
from collections import defaultdict
import hashlib
import json

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class LearningSource(Enum):
    """Source of learning signal."""
    EXECUTION = "execution"         # Code was executed, result observed
    CROSS_VALIDATION = "cross_validation"  # Multiple LLMs agree/disagree
    USER_FEEDBACK = "user_feedback" # User said it's good/bad
    AUTOMATED_TEST = "automated_test"  # Test suite result
    BAIS_ANALYSIS = "bais_analysis"    # BAIS own analysis


class EvidenceType(Enum):
    """Type of evidence."""
    RUNTIME_OUTPUT = "runtime_output"
    TEST_RESULT = "test_result"
    USER_LABEL = "user_label"
    LLM_CONSENSUS = "llm_consensus"
    STATIC_ANALYSIS = "static_analysis"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LearningSignal:
    """A single learning signal."""
    signal_id: str
    source: LearningSource
    evidence_type: EvidenceType
    
    # The learning data
    module_name: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    was_correct: bool
    
    # Trust factors
    raw_confidence: float  # 0-1, confidence before skepticism
    
    # Metadata
    timestamp: str = ""
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdjustedLearningSignal:
    """Learning signal after skepticism adjustment."""
    original: LearningSignal
    adjusted_confidence: float
    trust_weight: float
    should_learn: bool
    reason: str


@dataclass
class CrossValidationResult:
    """Result from cross-validating a signal."""
    signal_id: str
    validated: bool
    validation_sources: List[str]
    agreement_score: float
    conflicts: List[str]


# =============================================================================
# Skeptical Learning Manager
# =============================================================================

class SkepticalLearningManager:
    """
    Manages learning with appropriate skepticism.
    
    NOT: "User said it's correct, so it is"
    YES: "User said it's correct, but let's verify"
    
    Trust hierarchy:
    1. Execution evidence: 1.0 weight
    2. Automated tests: 0.95 weight
    3. LLM consensus: 0.85 weight
    4. User feedback: 0.7 weight (configurable)
    5. Static analysis: 0.6 weight
    """
    
    def __init__(
        self,
        user_label_discount: float = 0.7,
        require_cross_validation: bool = True,
        min_confidence_to_learn: float = 0.3
    ):
        self.user_label_discount = user_label_discount
        self.require_cross_validation = require_cross_validation
        self.min_confidence_to_learn = min_confidence_to_learn
        
        # Trust weights by source
        self._trust_weights = {
            LearningSource.EXECUTION: 1.0,
            LearningSource.AUTOMATED_TEST: 0.95,
            LearningSource.CROSS_VALIDATION: 0.85,
            LearningSource.USER_FEEDBACK: user_label_discount,
            LearningSource.BAIS_ANALYSIS: 0.75
        }
        
        # Learning history
        self._signals: List[AdjustedLearningSignal] = []
        self._learned_patterns: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self._stats = {
            'total_signals': 0,
            'learned': 0,
            'rejected': 0,
            'cross_validated': 0,
            'by_source': defaultdict(int),
            'avg_adjustment': 0.0
        }
        
        logger.info(f"[SkepticalLearning] Initialized with user_label_discount={user_label_discount}")
        self._outcomes = []

    def process_signal(self, signal: LearningSignal) -> AdjustedLearningSignal:
        """
        Process a learning signal with appropriate skepticism.
        
        Returns adjusted signal with learning decision.
        """
        self._stats['total_signals'] += 1
        self._stats['by_source'][signal.source.value] += 1
        
        # Get trust weight for this source
        trust_weight = getattr(self, '_trust_weights', {}).get(signal.source, 0.5)
        
        # Apply skepticism adjustment
        adjusted_confidence = signal.raw_confidence * trust_weight
        
        # Additional penalties
        if signal.source == LearningSource.USER_FEEDBACK:
            # Extra skepticism for user feedback without execution proof
            if signal.evidence_type == EvidenceType.USER_LABEL:
                adjusted_confidence *= 0.9  # Extra 10% discount
        
        # Determine if we should learn
        should_learn = adjusted_confidence >= self.min_confidence_to_learn
        
        # Generate reason
        if should_learn:
            reason = f"Accepted: {signal.source.value} with adjusted confidence {adjusted_confidence:.2f}"
        else:
            reason = f"Rejected: confidence {adjusted_confidence:.2f} below threshold {self.min_confidence_to_learn}"
        
        adjusted = AdjustedLearningSignal(
            original=signal,
            adjusted_confidence=adjusted_confidence,
            trust_weight=trust_weight,
            should_learn=should_learn,
            reason=reason
        )
        
        # Update stats
        if should_learn:
            self._stats['learned'] += 1
            getattr(self, '_signals', {}).append(adjusted)
        else:
            self._stats['rejected'] += 1
        
        self._update_average_adjustment(signal.raw_confidence, adjusted_confidence)
        
        return adjusted
    
    def cross_validate(
        self,
        signal: LearningSignal,
        validation_signals: List[LearningSignal]
    ) -> CrossValidationResult:
        """
        Cross-validate a signal against other signals.
        
        If multiple sources agree, increase confidence.
        If sources conflict, flag for review.
        """
        self._stats['cross_validated'] += 1
        
        agreements = []
        conflicts = []
        validation_sources = []
        
        for vs in validation_signals:
            validation_sources.append(vs.source.value)
            
            # Check if they agree on correctness
            if vs.was_correct == signal.was_correct:
                agreements.append(vs)
            else:
                conflicts.append(f"{vs.source.value}: {'correct' if vs.was_correct else 'incorrect'}")
        
        # Calculate agreement score
        total = len(validation_signals)
        if total > 0:
            agreement_score = len(agreements) / total
        else:
            agreement_score = 0.0
        
        validated = agreement_score >= 0.5 and len(conflicts) == 0
        
        return CrossValidationResult(
            signal_id=signal.signal_id,
            validated=validated,
            validation_sources=validation_sources,
            agreement_score=agreement_score,
            conflicts=conflicts
        )
    
    def learn_from_execution(
        self,
        module_name: str,
        input_data: Dict,
        output_data: Dict,
        execution_succeeded: bool,
        execution_output: str
    ) -> AdjustedLearningSignal:
        """
        Learn from actual execution result (highest trust).
        """
        signal = LearningSignal(
            signal_id=f"exec-{hashlib.md5(str(input_data).encode()).hexdigest()[:8]}",
            source=LearningSource.EXECUTION,
            evidence_type=EvidenceType.RUNTIME_OUTPUT,
            module_name=module_name,
            input_data=input_data,
            output_data=output_data,
            was_correct=execution_succeeded,
            raw_confidence=1.0 if execution_succeeded else 0.9,  # Even failure is high-confidence learning
            timestamp=datetime.utcnow().isoformat(),
            context={'execution_output': execution_output[:500]}
        )
        
        return self.process_signal(signal)
    
    def learn_from_user_feedback(
        self,
        module_name: str,
        input_data: Dict,
        output_data: Dict,
        user_said_correct: bool,
        user_comment: Optional[str] = None
    ) -> AdjustedLearningSignal:
        """
        Learn from user feedback (discounted by skepticism).
        """
        signal = LearningSignal(
            signal_id=f"user-{hashlib.md5(str(input_data).encode()).hexdigest()[:8]}",
            source=LearningSource.USER_FEEDBACK,
            evidence_type=EvidenceType.USER_LABEL,
            module_name=module_name,
            input_data=input_data,
            output_data=output_data,
            was_correct=user_said_correct,
            raw_confidence=0.8,  # User confidence is moderate
            timestamp=datetime.utcnow().isoformat(),
            context={'user_comment': user_comment or ''}
        )
        
        return self.process_signal(signal)
    
    def learn_from_consensus(
        self,
        module_name: str,
        input_data: Dict,
        output_data: Dict,
        llm_votes: Dict[str, bool],  # llm_name -> voted_correct
        consensus_correct: bool
    ) -> AdjustedLearningSignal:
        """
        Learn from LLM consensus.
        """
        # Calculate consensus strength
        total_votes = len(llm_votes)
        agreeing_votes = sum(1 for v in llm_votes.values() if v == consensus_correct)
        consensus_strength = agreeing_votes / total_votes if total_votes > 0 else 0.5
        
        signal = LearningSignal(
            signal_id=f"cons-{hashlib.md5(str(input_data).encode()).hexdigest()[:8]}",
            source=LearningSource.CROSS_VALIDATION,
            evidence_type=EvidenceType.LLM_CONSENSUS,
            module_name=module_name,
            input_data=input_data,
            output_data=output_data,
            was_correct=consensus_correct,
            raw_confidence=consensus_strength,
            timestamp=datetime.utcnow().isoformat(),
            context={'llm_votes': llm_votes}
        )
        
        return self.process_signal(signal)
    
    def get_learned_patterns(self, module_name: Optional[str] = None) -> Dict[str, Any]:
        """Get learned patterns, optionally filtered by module."""
        if module_name:
            return getattr(self, '_learned_patterns', {}).get(module_name, {})
        return getattr(self, '_learned_patterns', {}).copy()
    
    def get_trust_weights(self) -> Dict[str, float]:
        """Get current trust weights."""
        return {k.value: v for k, v in getattr(self, '_trust_weights', {}).items()}
    
    def set_user_label_discount(self, discount: float) -> None:
        """Update user label discount (skepticism)."""
        self.user_label_discount = max(0.1, min(1.0, discount))
        self._trust_weights[LearningSource.USER_FEEDBACK] = self.user_label_discount
        logger.info(f"[SkepticalLearning] User label discount set to {self.user_label_discount}")
    
    def _update_average_adjustment(self, raw: float, adjusted: float) -> None:
        """Update running average of confidence adjustments."""
        n = self._stats['total_signals']
        current_avg = self._stats['avg_adjustment']
        adjustment = raw - adjusted
        self._stats['avg_adjustment'] = (current_avg * (n - 1) + adjustment) / n
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        stats = getattr(self, '_stats', {}).copy()
        stats['by_source'] = dict(stats['by_source'])
        stats['user_label_discount'] = self.user_label_discount
        stats['acceptance_rate'] = self._stats['learned'] / max(1, self._stats['total_signals'])
        stats['learning_params'] = getattr(self, '_learning_params', {})
        stats['history_size'] = len(getattr(self, '_outcome_history', []))
        return stats
    
    # =========================================================================
    # Learning Interface (5/5 methods) - Additional for compatibility
    # =========================================================================
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict[str, Any]) -> None:
        """Record learning outcome (compatibility wrapper)."""
        if not hasattr(self, '_outcome_history'):
            self._outcome_history: List[Dict] = []
        
        getattr(self, '_outcome_history', {}).append({
            'timestamp': datetime.utcnow().isoformat(),
            **outcome
        })
        if len(self._outcome_history) > 1000:
            self._outcome_history = self._outcome_history[-500:]
    
    def learn_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """Learn from feedback to adjust skepticism."""
        if not hasattr(self, '_learning_params'):
            self._learning_params = {'skepticism_adjustment': 0.0}
        
        was_correct = feedback.get('was_correct', True)
        if not was_correct:
            if feedback.get('user_was_right', False):
                # User label was correct, reduce skepticism
                self.user_label_discount = min(1.0, self.user_label_discount + 0.05)
            else:
                # User label was wrong, increase skepticism
                self.user_label_discount = max(0.3, self.user_label_discount - 0.05)
    
    def serialize_state(self) -> Dict[str, Any]:
        """Serialize state for persistence."""
        return {
            'outcome_history': getattr(self, '_outcome_history', [])[-100:],
            'learning_params': getattr(self, '_learning_params', {}),
            'stats': {k: dict(v) if isinstance(v, defaultdict) else v for k, v in getattr(self, '_stats', {}).items()},
            'user_label_discount': self.user_label_discount,
            'learning_history': getattr(self, '_learning_history', [])[-100:],
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict[str, Any]) -> None:
        """Restore state from serialized data."""
        self._outcome_history = state.get('outcome_history', [])
        self._learning_params = state.get('learning_params', {})
        self.user_label_discount = state.get('user_label_discount', 0.7)
        self._learning_history = state.get('learning_history', [])
        for k, v in state.get('stats', {}).items():
            if k in self._stats:
                if isinstance(self._stats[k], defaultdict):
                    self._stats[k].update(v)
                else:
                    self._stats[k] = v


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SKEPTICAL LEARNING MANAGER TEST")
    print("=" * 80)
    
    manager = SkepticalLearningManager(user_label_discount=0.7)
    
    # Test 1: Execution evidence (highest trust)
    print("\n[1] Testing execution evidence (trust=1.0)...")
    result1 = manager.learn_from_execution(
        module_name="test_module",
        input_data={"query": "test"},
        output_data={"result": "success"},
        execution_succeeded=True,
        execution_output="All tests passed"
    )
    print(f"    Raw confidence: {result1.original.raw_confidence:.2f}")
    print(f"    Adjusted confidence: {result1.adjusted_confidence:.2f}")
    print(f"    Trust weight: {result1.trust_weight:.2f}")
    print(f"    Should learn: {result1.should_learn}")
    
    # Test 2: User feedback (discounted)
    print("\n[2] Testing user feedback (trust=0.7)...")
    result2 = manager.learn_from_user_feedback(
        module_name="test_module",
        input_data={"query": "test2"},
        output_data={"result": "looks good"},
        user_said_correct=True,
        user_comment="This worked for me"
    )
    print(f"    Raw confidence: {result2.original.raw_confidence:.2f}")
    print(f"    Adjusted confidence: {result2.adjusted_confidence:.2f}")
    print(f"    Trust weight: {result2.trust_weight:.2f}")
    print(f"    Should learn: {result2.should_learn}")
    print(f"    Discount applied: {result2.original.raw_confidence - result2.adjusted_confidence:.2f}")
    
    # Test 3: LLM consensus
    print("\n[3] Testing LLM consensus (trust=0.85)...")
    result3 = manager.learn_from_consensus(
        module_name="test_module",
        input_data={"query": "test3"},
        output_data={"result": "consensus result"},
        llm_votes={"grok": True, "openai": True, "anthropic": False},
        consensus_correct=True
    )
    print(f"    Consensus strength: {result3.original.raw_confidence:.2f}")
    print(f"    Adjusted confidence: {result3.adjusted_confidence:.2f}")
    print(f"    Trust weight: {result3.trust_weight:.2f}")
    print(f"    Should learn: {result3.should_learn}")
    
    # Test 4: Cross-validation
    print("\n[4] Testing cross-validation...")
    signal = LearningSignal(
        signal_id="test-sig",
        source=LearningSource.USER_FEEDBACK,
        evidence_type=EvidenceType.USER_LABEL,
        module_name="test_module",
        input_data={},
        output_data={},
        was_correct=True,
        raw_confidence=0.8
    )
    validation_signals = [
        LearningSignal(
            signal_id="val-1",
            source=LearningSource.EXECUTION,
            evidence_type=EvidenceType.RUNTIME_OUTPUT,
            module_name="test_module",
            input_data={},
            output_data={},
            was_correct=True,
            raw_confidence=1.0
        ),
        LearningSignal(
            signal_id="val-2",
            source=LearningSource.AUTOMATED_TEST,
            evidence_type=EvidenceType.TEST_RESULT,
            module_name="test_module",
            input_data={},
            output_data={},
            was_correct=False,  # Conflict!
            raw_confidence=0.95
        )
    ]
    
    cv_result = manager.cross_validate(signal, validation_signals)
    print(f"    Validated: {cv_result.validated}")
    print(f"    Agreement score: {cv_result.agreement_score:.2f}")
    print(f"    Conflicts: {cv_result.conflicts}")
    
    # Statistics
    print("\n[5] Statistics:")
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"    {key}: {value}")
    
    print("\n" + "=" * 80)
    print("✓ SKEPTICAL LEARNING MANAGER TEST COMPLETE")
    print("=" * 80)

