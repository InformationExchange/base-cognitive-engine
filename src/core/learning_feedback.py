"""
BAIS Learning Feedback (Layer 6 - Basal Ganglia)

Manages the feedback loop for continuous learning:
1. Outcome recording
2. Reward signal computation
3. Policy updates
4. Performance tracking

Patent Alignment:
- PPA2-Comp1 through PPA2-Comp7: Learning algorithms
- Brain Layer: 6 (Basal Ganglia - Learning/Feedback)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum
import time


class FeedbackType(Enum):
    """Types of feedback."""
    EXPLICIT = "explicit"      # User provided feedback
    IMPLICIT = "implicit"      # Inferred from behavior
    OUTCOME = "outcome"        # Task completion result
    CORRECTION = "correction"  # Error correction
    PREFERENCE = "preference"  # User preference signal


@dataclass
class FeedbackSignal:
    """A feedback signal for learning."""
    signal_id: str
    feedback_type: FeedbackType
    value: float  # -1 to 1, where -1 is negative, 1 is positive
    context: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    source: str = "system"


@dataclass
class LearningUpdate:
    """A learning update to apply."""
    target_module: str
    parameter_updates: Dict[str, float]
    confidence: float
    reason: str


@dataclass
class FeedbackSummary:
    """Summary of feedback over a period."""
    total_signals: int
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    average_value: float
    by_type: Dict[FeedbackType, int]


class LearningFeedback:
    """
    Manages learning feedback for BAIS modules.
    
    Brain Layer: 6 (Basal Ganglia)
    
    Responsibilities:
    1. Collect feedback signals
    2. Compute reward signals
    3. Determine policy updates
    4. Track learning progress
    """
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def __init__(self, learning_rate: float = 0.01,
                 discount_factor: float = 0.95):
        """
        Initialize the learning feedback manager.
        
        Args:
            learning_rate: Learning rate for updates
            discount_factor: Discount for future rewards
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Feedback storage
        self._feedback_buffer: List[FeedbackSignal] = []
        self._processed_feedback: List[FeedbackSignal] = []
        
        # Learning state
        self._module_performance: Dict[str, List[float]] = {}
        self._domain_adjustments: Dict[str, float] = {}
        self._total_feedback: int = 0
        
        # Registered callbacks
        self._update_callbacks: Dict[str, Callable] = {}
    
    def record(self, feedback: FeedbackSignal) -> None:
        """
        Record a feedback signal.
        
        Args:
            feedback: Feedback signal to record
        """
        self._total_feedback += 1
        self._feedback_buffer.append(feedback)
        
        # Keep buffer bounded
        if len(self._feedback_buffer) > 1000:
            self._feedback_buffer = self._feedback_buffer[-500:]
    
    def record_outcome(self, module: str, outcome: float,
                       context: Dict[str, Any] = None) -> None:
        """
        Record an outcome for a module.
        
        Args:
            module: Module name
            outcome: Outcome value (0-1)
            context: Optional context
        """
        feedback = FeedbackSignal(
            signal_id=f"outcome_{self._total_feedback}",
            feedback_type=FeedbackType.OUTCOME,
            value=(outcome * 2) - 1,  # Convert 0-1 to -1 to 1
            context={'module': module, **(context or {})}
        )
        self.record(feedback)
        
        # Track module performance
        if module not in self._module_performance:
            self._module_performance[module] = []
        self._module_performance[module].append(outcome)
        
        # Keep bounded
        if len(self._module_performance[module]) > 100:
            self._module_performance[module] = self._module_performance[module][-100:]
    
    def compute_reward(self, signals: List[FeedbackSignal] = None) -> float:
        """
        Compute aggregate reward from feedback signals.
        
        Args:
            signals: Signals to compute from (uses buffer if None)
            
        Returns:
            Computed reward value
        """
        signals = signals or self._feedback_buffer
        
        if not signals:
            return 0.0
        
        # Weight by recency
        total_weight = 0.0
        weighted_sum = 0.0
        
        now = time.time()
        for i, signal in enumerate(signals):
            age = now - signal.timestamp
            weight = self.discount_factor ** (age / 3600)  # Decay per hour
            
            # Type weighting
            type_weight = {
                FeedbackType.EXPLICIT: 1.0,
                FeedbackType.IMPLICIT: 0.5,
                FeedbackType.OUTCOME: 0.8,
                FeedbackType.CORRECTION: 0.9,
                FeedbackType.PREFERENCE: 0.6,
            }.get(signal.feedback_type, 0.5)
            
            combined_weight = weight * type_weight
            weighted_sum += signal.value * combined_weight
            total_weight += combined_weight
        
        return weighted_sum / max(total_weight, 0.001)
    
    def get_updates(self, module: str = None) -> List[LearningUpdate]:
        """
        Get pending learning updates.
        
        Args:
            module: Filter by module (None for all)
            
        Returns:
            List of learning updates
        """
        updates = []
        
        # Analyze feedback for each module
        for mod, performance in self._module_performance.items():
            if module and mod != module:
                continue
            
            if len(performance) < 10:
                continue
            
            recent = performance[-20:]
            avg_performance = sum(recent) / len(recent)
            
            # Determine if update needed
            if avg_performance < 0.5:
                # Needs improvement
                updates.append(LearningUpdate(
                    target_module=mod,
                    parameter_updates={'threshold': -self.learning_rate},
                    confidence=0.7,
                    reason=f"Low performance ({avg_performance:.2f})"
                ))
            elif avg_performance > 0.9:
                # Can be more aggressive
                updates.append(LearningUpdate(
                    target_module=mod,
                    parameter_updates={'threshold': self.learning_rate},
                    confidence=0.6,
                    reason=f"High performance ({avg_performance:.2f})"
                ))
        
        return updates
    
    def register_callback(self, module: str, callback: Callable) -> None:
        """
        Register a callback for learning updates.
        
        Args:
            module: Module to register for
            callback: Callback function
        """
        self._update_callbacks[module] = callback
    
    def apply_updates(self) -> int:
        """
        Apply pending updates via callbacks.
        
        Returns:
            Number of updates applied
        """
        updates = self.get_updates()
        applied = 0
        
        for update in updates:
            if update.target_module in self._update_callbacks:
                try:
                    self._update_callbacks[update.target_module](update)
                    applied += 1
                except Exception:
                    pass
        
        return applied
    
    def get_summary(self, last_n: int = 100) -> FeedbackSummary:
        """
        Get summary of recent feedback.
        
        Args:
            last_n: Number of recent signals to summarize
            
        Returns:
            FeedbackSummary
        """
        signals = self._feedback_buffer[-last_n:] if self._feedback_buffer else []
        
        if not signals:
            return FeedbackSummary(
                total_signals=0,
                positive_ratio=0.0,
                negative_ratio=0.0,
                neutral_ratio=1.0,
                average_value=0.0,
                by_type={}
            )
        
        positive = sum(1 for s in signals if s.value > 0.2)
        negative = sum(1 for s in signals if s.value < -0.2)
        neutral = len(signals) - positive - negative
        
        by_type = {}
        for s in signals:
            by_type[s.feedback_type] = by_type.get(s.feedback_type, 0) + 1
        
        avg_value = sum(s.value for s in signals) / len(signals)
        
        return FeedbackSummary(
            total_signals=len(signals),
            positive_ratio=positive / len(signals),
            negative_ratio=negative / len(signals),
            neutral_ratio=neutral / len(signals),
            average_value=avg_value,
            by_type=by_type
        )
    
    # ===== Learning Interface Methods =====
    
    def record_feedback(self, feedback: Dict) -> None:
        """Record feedback for learning adjustment."""
        signal = FeedbackSignal(
            signal_id=f"fb_{self._total_feedback}",
            feedback_type=FeedbackType.EXPLICIT,
            value=feedback.get('value', 0.0),
            context=feedback
        )
        self.record(signal)
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt learning thresholds."""
        if performance_data:
            avg = performance_data.get('average', 0.5)
            if avg < 0.4:
                self.learning_rate = min(0.05, self.learning_rate * 1.1)
            elif avg > 0.8:
                self.learning_rate = max(0.001, self.learning_rate * 0.9)
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get domain adjustment."""
        return self._domain_adjustments.get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics."""
        summary = self.get_summary()
        
        module_perf = {}
        for mod, perf in self._module_performance.items():
            if perf:
                module_perf[mod] = sum(perf[-20:]) / len(perf[-20:])
        
        return {
            'total_feedback': self._total_feedback,
            'buffer_size': len(self._feedback_buffer),
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'positive_ratio': summary.positive_ratio,
            'negative_ratio': summary.negative_ratio,
            'average_value': summary.average_value,
            'module_performance': module_perf,
            'domain_adjustments': dict(self._domain_adjustments)
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
    feedback = LearningFeedback()
    
    print("=" * 60)
    print("LEARNING FEEDBACK TEST")
    print("=" * 60)
    
    # Simulate some outcomes
    for i in range(20):
        outcome = 0.7 + (i % 3) * 0.1  # Varying outcomes
        feedback.record_outcome("test_module", outcome, {'iteration': i})
    
    # Get summary
    summary = feedback.get_summary()
    print(f"\nFeedback Summary:")
    print(f"  Total signals: {summary.total_signals}")
    print(f"  Positive ratio: {summary.positive_ratio:.2f}")
    print(f"  Average value: {summary.average_value:.2f}")
    
    # Get updates
    updates = feedback.get_updates()
    print(f"\nPending Updates: {len(updates)}")
    for u in updates:
        print(f"  - {u.target_module}: {u.reason}")
    
    print(f"\nLearning stats: {feedback.get_learning_statistics()}")

