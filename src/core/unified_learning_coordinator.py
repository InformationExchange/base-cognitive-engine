"""
BASE Unified Learning Coordinator

Phase 16B: Wires together all learning components for cross-session neuroplasticity.

Integrates:
- PPA1-Inv22: Feedback Loop with Continuous Learning
- PPA2-Inv27: OCO Threshold Adapter (via threshold_optimizer)
- NOVEL-30: Dimensional Learning Loop
- LLMAwareLearning: Cross-LLM persistence
- PerformanceMetrics: Per-invention tracking

Patent Alignment:
- PPA1-Inv22: Continuous feedback loop learning
- PPA1-Inv24: Dynamic Bias Evolution (neuroplasticity)
- PPA2-Inv27: OCO Threshold Adapter
- NOVEL-30: Dimensional Learning Loop
- NOVEL-18: Learning Memory Persistence

Brain Layer: Layer 4 (Memory/Hippocampus)

This coordinator ensures learnings from one session transfer to the next,
enabling BASE to become more effective over time (neuroplasticity).
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from enum import Enum
import json
import threading
import time


class LearningSignalType(Enum):
    """Types of learning signals flowing through the system."""
    FEEDBACK = "feedback"              # Human/system feedback
    OUTCOME = "outcome"                # Evaluation outcome
    THRESHOLD_UPDATE = "threshold"     # Threshold adaptation
    DIMENSION_EFFECTIVENESS = "dimension"  # Dimension learning
    PATTERN_EFFECTIVENESS = "pattern"  # Pattern learning
    AB_RESULT = "ab_result"            # A/B test result
    BIAS_PROFILE = "bias_profile"      # LLM bias profile update


@dataclass
class LearningSignal:
    """A signal flowing through the learning system."""
    signal_type: LearningSignalType
    source_module: str                 # Which module generated this
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    domain: str = "general"
    llm_provider: str = None
    
    def to_dict(self) -> Dict:
        return {
            "signal_type": self.signal_type.value,
            "source_module": self.source_module,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "domain": self.domain,
            "llm_provider": self.llm_provider
        }


@dataclass
class LearningState:
    """Aggregated learning state across all components."""
    total_signals_processed: int = 0
    feedback_signals: int = 0
    outcome_signals: int = 0
    threshold_adaptations: int = 0
    dimension_updates: int = 0
    pattern_updates: int = 0
    ab_results_recorded: int = 0
    
    # Effectiveness metrics
    learning_velocity: float = 0.0     # Signals per minute
    improvement_trend: str = "stable"  # improving, stable, declining
    
    # Session info
    session_start: datetime = field(default_factory=datetime.now)
    last_signal: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            "total_signals_processed": self.total_signals_processed,
            "feedback_signals": self.feedback_signals,
            "outcome_signals": self.outcome_signals,
            "threshold_adaptations": self.threshold_adaptations,
            "dimension_updates": self.dimension_updates,
            "pattern_updates": self.pattern_updates,
            "ab_results_recorded": self.ab_results_recorded,
            "learning_velocity": round(self.learning_velocity, 2),
            "improvement_trend": self.improvement_trend,
            "session_start": self.session_start.isoformat(),
            "last_signal": self.last_signal.isoformat() if self.last_signal else None
        }


class UnifiedLearningCoordinator:
    """
    Central coordinator for all BASE learning activities.
    
    Implements the "neuroplasticity" concept where BASE:
    1. Receives signals from various sources (feedback, outcomes, A/B tests)
    2. Routes signals to appropriate learning modules
    3. Aggregates learnings across modules
    4. Persists state for cross-session continuity
    5. Provides unified API for learning operations
    
    Signal Flow:
    
    [Evaluation] ──► [Coordinator] ──► [ThresholdOptimizer]
                         │
    [Feedback]   ──►     ├──────────► [DimensionalLearning]
                         │
    [A/B Test]   ──►     ├──────────► [LLMAwareLearning]
                         │
    [Metrics]    ──►     └──────────► [PerformanceTracker]
    """
    
    def __init__(self, storage_path: Path = None):
        """Initialize the unified learning coordinator."""
        self.storage_path = storage_path or Path("unified_learning")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Learning state
        self.state = LearningState()
        
        # Signal queue for batch processing
        self._signal_queue: List[LearningSignal] = []
        
        # Learning module references (lazy-loaded)
        self._threshold_optimizer = None
        self._dimensional_learning = None
        self._llm_aware_learning = None
        self._feedback_loop = None
        self._performance_tracker = None
        self._domain_pattern_learner = None
        
        # Load persisted state
        self._load_state()
    
    def _get_threshold_optimizer(self):
        """Lazy-load threshold optimizer."""
        if self._threshold_optimizer is None:
            try:
                from learning.threshold_optimizer import AdaptiveThresholdOptimizer
                self._threshold_optimizer = AdaptiveThresholdOptimizer(
                    data_dir=self.storage_path / "thresholds"
                )
            except Exception as e:
                print(f"[UnifiedLearning] ThresholdOptimizer load error: {e}")
        return self._threshold_optimizer
    
    def _get_dimensional_learning(self):
        """Lazy-load dimensional learning."""
        if self._dimensional_learning is None:
            try:
                from core.dimensional_learning import DimensionalLearning
                self._dimensional_learning = DimensionalLearning(
                    storage_path=self.storage_path / "dimensions"
                )
            except Exception as e:
                print(f"[UnifiedLearning] DimensionalLearning load error: {e}")
        return self._dimensional_learning
    
    def _get_llm_aware_learning(self):
        """Lazy-load LLM-aware learning."""
        if self._llm_aware_learning is None:
            try:
                from core.llm_aware_learning import LLMAwareLearningManager
                self._llm_aware_learning = LLMAwareLearningManager(
                    storage_path=self.storage_path / "llm_aware"
                )
            except Exception as e:
                print(f"[UnifiedLearning] LLMAwareLearning load error: {e}")
        return self._llm_aware_learning
    
    def _get_feedback_loop(self):
        """Lazy-load feedback loop."""
        if self._feedback_loop is None:
            try:
                from learning.feedback_loop import ContinuousFeedbackLoop
                self._feedback_loop = ContinuousFeedbackLoop(
                    storage_path=self.storage_path / "feedback"
                )
            except Exception as e:
                print(f"[UnifiedLearning] FeedbackLoop load error: {e}")
        return self._feedback_loop
    
    def _get_performance_tracker(self):
        """Lazy-load performance tracker."""
        if self._performance_tracker is None:
            try:
                from core.performance_metrics import PerformanceTracker
                self._performance_tracker = PerformanceTracker(
                    storage_path=self.storage_path / "metrics"
                )
            except Exception as e:
                print(f"[UnifiedLearning] PerformanceTracker load error: {e}")
        return self._performance_tracker
    
    def _get_domain_pattern_learner(self):
        """Lazy-load domain pattern learner."""
        if self._domain_pattern_learner is None:
            try:
                from core.domain_pattern_learning import DomainPatternLearner
                self._domain_pattern_learner = DomainPatternLearner(
                    storage_path=self.storage_path / "patterns"
                )
            except Exception as e:
                print(f"[UnifiedLearning] DomainPatternLearner load error: {e}")
        return self._domain_pattern_learner
    
    def record_signal(self, signal: LearningSignal):
        """
        Record a learning signal and route to appropriate handlers.
        
        This is the main entry point for all learning activities.
        """
        with self._lock:
            self.state.total_signals_processed += 1
            self.state.last_signal = datetime.now()
            
            # Update velocity
            elapsed = (datetime.now() - self.state.session_start).total_seconds() / 60
            if elapsed > 0:
                self.state.learning_velocity = self.state.total_signals_processed / elapsed
            
            # Route to appropriate handler
            self._route_signal(signal)
            
            # Queue for batch persistence
            self._signal_queue.append(signal)
            
            # Persist if queue is large enough
            if len(self._signal_queue) >= 10:
                self._flush_signals()
    
    def _route_signal(self, signal: LearningSignal):
        """Route signal to appropriate learning modules."""
        
        if signal.signal_type == LearningSignalType.FEEDBACK:
            self.state.feedback_signals += 1
            self._handle_feedback(signal)
            
        elif signal.signal_type == LearningSignalType.OUTCOME:
            self.state.outcome_signals += 1
            self._handle_outcome(signal)
            
        elif signal.signal_type == LearningSignalType.THRESHOLD_UPDATE:
            self.state.threshold_adaptations += 1
            self._handle_threshold_update(signal)
            
        elif signal.signal_type == LearningSignalType.DIMENSION_EFFECTIVENESS:
            self.state.dimension_updates += 1
            self._handle_dimension_update(signal)
            
        elif signal.signal_type == LearningSignalType.PATTERN_EFFECTIVENESS:
            self.state.pattern_updates += 1
            self._handle_pattern_update(signal)
            
        elif signal.signal_type == LearningSignalType.AB_RESULT:
            self.state.ab_results_recorded += 1
            self._handle_ab_result(signal)
            
        elif signal.signal_type == LearningSignalType.BIAS_PROFILE:
            self._handle_bias_profile_update(signal)
    
    def _handle_feedback(self, signal: LearningSignal):
        """Handle feedback signal - route to FeedbackLoop and ThresholdOptimizer."""
        data = signal.data
        
        # Update feedback loop
        feedback_loop = self._get_feedback_loop()
        if feedback_loop:
            try:
                feedback_loop.record_feedback(
                    decision_id=data.get("decision_id", "unknown"),
                    was_correct=data.get("was_correct", False),
                    domain=signal.domain,
                    feedback_type=data.get("feedback_type", "accuracy")
                )
            except Exception as e:
                print(f"[UnifiedLearning] Feedback record error: {e}")
        
        # Update threshold optimizer if there's an outcome
        if "was_correct" in data:
            threshold_optimizer = self._get_threshold_optimizer()
            if threshold_optimizer:
                try:
                    threshold_optimizer.record_outcome(
                        domain=signal.domain,
                        decision="accept" if data.get("was_correct") else "reject",
                        outcome="correct" if data.get("was_correct") else "incorrect",
                        context=data.get("context", {})
                    )
                except Exception as e:
                    print(f"[UnifiedLearning] Threshold update error: {e}")
    
    def _handle_outcome(self, signal: LearningSignal):
        """Handle evaluation outcome - update multiple learning systems."""
        data = signal.data
        
        # Update performance tracker
        tracker = self._get_performance_tracker()
        if tracker and "invention_id" in data:
            try:
                tracker.record_invention_outcome(
                    invention_id=data["invention_id"],
                    was_correct=data.get("was_correct", True),
                    domain=signal.domain
                )
            except Exception as e:
                print(f"[UnifiedLearning] Performance update error: {e}")
        
        # Update LLM-aware learning if provider specified
        if signal.llm_provider:
            llm_learning = self._get_llm_aware_learning()
            if llm_learning:
                try:
                    llm_learning.record_learning_outcome(
                        learning_id=data.get("learning_id", "unknown"),
                        was_successful=data.get("was_correct", True)
                    )
                except Exception as e:
                    print(f"[UnifiedLearning] LLM learning update error: {e}")
    
    def _handle_threshold_update(self, signal: LearningSignal):
        """Handle explicit threshold update request."""
        data = signal.data
        
        threshold_optimizer = self._get_threshold_optimizer()
        if threshold_optimizer:
            try:
                threshold_optimizer.record_outcome(
                    domain=signal.domain,
                    decision=data.get("decision", "accept"),
                    outcome=data.get("outcome", "correct"),
                    context=data.get("context", {})
                )
            except Exception as e:
                print(f"[UnifiedLearning] Threshold update error: {e}")
    
    def _handle_dimension_update(self, signal: LearningSignal):
        """Handle dimension effectiveness update."""
        data = signal.data
        
        dim_learning = self._get_dimensional_learning()
        if dim_learning:
            try:
                dim_learning.record_outcome(
                    query_hash=data.get("query_hash", "unknown"),
                    task_type=data.get("task_type"),
                    complexity=data.get("complexity"),
                    dimensions_used=data.get("dimensions_used", []),
                    outcome=data.get("outcome"),
                    confidence=data.get("confidence", 0.7),
                    feedback_source="coordinator"
                )
            except Exception as e:
                print(f"[UnifiedLearning] Dimension update error: {e}")
    
    def _handle_pattern_update(self, signal: LearningSignal):
        """Handle pattern effectiveness update."""
        data = signal.data
        
        pattern_learner = self._get_domain_pattern_learner()
        if pattern_learner:
            try:
                pattern_learner.record_pattern_outcome(
                    pattern_id=data.get("pattern_id"),
                    domain=signal.domain,
                    was_correct=data.get("was_correct", True)
                )
            except Exception as e:
                print(f"[UnifiedLearning] Pattern update error: {e}")
    
    def _handle_ab_result(self, signal: LearningSignal):
        """Handle A/B test result - update performance and learning."""
        data = signal.data
        
        # Update performance tracker
        tracker = self._get_performance_tracker()
        if tracker and "invention_id" in data:
            try:
                tracker.record_ab_result(
                    invention_id=data["invention_id"],
                    result=data.get("result", "tie")  # win, loss, tie
                )
            except Exception as e:
                print(f"[UnifiedLearning] A/B result error: {e}")
        
        # Update dimensional learning if dimensions were involved
        if "dimensions_used" in data:
            self._handle_dimension_update(signal)
    
    def _handle_bias_profile_update(self, signal: LearningSignal):
        """Handle LLM bias profile update."""
        data = signal.data
        
        llm_learning = self._get_llm_aware_learning()
        if llm_learning and signal.llm_provider:
            try:
                llm_learning.update_bias_profile(
                    bias_type=data.get("bias_type"),
                    severity=data.get("severity", 0.5),
                    llm=None  # Use current
                )
            except Exception as e:
                print(f"[UnifiedLearning] Bias profile update error: {e}")
    
    # === Convenience Methods ===
    
    def record_evaluation_outcome(
        self,
        query: str,
        response: str,
        decision: str,
        was_correct: bool,
        domain: str = "general",
        issues_found: List[str] = None,
        inventions_used: List[str] = None,
        llm_provider: str = None
    ):
        """
        Record the outcome of a BASE evaluation.
        
        This is the main method called after each evaluation to enable learning.
        """
        # Record outcome signal
        self.record_signal(LearningSignal(
            signal_type=LearningSignalType.OUTCOME,
            source_module="evaluation",
            domain=domain,
            llm_provider=llm_provider,
            data={
                "query_hash": hash(query) % 10**8,
                "decision": decision,
                "was_correct": was_correct,
                "issues_found": issues_found or [],
                "inventions_used": inventions_used or []
            }
        ))
        
        # Record per-invention outcomes
        if inventions_used:
            for inv_id in inventions_used:
                self.record_signal(LearningSignal(
                    signal_type=LearningSignalType.OUTCOME,
                    source_module="evaluation",
                    domain=domain,
                    llm_provider=llm_provider,
                    data={
                        "invention_id": inv_id,
                        "was_correct": was_correct
                    }
                ))
        
        # Record threshold feedback
        self.record_signal(LearningSignal(
            signal_type=LearningSignalType.FEEDBACK,
            source_module="evaluation",
            domain=domain,
            data={
                "decision_id": f"eval_{hash(query) % 10**8}",
                "was_correct": was_correct,
                "feedback_type": "accuracy"
            }
        ))
    
    def record_ab_test_result(
        self,
        query: str,
        track_a_result: Dict[str, Any],
        track_b_result: Dict[str, Any],
        winner: str,  # "A", "B", or "tie"
        domain: str = "general",
        inventions_used: List[str] = None
    ):
        """Record the result of an A/B test."""
        # Determine result for Track B (BASE)
        if winner == "B":
            result = "win"
        elif winner == "A":
            result = "loss"
        else:
            result = "tie"
        
        # Record for each invention
        if inventions_used:
            for inv_id in inventions_used:
                self.record_signal(LearningSignal(
                    signal_type=LearningSignalType.AB_RESULT,
                    source_module="ab_test",
                    domain=domain,
                    data={
                        "invention_id": inv_id,
                        "result": result,
                        "track_a": track_a_result,
                        "track_b": track_b_result
                    }
                ))
    
    def record_dimension_effectiveness(
        self,
        query: str,
        task_type: str,
        complexity: str,
        dimensions_used: List[str],
        was_helpful: bool,
        domain: str = "general"
    ):
        """Record how effective dimensions were for a query."""
        outcome = "HELPFUL" if was_helpful else "NOT_HELPFUL"
        
        self.record_signal(LearningSignal(
            signal_type=LearningSignalType.DIMENSION_EFFECTIVENESS,
            source_module="dimension_analysis",
            domain=domain,
            data={
                "query_hash": hash(query) % 10**8,
                "task_type": task_type,
                "complexity": complexity,
                "dimensions_used": dimensions_used,
                "outcome": outcome,
                "confidence": 0.8
            }
        ))
    
    def record_pattern_effectiveness(
        self,
        pattern_id: str,
        was_correct: bool,
        domain: str = "general"
    ):
        """Record whether a pattern detection was correct."""
        self.record_signal(LearningSignal(
            signal_type=LearningSignalType.PATTERN_EFFECTIVENESS,
            source_module="pattern_detection",
            domain=domain,
            data={
                "pattern_id": pattern_id,
                "was_correct": was_correct
            }
        ))
    
    # === Query Methods ===
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics."""
        stats = {
            "coordinator_state": self.state.to_dict(),
            "modules": {}
        }
        
        # Threshold optimizer stats
        threshold_opt = self._get_threshold_optimizer()
        if threshold_opt:
            try:
                stats["modules"]["threshold_optimizer"] = {
                    "algorithm": threshold_opt.algorithm_name,
                    "thresholds_learned": len(getattr(threshold_opt, '_domain_thresholds', {}))
                }
            except:
                pass
        
        # Dimensional learning stats
        dim_learning = self._get_dimensional_learning()
        if dim_learning:
            try:
                stats["modules"]["dimensional_learning"] = dim_learning.get_learning_report()
            except:
                pass
        
        # LLM-aware learning stats
        llm_learning = self._get_llm_aware_learning()
        if llm_learning:
            try:
                stats["modules"]["llm_aware_learning"] = llm_learning.get_transfer_statistics()
            except:
                pass
        
        # Performance tracker stats
        tracker = self._get_performance_tracker()
        if tracker:
            try:
                stats["modules"]["performance_tracker"] = {
                    "inventions_tracked": len(tracker.invention_metrics),
                    "overall_accuracy": tracker._calculate_overall_accuracy()
                }
            except:
                pass
        
        return stats
    
    def get_improvement_recommendations(self) -> List[str]:
        """Get recommendations for improving BASE effectiveness."""
        recommendations = []
        
        # Check learning velocity
        if self.state.learning_velocity < 0.1:
            recommendations.append(
                "Low learning velocity - consider enabling more feedback collection"
            )
        
        # Check A/B results
        if self.state.ab_results_recorded < 10:
            recommendations.append(
                "Limited A/B test data - run more A/B tests to validate BASE effectiveness"
            )
        
        # Check dimension updates
        if self.state.dimension_updates < self.state.outcome_signals * 0.1:
            recommendations.append(
                "Dimensional learning underutilized - enable dimension tracking for evaluations"
            )
        
        # Check pattern updates
        if self.state.pattern_updates < 5:
            recommendations.append(
                "Pattern learning not active - record pattern effectiveness feedback"
            )
        
        return recommendations
    
    # === Persistence ===
    
    def _flush_signals(self):
        """Persist queued signals."""
        if not self._signal_queue:
            return
        
        signals_file = self.storage_path / "signal_log.jsonl"
        try:
            with open(signals_file, 'a') as f:
                for signal in self._signal_queue:
                    f.write(json.dumps(signal.to_dict()) + "\n")
            self._signal_queue.clear()
        except Exception as e:
            print(f"[UnifiedLearning] Signal flush error: {e}")
    
    def _save_state(self):
        """Save coordinator state."""
        state_file = self.storage_path / "coordinator_state.json"
        try:
            with open(state_file, 'w') as f:
                json.dump(self.state.to_dict(), f, indent=2)
        except Exception as e:
            print(f"[UnifiedLearning] State save error: {e}")
    
    def _load_state(self):
        """Load coordinator state."""
        state_file = self.storage_path / "coordinator_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    self.state.total_signals_processed = data.get("total_signals_processed", 0)
                    self.state.feedback_signals = data.get("feedback_signals", 0)
                    self.state.outcome_signals = data.get("outcome_signals", 0)
                    self.state.threshold_adaptations = data.get("threshold_adaptations", 0)
                    self.state.dimension_updates = data.get("dimension_updates", 0)
                    self.state.pattern_updates = data.get("pattern_updates", 0)
                    self.state.ab_results_recorded = data.get("ab_results_recorded", 0)
            except Exception as e:
                print(f"[UnifiedLearning] State load error: {e}")
    
    def shutdown(self):
        """Gracefully shutdown the coordinator."""
        with self._lock:
            # Flush remaining signals
            self._flush_signals()
            
            # Save state
            self._save_state()
            
            # Save module states
            tracker = self._get_performance_tracker()
            if tracker:
                try:
                    tracker.save_metrics()
                except:
                    pass

    # Learning Interface
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        if not hasattr(self, '_outcomes'): self._outcomes = []
        self._outcomes.append(outcome)
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        pass
    
    def get_statistics(self) -> Dict:
        return {'outcomes': len(getattr(self, '_outcomes', []))}
    
    def serialize_state(self) -> Dict:
        return {'outcomes': getattr(self, '_outcomes', [])[-100:]}
    
    def deserialize_state(self, state: Dict) -> None:
        self._outcomes = state.get('outcomes', [])


# Singleton instance
unified_learning = UnifiedLearningCoordinator()

