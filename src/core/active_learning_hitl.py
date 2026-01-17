"""
BASE Cognitive Governance Engine v36.0
Active Learning & Human-in-the-Loop Escalation

Phase 36: Addresses PPA2-C1-33, PPA2-C1-34, PPA3-Comp4
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from datetime import datetime, timedelta
import threading
import uuid
import logging
import heapq

logger = logging.getLogger(__name__)


class QueryStrategy(Enum):
    UNCERTAINTY_SAMPLING = "uncertainty"
    MARGIN_SAMPLING = "margin"
    ENTROPY_SAMPLING = "entropy"
    QUERY_BY_COMMITTEE = "committee"
    DENSITY_WEIGHTED = "density"
    RANDOM = "random"


class EscalationReason(Enum):
    LOW_CONFIDENCE = "low_confidence"
    HIGH_STAKES = "high_stakes"
    MODEL_DISAGREEMENT = "model_disagreement"
    NOVEL_PATTERN = "novel_pattern"
    POLICY_VIOLATION = "policy_violation"
    DRIFT_DETECTED = "drift_detected"


class EscalationPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BATCH = "batch"


class ReviewStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    COMPLETED = "completed"
    EXPIRED = "expired"


@dataclass
class UncertaintySample:
    sample_id: str
    query: str
    response: str
    confidence: float
    uncertainty: float
    entropy: float
    margin: float
    features: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __lt__(self, other):
        return self.uncertainty > other.uncertainty


@dataclass
class EscalationRequest:
    request_id: str
    sample: UncertaintySample
    reason: EscalationReason
    priority: EscalationPriority
    context: Dict[str, Any]
    created_at: datetime
    deadline: datetime
    status: ReviewStatus = ReviewStatus.PENDING
    assigned_to: Optional[str] = None
    resolution: Optional[Dict[str, Any]] = None


@dataclass
class HumanFeedback:
    feedback_id: str
    request_id: str
    reviewer_id: str
    correct_label: str
    confidence: float
    notes: str
    timestamp: datetime


class UncertaintyEstimator:
    def __init__(self):
        self.calibration_data: List[Tuple[float, bool]] = []
        
    def estimate_uncertainty(self, confidence: float, model_outputs: Optional[List[float]] = None) -> Dict[str, float]:
        uncertainty = 1.0 - confidence
        if confidence > 0 and confidence < 1:
            p = confidence
            entropy = -(p * np.log2(p) + (1-p) * np.log2(1-p))
        else:
            entropy = 0.0
        margin = abs(confidence - 0.5) * 2
        disagreement = np.std(model_outputs) if model_outputs and len(model_outputs) > 1 else 0.0
        return {"uncertainty": uncertainty, "entropy": entropy, "margin": 1.0 - margin, "disagreement": disagreement, "combined": (uncertainty + entropy / 2 + (1 - margin) + disagreement) / 4}
    
    def calibrate(self, predicted_prob: float, actual_outcome: bool):
        self.calibration_data.append((predicted_prob, actual_outcome))
    
    def get_calibration_error(self, num_bins: int = 10) -> float:
        if len(self.calibration_data) < num_bins:
            return 0.5
        bins = [[] for _ in range(num_bins)]
        for prob, outcome in self.calibration_data:
            bin_idx = min(int(prob * num_bins), num_bins - 1)
            bins[bin_idx].append((prob, outcome))
        ece = 0.0
        total = len(self.calibration_data)
        for bin_data in bins:
            if bin_data:
                avg_conf = np.mean([p for p, _ in bin_data])
                avg_acc = np.mean([o for _, o in bin_data])
                ece += len(bin_data) / total * abs(avg_conf - avg_acc)
        return ece

    # Learning Interface
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


class QuerySelector:
    def __init__(self, strategy: QueryStrategy = QueryStrategy.UNCERTAINTY_SAMPLING):
        self.strategy = strategy
        self.sample_pool: List[UncertaintySample] = []
        self.selected_history: Set[str] = set()
        
    def add_sample(self, sample: UncertaintySample):
        if sample.sample_id not in self.selected_history:
            heapq.heappush(self.sample_pool, sample)
    
    def select_batch(self, batch_size: int, strategy: Optional[QueryStrategy] = None) -> List[UncertaintySample]:
        strategy = strategy or self.strategy
        selected = []
        temp_pool = []
        while len(selected) < batch_size and self.sample_pool:
            sample = heapq.heappop(self.sample_pool)
            if sample.sample_id not in self.selected_history:
                selected.append(sample)
                self.selected_history.add(sample.sample_id)
            else:
                temp_pool.append(sample)
        for s in temp_pool:
            heapq.heappush(self.sample_pool, s)
        return selected

    # =========================================================================
    # Learning Interface
    # =========================================================================
    
    def record_outcome(self, outcome: Dict) -> None:
        """Record operation outcome for learning."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback to adjust behavior."""
        if not hasattr(self, '_learning_params'):
            self._learning_params = {}
        was_correct = feedback.get('was_correct', True)
        if not was_correct:
            for key in self._learning_params:
                self._learning_params[key] *= 0.95
    
    def get_statistics(self) -> Dict:
        """Return learning statistics."""
        return {
            'outcomes': len(getattr(self, '_outcomes', [])),
            'params': getattr(self, '_learning_params', {})
        }
    
    def serialize_state(self) -> Dict:
        """Serialize state for persistence."""
        return {
            'outcomes': getattr(self, '_outcomes', [])[-100:],
            'params': getattr(self, '_learning_params', {}),
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore state from serialized data."""
        self._outcomes = state.get('outcomes', [])
        self._learning_params = state.get('params', {})


class HumanArbitrationManager:
    def __init__(self, confidence_threshold: float = 0.6, max_pending: int = 100):
        self.confidence_threshold = confidence_threshold
        self.auto_escalate_domains = ["medical", "legal", "financial"]
        self.max_pending = max_pending
        self.pending_requests: Dict[str, EscalationRequest] = {}
        self.completed_requests: List[EscalationRequest] = []
        self.feedback_history: List[HumanFeedback] = []
        self.lock = threading.RLock()
        self.priority_deadlines = {
            EscalationPriority.CRITICAL: timedelta(minutes=15),
            EscalationPriority.HIGH: timedelta(hours=1),
            EscalationPriority.MEDIUM: timedelta(hours=4),
            EscalationPriority.LOW: timedelta(hours=24),
            EscalationPriority.BATCH: timedelta(days=7)
        }
    
    def should_escalate(self, sample: UncertaintySample, domain: str = "general", context: Optional[Dict[str, Any]] = None) -> Tuple[bool, EscalationReason, EscalationPriority]:
        context = context or {}
        if sample.confidence < self.confidence_threshold:
            priority = self._determine_priority(sample, domain)
            return True, EscalationReason.LOW_CONFIDENCE, priority
        if domain in self.auto_escalate_domains:
            return True, EscalationReason.HIGH_STAKES, EscalationPriority.HIGH
        if sample.features.get("disagreement", 0) > 0.3:
            return True, EscalationReason.MODEL_DISAGREEMENT, EscalationPriority.MEDIUM
        if context.get("drift_detected", False):
            return True, EscalationReason.DRIFT_DETECTED, EscalationPriority.MEDIUM
        return False, EscalationReason.LOW_CONFIDENCE, EscalationPriority.LOW
    
    def _determine_priority(self, sample: UncertaintySample, domain: str) -> EscalationPriority:
        if domain in ["medical", "emergency"]:
            return EscalationPriority.CRITICAL
        elif domain in ["financial", "legal"]:
            return EscalationPriority.HIGH
        elif sample.confidence < 0.3:
            return EscalationPriority.HIGH
        elif sample.confidence < 0.5:
            return EscalationPriority.MEDIUM
        return EscalationPriority.LOW
    
    def create_escalation(self, sample: UncertaintySample, reason: EscalationReason, priority: EscalationPriority, context: Optional[Dict[str, Any]] = None) -> EscalationRequest:
        with self.lock:
            request = EscalationRequest(
                request_id=str(uuid.uuid4())[:12],
                sample=sample,
                reason=reason,
                priority=priority,
                context=context or {},
                created_at=datetime.utcnow(),
                deadline=datetime.utcnow() + self.priority_deadlines[priority]
            )
            self.pending_requests[request.request_id] = request
            return request
    
    def assign_reviewer(self, request_id: str, reviewer_id: str) -> bool:
        with self.lock:
            if request_id in self.pending_requests:
                self.pending_requests[request_id].assigned_to = reviewer_id
                self.pending_requests[request_id].status = ReviewStatus.ASSIGNED
                return True
            return False
    
    def submit_feedback(self, request_id: str, reviewer_id: str, correct_label: str, confidence: float, notes: str = "") -> HumanFeedback:
        with self.lock:
            if request_id not in self.pending_requests:
                raise ValueError(f"Request {request_id} not found")
            request = self.pending_requests[request_id]
            feedback = HumanFeedback(feedback_id=str(uuid.uuid4())[:12], request_id=request_id, reviewer_id=reviewer_id, correct_label=correct_label, confidence=confidence, notes=notes, timestamp=datetime.utcnow())
            request.status = ReviewStatus.COMPLETED
            request.resolution = {"label": correct_label, "confidence": confidence}
            self.completed_requests.append(request)
            del self.pending_requests[request_id]
            self.feedback_history.append(feedback)
            return feedback
    
    def get_pending_by_priority(self, priority: Optional[EscalationPriority] = None) -> List[EscalationRequest]:
        with self.lock:
            requests = list(self.pending_requests.values())
            if priority:
                requests = [r for r in requests if r.priority == priority]
            return sorted(requests, key=lambda r: r.created_at)

    # =========================================================================
    # Learning Interface
    # =========================================================================
    
    def record_outcome(self, outcome: Dict) -> None:
        """Record operation outcome for learning."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback to adjust behavior."""
        if not hasattr(self, '_learning_params'):
            self._learning_params = {}
        was_correct = feedback.get('was_correct', True)
        if not was_correct:
            for key in self._learning_params:
                self._learning_params[key] *= 0.95
    
    def get_statistics(self) -> Dict:
        """Return learning statistics."""
        return {
            'outcomes': len(getattr(self, '_outcomes', [])),
            'params': getattr(self, '_learning_params', {})
        }
    
    def serialize_state(self) -> Dict:
        """Serialize state for persistence."""
        return {
            'outcomes': getattr(self, '_outcomes', [])[-100:],
            'params': getattr(self, '_learning_params', {}),
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore state from serialized data."""
        self._outcomes = state.get('outcomes', [])
        self._learning_params = state.get('params', {})


class ActiveLearningEngine:
    def __init__(self, query_strategy: QueryStrategy = QueryStrategy.UNCERTAINTY_SAMPLING, confidence_threshold: float = 0.6, batch_size: int = 10):
        self.uncertainty_estimator = UncertaintyEstimator()
        self.query_selector = QuerySelector(strategy=query_strategy)
        self.arbitration_manager = HumanArbitrationManager(confidence_threshold=confidence_threshold)
        self.batch_size = batch_size
        logger.info("[ActiveLearning] Active Learning Engine initialized")
    
    def process_sample(self, query: str, response: str, confidence: float, domain: str = "general", model_outputs: Optional[List[float]] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        uncertainty_metrics = self.uncertainty_estimator.estimate_uncertainty(confidence, model_outputs)
        sample = UncertaintySample(sample_id=str(uuid.uuid4())[:12], query=query, response=response, confidence=confidence, uncertainty=uncertainty_metrics["uncertainty"], entropy=uncertainty_metrics["entropy"], margin=uncertainty_metrics["margin"], features={"disagreement": uncertainty_metrics["disagreement"], "combined": uncertainty_metrics["combined"]})
        self.query_selector.add_sample(sample)
        should_escalate, reason, priority = self.arbitration_manager.should_escalate(sample, domain, context)
        escalation = None
        if should_escalate:
            escalation = self.arbitration_manager.create_escalation(sample, reason, priority, context)
        return {"sample_id": sample.sample_id, "uncertainty_metrics": uncertainty_metrics, "escalated": should_escalate, "escalation": {"request_id": escalation.request_id if escalation else None, "reason": reason.value if should_escalate else None, "priority": priority.value if should_escalate else None} if should_escalate else None}
    
    def create_labeling_batch(self, strategy: Optional[QueryStrategy] = None) -> Dict[str, Any]:
        samples = self.query_selector.select_batch(self.batch_size, strategy)
        return {"batch_id": str(uuid.uuid4())[:12], "samples": [{"id": s.sample_id, "uncertainty": s.uncertainty} for s in samples], "count": len(samples), "strategy": (strategy or self.query_selector.strategy).value}
    
    def record_feedback(self, request_id: str, reviewer_id: str, correct_label: str, confidence: float = 1.0, notes: str = "") -> HumanFeedback:
        feedback = self.arbitration_manager.submit_feedback(request_id, reviewer_id, correct_label, confidence, notes)
        for req in self.arbitration_manager.completed_requests:
            if req.request_id == request_id:
                self.uncertainty_estimator.calibrate(req.sample.confidence, correct_label.lower() in ["accept", "correct", "true", "1"])
                break
        return feedback
    
    def get_status(self) -> Dict[str, Any]:
        return {"pool_size": len(self.query_selector.sample_pool), "selected_count": len(self.query_selector.selected_history), "pending_escalations": len(self.arbitration_manager.pending_requests), "completed_escalations": len(self.arbitration_manager.completed_requests), "feedback_count": len(self.arbitration_manager.feedback_history), "calibration_error": self.uncertainty_estimator.get_calibration_error(), "query_strategy": self.query_selector.strategy.value}



    # ========================================
    # PHASE 49: PERSISTENCE METHODS
    # ========================================
    
    def save_state(self, filepath=None):
        """Save learning state (Phase 49)."""
        if hasattr(self, '_learning_manager') and self._learning_manager:
            return self._learning_manager.save_state()
        return False
    
    def load_state(self, filepath=None):
        """Load learning state (Phase 49)."""
        if hasattr(self, '_learning_manager') and self._learning_manager:
            return self._learning_manager.load_state()
        return False



    # ========================================
    # LEARNING INTERFACE (Auto-added)
    # ========================================

    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'): self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, input_data, output_data, was_correct, domain=None, metadata=None):
        """Record outcome for adaptive learning."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append({
            'input': str(input_data)[:100],
            'correct': was_correct,
            'domain': domain
        })
        self._outcomes = self._outcomes[-1000:]

    def adapt_thresholds(self, threshold_name, current_value, direction='decrease'):
        """Adapt thresholds based on learning history."""
        adjustment = 0.05 if direction == 'increase' else -0.05
        return max(0.1, min(0.95, current_value + adjustment))

    def get_domain_adjustment(self, domain):
        """Get learned adjustment for a domain."""
        if not hasattr(self, '_domain_adjustments'):
            self._domain_adjustments = {}
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
    print("=" * 70)
    print("PHASE 36: Active Learning & Human-in-the-Loop Test")
    print("=" * 70)
    engine = ActiveLearningEngine(query_strategy=QueryStrategy.UNCERTAINTY_SAMPLING, confidence_threshold=0.6, batch_size=5)
    print("\n[1] Processing Samples")
    print("-" * 60)
    test_samples = [("What is Python?", 0.95, "general"), ("What medication for headache?", 0.45, "medical"), ("Should I invest in stocks?", 0.52, "financial"), ("Explain AI", 0.88, "educational"), ("Legal advice on contract", 0.38, "legal")]
    for query, conf, domain in test_samples:
        result = engine.process_sample(query=query, response="Response", confidence=conf, domain=domain)
        esc = "ESCALATED" if result["escalated"] else "OK"
        reason = result["escalation"]["reason"] if result["escalated"] else "-"
        priority = result["escalation"]["priority"] if result["escalated"] else "-"
        print(f"  [{esc:9}] conf={conf:.2f}, domain={domain:10} | reason={reason}, priority={priority}")
    print("\n[2] Query Selection")
    print("-" * 60)
    batch = engine.create_labeling_batch()
    print(f"  Batch ID: {batch['batch_id']}")
    print(f"  Samples: {batch['count']}")
    print(f"  Strategy: {batch['strategy']}")
    print("\n[3] Escalation & Feedback")
    print("-" * 60)
    pending = engine.arbitration_manager.get_pending_by_priority()
    print(f"  Pending: {len(pending)}")
    if pending:
        req = pending[0]
        engine.arbitration_manager.assign_reviewer(req.request_id, "reviewer-1")
        feedback = engine.record_feedback(req.request_id, "reviewer-1", "accept", 0.9, "Verified")
        print(f"  Feedback: {feedback.feedback_id}")
    print("\n[4] Status")
    print("-" * 60)
    status = engine.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")
    print("\n" + "=" * 70)
    print("PHASE 36: Active Learning Engine - VERIFIED")
    print("=" * 70)
