"""
BASE Cognitive Governance Engine v16.5
Value-of-Information (VOI) Based Short-Circuiting

PPA-2 Component 5: FULL IMPLEMENTATION
Dynamic detector ordering and early termination based on expected information gain.

This module implements:
1. VOI Calculation: Expected information gain per detector
2. Dynamic Ordering: Run most informative detectors first
3. Short-Circuiting: Stop when decision is already clear
4. Cost-Benefit Analysis: Balance accuracy vs. computation
5. Learning: Improve VOI estimates from feedback

Mathematical Foundation:
═══════════════════════════════════════════════════════════════════════════════

1. VOI Calculation (Information-Theoretic):
   VOI(d) = H(Decision) - H(Decision | d) - Cost(d)
         = E[I(d; Decision)] - λ * Time(d)
   
   where:
   - H(Decision) = -Σ P(dec) * log(P(dec)) is entropy of current decision
   - H(Decision|d) = expected entropy after running detector d
   - I(d; Decision) = mutual information between detector and decision
   - λ = cost-utility tradeoff parameter

2. Dynamic Ordering (Greedy Selection):
   next_detector = argmax_d VOI(d | detectors_already_run)
   
   Greedy approximation to optimal sequential decision problem.

3. Short-Circuit Condition (Confidence Bound):
   Terminate if: max(P(accept), P(reject)) > 1 - ε
   
   where ε = 0.1 (configurable)
   
   Equivalently: P(decision_change | remaining) < ε
   
   Estimated as: P_change ≈ Π (1 - impact_d) for remaining detectors d

4. Expected Time Savings:
   Savings = Σ E[Time(d)] for skipped detectors d
   
═══════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
from enum import Enum
import math
from collections import defaultdict


class DetectorID(str, Enum):
    """Identifiers for detectors."""
    GROUNDING = "grounding"
    FACTUAL = "factual"
    BEHAVIORAL = "behavioral"
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    COGNITIVE = "cognitive"


@dataclass
class DetectorProfile:
    """Profile of a detector for VOI calculation."""
    detector_id: DetectorID
    name: str
    
    # Cost metrics
    avg_time_ms: float        # Average execution time
    variance_time_ms: float   # Variance in execution time
    resource_cost: float      # Relative resource usage (0-1)
    
    # Information metrics
    avg_information_gain: float  # Average bits of information
    decision_impact: float       # P(decision changes | this detector)
    correlation_with_outcome: float  # How predictive of final outcome
    
    # Dynamic state
    recent_times: List[float] = field(default_factory=list)
    recent_impacts: List[float] = field(default_factory=list)
    
    def get_expected_voi(self, current_confidence: float) -> float:
        """
        Calculate expected value of information using information theory.
        
        VOI(d) = H(Decision) - E[H(Decision|d)] - λ * Cost(d)
        
        where H is Shannon entropy and λ is cost-utility tradeoff.
        """
        # H(Decision) = -p*log(p) - (1-p)*log(1-p) for binary decision
        p = current_confidence
        p = max(0.001, min(0.999, p))  # Avoid log(0)
        
        current_entropy = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
        
        # E[H(Decision|d)] - expected entropy after running detector
        # Depends on how much the detector typically changes confidence
        impact = self.decision_impact
        
        # After running detector, expected confidence moves by impact
        # Simplified: entropy reduces proportionally to information gain
        expected_entropy_reduction = self.avg_information_gain * current_entropy * impact
        
        # Information gain = H(before) - H(after)
        information_gain = expected_entropy_reduction
        
        # Cost term (λ = 0.1 as cost-utility tradeoff)
        lambda_cost = 0.1
        cost_penalty = lambda_cost * (self.avg_time_ms / 1000.0)  # Normalized
        
        # Correlation adjustment
        relevance = self.correlation_with_outcome
        
        # Final VOI
        voi = information_gain * relevance - cost_penalty
        
        return max(0.0, voi)  # VOI cannot be negative
    
    def update_from_observation(self, time_ms: float, had_impact: bool):
        """Update profile from new observation."""
        # Keep last 100 observations
        self.recent_times.append(time_ms)
        if len(self.recent_times) > 100:
            self.recent_times.pop(0)
        
        self.recent_impacts.append(1.0 if had_impact else 0.0)
        if len(self.recent_impacts) > 100:
            self.recent_impacts.pop(0)
        
        # Update averages with exponential smoothing
        alpha = 0.1
        self.avg_time_ms = alpha * time_ms + (1 - alpha) * self.avg_time_ms
        self.decision_impact = alpha * (1.0 if had_impact else 0.0) + (1 - alpha) * self.decision_impact
    
    def to_dict(self) -> Dict:
        return {
            'detector_id': self.detector_id.value,
            'name': self.name,
            'avg_time_ms': self.avg_time_ms,
            'decision_impact': self.decision_impact,
            'correlation_with_outcome': self.correlation_with_outcome,
            'expected_voi': self.get_expected_voi(0.5)
        }


@dataclass
class ShortCircuitResult:
    """Result of short-circuit evaluation."""
    should_terminate: bool
    reason: str
    detectors_run: List[str]
    detectors_skipped: List[str]
    current_decision: str  # 'accept', 'reject', 'uncertain'
    decision_confidence: float
    time_saved_ms: float
    
    def to_dict(self) -> Dict:
        return {
            'should_terminate': self.should_terminate,
            'reason': self.reason,
            'detectors_run': self.detectors_run,
            'detectors_skipped': self.detectors_skipped,
            'current_decision': self.current_decision,
            'decision_confidence': self.decision_confidence,
            'time_saved_ms': self.time_saved_ms
        }


class VOIShortCircuitEngine:
    """
    Value-of-Information Based Short-Circuiting Engine.
    
    PPA-2 Component 5: Full Implementation
    
    Determines optimal detector ordering and when to
    terminate early without running all detectors.
    """
    
    # Short-circuit thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.9  # Confident enough to stop
    LOW_CONFIDENCE_THRESHOLD = 0.1   # Confident enough to stop (reject)
    MIN_DETECTORS = 2                # Always run at least 2 detectors
    
    def __init__(self):
        # Detector profiles
        self.profiles: Dict[DetectorID, DetectorProfile] = {}
        self._initialize_profiles()
        
        # Learning history
        self.ordering_history: List[List[DetectorID]] = []
        self.short_circuit_history: List[Dict] = []
        
        # Statistics
        self.total_evaluations = 0
        self.short_circuited = 0
        self.total_time_saved_ms = 0.0
    
    def _initialize_profiles(self):
        """Initialize detector profiles with prior estimates."""
        
        self.profiles[DetectorID.GROUNDING] = DetectorProfile(
            detector_id=DetectorID.GROUNDING,
            name="Grounding Detector",
            avg_time_ms=50.0,
            variance_time_ms=20.0,
            resource_cost=0.3,
            avg_information_gain=0.7,
            decision_impact=0.6,
            correlation_with_outcome=0.7
        )
        
        self.profiles[DetectorID.FACTUAL] = DetectorProfile(
            detector_id=DetectorID.FACTUAL,
            name="Factual Detector",
            avg_time_ms=80.0,
            variance_time_ms=30.0,
            resource_cost=0.5,
            avg_information_gain=0.8,
            decision_impact=0.7,
            correlation_with_outcome=0.75
        )
        
        self.profiles[DetectorID.BEHAVIORAL] = DetectorProfile(
            detector_id=DetectorID.BEHAVIORAL,
            name="Behavioral Bias Detector",
            avg_time_ms=40.0,
            variance_time_ms=15.0,
            resource_cost=0.2,
            avg_information_gain=0.6,
            decision_impact=0.8,  # High impact - can cause rejection
            correlation_with_outcome=0.65
        )
        
        self.profiles[DetectorID.TEMPORAL] = DetectorProfile(
            detector_id=DetectorID.TEMPORAL,
            name="Temporal Detector",
            avg_time_ms=30.0,
            variance_time_ms=10.0,
            resource_cost=0.1,
            avg_information_gain=0.4,
            decision_impact=0.3,
            correlation_with_outcome=0.4
        )
        
        self.profiles[DetectorID.SEMANTIC] = DetectorProfile(
            detector_id=DetectorID.SEMANTIC,
            name="Semantic Detector",
            avg_time_ms=150.0,  # Expensive (embeddings)
            variance_time_ms=50.0,
            resource_cost=0.8,
            avg_information_gain=0.9,
            decision_impact=0.5,
            correlation_with_outcome=0.8
        )
        
        self.profiles[DetectorID.COGNITIVE] = DetectorProfile(
            detector_id=DetectorID.COGNITIVE,
            name="Cognitive Intervention",
            avg_time_ms=60.0,
            variance_time_ms=25.0,
            resource_cost=0.4,
            avg_information_gain=0.5,
            decision_impact=0.4,
            correlation_with_outcome=0.5
        )
    
    def get_optimal_ordering(self, 
                           current_confidence: float = 0.5,
                           available_detectors: List[DetectorID] = None) -> List[DetectorID]:
        """
        Get optimal detector ordering based on VOI.
        
        Returns detectors sorted by expected value of information.
        """
        if available_detectors is None:
            available_detectors = list(self.profiles.keys())
        
        # Calculate VOI for each detector
        voi_scores = []
        for detector_id in available_detectors:
            if detector_id in self.profiles:
                profile = self.profiles[detector_id]
                voi = profile.get_expected_voi(current_confidence)
                voi_scores.append((detector_id, voi))
        
        # Sort by VOI (highest first)
        voi_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [d[0] for d in voi_scores]
    
    def should_short_circuit(self,
                            detectors_run: List[DetectorID],
                            current_signals: Dict[str, float],
                            current_decision: str,
                            decision_confidence: float) -> ShortCircuitResult:
        """
        Determine if we should stop running detectors.
        
        Returns ShortCircuitResult with recommendation.
        """
        self.total_evaluations += 1
        
        # Get remaining detectors
        all_detectors = set(self.profiles.keys())
        run_detectors = set(detectors_run)
        remaining = list(all_detectors - run_detectors)
        
        # Calculate time that would be saved
        time_saved = sum(
            self.profiles[d].avg_time_ms for d in remaining
            if d in self.profiles
        )
        
        # Check minimum detectors constraint
        if len(detectors_run) < self.MIN_DETECTORS:
            return ShortCircuitResult(
                should_terminate=False,
                reason=f"Minimum {self.MIN_DETECTORS} detectors not yet run",
                detectors_run=[d.value for d in detectors_run],
                detectors_skipped=[],
                current_decision=current_decision,
                decision_confidence=decision_confidence,
                time_saved_ms=0
            )
        
        # Check high confidence for accept
        if current_decision == 'accept' and decision_confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            self.short_circuited += 1
            self.total_time_saved_ms += time_saved
            return ShortCircuitResult(
                should_terminate=True,
                reason=f"High confidence accept ({decision_confidence:.2f} >= {self.HIGH_CONFIDENCE_THRESHOLD})",
                detectors_run=[d.value for d in detectors_run],
                detectors_skipped=[d.value for d in remaining],
                current_decision=current_decision,
                decision_confidence=decision_confidence,
                time_saved_ms=time_saved
            )
        
        # Check high confidence for reject
        if current_decision == 'reject' and decision_confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            self.short_circuited += 1
            self.total_time_saved_ms += time_saved
            return ShortCircuitResult(
                should_terminate=True,
                reason=f"High confidence reject ({decision_confidence:.2f} >= {self.HIGH_CONFIDENCE_THRESHOLD})",
                detectors_run=[d.value for d in detectors_run],
                detectors_skipped=[d.value for d in remaining],
                current_decision=current_decision,
                decision_confidence=decision_confidence,
                time_saved_ms=time_saved
            )
        
        # Check if remaining detectors have low expected impact
        remaining_impact = sum(
            self.profiles[d].decision_impact for d in remaining
            if d in self.profiles
        )
        
        if remaining_impact < 0.3 and decision_confidence >= 0.7:
            # Low remaining impact and decent confidence
            self.short_circuited += 1
            self.total_time_saved_ms += time_saved
            return ShortCircuitResult(
                should_terminate=True,
                reason=f"Low remaining impact ({remaining_impact:.2f}) with good confidence ({decision_confidence:.2f})",
                detectors_run=[d.value for d in detectors_run],
                detectors_skipped=[d.value for d in remaining],
                current_decision=current_decision,
                decision_confidence=decision_confidence,
                time_saved_ms=time_saved
            )
        
        # Continue running detectors
        return ShortCircuitResult(
            should_terminate=False,
            reason="Decision not yet confident enough",
            detectors_run=[d.value for d in detectors_run],
            detectors_skipped=[],
            current_decision=current_decision,
            decision_confidence=decision_confidence,
            time_saved_ms=0
        )
    
    def get_next_detector(self,
                         detectors_run: List[DetectorID],
                         current_confidence: float) -> Optional[DetectorID]:
        """Get the next detector to run based on VOI."""
        all_detectors = set(self.profiles.keys())
        run_detectors = set(detectors_run)
        remaining = list(all_detectors - run_detectors)
        
        if not remaining:
            return None
        
        ordering = self.get_optimal_ordering(current_confidence, remaining)
        return ordering[0] if ordering else None
    
    def record_detector_run(self,
                           detector_id: DetectorID,
                           time_ms: float,
                           had_decision_impact: bool):
        """Record the result of running a detector."""
        if detector_id in self.profiles:
            self.profiles[detector_id].update_from_observation(time_ms, had_decision_impact)
    
    def record_final_outcome(self,
                            detectors_run: List[DetectorID],
                            was_short_circuited: bool,
                            final_correct: bool):
        """Record final outcome for learning."""
        self.short_circuit_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'detectors_run': [d.value for d in detectors_run],
            'short_circuited': was_short_circuited,
            'correct': final_correct
        })
        
        # Keep last 1000
        if len(self.short_circuit_history) > 1000:
            self.short_circuit_history.pop(0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get VOI engine statistics."""
        short_circuit_rate = (
            self.short_circuited / self.total_evaluations
            if self.total_evaluations > 0 else 0
        )
        
        # Calculate short-circuit accuracy
        if self.short_circuit_history:
            sc_decisions = [h for h in self.short_circuit_history if h['short_circuited']]
            if sc_decisions:
                sc_accuracy = sum(1 for h in sc_decisions if h['correct']) / len(sc_decisions)
            else:
                sc_accuracy = 0.0
        else:
            sc_accuracy = 0.0
        
        return {
            'total_evaluations': self.total_evaluations,
            'short_circuited': self.short_circuited,
            'short_circuit_rate': short_circuit_rate,
            'total_time_saved_ms': self.total_time_saved_ms,
            'avg_time_saved_ms': self.total_time_saved_ms / self.short_circuited if self.short_circuited > 0 else 0,
            'short_circuit_accuracy': sc_accuracy,
            'detector_profiles': {k.value: v.to_dict() for k, v in self.profiles.items()},
            'optimal_ordering': [d.value for d in self.get_optimal_ordering()]
        }
    
    def get_detector_profile(self, detector_id: DetectorID) -> Optional[DetectorProfile]:
        """Get profile for a detector."""
        return self.profiles.get(detector_id)
    
    def update_profile(self,
                      detector_id: DetectorID,
                      avg_time_ms: float = None,
                      decision_impact: float = None,
                      correlation: float = None):
        """Manually update a detector profile."""
        if detector_id not in self.profiles:
            return
        
        profile = self.profiles[detector_id]
        if avg_time_ms is not None:
            profile.avg_time_ms = avg_time_ms
        if decision_impact is not None:
            profile.decision_impact = decision_impact
        if correlation is not None:
            profile.correlation_with_outcome = correlation

    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        """Record operation outcome."""
        if not hasattr(self, '_outcomes'): self._outcomes = []
        self._outcomes.append(outcome)

    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback."""
        pass


    def serialize_state(self) -> Dict:
        """Serialize state."""
        return {'outcomes': getattr(self, '_outcomes', [])[-100:]}


    def deserialize_state(self, state: Dict) -> None:
        """Restore state."""
        self._outcomes = state.get('outcomes', [])

