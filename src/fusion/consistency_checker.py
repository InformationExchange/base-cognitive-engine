"""
BASE Cross-Pathway Consistency Checker
Migrated from Onyx Governance - Enhancement 6

Validates consistency across all governance signals and resolves conflicts.

Mathematical Formulation:
C = 1 - (1/|P|²) * Σ|signal_i - signal_j|

Where P is the set of all pathways/detectors.

Patent Claims: Enhancement-6
Proprietary IP - 100% owned by Invitas Inc.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class SignalConflict:
    """Detected conflict between two signals."""
    signal_a: str
    signal_b: str
    value_a: float
    value_b: float
    difference: float
    severity: str  # low, medium, high
    resolution: Optional[str] = None


@dataclass
class ConsistencyResult:
    """Consistency checking result."""
    consistent: bool
    consistency_score: float
    conflicts: List[SignalConflict]
    resolved_signals: Dict[str, float]
    confidence_adjustment: float  # How much to adjust fusion confidence
    warnings: List[str]
    timestamp: float
    processing_time_ms: float = 0.0


class CrossPathwayConsistencyChecker:
    """
    Cross-Pathway Consistency Checker for BASE.
    
    Validates that signals from different detectors/pathways are consistent.
    Detects and resolves conflicts to improve decision quality.
    
    Methods:
    - Pairwise consistency computation
    - Conflict detection
    - Weighted voting resolution
    - Confidence adjustment
    """
    
    # Expected signal relationships (for detecting anomalies)
    EXPECTED_CORRELATIONS = {
        # Signals that should generally agree
        ('grounding_score', 'factual_score'): 0.7,
        ('grounding_score', 'rag_quality_score'): 0.6,
        ('factual_score', 'fact_check_coverage'): 0.8,
        ('kg_alignment_score', 'factual_score'): 0.5,
    }
    
    def __init__(self,
                 consistency_threshold: float = 0.65,
                 conflict_threshold: float = 0.30,
                 resolution_method: str = "weighted_voting"):
        """
        Initialize Consistency Checker.
        
        Args:
            consistency_threshold: Minimum consistency to consider signals consistent
            conflict_threshold: Maximum difference before flagging as conflict
            resolution_method: Method for resolving conflicts
                ("weighted_voting", "majority", "average", "conservative")
        """
        self.consistency_threshold = consistency_threshold
        self.conflict_threshold = conflict_threshold
        self.resolution_method = resolution_method
        
        # Default signal weights (can be updated)
        self.signal_weights = {
            'grounding_score': 0.25,
            'factual_score': 0.20,
            'behavioral_score': 0.20,
            'temporal_score': 0.15,
            'rag_quality_score': 0.10,
            'kg_alignment_score': 0.05,
            'fact_check_coverage': 0.05,
        }
    
    def check(self,
              signals: Dict[str, float],
              weights: Dict[str, float] = None) -> ConsistencyResult:
        """
        Check consistency across signals.
        
        Args:
            signals: Dictionary of signal_name -> signal_value
            weights: Optional custom weights for signals
        
        Returns:
            ConsistencyResult with consistency score and conflicts
        """
        start_time = time.time()
        
        if len(signals) < 2:
            return ConsistencyResult(
                consistent=True,
                consistency_score=1.0,
                conflicts=[],
                resolved_signals=signals,
                confidence_adjustment=0.0,
                warnings=[],
                timestamp=time.time(),
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        weights = weights or self.signal_weights
        warnings = []
        
        # Normalize signals to [0, 1]
        normalized_signals = self._normalize_signals(signals)
        
        # Compute pairwise consistency matrix
        consistency_matrix = self._compute_pairwise_consistency(normalized_signals)
        
        # Compute overall consistency score
        consistency_score = self._compute_overall_consistency(consistency_matrix)
        
        # Detect conflicts
        conflicts = self._detect_conflicts(normalized_signals, consistency_matrix)
        
        # Check for anomalous correlations
        anomalies = self._check_expected_correlations(normalized_signals)
        for anomaly in anomalies:
            warnings.append(anomaly)
        
        # Resolve conflicts if any
        if conflicts:
            resolved_signals = self._resolve_conflicts(
                normalized_signals, conflicts, weights
            )
            warnings.append(f"Resolved {len(conflicts)} conflicts using {self.resolution_method}")
        else:
            resolved_signals = normalized_signals
        
        # Determine consistency
        consistent = consistency_score >= self.consistency_threshold and len(conflicts) == 0
        
        # Calculate confidence adjustment
        confidence_adjustment = self._compute_confidence_adjustment(
            consistency_score, len(conflicts)
        )
        
        if not consistent:
            warnings.append(f"Signals inconsistent (score: {consistency_score:.2f})")
        
        processing_time = (time.time() - start_time) * 1000
        
        return ConsistencyResult(
            consistent=consistent,
            consistency_score=consistency_score,
            conflicts=conflicts,
            resolved_signals=resolved_signals,
            confidence_adjustment=confidence_adjustment,
            warnings=warnings,
            timestamp=time.time(),
            processing_time_ms=processing_time
        )
    
    def _normalize_signals(self, signals: Dict[str, float]) -> Dict[str, float]:
        """Normalize signals to [0, 1] range."""
        normalized = {}
        for name, value in signals.items():
            # Handle percentage values
            if value > 1.0:
                value = value / 100.0
            # Clamp to [0, 1]
            normalized[name] = max(0.0, min(1.0, value))
        return normalized
    
    def _compute_pairwise_consistency(self, 
                                       signals: Dict[str, float]) -> Dict[Tuple[str, str], float]:
        """
        Compute pairwise consistency between all signal pairs.
        
        C_ij = 1 - |signal_i - signal_j|
        """
        signal_names = list(signals.keys())
        consistency_matrix = {}
        
        for i, name_i in enumerate(signal_names):
            for j, name_j in enumerate(signal_names):
                if i < j:  # Upper triangle only
                    diff = abs(signals[name_i] - signals[name_j])
                    consistency = 1.0 - diff
                    consistency_matrix[(name_i, name_j)] = consistency
        
        return consistency_matrix
    
    def _compute_overall_consistency(self,
                                      consistency_matrix: Dict[Tuple[str, str], float]) -> float:
        """Compute overall consistency score (mean of pairwise)."""
        if not consistency_matrix:
            return 1.0
        return sum(consistency_matrix.values()) / len(consistency_matrix)
    
    def _detect_conflicts(self,
                          signals: Dict[str, float],
                          consistency_matrix: Dict[Tuple[str, str], float]) -> List[SignalConflict]:
        """Detect conflicts where signals significantly disagree."""
        conflicts = []
        
        for (name_a, name_b), consistency in consistency_matrix.items():
            difference = 1.0 - consistency
            
            if difference > self.conflict_threshold:
                severity = "high" if difference > 0.5 else "medium" if difference > 0.35 else "low"
                
                conflict = SignalConflict(
                    signal_a=name_a,
                    signal_b=name_b,
                    value_a=signals[name_a],
                    value_b=signals[name_b],
                    difference=difference,
                    severity=severity
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _check_expected_correlations(self, 
                                      signals: Dict[str, float]) -> List[str]:
        """Check if expected signal correlations hold."""
        anomalies = []
        
        for (sig_a, sig_b), expected_corr in self.EXPECTED_CORRELATIONS.items():
            if sig_a in signals and sig_b in signals:
                # Check if they're correlated as expected
                diff = abs(signals[sig_a] - signals[sig_b])
                
                # If both are high but one is much lower than expected
                if signals[sig_a] > 0.7 and signals[sig_b] < 0.4:
                    anomalies.append(
                        f"Anomaly: {sig_a} ({signals[sig_a]:.2f}) and {sig_b} ({signals[sig_b]:.2f}) "
                        f"unexpectedly divergent"
                    )
                elif signals[sig_b] > 0.7 and signals[sig_a] < 0.4:
                    anomalies.append(
                        f"Anomaly: {sig_b} ({signals[sig_b]:.2f}) and {sig_a} ({signals[sig_a]:.2f}) "
                        f"unexpectedly divergent"
                    )
        
        return anomalies
    
    def _resolve_conflicts(self,
                           signals: Dict[str, float],
                           conflicts: List[SignalConflict],
                           weights: Dict[str, float]) -> Dict[str, float]:
        """
        Resolve conflicts using specified method.
        """
        if self.resolution_method == "weighted_voting":
            return self._weighted_voting_resolution(signals, weights)
        elif self.resolution_method == "majority":
            return self._majority_resolution(signals)
        elif self.resolution_method == "conservative":
            return self._conservative_resolution(signals)
        else:  # average
            return self._average_resolution(signals)
    
    def _weighted_voting_resolution(self,
                                     signals: Dict[str, float],
                                     weights: Dict[str, float]) -> Dict[str, float]:
        """Resolve using weighted voting - weight by signal importance."""
        resolved = {}
        
        # Compute weighted average
        total_weight = sum(weights.get(name, 0.1) for name in signals.keys())
        weighted_avg = sum(
            signals[name] * weights.get(name, 0.1)
            for name in signals.keys()
        ) / total_weight if total_weight > 0 else 0.5
        
        # Adjust signals toward weighted average
        for name, value in signals.items():
            weight = weights.get(name, 0.1)
            # Higher weight = keep original, lower weight = move toward average
            adjustment = (weighted_avg - value) * (1 - weight)
            resolved[name] = value + adjustment * 0.3  # Partial adjustment
        
        return resolved
    
    def _majority_resolution(self, signals: Dict[str, float]) -> Dict[str, float]:
        """Resolve using median value."""
        values = list(signals.values())
        median = float(np.median(values))
        
        # Move extreme values toward median
        resolved = {}
        for name, value in signals.items():
            if abs(value - median) > self.conflict_threshold:
                resolved[name] = (value + median) / 2  # Average with median
            else:
                resolved[name] = value
        
        return resolved
    
    def _conservative_resolution(self, signals: Dict[str, float]) -> Dict[str, float]:
        """Resolve conservatively - take minimum values."""
        # For governance, being conservative means lower scores (more cautious)
        min_val = min(signals.values())
        
        resolved = {}
        for name, value in signals.items():
            # Pull values toward minimum
            resolved[name] = (value + min_val) / 2
        
        return resolved
    
    def _average_resolution(self, signals: Dict[str, float]) -> Dict[str, float]:
        """Resolve using simple average."""
        avg = sum(signals.values()) / len(signals)
        
        resolved = {}
        for name, value in signals.items():
            # Move halfway toward average
            resolved[name] = (value + avg) / 2
        
        return resolved
    
    def _compute_confidence_adjustment(self,
                                        consistency_score: float,
                                        num_conflicts: int) -> float:
        """
        Compute how much to adjust fusion confidence.
        
        Positive = increase confidence, Negative = decrease confidence
        """
        # Base adjustment from consistency
        if consistency_score >= 0.9:
            base_adj = 0.1  # High consistency = slight boost
        elif consistency_score >= 0.7:
            base_adj = 0.0  # Normal
        elif consistency_score >= 0.5:
            base_adj = -0.1  # Low consistency = reduce
        else:
            base_adj = -0.2  # Very low = significant reduction
        
        # Additional penalty for conflicts
        conflict_penalty = num_conflicts * 0.05
        
        return base_adj - conflict_penalty

    # =========================================================================
    # Learning Interface
    # =========================================================================
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

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


# Self-test
if __name__ == "__main__":
    checker = CrossPathwayConsistencyChecker()
    
    # Test 1: Consistent signals
    signals_consistent = {
        'grounding_score': 0.85,
        'factual_score': 0.80,
        'behavioral_score': 0.10,  # Low is good for behavioral
        'temporal_score': 0.78,
    }
    
    result = checker.check(signals_consistent)
    print(f"\nTest 1 (Consistent):")
    print(f"  Consistent: {result.consistent}")
    print(f"  Consistency Score: {result.consistency_score:.3f}")
    print(f"  Conflicts: {len(result.conflicts)}")
    print(f"  Confidence Adj: {result.confidence_adjustment:+.2f}")
    
    # Test 2: Inconsistent signals
    signals_inconsistent = {
        'grounding_score': 0.90,
        'factual_score': 0.30,  # Conflict!
        'behavioral_score': 0.15,
        'temporal_score': 0.85,
    }
    
    result = checker.check(signals_inconsistent)
    print(f"\nTest 2 (Inconsistent):")
    print(f"  Consistent: {result.consistent}")
    print(f"  Consistency Score: {result.consistency_score:.3f}")
    print(f"  Conflicts: {len(result.conflicts)}")
    if result.conflicts:
        for c in result.conflicts:
            print(f"    - {c.signal_a} vs {c.signal_b}: diff={c.difference:.2f} ({c.severity})")
    print(f"  Warnings: {result.warnings}")
    print(f"  Confidence Adj: {result.confidence_adjustment:+.2f}")






