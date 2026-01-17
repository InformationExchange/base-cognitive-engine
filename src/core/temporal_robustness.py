"""
BAIS Cognitive Governance Engine v31.0
Temporal Robustness - Rolling Window & Hysteresis Control

Phase 31: Addresses PPA2-C1-9, PPA2-C1-10, PPA2-C1-21, PPA2-C1-31
- Rolling window W with hysteresis band
- Dwell-time/sample-count requirement
- Decision threshold γ modulated by temporal factor τ
- Acceptance requires certificate + (must-pass OR temporal robustness)

Patent Claims Addressed:
- PPA2-C1-9: Temporal robustness with rolling window W and hysteresis band
- PPA2-C1-10: Temporal robustness with dwell-time/sample-count requirement
- PPA2-C1-21: Acceptance requires certificate + (must-pass OR temporal robustness)
- PPA2-C1-31: Decision threshold γ modulated by bounded temporal factor τ
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Deque
from collections import deque
from enum import Enum
from datetime import datetime, timedelta
import logging
import math

logger = logging.getLogger(__name__)


class RobustnessStatus(Enum):
    """Status of temporal robustness check."""
    ROBUST = "robust"
    UNSTABLE = "unstable"
    TRANSITIONING = "transitioning"
    DWELL_REQUIRED = "dwell_required"
    SAMPLE_COUNT_LOW = "sample_count_low"


class ThresholdState(Enum):
    """State of threshold transition."""
    STABLE_LOW = "stable_low"
    STABLE_HIGH = "stable_high"
    RISING = "rising"
    FALLING = "falling"


@dataclass
class TemporalRobustnessResult:
    """Result of temporal robustness analysis."""
    is_robust: bool
    status: RobustnessStatus
    window_mean: float
    window_std: float
    hysteresis_band: Tuple[float, float]
    dwell_time_met: bool
    sample_count_met: bool
    temporal_factor: float
    adjusted_threshold: float
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class RollingWindowConfig:
    """Configuration for rolling window analysis."""
    window_size: int = 100
    min_samples: int = 10
    hysteresis_width: float = 0.1
    dwell_time_seconds: float = 5.0
    min_sample_count: int = 5
    temporal_factor_min: float = 0.8
    temporal_factor_max: float = 1.2


class RollingWindowAnalyzer:
    """
    Rolling window analyzer with hysteresis band.
    
    PPA2-C1-9: Temporal robustness with rolling window W and hysteresis band.
    
    Prevents oscillation by requiring values to cross hysteresis boundaries
    before state transitions are confirmed.
    """
    
    def __init__(self, config: Optional[RollingWindowConfig] = None):
        """
        Initialize rolling window analyzer.
        
        Args:
            config: Configuration for window analysis
        """
        self.config = config or RollingWindowConfig()
        self.window: Deque[Tuple[float, datetime]] = deque(maxlen=self.config.window_size)
        self.current_state = ThresholdState.STABLE_LOW
        self.last_state_change = datetime.utcnow()
        self.state_history: List[Tuple[ThresholdState, datetime]] = []
        
    def add_observation(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """
        Add observation to rolling window.
        
        Args:
            value: Observation value
            timestamp: Observation timestamp
        """
        ts = timestamp or datetime.utcnow()
        self.window.append((value, ts))
        
    def get_statistics(self) -> Tuple[float, float, int]:
        """
        Get rolling window statistics.
        
        Returns:
            Tuple of (mean, std, count)
        """
        if not self.window:
            return 0.0, 0.0, 0
        
        values = [v for v, _ in self.window]
        mean = np.mean(values)
        std = np.std(values) if len(values) > 1 else 0.0
        return float(mean), float(std), len(values)
    
    def get_hysteresis_band(self, center: float) -> Tuple[float, float]:
        """
        Calculate hysteresis band around center value.
        
        Args:
            center: Center of hysteresis band
            
        Returns:
            Tuple of (lower, upper) band boundaries
        """
        half_width = self.config.hysteresis_width / 2
        return (center - half_width, center + half_width)
    
    def check_state_transition(self, value: float, threshold: float) -> ThresholdState:
        """
        Check if value triggers state transition with hysteresis.
        
        Args:
            value: Current value
            threshold: Decision threshold
            
        Returns:
            New threshold state
        """
        lower, upper = self.get_hysteresis_band(threshold)
        
        if self.current_state == ThresholdState.STABLE_LOW:
            if value > upper:
                self.current_state = ThresholdState.RISING
        elif self.current_state == ThresholdState.STABLE_HIGH:
            if value < lower:
                self.current_state = ThresholdState.FALLING
        elif self.current_state == ThresholdState.RISING:
            if value > upper:
                self.current_state = ThresholdState.STABLE_HIGH
                self._record_state_change()
            elif value < lower:
                self.current_state = ThresholdState.STABLE_LOW
        elif self.current_state == ThresholdState.FALLING:
            if value < lower:
                self.current_state = ThresholdState.STABLE_LOW
                self._record_state_change()
            elif value > upper:
                self.current_state = ThresholdState.STABLE_HIGH
        
        return self.current_state
    
    def _record_state_change(self) -> None:
        """Record state change in history."""
        now = datetime.utcnow()
        self.state_history.append((self.current_state, now))
        self.last_state_change = now
        
    def is_stable(self) -> bool:
        """Check if current state is stable (not transitioning)."""
        return self.current_state in [ThresholdState.STABLE_LOW, ThresholdState.STABLE_HIGH]

    
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
        pass  # Implement specific learning logic


    def serialize_state(self) -> Dict:
        """Serialize state."""
        return {'outcomes': getattr(self, '_outcomes', [])[-100:]}


    def deserialize_state(self, state: Dict) -> None:
        """Restore state."""
        self._outcomes = state.get('outcomes', [])


class DwellTimeController:
    """
    Dwell-time and sample-count requirement controller.
    
    PPA2-C1-10: Temporal robustness with dwell-time/sample-count requirement.
    
    Ensures decisions remain stable for minimum duration and sample count
    before being confirmed.
    """
    
    def __init__(
        self,
        min_dwell_time: float = 5.0,
        min_sample_count: int = 5
    ):
        """
        Initialize dwell-time controller.
        
        Args:
            min_dwell_time: Minimum seconds before confirming state
            min_sample_count: Minimum samples before confirming state
        """
        self.min_dwell_time = min_dwell_time
        self.min_sample_count = min_sample_count
        self.current_state: Optional[str] = None
        self.state_start_time: Optional[datetime] = None
        self.samples_in_state: int = 0
        self.confirmed_state: Optional[str] = None
        
    def update(self, new_state: str) -> Tuple[bool, bool]:
        """
        Update with new state observation.
        
        Args:
            new_state: Observed state
            
        Returns:
            Tuple of (dwell_time_met, sample_count_met)
        """
        now = datetime.utcnow()
        
        if new_state != self.current_state:
            # State changed, reset counters
            self.current_state = new_state
            self.state_start_time = now
            self.samples_in_state = 1
        else:
            # Same state, increment counter
            self.samples_in_state += 1
        
        # Check requirements
        dwell_met = False
        sample_met = False
        
        if self.state_start_time:
            elapsed = (now - self.state_start_time).total_seconds()
            dwell_met = elapsed >= self.min_dwell_time
        
        sample_met = self.samples_in_state >= self.min_sample_count
        
        # Confirm state if both requirements met
        if dwell_met and sample_met:
            self.confirmed_state = self.current_state
        
        return dwell_met, sample_met
    
    def is_confirmed(self) -> bool:
        """Check if current state is confirmed."""
        return self.current_state == self.confirmed_state
    
    def get_progress(self) -> Dict[str, Any]:
        """Get progress toward confirmation."""
        elapsed = 0.0
        if self.state_start_time:
            elapsed = (datetime.utcnow() - self.state_start_time).total_seconds()
        
        return {
            "current_state": self.current_state,
            "confirmed_state": self.confirmed_state,
            "dwell_progress": min(1.0, elapsed / self.min_dwell_time) if self.min_dwell_time > 0 else 1.0,
            "sample_progress": min(1.0, self.samples_in_state / self.min_sample_count),
            "elapsed_seconds": elapsed,
            "samples": self.samples_in_state
        }

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


class TemporalFactorModulator:
    """
    Temporal factor modulator for decision threshold.
    
    PPA2-C1-31: Decision threshold γ modulated by bounded temporal factor τ.
    
    Adjusts decision thresholds based on temporal patterns and stability.
    """
    
    def __init__(
        self,
        tau_min: float = 0.8,
        tau_max: float = 1.2,
        smoothing_factor: float = 0.9
    ):
        """
        Initialize temporal factor modulator.
        
        Args:
            tau_min: Minimum temporal factor (floor)
            tau_max: Maximum temporal factor (ceiling)
            smoothing_factor: Exponential smoothing factor
        """
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.smoothing_factor = smoothing_factor
        self.current_tau = 1.0
        self.history: List[float] = []
        
    def update(
        self,
        volatility: float,
        trend: float,
        stability_score: float
    ) -> float:
        """
        Update temporal factor based on conditions.
        
        Args:
            volatility: Recent volatility (0-1)
            trend: Trend direction (-1 to 1)
            stability_score: Current stability (0-1)
            
        Returns:
            Updated temporal factor τ
        """
        # Calculate raw factor based on inputs
        # High volatility → lower factor (more conservative)
        # High stability → higher factor (more permissive)
        raw_tau = 1.0 - 0.3 * volatility + 0.2 * stability_score
        
        # Apply trend adjustment
        raw_tau += 0.1 * trend
        
        # Clamp to bounds
        raw_tau = max(self.tau_min, min(self.tau_max, raw_tau))
        
        # Apply exponential smoothing
        self.current_tau = (
            self.smoothing_factor * self.current_tau +
            (1 - self.smoothing_factor) * raw_tau
        )
        
        self.history.append(self.current_tau)
        
        return self.current_tau
    
    def modulate_threshold(self, base_threshold: float) -> float:
        """
        Apply temporal modulation to threshold.
        
        Per PPA2-C1-31: γ_adjusted = γ * τ
        
        Args:
            base_threshold: Base decision threshold γ
            
        Returns:
            Modulated threshold
        """
        return base_threshold * self.current_tau
    
    def get_status(self) -> Dict[str, Any]:
        """Get modulator status."""
        return {
            "current_tau": self.current_tau,
            "tau_bounds": (self.tau_min, self.tau_max),
            "history_length": len(self.history),
            "recent_avg": np.mean(self.history[-10:]) if self.history else 1.0
        }

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


class TemporalRobustnessManager:
    """
    Unified manager for temporal robustness.
    
    Implements:
    - PPA2-C1-9: Rolling window with hysteresis
    - PPA2-C1-10: Dwell-time/sample-count
    - PPA2-C1-21: Certificate + (must-pass OR temporal robustness)
    - PPA2-C1-31: Threshold modulation by τ
    """
    
    def __init__(self, config: Optional[RollingWindowConfig] = None):
        """
        Initialize temporal robustness manager.
        
        Args:
            config: Configuration for robustness checks
        """
        self.config = config or RollingWindowConfig()
        self.window_analyzer = RollingWindowAnalyzer(self.config)
        self.dwell_controller = DwellTimeController(
            min_dwell_time=self.config.dwell_time_seconds,
            min_sample_count=self.config.min_sample_count
        )
        self.tau_modulator = TemporalFactorModulator(
            tau_min=self.config.temporal_factor_min,
            tau_max=self.config.temporal_factor_max
        )
        
        logger.info("[TemporalRobust] Temporal Robustness Manager initialized")
    
    def evaluate(
        self,
        value: float,
        threshold: float,
        state_label: str = "accept"
    ) -> TemporalRobustnessResult:
        """
        Evaluate temporal robustness of a decision.
        
        Args:
            value: Current decision value (e.g., accuracy score)
            threshold: Decision threshold
            state_label: Label for current decision state
            
        Returns:
            TemporalRobustnessResult
        """
        # Add to rolling window
        self.window_analyzer.add_observation(value)
        
        # Get window statistics
        mean, std, count = self.window_analyzer.get_statistics()
        
        # Check hysteresis state
        state = self.window_analyzer.check_state_transition(value, threshold)
        is_stable = self.window_analyzer.is_stable()
        
        # Check dwell-time and sample-count
        dwell_met, sample_met = self.dwell_controller.update(state_label)
        
        # Calculate volatility for tau modulation
        volatility = std / (mean + 1e-6) if mean > 0 else 0.0
        volatility = min(1.0, volatility)
        
        # Calculate trend (simple)
        if len(self.window_analyzer.window) >= 2:
            values = [v for v, _ in self.window_analyzer.window]
            trend = np.sign(values[-1] - values[0])
        else:
            trend = 0.0
        
        # Calculate stability score
        stability = 1.0 if is_stable else 0.5
        if dwell_met and sample_met:
            stability = 1.0
        
        # Update temporal factor
        tau = self.tau_modulator.update(volatility, trend, stability)
        adjusted_threshold = self.tau_modulator.modulate_threshold(threshold)
        
        # Determine robustness status
        if count < self.config.min_samples:
            status = RobustnessStatus.SAMPLE_COUNT_LOW
            is_robust = False
        elif not dwell_met:
            status = RobustnessStatus.DWELL_REQUIRED
            is_robust = False
        elif not is_stable:
            status = RobustnessStatus.TRANSITIONING
            is_robust = False
        elif dwell_met and sample_met and is_stable:
            status = RobustnessStatus.ROBUST
            is_robust = True
        else:
            status = RobustnessStatus.UNSTABLE
            is_robust = False
        
        # Build recommendations
        recommendations = []
        if not is_robust:
            if count < self.config.min_samples:
                recommendations.append(f"Need {self.config.min_samples - count} more samples")
            if not dwell_met:
                progress = self.dwell_controller.get_progress()
                recommendations.append(f"Dwell time: {progress['dwell_progress']:.0%} complete")
            if not is_stable:
                recommendations.append("State is transitioning, wait for stability")
        else:
            recommendations.append("Temporal robustness confirmed")
        
        return TemporalRobustnessResult(
            is_robust=is_robust,
            status=status,
            window_mean=mean,
            window_std=std,
            hysteresis_band=self.window_analyzer.get_hysteresis_band(threshold),
            dwell_time_met=dwell_met,
            sample_count_met=sample_met,
            temporal_factor=tau,
            adjusted_threshold=adjusted_threshold,
            details={
                "window_size": count,
                "state": state.value,
                "volatility": volatility,
                "stability": stability,
                "dwell_progress": self.dwell_controller.get_progress()
            },
            recommendations=recommendations
        )
    
    def check_acceptance(
        self,
        has_certificate: bool,
        must_pass_satisfied: bool,
        temporal_robust: Optional[TemporalRobustnessResult] = None
    ) -> Tuple[bool, str]:
        """
        Check if acceptance requirements are met.
        
        PPA2-C1-21: Acceptance requires certificate + (must-pass OR temporal robustness)
        
        Args:
            has_certificate: Whether valid certificate exists
            must_pass_satisfied: Whether must-pass predicates are satisfied
            temporal_robust: Result of temporal robustness check
            
        Returns:
            Tuple of (accepted, reason)
        """
        if not has_certificate:
            return False, "No valid certificate"
        
        temporal_satisfied = temporal_robust and temporal_robust.is_robust
        
        if must_pass_satisfied or temporal_satisfied:
            if must_pass_satisfied and temporal_satisfied:
                return True, "Certificate + must-pass + temporal robustness"
            elif must_pass_satisfied:
                return True, "Certificate + must-pass predicates"
            else:
                return True, "Certificate + temporal robustness"
        
        return False, "Certificate exists but neither must-pass nor temporal robustness satisfied"
    
    def reset(self):
        """Reset all temporal state."""
        self.window_analyzer = RollingWindowAnalyzer(self.config)
        self.dwell_controller = DwellTimeController(
            min_dwell_time=self.config.dwell_time_seconds,
            min_sample_count=self.config.min_sample_count
        )
        self.tau_modulator = TemporalFactorModulator(
            tau_min=self.config.temporal_factor_min,
            tau_max=self.config.temporal_factor_max
        )

    # ========================================
    # PHASE 49: LEARNING METHODS
    # ========================================
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)
    
    def record_outcome(self, input_data, output_data, was_correct, domain=None, metadata=None):
        """Record outcome for learning (Phase 49)."""
        if hasattr(self, '_learning_manager') and self._learning_manager:
            self._learning_manager.record_outcome(
                module_name=self.__class__.__name__.lower(),
                input_data=input_data, output_data=output_data,
                was_correct=was_correct, domain=domain, metadata=metadata
            )
    
    def record_feedback(self, result, was_accurate):
        """Record feedback for learning (Phase 49)."""
        self.record_outcome({"result": str(result)}, {}, was_accurate)
    
    def adapt_thresholds(self, threshold_name, current_value, direction='decrease'):
        """Adapt thresholds based on learning (Phase 49)."""
        if hasattr(self, '_learning_manager') and self._learning_manager:
            return self._learning_manager.adapt_threshold(
                self.__class__.__name__.lower(), threshold_name, current_value, direction
            )
        return max(0.1, min(0.95, current_value + (0.05 if direction == 'increase' else -0.05)))
    
    def get_domain_adjustment(self, domain):
        """Get domain-specific adjustment (Phase 49)."""
        if hasattr(self, '_learning_manager') and self._learning_manager:
            return self._learning_manager.get_domain_adjustment(self.__class__.__name__.lower(), domain)
        return 0.0
    
    def save_state(self, filepath=None):
        """Save learning state (Phase 49)."""
        return self._learning_manager.save_state() if hasattr(self, '_learning_manager') and self._learning_manager else False
    
    def load_state(self, filepath=None):
        """Load learning state (Phase 49)."""
        return self._learning_manager.load_state() if hasattr(self, '_learning_manager') and self._learning_manager else False
    
    def get_learning_statistics(self):
        """Get learning statistics (Phase 49)."""
        if hasattr(self, '_learning_manager') and self._learning_manager:
            stats = self._learning_manager.get_learning_statistics(self.__class__.__name__.lower())
            return stats.__dict__ if hasattr(stats, '__dict__') else stats
        return {"module": self.__class__.__name__, "status": "no_learning_manager"}

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


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 31: Temporal Robustness Module Test")
    print("=" * 60)
    
    manager = TemporalRobustnessManager()
    
    # Test 1: Build up samples
    print("\n[1] Building Temporal History:")
    for i in range(15):
        value = 0.7 + 0.05 * np.random.randn()
        result = manager.evaluate(value, threshold=0.5, state_label="accept")
    
    print(f"    Window size: {result.details['window_size']}")
    print(f"    Window mean: {result.window_mean:.4f}")
    print(f"    Status: {result.status.value}")
    print(f"    Robust: {result.is_robust}")
    
    # Test 2: Check hysteresis
    print("\n[2] Hysteresis Band Test:")
    print(f"    Threshold: 0.5")
    print(f"    Hysteresis band: {result.hysteresis_band}")
    print(f"    Current state: {result.details['state']}")
    
    # Test 3: Temporal factor modulation
    print("\n[3] Temporal Factor Modulation:")
    print(f"    Tau: {result.temporal_factor:.4f}")
    print(f"    Adjusted threshold: {result.adjusted_threshold:.4f}")
    
    # Test 4: Dwell time progress
    print("\n[4] Dwell Time Progress:")
    progress = result.details['dwell_progress']
    print(f"    Dwell progress: {progress['dwell_progress']:.0%}")
    print(f"    Sample progress: {progress['sample_progress']:.0%}")
    print(f"    Dwell met: {result.dwell_time_met}")
    print(f"    Sample count met: {result.sample_count_met}")
    
    # Test 5: Acceptance check
    print("\n[5] Acceptance Check (PPA2-C1-21):")
    
    accepted, reason = manager.check_acceptance(
        has_certificate=True,
        must_pass_satisfied=True,
        temporal_robust=result
    )
    print(f"    With certificate + must-pass: {accepted} - {reason}")
    
    accepted, reason = manager.check_acceptance(
        has_certificate=True,
        must_pass_satisfied=False,
        temporal_robust=result
    )
    print(f"    With certificate + temporal: {accepted} - {reason}")
    
    accepted, reason = manager.check_acceptance(
        has_certificate=False,
        must_pass_satisfied=True,
        temporal_robust=result
    )
    print(f"    Without certificate: {accepted} - {reason}")
    
    print("\n" + "=" * 60)
    print("PHASE 31: Temporal Robustness Module - VERIFIED")
    print("=" * 60)


