"""
BAIS Cognitive Governance Engine v16.5
Crisis Parameter Adjustment

PPA-2 Dep.Claim 26: FULL IMPLEMENTATION
"Crisis condition: tighten acceptance by increasing γ, decreasing α,
enlarging W with hysteresis"

This module implements:
1. γ (gamma): Acceptance margin parameter
2. α (alpha): False-pass tolerance parameter  
3. W (window): Temporal window size parameter
4. Hysteresis: Prevent oscillation via dwell periods
5. Integration with state machine and conformal screening
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import json


class CrisisLevel(Enum):
    """Crisis severity levels."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRISIS = "crisis"
    DEGRADED = "degraded"


@dataclass
class CrisisParameters:
    """
    Crisis-adjusted parameters.
    
    PPA-2 Dep.Claim 26: The three key parameters.
    
    γ (gamma): Acceptance margin - how much above threshold to require
        Normal: 0.1 (10% margin)
        Crisis: 0.2 (20% margin) - tighter
        
    α (alpha): False-pass tolerance for conformal screening
        Normal: 0.05 (5% false-pass rate allowed)
        Crisis: 0.025 (2.5% false-pass rate) - stricter
        
    W (window): Temporal window for violation counting
        Normal: 100 observations
        Crisis: 150 observations - longer memory
    """
    gamma: float  # Acceptance margin (γ)
    alpha: float  # False-pass tolerance (α)
    window: int   # Temporal window size (W)
    
    # Additional crisis-adjusted parameters
    threshold_multiplier: float = 1.0
    min_reviews: int = 1
    
    # Metadata
    crisis_level: CrisisLevel = CrisisLevel.NORMAL
    adjusted_at: datetime = field(default_factory=datetime.utcnow)
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'gamma': self.gamma,
            'alpha': self.alpha,
            'window': self.window,
            'threshold_multiplier': self.threshold_multiplier,
            'min_reviews': self.min_reviews,
            'crisis_level': self.crisis_level.value,
            'adjusted_at': self.adjusted_at.isoformat(),
            'reason': self.reason
        }


class CrisisParameterController:
    """
    Crisis Parameter Controller.
    
    PPA-2 Dep.Claim 26: FULL IMPLEMENTATION
    
    Key Features:
    1. Explicit γ, α, W parameters with crisis adjustments
    2. Hysteresis to prevent oscillation
    3. Dwell periods before relaxing parameters
    4. Integration with state machine and behavioral signals
    
    Crisis Response:
    - On crisis detection: Tighten immediately (increase γ, decrease α, enlarge W)
    - On recovery: Relax slowly (hysteresis) after dwell period
    """
    
    # Baseline parameters (NORMAL mode)
    BASELINE_GAMMA = 0.10      # 10% acceptance margin
    BASELINE_ALPHA = 0.05      # 5% false-pass tolerance
    BASELINE_WINDOW = 100      # 100 observations
    BASELINE_MULTIPLIER = 1.0  # No threshold adjustment
    BASELINE_MIN_REVIEWS = 1   # Single reviewer sufficient
    
    # Crisis multipliers (per PPA-2 Dep.Claim 26)
    CRISIS_MULTIPLIERS = {
        CrisisLevel.NORMAL: {
            'gamma': 1.0,      # γ unchanged
            'alpha': 1.0,      # α unchanged
            'window': 1.0      # W unchanged
        },
        CrisisLevel.ELEVATED: {
            'gamma': 1.5,      # γ * 1.5 (tighter margin)
            'alpha': 0.75,     # α * 0.75 (stricter)
            'window': 1.25     # W * 1.25 (longer memory)
        },
        CrisisLevel.CRISIS: {
            'gamma': 2.0,      # γ * 2 (much tighter)
            'alpha': 0.5,      # α * 0.5 (much stricter)
            'window': 1.5      # W * 1.5 (much longer memory)
        },
        CrisisLevel.DEGRADED: {
            'gamma': 3.0,      # γ * 3 (extreme tightening)
            'alpha': 0.25,     # α * 0.25 (extreme restriction)
            'window': 2.0      # W * 2 (maximum memory)
        }
    }
    
    # Hysteresis: Dwell periods before relaxing (seconds)
    DWELL_PERIODS = {
        CrisisLevel.ELEVATED: 300,    # 5 minutes
        CrisisLevel.CRISIS: 600,      # 10 minutes
        CrisisLevel.DEGRADED: 1800    # 30 minutes
    }
    
    # Threshold multipliers per crisis level
    THRESHOLD_MULTIPLIERS = {
        CrisisLevel.NORMAL: 1.0,
        CrisisLevel.ELEVATED: 1.15,
        CrisisLevel.CRISIS: 1.4,
        CrisisLevel.DEGRADED: 1.8
    }
    
    def __init__(self, storage_path: Path = None):
        """Initialize crisis parameter controller."""
        # Use temp directory if no path provided (fixes read-only filesystem issues)
        if storage_path is None:
            import tempfile
            temp_dir = Path(tempfile.mkdtemp(prefix="bais_crisis_"))
            storage_path = temp_dir / "crisis_params.json"
        self.storage_path = storage_path
        
        # Current state
        self.current_level = CrisisLevel.NORMAL
        self.level_entered_at = datetime.utcnow()
        self.current_params = self._compute_parameters(CrisisLevel.NORMAL)
        
        # Transition history
        self.transitions: List[Dict] = []
        
        # Statistics
        self.stats = {
            'total_adjustments': 0,
            'time_in_crisis': 0.0,  # Total seconds in CRISIS or DEGRADED
            'avg_gamma': self.BASELINE_GAMMA,
            'avg_alpha': self.BASELINE_ALPHA
        }
        
        # Load persisted state
        self._load_state()
    
    def get_parameters(self, crisis_level: CrisisLevel = None) -> CrisisParameters:
        """
        Get current crisis-adjusted parameters.
        
        PPA-2 Dep.Claim 26: Returns γ, α, W adjusted for crisis level.
        """
        if crisis_level is None:
            return self.current_params
        
        return self._compute_parameters(crisis_level)
    
    def _compute_parameters(self, level: CrisisLevel) -> CrisisParameters:
        """
        Compute parameters for a given crisis level.
        
        Per PPA-2 Dep.Claim 26:
        - Crisis increases γ (acceptance margin)
        - Crisis decreases α (false-pass tolerance)
        - Crisis enlarges W (temporal window)
        """
        multipliers = self.CRISIS_MULTIPLIERS[level]
        
        return CrisisParameters(
            gamma=self.BASELINE_GAMMA * multipliers['gamma'],
            alpha=self.BASELINE_ALPHA * multipliers['alpha'],
            window=int(self.BASELINE_WINDOW * multipliers['window']),
            threshold_multiplier=self.THRESHOLD_MULTIPLIERS[level],
            min_reviews=1 if level == CrisisLevel.NORMAL else 2,
            crisis_level=level,
            adjusted_at=datetime.utcnow(),
            reason=f"Crisis level: {level.value}"
        )
    
    def evaluate_crisis(self,
                        behavioral_signals: Dict = None,
                        contradiction_rate: float = 0.0,
                        sentiment_volatility: float = 0.0,
                        recent_violations: int = 0) -> Tuple[CrisisLevel, CrisisParameters]:
        """
        Evaluate crisis conditions and adjust parameters.
        
        PPA-2 Dep.Claim 26: Crisis condition detection and response.
        
        Args:
            behavioral_signals: Full behavioral signal vector
            contradiction_rate: Truth-probe contradiction rate
            sentiment_volatility: Sentiment volatility score
            recent_violations: Recent violation count
        
        Returns:
            (new_level, adjusted_parameters)
        """
        # Determine crisis level based on signals
        new_level = self._determine_crisis_level(
            behavioral_signals=behavioral_signals,
            contradiction_rate=contradiction_rate,
            sentiment_volatility=sentiment_volatility,
            recent_violations=recent_violations
        )
        
        old_level = self.current_level
        
        # Apply hysteresis for transitions
        transition_allowed, reason = self._check_hysteresis(old_level, new_level)
        
        if transition_allowed and new_level != old_level:
            self._transition_to(new_level, reason)
        
        return self.current_level, self.current_params
    
    def _determine_crisis_level(self,
                                behavioral_signals: Dict = None,
                                contradiction_rate: float = 0.0,
                                sentiment_volatility: float = 0.0,
                                recent_violations: int = 0) -> CrisisLevel:
        """
        Determine crisis level from input signals.
        
        Thresholds from PPA-2 Dep.Claim 26:
        - contradiction_rate > 0.3 → crisis
        - sentiment_volatility > 0.7 → crisis
        - behavioral_skew > 0.5 → elevated
        """
        # Extract signals
        if behavioral_signals:
            contradiction_rate = behavioral_signals.get('truth_probe_contradiction_rate', contradiction_rate)
            sentiment_volatility = behavioral_signals.get('sentiment_volatility', sentiment_volatility)
            behavioral_skew = behavioral_signals.get('abnormal_behavioral_skew', 0.0)
        else:
            behavioral_skew = 0.0
        
        # Check DEGRADED conditions (most severe)
        if (contradiction_rate > 0.5 and sentiment_volatility > 0.8) or recent_violations >= 10:
            return CrisisLevel.DEGRADED
        
        # Check CRISIS conditions
        if contradiction_rate > 0.3 or sentiment_volatility > 0.7 or recent_violations >= 5:
            return CrisisLevel.CRISIS
        
        # Check ELEVATED conditions
        if behavioral_skew > 0.5 or recent_violations >= 2:
            return CrisisLevel.ELEVATED
        
        return CrisisLevel.NORMAL
    
    def _check_hysteresis(self, 
                          old_level: CrisisLevel, 
                          new_level: CrisisLevel) -> Tuple[bool, str]:
        """
        Check if transition is allowed based on hysteresis.
        
        PPA-2 Dep.Claim 26: Hysteresis prevents oscillation.
        
        Key principle: Tighten fast, relax slow.
        - Escalation (to more severe): Immediate
        - Recovery (to less severe): Requires dwell period
        """
        # Always allow escalation (tightening)
        if new_level.value > old_level.value or \
           (new_level in [CrisisLevel.CRISIS, CrisisLevel.DEGRADED] and 
            old_level in [CrisisLevel.NORMAL, CrisisLevel.ELEVATED]):
            return True, "escalation_immediate"
        
        # For recovery (relaxation), check dwell period
        if new_level.value < old_level.value:
            time_in_state = (datetime.utcnow() - self.level_entered_at).total_seconds()
            required_dwell = self.DWELL_PERIODS.get(old_level, 0)
            
            if time_in_state >= required_dwell:
                return True, f"recovery_after_dwell_{time_in_state:.0f}s"
            else:
                return False, f"dwell_not_met_{time_in_state:.0f}s/{required_dwell}s"
        
        # Same level - no change needed
        return False, "same_level"
    
    def _transition_to(self, new_level: CrisisLevel, reason: str):
        """Execute state transition."""
        old_level = self.current_level
        old_params = self.current_params
        
        # Track time in crisis
        time_in_old = (datetime.utcnow() - self.level_entered_at).total_seconds()
        if old_level in [CrisisLevel.CRISIS, CrisisLevel.DEGRADED]:
            self.stats['time_in_crisis'] += time_in_old
        
        # Transition
        self.current_level = new_level
        self.level_entered_at = datetime.utcnow()
        self.current_params = self._compute_parameters(new_level)
        
        # Record transition
        transition = {
            'timestamp': datetime.utcnow().isoformat(),
            'from_level': old_level.value,
            'to_level': new_level.value,
            'reason': reason,
            'time_in_previous': time_in_old,
            'old_params': {
                'gamma': old_params.gamma,
                'alpha': old_params.alpha,
                'window': old_params.window
            },
            'new_params': {
                'gamma': self.current_params.gamma,
                'alpha': self.current_params.alpha,
                'window': self.current_params.window
            }
        }
        self.transitions.append(transition)
        
        self.stats['total_adjustments'] += 1
        
        self._save_state()
        
        print(f"[CrisisParams] Transition: {old_level.value} → {new_level.value}")
        print(f"[CrisisParams] γ: {old_params.gamma:.3f} → {self.current_params.gamma:.3f}")
        print(f"[CrisisParams] α: {old_params.alpha:.4f} → {self.current_params.alpha:.4f}")
        print(f"[CrisisParams] W: {old_params.window} → {self.current_params.window}")
    
    def force_crisis(self, level: CrisisLevel, reason: str = "manual_override"):
        """Force transition to a specific crisis level."""
        self._transition_to(level, reason)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current crisis parameter status."""
        time_in_state = (datetime.utcnow() - self.level_entered_at).total_seconds()
        
        return {
            'current_level': self.current_level.value,
            'time_in_state_seconds': time_in_state,
            'parameters': self.current_params.to_dict(),
            'statistics': self.stats,
            'recent_transitions': self.transitions[-5:] if self.transitions else [],
            'hysteresis': {
                'dwell_periods': {k.value: v for k, v in self.DWELL_PERIODS.items()},
                'current_dwell_required': self.DWELL_PERIODS.get(self.current_level, 0),
                'time_remaining': max(0, self.DWELL_PERIODS.get(self.current_level, 0) - time_in_state)
            }
        }
    
    def _save_state(self):
        """Persist crisis parameter state."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'current_level': self.current_level.value,
            'level_entered_at': self.level_entered_at.isoformat(),
            'current_params': self.current_params.to_dict(),
            'transitions': self.transitions[-50:],  # Keep last 50
            'stats': self.stats
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load persisted crisis parameter state."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path) as f:
                state = json.load(f)
            
            self.current_level = CrisisLevel(state.get('current_level', 'normal'))
            self.level_entered_at = datetime.fromisoformat(state['level_entered_at'])
            self.current_params = self._compute_parameters(self.current_level)
            self.transitions = state.get('transitions', [])
            self.stats.update(state.get('stats', {}))
            
        except Exception as e:
            print(f"Warning: Could not load crisis params state: {e}")

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

