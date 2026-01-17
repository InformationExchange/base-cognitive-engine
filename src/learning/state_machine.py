"""
BAIS Cognitive Governance Engine v16.0
State Machine with Hysteresis

Per PPA 3, Invention 1:
- States: NORMAL → ELEVATED → CRISIS → DEGRADED
- Hysteresis: Different thresholds for entering vs exiting states
- Bayesian belief tracking over states
- Full persistence across restarts
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
from collections import deque
import json
from pathlib import Path
import math
import numpy as np


class OperationalState(Enum):
    """Operational states per PPA 3."""
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"  
    CRISIS = "CRISIS"
    DEGRADED = "DEGRADED"
    
    @property
    def severity(self) -> int:
        """Numerical severity for comparison."""
        return {
            OperationalState.NORMAL: 0,
            OperationalState.ELEVATED: 1,
            OperationalState.CRISIS: 2,
            OperationalState.DEGRADED: 3
        }[self]
    
    @classmethod
    def from_string(cls, s: str) -> 'OperationalState':
        return cls[s.upper()]


@dataclass
class Violation:
    """Record of a governance violation."""
    timestamp: datetime
    violation_type: str  # 'false_positive' or 'false_negative'
    domain: str
    accuracy: float
    threshold: float
    details: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'violation_type': self.violation_type,
            'domain': self.domain,
            'accuracy': self.accuracy,
            'threshold': self.threshold,
            'details': self.details
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'Violation':
        return cls(
            timestamp=datetime.fromisoformat(d['timestamp']),
            violation_type=d['violation_type'],
            domain=d['domain'],
            accuracy=d['accuracy'],
            threshold=d['threshold'],
            details=d.get('details', {})
        )


@dataclass
class StateTransition:
    """Record of a state transition."""
    from_state: OperationalState
    to_state: OperationalState
    timestamp: datetime
    violations_at_transition: int
    trigger: str  # 'escalation' or 'recovery'
    duration_in_previous_state: float  # seconds
    
    def to_dict(self) -> Dict:
        return {
            'from_state': self.from_state.value,
            'to_state': self.to_state.value,
            'timestamp': self.timestamp.isoformat(),
            'violations_at_transition': self.violations_at_transition,
            'trigger': self.trigger,
            'duration_in_previous_state': self.duration_in_previous_state
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'StateTransition':
        return cls(
            from_state=OperationalState.from_string(d['from_state']),
            to_state=OperationalState.from_string(d['to_state']),
            timestamp=datetime.fromisoformat(d['timestamp']),
            violations_at_transition=d['violations_at_transition'],
            trigger=d['trigger'],
            duration_in_previous_state=d['duration_in_previous_state']
        )


class StateMachineWithHysteresis:
    """
    Operational state machine with hysteresis control.
    
    Key Features:
    1. Asymmetric enter/exit thresholds (hysteresis)
    2. Time-windowed violation counting
    3. Bayesian belief tracking
    4. Full state persistence
    5. Learned transition thresholds
    
    Hysteresis prevents oscillation:
    - Entering ELEVATED requires 2 violations
    - Exiting ELEVATED requires 0 violations (must be clean)
    """
    
    # Default thresholds (can be learned)
    DEFAULT_ENTER_THRESHOLDS = {
        OperationalState.ELEVATED: 2,
        OperationalState.CRISIS: 5,
        OperationalState.DEGRADED: 10
    }
    
    DEFAULT_EXIT_THRESHOLDS = {
        OperationalState.ELEVATED: 0,
        OperationalState.CRISIS: 2,
        OperationalState.DEGRADED: 5
    }
    
    # Threshold multipliers per state
    STATE_MULTIPLIERS = {
        OperationalState.NORMAL: 1.0,
        OperationalState.ELEVATED: 1.15,
        OperationalState.CRISIS: 1.4,
        OperationalState.DEGRADED: 1.8
    }
    
    # Minimum time in state before allowing transition (debounce)
    MIN_TIME_IN_STATE = {
        OperationalState.NORMAL: 0,        # Can escalate immediately
        OperationalState.ELEVATED: 60,     # 1 minute before escalating further
        OperationalState.CRISIS: 120,      # 2 minutes
        OperationalState.DEGRADED: 300     # 5 minutes before recovery
    }
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def __init__(self, 
                 storage_path: Path = None,
                 violation_window_minutes: int = 30,
                 max_violations_stored: int = 1000):
        
        # Use temp file if no path provided (fixes read-only filesystem issues)
        if storage_path is None:
            import tempfile
            temp_dir = Path(tempfile.mkdtemp(prefix="bais_state_"))
            storage_path = temp_dir / "state.json"
        self.storage_path = storage_path
        self.violation_window_minutes = violation_window_minutes
        self.max_violations_stored = max_violations_stored
        
        # Current state
        self.state = OperationalState.NORMAL
        self.state_entered_at = datetime.utcnow()
        
        # Violations (circular buffer)
        self.violations: deque = deque(maxlen=max_violations_stored)
        
        # Transition history
        self.transitions: List[StateTransition] = []
        
        # Learned thresholds (can be adapted)
        self.enter_thresholds = dict(self.DEFAULT_ENTER_THRESHOLDS)
        self.exit_thresholds = dict(self.DEFAULT_EXIT_THRESHOLDS)
        
        # Bayesian belief over states
        self.state_beliefs = {
            OperationalState.NORMAL: 0.85,
            OperationalState.ELEVATED: 0.10,
            OperationalState.CRISIS: 0.04,
            OperationalState.DEGRADED: 0.01
        }
        
        # Statistics
        self.total_violations = 0
        self.total_successes = 0
        self.violations_by_type: Dict[str, int] = {'false_positive': 0, 'false_negative': 0}
        self.violations_by_domain: Dict[str, int] = {}
        
        # Load persisted state
        self._load_state()
    
    @property
    def current(self) -> OperationalState:
        """Alias for state - standard interface compatibility."""
        return self.state
    
    @property
    def current_state(self) -> OperationalState:
        """Alias for state - audit interface compatibility."""
        return self.state
    
    def record_outcome(self, 
                       domain: str,
                       accuracy: float,
                       threshold: float,
                       was_accepted: bool,
                       was_correct: bool,
                       details: Dict = None) -> Dict:
        """
        Record the outcome of a governance decision.
        
        Returns dict with:
        - state_changed: bool
        - new_state: str
        - multiplier: float
        - violation_recorded: bool
        """
        is_violation = (was_accepted and not was_correct) or (not was_accepted and was_correct)
        
        if is_violation:
            violation = Violation(
                timestamp=datetime.utcnow(),
                violation_type='false_positive' if was_accepted else 'false_negative',
                domain=domain,
                accuracy=accuracy,
                threshold=threshold,
                details=details or {}
            )
            self.violations.append(violation)
            self.total_violations += 1
            self.violations_by_type[violation.violation_type] += 1
            self.violations_by_domain[domain] = self.violations_by_domain.get(domain, 0) + 1
            
            # Update Bayesian beliefs
            self._update_beliefs_on_violation()
            
            # Check for state escalation
            old_state = self.state
            self._check_escalation()
            state_changed = old_state != self.state
        else:
            self.total_successes += 1
            
            # Update Bayesian beliefs
            self._update_beliefs_on_success()
            
            # Check for state recovery
            old_state = self.state
            self._check_recovery()
            state_changed = old_state != self.state
        
        # Persist state
        self._save_state()
        
        return {
            'state_changed': state_changed,
            'new_state': self.state.value,
            'multiplier': self.get_multiplier(),
            'violation_recorded': is_violation,
            'recent_violations': self.count_recent_violations(),
            'beliefs': dict(self.state_beliefs)
        }
    
    def count_recent_violations(self, window_minutes: int = None) -> int:
        """Count violations in recent time window."""
        window = window_minutes or self.violation_window_minutes
        cutoff = datetime.utcnow() - timedelta(minutes=window)
        return sum(1 for v in self.violations if v.timestamp > cutoff)
    
    def count_recent_by_type(self, violation_type: str, window_minutes: int = None) -> int:
        """Count violations of specific type in recent window."""
        window = window_minutes or self.violation_window_minutes
        cutoff = datetime.utcnow() - timedelta(minutes=window)
        return sum(1 for v in self.violations 
                   if v.timestamp > cutoff and v.violation_type == violation_type)
    
    def get_multiplier(self) -> float:
        """Get threshold multiplier for current state."""
        return self.STATE_MULTIPLIERS[self.state]
    
    def get_effective_threshold(self, base_threshold: float) -> float:
        """Apply state multiplier to base threshold."""
        return base_threshold * self.get_multiplier()
    
    def _check_escalation(self):
        """Check if state should escalate (get more severe)."""
        recent = self.count_recent_violations()
        time_in_state = (datetime.utcnow() - self.state_entered_at).total_seconds()
        
        # Must be in state for minimum time before further escalation
        if time_in_state < self.MIN_TIME_IN_STATE.get(self.state, 0):
            return
        
        old_state = self.state
        
        if self.state == OperationalState.NORMAL:
            if recent >= self.enter_thresholds[OperationalState.ELEVATED]:
                self.state = OperationalState.ELEVATED
        
        elif self.state == OperationalState.ELEVATED:
            if recent >= self.enter_thresholds[OperationalState.CRISIS]:
                self.state = OperationalState.CRISIS
        
        elif self.state == OperationalState.CRISIS:
            if recent >= self.enter_thresholds[OperationalState.DEGRADED]:
                self.state = OperationalState.DEGRADED
        
        if self.state != old_state:
            self._record_transition(old_state, self.state, recent, 'escalation')
    
    def _check_recovery(self):
        """Check if state should recover (get less severe)."""
        recent = self.count_recent_violations()
        time_in_state = (datetime.utcnow() - self.state_entered_at).total_seconds()
        
        # Must be in state for minimum time before recovery
        if time_in_state < self.MIN_TIME_IN_STATE.get(self.state, 0):
            return
        
        old_state = self.state
        
        # Exit thresholds are LOWER than enter thresholds (hysteresis)
        if self.state == OperationalState.DEGRADED:
            if recent <= self.exit_thresholds[OperationalState.DEGRADED]:
                self.state = OperationalState.CRISIS
        
        elif self.state == OperationalState.CRISIS:
            if recent <= self.exit_thresholds[OperationalState.CRISIS]:
                self.state = OperationalState.ELEVATED
        
        elif self.state == OperationalState.ELEVATED:
            if recent <= self.exit_thresholds[OperationalState.ELEVATED]:
                self.state = OperationalState.NORMAL
        
        if self.state != old_state:
            self._record_transition(old_state, self.state, recent, 'recovery')
    
    def _record_transition(self, from_state: OperationalState, 
                          to_state: OperationalState,
                          violations: int, trigger: str):
        """Record a state transition."""
        duration = (datetime.utcnow() - self.state_entered_at).total_seconds()
        
        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            timestamp=datetime.utcnow(),
            violations_at_transition=violations,
            trigger=trigger,
            duration_in_previous_state=duration
        )
        
        self.transitions.append(transition)
        self.state_entered_at = datetime.utcnow()
        
        # Keep only recent transitions
        if len(self.transitions) > 100:
            self.transitions = self.transitions[-100:]
    
    def _update_beliefs_on_violation(self):
        """Update Bayesian beliefs when violation occurs."""
        # Violations increase belief in more severe states
        self.state_beliefs[OperationalState.NORMAL] *= 0.9
        self.state_beliefs[OperationalState.ELEVATED] *= 1.05
        self.state_beliefs[OperationalState.CRISIS] *= 1.1
        self.state_beliefs[OperationalState.DEGRADED] *= 1.15
        
        self._normalize_beliefs()
    
    def _update_beliefs_on_success(self):
        """Update Bayesian beliefs when success occurs."""
        # Successes increase belief in less severe states
        self.state_beliefs[OperationalState.NORMAL] *= 1.02
        self.state_beliefs[OperationalState.ELEVATED] *= 0.99
        self.state_beliefs[OperationalState.CRISIS] *= 0.98
        self.state_beliefs[OperationalState.DEGRADED] *= 0.97
        
        self._normalize_beliefs()
    
    def _normalize_beliefs(self):
        """Normalize beliefs to sum to 1."""
        total = sum(self.state_beliefs.values())
        for state in self.state_beliefs:
            self.state_beliefs[state] /= total
    
    def get_most_likely_state(self) -> OperationalState:
        """Get state with highest belief probability."""
        return max(self.state_beliefs, key=self.state_beliefs.get)
    
    def adapt_thresholds(self, feedback: Dict):
        """
        Adapt enter/exit thresholds based on feedback.
        
        feedback:
        - too_sensitive: bool - Too many false escalations
        - too_slow: bool - Missed real problems
        """
        if feedback.get('too_sensitive'):
            # Increase enter thresholds
            for state in self.enter_thresholds:
                self.enter_thresholds[state] = min(
                    self.enter_thresholds[state] + 1,
                    self.DEFAULT_ENTER_THRESHOLDS[state] * 2  # Max 2x default
                )
        
        if feedback.get('too_slow'):
            # Decrease enter thresholds
            for state in self.enter_thresholds:
                self.enter_thresholds[state] = max(
                    self.enter_thresholds[state] - 1,
                    max(1, self.DEFAULT_ENTER_THRESHOLDS[state] // 2)  # Min half default
                )
        
        self._save_state()
    
    def get_status(self) -> Dict:
        """Get comprehensive state machine status."""
        return {
            'current_state': self.state.value,
            'multiplier': self.get_multiplier(),
            'time_in_state_seconds': (datetime.utcnow() - self.state_entered_at).total_seconds(),
            'recent_violations': self.count_recent_violations(),
            'recent_false_positives': self.count_recent_by_type('false_positive'),
            'recent_false_negatives': self.count_recent_by_type('false_negative'),
            'beliefs': {s.value: p for s, p in self.state_beliefs.items()},
            'most_likely_state': self.get_most_likely_state().value,
            'thresholds': {
                'enter': {s.value: t for s, t in self.enter_thresholds.items()},
                'exit': {s.value: t for s, t in self.exit_thresholds.items()}
            },
            'hysteresis_gaps': {
                s.value: self.enter_thresholds[s] - self.exit_thresholds[s]
                for s in [OperationalState.ELEVATED, OperationalState.CRISIS, OperationalState.DEGRADED]
            },
            'statistics': {
                'total_violations': self.total_violations,
                'total_successes': self.total_successes,
                'violation_rate': self.total_violations / max(1, self.total_violations + self.total_successes),
                'violations_by_type': dict(self.violations_by_type),
                'violations_by_domain': dict(self.violations_by_domain)
            },
            'recent_transitions': [t.to_dict() for t in self.transitions[-5:]]
        }
    
    def get_health_assessment(self) -> Dict:
        """Get health assessment based on current state and history."""
        recent = self.count_recent_violations()
        total_ops = self.total_violations + self.total_successes
        
        if self.state == OperationalState.NORMAL and recent == 0:
            health = 'healthy'
            score = 100
        elif self.state == OperationalState.NORMAL:
            health = 'good'
            score = 90 - recent * 5
        elif self.state == OperationalState.ELEVATED:
            health = 'warning'
            score = 70 - recent * 3
        elif self.state == OperationalState.CRISIS:
            health = 'critical'
            score = 40 - recent * 2
        else:
            health = 'degraded'
            score = 20 - recent
        
        return {
            'health': health,
            'score': max(0, min(100, score)),
            'recommendation': self._get_recommendation()
        }
    
    def _get_recommendation(self) -> str:
        """Get operational recommendation based on state."""
        if self.state == OperationalState.NORMAL:
            return "System operating normally."
        elif self.state == OperationalState.ELEVATED:
            return "Elevated error rate. Consider reviewing recent decisions."
        elif self.state == OperationalState.CRISIS:
            return "Crisis mode. Human review recommended for critical decisions."
        else:
            return "System degraded. Reduce throughput and enable manual review."
    
    def _save_state(self):
        """Persist state to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'state': self.state.value,
            'state_entered_at': self.state_entered_at.isoformat(),
            'violations': [v.to_dict() for v in list(self.violations)[-500:]],
            'transitions': [t.to_dict() for t in self.transitions[-100:]],
            'enter_thresholds': {s.value: t for s, t in self.enter_thresholds.items()},
            'exit_thresholds': {s.value: t for s, t in self.exit_thresholds.items()},
            'beliefs': {s.value: p for s, p in self.state_beliefs.items()},
            'statistics': {
                'total_violations': self.total_violations,
                'total_successes': self.total_successes,
                'violations_by_type': self.violations_by_type,
                'violations_by_domain': self.violations_by_domain
            }
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load state from disk."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path) as f:
                state = json.load(f)
            
            self.state = OperationalState.from_string(state['state'])
            self.state_entered_at = datetime.fromisoformat(state['state_entered_at'])
            
            self.violations = deque(
                [Violation.from_dict(v) for v in state.get('violations', [])],
                maxlen=self.max_violations_stored
            )
            
            self.transitions = [
                StateTransition.from_dict(t) for t in state.get('transitions', [])
            ]
            
            self.enter_thresholds = {
                OperationalState.from_string(s): t 
                for s, t in state.get('enter_thresholds', {}).items()
            }
            
            self.exit_thresholds = {
                OperationalState.from_string(s): t 
                for s, t in state.get('exit_thresholds', {}).items()
            }
            
            self.state_beliefs = {
                OperationalState.from_string(s): p 
                for s, p in state.get('beliefs', {}).items()
            }
            
            stats = state.get('statistics', {})
            self.total_violations = stats.get('total_violations', 0)
            self.total_successes = stats.get('total_successes', 0)
            self.violations_by_type = stats.get('violations_by_type', {'false_positive': 0, 'false_negative': 0})
            self.violations_by_domain = stats.get('violations_by_domain', {})
            
        except Exception as e:
            print(f"Warning: Could not load state machine state: {e}")

    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback."""
        pass


    def get_statistics(self) -> Dict:
        """Return statistics."""
        return {'outcomes': len(getattr(self, '_outcomes', []))}


    def serialize_state(self) -> Dict:
        """Serialize state."""
        return {'outcomes': getattr(self, '_outcomes', [])[-100:]}


    def deserialize_state(self, state: Dict) -> None:
        """Restore state."""
        self._outcomes = state.get('outcomes', [])

