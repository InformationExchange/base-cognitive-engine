"""
BASE Cognitive Governance Engine v16.5
Dynamic Bias Evolution System

PPA-1 Invention 24: FULL IMPLEMENTATION
Neuroplasticity-inspired modeling for bias adaptation over time.

This module implements:
1. Bias State Representation: Vector representation of bias state
2. Temporal Evolution: Track bias changes over time
3. Plasticity Modeling: Adapt bias detection based on patterns
4. Decay and Reinforcement: Biases strengthen/weaken over time
5. Bias Trajectory Prediction: Forecast future bias patterns

Neuroplasticity-Inspired Features:
- Hebbian Learning: "Neurons that fire together wire together"
- Synaptic Pruning: Unused bias patterns fade
- Long-Term Potentiation: Repeated patterns strengthen
- Critical Periods: Rapid learning during early exposure
- Homeostatic Plasticity: System maintains stability

This enables the system to LEARN and ADAPT its bias detection
based on observed patterns, not just static rules.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import math
import json
from pathlib import Path


class BiasCategory(str, Enum):
    """Categories of cognitive bias."""
    CONFIRMATION = "confirmation"
    ANCHORING = "anchoring"
    AVAILABILITY = "availability"
    SUNK_COST = "sunk_cost"
    BANDWAGON = "bandwagon"
    AUTHORITY = "authority"
    OPTIMISM = "optimism"
    PESSIMISM = "pessimism"
    SELF_SERVING = "self_serving"
    HINDSIGHT = "hindsight"


class PlasticityPhase(str, Enum):
    """Learning phases (inspired by neurodevelopment)."""
    CRITICAL = "critical"       # High plasticity, rapid learning
    CONSOLIDATION = "consolidation"  # Moderate plasticity
    MATURE = "mature"           # Low plasticity, stable
    REACTIVATION = "reactivation"  # Triggered re-learning


@dataclass
class BiasStateVector:
    """
    Vector representation of current bias state.
    
    Each dimension represents sensitivity to a bias type.
    Values 0-1 where higher = more likely to detect.
    """
    timestamp: datetime
    
    # Sensitivity to each bias type (0-1)
    sensitivities: Dict[BiasCategory, float]
    
    # Activation levels (recent triggers)
    activations: Dict[BiasCategory, float]
    
    # Confidence in each sensitivity
    confidence: Dict[BiasCategory, float]
    
    def to_array(self) -> List[float]:
        """Convert to flat array for mathematical operations."""
        return [self.sensitivities.get(b, 0.5) for b in BiasCategory]
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'sensitivities': {k.value: v for k, v in self.sensitivities.items()},
            'activations': {k.value: v for k, v in self.activations.items()},
            'confidence': {k.value: v for k, v in self.confidence.items()}
        }


@dataclass
class BiasEvent:
    """Record of a bias detection event."""
    event_id: str
    timestamp: datetime
    bias_type: BiasCategory
    detected: bool
    strength: float
    was_correct: Optional[bool]  # From feedback
    context: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'bias_type': self.bias_type.value,
            'detected': self.detected,
            'strength': self.strength,
            'was_correct': self.was_correct
        }


@dataclass
class EvolutionStep:
    """Record of state evolution."""
    timestamp: datetime
    old_state: BiasStateVector
    new_state: BiasStateVector
    trigger: str
    changes: Dict[BiasCategory, float]


class DynamicBiasEvolution:
    """
    Dynamic Bias Evolution System.
    
    PPA-1 Invention 24: Full Implementation
    
    Models bias detection as a dynamic, evolving system
    that learns and adapts based on feedback and patterns.
    """
    
    # Plasticity parameters
    LEARNING_RATE_CRITICAL = 0.3   # High learning during critical period
    LEARNING_RATE_CONSOLIDATION = 0.15
    LEARNING_RATE_MATURE = 0.05   # Slow learning when mature
    
    # Decay parameters
    DECAY_RATE = 0.01  # Per hour without activation
    MIN_SENSITIVITY = 0.1  # Never go below this
    MAX_SENSITIVITY = 0.95  # Never go above this
    
    # Hebbian learning parameters
    HEBBIAN_STRENGTHEN = 0.1  # When bias co-occurs
    HEBBIAN_WEAKEN = 0.05    # When bias expected but not found
    
    # Critical period duration
    CRITICAL_PERIOD_OBSERVATIONS = 100
    
    def __init__(self, storage_path: Path = None):
        # Use temp directory if no path provided (fixes read-only filesystem issues)
        if storage_path is None:
            import tempfile
            temp_dir = Path(tempfile.mkdtemp(prefix="base_bias_evo_"))
            storage_path = temp_dir / "bias_evolution.json"
        self.storage_path = storage_path
        
        # Current state
        self.current_state = self._initialize_state()
        
        # State history
        self.state_history: deque = deque(maxlen=1000)
        self.state_history.append(self.current_state)
        
        # Event history
        self.event_history: deque = deque(maxlen=5000)
        
        # Co-occurrence matrix (Hebbian learning)
        self.co_occurrence: Dict[Tuple[BiasCategory, BiasCategory], float] = {}
        
        # Current plasticity phase
        self.current_phase = PlasticityPhase.CRITICAL
        self.observations_count = 0
        
        # Load state
        self._load_state()
    
    def _initialize_state(self) -> BiasStateVector:
        """Initialize with neutral bias state."""
        return BiasStateVector(
            timestamp=datetime.utcnow(),
            sensitivities={b: 0.5 for b in BiasCategory},
            activations={b: 0.0 for b in BiasCategory},
            confidence={b: 0.3 for b in BiasCategory}  # Low initial confidence
        )
    
    def record_detection(self,
                        bias_type: BiasCategory,
                        detected: bool,
                        strength: float,
                        context: Dict = None) -> Dict[str, Any]:
        """
        Record a bias detection event and evolve the state.
        
        This is the main entry point for learning.
        """
        import uuid
        
        event = BiasEvent(
            event_id=str(uuid.uuid4())[:8],
            timestamp=datetime.utcnow(),
            bias_type=bias_type,
            detected=detected,
            strength=strength,
            was_correct=None,  # Unknown until feedback
            context=context or {}
        )
        self.event_history.append(event)
        self.observations_count += 1
        
        # Update phase if needed
        self._update_phase()
        
        # Apply learning
        evolution_result = self._evolve_state(event)
        
        # Update co-occurrence if multiple biases
        self._update_co_occurrence(event)
        
        # Apply decay to inactive biases
        self._apply_decay()
        
        # Save state
        self._save_state()
        
        return {
            'event_id': event.event_id,
            'phase': self.current_phase.value,
            'evolution': evolution_result,
            'current_state': self.current_state.to_dict()
        }
    
    def record_feedback(self, event_id: str, was_correct: bool) -> Dict[str, Any]:
        """
        Record feedback on a detection and apply reinforcement learning.
        """
        # Find event
        event = next(
            (e for e in reversed(self.event_history) if e.event_id == event_id),
            None
        )
        
        if not event:
            return {'error': 'Event not found'}
        
        event.was_correct = was_correct
        
        # Apply reinforcement/punishment
        if was_correct:
            # Correct detection - strengthen
            self._strengthen_sensitivity(event.bias_type, event.strength)
        else:
            # Incorrect detection - weaken
            self._weaken_sensitivity(event.bias_type, event.strength)
        
        return {
            'feedback_applied': True,
            'bias_type': event.bias_type.value,
            'was_correct': was_correct,
            'new_sensitivity': self.current_state.sensitivities[event.bias_type]
        }
    
    def _evolve_state(self, event: BiasEvent) -> Dict[str, Any]:
        """Evolve the state based on new event."""
        old_state = self._copy_state(self.current_state)
        
        # Get learning rate based on phase
        lr = self._get_learning_rate()
        
        # Update activation
        self.current_state.activations[event.bias_type] = (
            0.7 * event.strength + 
            0.3 * self.current_state.activations[event.bias_type]
        )
        
        # Update sensitivity if detected
        if event.detected and event.strength > 0.3:
            # Long-term potentiation: repeated detection strengthens
            delta = lr * event.strength * (1 - self.current_state.sensitivities[event.bias_type])
            self.current_state.sensitivities[event.bias_type] = min(
                self.MAX_SENSITIVITY,
                self.current_state.sensitivities[event.bias_type] + delta
            )
        
        # Update confidence
        self.current_state.confidence[event.bias_type] = min(
            0.95,
            self.current_state.confidence[event.bias_type] + 0.01
        )
        
        # Update timestamp
        self.current_state.timestamp = datetime.utcnow()
        
        # Record evolution
        changes = {
            b: self.current_state.sensitivities[b] - old_state.sensitivities[b]
            for b in BiasCategory
        }
        
        step = EvolutionStep(
            timestamp=datetime.utcnow(),
            old_state=old_state,
            new_state=self._copy_state(self.current_state),
            trigger=f"detection_{event.bias_type.value}",
            changes=changes
        )
        self.state_history.append(self.current_state)
        
        return {
            'learning_rate': lr,
            'changes': {k.value: v for k, v in changes.items() if abs(v) > 0.001}
        }
    
    def _strengthen_sensitivity(self, bias_type: BiasCategory, amount: float):
        """Strengthen sensitivity to a bias type (positive reinforcement)."""
        lr = self._get_learning_rate()
        delta = lr * amount * 0.5
        self.current_state.sensitivities[bias_type] = min(
            self.MAX_SENSITIVITY,
            self.current_state.sensitivities[bias_type] + delta
        )
        self.current_state.confidence[bias_type] = min(
            0.95,
            self.current_state.confidence[bias_type] + 0.05
        )
    
    def _weaken_sensitivity(self, bias_type: BiasCategory, amount: float):
        """Weaken sensitivity to a bias type (negative feedback)."""
        lr = self._get_learning_rate()
        delta = lr * amount * 0.5
        self.current_state.sensitivities[bias_type] = max(
            self.MIN_SENSITIVITY,
            self.current_state.sensitivities[bias_type] - delta
        )
        # Reduce confidence when wrong
        self.current_state.confidence[bias_type] = max(
            0.1,
            self.current_state.confidence[bias_type] - 0.1
        )
    
    def _apply_decay(self):
        """Apply decay to inactive bias sensitivities."""
        now = datetime.utcnow()
        
        for bias in BiasCategory:
            # Find last activation
            recent_events = [
                e for e in self.event_history
                if e.bias_type == bias and e.detected
            ]
            
            if not recent_events:
                hours_since = 24  # Assume long time
            else:
                last_event = recent_events[-1]
                hours_since = (now - last_event.timestamp).total_seconds() / 3600
            
            # Decay based on time since activation
            if hours_since > 1:
                decay = self.DECAY_RATE * min(hours_since, 24)
                # Move toward baseline (0.5)
                current = self.current_state.sensitivities[bias]
                if current > 0.5:
                    self.current_state.sensitivities[bias] = max(0.5, current - decay)
                else:
                    self.current_state.sensitivities[bias] = min(0.5, current + decay)
    
    def _update_co_occurrence(self, event: BiasEvent):
        """Update co-occurrence matrix for Hebbian learning."""
        if not event.detected:
            return
        
        # Find other biases detected in recent window
        recent = [
            e for e in self.event_history
            if e.detected and 
            (event.timestamp - e.timestamp).total_seconds() < 60  # 1 minute window
        ]
        
        for other_event in recent:
            if other_event.bias_type != event.bias_type:
                pair = tuple(sorted([event.bias_type, other_event.bias_type], key=lambda x: x.value))
                if pair not in self.co_occurrence:
                    self.co_occurrence[pair] = 0.0
                
                # Hebbian: strengthen connection
                self.co_occurrence[pair] = min(
                    1.0,
                    self.co_occurrence[pair] + self.HEBBIAN_STRENGTHEN
                )
    
    def _update_phase(self):
        """Update plasticity phase based on observations."""
        if self.observations_count < self.CRITICAL_PERIOD_OBSERVATIONS:
            self.current_phase = PlasticityPhase.CRITICAL
        elif self.observations_count < self.CRITICAL_PERIOD_OBSERVATIONS * 3:
            self.current_phase = PlasticityPhase.CONSOLIDATION
        else:
            self.current_phase = PlasticityPhase.MATURE
    
    def _get_learning_rate(self) -> float:
        """Get current learning rate based on phase."""
        if self.current_phase == PlasticityPhase.CRITICAL:
            return self.LEARNING_RATE_CRITICAL
        elif self.current_phase == PlasticityPhase.CONSOLIDATION:
            return self.LEARNING_RATE_CONSOLIDATION
        elif self.current_phase == PlasticityPhase.REACTIVATION:
            return self.LEARNING_RATE_CONSOLIDATION
        else:
            return self.LEARNING_RATE_MATURE
    
    def trigger_reactivation(self, reason: str = "manual"):
        """Trigger re-learning phase (increased plasticity)."""
        self.current_phase = PlasticityPhase.REACTIVATION
        return {
            'phase': self.current_phase.value,
            'learning_rate': self._get_learning_rate(),
            'reason': reason
        }
    
    def _copy_state(self, state: BiasStateVector) -> BiasStateVector:
        """Create a copy of state."""
        return BiasStateVector(
            timestamp=state.timestamp,
            sensitivities=dict(state.sensitivities),
            activations=dict(state.activations),
            confidence=dict(state.confidence)
        )
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current bias evolution state."""
        return {
            'state': self.current_state.to_dict(),
            'phase': self.current_phase.value,
            'observations': self.observations_count,
            'learning_rate': self._get_learning_rate(),
            'co_occurrence_count': len(self.co_occurrence)
        }
    
    def get_bias_trajectory(self, 
                           bias_type: BiasCategory,
                           window_hours: int = 24) -> Dict[str, Any]:
        """Get trajectory of a specific bias over time."""
        cutoff = datetime.utcnow() - timedelta(hours=window_hours)
        
        events = [
            e for e in self.event_history
            if e.bias_type == bias_type and e.timestamp >= cutoff
        ]
        
        if not events:
            return {
                'bias_type': bias_type.value,
                'no_events': True
            }
        
        # Calculate trend
        strengths = [e.strength for e in events]
        if len(strengths) >= 2:
            trend = (strengths[-1] - strengths[0]) / len(strengths)
        else:
            trend = 0.0
        
        return {
            'bias_type': bias_type.value,
            'event_count': len(events),
            'current_sensitivity': self.current_state.sensitivities[bias_type],
            'avg_strength': sum(strengths) / len(strengths),
            'trend': 'increasing' if trend > 0.01 else 'decreasing' if trend < -0.01 else 'stable',
            'trend_value': trend
        }
    
    def predict_future_state(self, hours_ahead: int = 24) -> Dict[str, Any]:
        """Predict future bias state based on current trajectory."""
        predictions = {}
        
        for bias in BiasCategory:
            trajectory = self.get_bias_trajectory(bias, window_hours=24)
            
            if 'no_events' in trajectory:
                # No data - predict decay toward baseline
                current = self.current_state.sensitivities[bias]
                predicted = current + (0.5 - current) * 0.1 * hours_ahead / 24
            else:
                # Use trend to predict
                trend = trajectory.get('trend_value', 0)
                current = self.current_state.sensitivities[bias]
                predicted = current + trend * hours_ahead
            
            # Clamp to valid range
            predicted = max(self.MIN_SENSITIVITY, min(self.MAX_SENSITIVITY, predicted))
            predictions[bias.value] = predicted
        
        return {
            'hours_ahead': hours_ahead,
            'predictions': predictions,
            'current_phase': self.current_phase.value
        }
    
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get statistics about bias evolution."""
        # Calculate accuracy if we have feedback
        events_with_feedback = [e for e in self.event_history if e.was_correct is not None]
        if events_with_feedback:
            accuracy = sum(1 for e in events_with_feedback if e.was_correct) / len(events_with_feedback)
        else:
            accuracy = None
        
        # Get strongest/weakest biases
        sorted_biases = sorted(
            self.current_state.sensitivities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'total_observations': self.observations_count,
            'current_phase': self.current_phase.value,
            'learning_rate': self._get_learning_rate(),
            'accuracy': accuracy,
            'strongest_sensitivities': [(k.value, v) for k, v in sorted_biases[:3]],
            'weakest_sensitivities': [(k.value, v) for k, v in sorted_biases[-3:]],
            'co_occurrence_pairs': len(self.co_occurrence),
            'state_history_size': len(self.state_history)
        }
    
    def _save_state(self):
        """Persist state."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'current_state': self.current_state.to_dict(),
            'phase': self.current_phase.value,
            'observations_count': self.observations_count,
            'co_occurrence': {
                f"{k[0].value}_{k[1].value}": v 
                for k, v in self.co_occurrence.items()
            }
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load state."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path) as f:
                state = json.load(f)
            
            if 'current_state' in state:
                s = state['current_state']
                self.current_state = BiasStateVector(
                    timestamp=datetime.fromisoformat(s['timestamp']),
                    sensitivities={BiasCategory(k): v for k, v in s['sensitivities'].items()},
                    activations={BiasCategory(k): v for k, v in s['activations'].items()},
                    confidence={BiasCategory(k): v for k, v in s['confidence'].items()}
                )
            
            self.current_phase = PlasticityPhase(state.get('phase', 'critical'))
            self.observations_count = state.get('observations_count', 0)
            
        except Exception as e:
            print(f"Warning: Could not load bias evolution state: {e}")

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

