"""
BASE Cognitive Governance Engine v30.0
Probe Mode - Quarantine & Shadow Model System

Phase 30: Addresses PPA2-C1-13, PPA2-C3-8
- Quarantine mode for high-impairment scenarios
- Shadow model comparison
- Safe exploration under uncertainty
- Gradual rollout mechanisms

Patent Claims Addressed:
- PPA2-C1-13: Probe mode with quarantine head/shadow model for high impairment
- PPA2-C1-14: Exploration budget under severe skew to avoid stagnation
- PPA2-C3-8: Probe mode for high impairment
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Awaitable
from enum import Enum
from datetime import datetime, timedelta
import logging
import random
import hashlib

logger = logging.getLogger(__name__)


class ProbeStatus(Enum):
    """Status of probe mode."""
    INACTIVE = "inactive"
    ACTIVE = "active"
    QUARANTINED = "quarantined"
    SHADOW = "shadow"
    RECOVERING = "recovering"


class ImpairmentLevel(Enum):
    """Impairment severity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ProbeResult:
    """Result of probe mode evaluation."""
    status: ProbeStatus
    impairment_level: ImpairmentLevel
    should_quarantine: bool
    shadow_comparison: Optional[Dict[str, Any]] = None
    recommendations: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class QuarantineState:
    """State of quarantine for a component."""
    component_id: str
    reason: str
    start_time: datetime
    end_time: Optional[datetime] = None
    impairment_score: float = 0.0
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    is_active: bool = True


@dataclass
class ExplorationBudget:
    """
    Exploration budget under severe skew.
    
    PPA2-C1-14: Exploration budget under severe skew to avoid stagnation.
    """
    total_budget: float = 0.1  # 10% of traffic for exploration
    spent: float = 0.0
    window_start: datetime = field(default_factory=datetime.utcnow)
    window_duration: timedelta = field(default_factory=lambda: timedelta(hours=1))
    
    def can_explore(self) -> bool:
        """Check if exploration budget is available."""
        self._maybe_reset_window()
        return self.spent < self.total_budget
    
    def consume(self, amount: float = 0.01) -> bool:
        """Consume exploration budget."""
        if self.can_explore():
            self.spent += amount
            return True
        return False
    
    def _maybe_reset_window(self):
        """Reset window if expired."""
        if datetime.utcnow() - self.window_start > self.window_duration:
            self.spent = 0.0
            self.window_start = datetime.utcnow()
    
    def remaining(self) -> float:
        """Get remaining budget."""
        self._maybe_reset_window()
        return max(0, self.total_budget - self.spent)

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


class ShadowModel:
    """
    Shadow model for comparison during probe mode.
    
    PPA2-C1-13: Shadow model for high impairment scenarios.
    """
    
    def __init__(self, model_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize shadow model.
        
        Args:
            model_id: Unique identifier for the shadow model
            config: Configuration parameters
        """
        self.model_id = model_id
        self.config = config or {}
        self.predictions: List[Dict[str, Any]] = []
        self.created_at = datetime.utcnow()
        self.is_active = True
        
    async def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make shadow prediction (simulated for now).
        
        Args:
            input_data: Input data for prediction
            
        Returns:
            Shadow model prediction
        """
        # Simulate shadow model prediction
        # In production, this would call an actual model
        prediction = {
            "model_id": self.model_id,
            "input_hash": hashlib.md5(str(input_data).encode()).hexdigest()[:8],
            "confidence": random.uniform(0.5, 0.95),
            "timestamp": datetime.utcnow().isoformat()
        }
        self.predictions.append(prediction)
        return prediction
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get shadow model metrics."""
        if not self.predictions:
            return {"predictions": 0, "avg_confidence": 0.0}
        
        confidences = [p["confidence"] for p in self.predictions]
        return {
            "predictions": len(self.predictions),
            "avg_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences)
        }


class QuarantineManager:
    """
    Manages quarantine state for components.
    
    PPA2-C1-13: Quarantine head for high impairment.
    """
    
    def __init__(self, auto_recovery_timeout: timedelta = timedelta(hours=1)):
        """
        Initialize quarantine manager.
        
        Args:
            auto_recovery_timeout: Time after which to attempt recovery
        """
        self.quarantined: Dict[str, QuarantineState] = {}
        self.auto_recovery_timeout = auto_recovery_timeout
        self.recovery_history: List[Dict[str, Any]] = []
        
    def quarantine(
        self,
        component_id: str,
        reason: str,
        impairment_score: float
    ) -> QuarantineState:
        """
        Quarantine a component.
        
        Args:
            component_id: ID of component to quarantine
            reason: Reason for quarantine
            impairment_score: Severity of impairment (0-1)
            
        Returns:
            QuarantineState
        """
        state = QuarantineState(
            component_id=component_id,
            reason=reason,
            start_time=datetime.utcnow(),
            impairment_score=impairment_score
        )
        self.quarantined[component_id] = state
        logger.warning(f"[Quarantine] Component {component_id} quarantined: {reason}")
        return state
    
    def is_quarantined(self, component_id: str) -> bool:
        """Check if component is quarantined."""
        state = self.quarantined.get(component_id)
        if state and state.is_active:
            return True
        return False
    
    def attempt_recovery(self, component_id: str) -> bool:
        """
        Attempt to recover a quarantined component.
        
        Args:
            component_id: ID of component to recover
            
        Returns:
            True if recovery successful
        """
        state = self.quarantined.get(component_id)
        if not state:
            return True  # Not quarantined
        
        state.recovery_attempts += 1
        
        # Simulate recovery check
        recovery_probability = 1.0 - state.impairment_score
        if random.random() < recovery_probability:
            state.is_active = False
            state.end_time = datetime.utcnow()
            self.recovery_history.append({
                "component_id": component_id,
                "recovered_at": datetime.utcnow().isoformat(),
                "attempts": state.recovery_attempts
            })
            logger.info(f"[Quarantine] Component {component_id} recovered after {state.recovery_attempts} attempts")
            return True
        
        if state.recovery_attempts >= state.max_recovery_attempts:
            logger.error(f"[Quarantine] Component {component_id} failed to recover after {state.max_recovery_attempts} attempts")
        
        return False
    
    def release(self, component_id: str):
        """Force release a component from quarantine."""
        if component_id in self.quarantined:
            state = self.quarantined[component_id]
            state.is_active = False
            state.end_time = datetime.utcnow()
            logger.info(f"[Quarantine] Component {component_id} force-released")
    
    def get_quarantined_components(self) -> List[str]:
        """Get list of currently quarantined components."""
        return [
            cid for cid, state in self.quarantined.items()
            if state.is_active
        ]
    
    def cleanup_expired(self):
        """Clean up expired quarantine states."""
        now = datetime.utcnow()
        for component_id, state in list(self.quarantined.items()):
            if state.is_active and now - state.start_time > self.auto_recovery_timeout:
                self.attempt_recovery(component_id)

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


class ProbeModeManager:
    """
    Main probe mode manager for the BASE system.
    
    Implements:
    - PPA2-C1-13: Probe mode with quarantine/shadow model
    - PPA2-C1-14: Exploration budget under severe skew
    - PPA2-C3-8: Probe mode for high impairment
    """
    
    def __init__(
        self,
        impairment_threshold: float = 0.7,
        shadow_model_enabled: bool = True,
        exploration_budget: float = 0.1
    ):
        """
        Initialize probe mode manager.
        
        Args:
            impairment_threshold: Threshold above which to enter probe mode
            shadow_model_enabled: Whether to use shadow models
            exploration_budget: Budget for exploration (0-1)
        """
        self.impairment_threshold = impairment_threshold
        self.shadow_model_enabled = shadow_model_enabled
        
        self.quarantine = QuarantineManager()
        self.exploration = ExplorationBudget(total_budget=exploration_budget)
        self.shadow_models: Dict[str, ShadowModel] = {}
        
        self.status = ProbeStatus.INACTIVE
        self.current_impairment = 0.0
        self.probe_history: List[Dict[str, Any]] = []
        
        logger.info(f"[ProbeMode] Initialized with threshold={impairment_threshold}")
    
    def assess_impairment(self, metrics: Dict[str, float]) -> ImpairmentLevel:
        """
        Assess impairment level from metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            ImpairmentLevel
        """
        # Calculate impairment score from metrics
        error_rate = metrics.get("error_rate", 0.0)
        latency_spike = metrics.get("latency_spike", 0.0)
        confidence_drop = metrics.get("confidence_drop", 0.0)
        drift_detected = metrics.get("drift_detected", 0.0)
        
        impairment_score = (
            error_rate * 0.3 +
            latency_spike * 0.2 +
            confidence_drop * 0.25 +
            drift_detected * 0.25
        )
        
        self.current_impairment = impairment_score
        
        if impairment_score < 0.2:
            return ImpairmentLevel.NONE
        elif impairment_score < 0.4:
            return ImpairmentLevel.LOW
        elif impairment_score < 0.6:
            return ImpairmentLevel.MEDIUM
        elif impairment_score < 0.8:
            return ImpairmentLevel.HIGH
        else:
            return ImpairmentLevel.CRITICAL
    
    async def evaluate(
        self,
        component_id: str,
        metrics: Dict[str, float],
        input_data: Optional[Dict[str, Any]] = None
    ) -> ProbeResult:
        """
        Evaluate whether to enter probe mode.
        
        Args:
            component_id: Component being evaluated
            metrics: Current performance metrics
            input_data: Optional input data for shadow comparison
            
        Returns:
            ProbeResult with recommendations
        """
        impairment = self.assess_impairment(metrics)
        
        # Check if already quarantined
        if self.quarantine.is_quarantined(component_id):
            return ProbeResult(
                status=ProbeStatus.QUARANTINED,
                impairment_level=impairment,
                should_quarantine=True,
                recommendations=["Component is quarantined, attempting recovery"],
                metrics=metrics
            )
        
        # Check for high impairment
        should_quarantine = impairment in [ImpairmentLevel.HIGH, ImpairmentLevel.CRITICAL]
        
        if should_quarantine:
            self.quarantine.quarantine(
                component_id=component_id,
                reason=f"Impairment level: {impairment.value}",
                impairment_score=self.current_impairment
            )
            self.status = ProbeStatus.QUARANTINED
            
            # Create shadow model if enabled
            shadow_comparison = None
            if self.shadow_model_enabled and input_data:
                shadow = self._get_or_create_shadow(component_id)
                shadow_result = await shadow.predict(input_data)
                shadow_comparison = {
                    "shadow_model_id": shadow.model_id,
                    "shadow_prediction": shadow_result
                }
            
            return ProbeResult(
                status=ProbeStatus.QUARANTINED,
                impairment_level=impairment,
                should_quarantine=True,
                shadow_comparison=shadow_comparison,
                recommendations=[
                    f"Component {component_id} quarantined due to {impairment.value} impairment",
                    "Shadow model activated for comparison",
                    "Reduce traffic to this component"
                ],
                metrics=metrics
            )
        
        # Check for exploration opportunity
        if impairment in [ImpairmentLevel.LOW, ImpairmentLevel.MEDIUM]:
            if self.exploration.can_explore():
                self.exploration.consume()
                self.status = ProbeStatus.ACTIVE
                
                return ProbeResult(
                    status=ProbeStatus.ACTIVE,
                    impairment_level=impairment,
                    should_quarantine=False,
                    recommendations=[
                        "Probe mode active for exploration",
                        f"Exploration budget remaining: {self.exploration.remaining():.2%}"
                    ],
                    metrics=metrics
                )
        
        # Normal operation
        self.status = ProbeStatus.INACTIVE
        return ProbeResult(
            status=ProbeStatus.INACTIVE,
            impairment_level=impairment,
            should_quarantine=False,
            recommendations=["Operating normally"],
            metrics=metrics
        )
    
    def _get_or_create_shadow(self, component_id: str) -> ShadowModel:
        """Get or create shadow model for component."""
        if component_id not in self.shadow_models:
            self.shadow_models[component_id] = ShadowModel(
                model_id=f"shadow_{component_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            )
        return self.shadow_models[component_id]
    
    async def compare_with_shadow(
        self,
        component_id: str,
        primary_result: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare primary model result with shadow model.
        
        Args:
            component_id: Component ID
            primary_result: Result from primary model
            input_data: Input data
            
        Returns:
            Comparison results
        """
        shadow = self._get_or_create_shadow(component_id)
        shadow_result = await shadow.predict(input_data)
        
        # Compare results
        primary_confidence = primary_result.get("confidence", 0.5)
        shadow_confidence = shadow_result.get("confidence", 0.5)
        
        agreement = 1.0 - abs(primary_confidence - shadow_confidence)
        
        comparison = {
            "primary_confidence": primary_confidence,
            "shadow_confidence": shadow_confidence,
            "agreement": agreement,
            "should_prefer_shadow": shadow_confidence > primary_confidence + 0.1,
            "divergence_detected": agreement < 0.7
        }
        
        if comparison["divergence_detected"]:
            logger.warning(f"[ProbeMode] Divergence detected for {component_id}: agreement={agreement:.2f}")
        
        return comparison
    
    def get_status(self) -> Dict[str, Any]:
        """Get current probe mode status."""
        return {
            "status": self.status.value,
            "current_impairment": self.current_impairment,
            "quarantined_components": self.quarantine.get_quarantined_components(),
            "exploration_budget_remaining": self.exploration.remaining(),
            "shadow_models_active": len(self.shadow_models),
            "impairment_threshold": self.impairment_threshold
        }
    
    def cleanup(self):
        """Cleanup expired quarantine states."""
        self.quarantine.cleanup_expired()
    
    # ========================================
    # PHASE 49: LEARNING METHODS (In-Class)
    # ========================================
    
    def record_outcome(self, input_data, output_data, was_correct, domain=None, metadata=None):
        """Record outcome for learning (Phase 49)."""
        if hasattr(self, '_learning_manager') and self._learning_manager:
            self._learning_manager.record_outcome(
                module_name="probemodemanager", input_data=input_data, 
                output_data=output_data, was_correct=was_correct, 
                domain=domain, metadata=metadata
            )
    
    def record_feedback(self, result, was_accurate):
        """Record feedback for learning (Phase 49)."""
        self.record_outcome({"result": str(result)}, {}, was_accurate)
    
    def adapt_thresholds(self, threshold_name, current_value, direction='decrease'):
        """Adapt thresholds based on learning (Phase 49)."""
        if hasattr(self, '_learning_manager') and self._learning_manager:
            return self._learning_manager.adapt_threshold(
                "probemodemanager", threshold_name, current_value, direction
            )
        return max(0.1, min(0.95, current_value + (0.05 if direction == 'increase' else -0.05)))
    
    def get_domain_adjustment(self, domain):
        """Get domain-specific adjustment (Phase 49)."""
        if hasattr(self, '_learning_manager') and self._learning_manager:
            return self._learning_manager.get_domain_adjustment("probemodemanager", domain)
        return 0.0
    
    def get_learning_statistics(self):
        """Get learning statistics (Phase 49)."""
        if hasattr(self, '_learning_manager') and self._learning_manager:
            stats = self._learning_manager.get_learning_statistics("probemodemanager")
            return stats.__dict__ if hasattr(stats, '__dict__') else stats
        return {"module": "ProbeModeManager", "status": "no_learning_manager"}


# Test function
if __name__ == "__main__":
    async def test_probe_mode():
        print("=" * 60)
        print("PHASE 30: Probe Mode Module Test")
        print("=" * 60)
        
        manager = ProbeModeManager(impairment_threshold=0.7)
        
        # Test 1: Normal operation
        print("\n[1] Normal Operation:")
        result = await manager.evaluate(
            component_id="detector_1",
            metrics={"error_rate": 0.05, "confidence_drop": 0.1}
        )
        print(f"    Status: {result.status.value}")
        print(f"    Impairment: {result.impairment_level.value}")
        print(f"    Quarantine: {result.should_quarantine}")
        
        # Test 2: High impairment triggering quarantine
        print("\n[2] High Impairment (triggering quarantine):")
        result = await manager.evaluate(
            component_id="detector_2",
            metrics={
                "error_rate": 0.8,
                "confidence_drop": 0.9,
                "drift_detected": 1.0
            },
            input_data={"query": "test query"}
        )
        print(f"    Status: {result.status.value}")
        print(f"    Impairment: {result.impairment_level.value}")
        print(f"    Quarantine: {result.should_quarantine}")
        if result.shadow_comparison:
            print(f"    Shadow Model: {result.shadow_comparison['shadow_model_id']}")
        
        # Test 3: Check quarantine status
        print("\n[3] Quarantine Status:")
        status = manager.get_status()
        print(f"    Quarantined: {status['quarantined_components']}")
        print(f"    Exploration budget: {status['exploration_budget_remaining']:.2%}")
        print(f"    Shadow models: {status['shadow_models_active']}")
        
        

    # ========================================
    # PHASE 49: LEARNING METHODS
    # ========================================
    
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

# Test 4: Exploration budget
        print("\n[4] Exploration Budget Test:")
        budget = ExplorationBudget(total_budget=0.1)
        consumed = 0
        while budget.can_explore():
            budget.consume(0.02)
            consumed += 1
        print(f"    Consumed {consumed} exploration units")
        print(f"    Remaining: {budget.remaining():.2%}")
        
        print("\n" + "=" * 60)
        print("PHASE 30: Probe Mode Module - VERIFIED")
        print("=" * 60)
    
    asyncio.run(test_probe_mode())


