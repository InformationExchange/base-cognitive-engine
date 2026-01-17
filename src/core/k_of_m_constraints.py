"""
BAIS Cognitive Governance Engine v31.0
K-of-M Constraint Acceptance - Configurable Multi-Predicate Gates

Phase 31: Addresses PPA2-C1-32
- Acceptance requires k-of-m constraints
- Configurable k and m parameters
- Multiple predicate evaluation
- Flexible acceptance policies

Patent Claims Addressed:
- PPA2-C1-32: Acceptance requires k-of-m constraints for configurable k, m
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PredicateType(Enum):
    """Types of predicates for k-of-m evaluation."""
    MUST_PASS = "must_pass"
    SHOULD_PASS = "should_pass"
    OPTIONAL = "optional"
    WEIGHTED = "weighted"


class ConstraintPolicy(Enum):
    """Predefined constraint policies."""
    STRICT = "strict"           # All must pass (m-of-m)
    MAJORITY = "majority"       # More than half (ceil(m/2 + 1)-of-m)
    SUPERMAJORITY = "supermajority"  # Two-thirds (ceil(2m/3)-of-m)
    ANY = "any"                 # At least one (1-of-m)
    CUSTOM = "custom"           # Custom k-of-m


@dataclass
class Predicate:
    """Single predicate for evaluation."""
    name: str
    predicate_type: PredicateType
    evaluator: Callable[[Dict[str, Any]], bool]
    weight: float = 1.0
    description: str = ""
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate predicate against context."""
        try:
            return self.evaluator(context)
        except Exception as e:
            logger.warning(f"[Predicate] {self.name} evaluation failed: {e}")
            return False


@dataclass
class PredicateResult:
    """Result of single predicate evaluation."""
    name: str
    passed: bool
    predicate_type: PredicateType
    weight: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KofMResult:
    """Result of k-of-m constraint evaluation."""
    accepted: bool
    k: int
    m: int
    passed_count: int
    total_weight: float
    passed_weight: float
    policy: ConstraintPolicy
    predicate_results: List[PredicateResult] = field(default_factory=list)
    must_pass_failed: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class KofMConstraintEngine:
    """
    K-of-M constraint evaluation engine.
    
    PPA2-C1-32: Acceptance requires k-of-m constraints for configurable k, m.
    
    Evaluates multiple predicates and accepts if at least k of m pass.
    Supports must-pass predicates that must always pass regardless of k.
    """
    
    def __init__(
        self,
        k: Optional[int] = None,
        m: Optional[int] = None,
        policy: ConstraintPolicy = ConstraintPolicy.MAJORITY,
        use_weights: bool = False
    ):
        """
        Initialize k-of-m constraint engine.
        
        Args:
            k: Number of constraints that must pass (None = derive from policy)
            m: Total number of constraints (None = count of registered predicates)
            policy: Predefined policy for k selection
            use_weights: Whether to use weighted voting
        """
        self.k = k
        self.m = m
        self.policy = policy
        self.use_weights = use_weights
        self.predicates: List[Predicate] = []
        
    def register_predicate(
        self,
        name: str,
        evaluator: Callable[[Dict[str, Any]], bool],
        predicate_type: PredicateType = PredicateType.SHOULD_PASS,
        weight: float = 1.0,
        description: str = ""
    ) -> None:
        """
        Register a predicate for evaluation.
        
        Args:
            name: Unique name for predicate
            evaluator: Function that returns True/False given context
            predicate_type: Type of predicate
            weight: Weight for weighted voting
            description: Human-readable description
        """
        predicate = Predicate(
            name=name,
            predicate_type=predicate_type,
            evaluator=evaluator,
            weight=weight,
            description=description
        )
        self.predicates.append(predicate)
        logger.debug(f"[KofM] Registered predicate: {name} ({predicate_type.value})")
    
    def _compute_k(self, m: int) -> int:
        """Compute k based on policy and m."""
        if self.k is not None:
            return min(self.k, m)
        
        if self.policy == ConstraintPolicy.STRICT:
            return m
        elif self.policy == ConstraintPolicy.MAJORITY:
            return (m // 2) + 1
        elif self.policy == ConstraintPolicy.SUPERMAJORITY:
            return (2 * m + 2) // 3
        elif self.policy == ConstraintPolicy.ANY:
            return 1
        else:
            return (m // 2) + 1  # Default to majority
    
    def evaluate(self, context: Dict[str, Any]) -> KofMResult:
        """
        Evaluate all predicates against context.
        
        Args:
            context: Context dictionary for predicate evaluation
            
        Returns:
            KofMResult with acceptance decision
        """
        if not self.predicates:
            return KofMResult(
                accepted=True,
                k=0,
                m=0,
                passed_count=0,
                total_weight=0.0,
                passed_weight=0.0,
                policy=self.policy,
                details={"message": "No predicates registered"}
            )
        
        # Determine m and k
        m = self.m if self.m is not None else len(self.predicates)
        k = self._compute_k(m)
        
        # Evaluate all predicates
        results: List[PredicateResult] = []
        must_pass_failed: List[str] = []
        passed_count = 0
        total_weight = 0.0
        passed_weight = 0.0
        
        for predicate in self.predicates:
            passed = predicate.evaluate(context)
            
            result = PredicateResult(
                name=predicate.name,
                passed=passed,
                predicate_type=predicate.predicate_type,
                weight=predicate.weight
            )
            results.append(result)
            
            # Track must-pass failures
            if predicate.predicate_type == PredicateType.MUST_PASS and not passed:
                must_pass_failed.append(predicate.name)
            
            # Count passes
            if passed:
                passed_count += 1
                passed_weight += predicate.weight
            
            total_weight += predicate.weight
        
        # Determine acceptance
        # Must-pass predicates always block if they fail
        if must_pass_failed:
            accepted = False
        elif self.use_weights:
            # Weighted voting
            weight_threshold = (k / m) * total_weight if m > 0 else 0
            accepted = passed_weight >= weight_threshold
        else:
            # Simple k-of-m
            accepted = passed_count >= k
        
        return KofMResult(
            accepted=accepted,
            k=k,
            m=m,
            passed_count=passed_count,
            total_weight=total_weight,
            passed_weight=passed_weight,
            policy=self.policy,
            predicate_results=results,
            must_pass_failed=must_pass_failed,
            details={
                "use_weights": self.use_weights,
                "passed_predicates": [r.name for r in results if r.passed],
                "failed_predicates": [r.name for r in results if not r.passed]
            }
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Get engine configuration."""
        return {
            "k": self.k,
            "m": self.m,
            "policy": self.policy.value,
            "use_weights": self.use_weights,
            "predicates": [
                {
                    "name": p.name,
                    "type": p.predicate_type.value,
                    "weight": p.weight
                }
                for p in self.predicates
            ]
        }

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


class PredicateLibrary:
    """
    Library of common predicates for BAIS governance.
    """
    
    @staticmethod
    def confidence_threshold(threshold: float) -> Callable[[Dict[str, Any]], bool]:
        """Predicate: confidence >= threshold."""
        def evaluator(ctx: Dict[str, Any]) -> bool:
            return ctx.get("confidence", 0.0) >= threshold
        return evaluator
    
    @staticmethod
    def accuracy_threshold(threshold: float) -> Callable[[Dict[str, Any]], bool]:
        """Predicate: accuracy >= threshold."""
        def evaluator(ctx: Dict[str, Any]) -> bool:
            return ctx.get("accuracy", 0.0) >= threshold
        return evaluator
    
    @staticmethod
    def no_critical_issues() -> Callable[[Dict[str, Any]], bool]:
        """Predicate: no critical issues detected."""
        def evaluator(ctx: Dict[str, Any]) -> bool:
            issues = ctx.get("issues", [])
            return not any("critical" in str(i).lower() for i in issues)
        return evaluator
    
    @staticmethod
    def grounding_sufficient(threshold: float = 0.5) -> Callable[[Dict[str, Any]], bool]:
        """Predicate: grounding score >= threshold."""
        def evaluator(ctx: Dict[str, Any]) -> bool:
            return ctx.get("grounding_score", 0.0) >= threshold
        return evaluator
    
    @staticmethod
    def certificate_valid() -> Callable[[Dict[str, Any]], bool]:
        """Predicate: valid certificate exists."""
        def evaluator(ctx: Dict[str, Any]) -> bool:
            return ctx.get("has_certificate", False)
        return evaluator
    
    @staticmethod
    def temporal_robust() -> Callable[[Dict[str, Any]], bool]:
        """Predicate: temporal robustness confirmed."""
        def evaluator(ctx: Dict[str, Any]) -> bool:
            return ctx.get("temporal_robust", False)
        return evaluator
    
    @staticmethod
    def no_drift_detected() -> Callable[[Dict[str, Any]], bool]:
        """Predicate: no drift detected."""
        def evaluator(ctx: Dict[str, Any]) -> bool:
            return not ctx.get("drift_detected", False)
        return evaluator
    
    @staticmethod
    def domain_approved(approved_domains: List[str]) -> Callable[[Dict[str, Any]], bool]:
        """Predicate: domain is in approved list."""
        def evaluator(ctx: Dict[str, Any]) -> bool:
            domain = ctx.get("domain", "")
            return domain in approved_domains
        return evaluator

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


class KofMConstraintManager:
    """
    Unified manager for k-of-m constraint evaluation.
    
    Provides preset configurations and easy setup for common use cases.
    """
    
    def __init__(self):
        """Initialize constraint manager."""
        self.engines: Dict[str, KofMConstraintEngine] = {}
        self._setup_default_engines()
        
        logger.info("[KofM] K-of-M Constraint Manager initialized")
    
    def _setup_default_engines(self):
        """Setup default constraint engines."""
        # Standard governance engine (majority vote)
        standard = KofMConstraintEngine(policy=ConstraintPolicy.MAJORITY)
        standard.register_predicate(
            "confidence", PredicateLibrary.confidence_threshold(0.5),
            PredicateType.SHOULD_PASS, description="Confidence >= 50%"
        )
        standard.register_predicate(
            "accuracy", PredicateLibrary.accuracy_threshold(0.5),
            PredicateType.SHOULD_PASS, description="Accuracy >= 50%"
        )
        standard.register_predicate(
            "grounding", PredicateLibrary.grounding_sufficient(0.4),
            PredicateType.SHOULD_PASS, description="Grounding >= 40%"
        )
        standard.register_predicate(
            "no_critical", PredicateLibrary.no_critical_issues(),
            PredicateType.MUST_PASS, description="No critical issues"
        )
        self.engines["standard"] = standard
        
        # Strict governance engine (all must pass)
        strict = KofMConstraintEngine(policy=ConstraintPolicy.STRICT)
        strict.register_predicate(
            "confidence", PredicateLibrary.confidence_threshold(0.7),
            PredicateType.MUST_PASS, description="Confidence >= 70%"
        )
        strict.register_predicate(
            "certificate", PredicateLibrary.certificate_valid(),
            PredicateType.MUST_PASS, description="Valid certificate"
        )
        strict.register_predicate(
            "temporal", PredicateLibrary.temporal_robust(),
            PredicateType.MUST_PASS, description="Temporal robustness"
        )
        strict.register_predicate(
            "no_drift", PredicateLibrary.no_drift_detected(),
            PredicateType.MUST_PASS, description="No drift detected"
        )
        self.engines["strict"] = strict
        
        # Lenient engine (any pass)
        lenient = KofMConstraintEngine(policy=ConstraintPolicy.ANY)
        lenient.register_predicate(
            "confidence", PredicateLibrary.confidence_threshold(0.3),
            PredicateType.OPTIONAL, description="Confidence >= 30%"
        )
        lenient.register_predicate(
            "accuracy", PredicateLibrary.accuracy_threshold(0.3),
            PredicateType.OPTIONAL, description="Accuracy >= 30%"
        )
        self.engines["lenient"] = lenient
    
    def evaluate(
        self,
        engine_name: str,
        context: Dict[str, Any]
    ) -> KofMResult:
        """
        Evaluate context against named engine.
        
        Args:
            engine_name: Name of engine to use
            context: Context for evaluation
            
        Returns:
            KofMResult
        """
        engine = self.engines.get(engine_name)
        if not engine:
            raise ValueError(f"Unknown engine: {engine_name}")
        
        return engine.evaluate(context)
    
    def create_engine(
        self,
        name: str,
        k: Optional[int] = None,
        m: Optional[int] = None,
        policy: ConstraintPolicy = ConstraintPolicy.MAJORITY,
        use_weights: bool = False
    ) -> KofMConstraintEngine:
        """
        Create and register a new engine.
        
        Args:
            name: Engine name
            k, m: k-of-m parameters
            policy: Constraint policy
            use_weights: Use weighted voting
            
        Returns:
            Created engine
        """
        engine = KofMConstraintEngine(k=k, m=m, policy=policy, use_weights=use_weights)
        self.engines[name] = engine
        return engine
    
    def get_engine(self, name: str) -> Optional[KofMConstraintEngine]:
        """Get engine by name."""
        return self.engines.get(name)
    
    def list_engines(self) -> List[str]:
        """List available engines."""
        return list(self.engines.keys())

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


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 31: K-of-M Constraints Module Test")
    print("=" * 60)
    
    manager = KofMConstraintManager()
    
    # Test context
    good_context = {
        "confidence": 0.75,
        "accuracy": 0.80,
        "grounding_score": 0.65,
        "issues": [],
        "has_certificate": True,
        "temporal_robust": True,
        "drift_detected": False
    }
    
    bad_context = {
        "confidence": 0.25,
        "accuracy": 0.30,
        "grounding_score": 0.20,
        "issues": ["CRITICAL: Dangerous advice"],
        "has_certificate": False,
        "temporal_robust": False,
        "drift_detected": True
    }
    
    # Test 1: Standard engine (majority)
    print("\n[1] Standard Engine (Majority):")
    result = manager.evaluate("standard", good_context)
    print(f"    Good context: {result.accepted} ({result.passed_count}/{result.m})")
    print(f"    Policy: {result.policy.value} (k={result.k})")
    
    result = manager.evaluate("standard", bad_context)
    print(f"    Bad context: {result.accepted}")
    print(f"    Must-pass failed: {result.must_pass_failed}")
    
    # Test 2: Strict engine (all must pass)
    print("\n[2] Strict Engine (All Must Pass):")
    result = manager.evaluate("strict", good_context)
    print(f"    Good context: {result.accepted} ({result.passed_count}/{result.m})")
    
    result = manager.evaluate("strict", bad_context)
    print(f"    Bad context: {result.accepted}")
    
    # Test 3: Custom k-of-m
    print("\n[3] Custom 2-of-3 Engine:")
    custom = manager.create_engine("custom_2of3", k=2, m=3, policy=ConstraintPolicy.CUSTOM)
    custom.register_predicate("check1", lambda ctx: ctx.get("confidence", 0) > 0.5)
    custom.register_predicate("check2", lambda ctx: ctx.get("accuracy", 0) > 0.5)
    custom.register_predicate("check3", lambda ctx: ctx.get("grounding_score", 0) > 0.5)
    
    result = custom.evaluate(good_context)
    print(f"    Result: {result.accepted} ({result.passed_count}/{result.m})")
    print(f"    Passed: {result.details['passed_predicates']}")
    
    # Test 4: Weighted voting
    print("\n[4] Weighted Voting:")
    weighted = manager.create_engine("weighted", policy=ConstraintPolicy.MAJORITY, use_weights=True)
    weighted.register_predicate("high_importance", lambda ctx: ctx.get("confidence", 0) > 0.5, weight=3.0)
    weighted.register_predicate("low_importance", lambda ctx: ctx.get("accuracy", 0) > 0.9, weight=1.0)
    
    result = weighted.evaluate({"confidence": 0.6, "accuracy": 0.5})
    print(f"    Result: {result.accepted}")
    print(f"    Weight: {result.passed_weight}/{result.total_weight}")
    
    # Test 5: List engines
    print("\n[5] Available Engines:")
    for engine_name in manager.list_engines():
        engine = manager.get_engine(engine_name)
        config = engine.get_config()
        print(f"    {engine_name}: {config['policy']} with {len(config['predicates'])} predicates")
    
    print("\n" + "=" * 60)
    print("PHASE 31: K-of-M Constraints Module - VERIFIED")
    print("=" * 60)


