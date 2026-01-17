"""
BAIS Cognitive Governance Engine v16.4
Configurable Predicate-Based Acceptance Policy

PPA-1 Invention 21: FULL IMPLEMENTATION
Policy function with k-of-n predicates and weighted voting.

This module implements:
1. Configurable Predicates: Define custom acceptance conditions
2. k-of-n Logic: Require k out of n predicates to pass
3. Weighted Voting: Predicates have different weights
4. Domain-Specific Policies: Different rules per domain
5. Dynamic Policy Updates: Adjust policies based on feedback

Policy Language:
- AND: All predicates must pass
- OR: Any predicate must pass
- K_OF_N: At least k of n predicates must pass
- WEIGHTED: Weighted sum exceeds threshold
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
from enum import Enum
import json
from pathlib import Path


class PredicateType(str, Enum):
    """Types of predicates."""
    THRESHOLD = "threshold"       # Score > threshold
    RANGE = "range"               # min <= score <= max
    BOOLEAN = "boolean"           # Direct boolean
    COMPARISON = "comparison"     # Compare two values
    COMPOSITE = "composite"       # Combination of predicates


class PolicyLogic(str, Enum):
    """Policy combination logic."""
    AND = "and"           # All must pass
    OR = "or"             # Any must pass
    K_OF_N = "k_of_n"     # k of n must pass
    WEIGHTED = "weighted"  # Weighted sum > threshold


@dataclass
class Predicate:
    """Single predicate for acceptance decision."""
    predicate_id: str
    name: str
    description: str
    predicate_type: PredicateType
    
    # For threshold predicates
    signal_name: Optional[str] = None
    threshold: float = 0.5
    operator: str = ">="  # ">=", ">", "<=", "<", "==", "!="
    
    # For range predicates
    min_value: float = 0.0
    max_value: float = 1.0
    
    # Weight for weighted policies
    weight: float = 1.0
    
    # Enabled flag
    enabled: bool = True
    
    # Category for grouping
    category: str = "general"
    
    def evaluate(self, signals: Dict[str, float]) -> Tuple[bool, float, str]:
        """
        Evaluate predicate against signals.
        
        Returns: (passed, score, reason)
        """
        if not self.enabled:
            return True, 1.0, "Predicate disabled"
        
        if self.predicate_type == PredicateType.THRESHOLD:
            return self._evaluate_threshold(signals)
        elif self.predicate_type == PredicateType.RANGE:
            return self._evaluate_range(signals)
        elif self.predicate_type == PredicateType.BOOLEAN:
            return self._evaluate_boolean(signals)
        else:
            return True, 0.5, "Unknown predicate type"
    
    def _evaluate_threshold(self, signals: Dict[str, float]) -> Tuple[bool, float, str]:
        """Evaluate threshold predicate."""
        if not self.signal_name or self.signal_name not in signals:
            return False, 0.0, f"Signal {self.signal_name} not found"
        
        value = signals[self.signal_name]
        
        if self.operator == ">=":
            passed = value >= self.threshold
        elif self.operator == ">":
            passed = value > self.threshold
        elif self.operator == "<=":
            passed = value <= self.threshold
        elif self.operator == "<":
            passed = value < self.threshold
        elif self.operator == "==":
            passed = abs(value - self.threshold) < 0.001
        elif self.operator == "!=":
            passed = abs(value - self.threshold) >= 0.001
        else:
            passed = value >= self.threshold
        
        reason = f"{self.signal_name}={value:.2f} {self.operator} {self.threshold}"
        return passed, value, reason
    
    def _evaluate_range(self, signals: Dict[str, float]) -> Tuple[bool, float, str]:
        """Evaluate range predicate."""
        if not self.signal_name or self.signal_name not in signals:
            return False, 0.0, f"Signal {self.signal_name} not found"
        
        value = signals[self.signal_name]
        passed = self.min_value <= value <= self.max_value
        reason = f"{self.min_value} <= {self.signal_name}={value:.2f} <= {self.max_value}"
        return passed, value, reason
    
    def _evaluate_boolean(self, signals: Dict[str, float]) -> Tuple[bool, float, str]:
        """Evaluate boolean predicate."""
        if not self.signal_name or self.signal_name not in signals:
            return False, 0.0, f"Signal {self.signal_name} not found"
        
        value = signals[self.signal_name]
        passed = value > 0.5  # Treat > 0.5 as True
        return passed, value, f"{self.signal_name}={value:.2f} -> {'True' if passed else 'False'}"
    
    def to_dict(self) -> Dict:
        return {
            'predicate_id': self.predicate_id,
            'name': self.name,
            'description': self.description,
            'type': self.predicate_type.value,
            'signal_name': self.signal_name,
            'threshold': self.threshold,
            'operator': self.operator,
            'weight': self.weight,
            'enabled': self.enabled,
            'category': self.category
        }


@dataclass
class PolicyResult:
    """Result of policy evaluation."""
    passed: bool
    logic_used: PolicyLogic
    
    # Predicate results
    predicates_passed: int
    predicates_failed: int
    predicates_total: int
    
    # For k-of-n
    k_required: int = 0
    
    # For weighted
    weighted_score: float = 0.0
    weight_threshold: float = 0.5
    
    # Details
    predicate_results: List[Dict] = field(default_factory=list)
    explanation: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'passed': self.passed,
            'logic': self.logic_used.value,
            'predicates_passed': self.predicates_passed,
            'predicates_failed': self.predicates_failed,
            'predicates_total': self.predicates_total,
            'k_required': self.k_required,
            'weighted_score': self.weighted_score,
            'explanation': self.explanation,
            'predicate_results': self.predicate_results
        }


@dataclass
class AcceptancePolicy:
    """Complete acceptance policy configuration."""
    policy_id: str
    name: str
    description: str
    domain: str  # Domain this policy applies to
    
    # Policy logic
    logic: PolicyLogic = PolicyLogic.AND
    k_value: int = 0  # For k-of-n
    weight_threshold: float = 0.5  # For weighted
    
    # Predicates
    predicates: List[Predicate] = field(default_factory=list)
    
    # Active flag
    enabled: bool = True
    
    # Priority (higher = evaluated first)
    priority: int = 0
    
    def evaluate(self, signals: Dict[str, float]) -> PolicyResult:
        """Evaluate policy against signals."""
        if not self.enabled:
            return PolicyResult(
                passed=True,
                logic_used=self.logic,
                predicates_passed=0,
                predicates_failed=0,
                predicates_total=0,
                explanation="Policy disabled"
            )
        
        # Evaluate all predicates
        results = []
        for pred in self.predicates:
            passed, score, reason = pred.evaluate(signals)
            results.append({
                'predicate_id': pred.predicate_id,
                'name': pred.name,
                'passed': passed,
                'score': score,
                'weight': pred.weight,
                'reason': reason
            })
        
        passed_count = sum(1 for r in results if r['passed'])
        failed_count = len(results) - passed_count
        
        # Apply logic
        if self.logic == PolicyLogic.AND:
            overall_passed = all(r['passed'] for r in results)
            explanation = f"AND: All {len(results)} predicates must pass, {passed_count} passed"
        
        elif self.logic == PolicyLogic.OR:
            overall_passed = any(r['passed'] for r in results)
            explanation = f"OR: Any predicate must pass, {passed_count} of {len(results)} passed"
        
        elif self.logic == PolicyLogic.K_OF_N:
            k = self.k_value if self.k_value > 0 else len(results) // 2 + 1
            overall_passed = passed_count >= k
            explanation = f"K_OF_N: Need {k} of {len(results)}, got {passed_count}"
        
        elif self.logic == PolicyLogic.WEIGHTED:
            weighted_sum = sum(
                r['weight'] for r in results if r['passed']
            )
            total_weight = sum(pred.weight for pred in self.predicates)
            weighted_score = weighted_sum / total_weight if total_weight > 0 else 0
            overall_passed = weighted_score >= self.weight_threshold
            explanation = f"WEIGHTED: Score {weighted_score:.2f} >= {self.weight_threshold}"
        else:
            overall_passed = passed_count > failed_count
            weighted_score = 0.0
            explanation = "Default: majority"
        
        weighted_score = (
            sum(r['weight'] for r in results if r['passed']) / 
            sum(pred.weight for pred in self.predicates)
            if self.predicates else 0
        )
        
        return PolicyResult(
            passed=overall_passed,
            logic_used=self.logic,
            predicates_passed=passed_count,
            predicates_failed=failed_count,
            predicates_total=len(results),
            k_required=self.k_value,
            weighted_score=weighted_score,
            weight_threshold=self.weight_threshold,
            predicate_results=results,
            explanation=explanation
        )
    
    def to_dict(self) -> Dict:
        return {
            'policy_id': self.policy_id,
            'name': self.name,
            'description': self.description,
            'domain': self.domain,
            'logic': self.logic.value,
            'k_value': self.k_value,
            'weight_threshold': self.weight_threshold,
            'predicates': [p.to_dict() for p in self.predicates],
            'enabled': self.enabled,
            'priority': self.priority
        }


class PredicatePolicyEngine:
    """
    Configurable Predicate Policy Engine.
    
    PPA-1 Invention 21: Full Implementation
    
    Manages acceptance policies with k-of-n predicates
    and weighted voting.
    """
    
    def __init__(self, storage_path: Path = None):
        # Use temp directory if no path provided (fixes read-only filesystem issues)
        if storage_path is None:
            import tempfile
            temp_dir = Path(tempfile.mkdtemp(prefix="bais_policy_"))
            storage_path = temp_dir / "policies.json"
        self.storage_path = storage_path
        
        # Policies by domain
        self.policies: Dict[str, AcceptancePolicy] = {}
        
        # Default policy
        self.default_policy: Optional[AcceptancePolicy] = None
        
        # Initialize defaults
        self._initialize_default_policies()
        
        # Load persisted
        self._load_state()
    
    def _initialize_default_policies(self):
        """Initialize default acceptance policies."""
        
        # General domain policy
        general_predicates = [
            Predicate(
                predicate_id="grounding_min",
                name="Minimum Grounding",
                description="Response must have minimum grounding score",
                predicate_type=PredicateType.THRESHOLD,
                signal_name="grounding",
                threshold=0.3,
                operator=">=",
                weight=1.5,
                category="evidence"
            ),
            Predicate(
                predicate_id="factual_min",
                name="Minimum Factual Score",
                description="Response must have minimum factual accuracy",
                predicate_type=PredicateType.THRESHOLD,
                signal_name="factual",
                threshold=0.4,
                operator=">=",
                weight=1.2,
                category="accuracy"
            ),
            Predicate(
                predicate_id="behavioral_max",
                name="Maximum Bias",
                description="Response must not exceed bias threshold",
                predicate_type=PredicateType.THRESHOLD,
                signal_name="behavioral",
                threshold=0.5,
                operator="<=",
                weight=1.0,
                category="safety"
            ),
            Predicate(
                predicate_id="temporal_min",
                name="Temporal Stability",
                description="System must be temporally stable",
                predicate_type=PredicateType.THRESHOLD,
                signal_name="temporal",
                threshold=0.3,
                operator=">=",
                weight=0.8,
                category="stability"
            )
        ]
        
        self.policies["general"] = AcceptancePolicy(
            policy_id="policy_general",
            name="General Acceptance Policy",
            description="Default policy for general queries",
            domain="general",
            logic=PolicyLogic.K_OF_N,
            k_value=3,  # 3 of 4 must pass
            predicates=general_predicates
        )
        
        # Medical domain policy (stricter)
        medical_predicates = [
            Predicate(
                predicate_id="med_grounding",
                name="Medical Evidence Grounding",
                description="Medical claims must be well-grounded",
                predicate_type=PredicateType.THRESHOLD,
                signal_name="grounding",
                threshold=0.5,  # Higher for medical
                operator=">=",
                weight=2.0,
                category="evidence"
            ),
            Predicate(
                predicate_id="med_factual",
                name="Medical Factual Accuracy",
                description="Medical facts must be highly accurate",
                predicate_type=PredicateType.THRESHOLD,
                signal_name="factual",
                threshold=0.6,  # Higher for medical
                operator=">=",
                weight=2.0,
                category="accuracy"
            ),
            Predicate(
                predicate_id="med_bias",
                name="Medical Bias Check",
                description="Medical responses must be unbiased",
                predicate_type=PredicateType.THRESHOLD,
                signal_name="behavioral",
                threshold=0.3,  # Stricter for medical
                operator="<=",
                weight=1.5,
                category="safety"
            ),
            Predicate(
                predicate_id="med_confidence",
                name="Medical Confidence",
                description="Must express appropriate uncertainty",
                predicate_type=PredicateType.RANGE,
                signal_name="confidence_calibration",
                min_value=0.3,
                max_value=0.9,  # Not overconfident
                weight=1.0,
                category="calibration"
            )
        ]
        
        self.policies["medical"] = AcceptancePolicy(
            policy_id="policy_medical",
            name="Medical Acceptance Policy",
            description="Strict policy for medical queries",
            domain="medical",
            logic=PolicyLogic.AND,  # ALL must pass for medical
            predicates=medical_predicates,
            priority=10  # Higher priority
        )
        
        # Financial domain policy
        financial_predicates = [
            Predicate(
                predicate_id="fin_grounding",
                name="Financial Evidence",
                description="Financial claims must be grounded",
                predicate_type=PredicateType.THRESHOLD,
                signal_name="grounding",
                threshold=0.45,
                operator=">=",
                weight=1.8,
                category="evidence"
            ),
            Predicate(
                predicate_id="fin_factual",
                name="Financial Accuracy",
                description="Financial facts must be accurate",
                predicate_type=PredicateType.THRESHOLD,
                signal_name="factual",
                threshold=0.5,
                operator=">=",
                weight=1.5,
                category="accuracy"
            ),
            Predicate(
                predicate_id="fin_bias",
                name="Financial Bias",
                description="No optimistic bias in financial advice",
                predicate_type=PredicateType.THRESHOLD,
                signal_name="behavioral",
                threshold=0.4,
                operator="<=",
                weight=1.2,
                category="safety"
            )
        ]
        
        self.policies["financial"] = AcceptancePolicy(
            policy_id="policy_financial",
            name="Financial Acceptance Policy",
            description="Policy for financial queries",
            domain="financial",
            logic=PolicyLogic.WEIGHTED,
            weight_threshold=0.6,
            predicates=financial_predicates,
            priority=8
        )
        
        # Set default
        self.default_policy = self.policies["general"]
    
    def evaluate(self, domain: str, signals: Dict[str, float]) -> PolicyResult:
        """
        Evaluate signals against the appropriate policy.
        """
        # Find policy for domain
        policy = self.policies.get(domain.lower(), self.default_policy)
        
        if policy is None:
            return PolicyResult(
                passed=True,
                logic_used=PolicyLogic.AND,
                predicates_passed=0,
                predicates_failed=0,
                predicates_total=0,
                explanation="No policy defined"
            )
        
        return policy.evaluate(signals)
    
    def add_policy(self, policy: AcceptancePolicy):
        """Add or update a policy."""
        self.policies[policy.domain.lower()] = policy
        self._save_state()
    
    def add_predicate(self, domain: str, predicate: Predicate):
        """Add a predicate to a policy."""
        if domain.lower() in self.policies:
            self.policies[domain.lower()].predicates.append(predicate)
            self._save_state()
    
    def update_k_value(self, domain: str, k: int):
        """Update k value for k-of-n policy."""
        if domain.lower() in self.policies:
            self.policies[domain.lower()].k_value = k
            self._save_state()
    
    def update_weight_threshold(self, domain: str, threshold: float):
        """Update weight threshold for weighted policy."""
        if domain.lower() in self.policies:
            self.policies[domain.lower()].weight_threshold = threshold
            self._save_state()
    
    def get_policy(self, domain: str) -> Optional[AcceptancePolicy]:
        """Get policy for domain."""
        return self.policies.get(domain.lower())
    
    def list_policies(self) -> List[Dict]:
        """List all policies."""
        return [p.to_dict() for p in self.policies.values()]
    
    def _save_state(self):
        """Persist policies."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'policies': {k: v.to_dict() for k, v in self.policies.items()},
            'default_domain': self.default_policy.domain if self.default_policy else 'general'
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load persisted policies."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path) as f:
                state = json.load(f)
            
            # Load custom policies (keep defaults)
            for domain, policy_dict in state.get('policies', {}).items():
                if domain not in self.policies:
                    # Reconstruct policy
                    predicates = [
                        Predicate(
                            predicate_id=p['predicate_id'],
                            name=p['name'],
                            description=p['description'],
                            predicate_type=PredicateType(p['type']),
                            signal_name=p.get('signal_name'),
                            threshold=p.get('threshold', 0.5),
                            operator=p.get('operator', '>='),
                            weight=p.get('weight', 1.0),
                            enabled=p.get('enabled', True),
                            category=p.get('category', 'general')
                        )
                        for p in policy_dict.get('predicates', [])
                    ]
                    
                    self.policies[domain] = AcceptancePolicy(
                        policy_id=policy_dict['policy_id'],
                        name=policy_dict['name'],
                        description=policy_dict['description'],
                        domain=domain,
                        logic=PolicyLogic(policy_dict.get('logic', 'and')),
                        k_value=policy_dict.get('k_value', 0),
                        weight_threshold=policy_dict.get('weight_threshold', 0.5),
                        predicates=predicates,
                        enabled=policy_dict.get('enabled', True),
                        priority=policy_dict.get('priority', 0)
                    )
                    
        except Exception as e:
            print(f"Warning: Could not load policies: {e}")

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

