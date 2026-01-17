"""
BAIS Cognitive Governance Engine v16.0
Pluggable Learning Algorithms

This module provides swappable learning algorithms for threshold optimization.
If one algorithm doesn't work, it can be replaced without changing the rest of the system.

Algorithms Implemented:
1. OCO (Online Convex Optimization) - Default, patent-specified
2. Bayesian Optimization - For uncertainty quantification
3. Thompson Sampling - For exploration/exploitation balance
4. UCB (Upper Confidence Bound) - For optimistic learning
5. EXP3 - For adversarial environments
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import math


@dataclass
class LearningOutcome:
    """Represents the outcome of a governance decision."""
    domain: str
    context_features: Dict[str, float]
    accuracy: float
    threshold_used: float
    was_accepted: bool
    was_correct: bool  # Ground truth feedback
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            'domain': self.domain,
            'context_features': self.context_features,
            'accuracy': self.accuracy,
            'threshold_used': self.threshold_used,
            'was_accepted': self.was_accepted,
            'was_correct': self.was_correct,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class LearningState:
    """Serializable state of a learning algorithm."""
    algorithm_name: str
    parameters: Dict[str, Any]
    history: List[Dict]
    metadata: Dict[str, Any]
    
    def to_json(self) -> str:
        return json.dumps({
            'algorithm_name': self.algorithm_name,
            'parameters': self.parameters,
            'history': self.history,
            'metadata': self.metadata
        }, default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'LearningState':
        data = json.loads(json_str)
        return cls(**data)


class LearningAlgorithm(ABC):
    """
    Abstract base class for pluggable learning algorithms.
    
    All algorithms must implement:
    - update(): Learn from an outcome
    - get_value(): Get current learned value for a context
    - get_state(): Export state for persistence
    - load_state(): Import state from persistence
    """
    
    @abstractmethod
    def update(self, outcome: LearningOutcome) -> Dict[str, float]:
        """
        Update the algorithm based on an outcome.
        
        Returns dict with:
        - old_value: Value before update
        - new_value: Value after update
        - gradient: Direction of change
        - learning_rate: Effective learning rate used
        """
        pass
    
    @abstractmethod
    def get_value(self, domain: str, context: Dict[str, float] = None) -> float:
        """
        Get the learned value (threshold) for a domain/context.
        
        Args:
            domain: The domain (medical, financial, etc.)
            context: Optional context features for context-dependent values
        
        Returns:
            The learned threshold value
        """
        pass
    
    @abstractmethod
    def get_state(self) -> LearningState:
        """Export algorithm state for persistence."""
        pass
    
    @abstractmethod
    def load_state(self, state: LearningState):
        """Load algorithm state from persistence."""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics for monitoring."""
        pass
    
    def initialize_from(self, other: 'LearningAlgorithm'):
        """
        Initialize this algorithm from another algorithm's learned values.
        Used when switching algorithms.
        """
        # Default: copy domain values if possible
        other_stats = other.get_statistics()
        if 'domain_values' in other_stats:
            for domain, value in other_stats['domain_values'].items():
                self._set_initial_value(domain, value)
    
    def _set_initial_value(self, domain: str, value: float):
        """Override in subclasses to set initial values."""
        pass
    
    # ===== Learning Interface Methods (Base Implementation) =====
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        """Record algorithm outcome for learning. Override in subclasses if needed."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)
    
    def record_feedback(self, feedback: Dict) -> None:
        """Record feedback for learning. Override in subclasses if needed."""
        if not hasattr(self, '_feedback'):
            self._feedback = []
        self._feedback.append(feedback)
        domain = feedback.get('domain', 'general')
        if not hasattr(self, '_domain_adjustments_base'):
            self._domain_adjustments_base = {}
        if feedback.get('algorithm_ineffective', False):
            self._domain_adjustments_base[domain] = self._domain_adjustments_base.get(domain, 0.0) - 0.05
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt thresholds. Override in subclasses for specific behavior."""
        pass
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get domain adjustment. Override in subclasses if needed."""
        if not hasattr(self, '_domain_adjustments_base'):
            self._domain_adjustments_base = {}
        return self._domain_adjustments_base.get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics. Combines with get_statistics()."""
        base_stats = self.get_statistics()
        base_stats['outcomes_recorded'] = len(getattr(self, '_outcomes', []))
        base_stats['feedback_recorded'] = len(getattr(self, '_feedback', []))
        base_stats['domain_adjustments'] = dict(getattr(self, '_domain_adjustments_base', {}))
        return base_stats

    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback."""
        pass  # Implement specific learning logic


    def serialize_state(self) -> Dict:
        """Serialize state."""
        return {'outcomes': getattr(self, '_outcomes', [])[-100:]}


    def deserialize_state(self, state: Dict) -> None:
        """Restore state."""
        self._outcomes = state.get('outcomes', [])


class OCOLearner(LearningAlgorithm):
    """
    Online Convex Optimization learner - FULL IMPLEMENTATION.
    
    PPA-2 Component 3: Adaptive Threshold Adaptation
    
    FULL PATENT FORMULAS:
    1. Threshold update with lexicographic projection:
       θ_{t+1} = Π_lex(θ_t + η_t * α ⊙ 1{v_t > 0} ⊙ s)
    
    2. Dual weights (primal-dual):
       m_t = (1 − η) * m_{t-1} + η * v_t
       λ_{t+1} = [λ_t + η_λ * (m_t − ρ)]_+
    
    3. Diligence weights (Exponentiated Gradient):
       ψ_{t+1,k} = ψ_{t,k} * exp(−η_ψ * g_{t,k}) / Z_t
    
    4. AdaGrad-style adaptive learning rate:
       η_t = η_0 / √(1 + Σ||∇||²)
    """
    
    def __init__(self, 
                 initial_lr: float = 0.5,
                 min_threshold: float = 30.0,
                 max_threshold: float = 90.0,
                 fp_cost: float = 0.6,   # Cost of false positive
                 fn_cost: float = 0.4,   # Cost of false negative
                 # PPA-2 specific parameters
                 dual_lr: float = 0.1,   # η_λ for dual weight updates
                 diligence_lr: float = 0.05,  # η_ψ for diligence weights
                 momentum: float = 0.9,  # (1-η) for exponential moving average
                 constraint_slack: float = 0.1):  # ρ in constraint
        
        self.initial_lr = initial_lr
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.fp_cost = fp_cost
        self.fn_cost = fn_cost
        self.dual_lr = dual_lr
        self.diligence_lr = diligence_lr
        self.momentum = momentum
        self.constraint_slack = constraint_slack
        
        # Per-domain state
        self.thresholds: Dict[str, float] = defaultdict(lambda: 50.0)
        self.gradient_accum: Dict[str, float] = defaultdict(lambda: 0.0)
        self.update_count: Dict[str, int] = defaultdict(lambda: 0)
        self.regret_history: Dict[str, List[float]] = defaultdict(list)
        
        # PPA-2: Dual weights (λ) - Lagrange multipliers for constraints
        self.dual_weights: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {'fp_rate': 0.0, 'fn_rate': 0.0, 'fairness': 0.0}
        )
        
        # PPA-2: Moving average of violations (m_t)
        self.violation_ema: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {'fp_rate': 0.0, 'fn_rate': 0.0, 'fairness': 0.0}
        )
        
        # PPA-2: Diligence weights (ψ) - per-constraint importance
        self.diligence_weights: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {'accuracy': 0.33, 'safety': 0.33, 'fairness': 0.33}
        )
        
        # Constraint priorities for lexicographic projection
        # Higher priority constraints are satisfied first
        self.constraint_priority = ['safety', 'accuracy', 'fairness']
        
        # Default thresholds per domain (with safety ordering: medical > financial > legal > technical > general)
        self._defaults = {
            'medical': 70.0,
            'financial': 65.0,
            'legal': 65.0,
            'technical': 55.0,
            'general': 50.0
        }
        
        # Domain ordering constraints (per PPA-2: medical must be strictest)
        self._domain_ordering = {
            'medical': {'min_margin_over_general': 15.0},
            'financial': {'min_margin_over_general': 10.0},
            'legal': {'min_margin_over_general': 10.0},
            'technical': {'min_margin_over_general': 5.0}
        }
        
        # Learning interface state
        self._outcomes: List[Dict] = []
        self._feedback: List[Dict] = []
        self._domain_adjustments_learning: Dict[str, float] = {}
        
        # Initialize with defaults
        for domain, value in self._defaults.items():
            self.thresholds[domain] = value
    
    @property
    def threshold(self) -> float:
        """Simple interface - returns default domain threshold."""
        return self.thresholds.get('default', 50.0)
    
    def update(self, outcome, was_correct: bool = None) -> Dict[str, float]:
        """
        Full OCO update with lexicographic projection, dual weights, and diligence weights.
        
        PPA-2 Component 3: Complete implementation of patent formulas.
        
        Args:
            outcome: LearningOutcome object OR accuracy value (float) for simple interface
            was_correct: If outcome is float, this indicates correctness
        """
        # Handle simple interface (accuracy, was_correct)
        if isinstance(outcome, (int, float)):
            accuracy = float(outcome)
            current_threshold = self.thresholds.get('default', 50.0)
            was_correct = was_correct if was_correct is not None else (accuracy > current_threshold)
            outcome = LearningOutcome(
                domain='default',
                context_features={},
                accuracy=accuracy,
                threshold_used=current_threshold,
                was_accepted=accuracy > current_threshold,
                was_correct=was_correct
            )
        
        domain = outcome.domain
        current_τ = self.thresholds[domain]
        
        # Step 1: Compute gradient based on decision outcome
        if outcome.was_accepted and not outcome.was_correct:
            # False positive: threshold was too low, increase it
            gradient = self.fp_cost
            regret = self.fp_cost * abs(outcome.accuracy - current_τ)
            violation_type = 'fp_rate'
            violation_value = 1.0
        elif not outcome.was_accepted and outcome.was_correct:
            # False negative: threshold was too high, decrease it
            gradient = -self.fn_cost
            regret = self.fn_cost * abs(current_τ - outcome.accuracy)
            violation_type = 'fn_rate'
            violation_value = 1.0
        else:
            # Correct decision
            gradient = 0.0
            regret = 0.0
            violation_type = None
            violation_value = 0.0
        
        # Step 2: Update violation EMA (m_t)
        # m_t = (1 − η) * m_{t-1} + η * v_t
        if violation_type:
            old_ema = self.violation_ema[domain][violation_type]
            new_ema = self.momentum * old_ema + (1 - self.momentum) * violation_value
            self.violation_ema[domain][violation_type] = new_ema
        
        # Step 3: Update dual weights (λ) - primal-dual update
        # λ_{t+1} = [λ_t + η_λ * (m_t − ρ)]_+
        for constraint_type in ['fp_rate', 'fn_rate', 'fairness']:
            m_t = self.violation_ema[domain][constraint_type]
            old_λ = self.dual_weights[domain][constraint_type]
            new_λ = max(0, old_λ + self.dual_lr * (m_t - self.constraint_slack))
            self.dual_weights[domain][constraint_type] = new_λ
        
        # Step 4: Update diligence weights (ψ) - Exponentiated Gradient
        # ψ_{t+1,k} = ψ_{t,k} * exp(−η_ψ * g_{t,k}) / Z_t
        self._update_diligence_weights(domain, gradient, violation_type)
        
        # Step 5: Compute weighted gradient using diligence weights
        weighted_gradient = gradient * self.diligence_weights[domain].get('accuracy', 0.33)
        
        # Add dual weight contribution
        for constraint_type, λ in self.dual_weights[domain].items():
            if constraint_type == 'fp_rate' and violation_type == 'fp_rate':
                weighted_gradient += λ * 0.5  # Push threshold up
            elif constraint_type == 'fn_rate' and violation_type == 'fn_rate':
                weighted_gradient -= λ * 0.5  # Push threshold down
        
        # Step 6: Accumulate squared gradient for AdaGrad
        self.gradient_accum[domain] += weighted_gradient ** 2
        
        # Step 7: Adaptive learning rate
        η = self.initial_lr / math.sqrt(1 + self.gradient_accum[domain])
        
        # Step 8: Basic threshold update
        old_τ = current_τ
        candidate_τ = current_τ - η * weighted_gradient
        
        # Step 9: Apply lexicographic projection (Π_lex)
        # This enforces constraints in priority order
        new_τ = self._lexicographic_projection(domain, candidate_τ)
        
        self.thresholds[domain] = new_τ
        self.update_count[domain] += 1
        self.regret_history[domain].append(regret)
        
        return {
            'old_value': old_τ,
            'new_value': new_τ,
            'gradient': gradient,
            'weighted_gradient': weighted_gradient,
            'learning_rate': η,
            'regret': regret,
            'cumulative_regret': sum(self.regret_history[domain]),
            # PPA-2 specific outputs
            'dual_weights': dict(self.dual_weights[domain]),
            'diligence_weights': dict(self.diligence_weights[domain]),
            'violation_ema': dict(self.violation_ema[domain]),
            'lexicographic_applied': new_τ != candidate_τ
        }
    
    def _update_diligence_weights(self, domain: str, gradient: float, violation_type: Optional[str]):
        """
        Update diligence weights using Exponentiated Gradient.
        
        PPA-2: ψ_{t+1,k} = ψ_{t,k} * exp(−η_ψ * g_{t,k}) / Z_t
        """
        weights = self.diligence_weights[domain]
        
        # Compute gradient for each weight dimension
        g = {'accuracy': 0.0, 'safety': 0.0, 'fairness': 0.0}
        
        if violation_type == 'fp_rate':
            # False positive: safety was compromised
            g['safety'] = abs(gradient)
            g['accuracy'] = -abs(gradient) * 0.5
        elif violation_type == 'fn_rate':
            # False negative: accuracy was compromised
            g['accuracy'] = abs(gradient)
            g['safety'] = -abs(gradient) * 0.3
        
        # Exponentiated gradient update
        new_weights = {}
        for k, ψ_k in weights.items():
            new_weights[k] = ψ_k * math.exp(-self.diligence_lr * g.get(k, 0))
        
        # Normalize (Z_t)
        Z = sum(new_weights.values())
        if Z > 0:
            for k in new_weights:
                new_weights[k] /= Z
        
        self.diligence_weights[domain] = new_weights
    
    def _lexicographic_projection(self, domain: str, candidate_τ: float) -> float:
        """
        Apply lexicographic projection to enforce constraints in priority order.
        
        PPA-2: Π_lex ensures higher-priority constraints are satisfied first.
        
        Constraint priority:
        1. Box constraints: τ ∈ [min_threshold, max_threshold]
        2. Domain ordering: medical ≥ financial ≥ legal ≥ technical ≥ general
        3. Safety margins: high-risk domains maintain minimum margin over general
        """
        τ = candidate_τ
        
        # Priority 1: Box constraints (always enforced)
        τ = np.clip(τ, self.min_threshold, self.max_threshold)
        
        # Priority 2: Domain ordering constraints
        general_τ = self.thresholds.get('general', 50.0)
        
        if domain in self._domain_ordering:
            min_margin = self._domain_ordering[domain]['min_margin_over_general']
            min_allowed = general_τ + min_margin
            τ = max(τ, min_allowed)
        
        # Priority 3: Ensure medical > financial >= legal > technical
        if domain == 'medical':
            financial_τ = self.thresholds.get('financial', 65.0)
            τ = max(τ, financial_τ + 5.0)  # Medical must be at least 5 above financial
        elif domain == 'financial':
            technical_τ = self.thresholds.get('technical', 55.0)
            τ = max(τ, technical_τ + 5.0)  # Financial must be at least 5 above technical
        
        # Final box constraint enforcement
        τ = np.clip(τ, self.min_threshold, self.max_threshold)
        
        return τ
    
    def get_constraint_violations(self, domain: str) -> Dict[str, float]:
        """Get current constraint violation rates for a domain."""
        return dict(self.violation_ema[domain])
    
    def get_value(self, domain: str, context: Dict[str, float] = None) -> float:
        """Get learned threshold for domain."""
        if domain not in self.thresholds:
            self.thresholds[domain] = self._defaults.get(domain, 50.0)
        return self.thresholds[domain]
    
    def get_state(self) -> LearningState:
        return LearningState(
            algorithm_name='OCO',
            parameters={
                'initial_lr': self.initial_lr,
                'min_threshold': self.min_threshold,
                'max_threshold': self.max_threshold,
                'fp_cost': self.fp_cost,
                'fn_cost': self.fn_cost
            },
            history=[
                {
                    'domain': d,
                    'threshold': self.thresholds[d],
                    'gradient_accum': self.gradient_accum[d],
                    'update_count': self.update_count[d],
                    'regret_history': self.regret_history[d][-100:]  # Last 100
                }
                for d in self.thresholds.keys()
            ],
            metadata={
                'total_updates': sum(self.update_count.values()),
                'last_updated': datetime.utcnow().isoformat()
            }
        )
    
    def load_state(self, state: LearningState):
        if state.algorithm_name != 'OCO':
            raise ValueError(f"Cannot load {state.algorithm_name} state into OCO")
        
        # Load parameters
        self.initial_lr = state.parameters.get('initial_lr', self.initial_lr)
        self.min_threshold = state.parameters.get('min_threshold', self.min_threshold)
        self.max_threshold = state.parameters.get('max_threshold', self.max_threshold)
        self.fp_cost = state.parameters.get('fp_cost', self.fp_cost)
        self.fn_cost = state.parameters.get('fn_cost', self.fn_cost)
        
        # Load per-domain state
        for entry in state.history:
            domain = entry['domain']
            self.thresholds[domain] = entry['threshold']
            self.gradient_accum[domain] = entry['gradient_accum']
            self.update_count[domain] = entry['update_count']
            self.regret_history[domain] = entry.get('regret_history', [])
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'algorithm': 'OCO',
            'domain_values': dict(self.thresholds),
            'update_counts': dict(self.update_count),
            'cumulative_regret': {
                d: sum(r) for d, r in self.regret_history.items()
            },
            'convergence_status': {
                d: self._check_convergence(d) for d in self.thresholds
            }
        }
    
    def _check_convergence(self, domain: str) -> str:
        """Check if threshold has converged for domain."""
        if self.update_count[domain] < 50:
            return 'insufficient_data'
        
        recent = self.regret_history[domain][-20:]
        if not recent:
            return 'no_data'
        
        avg_recent = sum(recent) / len(recent)
        if avg_recent < 0.5:
            return 'converged'
        elif avg_recent < 1.0:
            return 'converging'
        else:
            return 'learning'
    
    def _set_initial_value(self, domain: str, value: float):
        self.thresholds[domain] = value
    
    # ===== Learning Interface Methods (OCOLearner) =====
    
    def record_outcome(self, outcome: Dict) -> None:
        """Record threshold adaptation outcome for learning."""
        self._outcomes.append(outcome)
    
    def record_feedback(self, feedback: Dict) -> None:
        """Record feedback on threshold decisions."""
        self._feedback.append(feedback)
        domain = feedback.get('domain', 'general')
        if feedback.get('feedback_type') == 'threshold_too_strict':
            self._domain_adjustments_learning[domain] = self._domain_adjustments_learning.get(domain, 0.0) - 0.05
        elif feedback.get('feedback_type') == 'threshold_too_lenient':
            self._domain_adjustments_learning[domain] = self._domain_adjustments_learning.get(domain, 0.0) + 0.05
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt thresholds based on performance."""
        if performance_data:
            fp_rate = performance_data.get('false_positive_rate', 0)
            fn_rate = performance_data.get('false_negative_rate', 0)
            if fp_rate > 0.2:
                self.thresholds[domain] = min(self.max_threshold, self.thresholds[domain] + 2.0)
            if fn_rate > 0.2:
                self.thresholds[domain] = max(self.min_threshold, self.thresholds[domain] - 2.0)
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get learned adjustment for domain."""
        return self._domain_adjustments_learning.get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics."""
        return {
            'thresholds': dict(self.thresholds),
            'update_counts': dict(self.update_count),
            'domain_adjustments': dict(self._domain_adjustments_learning),
            'outcomes_recorded': len(self._outcomes),
            'feedback_recorded': len(self._feedback)
        }

    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback."""
        pass  # Implement specific learning logic


    def serialize_state(self) -> Dict:
        """Serialize state."""
        return {'outcomes': getattr(self, '_outcomes', [])[-100:]}


    def deserialize_state(self, state: Dict) -> None:
        """Restore state."""
        self._outcomes = state.get('outcomes', [])


class BayesianLearner(LearningAlgorithm):
    """
    Bayesian learning with uncertainty quantification.
    
    Maintains a posterior distribution over thresholds,
    allowing for uncertainty-aware decisions.
    
    Uses Normal-Gamma conjugate prior for computational efficiency.
    """
    
    def __init__(self,
                 prior_mean: float = 50.0,
                 prior_precision: float = 0.1,  # Low precision = high uncertainty
                 prior_alpha: float = 1.0,      # Shape of variance prior
                 prior_beta: float = 1.0):      # Rate of variance prior
        
        self.prior_mean = prior_mean
        self.prior_precision = prior_precision
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        
        # Per-domain posterior parameters
        self.posteriors: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                'mean': prior_mean,
                'precision': prior_precision,
                'alpha': prior_alpha,
                'beta': prior_beta,
                'n': 0
            }
        )
        
        self.observations: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)
    
    def update(self, outcome: LearningOutcome) -> Dict[str, float]:
        """
        Bayesian update using conjugate prior.
        """
        domain = outcome.domain
        posterior = self.posteriors[domain]
        
        # Observation: what threshold SHOULD have been used?
        # If false positive: threshold should have been higher
        # If false negative: threshold should have been lower
        if outcome.was_accepted and not outcome.was_correct:
            optimal_threshold = outcome.accuracy + 10  # Should have been higher
        elif not outcome.was_accepted and outcome.was_correct:
            optimal_threshold = outcome.accuracy - 10  # Should have been lower
        else:
            optimal_threshold = outcome.threshold_used  # Was correct
        
        # Record observation
        self.observations[domain].append((optimal_threshold, outcome.was_correct))
        
        old_mean = posterior['mean']
        
        # Bayesian update (Normal-Gamma)
        n = posterior['n'] + 1
        precision = posterior['precision']
        
        # Updated mean (weighted average of prior and observation)
        new_mean = (precision * posterior['mean'] + optimal_threshold) / (precision + 1)
        new_precision = precision + 1
        
        # Update variance parameters
        new_alpha = posterior['alpha'] + 0.5
        new_beta = posterior['beta'] + 0.5 * precision * (optimal_threshold - posterior['mean'])**2 / (precision + 1)
        
        posterior['mean'] = new_mean
        posterior['precision'] = new_precision
        posterior['alpha'] = new_alpha
        posterior['beta'] = new_beta
        posterior['n'] = n
        
        return {
            'old_value': old_mean,
            'new_value': new_mean,
            'gradient': new_mean - old_mean,
            'learning_rate': 1 / new_precision,
            'uncertainty': self._get_uncertainty(domain)
        }
    
    def get_value(self, domain: str, context: Dict[str, float] = None) -> float:
        """Get posterior mean as threshold."""
        return self.posteriors[domain]['mean']
    
    def get_uncertainty(self, domain: str) -> float:
        """Get uncertainty (standard deviation) of threshold estimate."""
        return self._get_uncertainty(domain)
    
    def _get_uncertainty(self, domain: str) -> float:
        """Compute posterior standard deviation."""
        p = self.posteriors[domain]
        if p['precision'] == 0:
            return float('inf')
        variance = p['beta'] / (p['alpha'] * p['precision'])
        return math.sqrt(variance)
    
    def sample_threshold(self, domain: str) -> float:
        """Sample from posterior distribution (for Thompson Sampling)."""
        p = self.posteriors[domain]
        # Sample variance from Inverse-Gamma
        variance = 1 / np.random.gamma(p['alpha'], 1/p['beta'])
        # Sample mean from Normal
        return np.random.normal(p['mean'], math.sqrt(variance / p['precision']))
    
    def get_state(self) -> LearningState:
        return LearningState(
            algorithm_name='Bayesian',
            parameters={
                'prior_mean': self.prior_mean,
                'prior_precision': self.prior_precision,
                'prior_alpha': self.prior_alpha,
                'prior_beta': self.prior_beta
            },
            history=[
                {
                    'domain': d,
                    'posterior': dict(p),
                    'observations': self.observations[d][-100:]
                }
                for d, p in self.posteriors.items()
            ],
            metadata={
                'total_observations': sum(p['n'] for p in self.posteriors.values())
            }
        )
    
    def load_state(self, state: LearningState):
        if state.algorithm_name != 'Bayesian':
            raise ValueError(f"Cannot load {state.algorithm_name} state into Bayesian")
        
        for entry in state.history:
            domain = entry['domain']
            self.posteriors[domain] = entry['posterior']
            self.observations[domain] = entry.get('observations', [])
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'algorithm': 'Bayesian',
            'domain_values': {d: p['mean'] for d, p in self.posteriors.items()},
            'uncertainties': {d: self._get_uncertainty(d) for d in self.posteriors},
            'observation_counts': {d: p['n'] for d, p in self.posteriors.items()}
        }
    
    def _set_initial_value(self, domain: str, value: float):
        self.posteriors[domain]['mean'] = value

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


class ThompsonSamplingLearner(LearningAlgorithm):
    """
    Thompson Sampling for threshold learning.
    
    Balances exploration and exploitation by sampling from
    posterior distribution of optimal thresholds.
    """
    
    def __init__(self):
        self.bayesian = BayesianLearner()
        self.samples_used: Dict[str, List[float]] = defaultdict(list)
    
    def update(self, outcome: LearningOutcome) -> Dict[str, float]:
        return self.bayesian.update(outcome)
    
    def get_value(self, domain: str, context: Dict[str, float] = None) -> float:
        """Sample threshold from posterior (Thompson Sampling)."""
        sampled = self.bayesian.sample_threshold(domain)
        sampled = np.clip(sampled, 30.0, 90.0)
        self.samples_used[domain].append(sampled)
        return sampled
    
    def get_mean_value(self, domain: str) -> float:
        """Get posterior mean (not sampled)."""
        return self.bayesian.get_value(domain)
    
    def get_state(self) -> LearningState:
        state = self.bayesian.get_state()
        state.algorithm_name = 'ThompsonSampling'
        state.metadata['samples_used'] = {d: s[-50:] for d, s in self.samples_used.items()}
        return state
    
    def load_state(self, state: LearningState):
        if state.algorithm_name not in ['ThompsonSampling', 'Bayesian']:
            raise ValueError(f"Cannot load {state.algorithm_name} into ThompsonSampling")
        state.algorithm_name = 'Bayesian'  # For internal Bayesian loader
        self.bayesian.load_state(state)
        self.samples_used = defaultdict(list, state.metadata.get('samples_used', {}))
    
    def get_statistics(self) -> Dict[str, Any]:
        stats = self.bayesian.get_statistics()
        stats['algorithm'] = 'ThompsonSampling'
        stats['exploration_rate'] = {
            d: np.std(s[-20:]) if len(s) >= 20 else float('inf')
            for d, s in self.samples_used.items()
        }
        return stats
    
    def _set_initial_value(self, domain: str, value: float):
        self.bayesian._set_initial_value(domain, value)

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


class UCBLearner(LearningAlgorithm):
    """
    Upper Confidence Bound learner.
    
    Optimistic learning: uses upper bound of confidence interval
    to encourage exploration of uncertain regions.
    """
    
    def __init__(self, exploration_weight: float = 2.0):
        self.exploration_weight = exploration_weight
        self.means: Dict[str, float] = defaultdict(lambda: 50.0)
        self.counts: Dict[str, int] = defaultdict(lambda: 0)
        self.sum_squares: Dict[str, float] = defaultdict(lambda: 0.0)
        self.total_count = 0
    
    def update(self, outcome: LearningOutcome) -> Dict[str, float]:
        domain = outcome.domain
        
        # Determine target threshold based on outcome
        if outcome.was_accepted and not outcome.was_correct:
            target = outcome.accuracy + 10
        elif not outcome.was_accepted and outcome.was_correct:
            target = outcome.accuracy - 10
        else:
            target = outcome.threshold_used
        
        old_mean = self.means[domain]
        n = self.counts[domain] + 1
        
        # Incremental mean update
        new_mean = old_mean + (target - old_mean) / n
        
        # Incremental variance update (Welford's algorithm)
        self.sum_squares[domain] += (target - old_mean) * (target - new_mean)
        
        self.means[domain] = new_mean
        self.counts[domain] = n
        self.total_count += 1
        
        return {
            'old_value': old_mean,
            'new_value': new_mean,
            'gradient': new_mean - old_mean,
            'learning_rate': 1 / n,
            'ucb': self._compute_ucb(domain)
        }
    
    def get_value(self, domain: str, context: Dict[str, float] = None) -> float:
        """Get UCB value (optimistic estimate)."""
        return self._compute_ucb(domain)
    
    def _compute_ucb(self, domain: str) -> float:
        """Compute Upper Confidence Bound."""
        n = self.counts[domain]
        if n == 0:
            return 50.0  # Default with high exploration
        
        mean = self.means[domain]
        variance = self.sum_squares[domain] / n if n > 1 else 100.0
        std = math.sqrt(variance)
        
        # UCB formula
        exploration_bonus = self.exploration_weight * std * math.sqrt(math.log(self.total_count + 1) / n)
        
        return np.clip(mean + exploration_bonus, 30.0, 90.0)
    
    def get_state(self) -> LearningState:
        return LearningState(
            algorithm_name='UCB',
            parameters={'exploration_weight': self.exploration_weight},
            history=[
                {
                    'domain': d,
                    'mean': self.means[d],
                    'count': self.counts[d],
                    'sum_squares': self.sum_squares[d]
                }
                for d in self.means.keys()
            ],
            metadata={'total_count': self.total_count}
        )
    
    def load_state(self, state: LearningState):
        if state.algorithm_name != 'UCB':
            raise ValueError(f"Cannot load {state.algorithm_name} into UCB")
        
        self.exploration_weight = state.parameters.get('exploration_weight', 2.0)
        self.total_count = state.metadata.get('total_count', 0)
        
        for entry in state.history:
            domain = entry['domain']
            self.means[domain] = entry['mean']
            self.counts[domain] = entry['count']
            self.sum_squares[domain] = entry['sum_squares']
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'algorithm': 'UCB',
            'domain_values': dict(self.means),
            'ucb_values': {d: self._compute_ucb(d) for d in self.means},
            'counts': dict(self.counts),
            'total_count': self.total_count
        }
    
    def _set_initial_value(self, domain: str, value: float):
        self.means[domain] = value

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


class EXP3Learner(LearningAlgorithm):
    """
    EXP3 (Exponential-weight algorithm for Exploration and Exploitation).
    
    Designed for adversarial environments where outcomes might not be i.i.d.
    Maintains distribution over discrete threshold levels.
    """
    
    def __init__(self, 
                 threshold_levels: List[float] = None,
                 gamma: float = 0.1):  # Exploration parameter
        
        self.threshold_levels = threshold_levels or [30, 40, 50, 60, 70, 80, 90]
        self.gamma = gamma
        self.K = len(self.threshold_levels)
        
        # Per-domain weights
        self.weights: Dict[str, np.ndarray] = defaultdict(
            lambda: np.ones(self.K)
        )
        self.last_action: Dict[str, int] = {}
    
    def _get_probabilities(self, domain: str) -> np.ndarray:
        """Get action probabilities from weights."""
        w = self.weights[domain]
        w_sum = np.sum(w)
        # Mix uniform exploration with weight-based exploitation
        probs = (1 - self.gamma) * (w / w_sum) + (self.gamma / self.K)
        return probs
    
    def update(self, outcome: LearningOutcome) -> Dict[str, float]:
        domain = outcome.domain
        
        if domain not in self.last_action:
            return {'old_value': 50.0, 'new_value': 50.0, 'gradient': 0.0, 'learning_rate': 0.0}
        
        action = self.last_action[domain]
        probs = self._get_probabilities(domain)
        
        # Reward: 1 if correct, 0 if wrong
        reward = 1.0 if outcome.was_correct else 0.0
        
        # Importance-weighted reward
        estimated_reward = reward / probs[action]
        
        # Update weight
        old_weight = self.weights[domain][action]
        self.weights[domain][action] *= np.exp(self.gamma * estimated_reward / self.K)
        
        old_value = self.threshold_levels[action]
        new_action = np.argmax(self.weights[domain])
        new_value = self.threshold_levels[new_action]
        
        return {
            'old_value': old_value,
            'new_value': new_value,
            'gradient': new_value - old_value,
            'learning_rate': self.gamma,
            'reward': reward,
            'action_taken': action
        }
    
    def get_value(self, domain: str, context: Dict[str, float] = None) -> float:
        """Sample threshold according to EXP3 distribution."""
        probs = self._get_probabilities(domain)
        action = np.random.choice(self.K, p=probs)
        self.last_action[domain] = action
        return self.threshold_levels[action]
    
    def get_state(self) -> LearningState:
        return LearningState(
            algorithm_name='EXP3',
            parameters={
                'threshold_levels': self.threshold_levels,
                'gamma': self.gamma
            },
            history=[
                {
                    'domain': d,
                    'weights': w.tolist()
                }
                for d, w in self.weights.items()
            ],
            metadata={}
        )
    
    def load_state(self, state: LearningState):
        if state.algorithm_name != 'EXP3':
            raise ValueError(f"Cannot load {state.algorithm_name} into EXP3")
        
        self.threshold_levels = state.parameters.get('threshold_levels', self.threshold_levels)
        self.gamma = state.parameters.get('gamma', self.gamma)
        self.K = len(self.threshold_levels)
        
        for entry in state.history:
            domain = entry['domain']
            self.weights[domain] = np.array(entry['weights'])
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'algorithm': 'EXP3',
            'domain_values': {
                d: self.threshold_levels[np.argmax(w)]
                for d, w in self.weights.items()
            },
            'weight_distributions': {
                d: dict(zip(self.threshold_levels, (w / np.sum(w)).tolist()))
                for d, w in self.weights.items()
            }
        }
    
    def _set_initial_value(self, domain: str, value: float):
        # Find closest threshold level
        idx = np.argmin([abs(t - value) for t in self.threshold_levels])
        self.weights[domain][idx] *= 2  # Boost that level

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


class PrimalDualAscent(LearningAlgorithm):
    """
    Primal-Dual Ascent Algorithm for Constrained Online Learning.
    
    PPA-2 Component 5: Primal-Dual Optimization
    
    This algorithm handles constrained optimization problems where we need to
    balance multiple objectives (accuracy, fairness, false positive rates, etc.)
    using Lagrangian duality.
    
    KEY FORMULAS:
    1. Primal update (threshold):
       θ_{t+1} = θ_t - η_θ * (∇L_θ + Σ_i λ_i * ∇g_i(θ))
    
    2. Dual update (Lagrange multipliers):
       λ_{t+1} = [λ_t + η_λ * g(θ_t)]_+
       where [x]_+ = max(0, x) projects to non-negative orthant
    
    3. Constraint functions g_i(θ):
       - g_fp: false_positive_rate - ε_fp ≤ 0
       - g_fn: false_negative_rate - ε_fn ≤ 0
       - g_fair: |group_a_rate - group_b_rate| - δ ≤ 0
    
    ADAPTIVE FEATURES:
    - Learning rate decay: η_t = η_0 / (1 + decay * t)
    - Constraint violation tracking with EMA smoothing
    - Automatic constraint tightening based on performance
    """
    
    def __init__(self,
                 primal_lr: float = 0.1,          # η_θ: Learning rate for threshold
                 dual_lr: float = 0.05,           # η_λ: Learning rate for multipliers
                 lr_decay: float = 0.001,         # Decay factor for adaptive LR
                 fp_constraint: float = 0.1,      # ε_fp: Max false positive rate
                 fn_constraint: float = 0.1,      # ε_fn: Max false negative rate
                 fairness_constraint: float = 0.05,  # δ: Max fairness gap
                 ema_alpha: float = 0.1,          # EMA smoothing for violations
                 min_threshold: float = 30.0,
                 max_threshold: float = 90.0,
                 constraint_adaptation: bool = True):  # Auto-adjust constraints
        
        self.primal_lr = primal_lr
        self.dual_lr = dual_lr
        self.lr_decay = lr_decay
        self.fp_constraint = fp_constraint
        self.fn_constraint = fn_constraint
        self.fairness_constraint = fairness_constraint
        self.ema_alpha = ema_alpha
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.constraint_adaptation = constraint_adaptation
        
        # Per-domain primal variables (thresholds)
        self.thresholds: Dict[str, float] = defaultdict(lambda: 50.0)
        
        # Per-domain dual variables (Lagrange multipliers)
        self.dual_vars: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {'fp': 0.1, 'fn': 0.1, 'fairness': 0.1}
        )
        
        # Constraint violation tracking (EMA smoothed)
        self.violation_ema: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {'fp': 0.0, 'fn': 0.0, 'fairness': 0.0}
        )
        
        # Per-domain statistics
        self.update_count: Dict[str, int] = defaultdict(int)
        self.fp_history: Dict[str, List[float]] = defaultdict(list)
        self.fn_history: Dict[str, List[float]] = defaultdict(list)
        self.duality_gap: Dict[str, List[float]] = defaultdict(list)
        
        # Adaptive constraint bounds (learned)
        self.learned_constraints: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {'fp': fp_constraint, 'fn': fn_constraint, 'fairness': fairness_constraint}
        )
        
        # Gradient accumulator for AdaGrad-style adaptation
        self.grad_accum: Dict[str, float] = defaultdict(lambda: 1.0)
    
    def _get_adaptive_lr(self, domain: str, lr_type: str = 'primal') -> float:
        """Compute adaptive learning rate with decay and gradient accumulation."""
        base_lr = self.primal_lr if lr_type == 'primal' else self.dual_lr
        t = self.update_count[domain]
        
        # Time-based decay
        time_factor = 1.0 / (1.0 + self.lr_decay * t)
        
        # AdaGrad-style scaling
        grad_factor = 1.0 / math.sqrt(self.grad_accum[domain])
        
        return base_lr * time_factor * grad_factor
    
    def _compute_constraint_violations(self, outcome: LearningOutcome) -> Dict[str, float]:
        """Compute constraint violation values g_i(θ)."""
        domain = outcome.domain
        
        # Current rates from history
        fp_rate = np.mean(self.fp_history[domain][-100:]) if self.fp_history[domain] else 0.0
        fn_rate = np.mean(self.fn_history[domain][-100:]) if self.fn_history[domain] else 0.0
        
        # Constraint violations (positive = violated)
        constraints = self.learned_constraints[domain]
        violations = {
            'fp': fp_rate - constraints['fp'],
            'fn': fn_rate - constraints['fn'],
            'fairness': 0.0  # Updated if fairness data available
        }
        
        return violations
    
    def _project_threshold(self, theta: float) -> float:
        """Project threshold to feasible set [min, max]."""
        return max(self.min_threshold, min(self.max_threshold, theta))
    
    def _project_dual(self, lambda_val: float) -> float:
        """Project dual variable to non-negative orthant."""
        return max(0.0, lambda_val)
    
    def update(self, outcome: LearningOutcome) -> Dict[str, float]:
        """
        Primal-dual update step.
        
        1. Compute constraint violations
        2. Update dual variables (ascent on Lagrangian)
        3. Update primal variable (descent on Lagrangian)
        """
        domain = outcome.domain
        self.update_count[domain] += 1
        
        # Record outcome for rate calculation
        is_fp = outcome.was_accepted and not outcome.was_correct
        is_fn = not outcome.was_accepted and outcome.was_correct
        self.fp_history[domain].append(1.0 if is_fp else 0.0)
        self.fn_history[domain].append(1.0 if is_fn else 0.0)
        
        # Get current state
        old_theta = self.thresholds[domain]
        old_duals = self.dual_vars[domain].copy()
        
        # Compute constraint violations
        violations = self._compute_constraint_violations(outcome)
        
        # Update violation EMA
        for key in violations:
            self.violation_ema[domain][key] = (
                self.ema_alpha * violations[key] + 
                (1 - self.ema_alpha) * self.violation_ema[domain][key]
            )
        
        # === DUAL UPDATE (Lagrange multiplier ascent) ===
        dual_lr = self._get_adaptive_lr(domain, 'dual')
        for key in self.dual_vars[domain]:
            # Ascent step: λ += η * g(θ)
            self.dual_vars[domain][key] += dual_lr * self.violation_ema[domain][key]
            # Project to non-negative
            self.dual_vars[domain][key] = self._project_dual(self.dual_vars[domain][key])
        
        # === PRIMAL UPDATE (threshold descent) ===
        primal_lr = self._get_adaptive_lr(domain, 'primal')
        
        # Compute primal gradient
        # Loss gradient: increases threshold if FN, decreases if FP
        loss_grad = -1.0 if is_fn else (1.0 if is_fp else 0.0)
        
        # Constraint gradients weighted by dual variables
        # Higher threshold → lower FP rate, higher FN rate
        constraint_grad = (
            -self.dual_vars[domain]['fp'] +  # Lower FP → decrease threshold
            self.dual_vars[domain]['fn']     # Lower FN → increase threshold
        )
        
        total_grad = loss_grad + constraint_grad
        
        # Update gradient accumulator
        self.grad_accum[domain] += total_grad ** 2
        
        # Descent step
        new_theta = old_theta - primal_lr * total_grad
        new_theta = self._project_threshold(new_theta)
        self.thresholds[domain] = new_theta
        
        # Compute duality gap for convergence monitoring
        primal_obj = (is_fp * 0.6 + is_fn * 0.4)  # Primal objective
        dual_obj = primal_obj + sum(
            self.dual_vars[domain][k] * violations[k] 
            for k in violations
        )
        gap = abs(primal_obj - dual_obj)
        self.duality_gap[domain].append(gap)
        
        # === ADAPTIVE CONSTRAINT TIGHTENING ===
        if self.constraint_adaptation and self.update_count[domain] % 100 == 0:
            # If consistently satisfying constraints, tighten them
            for key in ['fp', 'fn']:
                recent_violation = np.mean([
                    v for v in list(self.violation_ema[domain].values())[-50:]
                ]) if self.violation_ema[domain] else 0.0
                if recent_violation < -0.05:  # Slack in constraint
                    self.learned_constraints[domain][key] *= 0.95  # Tighten
        
        return {
            'old_value': old_theta,
            'new_value': new_theta,
            'gradient': total_grad,
            'learning_rate': primal_lr,
            'dual_vars': self.dual_vars[domain].copy(),
            'violations': violations,
            'duality_gap': gap
        }
    
    def get_value(self, domain: str, context: Dict[str, float] = None) -> float:
        """Get learned threshold for domain."""
        return self.thresholds[domain]
    
    def get_state(self) -> LearningState:
        """Export state for persistence."""
        return LearningState(
            algorithm_name='PrimalDualAscent',
            parameters={
                'primal_lr': self.primal_lr,
                'dual_lr': self.dual_lr,
                'lr_decay': self.lr_decay,
                'fp_constraint': self.fp_constraint,
                'fn_constraint': self.fn_constraint,
                'fairness_constraint': self.fairness_constraint,
                'min_threshold': self.min_threshold,
                'max_threshold': self.max_threshold
            },
            history=[
                {
                    'domain': domain,
                    'threshold': self.thresholds[domain],
                    'dual_vars': self.dual_vars[domain],
                    'update_count': self.update_count[domain],
                    'learned_constraints': self.learned_constraints[domain]
                }
                for domain in self.thresholds
            ],
            metadata={
                'total_updates': sum(self.update_count.values()),
                'avg_duality_gap': np.mean([
                    np.mean(gaps[-10:]) if gaps else 0.0
                    for gaps in self.duality_gap.values()
                ])
            }
        )
    
    def load_state(self, state: LearningState):
        """Load state from persistence."""
        if state.algorithm_name != 'PrimalDualAscent':
            raise ValueError(f"Cannot load {state.algorithm_name} into PrimalDualAscent")
        
        params = state.parameters
        self.primal_lr = params.get('primal_lr', self.primal_lr)
        self.dual_lr = params.get('dual_lr', self.dual_lr)
        self.lr_decay = params.get('lr_decay', self.lr_decay)
        
        for entry in state.history:
            domain = entry['domain']
            self.thresholds[domain] = entry['threshold']
            self.dual_vars[domain] = entry['dual_vars']
            self.update_count[domain] = entry['update_count']
            if 'learned_constraints' in entry:
                self.learned_constraints[domain] = entry['learned_constraints']
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            'algorithm': 'PrimalDualAscent',
            'domain_values': dict(self.thresholds),
            'dual_variables': {d: dict(v) for d, v in self.dual_vars.items()},
            'constraint_violations': {d: dict(v) for d, v in self.violation_ema.items()},
            'learned_constraints': {d: dict(c) for d, c in self.learned_constraints.items()},
            'update_counts': dict(self.update_count),
            'avg_duality_gap': {
                d: np.mean(gaps[-50:]) if gaps else 0.0
                for d, gaps in self.duality_gap.items()
            },
            'convergence': {
                d: gaps[-1] < 0.01 if gaps else False
                for d, gaps in self.duality_gap.items()
            }
        }
    
    def _set_initial_value(self, domain: str, value: float):
        """Set initial threshold value."""
        self.thresholds[domain] = self._project_threshold(value)

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


class ExponentiatedGradient(LearningAlgorithm):
    """
    Exponentiated Gradient Algorithm for Online Learning.
    
    PPA-2 Component 6: Multiplicative Weights Update
    
    This algorithm uses multiplicative (exponential) weight updates which are
    particularly effective for:
    - Maintaining valid probability distributions
    - Handling constraints that must sum to 1
    - Achieving low regret in adversarial settings
    
    KEY FORMULAS:
    1. Weight update (multiplicative):
       w_{t+1,i} = w_{t,i} * exp(-η * ℓ_{t,i}) / Z_t
       where Z_t is normalization factor
    
    2. Diligence weights for bias categories:
       ψ_{t+1,k} = ψ_{t,k} * exp(-η_ψ * g_{t,k}) / Z_t
    
    3. Adaptive learning rate:
       η_t = sqrt(ln(K) / t) where K = number of weights
    
    ADAPTIVE FEATURES:
    - Per-dimension learning rates
    - Automatic weight normalization
    - Regret tracking for convergence monitoring
    - Feature-based context conditioning
    """
    
    def __init__(self,
                 base_lr: float = 0.1,           # Base learning rate
                 min_weight: float = 1e-6,       # Minimum weight (numerical stability)
                 n_threshold_bins: int = 7,       # Discretization bins
                 threshold_range: Tuple[float, float] = (30.0, 90.0),
                 context_dims: int = 5,           # Context feature dimensions
                 use_context: bool = True,        # Enable context-dependent learning
                 regret_window: int = 100):       # Window for regret calculation
        
        self.base_lr = base_lr
        self.min_weight = min_weight
        self.n_bins = n_threshold_bins
        self.threshold_range = threshold_range
        self.context_dims = context_dims
        self.use_context = use_context
        self.regret_window = regret_window
        
        # Threshold discretization
        self.threshold_bins = np.linspace(
            threshold_range[0], threshold_range[1], n_threshold_bins
        )
        
        # Per-domain weights over threshold bins
        self.weights: Dict[str, np.ndarray] = defaultdict(
            lambda: np.ones(self.n_bins) / self.n_bins
        )
        
        # Per-domain diligence weights (for bias categories)
        self.diligence_weights: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                'grounding': 1.0,
                'factual': 1.0,
                'behavioral': 1.0,
                'temporal': 1.0
            }
        )
        
        # Context-conditioned weights (if enabled)
        self.context_weights: Dict[str, np.ndarray] = defaultdict(
            lambda: np.ones((self.context_dims, self.n_bins)) / self.n_bins
        )
        
        # Loss history for regret calculation
        self.loss_history: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self.best_fixed_loss: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_bins)
        )
        
        # Statistics
        self.update_count: Dict[str, int] = defaultdict(int)
        self.regret_history: Dict[str, List[float]] = defaultdict(list)
        self.last_action: Dict[str, int] = {}
    
    def _get_adaptive_lr(self, domain: str) -> float:
        """Compute adaptive learning rate based on time and number of weights."""
        t = max(1, self.update_count[domain])
        # Optimal rate for EG: sqrt(ln(K) / T)
        return self.base_lr * math.sqrt(math.log(self.n_bins) / t)
    
    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Normalize weights to form probability distribution."""
        weights = np.maximum(weights, self.min_weight)
        return weights / np.sum(weights)
    
    def _compute_context_features(self, context: Dict[str, float]) -> np.ndarray:
        """Extract context features for context-conditioned learning."""
        if not context:
            return np.ones(self.context_dims) / self.context_dims
        
        # Extract relevant features
        features = np.array([
            context.get('complexity', 0.5),
            context.get('domain_confidence', 0.5),
            context.get('query_length', 0.5),
            context.get('has_citations', 0.5),
            context.get('risk_level', 0.5)
        ])[:self.context_dims]
        
        # Pad if needed
        if len(features) < self.context_dims:
            features = np.pad(features, (0, self.context_dims - len(features)), 
                            constant_values=0.5)
        
        return features
    
    def _sample_threshold(self, domain: str, context: Dict[str, float] = None) -> Tuple[int, float]:
        """Sample threshold from current distribution."""
        if self.use_context and context:
            # Context-conditioned sampling
            ctx_features = self._compute_context_features(context)
            # Weighted combination of context weights
            combined = np.zeros(self.n_bins)
            for i, feat in enumerate(ctx_features):
                combined += feat * self.context_weights[domain][i]
            probs = self._normalize_weights(combined)
        else:
            probs = self._normalize_weights(self.weights[domain])
        
        action = np.random.choice(self.n_bins, p=probs)
        return action, self.threshold_bins[action]
    
    def update(self, outcome: LearningOutcome) -> Dict[str, float]:
        """
        Exponentiated gradient update.
        
        1. Compute loss for taken action
        2. Update weights using multiplicative rule
        3. Update diligence weights
        4. Track regret
        """
        domain = outcome.domain
        self.update_count[domain] += 1
        
        # Get action that was taken
        if domain not in self.last_action:
            action = self.n_bins // 2  # Default to middle
        else:
            action = self.last_action[domain]
        
        old_value = self.threshold_bins[action]
        
        # Compute loss (0-1 loss: 1 if wrong, 0 if correct)
        loss = 0.0 if outcome.was_correct else 1.0
        
        # Also compute losses for all bins (for regret calculation)
        all_losses = np.zeros(self.n_bins)
        for i, thresh in enumerate(self.threshold_bins):
            # Simulate: would we have accepted with this threshold?
            would_accept = outcome.accuracy >= (thresh / 100.0)
            if would_accept == outcome.was_correct:
                all_losses[i] = 0.0
            else:
                all_losses[i] = 1.0
        
        # Track cumulative loss for best fixed action (regret bound)
        self.best_fixed_loss[domain] += all_losses
        self.loss_history[domain].append((action, loss))
        
        # === EXPONENTIATED GRADIENT UPDATE ===
        eta = self._get_adaptive_lr(domain)
        
        # Multiplicative update: w *= exp(-η * loss)
        # For the chosen action
        old_weights = self.weights[domain].copy()
        
        # Update based on loss gradient
        loss_gradient = np.zeros(self.n_bins)
        loss_gradient[action] = loss
        
        self.weights[domain] = old_weights * np.exp(-eta * loss_gradient)
        self.weights[domain] = self._normalize_weights(self.weights[domain])
        
        # === CONTEXT WEIGHT UPDATE (if enabled) ===
        if self.use_context:
            ctx_features = self._compute_context_features(outcome.context_features)
            for i, feat in enumerate(ctx_features):
                if feat > 0.1:  # Only update relevant context dimensions
                    self.context_weights[domain][i] *= np.exp(-eta * feat * loss_gradient)
                    self.context_weights[domain][i] = self._normalize_weights(
                        self.context_weights[domain][i]
                    )
        
        # === DILIGENCE WEIGHT UPDATE ===
        # Update category importance based on which signals were reliable
        if outcome.was_correct:
            # Boost weights for categories that contributed
            for cat in self.diligence_weights[domain]:
                if outcome.context_features.get(f'{cat}_score', 0) > 0.7:
                    self.diligence_weights[domain][cat] *= math.exp(0.1)
        else:
            # Reduce weights for categories that may have misled
            for cat in self.diligence_weights[domain]:
                if outcome.context_features.get(f'{cat}_score', 0) > 0.7:
                    self.diligence_weights[domain][cat] *= math.exp(-0.1)
        
        # Normalize diligence weights
        total = sum(self.diligence_weights[domain].values())
        for cat in self.diligence_weights[domain]:
            self.diligence_weights[domain][cat] /= total
        
        # === REGRET CALCULATION ===
        cumulative_algo_loss = sum(l for _, l in self.loss_history[domain])
        best_fixed = np.min(self.best_fixed_loss[domain])
        regret = cumulative_algo_loss - best_fixed
        self.regret_history[domain].append(regret)
        
        # Get new value (mode of distribution)
        new_action = np.argmax(self.weights[domain])
        new_value = self.threshold_bins[new_action]
        
        return {
            'old_value': old_value,
            'new_value': new_value,
            'gradient': -(new_value - old_value),
            'learning_rate': eta,
            'loss': loss,
            'regret': regret,
            'weight_entropy': -np.sum(self.weights[domain] * np.log(self.weights[domain] + 1e-10))
        }
    
    def get_value(self, domain: str, context: Dict[str, float] = None) -> float:
        """Sample threshold from learned distribution."""
        action, value = self._sample_threshold(domain, context)
        self.last_action[domain] = action
        return value
    
    def get_state(self) -> LearningState:
        """Export state for persistence."""
        return LearningState(
            algorithm_name='ExponentiatedGradient',
            parameters={
                'base_lr': self.base_lr,
                'n_bins': self.n_bins,
                'threshold_range': self.threshold_range,
                'context_dims': self.context_dims,
                'use_context': self.use_context
            },
            history=[
                {
                    'domain': domain,
                    'weights': self.weights[domain].tolist(),
                    'diligence_weights': self.diligence_weights[domain],
                    'update_count': self.update_count[domain],
                    'context_weights': self.context_weights[domain].tolist() if self.use_context else None
                }
                for domain in self.weights
            ],
            metadata={
                'total_updates': sum(self.update_count.values()),
                'avg_regret': np.mean([
                    self.regret_history[d][-1] if self.regret_history[d] else 0.0
                    for d in self.weights
                ])
            }
        )
    
    def load_state(self, state: LearningState):
        """Load state from persistence."""
        if state.algorithm_name != 'ExponentiatedGradient':
            raise ValueError(f"Cannot load {state.algorithm_name} into ExponentiatedGradient")
        
        params = state.parameters
        self.base_lr = params.get('base_lr', self.base_lr)
        self.n_bins = params.get('n_bins', self.n_bins)
        self.threshold_range = tuple(params.get('threshold_range', self.threshold_range))
        self.use_context = params.get('use_context', self.use_context)
        
        # Rebuild threshold bins
        self.threshold_bins = np.linspace(
            self.threshold_range[0], self.threshold_range[1], self.n_bins
        )
        
        for entry in state.history:
            domain = entry['domain']
            self.weights[domain] = np.array(entry['weights'])
            self.diligence_weights[domain] = entry['diligence_weights']
            self.update_count[domain] = entry['update_count']
            if self.use_context and entry.get('context_weights'):
                self.context_weights[domain] = np.array(entry['context_weights'])
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            'algorithm': 'ExponentiatedGradient',
            'domain_values': {
                d: self.threshold_bins[np.argmax(w)]
                for d, w in self.weights.items()
            },
            'weight_distributions': {
                d: dict(zip(self.threshold_bins.tolist(), w.tolist()))
                for d, w in self.weights.items()
            },
            'diligence_weights': {d: dict(w) for d, w in self.diligence_weights.items()},
            'update_counts': dict(self.update_count),
            'regret': {
                d: self.regret_history[d][-1] if self.regret_history[d] else 0.0
                for d in self.weights
            },
            'regret_per_round': {
                d: (self.regret_history[d][-1] / self.update_count[d]) 
                   if self.regret_history[d] and self.update_count[d] > 0 else 0.0
                for d in self.weights
            }
        }
    
    def _set_initial_value(self, domain: str, value: float):
        """Set initial value by boosting nearest bin."""
        idx = np.argmin(np.abs(self.threshold_bins - value))
        self.weights[domain][idx] *= 3.0  # Boost this bin
        self.weights[domain] = self._normalize_weights(self.weights[domain])

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


# =============================================================================
# MIRROR DESCENT (PPA2-Comp2)
# =============================================================================

class MirrorDescent(LearningAlgorithm):
    """
    Mirror Descent learner for adaptive thresholding.
    PPA-2 Component 2: Mirror Descent
    
    Uses Bregman divergence with entropy (KL) as the mirror map.
    Provides better convergence for constrained optimization.
    """
    
    def __init__(self,
                 learning_rate: float = 0.1,
                 min_threshold: float = 30.0,
                 max_threshold: float = 90.0,
                 regularization: float = 0.01):
        self.learning_rate = learning_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.regularization = regularization
        
        # Dual space parameters (log space for entropy mirror)
        self.dual_params: Dict[str, float] = defaultdict(lambda: 0.0)
        self.thresholds: Dict[str, float] = defaultdict(lambda: 50.0)
        self.update_count: Dict[str, int] = defaultdict(int)
        self.gradient_history: Dict[str, List[float]] = defaultdict(list)
        self.regret_history: Dict[str, List[float]] = defaultdict(list)
    
    def _mirror_map(self, dual: float) -> float:
        """Map from dual to primal space (exp for entropy)."""
        # Softmax-style mapping
        return self.min_threshold + (self.max_threshold - self.min_threshold) / (1 + np.exp(-dual))
    
    def _inverse_mirror(self, primal: float) -> float:
        """Map from primal to dual space (log for entropy)."""
        normalized = (primal - self.min_threshold) / (self.max_threshold - self.min_threshold)
        normalized = np.clip(normalized, 0.01, 0.99)
        return np.log(normalized / (1 - normalized))
    
    def update(self, outcome: LearningOutcome) -> Dict[str, float]:
        domain = outcome.domain
        old_value = self.thresholds[domain]
        
        # Compute gradient based on outcome
        if outcome.was_correct:
            gradient = 0.0  # No adjustment needed
        else:
            if outcome.was_accepted:
                gradient = 1.0  # FP: increase threshold
            else:
                gradient = -1.0  # FN: decrease threshold
        
        # Add regularization toward default
        gradient += self.regularization * (self.thresholds[domain] - 50.0)
        
        # Update in dual space
        self.dual_params[domain] -= self.learning_rate * gradient
        
        # Mirror back to primal space
        self.thresholds[domain] = self._mirror_map(self.dual_params[domain])
        
        # Track history
        self.update_count[domain] += 1
        self.gradient_history[domain].append(gradient)
        
        # Compute regret
        loss = 0 if outcome.was_correct else 1
        best_loss = 0  # Optimal hindsight
        regret = loss - best_loss
        prev_regret = self.regret_history[domain][-1] if self.regret_history[domain] else 0
        self.regret_history[domain].append(prev_regret + regret)
        
        return {
            'old_value': old_value,
            'new_value': self.thresholds[domain],
            'gradient': gradient,
            'learning_rate': self.learning_rate,
            'dual_param': self.dual_params[domain]
        }
    
    def get_value(self, domain: str, context: Dict[str, float] = None) -> float:
        return self.thresholds[domain]
    
    def get_state(self) -> LearningState:
        return LearningState(
            algorithm_name='MirrorDescent',
            parameters={
                'learning_rate': self.learning_rate,
                'min_threshold': self.min_threshold,
                'max_threshold': self.max_threshold,
                'regularization': self.regularization
            },
            history=[{
                'domain': d,
                'threshold': self.thresholds[d],
                'dual_param': self.dual_params[d],
                'update_count': self.update_count[d]
            } for d in self.thresholds],
            metadata={'regret': {d: r[-1] if r else 0 for d, r in self.regret_history.items()}}
        )
    
    def load_state(self, state: LearningState):
        for entry in state.history:
            domain = entry['domain']
            self.thresholds[domain] = entry['threshold']
            self.dual_params[domain] = entry['dual_param']
            self.update_count[domain] = entry['update_count']
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'algorithm': 'MirrorDescent',
            'thresholds': dict(self.thresholds),
            'dual_params': dict(self.dual_params),
            'update_counts': dict(self.update_count),
            'regret_per_domain': {
                d: r[-1] / self.update_count[d] if r and self.update_count[d] > 0 else 0
                for d, r in self.regret_history.items()
            }
        }
    
    def _set_initial_value(self, domain: str, value: float):
        self.thresholds[domain] = value
        self.dual_params[domain] = self._inverse_mirror(value)

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


# =============================================================================
# FOLLOW THE REGULARIZED LEADER (PPA2-Comp3)
# =============================================================================

class FollowTheRegularizedLeader(LearningAlgorithm):
    """
    Follow The Regularized Leader (FTRL) for adaptive thresholding.
    PPA-2 Component 3: FTRL
    
    Combines cumulative gradients with L2 regularization.
    Better for sparse/noisy feedback environments.
    """
    
    def __init__(self,
                 learning_rate: float = 0.1,
                 l2_regularization: float = 0.01,
                 min_threshold: float = 30.0,
                 max_threshold: float = 90.0):
        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        
        # Cumulative gradients
        self.cumulative_gradients: Dict[str, float] = defaultdict(float)
        self.cumulative_squared: Dict[str, float] = defaultdict(float)  # For adaptive LR
        self.thresholds: Dict[str, float] = defaultdict(lambda: 50.0)
        self.update_count: Dict[str, int] = defaultdict(int)
        self.regret_history: Dict[str, List[float]] = defaultdict(list)
    
    def update(self, outcome: LearningOutcome) -> Dict[str, float]:
        domain = outcome.domain
        old_value = self.thresholds[domain]
        
        # Compute gradient
        if outcome.was_correct:
            gradient = 0.0
        else:
            if outcome.was_accepted:
                gradient = 1.0  # FP
            else:
                gradient = -1.0  # FN
        
        # Accumulate gradients
        self.cumulative_gradients[domain] += gradient
        self.cumulative_squared[domain] += gradient ** 2
        
        # Adaptive learning rate (AdaGrad style)
        adaptive_lr = self.learning_rate / (1 + np.sqrt(self.cumulative_squared[domain]))
        
        # FTRL update: minimize cumulative loss + regularization
        # theta = -cumulative_grad / (regularization + sqrt(sum_squared))
        denominator = self.l2_regularization + np.sqrt(self.cumulative_squared[domain])
        new_threshold = 50.0 - (self.cumulative_gradients[domain] * adaptive_lr / denominator) * 10
        
        # Clip to bounds
        self.thresholds[domain] = np.clip(new_threshold, self.min_threshold, self.max_threshold)
        
        self.update_count[domain] += 1
        
        # Track regret
        loss = 0 if outcome.was_correct else 1
        prev_regret = self.regret_history[domain][-1] if self.regret_history[domain] else 0
        self.regret_history[domain].append(prev_regret + loss)
        
        return {
            'old_value': old_value,
            'new_value': self.thresholds[domain],
            'gradient': gradient,
            'learning_rate': adaptive_lr,
            'cumulative_gradient': self.cumulative_gradients[domain]
        }
    
    def get_value(self, domain: str, context: Dict[str, float] = None) -> float:
        return self.thresholds[domain]
    
    def get_state(self) -> LearningState:
        return LearningState(
            algorithm_name='FollowTheRegularizedLeader',
            parameters={
                'learning_rate': self.learning_rate,
                'l2_regularization': self.l2_regularization
            },
            history=[{
                'domain': d,
                'threshold': self.thresholds[d],
                'cumulative_grad': self.cumulative_gradients[d],
                'cumulative_sq': self.cumulative_squared[d],
                'update_count': self.update_count[d]
            } for d in self.thresholds],
            metadata={}
        )
    
    def load_state(self, state: LearningState):
        for entry in state.history:
            domain = entry['domain']
            self.thresholds[domain] = entry['threshold']
            self.cumulative_gradients[domain] = entry['cumulative_grad']
            self.cumulative_squared[domain] = entry['cumulative_sq']
            self.update_count[domain] = entry['update_count']
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'algorithm': 'FollowTheRegularizedLeader',
            'thresholds': dict(self.thresholds),
            'cumulative_gradients': dict(self.cumulative_gradients),
            'update_counts': dict(self.update_count),
            'regret_bounds': {
                d: np.sqrt(self.update_count[d]) * 2 if self.update_count[d] > 0 else 0
                for d in self.thresholds
            }
        }
    
    def _set_initial_value(self, domain: str, value: float):
        self.thresholds[domain] = value

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


# =============================================================================
# BANDIT FEEDBACK (PPA2-Comp4)
# =============================================================================

class BanditFeedback(LearningAlgorithm):
    """
    Bandit Feedback learner for partial information settings.
    PPA-2 Component 4: Bandit Feedback
    
    Handles scenarios where we only observe reward for chosen actions,
    using importance weighting for unbiased gradient estimates.
    """
    
    def __init__(self,
                 num_arms: int = 10,
                 exploration_rate: float = 0.1,
                 min_threshold: float = 30.0,
                 max_threshold: float = 90.0):
        self.num_arms = num_arms
        self.exploration_rate = exploration_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        
        # Arms correspond to threshold levels
        self.arm_values = np.linspace(min_threshold, max_threshold, num_arms)
        
        # Estimated rewards per arm per domain
        self.estimated_rewards: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(num_arms))
        self.arm_counts: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(num_arms))
        self.current_arm: Dict[str, int] = defaultdict(lambda: num_arms // 2)
        self.update_count: Dict[str, int] = defaultdict(int)
        self.cumulative_reward: Dict[str, float] = defaultdict(float)
    
    def _select_arm(self, domain: str) -> int:
        """Epsilon-greedy arm selection."""
        if np.random.random() < self.exploration_rate:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.estimated_rewards[domain])
    
    def update(self, outcome: LearningOutcome) -> Dict[str, float]:
        domain = outcome.domain
        arm = self.current_arm[domain]
        old_value = self.arm_values[arm]
        
        # Reward: 1 if correct, 0 otherwise
        reward = 1.0 if outcome.was_correct else 0.0
        
        # Importance weighting for unbiased estimate
        prob_selected = (1 - self.exploration_rate) * (1 if arm == np.argmax(self.estimated_rewards[domain]) else 0) + \
                       self.exploration_rate / self.num_arms
        prob_selected = max(prob_selected, 0.01)  # Avoid division by zero
        
        importance_weight = 1.0 / prob_selected
        
        # Update estimated reward with importance weighting
        self.arm_counts[domain][arm] += 1
        n = self.arm_counts[domain][arm]
        self.estimated_rewards[domain][arm] += (importance_weight * reward - self.estimated_rewards[domain][arm]) / n
        
        # Select next arm
        self.current_arm[domain] = self._select_arm(domain)
        new_value = self.arm_values[self.current_arm[domain]]
        
        self.update_count[domain] += 1
        self.cumulative_reward[domain] += reward
        
        return {
            'old_value': old_value,
            'new_value': new_value,
            'gradient': 0.0,  # Bandit doesn't use gradients
            'learning_rate': 1.0 / n,
            'arm_selected': self.current_arm[domain],
            'reward': reward
        }
    
    def get_value(self, domain: str, context: Dict[str, float] = None) -> float:
        arm = self.current_arm[domain]
        return self.arm_values[arm]
    
    def get_state(self) -> LearningState:
        return LearningState(
            algorithm_name='BanditFeedback',
            parameters={
                'num_arms': self.num_arms,
                'exploration_rate': self.exploration_rate
            },
            history=[{
                'domain': d,
                'estimated_rewards': self.estimated_rewards[d].tolist(),
                'arm_counts': self.arm_counts[d].tolist(),
                'current_arm': self.current_arm[d],
                'update_count': self.update_count[d]
            } for d in set(list(self.estimated_rewards.keys()) + list(self.current_arm.keys()))],
            metadata={}
        )
    
    def load_state(self, state: LearningState):
        for entry in state.history:
            domain = entry['domain']
            self.estimated_rewards[domain] = np.array(entry['estimated_rewards'])
            self.arm_counts[domain] = np.array(entry['arm_counts'])
            self.current_arm[domain] = entry['current_arm']
            self.update_count[domain] = entry['update_count']
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'algorithm': 'BanditFeedback',
            'current_thresholds': {d: self.arm_values[a] for d, a in self.current_arm.items()},
            'best_arms': {d: np.argmax(r) for d, r in self.estimated_rewards.items()},
            'arm_counts': {d: c.tolist() for d, c in self.arm_counts.items()},
            'average_reward': {
                d: self.cumulative_reward[d] / self.update_count[d] if self.update_count[d] > 0 else 0
                for d in self.cumulative_reward
            }
        }
    
    def _set_initial_value(self, domain: str, value: float):
        # Find nearest arm
        arm = np.argmin(np.abs(self.arm_values - value))
        self.current_arm[domain] = arm
        self.estimated_rewards[domain][arm] = 0.5  # Neutral prior

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


# =============================================================================
# CONTEXTUAL BANDIT (PPA2-Comp7)
# =============================================================================

class ContextualBandit(LearningAlgorithm):
    """
    Contextual Bandit learner using LinUCB.
    PPA-2 Component 7: Contextual Bandit
    
    Uses context features to select thresholds, combining
    exploration (UCB) with linear reward modeling.
    """
    
    def __init__(self,
                 num_arms: int = 10,
                 context_dim: int = 5,
                 alpha: float = 1.0,
                 min_threshold: float = 30.0,
                 max_threshold: float = 90.0):
        self.num_arms = num_arms
        self.context_dim = context_dim
        self.alpha = alpha  # Exploration parameter
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        
        self.arm_values = np.linspace(min_threshold, max_threshold, num_arms)
        
        # LinUCB parameters per arm per domain
        # A_a = d x d matrix, b_a = d vector
        self.A: Dict[str, List[np.ndarray]] = {}
        self.b: Dict[str, List[np.ndarray]] = {}
        self.current_arm: Dict[str, int] = defaultdict(lambda: num_arms // 2)
        self.update_count: Dict[str, int] = defaultdict(int)
    
    def _ensure_domain(self, domain: str):
        if domain not in self.A:
            self.A[domain] = [np.eye(self.context_dim) for _ in range(self.num_arms)]
            self.b[domain] = [np.zeros(self.context_dim) for _ in range(self.num_arms)]
    
    def _get_context_vector(self, context: Dict[str, float]) -> np.ndarray:
        """Convert context dict to fixed-size vector."""
        # Use standard features
        features = ['confidence', 'accuracy', 'grounding', 'behavioral', 'temporal']
        vec = np.zeros(self.context_dim)
        for i, f in enumerate(features[:self.context_dim]):
            vec[i] = context.get(f, 0.5)
        return vec
    
    def _select_arm(self, domain: str, context: np.ndarray) -> int:
        """LinUCB arm selection."""
        self._ensure_domain(domain)
        
        ucb_values = np.zeros(self.num_arms)
        for a in range(self.num_arms):
            A_inv = np.linalg.inv(self.A[domain][a])
            theta = A_inv @ self.b[domain][a]
            
            # UCB: theta^T x + alpha * sqrt(x^T A^-1 x)
            mean_reward = theta @ context
            uncertainty = self.alpha * np.sqrt(context @ A_inv @ context)
            ucb_values[a] = mean_reward + uncertainty
        
        return np.argmax(ucb_values)
    
    def update(self, outcome: LearningOutcome) -> Dict[str, float]:
        domain = outcome.domain
        self._ensure_domain(domain)
        
        context = self._get_context_vector(outcome.context_features)
        arm = self.current_arm[domain]
        old_value = self.arm_values[arm]
        
        # Reward
        reward = 1.0 if outcome.was_correct else 0.0
        
        # LinUCB update
        self.A[domain][arm] += np.outer(context, context)
        self.b[domain][arm] += reward * context
        
        # Select next arm
        self.current_arm[domain] = self._select_arm(domain, context)
        new_value = self.arm_values[self.current_arm[domain]]
        
        self.update_count[domain] += 1
        
        return {
            'old_value': old_value,
            'new_value': new_value,
            'gradient': 0.0,
            'learning_rate': self.alpha,
            'arm_selected': self.current_arm[domain],
            'reward': reward
        }
    
    def get_value(self, domain: str, context: Dict[str, float] = None) -> float:
        if context is not None:
            self._ensure_domain(domain)
            context_vec = self._get_context_vector(context)
            arm = self._select_arm(domain, context_vec)
            return self.arm_values[arm]
        return self.arm_values[self.current_arm[domain]]
    
    def get_state(self) -> LearningState:
        return LearningState(
            algorithm_name='ContextualBandit',
            parameters={
                'num_arms': self.num_arms,
                'context_dim': self.context_dim,
                'alpha': self.alpha
            },
            history=[{
                'domain': d,
                'A': [a.tolist() for a in self.A[d]],
                'b': [b.tolist() for b in self.b[d]],
                'current_arm': self.current_arm[d],
                'update_count': self.update_count[d]
            } for d in self.A],
            metadata={}
        )
    
    def load_state(self, state: LearningState):
        for entry in state.history:
            domain = entry['domain']
            self.A[domain] = [np.array(a) for a in entry['A']]
            self.b[domain] = [np.array(b) for b in entry['b']]
            self.current_arm[domain] = entry['current_arm']
            self.update_count[domain] = entry['update_count']
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'algorithm': 'ContextualBandit',
            'current_thresholds': {d: self.arm_values[a] for d, a in self.current_arm.items()},
            'update_counts': dict(self.update_count),
            'exploration_param': self.alpha
        }
    
    def _set_initial_value(self, domain: str, value: float):
        arm = np.argmin(np.abs(self.arm_values - value))
        self.current_arm[domain] = arm

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


# =============================================================================
# ALIASES FOR DOCUMENTATION COMPATIBILITY
# =============================================================================

# PPA2 documented names -> actual implementations
OnlineConvexOptimization = OCOLearner
ThompsonSampling = ThompsonSamplingLearner
FTRL = FollowTheRegularizedLeader


# Factory function
def create_algorithm(name: str, **kwargs) -> LearningAlgorithm:
    """Create a learning algorithm by name."""
    algorithms = {
        'oco': OCOLearner,
        'bayesian': BayesianLearner,
        'thompson': ThompsonSamplingLearner,
        'ucb': UCBLearner,
        'exp3': EXP3Learner,
        'primal_dual': PrimalDualAscent,
        'exponentiated_gradient': ExponentiatedGradient,
        'mirror_descent': MirrorDescent,
        'ftrl': FollowTheRegularizedLeader,
        'bandit': BanditFeedback,
        'contextual_bandit': ContextualBandit
    }
    
    if name.lower() not in algorithms:
        raise ValueError(f"Unknown algorithm: {name}. Available: {list(algorithms.keys())}")
    
    return algorithms[name.lower()](**kwargs)


# Algorithm registry for dynamic switching
class AlgorithmRegistry:
    """
    Registry for managing and switching between algorithms.
    """
    
    def __init__(self, storage_path: Path = None):
        # Use temp directory if no path provided (fixes read-only filesystem issues)
        if storage_path is None:
            import tempfile
            temp_dir = Path(tempfile.mkdtemp(prefix="bais_algo_"))
            storage_path = temp_dir / "algorithm_state.json"
        self.storage_path = storage_path
        self.current_algorithm: Optional[LearningAlgorithm] = None
        self.algorithm_history: List[Dict] = []
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
    
    def set_algorithm(self, algorithm: LearningAlgorithm):
        """Set current algorithm, optionally transferring state from previous."""
        if self.current_algorithm is not None:
            # Transfer learned values
            algorithm.initialize_from(self.current_algorithm)
            
            # Record switch
            self.algorithm_history.append({
                'from': self.current_algorithm.get_statistics().get('algorithm', 'unknown'),
                'to': algorithm.get_statistics().get('algorithm', 'unknown'),
                'timestamp': datetime.utcnow().isoformat(),
                'reason': 'manual_switch'
            })
        
        self.current_algorithm = algorithm
    
    def record_performance(self, algorithm_name: str, accuracy: float):
        """Record algorithm performance for comparison."""
        self.performance_metrics[algorithm_name].append(accuracy)
    
    def get_best_algorithm(self) -> str:
        """Get name of best-performing algorithm based on recent accuracy."""
        if not self.performance_metrics:
            return 'oco'  # Default
        
        recent_performance = {}
        for name, accuracies in self.performance_metrics.items():
            if len(accuracies) >= 10:
                recent_performance[name] = np.mean(accuracies[-50:])
        
        if not recent_performance:
            return 'oco'
        
        return max(recent_performance, key=recent_performance.get)
    
    def save(self):
        """Save registry state."""
        if self.current_algorithm is None:
            return
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'current_algorithm': self.current_algorithm.get_state().to_json(),
            'algorithm_history': self.algorithm_history,
            'performance_metrics': {k: v[-100:] for k, v in self.performance_metrics.items()}
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load(self):
        """Load registry state."""
        if not self.storage_path.exists():
            return
        
        with open(self.storage_path) as f:
            state = json.load(f)
        
        # Reconstruct algorithm
        algo_state = LearningState.from_json(state['current_algorithm'])
        self.current_algorithm = create_algorithm(algo_state.algorithm_name.lower())
        self.current_algorithm.load_state(algo_state)
        
        self.algorithm_history = state.get('algorithm_history', [])
        self.performance_metrics = defaultdict(list, state.get('performance_metrics', {}))

