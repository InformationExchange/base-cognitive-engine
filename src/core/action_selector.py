"""
BASE Action Selector (Layer 6 - Basal Ganglia)

Selects the best action/response from multiple candidates:
1. Action evaluation
2. Risk assessment
3. Reward prediction
4. Final selection

Patent Alignment:
- Part of decision-making layer
- Brain Layer: 6 (Basal Ganglia - Action Selection)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum


class ActionType(Enum):
    """Types of actions."""
    RESPOND = "respond"
    CLARIFY = "clarify"
    REFUSE = "refuse"
    DEFER = "defer"
    ESCALATE = "escalate"


@dataclass
class ActionCandidate:
    """A candidate action to evaluate."""
    action_id: str
    action_type: ActionType
    content: str
    predicted_reward: float = 0.0
    risk_score: float = 0.0
    confidence: float = 0.5


@dataclass
class ActionEvaluation:
    """Evaluation of an action candidate."""
    candidate: ActionCandidate
    utility_score: float
    risk_adjusted_score: float
    factors: Dict[str, float]


@dataclass
class ActionSelection:
    """Result of action selection."""
    selected_action: ActionCandidate
    alternatives: List[ActionCandidate]
    selection_reason: str
    confidence: float


class ActionSelector:
    """
    Selects the best action from multiple candidates.
    
    Brain Layer: 6 (Basal Ganglia)
    
    Responsibilities:
    1. Evaluate action candidates
    2. Predict rewards
    3. Assess risks
    4. Select optimal action
    """
    
    # Risk weights by action type
    RISK_WEIGHTS = {
        ActionType.RESPOND: 0.3,
        ActionType.CLARIFY: 0.1,
        ActionType.REFUSE: 0.2,
        ActionType.DEFER: 0.15,
        ActionType.ESCALATE: 0.25,
    }
    
    # Base reward by action type
    BASE_REWARDS = {
        ActionType.RESPOND: 0.7,
        ActionType.CLARIFY: 0.5,
        ActionType.REFUSE: 0.3,
        ActionType.DEFER: 0.4,
        ActionType.ESCALATE: 0.4,
    }
    
    def __init__(self, risk_tolerance: float = 0.5):
        """
        Initialize the action selector.
        
        Args:
            risk_tolerance: Risk tolerance level (0=risk-averse, 1=risk-seeking)
        """
        self.risk_tolerance = risk_tolerance
        
        # Learning state
        self._outcomes: List[Dict] = []
        self._feedback: List[Dict] = []
        self._domain_adjustments: Dict[str, float] = {}
        self._total_selections: int = 0
        self._action_type_performance: Dict[ActionType, List[float]] = {}
    
    def evaluate(self, candidate: ActionCandidate,
                 context: Dict[str, Any] = None) -> ActionEvaluation:
        """
        Evaluate an action candidate.
        
        Args:
            candidate: Action to evaluate
            context: Optional context
            
        Returns:
            ActionEvaluation with scores
        """
        context = context or {}
        
        # Calculate factors
        factors = {}
        
        # Base reward
        base_reward = self.BASE_REWARDS.get(candidate.action_type, 0.5)
        factors['base_reward'] = base_reward
        
        # Content quality (simplified)
        content_quality = min(1.0, len(candidate.content) / 200) if candidate.content else 0.3
        factors['content_quality'] = content_quality
        
        # Risk factor
        risk_weight = self.RISK_WEIGHTS.get(candidate.action_type, 0.3)
        adjusted_risk = candidate.risk_score * risk_weight
        factors['risk_factor'] = adjusted_risk
        
        # Confidence bonus
        confidence_bonus = candidate.confidence * 0.2
        factors['confidence_bonus'] = confidence_bonus
        
        # Calculate utility
        utility_score = (base_reward * 0.4 + 
                        content_quality * 0.3 + 
                        confidence_bonus * 0.3)
        
        # Risk-adjusted score
        risk_penalty = adjusted_risk * (1 - self.risk_tolerance)
        risk_adjusted_score = utility_score - risk_penalty
        
        factors['utility'] = utility_score
        factors['risk_penalty'] = risk_penalty
        
        return ActionEvaluation(
            candidate=candidate,
            utility_score=utility_score,
            risk_adjusted_score=risk_adjusted_score,
            factors=factors
        )
    
    def select(self, candidates: List[ActionCandidate],
               context: Dict[str, Any] = None) -> ActionSelection:
        """
        Select the best action from candidates.
        
        Args:
            candidates: List of action candidates
            context: Optional context
            
        Returns:
            ActionSelection with chosen action
        """
        self._total_selections += 1
        
        if not candidates:
            # Default action
            default = ActionCandidate(
                action_id="default",
                action_type=ActionType.CLARIFY,
                content="Could you please clarify your request?",
                confidence=0.5
            )
            return ActionSelection(
                selected_action=default,
                alternatives=[],
                selection_reason="No candidates provided",
                confidence=0.3
            )
        
        # Evaluate all candidates
        evaluations = [self.evaluate(c, context) for c in candidates]
        
        # Sort by risk-adjusted score
        evaluations.sort(key=lambda e: e.risk_adjusted_score, reverse=True)
        
        best = evaluations[0]
        alternatives = [e.candidate for e in evaluations[1:4]]  # Top 3 alternatives
        
        # Determine selection reason
        if best.risk_adjusted_score > 0.7:
            reason = "High utility with acceptable risk"
        elif best.factors.get('risk_penalty', 0) > 0.2:
            reason = "Best balance of utility and risk"
        else:
            reason = "Highest overall score"
        
        return ActionSelection(
            selected_action=best.candidate,
            alternatives=alternatives,
            selection_reason=reason,
            confidence=best.risk_adjusted_score
        )
    
    # ===== Learning Interface Methods =====
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        """Record selection outcome for learning."""
        self._outcomes.append(outcome)
        
        # Track performance by action type
        action_type = outcome.get('action_type')
        success = outcome.get('success', 0.5)
        
        if action_type:
            at = ActionType(action_type) if isinstance(action_type, str) else action_type
            if at not in self._action_type_performance:
                self._action_type_performance[at] = []
            self._action_type_performance[at].append(success)
    
    def record_feedback(self, feedback: Dict) -> None:
        """Record feedback on action selection."""
        self._feedback.append(feedback)
        domain = feedback.get('domain', 'general')
        if feedback.get('wrong_action', False):
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) - 0.05
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt selection thresholds."""
        pass
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get domain adjustment."""
        return self._domain_adjustments.get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics."""
        action_perf = {}
        for at, scores in self._action_type_performance.items():
            if scores:
                action_perf[at.value] = sum(scores) / len(scores)
        
        return {
            'total_selections': self._total_selections,
            'action_type_performance': action_perf,
            'risk_tolerance': self.risk_tolerance,
            'domain_adjustments': dict(self._domain_adjustments),
            'outcomes_recorded': len(self._outcomes),
            'feedback_recorded': len(self._feedback)
        }


    # =========================================================================
    # Learning Interface Completion
    # =========================================================================
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback to adjust behavior."""
        if not hasattr(self, '_learning_params'):
            self._learning_params = {}
        was_correct = feedback.get('was_correct', True)
        if not was_correct:
            adjustment = feedback.get('adjustment', 0.05)
            for key in self._learning_params:
                self._learning_params[key] *= (1 - adjustment)
    
    def get_statistics(self) -> Dict:
        """Return learning statistics."""
        return {
            'learning_params': getattr(self, '_learning_params', {}),
            'history_size': len(getattr(self, '_outcomes', []))
        }
    
    def serialize_state(self) -> Dict:
        """Serialize state for persistence."""
        return {
            'outcomes': getattr(self, '_outcomes', [])[-100:],
            'learning_params': getattr(self, '_learning_params', {}),
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore state from serialized data."""
        self._outcomes = state.get('outcomes', [])
        self._learning_params = state.get('learning_params', {})

if __name__ == "__main__":
    selector = ActionSelector()
    
    candidates = [
        ActionCandidate("a1", ActionType.RESPOND, "Here is a detailed answer...", 0.7, 0.2, 0.8),
        ActionCandidate("a2", ActionType.CLARIFY, "Could you specify what you mean?", 0.5, 0.1, 0.9),
        ActionCandidate("a3", ActionType.REFUSE, "I cannot help with this.", 0.3, 0.4, 0.6),
    ]
    
    print("=" * 60)
    print("ACTION SELECTOR TEST")
    print("=" * 60)
    
    result = selector.select(candidates)
    
    print(f"\nSelected: {result.selected_action.action_type.value}")
    print(f"Reason: {result.selection_reason}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Alternatives: {[a.action_type.value for a in result.alternatives]}")
    print(f"\nLearning stats: {selector.get_learning_statistics()}")

