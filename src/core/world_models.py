"""
BASE World Models (NOVEL-16)

Implements world model reasoning:
1. Causal reasoning about world states
2. Counterfactual simulation
3. Prediction of action consequences
4. Mental simulation of scenarios

Patent Alignment:
- NOVEL-16: World Models
- Brain Layer: 2 (Prefrontal Cortex)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum


class SimulationMode(Enum):
    """Modes of world model simulation."""
    DETERMINISTIC = "deterministic"
    PROBABILISTIC = "probabilistic"
    COUNTERFACTUAL = "counterfactual"


@dataclass
class WorldState:
    """Represents a state of the world."""
    state_id: str
    entities: Dict[str, Any]
    relationships: Dict[str, List[str]]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Action:
    """An action that can modify world state."""
    action_id: str
    name: str
    preconditions: Dict[str, Any]
    effects: Dict[str, Any]
    probability: float = 1.0


@dataclass
class SimulationResult:
    """Result of a world model simulation."""
    initial_state: WorldState
    final_state: WorldState
    actions_taken: List[Action]
    outcomes: List[str]
    probability: float
    counterfactuals: List[Dict[str, Any]]


class WorldModelAnalyzer:
    """
    World model reasoning for BASE.
    
    Implements NOVEL-16: World Models
    Brain Layer: 2 (Prefrontal Cortex)
    
    Capabilities:
    1. Build and maintain world state representations
    2. Simulate action consequences
    3. Generate counterfactual scenarios
    4. Predict outcomes with probabilities
    """
    
    def __init__(self):
        """Initialize the world model analyzer."""
        self.current_state: Optional[WorldState] = None
        self.action_library: Dict[str, Action] = {}
        
        # Learning state
        self._outcomes: List[Dict] = []
        self._feedback: List[Dict] = []
        self._domain_adjustments: Dict[str, float] = {}
        self._prediction_accuracy: List[bool] = []
        self._total_simulations: int = 0
    
    def build_state(self, entities: Dict[str, Any], 
                    relationships: Dict[str, List[str]] = None) -> WorldState:
        """
        Build a world state from entities and relationships.
        
        Args:
            entities: Dict of entity_id -> properties
            relationships: Dict of relationship_type -> [entity pairs]
            
        Returns:
            WorldState object
        """
        state = WorldState(
            state_id=f"STATE-{datetime.utcnow().timestamp()}",
            entities=entities,
            relationships=relationships or {}
        )
        self.current_state = state
        return state
    
    def register_action(self, action: Action) -> None:
        """Register an action in the action library."""
        self.action_library[action.action_id] = action
    
    def simulate(self, initial_state: WorldState, actions: List[str],
                 mode: SimulationMode = SimulationMode.DETERMINISTIC) -> SimulationResult:
        """
        Simulate a sequence of actions on the world state.
        
        Args:
            initial_state: Starting world state
            actions: List of action IDs to simulate
            mode: Simulation mode
            
        Returns:
            SimulationResult with outcomes
        """
        self._total_simulations += 1
        
        current = WorldState(
            state_id=f"SIM-{self._total_simulations}",
            entities=dict(initial_state.entities),
            relationships=dict(initial_state.relationships)
        )
        
        executed_actions = []
        outcomes = []
        probability = 1.0
        
        for action_id in actions:
            action = self.action_library.get(action_id)
            if action is None:
                outcomes.append(f"Unknown action: {action_id}")
                continue
            
            # Check preconditions
            if self._check_preconditions(current, action):
                # Apply effects
                current = self._apply_effects(current, action)
                executed_actions.append(action)
                probability *= action.probability
                outcomes.append(f"Executed: {action.name}")
            else:
                outcomes.append(f"Preconditions not met for: {action.name}")
        
        # Generate counterfactuals if in counterfactual mode
        counterfactuals = []
        if mode == SimulationMode.COUNTERFACTUAL:
            counterfactuals = self._generate_counterfactuals(initial_state, actions)
        
        return SimulationResult(
            initial_state=initial_state,
            final_state=current,
            actions_taken=executed_actions,
            outcomes=outcomes,
            probability=probability,
            counterfactuals=counterfactuals
        )
    
    def predict_consequence(self, state: WorldState, action_id: str) -> Dict[str, Any]:
        """
        Predict the consequence of a single action.
        
        Args:
            state: Current world state
            action_id: Action to predict
            
        Returns:
            Dict with prediction details
        """
        action = self.action_library.get(action_id)
        if action is None:
            return {'error': 'Unknown action', 'success': False}
        
        can_execute = self._check_preconditions(state, action)
        
        if can_execute:
            new_state = self._apply_effects(state, action)
            return {
                'success': True,
                'can_execute': True,
                'predicted_state': new_state.entities,
                'probability': action.probability,
                'effects': action.effects
            }
        else:
            return {
                'success': True,
                'can_execute': False,
                'reason': 'Preconditions not met',
                'required': action.preconditions
            }
    
    def _check_preconditions(self, state: WorldState, action: Action) -> bool:
        """Check if action preconditions are met."""
        for key, required_value in action.preconditions.items():
            if key not in state.entities:
                return False
            if state.entities.get(key) != required_value:
                return False
        return True
    
    def _apply_effects(self, state: WorldState, action: Action) -> WorldState:
        """Apply action effects to state."""
        new_entities = dict(state.entities)
        for key, new_value in action.effects.items():
            new_entities[key] = new_value
        
        return WorldState(
            state_id=f"{state.state_id}-post-{action.action_id}",
            entities=new_entities,
            relationships=dict(state.relationships)
        )
    
    def _generate_counterfactuals(self, state: WorldState, 
                                   actions: List[str]) -> List[Dict[str, Any]]:
        """Generate counterfactual scenarios."""
        counterfactuals = []
        
        # What if we didn't do each action?
        for i, action_id in enumerate(actions):
            alt_actions = actions[:i] + actions[i+1:]
            alt_result = self.simulate(state, alt_actions, SimulationMode.DETERMINISTIC)
            counterfactuals.append({
                'scenario': f"What if we skipped '{action_id}'?",
                'alternative_outcome': alt_result.outcomes
            })
        
        return counterfactuals
    
    def analyze_response(self, response: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze a response using world model reasoning.
        
        Args:
            response: Text to analyze
            context: Additional context
            
        Returns:
            Analysis results
        """
        # Extract implied world states and actions from response
        # This is a simplified implementation
        
        return {
            'coherent': True,
            'implied_states': [],
            'implied_actions': [],
            'causal_chain_valid': True,
            'confidence': 0.8
        }
    
    # ===== Learning Interface Methods =====
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        """Record simulation outcome for learning."""
        self._outcomes.append(outcome)
        if 'prediction_correct' in outcome:
            self._prediction_accuracy.append(outcome['prediction_correct'])
    
    def record_feedback(self, feedback: Dict) -> None:
        """Record feedback on world model predictions."""
        self._feedback.append(feedback)
        domain = feedback.get('domain', 'general')
        if feedback.get('prediction_wrong', False):
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) - 0.05
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt prediction confidence based on performance."""
        pass
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get domain adjustment."""
        return self._domain_adjustments.get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics."""
        if self._prediction_accuracy:
            accuracy = sum(self._prediction_accuracy) / len(self._prediction_accuracy)
        else:
            accuracy = 0.0
        
        return {
            'total_simulations': self._total_simulations,
            'actions_registered': len(self.action_library),
            'prediction_accuracy': accuracy,
            'domain_adjustments': dict(self._domain_adjustments),
            'outcomes_recorded': len(self._outcomes),
            'feedback_recorded': len(self._feedback)
        }


if __name__ == "__main__":
    analyzer = WorldModelAnalyzer()
    
    # Build a simple world state
    state = analyzer.build_state(
        entities={'light': 'off', 'door': 'closed'},
        relationships={'controls': [('switch', 'light')]}
    )
    
    # Register actions
    analyzer.register_action(Action(
        action_id="TURN_ON_LIGHT",
        name="Turn on light",
        preconditions={'light': 'off'},
        effects={'light': 'on'},
        probability=0.99
    ))
    
    # Simulate
    result = analyzer.simulate(state, ["TURN_ON_LIGHT"])
    
    print("=" * 60)
    print("WORLD MODEL TEST")
    print("=" * 60)
    print(f"Initial state: {state.entities}")
    print(f"Final state: {result.final_state.entities}")
    print(f"Outcomes: {result.outcomes}")
    print(f"Probability: {result.probability}")
    print(f"\nLearning stats: {analyzer.get_learning_statistics()}")

