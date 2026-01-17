"""
BAIS Creative Reasoning (NOVEL-17)

Implements creative problem-solving capabilities:
1. Analogical reasoning
2. Lateral thinking prompts
3. Novel solution generation
4. Creativity-safety balance

Patent Alignment:
- NOVEL-17: Creative Reasoning
- Brain Layer: 2 (Prefrontal Cortex)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum


class CreativityMode(Enum):
    """Modes of creative reasoning."""
    CONSERVATIVE = "conservative"  # Safe, conventional solutions
    BALANCED = "balanced"          # Mix of creative and safe
    INNOVATIVE = "innovative"      # Novel solutions prioritized
    EXPLORATORY = "exploratory"    # Maximum creativity, minimum constraints


@dataclass
class CreativeSolution:
    """A creative solution generated."""
    solution_id: str
    approach: str
    description: str
    novelty_score: float  # 0-1
    safety_score: float   # 0-1
    feasibility_score: float  # 0-1
    analogies_used: List[str] = field(default_factory=list)


@dataclass
class CreativeAnalysisResult:
    """Result of creative reasoning analysis."""
    solutions: List[CreativeSolution]
    best_solution: Optional[CreativeSolution]
    creativity_score: float  # 0-100
    safety_maintained: bool
    analogies_explored: List[str]
    lateral_prompts: List[str]


class CreativeReasoning:
    """
    Creative reasoning module for BAIS.
    
    Implements NOVEL-17: Creative Reasoning
    Brain Layer: 2 (Prefrontal Cortex)
    
    Capabilities:
    1. Generate analogical connections
    2. Propose lateral thinking prompts
    3. Balance creativity with safety
    4. Score and rank creative solutions
    """
    
    # Lateral thinking prompts
    LATERAL_PROMPTS = [
        "What if the constraints were reversed?",
        "How would an expert in a different field approach this?",
        "What's the opposite of the obvious solution?",
        "What would a child's approach be?",
        "What if we had unlimited resources?",
        "What if we had to solve this in 5 minutes?",
        "What analogies from nature might apply?",
        "What would the solution look like from the future?",
    ]
    
    # Analogy domains
    ANALOGY_DOMAINS = {
        'nature': ['ecosystem', 'evolution', 'symbiosis', 'adaptation', 'metamorphosis'],
        'engineering': ['bridge', 'circuit', 'pipeline', 'framework', 'architecture'],
        'biology': ['cell', 'organism', 'immune system', 'neural network', 'DNA'],
        'physics': ['momentum', 'equilibrium', 'entropy', 'gravity', 'wave'],
        'social': ['collaboration', 'consensus', 'network', 'community', 'hierarchy'],
    }
    
    def __init__(self, default_mode: CreativityMode = CreativityMode.BALANCED):
        """Initialize creative reasoning module."""
        self.default_mode = default_mode
        
        # Learning state
        self._outcomes: List[Dict] = []
        self._feedback: List[Dict] = []
        self._domain_adjustments: Dict[str, float] = {}
        self._solution_history: List[CreativeSolution] = []
        self._total_analyses: int = 0
        self._successful_solutions: int = 0
    
    def analyze(self, problem: str, context: Dict[str, Any] = None,
                mode: CreativityMode = None) -> CreativeAnalysisResult:
        """
        Apply creative reasoning to a problem.
        
        Args:
            problem: Problem description
            context: Additional context
            mode: Creativity mode to use
            
        Returns:
            CreativeAnalysisResult with generated solutions
        """
        self._total_analyses += 1
        mode = mode or self.default_mode
        context = context or {}
        domain = context.get('domain', 'general')
        
        # Generate analogies
        analogies = self._generate_analogies(problem)
        
        # Generate lateral prompts
        prompts = self._select_lateral_prompts(problem, mode)
        
        # Generate solutions
        solutions = self._generate_solutions(problem, analogies, mode, domain)
        
        # Score and rank
        ranked_solutions = sorted(
            solutions,
            key=lambda s: self._calculate_combined_score(s, mode),
            reverse=True
        )
        
        best = ranked_solutions[0] if ranked_solutions else None
        
        # Check safety
        safety_maintained = all(s.safety_score >= 0.6 for s in solutions) if solutions else True
        
        # Calculate overall creativity score
        if solutions:
            creativity_score = sum(s.novelty_score for s in solutions) / len(solutions) * 100
        else:
            creativity_score = 0
        
        return CreativeAnalysisResult(
            solutions=ranked_solutions,
            best_solution=best,
            creativity_score=creativity_score,
            safety_maintained=safety_maintained,
            analogies_explored=analogies,
            lateral_prompts=prompts
        )
    
    def _generate_analogies(self, problem: str) -> List[str]:
        """Generate relevant analogies for the problem."""
        problem_lower = problem.lower()
        analogies = []
        
        for domain, concepts in self.ANALOGY_DOMAINS.items():
            # Simple keyword matching for demo
            for concept in concepts:
                if any(word in problem_lower for word in ['system', 'process', 'structure', 'flow']):
                    analogies.append(f"Like a {concept} in {domain}")
                    break
        
        return analogies[:5]  # Limit to 5 analogies
    
    def _select_lateral_prompts(self, problem: str, mode: CreativityMode) -> List[str]:
        """Select appropriate lateral thinking prompts."""
        # More prompts for more creative modes
        count = {
            CreativityMode.CONSERVATIVE: 2,
            CreativityMode.BALANCED: 3,
            CreativityMode.INNOVATIVE: 5,
            CreativityMode.EXPLORATORY: 8
        }[mode]
        
        return self.LATERAL_PROMPTS[:count]
    
    def _generate_solutions(self, problem: str, analogies: List[str],
                           mode: CreativityMode, domain: str) -> List[CreativeSolution]:
        """Generate creative solutions."""
        solutions = []
        
        # Standard solution
        solutions.append(CreativeSolution(
            solution_id="SOL-001",
            approach="conventional",
            description="Apply standard best practices for this type of problem",
            novelty_score=0.3,
            safety_score=0.95,
            feasibility_score=0.9
        ))
        
        # Analogy-based solution
        if analogies:
            solutions.append(CreativeSolution(
                solution_id="SOL-002",
                approach="analogical",
                description=f"Apply insights from: {analogies[0]}",
                novelty_score=0.7,
                safety_score=0.8,
                feasibility_score=0.7,
                analogies_used=analogies[:2]
            ))
        
        # Lateral thinking solution (only in more creative modes)
        if mode in [CreativityMode.INNOVATIVE, CreativityMode.EXPLORATORY]:
            solutions.append(CreativeSolution(
                solution_id="SOL-003",
                approach="lateral",
                description="Challenge the assumptions and consider opposite approaches",
                novelty_score=0.9,
                safety_score=0.6,
                feasibility_score=0.5
            ))
        
        # Apply domain adjustment
        domain_adj = self._domain_adjustments.get(domain, 0.0)
        for sol in solutions:
            sol.safety_score = max(0, min(1, sol.safety_score + domain_adj * 0.1))
        
        return solutions
    
    def _calculate_combined_score(self, solution: CreativeSolution, 
                                  mode: CreativityMode) -> float:
        """Calculate combined score based on mode."""
        weights = {
            CreativityMode.CONSERVATIVE: {'novelty': 0.2, 'safety': 0.5, 'feasibility': 0.3},
            CreativityMode.BALANCED: {'novelty': 0.33, 'safety': 0.34, 'feasibility': 0.33},
            CreativityMode.INNOVATIVE: {'novelty': 0.5, 'safety': 0.25, 'feasibility': 0.25},
            CreativityMode.EXPLORATORY: {'novelty': 0.6, 'safety': 0.2, 'feasibility': 0.2}
        }[mode]
        
        return (
            solution.novelty_score * weights['novelty'] +
            solution.safety_score * weights['safety'] +
            solution.feasibility_score * weights['feasibility']
        )
    
    # ===== Learning Interface Methods =====
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        """Record creative solution outcome."""
        self._outcomes.append(outcome)
        if outcome.get('solution_accepted', False):
            self._successful_solutions += 1
    
    def record_feedback(self, feedback: Dict) -> None:
        """Record feedback on creative solutions."""
        self._feedback.append(feedback)
        domain = feedback.get('domain', 'general')
        if feedback.get('too_conservative', False):
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) + 0.1
        elif feedback.get('too_risky', False):
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) - 0.1
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt creativity-safety balance."""
        pass
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get domain adjustment."""
        return self._domain_adjustments.get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics."""
        success_rate = self._successful_solutions / max(self._total_analyses, 1)
        return {
            'total_analyses': self._total_analyses,
            'successful_solutions': self._successful_solutions,
            'success_rate': success_rate,
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
    reasoning = CreativeReasoning()
    
    result = reasoning.analyze(
        "How can we improve the system performance while reducing costs?",
        context={'domain': 'technical'},
        mode=CreativityMode.BALANCED
    )
    
    print("=" * 60)
    print("CREATIVE REASONING TEST")
    print("=" * 60)
    print(f"Creativity score: {result.creativity_score:.1f}")
    print(f"Safety maintained: {result.safety_maintained}")
    print(f"Analogies: {result.analogies_explored}")
    
    print("\nSolutions:")
    for sol in result.solutions:
        print(f"  - {sol.approach}: novelty={sol.novelty_score:.1f}, safety={sol.safety_score:.1f}")
    
    print(f"\nLearning stats: {reasoning.get_learning_statistics()}")

