"""
BAIS Research Modules - Advanced Cognitive Governance

These modules represent advanced R&D capabilities that extend BAIS
beyond standard governance into cognitive AI territory.

Modules:
1. Theory of Mind - Mental state inference, perspective taking
2. Neuro-Symbolic - Logical reasoning with formal verification
3. World Models - Causal reasoning and forward prediction
4. Creative Reasoning - Divergent thinking and novel solution generation

All modules integrate with:
- Query Router (selection criteria)
- Module Registry (availability tracking)
- Shared Services (learning, storage, audit)

Status: R&D / Experimental
"""

from research.theory_of_mind import TheoryOfMindModule, MentalStateAnalysis
from research.neurosymbolic import NeuroSymbolicModule, LogicalVerification
from research.world_models import WorldModelsModule, CausalPrediction
from research.creative_reasoning import CreativeReasoningModule, DivergentAnalysis

__all__ = [
    'TheoryOfMindModule',
    'MentalStateAnalysis',
    'NeuroSymbolicModule',
    'LogicalVerification',
    'WorldModelsModule',
    'CausalPrediction',
    'CreativeReasoningModule',
    'DivergentAnalysis',
]

# Module Selection Criteria (for Query Router integration)
MODULE_TRIGGERS = {
    'theory_of_mind': {
        'query_patterns': [
            r'\b(why did|why would|what was .* thinking|perspective|intent|motive)\b',
            r'\b(feel|believe|want|expect|assume|understand)\b',
            r'\b(persuade|manipulate|deceive|mislead)\b',
            r'\b(social|emotional|relationship|conflict)\b',
        ],
        'response_patterns': [
            r'\b(they might feel|could believe|probably thinks)\b',
            r'\b(from .* perspective|in .* view|according to)\b',
        ],
        'domains': ['social', 'psychological', 'communication', 'hr'],
        'query_types': ['analytical', 'social'],
        'description': 'Activated when understanding mental states, intentions, or perspectives is needed'
    },
    
    'neurosymbolic': {
        'query_patterns': [
            r'\b(prove|verify|valid|logical|if .* then|implies|therefore)\b',
            r'\b(constraint|satisfy|requirement|condition|rule)\b',
            r'\b(always|never|all|none|every|must)\b',
            r'\b(contradiction|inconsistent|consistent)\b',
        ],
        'response_patterns': [
            r'\b(follows that|we can conclude|logically)\b',
            r'\b(necessary|sufficient|required)\b',
        ],
        'domains': ['legal', 'technical', 'mathematical', 'compliance'],
        'query_types': ['analytical', 'verification'],
        'description': 'Activated for formal logic verification, constraint satisfaction, and proof checking'
    },
    
    'world_models': {
        'query_patterns': [
            r'\b(what if|what would happen|predict|forecast|future|scenario)\b',
            r'\b(cause|effect|lead to|result in|impact|consequence)\b',
            r'\b(simulate|model|plan|strategy)\b',
            r'\b(risk|probability|likelihood|chance)\b',
        ],
        'response_patterns': [
            r'\b(would likely|could lead to|may result in)\b',
            r'\b(in .* scenario|if .* then)\b',
        ],
        'domains': ['financial', 'strategic', 'scientific', 'policy'],
        'query_types': ['analytical', 'predictive', 'planning'],
        'description': 'Activated for causal reasoning, predictions, and counterfactual analysis'
    },
    
    'creative_reasoning': {
        'query_patterns': [
            r'\b(creative|innovative|novel|unique|original|new idea)\b',
            r'\b(brainstorm|generate|imagine|invent|design)\b',
            r'\b(like|similar to|analogy|metaphor|comparison)\b',
            r'\b(alternative|different way|outside the box)\b',
        ],
        'response_patterns': [
            r'\b(could also|another approach|alternatively)\b',
            r'\b(like .* but|similar to|analogous)\b',
        ],
        'domains': ['creative', 'marketing', 'design', 'innovation'],
        'query_types': ['creative', 'generative', 'exploratory'],
        'description': 'Activated for divergent thinking, analogical reasoning, and creative generation'
    },
}






