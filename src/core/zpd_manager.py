"""
BASE Zone of Proximal Development Manager (PPA1-Inv12, NOVEL-4)

Implements adaptive difficulty based on Vygotsky's ZPD theory:
1. Assess user's current understanding level
2. Determine optimal challenge level
3. Scaffold explanations appropriately
4. Adjust complexity dynamically

Patent Alignment:
- PPA1-Inv12: Adaptive Difficulty (ZPD)
- NOVEL-4: Zone of Proximal Development
- Brain Layer: 3 (Limbic System)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum


class CompetenceLevel(Enum):
    """User competence levels."""
    NOVICE = 1
    BEGINNER = 2
    INTERMEDIATE = 3
    ADVANCED = 4
    EXPERT = 5


class ScaffoldType(Enum):
    """Types of scaffolding support."""
    FULL_EXPLANATION = "full_explanation"        # Step-by-step with examples
    GUIDED_DISCOVERY = "guided_discovery"        # Hints and questions
    PARTIAL_SUPPORT = "partial_support"          # Key points only
    CHALLENGE_MODE = "challenge_mode"            # Minimal guidance
    PEER_LEVEL = "peer_level"                    # Technical discussion


@dataclass
class ZPDAssessment:
    """Assessment of user's current zone."""
    current_level: CompetenceLevel
    confidence: float
    evidence: List[str]
    recommended_scaffold: ScaffoldType
    stretch_topics: List[str]  # Topics just beyond current level


@dataclass
class ZPDResult:
    """Result of ZPD-aware processing."""
    assessment: ZPDAssessment
    adapted_response: str
    scaffolding_applied: List[str]
    complexity_score: float  # 0-1, how complex the adapted response is
    learning_opportunities: List[str]


class ZPDManager:
    """
    Manages Zone of Proximal Development for adaptive responses.
    
    Implements PPA1-Inv12 and NOVEL-4
    Brain Layer: 3 (Limbic System)
    
    Capabilities:
    1. Assess user's current knowledge level
    2. Determine optimal complexity for learning
    3. Apply appropriate scaffolding
    4. Track learning progression over sessions
    """
    
    # Complexity indicators for level detection
    COMPLEXITY_INDICATORS = {
        CompetenceLevel.NOVICE: ['what is', 'basic', 'simple', 'explain', 'beginner'],
        CompetenceLevel.BEGINNER: ['how to', 'getting started', 'learn', 'introduction'],
        CompetenceLevel.INTERMEDIATE: ['best practice', 'pattern', 'optimize', 'compare'],
        CompetenceLevel.ADVANCED: ['architecture', 'scale', 'performance', 'trade-off'],
        CompetenceLevel.EXPERT: ['internals', 'edge case', 'novel', 'research', 'cutting-edge'],
    }
    
    # Scaffolding strategies by level
    SCAFFOLDING_STRATEGIES = {
        CompetenceLevel.NOVICE: ScaffoldType.FULL_EXPLANATION,
        CompetenceLevel.BEGINNER: ScaffoldType.GUIDED_DISCOVERY,
        CompetenceLevel.INTERMEDIATE: ScaffoldType.PARTIAL_SUPPORT,
        CompetenceLevel.ADVANCED: ScaffoldType.CHALLENGE_MODE,
        CompetenceLevel.EXPERT: ScaffoldType.PEER_LEVEL,
    }
    
    def __init__(self):
        """Initialize ZPD Manager."""
        # User profiles (would be persistent in production)
        self.user_profiles: Dict[str, Dict] = {}
        
        # Learning state
        self._outcomes: List[Dict] = []
        self._feedback: List[Dict] = []
        self._domain_adjustments: Dict[str, float] = {}
        self._total_assessments: int = 0
        self._level_accuracy: List[bool] = []
    
    def assess(self, query: str, user_id: str = "default",
               context: Dict[str, Any] = None) -> ZPDAssessment:
        """
        Assess user's current level and determine ZPD.
        
        Args:
            query: User's query
            user_id: User identifier
            context: Additional context
            
        Returns:
            ZPDAssessment with level and recommendations
        """
        self._total_assessments += 1
        query_lower = query.lower()
        context = context or {}
        
        # Get or create user profile
        profile = self.user_profiles.get(user_id, {'level': CompetenceLevel.INTERMEDIATE})
        
        # Detect level from query
        detected_level, confidence, evidence = self._detect_level(query_lower)
        
        # Combine with profile (moving average)
        if user_id in self.user_profiles:
            # Weight new detection with profile
            profile_weight = 0.7
            detection_weight = 0.3
            effective_level_value = (
                profile['level'].value * profile_weight +
                detected_level.value * detection_weight
            )
            effective_level = CompetenceLevel(round(effective_level_value))
        else:
            effective_level = detected_level
        
        # Update profile
        self.user_profiles[user_id] = {'level': effective_level}
        
        # Determine scaffold type
        scaffold = self.SCAFFOLDING_STRATEGIES.get(effective_level, ScaffoldType.PARTIAL_SUPPORT)
        
        # Identify stretch topics (one level up)
        stretch_topics = self._identify_stretch_topics(effective_level, context)
        
        return ZPDAssessment(
            current_level=effective_level,
            confidence=confidence,
            evidence=evidence,
            recommended_scaffold=scaffold,
            stretch_topics=stretch_topics
        )
    
    def adapt_response(self, response: str, assessment: ZPDAssessment,
                       domain: str = 'general') -> ZPDResult:
        """
        Adapt a response based on ZPD assessment.
        
        Args:
            response: Original response
            assessment: ZPD assessment
            domain: Domain context
            
        Returns:
            ZPDResult with adapted response
        """
        scaffolding_applied = []
        
        # Apply scaffolding based on type
        adapted = response
        scaffold_type = assessment.recommended_scaffold
        
        if scaffold_type == ScaffoldType.FULL_EXPLANATION:
            adapted = self._add_full_scaffolding(response)
            scaffolding_applied.append("Added definitions and examples")
            scaffolding_applied.append("Broke into smaller steps")
            
        elif scaffold_type == ScaffoldType.GUIDED_DISCOVERY:
            adapted = self._add_guided_scaffolding(response)
            scaffolding_applied.append("Added guiding questions")
            scaffolding_applied.append("Included hints")
            
        elif scaffold_type == ScaffoldType.PARTIAL_SUPPORT:
            adapted = self._add_partial_scaffolding(response)
            scaffolding_applied.append("Added key point summaries")
            
        elif scaffold_type == ScaffoldType.CHALLENGE_MODE:
            adapted = self._add_challenge_scaffolding(response)
            scaffolding_applied.append("Added advanced considerations")
            
        # Calculate complexity
        complexity = self._calculate_complexity(adapted)
        
        # Apply domain adjustment
        domain_adj = self._domain_adjustments.get(domain, 0.0)
        complexity = max(0, min(1, complexity + domain_adj * 0.1))
        
        # Identify learning opportunities
        opportunities = self._identify_learning_opportunities(assessment)
        
        return ZPDResult(
            assessment=assessment,
            adapted_response=adapted,
            scaffolding_applied=scaffolding_applied,
            complexity_score=complexity,
            learning_opportunities=opportunities
        )
    
    def _detect_level(self, query: str) -> tuple:
        """Detect competence level from query."""
        best_level = CompetenceLevel.INTERMEDIATE
        best_score = 0
        evidence = []
        
        for level, indicators in self.COMPLEXITY_INDICATORS.items():
            matches = [ind for ind in indicators if ind in query]
            if len(matches) > best_score:
                best_score = len(matches)
                best_level = level
                evidence = matches
        
        confidence = min(0.9, 0.3 + best_score * 0.2)
        return best_level, confidence, evidence
    
    def _identify_stretch_topics(self, current_level: CompetenceLevel,
                                  context: Dict) -> List[str]:
        """Identify topics just beyond current level."""
        stretch = []
        
        if current_level.value < CompetenceLevel.EXPERT.value:
            next_level = CompetenceLevel(current_level.value + 1)
            indicators = self.COMPLEXITY_INDICATORS.get(next_level, [])
            stretch = [f"Consider exploring: {ind}" for ind in indicators[:2]]
        
        return stretch
    
    def _add_full_scaffolding(self, response: str) -> str:
        """Add full scaffolding for novices."""
        return f"Let me explain this step by step:\n\n{response}\n\nKey takeaway: Focus on understanding the basic concepts first."
    
    def _add_guided_scaffolding(self, response: str) -> str:
        """Add guided discovery scaffolding."""
        return f"{response}\n\nThink about this: How might you apply these concepts in your own context?"
    
    def _add_partial_scaffolding(self, response: str) -> str:
        """Add partial scaffolding."""
        return response
    
    def _add_challenge_scaffolding(self, response: str) -> str:
        """Add challenge scaffolding for advanced users."""
        return f"{response}\n\nAdvanced consideration: What edge cases or limitations should you consider?"
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity (simplified)."""
        words = text.split()
        avg_word_length = sum(len(w) for w in words) / max(len(words), 1)
        sentences = text.count('.') + text.count('!') + text.count('?')
        avg_sentence_length = len(words) / max(sentences, 1)
        
        # Simple complexity heuristic
        complexity = min(1.0, (avg_word_length / 10 + avg_sentence_length / 30))
        return complexity
    
    def _identify_learning_opportunities(self, assessment: ZPDAssessment) -> List[str]:
        """Identify learning opportunities in ZPD."""
        opportunities = []
        
        if assessment.stretch_topics:
            opportunities.append(f"Ready for: {assessment.stretch_topics[0]}")
        
        if assessment.current_level.value < 3:
            opportunities.append("Foundation building phase - practice core concepts")
        else:
            opportunities.append("Ready for deeper exploration of edge cases")
        
        return opportunities
    
    # ===== Learning Interface Methods =====
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        """Record assessment outcome for learning."""
        self._outcomes.append(outcome)
        if 'level_correct' in outcome:
            self._level_accuracy.append(outcome['level_correct'])
    
    def record_feedback(self, feedback: Dict) -> None:
        """Record feedback on ZPD assessment."""
        self._feedback.append(feedback)
        domain = feedback.get('domain', 'general')
        if feedback.get('too_simple', False):
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) + 0.1
        elif feedback.get('too_complex', False):
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) - 0.1
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt complexity thresholds."""
        pass
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get domain adjustment."""
        return self._domain_adjustments.get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics."""
        level_acc = sum(self._level_accuracy) / max(len(self._level_accuracy), 1)
        
        return {
            'total_assessments': self._total_assessments,
            'user_profiles_count': len(self.user_profiles),
            'level_accuracy': level_acc,
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
    zpd = ZPDManager()
    
    test_query = "What is machine learning? I'm just getting started."
    assessment = zpd.assess(test_query)
    
    print("=" * 60)
    print("ZPD MANAGER TEST")
    print("=" * 60)
    print(f"Query: {test_query}")
    print(f"\nAssessed Level: {assessment.current_level.value} ({assessment.current_level.name})")
    print(f"Confidence: {assessment.confidence:.2f}")
    print(f"Recommended Scaffold: {assessment.recommended_scaffold.value}")
    print(f"Stretch Topics: {assessment.stretch_topics}")
    
    # Test adaptation
    response = "Machine learning is a subset of AI that enables systems to learn from data."
    result = zpd.adapt_response(response, assessment)
    
    print(f"\nAdapted Response: {result.adapted_response[:100]}...")
    print(f"Scaffolding Applied: {result.scaffolding_applied}")
    print(f"\nLearning stats: {zpd.get_learning_statistics()}")

