"""
BASE Theory of Mind (NOVEL-14)

Implements theory of mind capabilities:
1. Intent inference - what does the user really want?
2. Belief modeling - what does the user believe?
3. Emotional state detection
4. Communication gap identification

Patent Alignment:
- NOVEL-14: Theory of Mind
- Brain Layer: 3 (Limbic System)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum


class IntentType(Enum):
    """Types of user intent."""
    INFORMATION_SEEKING = "information_seeking"
    TASK_COMPLETION = "task_completion"
    EMOTIONAL_SUPPORT = "emotional_support"
    VALIDATION = "validation"
    EXPLORATION = "exploration"
    CHALLENGE = "challenge"
    CLARIFICATION = "clarification"


class EmotionalState(Enum):
    """Detected emotional states."""
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    FRUSTRATED = "frustrated"
    ANXIOUS = "anxious"
    CURIOUS = "curious"
    CONFUSED = "confused"


@dataclass
class MentalModel:
    """Model of user's mental state."""
    inferred_intent: IntentType
    intent_confidence: float
    emotional_state: EmotionalState
    emotion_confidence: float
    beliefs: List[str]
    knowledge_gaps: List[str]
    communication_preferences: Dict[str, Any]


@dataclass
class TheoryOfMindResult:
    """Result of theory of mind analysis."""
    mental_model: MentalModel
    response_recommendations: List[str]
    potential_misunderstandings: List[str]
    rapport_strategies: List[str]
    confidence: float


class TheoryOfMind:
    """
    Theory of Mind module for BASE.
    
    Implements NOVEL-14: Theory of Mind
    Brain Layer: 3 (Limbic System)
    
    Capabilities:
    1. Infer user intent beyond literal meaning
    2. Model user beliefs and knowledge
    3. Detect emotional state
    4. Identify potential communication gaps
    5. Recommend appropriate response strategies
    """
    
    # Intent indicators
    INTENT_PATTERNS = {
        IntentType.INFORMATION_SEEKING: ['what is', 'how does', 'why', 'explain', 'tell me'],
        IntentType.TASK_COMPLETION: ['help me', 'can you', 'please', 'need to', 'want to'],
        IntentType.EMOTIONAL_SUPPORT: ['feeling', 'worried', 'stressed', 'scared', 'upset'],
        IntentType.VALIDATION: ['am i right', 'correct', 'agree', 'think so', 'makes sense'],
        IntentType.EXPLORATION: ['what if', 'suppose', 'imagine', 'possibilities', 'alternatives'],
        IntentType.CHALLENGE: ['prove', 'wrong', 'disagree', 'but', 'actually'],
        IntentType.CLARIFICATION: ['mean', 'understand', 'confused', 'unclear', 'rephrase'],
    }
    
    # Emotion indicators
    EMOTION_PATTERNS = {
        EmotionalState.POSITIVE: ['great', 'thanks', 'love', 'excited', 'happy', 'wonderful'],
        EmotionalState.NEGATIVE: ['hate', 'terrible', 'bad', 'awful', 'disappointing'],
        EmotionalState.FRUSTRATED: ['frustrated', 'annoying', 'again', 'still', 'why won\'t'],
        EmotionalState.ANXIOUS: ['worried', 'nervous', 'scared', 'anxious', 'uncertain'],
        EmotionalState.CURIOUS: ['interesting', 'curious', 'wonder', 'fascinating'],
        EmotionalState.CONFUSED: ['confused', 'don\'t understand', 'unclear', 'lost'],
    }
    
    def __init__(self):
        """Initialize Theory of Mind module."""
        # Learning state
        self._outcomes: List[Dict] = []
        self._feedback: List[Dict] = []
        self._domain_adjustments: Dict[str, float] = {}
        self._total_analyses: int = 0
        self._intent_accuracy: List[bool] = []
        self._emotion_accuracy: List[bool] = []
    
    def analyze(self, query: str, response: str = None, 
                context: Dict[str, Any] = None) -> TheoryOfMindResult:
        """
        Analyze query to understand user's mental state.
        
        Args:
            query: User's query
            response: Optional response to analyze
            context: Optional context (history, domain, etc.)
            
        Returns:
            TheoryOfMindResult with mental model
        """
        self._total_analyses += 1
        query_lower = query.lower()
        context = context or {}
        
        # Infer intent
        intent, intent_conf = self._infer_intent(query_lower)
        
        # Detect emotional state
        emotion, emotion_conf = self._detect_emotion(query_lower)
        
        # Infer beliefs
        beliefs = self._infer_beliefs(query_lower, context)
        
        # Identify knowledge gaps
        gaps = self._identify_knowledge_gaps(query_lower)
        
        # Determine communication preferences
        prefs = self._infer_communication_preferences(query_lower, intent, emotion)
        
        mental_model = MentalModel(
            inferred_intent=intent,
            intent_confidence=intent_conf,
            emotional_state=emotion,
            emotion_confidence=emotion_conf,
            beliefs=beliefs,
            knowledge_gaps=gaps,
            communication_preferences=prefs
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(mental_model)
        misunderstandings = self._identify_potential_misunderstandings(query_lower, mental_model)
        rapport = self._suggest_rapport_strategies(mental_model)
        
        # Overall confidence
        confidence = (intent_conf + emotion_conf) / 2
        
        return TheoryOfMindResult(
            mental_model=mental_model,
            response_recommendations=recommendations,
            potential_misunderstandings=misunderstandings,
            rapport_strategies=rapport,
            confidence=confidence
        )
    
    def _infer_intent(self, query: str) -> tuple:
        """Infer user's intent."""
        best_intent = IntentType.INFORMATION_SEEKING
        best_confidence = 0.3
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            matches = sum(1 for p in patterns if p in query)
            confidence = min(0.9, 0.3 + matches * 0.2)
            if confidence > best_confidence:
                best_intent = intent
                best_confidence = confidence
        
        return best_intent, best_confidence
    
    def _detect_emotion(self, query: str) -> tuple:
        """Detect emotional state."""
        best_emotion = EmotionalState.NEUTRAL
        best_confidence = 0.5
        
        for emotion, patterns in self.EMOTION_PATTERNS.items():
            matches = sum(1 for p in patterns if p in query)
            if matches > 0:
                confidence = min(0.9, 0.4 + matches * 0.2)
                if confidence > best_confidence:
                    best_emotion = emotion
                    best_confidence = confidence
        
        return best_emotion, best_confidence
    
    def _infer_beliefs(self, query: str, context: Dict) -> List[str]:
        """Infer user beliefs from query."""
        beliefs = []
        
        # Look for assumption indicators
        if 'i think' in query or 'i believe' in query:
            beliefs.append("User has pre-existing beliefs about the topic")
        if 'always' in query or 'never' in query:
            beliefs.append("User may hold absolute beliefs")
        if '?' in query and len(query.split('?')) > 2:
            beliefs.append("User has multiple questions, may be uncertain")
        
        return beliefs
    
    def _identify_knowledge_gaps(self, query: str) -> List[str]:
        """Identify what the user might not know."""
        gaps = []
        
        if 'what is' in query or 'what are' in query:
            gaps.append("May lack basic understanding of the topic")
        if 'how do' in query or 'how can' in query:
            gaps.append("May lack procedural knowledge")
        if 'why' in query:
            gaps.append("May lack understanding of reasoning/causation")
        
        return gaps
    
    def _infer_communication_preferences(self, query: str, intent: IntentType,
                                         emotion: EmotionalState) -> Dict[str, Any]:
        """Infer preferred communication style."""
        prefs = {
            'detail_level': 'moderate',
            'tone': 'professional',
            'structure': 'organized'
        }
        
        if intent == IntentType.EMOTIONAL_SUPPORT:
            prefs['tone'] = 'empathetic'
        elif intent == IntentType.TASK_COMPLETION:
            prefs['structure'] = 'step-by-step'
        
        if emotion in [EmotionalState.FRUSTRATED, EmotionalState.ANXIOUS]:
            prefs['tone'] = 'reassuring'
        
        return prefs
    
    def _generate_recommendations(self, model: MentalModel) -> List[str]:
        """Generate response recommendations."""
        recs = []
        
        if model.emotional_state == EmotionalState.FRUSTRATED:
            recs.append("Acknowledge frustration before providing solution")
        if model.emotional_state == EmotionalState.CONFUSED:
            recs.append("Use simpler explanations and examples")
        if model.inferred_intent == IntentType.VALIDATION:
            recs.append("Provide balanced perspective, not just agreement")
        if model.knowledge_gaps:
            recs.append(f"Address knowledge gap: {model.knowledge_gaps[0]}")
        
        return recs
    
    def _identify_potential_misunderstandings(self, query: str, 
                                               model: MentalModel) -> List[str]:
        """Identify potential communication gaps."""
        misunderstandings = []
        
        if model.beliefs and 'absolute beliefs' in str(model.beliefs):
            misunderstandings.append("User may resist nuanced answers")
        if model.intent_confidence < 0.5:
            misunderstandings.append("Intent unclear - consider asking for clarification")
        
        return misunderstandings
    
    def _suggest_rapport_strategies(self, model: MentalModel) -> List[str]:
        """Suggest strategies to build rapport."""
        strategies = []
        
        if model.emotional_state != EmotionalState.NEUTRAL:
            strategies.append(f"Acknowledge {model.emotional_state.value} state")
        strategies.append("Mirror communication style in response")
        if model.inferred_intent == IntentType.EXPLORATION:
            strategies.append("Encourage exploration with open-ended follow-ups")
        
        return strategies
    
    # ===== Learning Interface Methods =====
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        """Record analysis outcome for learning."""
        self._outcomes.append(outcome)
        if 'intent_correct' in outcome:
            self._intent_accuracy.append(outcome['intent_correct'])
        if 'emotion_correct' in outcome:
            self._emotion_accuracy.append(outcome['emotion_correct'])
    
    def record_feedback(self, feedback: Dict) -> None:
        """Record feedback on theory of mind analysis."""
        self._feedback.append(feedback)
        domain = feedback.get('domain', 'general')
        if feedback.get('misread_intent', False):
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) - 0.05
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt thresholds based on performance."""
        pass
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get domain adjustment."""
        return self._domain_adjustments.get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics."""
        intent_acc = sum(self._intent_accuracy) / max(len(self._intent_accuracy), 1)
        emotion_acc = sum(self._emotion_accuracy) / max(len(self._emotion_accuracy), 1)
        
        return {
            'total_analyses': self._total_analyses,
            'intent_accuracy': intent_acc,
            'emotion_accuracy': emotion_acc,
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
    tom = TheoryOfMind()
    
    test_query = "I'm really frustrated. I've tried everything but nothing works. Can you help me?"
    result = tom.analyze(test_query)
    
    print("=" * 60)
    print("THEORY OF MIND TEST")
    print("=" * 60)
    print(f"Query: {test_query[:50]}...")
    print(f"\nInferred Intent: {result.mental_model.inferred_intent.value}")
    print(f"Intent Confidence: {result.mental_model.intent_confidence:.2f}")
    print(f"Emotional State: {result.mental_model.emotional_state.value}")
    print(f"\nRecommendations: {result.response_recommendations}")
    print(f"Rapport Strategies: {result.rapport_strategies}")
    print(f"\nLearning stats: {tom.get_learning_statistics()}")

