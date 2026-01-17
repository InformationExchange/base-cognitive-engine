"""
BASE Big5 Personality Traits Detector (PPA2-Big5)

Implements OCEAN personality trait detection:
1. Openness - creativity, curiosity, intellectual interests
2. Conscientiousness - organization, dependability, self-discipline
3. Extraversion - energy, positive emotions, sociability
4. Agreeableness - compassion, cooperation, trust
5. Neuroticism - emotional instability, anxiety, moodiness

Patent Alignment:
- PPA2-Big5: OCEAN Personality Traits
- Brain Layer: 3 (Limbic System)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum


class PersonalityTrait(Enum):
    """Big Five personality traits."""
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"


@dataclass
class TraitScore:
    """Score for a single personality trait."""
    trait: PersonalityTrait
    score: float  # 0-1, where 0.5 is neutral
    confidence: float
    evidence: List[str]


@dataclass
class Big5Result:
    """Result of Big5 personality analysis."""
    traits: Dict[PersonalityTrait, TraitScore]
    dominant_traits: List[PersonalityTrait]
    overall_profile: str
    bias_indicators: List[str]
    confidence: float


class Big5Detector:
    """
    Detects Big Five personality traits in text.
    
    Implements PPA2-Big5: OCEAN Personality Traits
    Brain Layer: 3 (Limbic System)
    
    This helps identify potential biases based on:
    - Author's personality tendencies
    - Target audience assumptions
    - Communication style biases
    """
    
    # Trait indicator patterns (simplified - production would use NLP models)
    TRAIT_INDICATORS = {
        PersonalityTrait.OPENNESS: {
            'high': ['creative', 'curious', 'imaginative', 'innovative', 'artistic', 'unconventional'],
            'low': ['practical', 'conventional', 'traditional', 'straightforward', 'concrete']
        },
        PersonalityTrait.CONSCIENTIOUSNESS: {
            'high': ['organized', 'careful', 'disciplined', 'responsible', 'thorough', 'reliable'],
            'low': ['flexible', 'spontaneous', 'careless', 'impulsive', 'disorganized']
        },
        PersonalityTrait.EXTRAVERSION: {
            'high': ['energetic', 'talkative', 'enthusiastic', 'outgoing', 'social', 'assertive'],
            'low': ['quiet', 'reserved', 'solitary', 'introspective', 'independent']
        },
        PersonalityTrait.AGREEABLENESS: {
            'high': ['helpful', 'trusting', 'cooperative', 'kind', 'compassionate', 'empathetic'],
            'low': ['competitive', 'skeptical', 'challenging', 'critical', 'detached']
        },
        PersonalityTrait.NEUROTICISM: {
            'high': ['anxious', 'worried', 'nervous', 'stressed', 'emotional', 'vulnerable'],
            'low': ['calm', 'stable', 'confident', 'secure', 'relaxed', 'resilient']
        }
    }
    
    def __init__(self):
        """Initialize the Big5 detector."""
        # Learning state
        self._outcomes: List[Dict] = []
        self._feedback: List[Dict] = []
        self._domain_adjustments: Dict[str, float] = {}
        self._total_analyses: int = 0
        self._trait_accuracy: Dict[str, List[bool]] = {}
    
    def detect(self, text: str, context: Dict[str, Any] = None) -> Big5Result:
        """
        Detect Big5 personality traits in text.
        
        Args:
            text: Text to analyze
            context: Optional context
            
        Returns:
            Big5Result with trait scores
        """
        self._total_analyses += 1
        text_lower = text.lower()
        
        traits = {}
        for trait in PersonalityTrait:
            score = self._analyze_trait(text_lower, trait)
            traits[trait] = score
        
        # Find dominant traits (significantly above 0.5)
        dominant = [t for t, s in traits.items() if s.score > 0.65]
        
        # Generate profile description
        profile = self._generate_profile(traits)
        
        # Identify bias indicators
        bias_indicators = self._identify_bias_indicators(traits)
        
        # Calculate overall confidence
        confidence = sum(s.confidence for s in traits.values()) / len(traits)
        
        return Big5Result(
            traits=traits,
            dominant_traits=dominant,
            overall_profile=profile,
            bias_indicators=bias_indicators,
            confidence=confidence
        )
    
    def _analyze_trait(self, text: str, trait: PersonalityTrait) -> TraitScore:
        """Analyze a single trait."""
        indicators = self.TRAIT_INDICATORS[trait]
        
        high_matches = []
        low_matches = []
        
        for word in indicators['high']:
            if word in text:
                high_matches.append(word)
        
        for word in indicators['low']:
            if word in text:
                low_matches.append(word)
        
        # Calculate score
        if high_matches and not low_matches:
            score = 0.5 + min(0.4, len(high_matches) * 0.1)
        elif low_matches and not high_matches:
            score = 0.5 - min(0.4, len(low_matches) * 0.1)
        elif high_matches and low_matches:
            diff = len(high_matches) - len(low_matches)
            score = 0.5 + min(0.3, max(-0.3, diff * 0.05))
        else:
            score = 0.5  # Neutral
        
        confidence = min(0.9, 0.3 + (len(high_matches) + len(low_matches)) * 0.1)
        evidence = high_matches + low_matches
        
        return TraitScore(
            trait=trait,
            score=score,
            confidence=confidence,
            evidence=evidence
        )
    
    def _generate_profile(self, traits: Dict[PersonalityTrait, TraitScore]) -> str:
        """Generate a profile description."""
        high_traits = [t.value for t, s in traits.items() if s.score > 0.6]
        low_traits = [t.value for t, s in traits.items() if s.score < 0.4]
        
        if high_traits:
            return f"High in: {', '.join(high_traits)}"
        elif low_traits:
            return f"Low in: {', '.join(low_traits)}"
        else:
            return "Balanced personality profile"
    
    def _identify_bias_indicators(self, traits: Dict[PersonalityTrait, TraitScore]) -> List[str]:
        """Identify potential biases based on traits."""
        indicators = []
        
        # High agreeableness might lead to sycophantic responses
        if traits[PersonalityTrait.AGREEABLENESS].score > 0.7:
            indicators.append("Potential sycophantic tendency (high agreeableness)")
        
        # Low openness might miss creative solutions
        if traits[PersonalityTrait.OPENNESS].score < 0.35:
            indicators.append("May favor conventional over innovative solutions")
        
        # High neuroticism might overstate risks
        if traits[PersonalityTrait.NEUROTICISM].score > 0.7:
            indicators.append("May overemphasize risks and negative outcomes")
        
        return indicators
    
    # ===== Learning Interface Methods =====
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        """Record detection outcome for learning."""
        self._outcomes.append(outcome)
        for trait, correct in outcome.get('trait_accuracy', {}).items():
            if trait not in self._trait_accuracy:
                self._trait_accuracy[trait] = []
            self._trait_accuracy[trait].append(correct)
    
    def record_feedback(self, feedback: Dict) -> None:
        """Record feedback on trait detection."""
        self._feedback.append(feedback)
        domain = feedback.get('domain', 'general')
        if feedback.get('detection_inaccurate', False):
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) - 0.05
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt detection thresholds."""
        pass
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get domain adjustment."""
        return self._domain_adjustments.get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics."""
        trait_stats = {}
        for trait, results in self._trait_accuracy.items():
            if results:
                trait_stats[trait] = sum(results) / len(results)
        
        return {
            'total_analyses': self._total_analyses,
            'trait_accuracy': trait_stats,
            'domain_adjustments': dict(self._domain_adjustments),
            'outcomes_recorded': len(self._outcomes),
            'feedback_recorded': len(self._feedback)
        }

    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback."""
        pass  # Implement specific learning logic


    def get_statistics(self) -> Dict:
        """Return statistics."""
        return {'outcomes': len(getattr(self, '_outcomes', []))}


    def serialize_state(self) -> Dict:
        """Serialize state."""
        return {'outcomes': getattr(self, '_outcomes', [])[-100:]}


    def deserialize_state(self, state: Dict) -> None:
        """Restore state."""
        self._outcomes = state.get('outcomes', [])


if __name__ == "__main__":
    detector = Big5Detector()
    
    test_text = """
    I think we should be careful and organized in our approach. 
    We need to be reliable and thorough. I'm a bit worried about the risks,
    but I'm open to creative solutions if they're practical.
    """
    
    result = detector.detect(test_text)
    
    print("=" * 60)
    print("BIG5 DETECTOR TEST")
    print("=" * 60)
    print(f"Profile: {result.overall_profile}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"\nDominant traits: {[t.value for t in result.dominant_traits]}")
    print(f"\nBias indicators: {result.bias_indicators}")
    print(f"\nLearning stats: {detector.get_learning_statistics()}")

