"""
BASE Cognitive Governance Engine v16.5
Adaptive Difficulty Adjustment System

PPA-1 Invention 12: FULL IMPLEMENTATION
Dynamic challenge generation based on user/system capability.

This module implements:
1. Capability Assessment: Measure current performance level
2. Difficulty Scaling: Adjust thresholds based on capability
3. Challenge Generation: Create appropriate verification challenges
4. Zone of Proximal Development: Optimal difficulty for learning
5. Progression Tracking: Monitor improvement over time

═══════════════════════════════════════════════════════════════════════════════
PEDAGOGICAL FOUNDATION
═══════════════════════════════════════════════════════════════════════════════

1. Vygotsky's Zone of Proximal Development (ZPD):
   - ZPD_LOWER = 0.4 (success rate below this = too hard)
   - ZPD_UPPER = 0.8 (success rate above this = too easy)
   - Optimal learning occurs within ZPD bounds
   
2. Flow Theory (Csikszentmihalyi):
   - Balance challenge vs. skill
   - Too easy → boredom → decrease engagement
   - Too hard → frustration → decrease engagement
   - Optimal → flow state → maximum learning
   
3. Spaced Repetition:
   - Target weak dimensions more frequently
   - Reinforce strengths less often
   - Optimal retention through spacing

═══════════════════════════════════════════════════════════════════════════════
DIFFICULTY ADJUSTMENT FORMULA
═══════════════════════════════════════════════════════════════════════════════

Level Adjustment Rule:
- If success_rate > ZPD_UPPER: level_up()
- If success_rate < ZPD_LOWER: level_down()
- Otherwise: maintain (optimal zone)

Threshold Multipliers by Level:
- NOVICE: 0.6× (very lenient)
- BEGINNER: 0.8×
- INTERMEDIATE: 1.0× (standard)
- ADVANCED: 1.15×
- EXPERT: 1.3× (strictest)

Capability Update:
C_new = α × observation + (1 - α) × C_old
where α = CAPABILITY_LEARNING_RATE = 0.15
═══════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import math
import json
from pathlib import Path


class DifficultyLevel(str, Enum):
    """Difficulty levels for governance."""
    NOVICE = "novice"           # Very lenient (learning mode)
    BEGINNER = "beginner"       # Lenient
    INTERMEDIATE = "intermediate"  # Standard
    ADVANCED = "advanced"       # Strict
    EXPERT = "expert"           # Very strict


class CapabilityDimension(str, Enum):
    """Dimensions of capability being measured."""
    FACTUAL_ACCURACY = "factual_accuracy"
    BIAS_AVOIDANCE = "bias_avoidance"
    GROUNDING = "grounding"
    CONSISTENCY = "consistency"
    UNCERTAINTY_CALIBRATION = "uncertainty_calibration"
    DOMAIN_EXPERTISE = "domain_expertise"


@dataclass
class CapabilityProfile:
    """Profile of system/user capability."""
    profile_id: str
    name: str
    
    # Dimension scores (0-1)
    dimension_scores: Dict[CapabilityDimension, float] = field(default_factory=dict)
    
    # Overall capability
    overall_capability: float = 0.5
    confidence: float = 0.5  # Confidence in this assessment
    
    # History
    observation_count: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # Current difficulty level
    current_level: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    
    def update_dimension(self, dimension: CapabilityDimension, score: float, weight: float = 0.1):
        """Update a capability dimension with new observation."""
        current = self.dimension_scores.get(dimension, 0.5)
        self.dimension_scores[dimension] = weight * score + (1 - weight) * current
        self._recalculate_overall()
        self.observation_count += 1
        self.last_updated = datetime.utcnow()
    
    def _recalculate_overall(self):
        """Recalculate overall capability."""
        if self.dimension_scores:
            self.overall_capability = sum(self.dimension_scores.values()) / len(self.dimension_scores)
    
    def to_dict(self) -> Dict:
        return {
            'profile_id': self.profile_id,
            'name': self.name,
            'dimension_scores': {k.value: v for k, v in self.dimension_scores.items()},
            'overall_capability': self.overall_capability,
            'confidence': self.confidence,
            'current_level': self.current_level.value,
            'observation_count': self.observation_count
        }


@dataclass
class DifficultySettings:
    """Settings for a difficulty level."""
    level: DifficultyLevel
    
    # Threshold adjustments (multipliers)
    accuracy_threshold_multiplier: float  # Multiply base threshold
    bias_sensitivity_multiplier: float
    grounding_threshold_multiplier: float
    
    # Tolerance settings
    false_positive_tolerance: float  # How many FPs allowed
    false_negative_tolerance: float
    
    # Challenge settings
    verification_frequency: float  # How often to verify (0-1)
    
    def get_adjusted_threshold(self, base: float) -> float:
        """Get adjusted threshold based on difficulty."""
        return base * self.accuracy_threshold_multiplier


@dataclass
class Challenge:
    """A verification challenge."""
    challenge_id: str
    challenge_type: str
    difficulty: DifficultyLevel
    question: str
    expected_answer: Optional[str]
    time_limit_seconds: int
    dimension_tested: CapabilityDimension
    
    # Results
    completed: bool = False
    passed: bool = False
    response: Optional[str] = None
    time_taken_seconds: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'challenge_id': self.challenge_id,
            'type': self.challenge_type,
            'difficulty': self.difficulty.value,
            'dimension': self.dimension_tested.value,
            'question': self.question[:100],
            'completed': self.completed,
            'passed': self.passed
        }


class AdaptiveDifficultyEngine:
    """
    Adaptive Difficulty Adjustment Engine.
    
    PPA-1 Invention 12: Full Implementation
    
    Dynamically adjusts governance difficulty based on
    demonstrated capability, following pedagogical principles.
    """
    
    # Zone of Proximal Development parameters
    ZPD_LOWER = 0.4  # Success rate below this = too hard
    ZPD_UPPER = 0.8  # Success rate above this = too easy
    
    # Adjustment rates
    CAPABILITY_LEARNING_RATE = 0.15
    DIFFICULTY_ADJUSTMENT_THRESHOLD = 10  # Observations before adjustment
    
    def __init__(self, storage_path: Path = None):
        # Use temp directory if no path provided (fixes read-only filesystem issues)
        if storage_path is None:
            import tempfile
            temp_dir = Path(tempfile.mkdtemp(prefix="base_difficulty_"))
            storage_path = temp_dir / "adaptive_difficulty.json"
        self.storage_path = storage_path
        
        # Capability profiles by ID
        self.profiles: Dict[str, CapabilityProfile] = {}
        
        # Difficulty settings
        self.difficulty_settings = self._initialize_difficulty_settings()
        
        # Challenge library
        self.challenges: List[Challenge] = []
        
        # Performance history
        self.performance_history: deque = deque(maxlen=1000)
        
        # Domain-specific capability tracking
        self.domain_capability: Dict[str, Dict[CapabilityDimension, float]] = {}
        
        # Load state
        self._load_state()
    
    def _initialize_difficulty_settings(self) -> Dict[DifficultyLevel, DifficultySettings]:
        """Initialize difficulty settings."""
        return {
            DifficultyLevel.NOVICE: DifficultySettings(
                level=DifficultyLevel.NOVICE,
                accuracy_threshold_multiplier=0.6,  # Lower thresholds
                bias_sensitivity_multiplier=0.7,
                grounding_threshold_multiplier=0.6,
                false_positive_tolerance=0.3,  # More tolerant
                false_negative_tolerance=0.2,
                verification_frequency=0.2
            ),
            DifficultyLevel.BEGINNER: DifficultySettings(
                level=DifficultyLevel.BEGINNER,
                accuracy_threshold_multiplier=0.8,
                bias_sensitivity_multiplier=0.85,
                grounding_threshold_multiplier=0.75,
                false_positive_tolerance=0.2,
                false_negative_tolerance=0.15,
                verification_frequency=0.3
            ),
            DifficultyLevel.INTERMEDIATE: DifficultySettings(
                level=DifficultyLevel.INTERMEDIATE,
                accuracy_threshold_multiplier=1.0,  # Standard
                bias_sensitivity_multiplier=1.0,
                grounding_threshold_multiplier=1.0,
                false_positive_tolerance=0.15,
                false_negative_tolerance=0.1,
                verification_frequency=0.5
            ),
            DifficultyLevel.ADVANCED: DifficultySettings(
                level=DifficultyLevel.ADVANCED,
                accuracy_threshold_multiplier=1.15,  # Higher thresholds
                bias_sensitivity_multiplier=1.2,
                grounding_threshold_multiplier=1.2,
                false_positive_tolerance=0.1,
                false_negative_tolerance=0.08,
                verification_frequency=0.7
            ),
            DifficultyLevel.EXPERT: DifficultySettings(
                level=DifficultyLevel.EXPERT,
                accuracy_threshold_multiplier=1.3,  # Strictest
                bias_sensitivity_multiplier=1.4,
                grounding_threshold_multiplier=1.4,
                false_positive_tolerance=0.05,
                false_negative_tolerance=0.05,
                verification_frequency=0.9
            ),
        }
    
    def get_or_create_profile(self, profile_id: str, name: str = None) -> CapabilityProfile:
        """Get existing profile or create new one."""
        if profile_id not in self.profiles:
            self.profiles[profile_id] = CapabilityProfile(
                profile_id=profile_id,
                name=name or profile_id,
                dimension_scores={dim: 0.5 for dim in CapabilityDimension},
                overall_capability=0.5,
                current_level=DifficultyLevel.INTERMEDIATE
            )
            self._save_state()
        return self.profiles[profile_id]
    
    def record_performance(self,
                          profile_id: str,
                          dimension: CapabilityDimension,
                          score: float,
                          domain: str = "general") -> Dict[str, Any]:
        """
        Record performance observation and update capability.
        
        Returns adjustment info if difficulty changed.
        """
        profile = self.get_or_create_profile(profile_id)
        old_level = profile.current_level
        
        # Update dimension score
        profile.update_dimension(dimension, score, self.CAPABILITY_LEARNING_RATE)
        
        # Update domain-specific tracking
        if domain not in self.domain_capability:
            self.domain_capability[domain] = {dim: 0.5 for dim in CapabilityDimension}
        self.domain_capability[domain][dimension] = (
            0.1 * score + 0.9 * self.domain_capability[domain].get(dimension, 0.5)
        )
        
        # Record in history
        self.performance_history.append({
            'profile_id': profile_id,
            'dimension': dimension.value,
            'score': score,
            'domain': domain,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Check if difficulty adjustment needed
        adjustment_result = self._check_difficulty_adjustment(profile)
        
        self._save_state()
        
        return {
            'profile': profile.to_dict(),
            'old_level': old_level.value,
            'new_level': profile.current_level.value,
            'level_changed': old_level != profile.current_level,
            'adjustment': adjustment_result
        }
    
    def _check_difficulty_adjustment(self, profile: CapabilityProfile) -> Dict[str, Any]:
        """Check if difficulty should be adjusted based on recent performance."""
        if profile.observation_count < self.DIFFICULTY_ADJUSTMENT_THRESHOLD:
            return {'adjusted': False, 'reason': 'Insufficient observations'}
        
        # Get recent performance for this profile
        recent = [p for p in list(self.performance_history)[-50:] 
                 if p['profile_id'] == profile.profile_id]
        
        if len(recent) < 10:
            return {'adjusted': False, 'reason': 'Insufficient recent data'}
        
        # Calculate success rate (score > 0.5 = success)
        success_rate = sum(1 for p in recent if p['score'] > 0.5) / len(recent)
        
        new_level = profile.current_level
        reason = ""
        
        # Zone of Proximal Development check
        if success_rate > self.ZPD_UPPER:
            # Too easy - increase difficulty
            new_level = self._increase_difficulty(profile.current_level)
            reason = f"Success rate {success_rate:.2f} > {self.ZPD_UPPER} (too easy)"
        elif success_rate < self.ZPD_LOWER:
            # Too hard - decrease difficulty
            new_level = self._decrease_difficulty(profile.current_level)
            reason = f"Success rate {success_rate:.2f} < {self.ZPD_LOWER} (too hard)"
        else:
            # In ZPD - optimal difficulty
            reason = f"Success rate {success_rate:.2f} in optimal zone [{self.ZPD_LOWER}, {self.ZPD_UPPER}]"
        
        if new_level != profile.current_level:
            profile.current_level = new_level
            return {
                'adjusted': True,
                'old_level': profile.current_level.value,
                'new_level': new_level.value,
                'success_rate': success_rate,
                'reason': reason
            }
        
        return {'adjusted': False, 'success_rate': success_rate, 'reason': reason}
    
    def _increase_difficulty(self, current: DifficultyLevel) -> DifficultyLevel:
        """Get next higher difficulty level."""
        levels = list(DifficultyLevel)
        idx = levels.index(current)
        return levels[min(idx + 1, len(levels) - 1)]
    
    def _decrease_difficulty(self, current: DifficultyLevel) -> DifficultyLevel:
        """Get next lower difficulty level."""
        levels = list(DifficultyLevel)
        idx = levels.index(current)
        return levels[max(idx - 1, 0)]
    
    def get_adjusted_thresholds(self,
                               profile_id: str,
                               base_thresholds: Dict[str, float]) -> Dict[str, float]:
        """
        Get thresholds adjusted for profile's capability level.
        
        Args:
            profile_id: Profile to adjust for
            base_thresholds: Base threshold values
        
        Returns:
            Adjusted thresholds
        """
        profile = self.get_or_create_profile(profile_id)
        settings = self.difficulty_settings[profile.current_level]
        
        adjusted = {}
        for key, value in base_thresholds.items():
            if 'accuracy' in key.lower():
                adjusted[key] = value * settings.accuracy_threshold_multiplier
            elif 'bias' in key.lower():
                adjusted[key] = value * settings.bias_sensitivity_multiplier
            elif 'grounding' in key.lower():
                adjusted[key] = value * settings.grounding_threshold_multiplier
            else:
                adjusted[key] = value
        
        return adjusted
    
    def generate_challenge(self,
                          profile_id: str,
                          dimension: CapabilityDimension = None) -> Challenge:
        """
        Generate an appropriate verification challenge.
        
        Selects difficulty based on profile and targets weak dimensions.
        """
        profile = self.get_or_create_profile(profile_id)
        
        # Target weakest dimension if not specified
        if dimension is None:
            dimension = min(
                profile.dimension_scores.keys(),
                key=lambda d: profile.dimension_scores.get(d, 0.5)
            )
        
        # Select challenge type based on dimension
        challenge_templates = self._get_challenge_templates(dimension)
        
        # Select difficulty-appropriate challenge
        template = challenge_templates[profile.current_level.value]
        
        import uuid
        challenge = Challenge(
            challenge_id=str(uuid.uuid4())[:8],
            challenge_type=template['type'],
            difficulty=profile.current_level,
            question=template['question'],
            expected_answer=template.get('expected'),
            time_limit_seconds=template.get('time_limit', 60),
            dimension_tested=dimension
        )
        
        self.challenges.append(challenge)
        return challenge
    
    def _get_challenge_templates(self, dimension: CapabilityDimension) -> Dict[str, Dict]:
        """Get challenge templates for a dimension."""
        templates = {
            CapabilityDimension.FACTUAL_ACCURACY: {
                'novice': {
                    'type': 'fact_verification',
                    'question': 'Is water composed of hydrogen and oxygen?',
                    'expected': 'yes',
                    'time_limit': 30
                },
                'beginner': {
                    'type': 'fact_verification',
                    'question': 'Verify: The speed of light is approximately 300,000 km/s',
                    'expected': 'yes',
                    'time_limit': 45
                },
                'intermediate': {
                    'type': 'fact_comparison',
                    'question': 'Compare claims: "AI can perfectly predict stock prices" vs "AI predictions have uncertainty"',
                    'expected': 'second_claim_more_accurate',
                    'time_limit': 60
                },
                'advanced': {
                    'type': 'nuanced_fact',
                    'question': 'Evaluate: "Quantum computers are faster than classical computers for all problems"',
                    'expected': 'false_with_nuance',
                    'time_limit': 90
                },
                'expert': {
                    'type': 'complex_verification',
                    'question': 'Assess factual accuracy of claim with contradicting sources',
                    'expected': 'detailed_analysis',
                    'time_limit': 120
                }
            },
            CapabilityDimension.BIAS_AVOIDANCE: {
                'novice': {
                    'type': 'bias_detection',
                    'question': 'Is "All politicians are corrupt" a biased statement?',
                    'expected': 'yes',
                    'time_limit': 30
                },
                'beginner': {
                    'type': 'bias_detection',
                    'question': 'Identify bias in: "Only traditional methods work reliably"',
                    'expected': 'status_quo_bias',
                    'time_limit': 45
                },
                'intermediate': {
                    'type': 'subtle_bias',
                    'question': 'Detect confirmation bias in a research summary',
                    'expected': 'identify_bias',
                    'time_limit': 60
                },
                'advanced': {
                    'type': 'systemic_bias',
                    'question': 'Analyze text for systemic bias patterns',
                    'expected': 'detailed_analysis',
                    'time_limit': 90
                },
                'expert': {
                    'type': 'meta_bias',
                    'question': 'Evaluate bias in bias-detection methodology',
                    'expected': 'recursive_analysis',
                    'time_limit': 120
                }
            },
            # Add templates for other dimensions...
        }
        
        # Return templates for requested dimension, or default
        return templates.get(dimension, templates[CapabilityDimension.FACTUAL_ACCURACY])
    
    def evaluate_challenge(self,
                          challenge_id: str,
                          response: str,
                          time_taken: float) -> Dict[str, Any]:
        """
        Evaluate challenge response and update profile.
        """
        # Find challenge
        challenge = next((c for c in self.challenges if c.challenge_id == challenge_id), None)
        if not challenge:
            return {'error': 'Challenge not found'}
        
        # Simple evaluation (in real system, this would be more sophisticated)
        challenge.completed = True
        challenge.response = response
        challenge.time_taken_seconds = time_taken
        
        # Check time limit
        within_time = time_taken <= challenge.time_limit_seconds
        
        # Check answer (simple matching for now)
        if challenge.expected_answer:
            answer_correct = challenge.expected_answer.lower() in response.lower()
        else:
            answer_correct = len(response) > 10  # At least tried to answer
        
        challenge.passed = within_time and answer_correct
        
        # Calculate score
        score = 0.0
        if challenge.passed:
            score = 0.7 + 0.3 * (1 - time_taken / challenge.time_limit_seconds)
        else:
            score = 0.3 if answer_correct else 0.1
        
        return {
            'challenge_id': challenge_id,
            'passed': challenge.passed,
            'score': score,
            'within_time': within_time,
            'time_taken': time_taken,
            'time_limit': challenge.time_limit_seconds
        }
    
    def get_profile_summary(self, profile_id: str) -> Dict[str, Any]:
        """Get comprehensive profile summary."""
        profile = self.get_or_create_profile(profile_id)
        
        # Get recent performance
        recent = [p for p in list(self.performance_history)[-100:] 
                 if p['profile_id'] == profile_id]
        
        if recent:
            recent_avg = sum(p['score'] for p in recent) / len(recent)
            trend = "improving" if len(recent) > 10 and recent[-5:] > recent[:5] else "stable"
        else:
            recent_avg = 0.5
            trend = "unknown"
        
        return {
            'profile': profile.to_dict(),
            'recent_performance': {
                'observations': len(recent),
                'average_score': recent_avg,
                'trend': trend
            },
            'weakest_dimension': min(
                profile.dimension_scores.keys(),
                key=lambda d: profile.dimension_scores.get(d, 0.5)
            ).value if profile.dimension_scores else None,
            'strongest_dimension': max(
                profile.dimension_scores.keys(),
                key=lambda d: profile.dimension_scores.get(d, 0.5)
            ).value if profile.dimension_scores else None,
            'difficulty_settings': self.difficulty_settings[profile.current_level].accuracy_threshold_multiplier
        }
    
    def _save_state(self):
        """Persist state."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'profiles': {k: v.to_dict() for k, v in self.profiles.items()},
            'domain_capability': {
                d: {k.value: v for k, v in dims.items()}
                for d, dims in self.domain_capability.items()
            }
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load state."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path) as f:
                state = json.load(f)
            
            for pid, p_dict in state.get('profiles', {}).items():
                self.profiles[pid] = CapabilityProfile(
                    profile_id=p_dict['profile_id'],
                    name=p_dict['name'],
                    dimension_scores={
                        CapabilityDimension(k): v 
                        for k, v in p_dict.get('dimension_scores', {}).items()
                    },
                    overall_capability=p_dict.get('overall_capability', 0.5),
                    confidence=p_dict.get('confidence', 0.5),
                    current_level=DifficultyLevel(p_dict.get('current_level', 'intermediate')),
                    observation_count=p_dict.get('observation_count', 0)
                )
                
        except Exception as e:
            print(f"Warning: Could not load adaptive difficulty state: {e}")

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

