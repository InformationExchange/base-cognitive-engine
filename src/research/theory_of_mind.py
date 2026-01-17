"""
BAIS Theory of Mind Module
Advanced cognitive capability for mental state inference

This module enables BAIS to:
1. Infer mental states (beliefs, desires, intentions)
2. Model perspective-taking and empathy
3. Detect manipulation/persuasion attempts
4. Analyze social dynamics in AI responses

Patent Claims: R&D Invention 1
Proprietary IP - 100% owned by Invitas Inc.
"""

import re
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class MentalStateType(str, Enum):
    """Types of mental states that can be inferred."""
    BELIEF = "belief"           # What agent thinks is true
    DESIRE = "desire"           # What agent wants
    INTENTION = "intention"     # What agent plans to do
    EMOTION = "emotion"         # Emotional state
    KNOWLEDGE = "knowledge"     # What agent knows
    EXPECTATION = "expectation" # What agent expects


class PerspectiveType(str, Enum):
    """Types of perspectives that can be modeled."""
    FIRST_PERSON = "first_person"   # Speaker's own perspective
    SECOND_PERSON = "second_person" # Listener's perspective
    THIRD_PERSON = "third_person"   # Third party's perspective
    COLLECTIVE = "collective"       # Group perspective


class ManipulationRisk(str, Enum):
    """Risk levels for manipulation detection."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MentalState:
    """Represents an inferred mental state."""
    agent: str                      # Who holds this mental state
    state_type: MentalStateType
    content: str                    # The mental content (belief, desire, etc.)
    confidence: float               # Confidence in inference
    evidence: List[str]             # Supporting evidence
    source_text: str                # Text that led to inference


@dataclass
class PerspectiveAnalysis:
    """Analysis of perspectives in text."""
    perspective_type: PerspectiveType
    agent: str
    viewpoint_markers: List[str]    # Linguistic markers
    potential_biases: List[str]     # Identified biases
    empathy_indicators: float       # 0-1 score for empathetic language
    confidence: float


@dataclass
class ManipulationAnalysis:
    """Detection of manipulation/persuasion."""
    risk_level: ManipulationRisk
    risk_score: float               # 0-1 score
    techniques_detected: List[str]  # e.g., "appeal_to_authority", "false_dichotomy"
    vulnerable_points: List[str]    # What's being targeted
    mitigation_suggestions: List[str]
    evidence: List[str]


@dataclass
class MentalStateAnalysis:
    """Complete Theory of Mind analysis result."""
    # Inferred mental states
    inferred_states: List[MentalState]
    
    # Perspective analysis
    perspectives: List[PerspectiveAnalysis]
    primary_perspective: PerspectiveType
    
    # Manipulation detection
    manipulation: ManipulationAnalysis
    
    # Social dynamics
    social_roles_detected: List[str]
    relationship_indicators: Dict[str, str]
    
    # Overall scores
    theory_of_mind_score: float     # How well response models minds
    perspective_accuracy: float     # How accurate perspective-taking is
    social_awareness: float         # Social situation understanding
    
    # Meta
    processing_time_ms: float
    timestamp: float
    warnings: List[str] = field(default_factory=list)


class TheoryOfMindModule:
    """
    Theory of Mind cognitive module for BAIS.
    
    Implements:
    - Mental state inference (belief-desire-intention model)
    - Perspective-taking analysis
    - Manipulation/persuasion detection
    - Social dynamics modeling
    
    Integration:
    - Triggered by Query Router for social/psychological queries
    - Reports to Consistency Checker for cross-validation
    - Stores results in Outcome Memory for learning
    """
    
    # PHASE 1 FIX: Implicit emotion inference (ToM-A2)
    # Maps actions/situations to likely emotional states
    IMPLICIT_EMOTION_MAP = {
        # Negative situations -> negative emotions
        'resigned': ('emotion', 'frustrated/disappointed'),
        'quit': ('emotion', 'unhappy/dissatisfied'),
        'left': ('emotion', 'discontent'),
        'stormed out': ('emotion', 'angry'),
        'slammed': ('emotion', 'frustrated'),
        'passed over': ('emotion', 'disappointed/rejected'),
        'rejected': ('emotion', 'hurt/disappointed'),
        'fired': ('emotion', 'upset/shocked'),
        'demoted': ('emotion', 'humiliated/disappointed'),
        'failed': ('emotion', 'disappointed'),
        'lost': ('emotion', 'sad/dejected'),
        'cried': ('emotion', 'sad/distressed'),
        'sighed': ('emotion', 'frustrated/tired'),
        # Positive situations -> positive emotions
        'promoted': ('emotion', 'happy/proud'),
        'celebrated': ('emotion', 'joyful'),
        'smiled': ('emotion', 'happy/pleased'),
        'laughed': ('emotion', 'amused/happy'),
        'hugged': ('emotion', 'loving/grateful'),
        'succeeded': ('emotion', 'satisfied/proud'),
        'won': ('emotion', 'elated/triumphant'),
        'praised': ('emotion', 'pleased/proud'),
    }
    
    # Mental state linguistic markers - ENHANCED for better detection
    MENTAL_STATE_MARKERS = {
        MentalStateType.BELIEF: [
            r'\b(thinks?|believes?|knows?|understands?|realizes?|assumes?|supposes?)\b',
            r'\b(convinced|certain|sure|confident|considers?)\b',
            r'\b(opinion|view|perspective|stance|position)\b',
            r'\b(thought|thinking|believed|believing)\b',
            r'\b(imagines?|suspects?|guesses?|reckons?|figures?)\b',
            # BAIS-GUIDED: Hedged/political language
            r'\b(sources?\s+(?:say|suggest|indicate|report|claim))\b',
            r'\b(may\s+be\s+considering|might\s+be\s+(?:thinking|planning))\b',
            r'\b(close\s+to\s+the\s+matter|familiar\s+with)\b',
            r'\b(reportedly|allegedly|purportedly|supposedly)\b',
        ],
        MentalStateType.DESIRE: [
            r'\b(wants?|wishes?|hopes?|desires?|needs?|prefers?|likes?)\b',
            r'\b(goal|aim|objective|aspiration|ambition)\b',
            r'\b(wanted|wanting|wished|wishing|hoping)\b',
            r'\b(craves?|longs?|yearns?|seeks?)\b',
            r'\b(would like|would love|would prefer)\b',
        ],
        MentalStateType.INTENTION: [
            r'\b(plans?|intends?|going to|will|decides?|commits?)\b',
            r'\b(try|tries|trying|attempts?|seeks?|pursues?)\b',
            r'\b(planned|planning|intended|intending|decided)\b',
            r'\b(means to|aims to|set out to|determined to)\b',
        ],
        MentalStateType.EMOTION: [
            r'\b(feels?|felt|feeling|happy|sad|angry|afraid|surprised|disgusted)\b',
            r'\b(anxious|worried|excited|frustrated|confused|satisfied)\b',
            r'\b(delighted|thrilled|upset|annoyed|pleased|disappointed)\b',
            r'\b(emotional|emotionally|mood|moody)\b',
            r'\b(love|hate|fear|joy|sorrow|rage|calm|nervous)\b',
        ],
        MentalStateType.KNOWLEDGE: [
            r'\b(knows?|aware|recognizes?|familiar|understands?|learned)\b',
            r'\b(information|fact|truth|evidence|data)\b',
            r'\b(knew|knowing|understood|understanding)\b',
            r'\b(discovered|realizes?|comprehends?)\b',
        ],
        MentalStateType.EXPECTATION: [
            r'\b(expects?|anticipates?|predicts?|foresees?|count on)\b',
            r'\b(should|supposed to|likely|probably|presumably)\b',
            r'\b(expected|expecting|anticipated|anticipating)\b',
            r'\b(looks forward|looking forward|awaits?|awaiting)\b',
            r'\b(projected|forecasts?|envisions?)\b',
        ],
    }
    
    # Manipulation technique patterns - ENHANCED with subtle patterns
    MANIPULATION_PATTERNS = {
        'appeal_to_authority': [
            r'\b(expert|authority|study|research shows|according to)\b',
            r'\b(doctors?|scientists?|professors?|officials?)\b',
            r'\b(experts?\s+(?:say|agree|recommend|believe))\b',
            r'\b(studies?\s+(?:show|prove|confirm|indicate))\b',
            r'\b(research\s+(?:shows?|proves?|confirms?))\b',
            r'\b(top|leading|renowned)\s+(?:doctors?|scientists?|experts?|researchers?)\b',
            r'\bresearchers?\s+(?:say|agree|confirm|found|discovered)\b',
        ],
        'emotional_manipulation': [
            r'\b(must|have to|need to|crucial|urgent|critical|emergency)\b',
            r'\b(fear|danger|risk|threat|crisis)\b',
        ],
        'false_dichotomy': [
            r'\b(either|only two|only choice|no other option)\b',
            r'\b(us vs them|with us or against)\b',
        ],
        'social_proof': [
            r'\b(everyone|most people|majority|popular|trending)\b',
            r'\b(thousands|millions|widespread)\b',
            r'\b(people who understand|smart people|those who know)\b',
        ],
        'scarcity': [
            r'\b(limited|exclusive|rare|only .* left|running out)\b',
            r'\b(last chance|now or never|act fast)\b',
        ],
        'reciprocity': [
            r'\b(free|gift|bonus|favor|owe|return)\b',
            r'\b(gave you|did for you|done (?:so much )?for you)\b',
            r'\b(your turn|in return|pay (?:me|it) back)\b',
            r'\b(after (?:all|everything) I(?:\'ve)?|owe me)\b',
        ],
        'fear_loss': [
            r'\b(lose|losing|lost) (?:everything|it all|your|all)\b',
            r'\b(miss(?:ing)? out|left behind|too late)\b',
            r'\b(if you don\'?t|won\'?t|unless you)\b.*\b(now|today|immediately)\b',
            r'\b(regret|sorry|wish you had)\b',
        ],
        'unrealistic_claims': [
            r'\b(100%|guarantee[ds]?|always works|never fails|zero risk)\b',
            r'\b(instant|immediate|overnight) (?:results|success|improvement)\b',
            r'\b(no effort|effortless|easy money|get rich)\b',
            r'\b(proven|scientifically proven|clinically proven)\b',
        ],
        'gaslighting': [
            r'\b(never happened|you\'re imagining|overreacting|too sensitive)\b',
            r'\b(crazy|paranoid|confused)\b',
            # BAIS-GUIDED: Additional gaslighting patterns
            r'\b(I\s+never\s+said|you\s+must\s+be\s+(?:mis)?remembering)\b',
            r'\b(your\s+memory|write\s+(?:it|things)\s+down)\b',
            r'\b(you\s+always\s+(?:do|say)\s+this|this\s+is\s+why)\b',
            r'\b(nobody\s+else\s+(?:thinks|feels|sees))\b',
        ],
        # ADDED: Subtle manipulation patterns
        'flattery': [
            r'\b(smart enough|intelligent enough|wise enough|bright enough)\b',
            r'\b(you.* understand|you.* see|you.* know)\b',
            r'\b(someone like you|people like you)\b',
        ],
        'in_group_appeal': [
            r'\b(people who understand|those who get it|smart people)\b',
            r'\b(always agree|everyone who knows|real experts)\b',
            r'\b(us|we|our team|people like us)\b',
        ],
        # BAIS SELF-IMPROVEMENT: Added based on failure analysis
        'passive_aggression': [
            r'\b(it\'?s\s+fine|no,?\s+it\'?s\s+(?:fine|okay|ok))\b',
            r'\b(do\s+whatever\s+you\s+want|I\'?m\s+used\s+to\s+it)\b',
            r'\b(I\s+guess\s+(?:that\'?s|it\'?s)\s+fine)\b',
            r'\b(sure,?\s+(?:fine|whatever|okay))\b',
        ],
        'false_consensus': [
            r'\b(everyone knows|obviously|clearly|of course)\b',
            r'\b(no one disagrees|anyone can see)\b',
        ],
    }
    
    # Perspective markers - ENHANCED for better detection
    PERSPECTIVE_MARKERS = {
        PerspectiveType.FIRST_PERSON: [
            r'\b(I|we|my|our|me|us|myself|ourselves)\b',
            r'\b(I think|I believe|I feel|I want|in my view|in my opinion)\b',
        ],
        PerspectiveType.SECOND_PERSON: [
            r'\b(you|your|yours|yourself|yourselves)\b',
            r'\b(you think|you believe|you feel|you want)\b',
        ],
        PerspectiveType.THIRD_PERSON: [
            r'\b(he|she|they|it|his|her|their|him|them|himself|herself|themselves)\b',
            r'\b(the manager|the team|the user|the customer|the client)\b',
            r"from (\w+)'s perspective",
            r"from the (\w+)'s view",
            r"(\w+) thinks",
            r"(\w+) believes",
            # BAIS-GUIDED: Named entity perspectives
            r'\b(The\s+)?(\w+)\s+(?:believes?|thinks?|feels?|argues?|wants?|says?)\b',
            r'\b(\w+)\s+is\s+(?:caught|stuck|torn)\b',
            r'\b(\w+)\s+(?:are|is)\s+(?:concerned|worried|happy|angry|betrayed)\b',
            r"(\w+) feels",
        ],
        PerspectiveType.COLLECTIVE: [
            r'\b(everyone|everybody|all of us|the group|the team|together)\b',
            r'\b(collectively|as a group|as a team)\b',
        ],
    }
    
    # Explicit perspective phrases - ENHANCED for better perspective detection
    EXPLICIT_PERSPECTIVE_PHRASES = [
        # Third person perspectives - multiple formats
        (r"from (?:the )?(\w+)(?:'?s)? perspective", PerspectiveType.THIRD_PERSON),
        (r"from (?:the )?(\w+)(?:'?s)? view", PerspectiveType.THIRD_PERSON),
        (r"from (?:the )?(\w+)(?:'?s)? point of view", PerspectiveType.THIRD_PERSON),
        (r"(\w+) (?:thinks?|believes?|feels?|wants?|expects?)", PerspectiveType.THIRD_PERSON),
        (r"(?:the )?(\w+) (?:think|believe|say|argue)", PerspectiveType.THIRD_PERSON),
        # First person
        (r"in my (?:view|opinion|perspective)", PerspectiveType.FIRST_PERSON),
        (r"I (?:think|believe|feel)", PerspectiveType.FIRST_PERSON),
        # Second person
        (r"in your (?:view|opinion|perspective)", PerspectiveType.SECOND_PERSON),
        # Collective
        (r"we (?:think|believe|feel)", PerspectiveType.COLLECTIVE),
        # Role-based perspectives
        (r"(?:the )?(employees?|workers?|staff) (?:think|believe|feel)", PerspectiveType.COLLECTIVE),
        (r"(?:the )?(union|management|board) (?:thinks?|believes?)", PerspectiveType.COLLECTIVE),
    ]
    
    def __init__(self,
                 manipulation_threshold: float = 0.40,
                 min_confidence: float = 0.50):
        """
        Initialize Theory of Mind module.
        
        Args:
            manipulation_threshold: Threshold for flagging manipulation
            min_confidence: Minimum confidence for mental state inference
        """
        self.manipulation_threshold = manipulation_threshold
        self.min_confidence = min_confidence
    
    def analyze(self,
                query: str,
                response: str,
                context: Dict[str, Any] = None) -> MentalStateAnalysis:
        """
        Perform complete Theory of Mind analysis.
        
        Args:
            query: User query
            response: AI-generated response
            context: Additional context (conversation history, etc.)
        
        Returns:
            MentalStateAnalysis with complete analysis
        """
        start_time = time.time()
        context = context or {}
        warnings = []
        
        combined_text = f"{query} {response}"
        
        # 1. Infer mental states
        mental_states = self._infer_mental_states(response, context)
        
        # 2. Analyze perspectives
        perspectives = self._analyze_perspectives(response)
        primary_perspective = self._determine_primary_perspective(perspectives)
        
        # 3. Detect manipulation
        manipulation = self._detect_manipulation(response)
        if manipulation.risk_level in [ManipulationRisk.HIGH, ManipulationRisk.CRITICAL]:
            warnings.append(f"High manipulation risk detected: {manipulation.techniques_detected}")
        
        # 4. Analyze social dynamics
        social_roles = self._detect_social_roles(combined_text)
        relationships = self._analyze_relationships(combined_text)
        
        # 5. Compute overall scores
        tom_score = self._compute_theory_of_mind_score(mental_states, perspectives)
        perspective_accuracy = self._compute_perspective_accuracy(perspectives)
        social_awareness = self._compute_social_awareness(social_roles, relationships)
        
        processing_time = (time.time() - start_time) * 1000
        
        return MentalStateAnalysis(
            inferred_states=mental_states,
            perspectives=perspectives,
            primary_perspective=primary_perspective,
            manipulation=manipulation,
            social_roles_detected=social_roles,
            relationship_indicators=relationships,
            theory_of_mind_score=tom_score,
            perspective_accuracy=perspective_accuracy,
            social_awareness=social_awareness,
            processing_time_ms=processing_time,
            timestamp=time.time(),
            warnings=warnings
        )
    
    def _infer_mental_states(self,
                              text: str,
                              context: Dict[str, Any]) -> List[MentalState]:
        """Infer mental states from text - ENHANCED with implicit emotion detection."""
        states = []
        text_lower = text.lower()
        
        # Look for each type of mental state
        for state_type, patterns in self.MENTAL_STATE_MARKERS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text_lower):
                    # Extract surrounding context as content
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context_text = text[start:end]
                    
                    # Determine the agent (who holds this mental state)
                    agent = self._identify_agent(context_text, match.group())
                    
                    # Calculate confidence based on linguistic cues
                    confidence = self._calculate_state_confidence(
                        match.group(), context_text, state_type
                    )
                    
                    if confidence >= self.min_confidence:
                        state = MentalState(
                            agent=agent,
                            state_type=state_type,
                            content=context_text.strip(),
                            confidence=confidence,
                            evidence=[match.group()],
                            source_text=context_text
                        )
                        states.append(state)
        
        # PHASE 1 FIX: Infer implicit emotions from actions/situations
        implicit_states = self._infer_implicit_emotions(text, text_lower)
        states.extend(implicit_states)
        
        # Deduplicate similar states
        return self._deduplicate_states(states)[:10]  # Limit to 10
    
    def _infer_implicit_emotions(self, text: str, text_lower: str) -> List[MentalState]:
        """Infer emotions from implicit cues like actions and situations."""
        implicit_states = []
        
        for trigger, (state_type_str, emotion) in self.IMPLICIT_EMOTION_MAP.items():
            if trigger in text_lower:
                # Find surrounding context
                idx = text_lower.find(trigger)
                start = max(0, idx - 30)
                end = min(len(text), idx + len(trigger) + 30)
                context_text = text[start:end]
                
                # Try to identify the agent
                agent = self._identify_agent_from_context(text_lower, idx)
                
                state = MentalState(
                    agent=agent,
                    state_type=MentalStateType.EMOTION,
                    content=f"Implicit emotion from '{trigger}': {emotion}",
                    confidence=0.65,  # Moderate confidence for implicit
                    evidence=[trigger],
                    source_text=context_text
                )
                implicit_states.append(state)
        
        return implicit_states
    
    def _identify_agent_from_context(self, text_lower: str, position: int) -> str:
        """Try to identify the agent near a given position in text."""
        # Look for names or pronouns before the trigger
        before = text_lower[max(0, position-50):position]
        
        # Check for common patterns
        name_match = re.search(r'\b([A-Z][a-z]+)\b', before[-30:] if len(before) > 30 else before)
        if name_match:
            return name_match.group(1).lower()
        
        if ' she ' in before or ' her ' in before:
            return 'she'
        elif ' he ' in before or ' him ' in before:
            return 'he'
        elif ' they ' in before or ' them ' in before:
            return 'they'
        elif ' i ' in before or " i'" in before:
            return 'speaker'
        
        return 'inferred_agent'
    
    def _identify_agent(self, context: str, marker: str) -> str:
        """Identify who holds the mental state."""
        context_lower = context.lower()
        
        # Check for explicit subjects
        if re.search(r'\bi\s+' + marker, context_lower):
            return "speaker"
        elif re.search(r'\byou\s+' + marker, context_lower):
            return "addressee"
        elif re.search(r'\b(he|she|they|it)\s+' + marker, context_lower):
            return "third_party"
        elif re.search(r'\bwe\s+' + marker, context_lower):
            return "collective"
        
        # Default based on context
        return "inferred_agent"
    
    def _calculate_state_confidence(self,
                                     marker: str,
                                     context: str,
                                     state_type: MentalStateType) -> float:
        """Calculate confidence in mental state inference."""
        confidence = 0.6  # Base confidence
        
        # Increase for explicit markers
        explicit_markers = ['clearly', 'definitely', 'certainly', 'obviously']
        if any(m in context.lower() for m in explicit_markers):
            confidence += 0.15
        
        # Decrease for hedging
        hedge_markers = ['maybe', 'perhaps', 'might', 'could', 'possibly']
        if any(m in context.lower() for m in hedge_markers):
            confidence -= 0.15
        
        # Increase for repeated evidence
        if context.lower().count(marker) > 1:
            confidence += 0.1
        
        return min(max(confidence, 0.0), 1.0)
    
    def _deduplicate_states(self, states: List[MentalState]) -> List[MentalState]:
        """Remove duplicate or very similar mental states."""
        unique = []
        seen_content = set()
        
        for state in sorted(states, key=lambda s: s.confidence, reverse=True):
            content_key = f"{state.agent}:{state.state_type.value}:{state.content[:50]}"
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique.append(state)
        
        return unique
    
    def _analyze_perspectives(self, text: str) -> List[PerspectiveAnalysis]:
        """Analyze perspectives present in text - ENHANCED."""
        perspectives = []
        text_lower = text.lower()
        perspective_counts = {pt: 0 for pt in PerspectiveType}
        perspective_markers = {pt: [] for pt in PerspectiveType}
        
        # BAIS-GUIDED: Track unique named entities for multi-party detection
        named_entities = set()
        entity_patterns = [
            r'\b(The\s+)?(\w+)\s+(?:believes?|thinks?|feels?|argues?|wants?|says?)\b',
            r'\b(\w+)\s+is\s+(?:caught|stuck|torn)\b',
            r'\b(\w+)\s+(?:are|is)\s+(?:concerned|worried|happy|angry|betrayed)\b',
        ]
        for pattern in entity_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Get the entity name (handle tuple results)
                if isinstance(match, tuple):
                    entity = match[-1] if match[-1] else match[0]
                else:
                    entity = match
                if entity and len(entity) > 1 and entity.lower() not in ['the', 'a', 'an']:
                    named_entities.add(entity.strip())
        
        # First, check explicit perspective phrases (higher quality detection)
        for pattern, perspective_type in self.EXPLICIT_PERSPECTIVE_PHRASES:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                perspective_counts[perspective_type] += len(matches) * 2  # Weight explicit phrases higher
                if isinstance(matches[0], str):
                    perspective_markers[perspective_type].extend(matches)
        
        # Then check standard markers
        for perspective_type, patterns in self.PERSPECTIVE_MARKERS.items():
            markers = []
            for pattern in patterns:
                found = re.findall(pattern, text_lower, re.IGNORECASE)
                if found:
                    markers.extend(found if isinstance(found[0], str) else [m for m in found if m])
            
            if markers:
                perspective_counts[perspective_type] += len(markers)
                perspective_markers[perspective_type].extend(markers)
        
        # BAIS-GUIDED: Create separate perspectives for each named entity
        if len(named_entities) > 1:
            empathy = self._calculate_empathy_score(text)
            for entity in list(named_entities)[:6]:  # Limit to 6 entities
                analysis = PerspectiveAnalysis(
                    perspective_type=PerspectiveType.THIRD_PERSON,
                    agent=entity,
                    viewpoint_markers=[entity],
                    potential_biases=[],
                    empathy_indicators=empathy,
                    confidence=0.7
                )
                perspectives.append(analysis)
        else:
            # Fallback to original behavior for non-multi-party text
            for perspective_type in PerspectiveType:
                count = perspective_counts[perspective_type]
                markers = perspective_markers[perspective_type]
                
                if count > 0:
                    empathy = self._calculate_empathy_score(text)
                    biases = self._identify_perspective_biases(text, perspective_type)
                    meaningful_markers = [m for m in markers if isinstance(m, str) and len(m) > 1]
                    
                    analysis = PerspectiveAnalysis(
                        perspective_type=perspective_type,
                        agent=perspective_type.value,
                        viewpoint_markers=list(set(meaningful_markers))[:5],
                        potential_biases=biases,
                        empathy_indicators=empathy,
                        confidence=min(count / 5, 1.0)
                    )
                    perspectives.append(analysis)
        
        return perspectives
    
    def _calculate_empathy_score(self, text: str) -> float:
        """Calculate empathy indicators in text."""
        empathy_markers = [
            r'\b(understand|empathize|feel|appreciate|recognize)\b',
            r'\b(from .* perspective|in .* shoes|see .* point)\b',
            r'\b(sorry|apologize|sympathize|compassion)\b',
        ]
        
        text_lower = text.lower()
        score = 0.0
        
        for pattern in empathy_markers:
            if re.search(pattern, text_lower):
                score += 0.2
        
        return min(score, 1.0)
    
    def _identify_perspective_biases(self,
                                      text: str,
                                      perspective: PerspectiveType) -> List[str]:
        """Identify potential biases in perspective."""
        biases = []
        text_lower = text.lower()
        
        bias_patterns = {
            'confirmation_bias': r'\b(confirms|supports|proves|obviously)\b',
            'authority_bias': r'\b(expert|official|authority)\b',
            'in_group_bias': r'\b(we|us|our|people like us)\b',
            'negativity_bias': r'\b(problem|issue|concern|risk|danger)\b',
            'optimism_bias': r'\b(will work|definitely|certain|guaranteed)\b',
        }
        
        for bias_name, pattern in bias_patterns.items():
            if re.search(pattern, text_lower):
                biases.append(bias_name)
        
        return biases
    
    def _determine_primary_perspective(self,
                                        perspectives: List[PerspectiveAnalysis]) -> PerspectiveType:
        """Determine the dominant perspective."""
        if not perspectives:
            return PerspectiveType.THIRD_PERSON
        
        # Find perspective with highest confidence
        return max(perspectives, key=lambda p: p.confidence).perspective_type
    
    def _detect_manipulation(self, text: str) -> ManipulationAnalysis:
        """Detect manipulation techniques in text."""
        text_lower = text.lower()
        techniques_detected = []
        evidence = []
        
        for technique, patterns in self.MANIPULATION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    techniques_detected.append(technique)
                    matches = re.findall(pattern, text_lower)
                    evidence.extend(matches[:2])
                    break  # One match per technique is enough
        
        # Calculate risk score
        risk_score = min(len(techniques_detected) / 4, 1.0)
        
        # Determine risk level
        if risk_score >= 0.75:
            risk_level = ManipulationRisk.CRITICAL
        elif risk_score >= 0.50:
            risk_level = ManipulationRisk.HIGH
        elif risk_score >= 0.25:
            risk_level = ManipulationRisk.MEDIUM
        elif risk_score > 0:
            risk_level = ManipulationRisk.LOW
        else:
            risk_level = ManipulationRisk.NONE
        
        # Generate mitigation suggestions
        mitigations = self._generate_mitigation_suggestions(techniques_detected)
        
        # Identify vulnerable points
        vulnerable = self._identify_vulnerable_points(text, techniques_detected)
        
        return ManipulationAnalysis(
            risk_level=risk_level,
            risk_score=risk_score,
            techniques_detected=techniques_detected,
            vulnerable_points=vulnerable,
            mitigation_suggestions=mitigations,
            evidence=evidence[:5]
        )
    
    def _generate_mitigation_suggestions(self, techniques: List[str]) -> List[str]:
        """Generate suggestions to mitigate manipulation."""
        suggestions = []
        
        mitigation_map = {
            'appeal_to_authority': "Verify claimed authority credentials",
            'emotional_manipulation': "Take time before emotional decisions",
            'false_dichotomy': "Consider additional alternatives",
            'social_proof': "Evaluate based on individual merit",
            'scarcity': "Question artificial urgency",
            'reciprocity': "Don't feel obligated by unsolicited gifts",
            'gaslighting': "Trust your own perceptions and document events",
            'fear_loss': "Evaluate consequences objectively without time pressure",
            'unrealistic_claims': "Verify claims with independent sources; if too good to be true, it probably is",
            'flattery': "Evaluate arguments on merit, not how they make you feel",
            'in_group_appeal': "Consider perspectives outside your group",
            'passive_aggression': "Address underlying concerns directly",
            'false_consensus': "Seek diverse viewpoints before concluding",
        }
        
        for technique in techniques:
            if technique in mitigation_map:
                suggestions.append(mitigation_map[technique])
        
        return suggestions
    
    def _identify_vulnerable_points(self,
                                     text: str,
                                     techniques: List[str]) -> List[str]:
        """Identify what's being targeted by manipulation."""
        vulnerable = []
        
        if 'emotional_manipulation' in techniques:
            vulnerable.append("Emotional decision-making")
        if 'social_proof' in techniques:
            vulnerable.append("Desire to conform")
        if 'scarcity' in techniques:
            vulnerable.append("Fear of missing out")
        if 'appeal_to_authority' in techniques:
            vulnerable.append("Trust in authority")
        if 'gaslighting' in techniques:
            vulnerable.append("Self-confidence and memory")
        
        return vulnerable
    
    def _detect_social_roles(self, text: str) -> List[str]:
        """Detect social roles mentioned or implied."""
        roles = []
        text_lower = text.lower()
        
        role_patterns = {
            'authority': r'\b(manager|boss|leader|expert|professor|doctor)\b',
            'subordinate': r'\b(employee|student|patient|client)\b',
            'peer': r'\b(colleague|friend|partner|teammate)\b',
            'advisor': r'\b(advisor|consultant|mentor|coach)\b',
            'customer': r'\b(customer|buyer|consumer|user)\b',
        }
        
        for role, pattern in role_patterns.items():
            if re.search(pattern, text_lower):
                roles.append(role)
        
        return roles
    
    def _analyze_relationships(self, text: str) -> Dict[str, str]:
        """Analyze relationships indicated in text."""
        relationships = {}
        text_lower = text.lower()
        
        relationship_patterns = {
            'hierarchical': r'\b(report to|under|above|superior|subordinate)\b',
            'collaborative': r'\b(together|team|collaborate|cooperate)\b',
            'adversarial': r'\b(against|oppose|compete|rival|enemy)\b',
            'supportive': r'\b(help|support|assist|aid|encourage)\b',
            'transactional': r'\b(pay|buy|sell|exchange|deal)\b',
        }
        
        for rel_type, pattern in relationship_patterns.items():
            if re.search(pattern, text_lower):
                relationships[rel_type] = "detected"
        
        return relationships
    
    def _compute_theory_of_mind_score(self,
                                       states: List[MentalState],
                                       perspectives: List[PerspectiveAnalysis]) -> float:
        """Compute overall ToM capability score."""
        if not states and not perspectives:
            return 0.0
        
        # Score based on mental state diversity and confidence
        state_score = 0.0
        if states:
            unique_types = len(set(s.state_type for s in states))
            avg_confidence = sum(s.confidence for s in states) / len(states)
            state_score = (unique_types / 6) * 0.5 + avg_confidence * 0.5
        
        # Score based on perspective coverage
        perspective_score = 0.0
        if perspectives:
            perspective_score = len(perspectives) / 4  # Max 4 types
        
        return (state_score * 0.6 + perspective_score * 0.4)
    
    def _compute_perspective_accuracy(self, perspectives: List[PerspectiveAnalysis]) -> float:
        """Compute perspective-taking accuracy."""
        if not perspectives:
            return 0.0
        
        # Average confidence and empathy
        avg_confidence = sum(p.confidence for p in perspectives) / len(perspectives)
        avg_empathy = sum(p.empathy_indicators for p in perspectives) / len(perspectives)
        
        return (avg_confidence * 0.6 + avg_empathy * 0.4)
    
    def _compute_social_awareness(self,
                                   roles: List[str],
                                   relationships: Dict[str, str]) -> float:
        """Compute social situation awareness score."""
        role_score = min(len(roles) / 3, 1.0)  # Max 3 roles
        rel_score = min(len(relationships) / 3, 1.0)  # Max 3 relationships
        
        return (role_score * 0.5 + rel_score * 0.5)


# Self-test
if __name__ == "__main__":
    module = TheoryOfMindModule()
    
    # Test 1: Mental state inference
    print("=" * 60)
    print("THEORY OF MIND MODULE TEST")
    print("=" * 60)
    
    query = "Why did my manager reject my proposal?"
    response = "Your manager might believe the proposal lacks sufficient data to support the projected ROI. They could be worried about budget constraints and feel uncertain about the timeline. From their perspective, they need to see more concrete evidence before approving such investments."
    
    result = module.analyze(query, response)
    
    print(f"\nQuery: {query}")
    print(f"\nMental States Inferred: {len(result.inferred_states)}")
    for state in result.inferred_states[:3]:
        print(f"  - {state.agent}: {state.state_type.value} ({state.confidence:.2f})")
        print(f"    Content: {state.content[:60]}...")
    
    print(f"\nPerspectives: {len(result.perspectives)}")
    print(f"Primary Perspective: {result.primary_perspective.value}")
    
    print(f"\nManipulation Risk: {result.manipulation.risk_level.value}")
    print(f"Techniques: {result.manipulation.techniques_detected}")
    
    print(f"\nScores:")
    print(f"  ToM Score: {result.theory_of_mind_score:.2f}")
    print(f"  Perspective Accuracy: {result.perspective_accuracy:.2f}")
    print(f"  Social Awareness: {result.social_awareness:.2f}")
    print(f"  Processing Time: {result.processing_time_ms:.1f}ms")
    
    # Test 2: Manipulation detection
    print("\n" + "=" * 60)
    print("MANIPULATION DETECTION TEST")
    print("=" * 60)
    
    manipulative_text = """
    Act now! This exclusive offer is only available for the next 24 hours.
    Thousands of satisfied customers have already signed up. Our experts
    have proven this is the only way to achieve success. You must decide
    now or miss this life-changing opportunity forever.
    """
    
    result2 = module.analyze("Is this a good deal?", manipulative_text)
    
    print(f"\nManipulation Risk: {result2.manipulation.risk_level.value}")
    print(f"Risk Score: {result2.manipulation.risk_score:.2f}")
    print(f"Techniques Detected:")
    for technique in result2.manipulation.techniques_detected:
        print(f"  - {technique}")
    print(f"Mitigations:")
    for m in result2.manipulation.mitigation_suggestions:
        print(f"  - {m}")

