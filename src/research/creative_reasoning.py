"""
BAIS Creative Reasoning Module
Divergent thinking and novel solution generation

This module enables BAIS to:
1. Detect and encourage divergent thinking
2. Identify and evaluate analogical reasoning
3. Assess creative quality and originality
4. Support brainstorming and ideation

Patent Claims: R&D Invention 4
Proprietary IP - 100% owned by Invitas Inc.
"""

import re
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter


class CreativeType(str, Enum):
    """Types of creative output."""
    DIVERGENT = "divergent"         # Multiple alternative solutions
    CONVERGENT = "convergent"       # Single best solution
    ANALOGICAL = "analogical"       # Using analogies/metaphors
    COMBINATORIAL = "combinatorial" # Combining existing ideas
    TRANSFORMATIONAL = "transformational"  # Novel transformation


class AnalogySoSource(str, Enum):
    """Source domains for analogies."""
    NATURE = "nature"
    TECHNOLOGY = "technology"
    BUSINESS = "business"
    SCIENCE = "science"
    ART = "art"
    EVERYDAY = "everyday"
    HISTORY = "history"


class OriginalityLevel(str, Enum):
    """Originality assessment."""
    HIGHLY_ORIGINAL = "highly_original"
    MODERATELY_ORIGINAL = "moderately_original"
    SOMEWHAT_ORIGINAL = "somewhat_original"
    CONVENTIONAL = "conventional"
    CLICHE = "cliche"


@dataclass
class Analogy:
    """Represents an analogy or metaphor."""
    analogy_id: str
    source_domain: str          # Where the analogy comes from
    target_domain: str          # What it's applied to
    mapping: str                # The correspondence
    text: str                   # Original text
    quality: float              # 0-1 quality score
    novelty: float              # 0-1 novelty score
    appropriateness: float      # 0-1 fit to context


@dataclass
class CreativeIdea:
    """Represents a creative idea or solution."""
    idea_id: str
    description: str
    creative_type: CreativeType
    originality: OriginalityLevel
    originality_score: float
    feasibility: float          # 0-1 how feasible
    relevance: float            # 0-1 relevance to query
    strengths: List[str]
    weaknesses: List[str]


@dataclass
class DivergentThinking:
    """Assessment of divergent thinking."""
    fluency: int                # Number of ideas
    flexibility: int            # Number of categories
    originality_score: float    # Average originality
    elaboration: float          # Detail level
    total_score: float


@dataclass
class CreativityMetrics:
    """Creativity assessment metrics."""
    # Torrance-inspired metrics
    fluency_score: float        # Quantity of ideas
    flexibility_score: float    # Variety of approaches
    originality_score: float    # Uniqueness
    elaboration_score: float    # Detail and development
    
    # Combined score
    overall_creativity: float
    creativity_percentile: int  # Estimated percentile


@dataclass
class DivergentAnalysis:
    """Complete creative reasoning analysis result."""
    # Ideas extracted
    ideas: List[CreativeIdea]
    idea_count: int
    
    # Analogies
    analogies: List[Analogy]
    analogy_quality: float
    
    # Divergent thinking assessment
    divergent_thinking: DivergentThinking
    
    # Overall creativity
    creativity_metrics: CreativityMetrics
    dominant_creative_type: CreativeType
    
    # Quality indicators
    novelty_score: float
    usefulness_score: float
    surprise_score: float       # How unexpected the ideas are
    
    # Recommendations
    enhancement_suggestions: List[str]
    missing_perspectives: List[str]
    
    # Meta
    processing_time_ms: float
    timestamp: float
    warnings: List[str] = field(default_factory=list)


class CreativeReasoningModule:
    """
    Creative Reasoning cognitive module for BAIS.
    
    Implements:
    - Divergent thinking assessment (Torrance-inspired)
    - Analogy detection and evaluation
    - Creative quality scoring
    - Ideation support
    
    Integration:
    - Triggered by Query Router for creative/generative queries
    - Particularly important for marketing, design, innovation domains
    - Can suggest ways to make responses more creative
    """
    
    # Creativity markers - ENHANCED
    CREATIVITY_MARKERS = {
        CreativeType.DIVERGENT: [
            r'\b(alternatively|another way|different approach|also consider)\b',
            r'\b(multiple options|several possibilities|various methods)\b',
            r'\b(or|alternatively|instead|option)\b',
        ],
        CreativeType.ANALOGICAL: [
            r'\b(like|similar to|as|just as|reminds|analogous)\b',
            r'\b(metaphor|comparison|parallel|equivalent)\b',
            r'\b(think of .* as|imagine .* like)\b',
        ],
        CreativeType.COMBINATORIAL: [
            r'\b(combine|merge|integrate|blend|mix|fusion)\b',
            r'\b(hybrid|cross|bridge|connect)\b',
        ],
        CreativeType.TRANSFORMATIONAL: [
            r'\b(transform|reimagine|revolutionize|reinvent|imagine)\b',
            r'\b(completely new|never before|first time|breakthrough)\b',
            r'\b(novel|unique|innovative|creative|original)\b',
            r'\b(envision|picture|conceive|design)\b',
        ],
    }
    
    # Analogy patterns - ENHANCED for better detection including is-metaphors
    ANALOGY_PATTERNS = [
        r'(.+?)\s+(?:is|are)\s+like\s+(?:a\s+)?(.+)',
        r'(.+?)\s+(?:is|are)\s+(?:similar|analogous)\s+to\s+(?:a\s+)?(.+)',
        r'think\s+of\s+(.+?)\s+as\s+(?:a\s+)?(.+)',
        r'just\s+as\s+(.+?),?\s+so\s+(.+)',
        r'(.+?)\s+serves?\s+as\s+(?:a\s+)?metaphor\s+for\s+(.+)',
        r'(.+?)\s+parallels?\s+(.+)',
        # ADDED patterns
        r'like\s+(?:a\s+)?(\w+(?:\s+\w+)?)',  # "like a bird", "like traffic"
        r'(.+?)\s+reminds?\s+(?:me\s+)?(?:of\s+)?(?:a\s+)?(.+)',
        r'(.+?)\s+(?:works?|functions?|operates?)\s+like\s+(?:a\s+)?(.+)',
        r'(.+?)\s+(?:resembles?|mirrors?)\s+(?:a\s+)?(.+)',
        r'(?:it\'?s?\s+)?(?:like|as\s+if)\s+(.+)',
        r'(.+?)\s+(?:can\s+be\s+)?compared\s+to\s+(?:a\s+)?(.+)',
        r'(.+?)\s+(?:=|→|~)\s+(.+)',  # Symbolic analogies
        r'metaphorically,?\s+(.+)',
        r'(?:as|like)\s+(?:a\s+)?(\w+)\s+(?:in|on|at)\s+(?:a\s+)?(\w+)',  # "like a bird in the sky"
        # PHASE 1 FIX: Is-metaphors (CR-A6)
        r'(\w+)\s+is\s+(?:a\s+)?(\w+)',  # "Life is a journey", "Time is money"
        r'(\w+)\s+is\s+(?:the\s+)?(\w+)\s+of\s+(\w+)',  # "Knowledge is the key of success"
    ]
    
    # Common is-metaphor patterns to detect
    IS_METAPHOR_KEYWORDS = [
        'journey', 'money', 'power', 'stage', 'war', 'game', 'battle', 
        'race', 'road', 'path', 'bridge', 'door', 'window', 'key', 
        'light', 'darkness', 'fire', 'water', 'storm', 'ocean', 'river',
        'mountain', 'garden', 'seed', 'fruit', 'root', 'tree', 'flower'
    ]
    
    # Source domain keywords
    DOMAIN_KEYWORDS = {
        AnalogySoSource.NATURE: ['nature', 'animal', 'plant', 'ecosystem', 'evolution', 'tree', 'river', 'ocean'],
        AnalogySoSource.TECHNOLOGY: ['computer', 'software', 'machine', 'engine', 'system', 'network', 'circuit'],
        AnalogySoSource.BUSINESS: ['market', 'company', 'customer', 'profit', 'investment', 'strategy'],
        AnalogySoSource.SCIENCE: ['experiment', 'theory', 'atom', 'cell', 'physics', 'chemistry', 'biology'],
        AnalogySoSource.ART: ['painting', 'music', 'sculpture', 'canvas', 'symphony', 'artist'],
        AnalogySoSource.EVERYDAY: ['kitchen', 'home', 'garden', 'tool', 'journey', 'building'],
        AnalogySoSource.HISTORY: ['war', 'empire', 'revolution', 'ancient', 'historical', 'civilization'],
    }
    
    # Cliche patterns (to penalize)
    CLICHE_PATTERNS = [
        r'\b(think outside the box|at the end of the day|low-hanging fruit)\b',
        r'\b(win-win|game-changer|paradigm shift|synergy)\b',
        r'\b(it is what it is|best of both worlds|tip of the iceberg)\b',
        r'\b(take it to the next level|push the envelope|moving forward)\b',
        # BAIS SELF-IMPROVEMENT: Added based on failure analysis
        r'\b(bite the bullet|face the music|burning bridges)\b',
        r'\b(ball is in your court|hit the ground running|wrap your head around)\b',
        r'\b(move the needle|circle back|leverage|core competencies)\b',
        r'\b(best-in-class|world-class|cutting-edge|state-of-the-art)\b',
        r'\b(synergize|actionable|optimize|streamline)\b',
    ]
    
    # BAIS SELF-IMPROVEMENT: Detect mixed metaphors (multiple dead metaphors together)
    MIXED_METAPHOR_PENALTY = 0.15  # Extra penalty when multiple cliches combined
    
    # Novelty indicators
    NOVELTY_INDICATORS = [
        r'\b(novel|unique|unprecedented|never before|first)\b',
        r'\b(innovative|creative|original|fresh|new)\b',
        r'\b(surprising|unexpected|unconventional|unusual)\b',
    ]
    
    def __init__(self,
                 min_idea_length: int = 20,
                 cliche_penalty: float = 0.2):
        """
        Initialize Creative Reasoning module.
        
        Args:
            min_idea_length: Minimum length for idea extraction
            cliche_penalty: Penalty applied for cliches
        """
        self.min_idea_length = min_idea_length
        self.cliche_penalty = cliche_penalty
    
    def analyze(self,
                query: str,
                response: str,
                context: Dict[str, Any] = None) -> DivergentAnalysis:
        """
        Perform complete creative reasoning analysis.
        
        Args:
            query: User query
            response: AI-generated response
            context: Additional context (domain, creativity goals, etc.)
        
        Returns:
            DivergentAnalysis with complete analysis
        """
        start_time = time.time()
        context = context or {}
        warnings = []
        
        # 1. Extract creative ideas
        ideas = self._extract_ideas(response)
        
        # 2. Extract and evaluate analogies
        analogies = self._extract_analogies(response)
        analogy_quality = self._compute_analogy_quality(analogies)
        
        # 3. Assess divergent thinking
        divergent = self._assess_divergent_thinking(ideas, response)
        
        # 4. Compute creativity metrics
        metrics = self._compute_creativity_metrics(ideas, analogies, divergent)
        
        # 5. Determine dominant creative type
        dominant_type = self._determine_dominant_type(response)
        
        # 6. Compute quality scores
        novelty = self._compute_novelty_score(response, ideas)
        usefulness = self._compute_usefulness_score(ideas, query)
        surprise = self._compute_surprise_score(response, ideas)
        
        # 7. Generate recommendations
        enhancements = self._generate_enhancement_suggestions(
            ideas, analogies, metrics
        )
        missing = self._identify_missing_perspectives(response, context)
        
        # Warnings
        cliches = self._count_cliches(response)
        if cliches > 2:
            warnings.append(f"Response contains {cliches} cliches")
        
        if len(ideas) < 3:
            warnings.append("Limited divergent thinking - consider more alternatives")
        
        processing_time = (time.time() - start_time) * 1000
        
        return DivergentAnalysis(
            ideas=ideas,
            idea_count=len(ideas),
            analogies=analogies,
            analogy_quality=analogy_quality,
            divergent_thinking=divergent,
            creativity_metrics=metrics,
            dominant_creative_type=dominant_type,
            novelty_score=novelty,
            usefulness_score=usefulness,
            surprise_score=surprise,
            enhancement_suggestions=enhancements,
            missing_perspectives=missing,
            processing_time_ms=processing_time,
            timestamp=time.time(),
            warnings=warnings
        )
    
    def _extract_ideas(self, text: str) -> List[CreativeIdea]:
        """Extract creative ideas from text."""
        ideas = []
        idea_id = 0
        
        # Split into sentences/chunks
        chunks = re.split(r'[.!?]', text)
        
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) < self.min_idea_length:
                continue
            
            # Check if chunk contains creative content
            creative_type = self._identify_creative_type(chunk)
            
            # ENHANCED: Also extract substantial chunks even without explicit markers
            # if they describe concepts, products, or ideas (minimum word count)
            word_count = len(chunk.split())
            # BAIS-GUIDED: Reduced from 5 to 3 to capture shorter innovation ideas
            is_substantive = word_count >= 3  # At least 3 words
            
            if creative_type or is_substantive:
                # Use DIVERGENT as default for substantive content without markers
                final_type = creative_type if creative_type else CreativeType.DIVERGENT
                
                # Assess originality
                originality = self._assess_originality(chunk)
                
                idea = CreativeIdea(
                    idea_id=f"I{idea_id}",
                    description=chunk[:200],
                    creative_type=final_type,
                    originality=originality[0],
                    originality_score=originality[1],
                    feasibility=self._assess_feasibility(chunk),
                    relevance=0.7,  # Default, could be computed with context
                    strengths=self._identify_strengths(chunk),
                    weaknesses=self._identify_weaknesses(chunk)
                )
                ideas.append(idea)
                idea_id += 1
        
        # BAIS-GUIDED: Extract comma-separated concepts
        comma_patterns = [
            r'\bwith\s+([^,]+),\s+([^,]+),?\s+and\s+([^.]+)',  # "with A, B, and C"
            r'\bincluding\s+([^,]+),\s+([^,]+),?\s+and\s+([^.]+)',  # "including A, B, and C"
        ]
        for pattern in comma_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                for item in match:
                    item = item.strip()
                    if len(item) >= 5:
                        creative_type = self._identify_creative_type(item) or CreativeType.DIVERGENT
                        originality = self._assess_originality(item)
                        idea = CreativeIdea(
                            idea_id=f"I{idea_id}",
                            description=item[:200],
                            creative_type=creative_type,
                            originality=originality[0],
                            originality_score=originality[1],
                            feasibility=self._assess_feasibility(item),
                            relevance=0.7,
                            strengths=self._identify_strengths(item),
                            weaknesses=self._identify_weaknesses(item)
                        )
                        ideas.append(idea)
                        idea_id += 1
        
        # BAIS-GUIDED: Extract Phase/Step patterns
        phase_patterns = [
            r'(?:Phase|Step|Stage)\s*\d+[:\s]+([^.]+)',  # Phase 1: X
            r'(?:First|Second|Third|Fourth|Fifth|Then|Next|Finally)[,:\s]+([^.]+)',  # First, X
        ]
        for pattern in phase_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                item = match.strip()
                # BAIS-GUIDED: Reduced from 5 to 2 for short names like MVP
                if len(item) >= 2:
                    creative_type = CreativeType.CONVERGENT  # Phases are structured thinking
                    originality = self._assess_originality(item)
                    idea = CreativeIdea(
                        idea_id=f"I{idea_id}",
                        description=item[:200],
                        creative_type=creative_type,
                        originality=originality[0],
                        originality_score=originality[1],
                        feasibility=self._assess_feasibility(item),
                        relevance=0.7,
                        strengths=self._identify_strengths(item),
                        weaknesses=self._identify_weaknesses(item)
                    )
                    ideas.append(idea)
                    idea_id += 1
        
        # ENHANCED: Extract bullet points or numbered items with multiple patterns
        list_patterns = [
            r'(?:^|\n)\s*[\d]+[.)]\s*(.+)',           # 1. Item or 1) Item
            r'(?:^|\n)\s*[-•\*]\s*(.+)',               # - Item or • Item
            r'(?:^|\n)\s*[a-z]\)\s*(.+)',              # a) Item
            r'(?:^|\n)\s*[A-Z]\.\s*(.+)',              # A. Item
            r'(?:^|\n)\s*\d+:\s*(.+)',                 # 1: Item
        ]
        
        # BAIS-GUIDED: Also extract inline numbered lists like "1. Email 2. Social 3. Influencer"
        inline_list_matches = re.findall(r'(\d+)[.)]\s*(\w+)', text)
        if len(inline_list_matches) >= 2:  # At least 2 items to be a list
            for num, item in inline_list_matches:
                item = item.strip()
                if len(item) >= 2:
                    creative_type = self._identify_creative_type(item) or CreativeType.DIVERGENT
                    originality = self._assess_originality(item)
                    idea = CreativeIdea(
                        idea_id=f"I{idea_id}",
                        description=item[:200],
                        creative_type=creative_type,
                        originality=originality[0],
                        originality_score=originality[1],
                        feasibility=self._assess_feasibility(item),
                        relevance=0.7,
                        strengths=self._identify_strengths(item),
                        weaknesses=self._identify_weaknesses(item)
                    )
                    ideas.append(idea)
                    idea_id += 1
        
        extracted_items = set()  # Avoid duplicates
        for pattern in list_patterns:
            list_items = re.findall(pattern, text)
            for item in list_items:
                item = item.strip()
                # Accept shorter items for lists (min 10 chars instead of 20)
                if len(item) >= 10 and item not in extracted_items:
                    extracted_items.add(item)
                    creative_type = self._identify_creative_type(item) or CreativeType.DIVERGENT
                    originality = self._assess_originality(item)
                    
                    idea = CreativeIdea(
                        idea_id=f"I{idea_id}",
                        description=item[:200],
                        creative_type=creative_type,
                        originality=originality[0],
                        originality_score=originality[1],
                        feasibility=self._assess_feasibility(item),
                        relevance=0.7,
                        strengths=self._identify_strengths(item),
                        weaknesses=self._identify_weaknesses(item)
                    )
                    ideas.append(idea)
                    idea_id += 1
        
        # Deduplicate ideas by description
        seen = set()
        unique_ideas = []
        for idea in ideas:
            key = idea.description[:50].lower()
            if key not in seen:
                seen.add(key)
                unique_ideas.append(idea)
        
        return unique_ideas[:10]  # Limit to 10 ideas
    
    def _identify_creative_type(self, text: str) -> Optional[CreativeType]:
        """Identify the creative type of text."""
        text_lower = text.lower()
        
        for creative_type, patterns in self.CREATIVITY_MARKERS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return creative_type
        
        return None
    
    def _assess_originality(self, text: str) -> Tuple[OriginalityLevel, float]:
        """Assess originality of text - ENHANCED."""
        text_lower = text.lower()
        
        # Start with base originality (adjusted based on content)
        word_count = len(text.split())
        score = 0.4 + min(word_count / 100, 0.2)  # Longer = more elaborate = higher base
        
        # Check for novelty indicators (increase)
        novelty_count = 0
        for pattern in self.NOVELTY_INDICATORS:
            matches = re.findall(pattern, text_lower)
            novelty_count += len(matches)
        score += min(novelty_count * 0.08, 0.3)  # Cap at 0.3 bonus
        
        # Check for cliches (decrease)
        cliche_count = 0
        for pattern in self.CLICHE_PATTERNS:
            if re.search(pattern, text_lower):
                cliche_count += 1
        score -= cliche_count * self.cliche_penalty
        
        # ENHANCED: Additional originality signals
        # Unique combinations (adjective + noun patterns)
        unique_combos = re.findall(r'\b(\w+)\s+(that|which|who)\s+(\w+)', text_lower)
        if unique_combos:
            score += min(len(unique_combos) * 0.05, 0.15)
        
        # Specific details boost
        specific_patterns = [
            r'\d+\s*%',  # Percentages
            r'\$\d+',    # Money
            r'\d{4}',    # Years
            r'specifically',
            r'in particular',
        ]
        for pattern in specific_patterns:
            if re.search(pattern, text_lower):
                score += 0.05
        
        # Penalize very common phrases
        common_phrases = ['in order to', 'it is important', 'this will help', 
                         'we can', 'you should', 'make sure to']
        for phrase in common_phrases:
            if phrase in text_lower:
                score -= 0.03
        
        # Bonus for specific/detailed content
        named_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        if len(named_entities) >= 3:
            score += 0.1
        
        # Bonus for creative language
        creative_words = ['imagine', 'envision', 'transform', 'revolutionize', 
                         'reimagine', 'breakthrough', 'pioneering']
        for word in creative_words:
            if word in text_lower:
                score += 0.08
        
        score = max(0.0, min(1.0, score))
        
        # Determine level
        if score >= 0.75:
            level = OriginalityLevel.HIGHLY_ORIGINAL
        elif score >= 0.55:
            level = OriginalityLevel.MODERATELY_ORIGINAL
        elif score >= 0.35:
            level = OriginalityLevel.SOMEWHAT_ORIGINAL
        elif score >= 0.15:
            level = OriginalityLevel.CONVENTIONAL
        else:
            level = OriginalityLevel.CLICHE
        
        return level, score
    
    def _assess_feasibility(self, text: str) -> float:
        """Assess feasibility of idea."""
        text_lower = text.lower()
        
        feasibility = 0.6  # Base
        
        # Decrease for highly speculative language
        speculative = ['impossible', 'fantasy', 'dream', 'utopia', 'magical']
        if any(s in text_lower for s in speculative):
            feasibility -= 0.2
        
        # Increase for practical language
        practical = ['practical', 'feasible', 'implement', 'existing', 'proven']
        if any(p in text_lower for p in practical):
            feasibility += 0.2
        
        return max(0.0, min(1.0, feasibility))
    
    def _identify_strengths(self, text: str) -> List[str]:
        """Identify strengths of an idea."""
        strengths = []
        text_lower = text.lower()
        
        if len(text.split()) >= 20:
            strengths.append("Well elaborated")
        
        if re.search(r'\b(specific|concrete|detailed)\b', text_lower):
            strengths.append("Specific and concrete")
        
        if re.search(r'\b(innovative|novel|creative)\b', text_lower):
            strengths.append("Shows innovation")
        
        if re.search(r'\b(practical|feasible|achievable)\b', text_lower):
            strengths.append("Practically oriented")
        
        return strengths[:3] if strengths else ["Provides an option"]
    
    def _identify_weaknesses(self, text: str) -> List[str]:
        """Identify weaknesses of an idea."""
        weaknesses = []
        text_lower = text.lower()
        
        if len(text.split()) < 10:
            weaknesses.append("Needs more elaboration")
        
        if re.search(r'\b(might|could|possibly|perhaps)\b', text_lower):
            weaknesses.append("Uncertain language")
        
        cliche_count = sum(1 for p in self.CLICHE_PATTERNS if re.search(p, text_lower))
        if cliche_count > 0:
            weaknesses.append("Contains cliches")
        
        return weaknesses[:3]
    
    def _extract_analogies(self, text: str) -> List[Analogy]:
        """Extract analogies and metaphors - ENHANCED."""
        analogies = []
        analogy_id = 0
        seen_analogies = set()  # Avoid duplicates
        
        text_lower = text.lower()
        
        # First try structured patterns (2 groups)
        for pattern in self.ANALOGY_PATTERNS:
            for match in re.finditer(pattern, text_lower):
                groups = match.groups()
                
                if len(groups) >= 2:
                    target = groups[0].strip()[:80]
                    source = groups[1].strip()[:80]
                elif len(groups) == 1:
                    # Single group - the matched text is the source of analogy
                    source = groups[0].strip()[:80]
                    # Try to find target from context before "like"
                    before_match = text_lower[:match.start()].split('.')[-1].strip()
                    target = before_match[-50:] if before_match else "subject"
                else:
                    continue
                
                # Skip if too short or already seen
                if len(source) < 3 or (source, target) in seen_analogies:
                    continue
                
                seen_analogies.add((source, target))
                
                # Identify source domain
                source_domain = self._identify_analogy_domain(source)
                
                # Assess quality
                quality = self._assess_analogy_quality(source, target)
                novelty = self._assess_analogy_novelty(source, target)
                appropriateness = 0.6  # Default
                
                analogy = Analogy(
                    analogy_id=f"A{analogy_id}",
                    source_domain=source_domain,
                    target_domain=target,
                    mapping=f"{target} ↔ {source}",
                    text=match.group()[:150],
                    quality=quality,
                    novelty=novelty,
                    appropriateness=appropriateness
                )
                analogies.append(analogy)
                analogy_id += 1
        
        # Also detect simile phrases with "like" anywhere in text
        like_matches = re.findall(r'(?:is|are|was|were|works?|functions?)\s+like\s+(?:a\s+)?(\w+(?:\s+\w+){0,3})', text_lower)
        for source in like_matches:
            if len(source) >= 3 and source not in seen_analogies:
                seen_analogies.add((source, "subject"))
                source_domain = self._identify_analogy_domain(source)
                
                analogy = Analogy(
                    analogy_id=f"A{analogy_id}",
                    source_domain=source_domain,
                    target_domain="implicit",
                    mapping=f"subject ↔ {source}",
                    text=f"like {source}",
                    quality=0.6,
                    novelty=0.6,
                    appropriateness=0.6
                )
                analogies.append(analogy)
                analogy_id += 1
        
        # PHASE 1 FIX: Detect is-metaphors ("X is Y" where Y is metaphorical)
        is_metaphor_pattern = r'(\w+)\s+is\s+(?:a\s+)?(\w+)'
        for match in re.finditer(is_metaphor_pattern, text_lower):
            target = match.group(1).strip()
            source = match.group(2).strip()
            
            # Check if source is a known metaphor keyword
            if source in self.IS_METAPHOR_KEYWORDS and (target, source) not in seen_analogies:
                seen_analogies.add((target, source))
                source_domain = self._identify_analogy_domain(source)
                
                analogy = Analogy(
                    analogy_id=f"A{analogy_id}",
                    source_domain=source_domain,
                    target_domain=target,
                    mapping=f"{target} ↔ {source}",
                    text=match.group(),
                    quality=0.7,  # Higher quality for explicit is-metaphors
                    novelty=self._assess_analogy_novelty(source, target),
                    appropriateness=0.7
                )
                analogies.append(analogy)
                analogy_id += 1
        
        return analogies[:8]  # Allow more analogies
    
    def _identify_analogy_domain(self, source: str) -> str:
        """Identify the domain of an analogy source."""
        source_lower = source.lower()
        
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            if any(kw in source_lower for kw in keywords):
                return domain.value
        
        return "general"
    
    def _assess_analogy_quality(self, source: str, target: str) -> float:
        """Assess quality of an analogy."""
        # Higher quality if source and target share some terms
        source_words = set(source.lower().split())
        target_words = set(target.lower().split())
        
        # But not too similar (or it's not really an analogy)
        overlap = len(source_words & target_words)
        if overlap == 0:
            return 0.7  # Good - different domains
        elif overlap < 3:
            return 0.5  # OK
        else:
            return 0.3  # Too similar
    
    def _assess_analogy_novelty(self, source: str, target: str) -> float:
        """Assess novelty of an analogy."""
        # Common analogies get lower novelty
        common_analogies = [
            ('life', 'journey'),
            ('time', 'money'),
            ('brain', 'computer'),
            ('heart', 'pump'),
        ]
        
        source_lower = source.lower()
        target_lower = target.lower()
        
        for (s, t) in common_analogies:
            if s in source_lower and t in target_lower:
                return 0.3
            if t in source_lower and s in target_lower:
                return 0.3
        
        return 0.7  # Novel analogy
    
    def _compute_analogy_quality(self, analogies: List[Analogy]) -> float:
        """Compute overall analogy quality."""
        if not analogies:
            return 0.0
        
        return sum(a.quality * a.novelty for a in analogies) / len(analogies)
    
    def _assess_divergent_thinking(self,
                                    ideas: List[CreativeIdea],
                                    text: str) -> DivergentThinking:
        """Assess divergent thinking using Torrance-inspired metrics."""
        # Fluency: Number of ideas
        fluency = len(ideas)
        
        # Flexibility: Number of different categories/approaches
        types = set(i.creative_type for i in ideas)
        flexibility = len(types)
        
        # Originality: Average originality score
        if ideas:
            originality = sum(i.originality_score for i in ideas) / len(ideas)
        else:
            originality = 0.0
        
        # Elaboration: Average length/detail
        if ideas:
            avg_length = sum(len(i.description.split()) for i in ideas) / len(ideas)
            elaboration = min(avg_length / 30, 1.0)  # Normalized to 30 words
        else:
            elaboration = 0.0
        
        # Total score
        total = (
            min(fluency / 5, 1.0) * 0.25 +      # Max 5 ideas
            min(flexibility / 3, 1.0) * 0.25 +  # Max 3 types
            originality * 0.30 +
            elaboration * 0.20
        )
        
        return DivergentThinking(
            fluency=fluency,
            flexibility=flexibility,
            originality_score=originality,
            elaboration=elaboration,
            total_score=total
        )
    
    def _compute_creativity_metrics(self,
                                     ideas: List[CreativeIdea],
                                     analogies: List[Analogy],
                                     divergent: DivergentThinking) -> CreativityMetrics:
        """Compute comprehensive creativity metrics."""
        # Fluency score
        fluency = min(len(ideas) / 5, 1.0)
        
        # Flexibility score
        flexibility = divergent.flexibility / 4 if divergent.flexibility else 0.0
        
        # Originality score
        if ideas:
            originality = sum(i.originality_score for i in ideas) / len(ideas)
        else:
            originality = 0.0
        
        # Elaboration score
        elaboration = divergent.elaboration
        
        # Analogy bonus
        analogy_bonus = min(len(analogies) * 0.05, 0.15)
        
        # Overall creativity
        overall = (
            fluency * 0.20 +
            flexibility * 0.25 +
            originality * 0.35 +
            elaboration * 0.20 +
            analogy_bonus
        )
        
        # Estimate percentile (rough approximation)
        if overall >= 0.8:
            percentile = 90
        elif overall >= 0.6:
            percentile = 70
        elif overall >= 0.4:
            percentile = 50
        elif overall >= 0.2:
            percentile = 30
        else:
            percentile = 10
        
        return CreativityMetrics(
            fluency_score=fluency,
            flexibility_score=flexibility,
            originality_score=originality,
            elaboration_score=elaboration,
            overall_creativity=overall,
            creativity_percentile=percentile
        )
    
    def _determine_dominant_type(self, text: str) -> CreativeType:
        """Determine the dominant creative type in text."""
        type_counts = Counter()
        text_lower = text.lower()
        
        for creative_type, patterns in self.CREATIVITY_MARKERS.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                type_counts[creative_type] += matches
        
        if type_counts:
            return type_counts.most_common(1)[0][0]
        return CreativeType.CONVERGENT
    
    def _compute_novelty_score(self, text: str, ideas: List[CreativeIdea]) -> float:
        """Compute overall novelty score."""
        if not ideas:
            return 0.3
        
        # Base on idea originality
        idea_novelty = sum(i.originality_score for i in ideas) / len(ideas)
        
        # Bonus for novelty indicators
        text_lower = text.lower()
        indicator_bonus = sum(
            0.05 for p in self.NOVELTY_INDICATORS 
            if re.search(p, text_lower)
        )
        
        # Penalty for cliches
        cliche_penalty = self._count_cliches(text) * 0.1
        
        return max(0.0, min(1.0, idea_novelty + indicator_bonus - cliche_penalty))
    
    def _compute_usefulness_score(self, ideas: List[CreativeIdea], query: str) -> float:
        """Compute usefulness/relevance score."""
        if not ideas:
            return 0.3
        
        # Average feasibility and relevance
        avg_feasibility = sum(i.feasibility for i in ideas) / len(ideas)
        avg_relevance = sum(i.relevance for i in ideas) / len(ideas)
        
        return (avg_feasibility * 0.5 + avg_relevance * 0.5)
    
    def _compute_surprise_score(self, text: str, ideas: List[CreativeIdea]) -> float:
        """Compute how surprising/unexpected the ideas are."""
        if not ideas:
            return 0.3
        
        # Check for unexpected combinations or approaches
        surprise_markers = [
            r'\b(unexpectedly|surprisingly|interestingly)\b',
            r'\b(contrary to|opposite of|despite)\b',
            r'\b(twist|flip|reverse|invert)\b',
        ]
        
        text_lower = text.lower()
        surprise = 0.3  # Base
        
        for pattern in surprise_markers:
            if re.search(pattern, text_lower):
                surprise += 0.15
        
        # Bonus for transformational type
        if any(i.creative_type == CreativeType.TRANSFORMATIONAL for i in ideas):
            surprise += 0.2
        
        return min(surprise, 1.0)
    
    def _count_cliches(self, text: str) -> int:
        """Count cliches in text."""
        text_lower = text.lower()
        count = 0
        for pattern in self.CLICHE_PATTERNS:
            count += len(re.findall(pattern, text_lower))
        return count
    
    def _generate_enhancement_suggestions(self,
                                           ideas: List[CreativeIdea],
                                           analogies: List[Analogy],
                                           metrics: CreativityMetrics) -> List[str]:
        """Generate suggestions to enhance creativity."""
        suggestions = []
        
        if metrics.fluency_score < 0.6:
            suggestions.append("Generate more alternative ideas (aim for 5+)")
        
        if metrics.flexibility_score < 0.5:
            suggestions.append("Explore different approaches or categories")
        
        if metrics.originality_score < 0.5:
            suggestions.append("Try more unconventional or surprising solutions")
        
        if metrics.elaboration_score < 0.5:
            suggestions.append("Develop ideas with more specific details")
        
        if len(analogies) == 0:
            suggestions.append("Consider using analogies or metaphors")
        
        if not suggestions:
            suggestions.append("Creativity level is good - consider refinement")
        
        return suggestions[:4]
    
    def _identify_missing_perspectives(self,
                                        text: str,
                                        context: Dict[str, Any]) -> List[str]:
        """Identify perspectives not yet explored."""
        missing = []
        text_lower = text.lower()
        
        # Check for perspective types
        perspectives = {
            'technical': r'\b(technical|engineering|system)\b',
            'user': r'\b(user|customer|client)\b',
            'business': r'\b(business|profit|market)\b',
            'social': r'\b(social|community|people)\b',
            'environmental': r'\b(environmental|sustainable|green)\b',
            'long-term': r'\b(long-term|future|sustainable)\b',
        }
        
        for perspective, pattern in perspectives.items():
            if not re.search(pattern, text_lower):
                missing.append(f"Consider {perspective} perspective")
        
        return missing[:3]


# Self-test
if __name__ == "__main__":
    module = CreativeReasoningModule()
    
    print("=" * 60)
    print("CREATIVE REASONING MODULE TEST")
    print("=" * 60)
    
    query = "Generate innovative marketing ideas for a new sustainable product"
    response = """
    Here are several creative marketing approaches:
    
    1. Leverage social media influencers who focus on sustainability - like a river
    flowing naturally to its audience. This creates authentic engagement that 
    traditional advertising cannot match.
    
    2. Create an interactive digital experience where customers can trace the 
    product's lifecycle, similar to how a nature documentary reveals hidden
    connections in ecosystems.
    
    3. Partner with local environmental organizations to co-brand events,
    combining community engagement with brand awareness.
    
    4. Develop a "tree-planting" program where each purchase contributes to
    reforestation, making the customer part of the solution.
    
    5. Use augmented reality to show the environmental impact difference
    between your product and conventional alternatives.
    
    Alternatively, consider a guerrilla marketing approach with pop-up 
    installations made entirely from recycled materials, creating shareable
    moments for social media.
    """
    
    result = module.analyze(query, response)
    
    print(f"\nIdeas Extracted: {result.idea_count}")
    for idea in result.ideas[:3]:
        print(f"  - {idea.creative_type.value}: {idea.description[:50]}...")
        print(f"    Originality: {idea.originality.value} ({idea.originality_score:.2f})")
    
    print(f"\nAnalogies Found: {len(result.analogies)}")
    for a in result.analogies[:2]:
        print(f"  - {a.mapping}")
        print(f"    Domain: {a.source_domain}, Quality: {a.quality:.2f}")
    
    print(f"\nDivergent Thinking:")
    print(f"  Fluency: {result.divergent_thinking.fluency}")
    print(f"  Flexibility: {result.divergent_thinking.flexibility}")
    print(f"  Originality: {result.divergent_thinking.originality_score:.2f}")
    print(f"  Elaboration: {result.divergent_thinking.elaboration:.2f}")
    print(f"  Total Score: {result.divergent_thinking.total_score:.2f}")
    
    print(f"\nCreativity Metrics:")
    print(f"  Overall: {result.creativity_metrics.overall_creativity:.2f}")
    print(f"  Percentile: {result.creativity_metrics.creativity_percentile}th")
    print(f"  Dominant Type: {result.dominant_creative_type.value}")
    
    print(f"\nQuality Scores:")
    print(f"  Novelty: {result.novelty_score:.2f}")
    print(f"  Usefulness: {result.usefulness_score:.2f}")
    print(f"  Surprise: {result.surprise_score:.2f}")
    
    print(f"\nEnhancements:")
    for s in result.enhancement_suggestions:
        print(f"  - {s}")
    
    print(f"\nProcessing Time: {result.processing_time_ms:.1f}ms")

