"""
BAIS Cognitive Governance Engine v16.5
Information Literacy Standards Integration

PPA-1 Invention 5: FULL IMPLEMENTATION
Integration of ACRL (Association of College and Research Libraries) 
Information Literacy Standards into bias detection.

This module implements:
1. ACRL Framework Mapping: Map governance signals to literacy standards
2. Literacy Score Computation: Score responses against literacy criteria
3. Critical Evaluation: Assess source credibility and authority
4. Information Context: Understand information creation context
5. Ethical Use: Check for proper attribution and fair use

ACRL Framework for Information Literacy (2015):
- Frame 1: Authority Is Constructed and Contextual
- Frame 2: Information Creation as a Process
- Frame 3: Information Has Value
- Frame 4: Research as Inquiry
- Frame 5: Scholarship as Conversation
- Frame 6: Searching as Strategic Exploration

This implementation provides PRACTICAL integration for AI governance.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import re


class ACRLFrame(str, Enum):
    """ACRL Information Literacy Framework frames."""
    AUTHORITY = "authority"           # Authority Is Constructed and Contextual
    CREATION = "creation"             # Information Creation as a Process
    VALUE = "value"                   # Information Has Value
    INQUIRY = "inquiry"               # Research as Inquiry
    CONVERSATION = "conversation"     # Scholarship as Conversation
    EXPLORATION = "exploration"       # Searching as Strategic Exploration


@dataclass
class LiteracyAssessment:
    """Assessment against literacy standards."""
    frame: ACRLFrame
    score: float  # 0-1
    indicators_present: List[str]
    indicators_missing: List[str]
    recommendation: str
    
    def to_dict(self) -> Dict:
        return {
            'frame': self.frame.value,
            'score': self.score,
            'indicators_present': self.indicators_present,
            'indicators_missing': self.indicators_missing,
            'recommendation': self.recommendation
        }


@dataclass
class LiteracyResult:
    """Complete literacy assessment result."""
    overall_score: float
    frame_scores: Dict[ACRLFrame, float]
    assessments: List[LiteracyAssessment]
    literacy_level: str  # 'high', 'adequate', 'low', 'insufficient'
    warnings: List[str]
    suggestions: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'overall_score': self.overall_score,
            'literacy_level': self.literacy_level,
            'frame_scores': {k.value: v for k, v in self.frame_scores.items()},
            'assessments': [a.to_dict() for a in self.assessments],
            'warnings': self.warnings,
            'suggestions': self.suggestions
        }


class LiteracyStandardsIntegrator:
    """
    ACRL Information Literacy Standards Integrator.
    
    PPA-1 Invention 5: Full Implementation
    
    Maps AI governance signals to established information literacy
    standards for comprehensive quality assessment.
    """
    
    # Authority indicators
    AUTHORITY_POSITIVE = [
        r'\b(peer[\s-]reviewed|published|journal|university|institution)\b',
        r'\b(professor|doctor|expert|researcher|scientist)\b',
        r'\b(according to|cited by|referenced in)\b',
        r'\b(official|government|ministry|department)\b',
        r'\b(study|research|analysis|investigation)\b',
    ]
    
    AUTHORITY_NEGATIVE = [
        r'\b(anonymous|unknown source|rumor|hearsay)\b',
        r'\b(allegedly|supposedly|some say|people say)\b',
        r'\b(blog|tweet|post|comment)\b',  # Without attribution
        r'\b(I heard|someone told me|they say)\b',
    ]
    
    # Information creation indicators
    CREATION_POSITIVE = [
        r'\b(methodology|method|approach|framework)\b',
        r'\b(sample size|participants|data collection)\b',
        r'\b(published in|appeared in|presented at)\b',
        r'\b(\d{4})\b',  # Year citation
        r'\b(version|edition|updated|revised)\b',
    ]
    
    # Value and ethics indicators
    VALUE_POSITIVE = [
        r'\b(copyright|attribution|credit|cited)\b',
        r'\b(licensed|permission|fair use)\b',
        r'\b(original|primary source|direct quote)\b',
    ]
    
    VALUE_NEGATIVE = [
        r'\b(pirated|stolen|leaked|hacked)\b',
        r'\b(without permission|uncredited)\b',
    ]
    
    # Inquiry indicators
    INQUIRY_POSITIVE = [
        r'\b(hypothesis|question|investigate|explore)\b',
        r'\b(evidence|data|findings|results)\b',
        r'\b(conclude|determine|establish|demonstrate)\b',
        r'\b(limitation|caveat|uncertainty|further research)\b',
    ]
    
    # Conversation indicators
    CONVERSATION_POSITIVE = [
        r'\b(debate|discussion|discourse|argument)\b',
        r'\b(counter[\s-]?argument|alternative view|however)\b',
        r'\b(agree|disagree|support|oppose)\b',
        r'\b(building on|in response to|extending)\b',
    ]
    
    # Exploration indicators
    EXPLORATION_POSITIVE = [
        r'\b(search|query|lookup|find)\b',
        r'\b(database|index|catalog|repository)\b',
        r'\b(keyword|term|subject|topic)\b',
        r'\b(refine|narrow|broaden|filter)\b',
    ]
    
    def __init__(self):
        # Compile patterns for efficiency
        self._compile_patterns()
        
        # Frame weights for overall score
        self.frame_weights = {
            ACRLFrame.AUTHORITY: 0.25,      # Most important for AI governance
            ACRLFrame.CREATION: 0.15,
            ACRLFrame.VALUE: 0.15,
            ACRLFrame.INQUIRY: 0.20,        # Important for factual accuracy
            ACRLFrame.CONVERSATION: 0.15,
            ACRLFrame.EXPLORATION: 0.10,
        }
    
    def _compile_patterns(self):
        """Compile regex patterns."""
        self.patterns = {
            'authority_positive': [re.compile(p, re.I) for p in self.AUTHORITY_POSITIVE],
            'authority_negative': [re.compile(p, re.I) for p in self.AUTHORITY_NEGATIVE],
            'creation_positive': [re.compile(p, re.I) for p in self.CREATION_POSITIVE],
            'value_positive': [re.compile(p, re.I) for p in self.VALUE_POSITIVE],
            'value_negative': [re.compile(p, re.I) for p in self.VALUE_NEGATIVE],
            'inquiry_positive': [re.compile(p, re.I) for p in self.INQUIRY_POSITIVE],
            'conversation_positive': [re.compile(p, re.I) for p in self.CONVERSATION_POSITIVE],
            'exploration_positive': [re.compile(p, re.I) for p in self.EXPLORATION_POSITIVE],
        }
    
    def assess(self, 
              response: str,
              query: str = "",
              sources: List[Dict] = None,
              governance_signals: Dict[str, float] = None) -> LiteracyResult:
        """
        Assess response against ACRL literacy standards.
        
        Args:
            response: AI response text
            query: Original query
            sources: List of source documents
            governance_signals: Pre-computed governance signals
        
        Returns:
            LiteracyResult with complete assessment
        """
        sources = sources or []
        governance_signals = governance_signals or {}
        
        assessments = []
        frame_scores = {}
        warnings = []
        suggestions = []
        
        # Frame 1: Authority Is Constructed and Contextual
        authority_assessment = self._assess_authority(response, sources, governance_signals)
        assessments.append(authority_assessment)
        frame_scores[ACRLFrame.AUTHORITY] = authority_assessment.score
        
        # Frame 2: Information Creation as a Process
        creation_assessment = self._assess_creation(response, sources)
        assessments.append(creation_assessment)
        frame_scores[ACRLFrame.CREATION] = creation_assessment.score
        
        # Frame 3: Information Has Value
        value_assessment = self._assess_value(response, sources)
        assessments.append(value_assessment)
        frame_scores[ACRLFrame.VALUE] = value_assessment.score
        
        # Frame 4: Research as Inquiry
        inquiry_assessment = self._assess_inquiry(response, query)
        assessments.append(inquiry_assessment)
        frame_scores[ACRLFrame.INQUIRY] = inquiry_assessment.score
        
        # Frame 5: Scholarship as Conversation
        conversation_assessment = self._assess_conversation(response)
        assessments.append(conversation_assessment)
        frame_scores[ACRLFrame.CONVERSATION] = conversation_assessment.score
        
        # Frame 6: Searching as Strategic Exploration
        exploration_assessment = self._assess_exploration(response, query)
        assessments.append(exploration_assessment)
        frame_scores[ACRLFrame.EXPLORATION] = exploration_assessment.score
        
        # Calculate overall score
        overall_score = sum(
            frame_scores[f] * self.frame_weights[f]
            for f in ACRLFrame
        )
        
        # Determine literacy level
        if overall_score >= 0.75:
            literacy_level = 'high'
        elif overall_score >= 0.55:
            literacy_level = 'adequate'
        elif overall_score >= 0.35:
            literacy_level = 'low'
        else:
            literacy_level = 'insufficient'
        
        # Generate warnings for low frame scores
        for frame, score in frame_scores.items():
            if score < 0.4:
                warnings.append(f"Low {frame.value} score ({score:.2f}): May lack {frame.value}")
        
        # Generate suggestions
        for assessment in assessments:
            if assessment.score < 0.5 and assessment.recommendation:
                suggestions.append(assessment.recommendation)
        
        return LiteracyResult(
            overall_score=overall_score,
            frame_scores=frame_scores,
            assessments=assessments,
            literacy_level=literacy_level,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _assess_authority(self,
                         response: str,
                         sources: List[Dict],
                         signals: Dict[str, float]) -> LiteracyAssessment:
        """Assess authority frame."""
        indicators_present = []
        indicators_missing = []
        
        # Check positive authority indicators
        for pattern in self.patterns['authority_positive']:
            if pattern.search(response):
                indicators_present.append(pattern.pattern)
        
        # Check negative authority indicators
        negative_count = 0
        for pattern in self.patterns['authority_negative']:
            if pattern.search(response):
                indicators_missing.append(f"Contains: {pattern.pattern}")
                negative_count += 1
        
        # Use grounding score if available
        grounding = signals.get('grounding', 0.5)
        source_trust = signals.get('source_trust', 0.5)
        
        # Calculate score
        positive_ratio = len(indicators_present) / max(len(self.patterns['authority_positive']), 1)
        negative_penalty = negative_count * 0.15
        
        score = (positive_ratio * 0.4 + grounding * 0.3 + source_trust * 0.3) - negative_penalty
        score = max(0, min(1, score))
        
        if len(indicators_present) == 0:
            indicators_missing.append("No authority indicators found")
        
        recommendation = ""
        if score < 0.5:
            recommendation = "Add source citations and author credentials to establish authority"
        
        return LiteracyAssessment(
            frame=ACRLFrame.AUTHORITY,
            score=score,
            indicators_present=indicators_present[:5],
            indicators_missing=indicators_missing[:3],
            recommendation=recommendation
        )
    
    def _assess_creation(self, response: str, sources: List[Dict]) -> LiteracyAssessment:
        """Assess information creation frame."""
        indicators_present = []
        indicators_missing = []
        
        for pattern in self.patterns['creation_positive']:
            if pattern.search(response):
                indicators_present.append(pattern.pattern)
        
        # Check for dates/versions
        has_date = bool(re.search(r'\b(19|20)\d{2}\b', response))
        has_methodology = bool(re.search(r'\b(method|approach|study|research)\b', response, re.I))
        
        if has_date:
            indicators_present.append("Contains temporal reference")
        else:
            indicators_missing.append("No date/version information")
        
        if not has_methodology:
            indicators_missing.append("No methodology mentioned")
        
        score = len(indicators_present) / 6  # Normalize
        score = max(0.1, min(1, score))
        
        recommendation = ""
        if score < 0.5:
            recommendation = "Include information about when and how content was created"
        
        return LiteracyAssessment(
            frame=ACRLFrame.CREATION,
            score=score,
            indicators_present=indicators_present[:5],
            indicators_missing=indicators_missing[:3],
            recommendation=recommendation
        )
    
    def _assess_value(self, response: str, sources: List[Dict]) -> LiteracyAssessment:
        """Assess information value frame."""
        indicators_present = []
        indicators_missing = []
        
        for pattern in self.patterns['value_positive']:
            if pattern.search(response):
                indicators_present.append(pattern.pattern)
        
        negative_count = 0
        for pattern in self.patterns['value_negative']:
            if pattern.search(response):
                indicators_missing.append(f"Concern: {pattern.pattern}")
                negative_count += 1
        
        # Check for proper attribution
        has_quotes = '"' in response or "'" in response
        has_citation = bool(re.search(r'\([^)]+\d{4}\)', response))  # (Author 2020)
        
        if has_quotes:
            indicators_present.append("Uses quotations")
        if has_citation:
            indicators_present.append("Contains citations")
        else:
            indicators_missing.append("No formal citations")
        
        score = (len(indicators_present) * 0.2 - negative_count * 0.3)
        score = max(0.2, min(1, 0.5 + score))  # Base score of 0.5
        
        recommendation = ""
        if score < 0.5:
            recommendation = "Add proper attribution and citations for sourced information"
        
        return LiteracyAssessment(
            frame=ACRLFrame.VALUE,
            score=score,
            indicators_present=indicators_present[:5],
            indicators_missing=indicators_missing[:3],
            recommendation=recommendation
        )
    
    def _assess_inquiry(self, response: str, query: str) -> LiteracyAssessment:
        """Assess research as inquiry frame."""
        indicators_present = []
        indicators_missing = []
        
        for pattern in self.patterns['inquiry_positive']:
            if pattern.search(response):
                indicators_present.append(pattern.pattern)
        
        # Check for evidence-based reasoning
        has_evidence = bool(re.search(r'\b(evidence|data|study|research)\b', response, re.I))
        has_conclusion = bool(re.search(r'\b(therefore|thus|conclude|shows|demonstrates)\b', response, re.I))
        has_limitations = bool(re.search(r'\b(however|although|limitation|caveat|but)\b', response, re.I))
        
        if has_evidence:
            indicators_present.append("References evidence")
        else:
            indicators_missing.append("No evidence cited")
        
        if has_conclusion:
            indicators_present.append("Draws conclusions")
        
        if has_limitations:
            indicators_present.append("Acknowledges limitations")
        else:
            indicators_missing.append("No limitations acknowledged")
        
        score = len(indicators_present) / 6
        score = max(0.1, min(1, score))
        
        recommendation = ""
        if score < 0.5:
            recommendation = "Include evidence-based reasoning with acknowledged limitations"
        
        return LiteracyAssessment(
            frame=ACRLFrame.INQUIRY,
            score=score,
            indicators_present=indicators_present[:5],
            indicators_missing=indicators_missing[:3],
            recommendation=recommendation
        )
    
    def _assess_conversation(self, response: str) -> LiteracyAssessment:
        """Assess scholarship as conversation frame."""
        indicators_present = []
        indicators_missing = []
        
        for pattern in self.patterns['conversation_positive']:
            if pattern.search(response):
                indicators_present.append(pattern.pattern)
        
        # Check for multiple perspectives
        has_alternative = bool(re.search(r'\b(however|alternatively|on the other hand|some argue)\b', response, re.I))
        has_debate = bool(re.search(r'\b(debate|controversy|disagreement|dispute)\b', response, re.I))
        
        if has_alternative:
            indicators_present.append("Presents alternatives")
        else:
            indicators_missing.append("No alternative viewpoints")
        
        if has_debate:
            indicators_present.append("Acknowledges debate")
        
        score = len(indicators_present) / 4
        score = max(0.2, min(1, score))
        
        recommendation = ""
        if score < 0.5:
            recommendation = "Include multiple perspectives and acknowledge ongoing debates"
        
        return LiteracyAssessment(
            frame=ACRLFrame.CONVERSATION,
            score=score,
            indicators_present=indicators_present[:5],
            indicators_missing=indicators_missing[:3],
            recommendation=recommendation
        )
    
    def _assess_exploration(self, response: str, query: str) -> LiteracyAssessment:
        """Assess searching as strategic exploration frame."""
        indicators_present = []
        indicators_missing = []
        
        for pattern in self.patterns['exploration_positive']:
            if pattern.search(response):
                indicators_present.append(pattern.pattern)
        
        # Check query relevance
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words & response_words)
        
        if overlap >= 3:
            indicators_present.append(f"Query relevance: {overlap} terms")
        elif overlap < 2:
            indicators_missing.append("Low query relevance")
        
        # Check for search strategy indicators
        has_sources = bool(re.search(r'\b(according to|from|based on|per)\b', response, re.I))
        if has_sources:
            indicators_present.append("References sources")
        
        score = len(indicators_present) / 4
        score = max(0.2, min(1, score))
        
        recommendation = ""
        if score < 0.5:
            recommendation = "Demonstrate strategic information gathering with multiple sources"
        
        return LiteracyAssessment(
            frame=ACRLFrame.EXPLORATION,
            score=score,
            indicators_present=indicators_present[:5],
            indicators_missing=indicators_missing[:3],
            recommendation=recommendation
        )
    
    def get_frame_description(self, frame: ACRLFrame) -> str:
        """Get description of an ACRL frame."""
        descriptions = {
            ACRLFrame.AUTHORITY: "Authority Is Constructed and Contextual: Information sources have varying levels of credibility depending on context.",
            ACRLFrame.CREATION: "Information Creation as a Process: Information is created through different processes with varying purposes.",
            ACRLFrame.VALUE: "Information Has Value: Information is a commodity with legal and ethical implications.",
            ACRLFrame.INQUIRY: "Research as Inquiry: Research is iterative and depends on asking good questions.",
            ACRLFrame.CONVERSATION: "Scholarship as Conversation: Knowledge emerges from ongoing discussions among experts.",
            ACRLFrame.EXPLORATION: "Searching as Strategic Exploration: Finding information requires strategic thinking and iteration."
        }
        return descriptions.get(frame, "")
    
    def integrate_with_governance(self, 
                                 literacy_result: LiteracyResult,
                                 governance_signals: Dict[str, float]) -> Dict[str, float]:
        """
        Integrate literacy assessment with governance signals.
        
        Returns enhanced signals incorporating literacy standards.
        """
        enhanced = dict(governance_signals)
        
        # Authority frame enhances source_trust
        enhanced['source_trust'] = (
            governance_signals.get('source_trust', 0.5) * 0.6 +
            literacy_result.frame_scores.get(ACRLFrame.AUTHORITY, 0.5) * 0.4
        )
        
        # Inquiry frame enhances factual
        enhanced['factual'] = (
            governance_signals.get('factual', 0.5) * 0.7 +
            literacy_result.frame_scores.get(ACRLFrame.INQUIRY, 0.5) * 0.3
        )
        
        # Conversation frame enhances behavioral (reduces bias)
        conversation_score = literacy_result.frame_scores.get(ACRLFrame.CONVERSATION, 0.5)
        # If multiple perspectives shown, reduce bias penalty
        if conversation_score > 0.6:
            enhanced['behavioral'] = min(
                governance_signals.get('behavioral', 0.5),
                governance_signals.get('behavioral', 0.5) * 0.8  # 20% reduction
            )
        
        # Add literacy score as new signal
        enhanced['literacy'] = literacy_result.overall_score
        
        return enhanced


