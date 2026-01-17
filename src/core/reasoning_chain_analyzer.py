"""
BASE Reasoning Chain Analyzer

Detects structural reasoning problems WITHOUT requiring domain knowledge.
Analyzes the logical flow of arguments to detect:
- Anchoring bias (first hypothesis → conclusion without alternatives)
- Selective reasoning (all evidence points same direction)
- Premature certainty (weak evidence → strong conclusion)
- Missing differential (single answer without considering alternatives)

Patent Alignment:
- NOVEL-14: Multi-Framework Convergence (reasoning structure analysis)
- NOVEL-15: Theory of Mind (understanding reasoning intent)
- PPA2-Inv4: Behavioral Pattern Detection (reasoning patterns)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
import re


class ContentType(Enum):
    """
    Content type classification for reasoning analysis.
    
    Phase 15 Enhancement: Factual content doesn't require alternatives.
    """
    FACTUAL = "factual"              # Objective facts - no alternatives needed
    ANALYTICAL = "analytical"        # Analysis requiring consideration of alternatives
    RECOMMENDATION = "recommendation"  # Advice requiring alternatives
    DIAGNOSTIC = "diagnostic"        # Diagnosis requiring differential
    OPINION = "opinion"              # Subjective opinion
    UNKNOWN = "unknown"              # Cannot classify


class ReasoningIssueType(Enum):
    """Types of reasoning issues detected."""
    ANCHORING = "anchoring"                    # First hypothesis → conclusion
    SELECTIVE_REASONING = "selective"          # All evidence one direction
    PREMATURE_CERTAINTY = "premature"          # Weak evidence → strong conclusion
    MISSING_ALTERNATIVES = "no_alternatives"   # No differential/alternatives
    CIRCULAR = "circular"                      # Conclusion in premises
    NON_SEQUITUR = "non_sequitur"             # Conclusion doesn't follow
    APPEAL_TO_AUTHORITY = "authority"          # "Expert says" without evidence
    CONFIRMATION_BIAS = "confirmation"         # Only supporting evidence cited


class ReasoningStrength(Enum):
    """Strength of reasoning chain."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


@dataclass
class ReasoningElement:
    """A single element in a reasoning chain."""
    element_type: str                # "observation", "hypothesis", "evidence", "conclusion"
    content: str
    position: int                    # Position in text
    strength_words: List[str]        # Words indicating confidence level
    hedging_words: List[str]         # Words indicating uncertainty


@dataclass
class ReasoningChain:
    """A complete reasoning chain extracted from text."""
    observations: List[ReasoningElement]
    hypotheses: List[ReasoningElement]
    evidence_for: List[ReasoningElement]      # Supporting evidence
    evidence_against: List[ReasoningElement]  # Contrary evidence
    conclusions: List[ReasoningElement]
    alternatives_mentioned: List[str]
    
    def get_balance_ratio(self) -> float:
        """Get ratio of evidence_for to evidence_against."""
        if not self.evidence_against:
            return 1.0 if self.evidence_for else 0.5
        return len(self.evidence_for) / (len(self.evidence_for) + len(self.evidence_against))


@dataclass
class ReasoningIssue:
    """A detected reasoning issue."""
    issue_type: ReasoningIssueType
    severity: float                  # 0.0 - 1.0
    description: str
    evidence: List[str]              # Quotes from text supporting this issue
    recommendation: str


@dataclass
class ReasoningAnalysisResult:
    """Complete reasoning analysis result."""
    chain: ReasoningChain
    issues: List[ReasoningIssue]
    overall_strength: ReasoningStrength
    confidence_mismatch: float       # Gap between stated confidence and evidence
    has_alternatives: bool
    has_contrary_evidence: bool
    anchoring_score: float           # 0.0 = no anchoring, 1.0 = severe anchoring
    selectivity_score: float         # 0.0 = balanced, 1.0 = completely one-sided
    warnings: List[str]
    content_type: str = "unknown"    # Content type classification (factual, analytical, etc.)
    
    def to_dict(self) -> Dict:
        return {
            "overall_strength": self.overall_strength.value,
            "content_type": self.content_type,
            "confidence_mismatch": self.confidence_mismatch,
            "has_alternatives": self.has_alternatives,
            "has_contrary_evidence": self.has_contrary_evidence,
            "anchoring_score": self.anchoring_score,
            "selectivity_score": self.selectivity_score,
            "issues": [
                {
                    "type": i.issue_type.value,
                    "severity": i.severity,
                    "description": i.description,
                    "recommendation": i.recommendation
                }
                for i in self.issues
            ],
            "warnings": self.warnings
        }


class ReasoningChainAnalyzer:
    """
    Analyzes reasoning structure without domain knowledge.
    
    Detects problems in HOW arguments are structured, not WHETHER
    the domain-specific content is correct.
    """
    
    # Indicators of high confidence in conclusions
    HIGH_CONFIDENCE_WORDS = [
        'clearly', 'obviously', 'definitely', 'certainly', 'undoubtedly',
        'without doubt', 'no question', 'conclusively', 'unquestionably',
        'absolutely', 'surely', 'plainly', '100%', 'proven', 'established',
        'confirmed', 'definitive', 'certain'
    ]
    
    # Indicators of hedging/uncertainty
    HEDGING_WORDS = [
        'might', 'may', 'could', 'possibly', 'perhaps', 'potentially',
        'likely', 'probably', 'appears', 'seems', 'suggests', 'indicates',
        'consider', 'evaluate', 'assess', 'uncertain', 'unclear'
    ]
    
    # Words indicating alternatives/differential
    ALTERNATIVE_INDICATORS = [
        'alternatively', 'however', 'on the other hand', 'but', 'although',
        'despite', 'conversely', 'whereas', 'while', 'instead', 'rather',
        'or', 'either', 'differential', 'rule out', 'consider also',
        'other possibilities', 'alternative', 'versus', 'vs'
    ]
    
    # Words indicating supporting evidence
    SUPPORT_INDICATORS = [
        'supports', 'confirms', 'proves', 'demonstrates', 'shows',
        'indicates', 'consistent with', 'in line with', 'agrees with',
        'validates', 'verifies', 'establishes', 'reinforces'
    ]
    
    # Words indicating contrary evidence
    CONTRARY_INDICATORS = [
        'however', 'but', 'although', 'despite', 'contrary to',
        'inconsistent with', 'conflicts with', 'contradicts', 'against',
        'opposes', 'challenges', 'undermines', 'weakens', 'refutes',
        'nevertheless', 'nonetheless', 'yet', 'still'
    ]
    
    # Conclusion indicators
    CONCLUSION_INDICATORS = [
        'therefore', 'thus', 'hence', 'consequently', 'so',
        'in conclusion', 'to conclude', 'it follows that',
        'we conclude', 'the conclusion is', 'this means',
        'as a result', 'accordingly', 'diagnosis:', 'verdict:',
        'recommendation:', 'decision:'
    ]
    
    # Hypothesis indicators
    HYPOTHESIS_INDICATORS = [
        'this is', 'this appears to be', 'likely', 'probably',
        'diagnosis', 'assessment', 'impression', 'finding',
        'conclusion', 'determination', 'it is'
    ]
    
    # Phase 15: Factual content indicators (no alternatives needed)
    FACTUAL_INDICATORS = [
        r"\bcapital\s+of\b",           # "capital of X is Y"
        r"\bis\s+(?:a|an|the)\s+\w+\b",  # "X is a/an/the Y"
        r"\b(?:was|were)\s+born\b",      # "born in X"
        r"\b(?:located|situated)\s+in\b",
        r"\bfounded\s+in\b",
        r"\b\d{4}\b",                   # Year - often factual
        r"\bpopulation\s+of\b",
        r"\bequals?\b",                 # Mathematical facts
        r"\bdefined\s+as\b",
        r"\bknown\s+as\b",
    ]
    
    # Phase 15: Analytical/recommendation indicators (alternatives needed)
    ANALYTICAL_INDICATORS = [
        r"\bshould\b",
        r"\brecommend\b",
        r"\badvise\b",
        r"\bsuggest\b",
        r"\bconsider\b",
        r"\bmight\s+want\s+to\b",
        r"\bbest\s+(?:option|approach|practice)\b",
        r"\bdiagnosis\b",
        r"\btreatment\b",
        r"\bprognosis\b",
        # Phase 16: Estimation patterns (subject to anchoring)
        r"\bestimate\b",
        r"\binitial\s+assessment\b",
        r"\bfirst\s+thought\b",
        r"\boriginal\s+estimate\b",
        r"\bbased\s+on\s+my\b",
    ]
    
    def __init__(self):
        """Initialize the reasoning chain analyzer."""
        # Learning state
        self._outcomes: List[Dict] = []
        self._feedback: List[Dict] = []
        self._domain_adjustments: Dict[str, float] = {}
        self._issue_accuracy: Dict[str, Dict] = {}  # Track accuracy per issue type
        self._total_analyses: int = 0
        self._correct_analyses: int = 0
    
    # ===== Learning Interface Methods =====
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        """
        Record the outcome of a reasoning analysis for learning.
        
        Args:
            outcome: Dict with 'text', 'predicted_issues', 'actual_issues', 'correct'
        """
        self._outcomes.append(outcome)
        self._total_analyses += 1
        if outcome.get('correct', False):
            self._correct_analyses += 1
        
        # Track per-issue accuracy
        for issue_type in outcome.get('predicted_issues', []):
            if issue_type not in self._issue_accuracy:
                self._issue_accuracy[issue_type] = {'correct': 0, 'total': 0}
            self._issue_accuracy[issue_type]['total'] += 1
            if outcome.get('correct'):
                self._issue_accuracy[issue_type]['correct'] += 1
    
    def record_feedback(self, feedback: Dict) -> None:
        """
        Record human feedback on reasoning analysis.
        
        Args:
            feedback: Dict with 'text', 'feedback_type', 'issue_corrections', 'domain'
        """
        self._feedback.append(feedback)
        
        domain = feedback.get('domain', 'general')
        if feedback.get('feedback_type') == 'false_positive':
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) - 0.05
        elif feedback.get('feedback_type') == 'false_negative':
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) + 0.05
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """
        Adapt detection thresholds based on performance.
        
        Args:
            domain: Domain to adapt thresholds for
            performance_data: Optional performance metrics
        """
        if performance_data is None:
            return
        
        false_positive_rate = performance_data.get('false_positive_rate', 0.0)
        false_negative_rate = performance_data.get('false_negative_rate', 0.0)
        
        if false_positive_rate > 0.2:
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) - 0.1
        if false_negative_rate > 0.2:
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) + 0.1
    
    def get_domain_adjustment(self, domain: str) -> float:
        """
        Get the threshold adjustment for a specific domain.
        
        Args:
            domain: Domain to get adjustment for
            
        Returns:
            Adjustment factor (-1.0 to 1.0)
        """
        return self._domain_adjustments.get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict:
        """
        Get statistics about the learning process.
        
        Returns:
            Dict with learning statistics
        """
        accuracy = self._correct_analyses / max(self._total_analyses, 1)
        return {
            'total_analyses': self._total_analyses,
            'correct_analyses': self._correct_analyses,
            'accuracy': accuracy,
            'total_feedback': len(self._feedback),
            'domain_adjustments': dict(self._domain_adjustments),
            'issue_accuracy': dict(self._issue_accuracy),
            'outcomes_recorded': len(self._outcomes)
        }
    
    def _classify_content_type(self, text: str) -> ContentType:
        """
        Classify the content type to determine if alternatives are required.
        
        Phase 15 Enhancement: Factual content doesn't need alternatives.
        "The capital of France is Paris" is factual - no alternatives needed.
        "The diagnosis is lupus" is diagnostic - alternatives needed.
        
        Args:
            text: Text to classify
        
        Returns:
            ContentType enum value
        """
        text_lower = text.lower()
        
        # Check length - very short responses are usually factual
        if len(text) < 50:
            # Additional check: does it look like a simple factual statement?
            if not any(re.search(p, text_lower) for p in self.ANALYTICAL_INDICATORS):
                return ContentType.FACTUAL
        
        # Check for analytical/recommendation indicators
        analytical_score = sum(1 for p in self.ANALYTICAL_INDICATORS if re.search(p, text_lower))
        factual_score = sum(1 for p in self.FACTUAL_INDICATORS if re.search(p, text_lower))
        
        # Strong analytical indicators
        if analytical_score >= 2:
            if 'diagnos' in text_lower:
                return ContentType.DIAGNOSTIC
            if any(w in text_lower for w in ['recommend', 'advise', 'suggest', 'should']):
                return ContentType.RECOMMENDATION
            return ContentType.ANALYTICAL
        
        # Strong factual indicators
        if factual_score >= 2 or (factual_score >= 1 and analytical_score == 0):
            return ContentType.FACTUAL
        
        # Default based on length and complexity
        sentences = len(re.findall(r'[.!?]', text))
        if sentences <= 2 and analytical_score == 0:
            return ContentType.FACTUAL
        
        return ContentType.UNKNOWN
    
    def analyze(self, text: str, domain: str = "general") -> ReasoningAnalysisResult:
        """
        Analyze reasoning structure in text.
        
        Phase 15 Enhancement: Content-type aware analysis.
        Factual content is not flagged for missing alternatives.
        
        Args:
            text: Text to analyze
            domain: Domain context (for domain-appropriate recommendations)
        
        Returns:
            ReasoningAnalysisResult with detected issues
        """
        # Phase 15: Classify content type FIRST
        content_type = self._classify_content_type(text)
        
        # Extract reasoning chain
        chain = self._extract_chain(text)
        
        # Detect issues
        issues = []
        warnings = []
        
        # Check for anchoring (only for analytical content)
        if content_type not in [ContentType.FACTUAL]:
            anchoring = self._detect_anchoring(chain, text)
            if anchoring:
                issues.append(anchoring)
        
        # Check for selective reasoning (only for analytical content)
        if content_type not in [ContentType.FACTUAL]:
            selective = self._detect_selective_reasoning(chain, text)
            if selective:
                issues.append(selective)
        
        # Check for premature certainty (reduced severity for factual content)
        premature = self._detect_premature_certainty(chain, text)
        if premature:
            if content_type == ContentType.FACTUAL:
                # Reduce severity significantly for factual content
                premature.severity *= 0.3
            if premature.severity >= 0.3:  # Only include if still meaningful
                issues.append(premature)
        
        # Phase 15: SKIP missing alternatives check for FACTUAL content
        # "The capital of France is Paris" doesn't need alternatives
        if content_type in [ContentType.ANALYTICAL, ContentType.DIAGNOSTIC, ContentType.RECOMMENDATION]:
            missing_alt = self._detect_missing_alternatives(chain, text, domain)
            if missing_alt:
                issues.append(missing_alt)
        
        # Check for confirmation bias (only for analytical content)
        if content_type not in [ContentType.FACTUAL]:
            confirmation = self._detect_confirmation_bias(chain, text)
            if confirmation:
                issues.append(confirmation)
        
        # Calculate scores
        anchoring_score = self._calculate_anchoring_score(chain)
        selectivity_score = self._calculate_selectivity_score(chain)
        confidence_mismatch = self._calculate_confidence_mismatch(chain, text)
        
        # Phase 15: Adjust scores for factual content
        if content_type == ContentType.FACTUAL:
            anchoring_score *= 0.3
            selectivity_score *= 0.3
            confidence_mismatch *= 0.3
        
        # Determine overall strength
        if len(issues) >= 3 or any(i.severity > 0.8 for i in issues):
            overall_strength = ReasoningStrength.WEAK
        elif len(issues) >= 1 or anchoring_score > 0.5:
            overall_strength = ReasoningStrength.MODERATE
        else:
            overall_strength = ReasoningStrength.STRONG
        
        # Generate warnings (content-type aware)
        if anchoring_score > 0.7 and content_type != ContentType.FACTUAL:
            warnings.append("High anchoring detected: First hypothesis treated as conclusion")
        if selectivity_score > 0.8 and content_type != ContentType.FACTUAL:
            warnings.append("Selective reasoning: All evidence supports single conclusion")
        if confidence_mismatch > 0.5 and content_type != ContentType.FACTUAL:
            warnings.append("Confidence mismatch: Stated certainty exceeds evidence strength")
        
        # Phase 15: Only flag missing alternatives for content that requires them
        requires_alternatives = content_type in [
            ContentType.ANALYTICAL, 
            ContentType.DIAGNOSTIC, 
            ContentType.RECOMMENDATION
        ]
        if not chain.alternatives_mentioned and requires_alternatives:
            warnings.append("No alternatives considered: Missing differential analysis")
        
        return ReasoningAnalysisResult(
            chain=chain,
            issues=issues,
            overall_strength=overall_strength,
            confidence_mismatch=confidence_mismatch,
            has_alternatives=len(chain.alternatives_mentioned) > 0,
            has_contrary_evidence=len(chain.evidence_against) > 0,
            anchoring_score=anchoring_score,
            selectivity_score=selectivity_score,
            warnings=warnings,
            content_type=content_type.value if hasattr(content_type, 'value') else str(content_type)
        )
    
    def _extract_chain(self, text: str) -> ReasoningChain:
        """Extract reasoning chain elements from text."""
        text_lower = text.lower()
        sentences = self._split_sentences(text)
        
        observations = []
        hypotheses = []
        evidence_for = []
        evidence_against = []
        conclusions = []
        alternatives = []
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Classify sentence
            has_support = any(w in sentence_lower for w in self.SUPPORT_INDICATORS)
            has_contrary = any(w in sentence_lower for w in self.CONTRARY_INDICATORS)
            has_conclusion = any(w in sentence_lower for w in self.CONCLUSION_INDICATORS)
            has_hypothesis = any(w in sentence_lower for w in self.HYPOTHESIS_INDICATORS)
            has_alternative = any(w in sentence_lower for w in self.ALTERNATIVE_INDICATORS)
            
            strength_words = [w for w in self.HIGH_CONFIDENCE_WORDS if w in sentence_lower]
            hedging_words = [w for w in self.HEDGING_WORDS if w in sentence_lower]
            
            element = ReasoningElement(
                element_type="unknown",
                content=sentence,
                position=i,
                strength_words=strength_words,
                hedging_words=hedging_words
            )
            
            if has_conclusion:
                element.element_type = "conclusion"
                conclusions.append(element)
            elif has_hypothesis and not has_support and not has_contrary:
                element.element_type = "hypothesis"
                hypotheses.append(element)
            elif has_support and not has_contrary:
                element.element_type = "evidence_for"
                evidence_for.append(element)
            elif has_contrary:
                element.element_type = "evidence_against"
                evidence_against.append(element)
            else:
                element.element_type = "observation"
                observations.append(element)
            
            if has_alternative:
                alternatives.append(sentence)
        
        return ReasoningChain(
            observations=observations,
            hypotheses=hypotheses,
            evidence_for=evidence_for,
            evidence_against=evidence_against,
            conclusions=conclusions,
            alternatives_mentioned=alternatives
        )
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _detect_anchoring(self, chain: ReasoningChain, text: str) -> Optional[ReasoningIssue]:
        """
        Detect anchoring bias: First hypothesis treated as conclusion.
        
        Phase 16 Enhancement: Also detects numeric anchoring patterns where
        an initial number heavily influences the final estimate.
        """
        text_lower = text.lower()
        
        # Pattern 1: Numeric anchoring (initial estimate influences final)
        numeric_anchoring_patterns = [
            # "initial assessment of X, ... estimate X-Y"
            (r'initial (?:assessment|estimate|thought|impression)[^.]*?(\d+)[^.]*?(?:estimate|conclude|think)[^.]*?(\d+)', 'initial assessment'),
            # "based on X, ... Y weeks/days"
            (r'based on[^.]*?(\d+)[^.]*?(?:weeks?|days?|hours?|months?)[^.]*?(\d+)', 'based on'),
            # "first thought X, ... X-Y"
            (r'(?:first|my|our) (?:thought|guess|estimate)[^.]*?(\d+)[^.]*?(?:now|final|conclude)[^.]*?(\d+)', 'first thought'),
            # "original estimate X ... revised to X-Y"
            (r'original[^.]*?(\d+)[^.]*?(?:revise|update|adjust)[^.]*?(\d+)', 'original estimate'),
        ]
        
        for pattern, anchor_type in numeric_anchoring_patterns:
            match = re.search(pattern, text_lower)
            if match:
                initial = int(match.group(1))
                final = int(match.group(2))
                
                # If final is within 50% of initial, it's likely anchored
                if 0.5 <= final / max(initial, 1) <= 1.5:
                    return ReasoningIssue(
                        issue_type=ReasoningIssueType.ANCHORING,
                        severity=0.7,
                        description=f"Numeric anchoring detected: {anchor_type} of {initial} influenced final estimate of {final}",
                        evidence=[match.group(0)[:150]],
                        recommendation="Consider estimates independently without reference to initial values"
                    )
        
        # Pattern 2: Original hypothesis-conclusion overlap detection
        if not chain.hypotheses or not chain.conclusions:
            return None
        
        # Check if first hypothesis is essentially the same as conclusion
        first_hyp = chain.hypotheses[0].content.lower()
        
        for conclusion in chain.conclusions:
            conc_lower = conclusion.content.lower()
            
            # Check for significant overlap
            hyp_words = set(first_hyp.split())
            conc_words = set(conc_lower.split())
            overlap = len(hyp_words & conc_words) / max(len(hyp_words), 1)
            
            if overlap > 0.5:
                # Check if alternatives were considered
                if not chain.alternatives_mentioned:
                    return ReasoningIssue(
                        issue_type=ReasoningIssueType.ANCHORING,
                        severity=0.8,
                        description="First hypothesis presented as conclusion without considering alternatives",
                        evidence=[first_hyp[:100], conclusion.content[:100]],
                        recommendation="Consider alternative hypotheses before concluding"
                    )
        
        return None
    
    def _detect_selective_reasoning(self, chain: ReasoningChain, text: str) -> Optional[ReasoningIssue]:
        """Detect selective reasoning: Only evidence supporting conclusion cited."""
        if not chain.evidence_for:
            return None
        
        # If there's supporting evidence but no contrary evidence mentioned
        if chain.evidence_for and not chain.evidence_against:
            # Check if the conclusion is strong
            has_strong_conclusion = any(
                c.strength_words for c in chain.conclusions
            )
            
            if has_strong_conclusion:
                return ReasoningIssue(
                    issue_type=ReasoningIssueType.SELECTIVE_REASONING,
                    severity=0.7,
                    description="Only supporting evidence cited; no contrary evidence considered",
                    evidence=[e.content[:100] for e in chain.evidence_for[:3]],
                    recommendation="Address potential contrary evidence or limitations"
                )
        
        return None
    
    def _detect_premature_certainty(self, chain: ReasoningChain, text: str) -> Optional[ReasoningIssue]:
        """Detect premature certainty: Strong conclusion from weak evidence."""
        text_lower = text.lower()
        
        # Count high confidence words
        high_conf_count = sum(1 for w in self.HIGH_CONFIDENCE_WORDS if w in text_lower)
        
        # Count hedging words
        hedge_count = sum(1 for w in self.HEDGING_WORDS if w in text_lower)
        
        # Count evidence elements
        evidence_count = len(chain.evidence_for) + len(chain.evidence_against)
        
        # Check for strong conclusions with weak evidence
        if high_conf_count > 0 and evidence_count < 2 and hedge_count == 0:
            return ReasoningIssue(
                issue_type=ReasoningIssueType.PREMATURE_CERTAINTY,
                severity=0.75,
                description="Strong certainty expressed with limited supporting evidence",
                evidence=[f"High confidence words: {high_conf_count}, Evidence elements: {evidence_count}"],
                recommendation="Add more evidence or use hedging language to match evidence strength"
            )
        
        return None
    
    def _detect_missing_alternatives(self, chain: ReasoningChain, text: str, domain: str) -> Optional[ReasoningIssue]:
        """Detect missing alternatives/differential."""
        if chain.alternatives_mentioned:
            return None
        
        # Check if this is a diagnostic/decision context where alternatives matter
        diagnostic_indicators = [
            'diagnosis', 'diagnose', 'assessment', 'evaluate', 'determine',
            'conclude', 'decision', 'recommend', 'prescribe', 'treatment',
            'solution', 'answer', 'verdict', 'ruling', 'finding'
        ]
        
        text_lower = text.lower()
        is_diagnostic_context = any(w in text_lower for w in diagnostic_indicators)
        
        if is_diagnostic_context and chain.conclusions:
            return ReasoningIssue(
                issue_type=ReasoningIssueType.MISSING_ALTERNATIVES,
                severity=0.65,
                description="Single conclusion reached without considering alternatives",
                evidence=[c.content[:100] for c in chain.conclusions[:2]],
                recommendation="Consider differential diagnosis or alternative explanations"
            )
        
        return None
    
    def _detect_confirmation_bias(self, chain: ReasoningChain, text: str) -> Optional[ReasoningIssue]:
        """Detect confirmation bias pattern."""
        if not chain.hypotheses or not chain.evidence_for:
            return None
        
        # Check if all evidence supports the first hypothesis
        balance = chain.get_balance_ratio()
        
        if balance > 0.9 and len(chain.evidence_for) >= 2:
            return ReasoningIssue(
                issue_type=ReasoningIssueType.CONFIRMATION_BIAS,
                severity=0.6,
                description="All cited evidence supports initial hypothesis; potential confirmation bias",
                evidence=[f"Support ratio: {balance:.0%}"],
                recommendation="Actively seek disconfirming evidence"
            )
        
        return None
    
    def _calculate_anchoring_score(self, chain: ReasoningChain) -> float:
        """Calculate anchoring score (0.0 = no anchoring, 1.0 = severe)."""
        if not chain.hypotheses or not chain.conclusions:
            return 0.0
        
        # Factors that increase anchoring score
        score = 0.0
        
        # Single hypothesis → single conclusion
        if len(chain.hypotheses) == 1 and len(chain.conclusions) == 1:
            score += 0.3
        
        # No alternatives mentioned
        if not chain.alternatives_mentioned:
            score += 0.3
        
        # No contrary evidence
        if not chain.evidence_against:
            score += 0.2
        
        # First hypothesis appears early and conclusion late
        if chain.hypotheses[0].position < 3:
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_selectivity_score(self, chain: ReasoningChain) -> float:
        """Calculate selectivity score (0.0 = balanced, 1.0 = one-sided)."""
        total_evidence = len(chain.evidence_for) + len(chain.evidence_against)
        
        if total_evidence == 0:
            return 0.5  # Neutral if no evidence
        
        return len(chain.evidence_for) / total_evidence
    
    def _calculate_confidence_mismatch(self, chain: ReasoningChain, text: str) -> float:
        """Calculate gap between stated confidence and evidence strength."""
        text_lower = text.lower()
        
        # Measure stated confidence
        high_conf_count = sum(1 for w in self.HIGH_CONFIDENCE_WORDS if w in text_lower)
        hedge_count = sum(1 for w in self.HEDGING_WORDS if w in text_lower)
        
        stated_confidence = high_conf_count / (high_conf_count + hedge_count + 1)
        
        # Measure evidence strength
        evidence_count = len(chain.evidence_for) + len(chain.evidence_against)
        has_alternatives = len(chain.alternatives_mentioned) > 0
        has_contrary = len(chain.evidence_against) > 0
        
        evidence_strength = min(evidence_count / 5, 1.0)  # Cap at 5 pieces of evidence
        if has_alternatives:
            evidence_strength += 0.1
        if has_contrary:
            evidence_strength += 0.1
        
        evidence_strength = min(evidence_strength, 1.0)
        
        # Mismatch is when confidence exceeds evidence
        mismatch = max(0, stated_confidence - evidence_strength)
        
        return mismatch


    # =========================================================================
    # Learning Interface (5/5 methods) - Completing interface
    # =========================================================================
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback to adjust behavior."""
        if not hasattr(self, '_learning_params'):
            self._learning_params = {}
        was_correct = feedback.get('was_correct', True)
        if not was_correct:
            adjustment = feedback.get('adjustment', 0.05)
            self._learning_params['threshold'] = self._learning_params.get('threshold', 0.5) * (1 - adjustment)
    
    def get_statistics(self) -> Dict:
        """Return learning statistics."""
        return {
            'total_operations': getattr(self, '_total_operations', 0),
            'learning_params': getattr(self, '_learning_params', {}),
            'history_size': len(getattr(self, '_outcomes', []))
        }
    
    def serialize_state(self) -> Dict:
        """Serialize learning state for persistence."""
        return {
            'outcomes': getattr(self, '_outcomes', [])[-100:],
            'learning_params': getattr(self, '_learning_params', {}),
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore learning state from serialized data."""
        self._outcomes = state.get('outcomes', [])
        self._learning_params = state.get('learning_params', {})
