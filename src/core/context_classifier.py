"""
BAIS Context Classifier Module

Classifies the context of text to enable context-aware detection.
Prevents false positives by understanding the PURPOSE of the content.

Key Features:
1. Planning/Estimation context - time estimates, effort planning
2. Technical Documentation context - specs, architecture docs
3. Audit/Report context - formal assessments, reviews
4. Response context - conversational responses to users

Patent Alignment:
- PPA1-Inv7: Signal Fusion (context as a signal)
- NOVEL-28: Intelligent Dimension Selection (context-aware dimensions)
- PPA2-Inv4: Behavioral Pattern Detection (context-adjusted patterns)

Phase 16 Enhancement: Reduces false positives in metric_gaming and other patterns
by understanding that planning documents, estimates, and technical docs have
different legitimate patterns than conversational responses.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import re


class ContentContext(Enum):
    """Context classification for content analysis."""
    PLANNING = "planning"              # Project plans, estimates, roadmaps
    DOCUMENTATION = "documentation"    # Technical docs, architecture, specs
    AUDIT_REPORT = "audit_report"      # Formal assessments, reviews
    RESPONSE = "response"              # Conversational responses
    CODE = "code"                      # Source code
    MEDICAL = "medical"                # Medical advice/diagnosis
    LEGAL = "legal"                    # Legal advice/analysis
    FINANCIAL = "financial"            # Financial advice/analysis
    GENERAL = "general"                # Default


class EstimateType(Enum):
    """Types of estimates that should NOT be flagged as metric gaming."""
    TIME_ESTIMATE = "time"             # "2-3 hours", "approximately 5 days"
    EFFORT_ESTIMATE = "effort"         # "medium effort", "high complexity"
    RANGE_ESTIMATE = "range"           # "10-15%", "$500-$1000"
    ROUGH_ESTIMATE = "rough"           # "roughly", "approximately", "about"
    NONE = "none"


@dataclass
class ContextSignals:
    """Signals extracted from context analysis."""
    primary_context: ContentContext
    secondary_contexts: List[ContentContext]
    confidence: float
    
    # Planning-specific signals
    is_planning_document: bool = False
    has_time_estimates: bool = False
    has_phase_structure: bool = False
    has_task_list: bool = False
    
    # Documentation-specific signals
    is_technical_doc: bool = False
    has_architecture_terms: bool = False
    has_code_references: bool = False
    
    # Audit-specific signals
    is_audit_report: bool = False
    has_verification_language: bool = False
    has_evidence_references: bool = False
    
    # Estimate detection
    estimate_types_found: List[EstimateType] = field(default_factory=list)
    estimate_count: int = 0
    
    # Adjustments for detection
    metric_gaming_adjustment: float = 1.0  # 1.0 = no adjustment, <1.0 = reduce confidence
    listing_adjustment: float = 1.0
    structure_adjustment: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            "primary_context": self.primary_context.value,
            "secondary_contexts": [c.value for c in self.secondary_contexts],
            "confidence": self.confidence,
            "is_planning_document": self.is_planning_document,
            "has_time_estimates": self.has_time_estimates,
            "has_phase_structure": self.has_phase_structure,
            "has_task_list": self.has_task_list,
            "is_technical_doc": self.is_technical_doc,
            "is_audit_report": self.is_audit_report,
            "estimate_types_found": [e.value for e in self.estimate_types_found],
            "estimate_count": self.estimate_count,
            "metric_gaming_adjustment": self.metric_gaming_adjustment,
            "listing_adjustment": self.listing_adjustment,
            "structure_adjustment": self.structure_adjustment
        }


class ContextClassifier:
    """
    Classifies content context to enable context-aware detection.
    
    Key insight: Planning documents legitimately contain:
    - Time estimates (not metric gaming)
    - Structured lists (not padding)
    - Phase breakdowns (not formulaic gaming)
    - Percentage estimates with ranges (not inflation)
    """
    
    # Planning context indicators
    PLANNING_INDICATORS = [
        r'\bphase\s*\d+\b',                           # Phase 1, Phase 2
        r'\b(?:estimated|estimate)\s*(?:effort|time|duration)\b',
        r'\b\d+\s*-\s*\d+\s*(?:hours?|days?|weeks?)\b',  # Time ranges
        r'\b(?:task|todo|action\s*item)\b',
        r'\b(?:roadmap|timeline|milestone)\b',
        r'\b(?:priority|prioritized)\b',
        r'\b(?:step\s*\d+|task\s*\d+)\b',
        r'\benhancement\s*(?:plan|proposal)\b',
        r'\b(?:sprint|iteration|backlog)\b',
        r'\b(?:scope|deliverable|objective)\b',
    ]
    
    # Documentation context indicators
    DOCUMENTATION_INDICATORS = [
        r'\barchitecture\b',
        r'\b(?:module|component|class|function)\b',
        r'\b(?:implementation|specification)\b',
        r'\b(?:diagram|flowchart)\b',
        r'\b(?:layer\s*\d+|brain\s*layer)\b',
        r'\b(?:invention|patent|claim)\b',
        r'\b(?:interface|endpoint|api)\b',
        r'`[a-zA-Z_][a-zA-Z0-9_.]*`',                   # Code references like `module.py`
        r'\b(?:src/|\.py|\.ts|\.js)\b',
    ]
    
    # Audit/Report context indicators
    AUDIT_INDICATORS = [
        r'\b(?:audit|assessment|evaluation|review)\b',
        r'\b(?:verified|verified|tested|confirmed)\b',
        r'\b(?:evidence|proof|documentation)\b',
        r'\b(?:findings?|results?|conclusions?)\b',
        r'\b(?:compliance|conformance)\b',
        r'\b(?:track\s*[ab]|dual-track|a/b\s*test)\b',
        r'\b(?:win\s*rate|success\s*rate|accuracy)\b',
    ]
    
    # Time/effort estimate patterns (these are LEGITIMATE, not gaming)
    ESTIMATE_PATTERNS = {
        EstimateType.TIME_ESTIMATE: [
            r'\b(\d+)\s*-\s*(\d+)\s*(hours?|days?|weeks?|months?)\b',
            r'\b(?:approximately|approx\.?|about|roughly|~)\s*(\d+)\s*(hours?|days?|weeks?)\b',
            r'\bestimated\s*(?:at|to\s*be)?\s*(\d+)\s*(hours?|days?|weeks?)\b',
        ],
        EstimateType.EFFORT_ESTIMATE: [
            r'\b(?:low|medium|high)\s*(?:effort|complexity|priority)\b',
            r'\b(?:effort|complexity)\s*:\s*(?:low|medium|high)\b',
        ],
        EstimateType.RANGE_ESTIMATE: [
            r'\b(\d+)\s*-\s*(\d+)\s*%\b',              # 10-15%
            r'\$(\d+)\s*-\s*\$?(\d+)\b',               # $100-$200
            r'\b(\d+)\s*(?:to|or)\s*(\d+)\b',          # 10 to 15
        ],
        EstimateType.ROUGH_ESTIMATE: [
            r'\b(?:roughly|approximately|about|around|~)\s*\d+',
            r'\b(?:estimated|estimate)\b',
        ],
    }
    
    def __init__(self):
        """Initialize the context classifier."""
        # Compile patterns for efficiency
        self._planning_patterns = [re.compile(p, re.IGNORECASE) for p in self.PLANNING_INDICATORS]
        self._doc_patterns = [re.compile(p, re.IGNORECASE) for p in self.DOCUMENTATION_INDICATORS]
        self._audit_patterns = [re.compile(p, re.IGNORECASE) for p in self.AUDIT_INDICATORS]
        
        self._estimate_patterns = {
            est_type: [re.compile(p, re.IGNORECASE) for p in patterns]
            for est_type, patterns in self.ESTIMATE_PATTERNS.items()
        }
    
    def classify(self, text: str, query: str = None) -> ContextSignals:
        """
        Classify the context of text.
        
        Args:
            text: The text to classify (typically a response)
            query: Optional query that prompted the text
        
        Returns:
            ContextSignals with context classification and adjustments
        """
        # Count indicator matches
        planning_score = sum(1 for p in self._planning_patterns if p.search(text))
        doc_score = sum(1 for p in self._doc_patterns if p.search(text))
        audit_score = sum(1 for p in self._audit_patterns if p.search(text))
        
        # Detect estimate types
        estimate_types = []
        estimate_count = 0
        for est_type, patterns in self._estimate_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    estimate_types.append(est_type)
                    estimate_count += len(matches)
        
        # Remove duplicates from estimate types
        estimate_types = list(set(estimate_types))
        
        # Determine primary context
        scores = {
            ContentContext.PLANNING: planning_score,
            ContentContext.DOCUMENTATION: doc_score,
            ContentContext.AUDIT_REPORT: audit_score,
        }
        
        # Check query for additional context
        if query:
            query_lower = query.lower()
            if any(w in query_lower for w in ['plan', 'estimate', 'timeline', 'roadmap', 'phase']):
                scores[ContentContext.PLANNING] += 3
            if any(w in query_lower for w in ['document', 'architecture', 'spec', 'design']):
                scores[ContentContext.DOCUMENTATION] += 3
            if any(w in query_lower for w in ['audit', 'review', 'assess', 'verify', 'test']):
                scores[ContentContext.AUDIT_REPORT] += 3
        
        # Find primary context
        max_score = max(scores.values())
        if max_score >= 2:
            primary_context = max(scores, key=scores.get)
            confidence = min(0.9, 0.5 + (max_score * 0.1))
        else:
            primary_context = ContentContext.GENERAL
            confidence = 0.5
        
        # Secondary contexts
        secondary_contexts = [
            ctx for ctx, score in scores.items()
            if score >= 1 and ctx != primary_context
        ]
        
        # Build signals
        signals = ContextSignals(
            primary_context=primary_context,
            secondary_contexts=secondary_contexts,
            confidence=confidence,
            
            # Planning signals
            is_planning_document=planning_score >= 3,
            has_time_estimates=EstimateType.TIME_ESTIMATE in estimate_types,
            has_phase_structure=bool(re.search(r'\bphase\s*\d+', text, re.IGNORECASE)),
            has_task_list=bool(re.search(r'^\s*(?:\d+\.|[-•*]|□)\s*', text, re.MULTILINE)),
            
            # Documentation signals
            is_technical_doc=doc_score >= 3,
            has_architecture_terms=bool(re.search(r'\b(?:layer|module|component)\b', text, re.IGNORECASE)),
            has_code_references=bool(re.search(r'`[^`]+`|\.py\b|\.ts\b', text)),
            
            # Audit signals
            is_audit_report=audit_score >= 3,
            has_verification_language=bool(re.search(r'\bverified\b|\bconfirmed\b|\btested\b', text, re.IGNORECASE)),
            has_evidence_references=bool(re.search(r'\bevidence\b|\bproof\b', text, re.IGNORECASE)),
            
            # Estimates
            estimate_types_found=estimate_types,
            estimate_count=estimate_count,
        )
        
        # Calculate adjustments based on context
        signals.metric_gaming_adjustment = self._calculate_metric_gaming_adjustment(signals)
        signals.listing_adjustment = self._calculate_listing_adjustment(signals)
        signals.structure_adjustment = self._calculate_structure_adjustment(signals)
        
        return signals
    
    def _calculate_metric_gaming_adjustment(self, signals: ContextSignals) -> float:
        """
        Calculate adjustment factor for metric gaming detection.
        
        Returns value < 1.0 to REDUCE false positives in legitimate contexts.
        """
        adjustment = 1.0
        
        # Planning documents with time estimates are NOT gaming
        if signals.is_planning_document:
            adjustment *= 0.4  # Reduce confidence by 60%
        
        # Time estimates with ranges show uncertainty, not gaming
        if signals.has_time_estimates:
            adjustment *= 0.3  # Reduce confidence by 70%
        
        # Range estimates are NOT gaming - they show honest uncertainty
        if EstimateType.RANGE_ESTIMATE in signals.estimate_types_found:
            adjustment *= 0.4
        
        # Rough estimates with hedging language are NOT gaming
        if EstimateType.ROUGH_ESTIMATE in signals.estimate_types_found:
            adjustment *= 0.5
        
        # Technical documentation with architecture terms
        if signals.is_technical_doc:
            adjustment *= 0.6
        
        # Audit reports with evidence references
        if signals.is_audit_report and signals.has_evidence_references:
            adjustment *= 0.5
        
        return max(0.1, adjustment)  # Floor at 0.1
    
    def _calculate_listing_adjustment(self, signals: ContextSignals) -> float:
        """
        Calculate adjustment for listing detection.
        
        Planning documents and technical docs legitimately have many list items.
        """
        adjustment = 1.0
        
        if signals.is_planning_document or signals.has_task_list:
            adjustment *= 0.3  # Planning docs can have many items
        
        if signals.is_technical_doc:
            adjustment *= 0.4  # Tech docs often have extensive lists
        
        if signals.is_audit_report:
            adjustment *= 0.5  # Audit reports list findings
        
        return max(0.1, adjustment)
    
    def _calculate_structure_adjustment(self, signals: ContextSignals) -> float:
        """
        Calculate adjustment for formulaic structure detection.
        
        Structured documents are SUPPOSED to be formulaic.
        """
        adjustment = 1.0
        
        if signals.has_phase_structure:
            adjustment *= 0.2  # Phase structure is intentional, not gaming
        
        if signals.is_planning_document:
            adjustment *= 0.3
        
        if signals.is_technical_doc:
            adjustment *= 0.4
        
        return max(0.1, adjustment)
    
    def should_skip_metric_gaming(self, text: str, query: str = None) -> Tuple[bool, str]:
        """
        Quick check: Should we skip metric gaming detection entirely?
        
        Returns:
            (should_skip, reason)
        """
        signals = self.classify(text, query)
        
        # Planning documents with time estimates
        if signals.is_planning_document and signals.has_time_estimates:
            return True, "Planning document with legitimate time estimates"
        
        # Multiple range estimates indicate uncertainty, not gaming
        if signals.estimate_count >= 3 and EstimateType.RANGE_ESTIMATE in signals.estimate_types_found:
            return True, "Multiple range estimates indicate honest uncertainty"
        
        # Technical documentation with code references
        if signals.is_technical_doc and signals.has_code_references:
            return True, "Technical documentation with implementation details"
        
        return False, ""
    
    def get_adjusted_threshold(self, 
                               base_threshold: float, 
                               signals: ContextSignals,
                               pattern_type: str) -> float:
        """
        Get adjusted detection threshold based on context.
        
        Args:
            base_threshold: Original detection threshold
            signals: Context signals
            pattern_type: Type of pattern being detected
        
        Returns:
            Adjusted threshold (higher = stricter, fewer detections)
        """
        if pattern_type == 'metric_gaming':
            # INCREASE threshold (make it harder to trigger) for legitimate contexts
            if signals.metric_gaming_adjustment < 1.0:
                # Inverse: lower adjustment = higher threshold
                return base_threshold / signals.metric_gaming_adjustment
        
        elif pattern_type == 'excessive_listing':
            if signals.listing_adjustment < 1.0:
                return base_threshold / signals.listing_adjustment
        
        elif pattern_type == 'formulaic_structure':
            if signals.structure_adjustment < 1.0:
                return base_threshold / signals.structure_adjustment
        
        return base_threshold

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


# Singleton instance for easy import
context_classifier = ContextClassifier()

