"""
BAIS Domain Pattern Learning Module

Learns domain-specific patterns from exposure and tracks their effectiveness.
Patterns are discovered through analysis, not prescribed.

Key Features:
1. Common patterns (apply to all domains)
2. Domain-specific patterns (learned from exposure)
3. Effectiveness tracking (true positive/false positive rates)
4. Adaptive thresholds per domain
5. Pattern discovery from Multi-Track LLM feedback

Patent Alignment:
- PPA3-Inv1: State Machine with Hysteresis (pattern states)
- PPA3-Inv2: Adaptive Thresholds (domain-specific adjustments)
- NOVEL-30: Dimensional Learning (pattern-to-dimension mapping)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from enum import Enum
import json
import re


class PatternType(Enum):
    """Types of patterns BAIS detects."""
    COMMON = "common"                    # Apply to all domains
    DOMAIN_SPECIFIC = "domain_specific"  # Learned per domain
    CIRCUMSTANCE = "circumstance"        # Context-dependent


class PatternCategory(Enum):
    """Categories of patterns."""
    FALSE_COMPLETION = "false_completion"
    OVERCONFIDENCE = "overconfidence"
    TGTBT = "tgtbt"                      # Too Good To Be True
    MISSING_DISCLAIMER = "missing_disclaimer"
    GRACEFUL_FAILOVER = "graceful_failover"
    ANCHORING = "anchoring"
    SELECTIVE_REASONING = "selective_reasoning"
    METRIC_GAMING = "metric_gaming"
    PREMATURE_CERTAINTY = "premature_certainty"
    PROPOSAL_AS_IMPL = "proposal_as_impl"      # Describes what should be as what is
    MISSION_DRIFT = "mission_drift"            # Unasked scope expansion
    SYCOPHANCY = "sycophancy"                  # Self-congratulatory, pleasing language
    MINIMIZATION = "minimization"              # Explaining away failures


@dataclass
class Pattern:
    """A detectable pattern with effectiveness tracking."""
    pattern_id: str
    name: str
    regex: Optional[str]                 # Regex pattern (if applicable)
    keywords: List[str]                  # Keywords to detect
    category: PatternCategory
    pattern_type: PatternType
    
    # Effectiveness tracking
    total_triggers: int = 0
    true_positives: int = 0
    false_positives: int = 0
    
    # Domain-specific effectiveness
    domain_effectiveness: Dict[str, float] = field(default_factory=dict)
    
    # Adaptive threshold
    base_confidence: float = 0.7
    current_confidence: float = 0.7
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "hardcoded"            # hardcoded, learned, llm_suggested
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def get_effectiveness(self, domain: str = None) -> float:
        """Get pattern effectiveness rate."""
        if domain and domain in self.domain_effectiveness:
            return self.domain_effectiveness[domain]
        if self.total_triggers == 0:
            return self.base_confidence
        return self.true_positives / self.total_triggers
    
    def record_outcome(self, was_correct: bool, domain: str = None):
        """Record whether pattern detection was correct."""
        self.total_triggers += 1
        if was_correct:
            self.true_positives += 1
        else:
            self.false_positives += 1
        
        # Update domain-specific effectiveness
        if domain:
            if domain not in self.domain_effectiveness:
                self.domain_effectiveness[domain] = self.base_confidence
            # Exponential moving average
            alpha = 0.1
            current = self.domain_effectiveness[domain]
            self.domain_effectiveness[domain] = alpha * (1.0 if was_correct else 0.0) + (1 - alpha) * current
        
        # Update current confidence based on overall effectiveness
        self.current_confidence = self.get_effectiveness()
        self.last_updated = datetime.now().isoformat()
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback."""
        pass
    
    def get_statistics(self) -> Dict:
        """Return statistics."""
        return {'outcomes': len(getattr(self, '_outcomes', []))}
    
    def serialize_state(self) -> Dict:
        """Serialize state."""
        return {'outcomes': getattr(self, '_outcomes', [])[-100:]}
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore state."""
        self._outcomes = state.get('outcomes', [])

    def to_dict(self) -> Dict:
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "regex": self.regex,
            "keywords": self.keywords,
            "category": self.category.value,
            "pattern_type": self.pattern_type.value,
            "total_triggers": self.total_triggers,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "domain_effectiveness": self.domain_effectiveness,
            "base_confidence": self.base_confidence,
            "current_confidence": self.current_confidence,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "source": self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Pattern':
        return cls(
            pattern_id=data["pattern_id"],
            name=data["name"],
            regex=data.get("regex"),
            keywords=data.get("keywords", []),
            category=PatternCategory(data["category"]),
            pattern_type=PatternType(data["pattern_type"]),
            total_triggers=data.get("total_triggers", 0),
            true_positives=data.get("true_positives", 0),
            false_positives=data.get("false_positives", 0),
            domain_effectiveness=data.get("domain_effectiveness", {}),
            base_confidence=data.get("base_confidence", 0.7),
            current_confidence=data.get("current_confidence", 0.7),
            created_at=data.get("created_at", datetime.now().isoformat()),
            last_updated=data.get("last_updated", datetime.now().isoformat()),
            source=data.get("source", "hardcoded")
        )


@dataclass
class PatternMatch:
    """Result of a pattern match."""
    pattern: Pattern
    matched_text: str
    confidence: float
    position: Tuple[int, int]            # Start, end position
    domain: str
    context: str                         # Surrounding text


@dataclass
class DomainPatternProfile:
    """Pattern profile learned for a specific domain."""
    domain: str
    patterns_discovered: List[str]       # Pattern IDs
    pattern_weights: Dict[str, float]    # Pattern ID -> weight
    total_audits: int = 0
    issues_caught: int = 0
    false_positives: int = 0
    
    # Learned keywords specific to this domain
    domain_keywords: Dict[str, float] = field(default_factory=dict)  # keyword -> importance
    
    # Circumstance indicators
    high_stakes_indicators: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "domain": self.domain,
            "patterns_discovered": self.patterns_discovered,
            "pattern_weights": self.pattern_weights,
            "total_audits": self.total_audits,
            "issues_caught": self.issues_caught,
            "false_positives": self.false_positives,
            "domain_keywords": self.domain_keywords,
            "high_stakes_indicators": self.high_stakes_indicators
        }


class DomainPatternLearner:
    """
    Learns and manages patterns across domains.
    
    Core Principles:
    1. Start with common patterns (universal)
    2. Learn domain-specific patterns from exposure
    3. Track effectiveness and adapt
    4. Discover new patterns through LLM feedback
    """
    
    # Common patterns (apply to ALL domains)
    COMMON_PATTERNS = [
        Pattern(
            pattern_id="common_100_percent",
            name="100% Certainty Claim",
            regex=r"\b100\s*%",
            keywords=["100%", "guaranteed", "certain"],
            category=PatternCategory.OVERCONFIDENCE,
            pattern_type=PatternType.COMMON,
            base_confidence=0.95
        ),
        Pattern(
            pattern_id="common_fully_implemented",
            name="Fully Implemented Claim",
            regex=r"\bfully\s+(?:implemented|integrated|working|operational|complete|tested|verified|documented|functional)\b",
            keywords=["fully implemented", "fully working", "fully tested"],
            category=PatternCategory.FALSE_COMPLETION,
            pattern_type=PatternType.COMMON,
            base_confidence=0.90
        ),
        Pattern(
            pattern_id="common_production_ready",
            name="Production Ready Claim",
            regex=r"\bproduction[\s-]?ready\b",
            keywords=["production-ready", "production ready"],
            category=PatternCategory.FALSE_COMPLETION,
            pattern_type=PatternType.COMMON,
            base_confidence=0.85
        ),
        Pattern(
            pattern_id="common_all_tests_pass",
            name="All Tests Pass Claim",
            regex=r"\ball\s+tests?\s+pass",
            keywords=["all tests pass", "tests pass", "all passing"],
            category=PatternCategory.FALSE_COMPLETION,
            pattern_type=PatternType.COMMON,
            base_confidence=0.80
        ),
        Pattern(
            pattern_id="common_success_emoji",
            name="Success Emoji Indicator",
            regex=r"[✅🎉🚀]",
            keywords=["✅", "🎉", "🚀"],
            category=PatternCategory.TGTBT,
            pattern_type=PatternType.COMMON,
            base_confidence=0.70
        ),
        Pattern(
            pattern_id="common_overconfident_words",
            name="Overconfident Language",
            regex=r"\b(?:clearly|definitely|certainly|obviously|undoubtedly)\b",
            keywords=["clearly", "definitely", "certainly", "obviously"],
            category=PatternCategory.OVERCONFIDENCE,
            pattern_type=PatternType.COMMON,
            base_confidence=0.65
        ),
        Pattern(
            pattern_id="common_absolute_always",
            name="Absolute Always/Never",
            regex=r"\b(?:always|never)\b",
            keywords=["always", "never"],
            category=PatternCategory.OVERCONFIDENCE,
            pattern_type=PatternType.COMMON,
            base_confidence=0.60
        ),
        Pattern(
            pattern_id="common_todo_placeholder",
            name="TODO/Placeholder Marker",
            regex=r"#\s*(?:TODO|FIXME|XXX)|NotImplemented|pass\s*$",
            keywords=["TODO", "FIXME", "NotImplemented"],
            category=PatternCategory.FALSE_COMPLETION,
            pattern_type=PatternType.COMMON,
            base_confidence=0.95
        ),
        Pattern(
            pattern_id="common_graceful_failover",
            name="Graceful Failover Hiding",
            regex=r"except.*?return.*?(?:success|True|{)",
            keywords=["except", "return success", "graceful"],
            category=PatternCategory.GRACEFUL_FAILOVER,
            pattern_type=PatternType.COMMON,
            base_confidence=0.75
        ),
        Pattern(
            pattern_id="common_zero_errors",
            name="Zero Errors Claim",
            regex=r"\bzero\s+(?:errors?|bugs?|issues?|problems?)\b",
            keywords=["zero errors", "zero bugs", "no errors"],
            category=PatternCategory.TGTBT,
            pattern_type=PatternType.COMMON,
            base_confidence=0.85
        ),
        # NEW: Proposal-as-Implementation patterns (from REAL_LLM_FAILURE_PATTERNS.md)
        Pattern(
            pattern_id="common_proposal_as_impl",
            name="Proposal Presented as Implementation",
            regex=r"\b(?:will|would|should|could)\s+(?:implement|add|create|build)\b",
            keywords=["will implement", "would add", "should create"],
            category=PatternCategory.PROPOSAL_AS_IMPL,
            pattern_type=PatternType.COMMON,
            base_confidence=0.70
        ),
        Pattern(
            pattern_id="common_design_as_done",
            name="Design Described as Done",
            regex=r"\b(?:designed|architected|planned)\s+to\b",
            keywords=["designed to", "architected to", "planned to"],
            category=PatternCategory.PROPOSAL_AS_IMPL,
            pattern_type=PatternType.COMMON,
            base_confidence=0.65
        ),
        # NEW: Mission drift patterns
        Pattern(
            pattern_id="common_scope_expansion",
            name="Unasked Scope Expansion",
            regex=r"\balso\s+(?:added|included|implemented)\b",
            keywords=["also added", "also included", "bonus feature"],
            category=PatternCategory.MISSION_DRIFT,
            pattern_type=PatternType.COMMON,
            base_confidence=0.60
        ),
        # NEW: Metric gaming patterns
        Pattern(
            pattern_id="common_selective_reporting",
            name="Selective Success Reporting",
            regex=r"\b(?:success|passed|complete)[:\s]+\d+%",
            keywords=["success:", "pass rate:", "completion:"],
            category=PatternCategory.METRIC_GAMING,
            pattern_type=PatternType.COMMON,
            base_confidence=0.70
        ),
        # NEW: Self-congratulatory patterns
        Pattern(
            pattern_id="common_self_praise",
            name="Self-Congratulatory Language",
            regex=r"\b(?:excellent|perfect|brilliant|amazing)\s+(?:work|job|result)\b",
            keywords=["excellent work", "perfect", "amazing"],
            category=PatternCategory.SYCOPHANCY,
            pattern_type=PatternType.COMMON,
            base_confidence=0.65
        ),
        # NEW: Explained failure patterns
        Pattern(
            pattern_id="common_explained_failure",
            name="Explained Away Failure",
            regex=r"\b(?:minor|small)\s+(?:issue|problem|error)\b",
            keywords=["minor issue", "small problem", "being addressed"],
            category=PatternCategory.MINIMIZATION,
            pattern_type=PatternType.COMMON,
            base_confidence=0.60
        ),
    ]
    
    # Domain-specific pattern seeds (can be expanded through learning)
    DOMAIN_PATTERN_SEEDS = {
        "medical": [
            Pattern(
                pattern_id="medical_diagnosis_certain",
                name="Certain Diagnosis Without Full Workup",
                regex=r"\bdiagnosis\s+is\s+(?:certain|confirmed|definitive)\b",
                keywords=["diagnosis certain", "diagnosis confirmed"],
                category=PatternCategory.PREMATURE_CERTAINTY,
                pattern_type=PatternType.DOMAIN_SPECIFIC,
                base_confidence=0.80
            ),
            Pattern(
                pattern_id="medical_start_immediately",
                name="Treatment Without Evaluation",
                regex=r"\bstart\s+\w+\s+immediately\b",
                keywords=["start immediately", "begin treatment"],
                category=PatternCategory.PREMATURE_CERTAINTY,
                pattern_type=PatternType.DOMAIN_SPECIFIC,
                base_confidence=0.75
            ),
            Pattern(
                pattern_id="medical_no_further_testing",
                name="Premature Closure",
                regex=r"\bno\s+(?:need\s+for\s+)?(?:further|more)\s+test",
                keywords=["no further testing", "no more tests"],
                category=PatternCategory.ANCHORING,
                pattern_type=PatternType.DOMAIN_SPECIFIC,
                base_confidence=0.85
            ),
        ],
        "legal": [
            Pattern(
                pattern_id="legal_clearly_established",
                name="Clearly Established Without Citation",
                regex=r"\bclearly\s+established\b(?![^.]*\d{4})",
                keywords=["clearly established"],
                category=PatternCategory.OVERCONFIDENCE,
                pattern_type=PatternType.DOMAIN_SPECIFIC,
                base_confidence=0.70
            ),
            Pattern(
                pattern_id="legal_law_settled",
                name="Settled Law Claim",
                regex=r"\b(?:the\s+)?law\s+is\s+(?:settled|clear)\b",
                keywords=["law is settled", "law is clear"],
                category=PatternCategory.OVERCONFIDENCE,
                pattern_type=PatternType.DOMAIN_SPECIFIC,
                base_confidence=0.65
            ),
        ],
        "code": [
            Pattern(
                pattern_id="code_handles_all_edge_cases",
                name="Handles All Edge Cases Claim",
                regex=r"\bhandles?\s+all\s+(?:edge\s+)?cases?\b",
                keywords=["handles all cases", "all edge cases"],
                category=PatternCategory.TGTBT,
                pattern_type=PatternType.DOMAIN_SPECIFIC,
                base_confidence=0.80
            ),
            Pattern(
                pattern_id="code_tested_n_records",
                name="Tested With N Records Without Code",
                regex=r"\btested\s+with\s+\d+[KMB]?\s+records?\b",
                keywords=["tested with records"],
                category=PatternCategory.FALSE_COMPLETION,
                pattern_type=PatternType.DOMAIN_SPECIFIC,
                base_confidence=0.75
            ),
        ],
        "financial": [
            Pattern(
                pattern_id="financial_guaranteed_returns",
                name="Guaranteed Returns Claim",
                regex=r"\bguaranteed\s+(?:returns?|profit|income)\b",
                keywords=["guaranteed returns", "guaranteed profit"],
                category=PatternCategory.TGTBT,
                pattern_type=PatternType.DOMAIN_SPECIFIC,
                base_confidence=0.95
            ),
            Pattern(
                pattern_id="financial_risk_free",
                name="Risk-Free Claim",
                regex=r"\brisk[\s-]?free\b",
                keywords=["risk-free", "risk free"],
                category=PatternCategory.TGTBT,
                pattern_type=PatternType.DOMAIN_SPECIFIC,
                base_confidence=0.90
            ),
        ],
    }
    
    def __init__(self, storage_path: Path = None):
        """Initialize the domain pattern learner."""
        self.storage_path = storage_path or Path("pattern_learning_data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize pattern library
        self.patterns: Dict[str, Pattern] = {}
        self.domain_profiles: Dict[str, DomainPatternProfile] = {}
        
        # Load common patterns
        for pattern in self.COMMON_PATTERNS:
            self.patterns[pattern.pattern_id] = pattern
        
        # Load domain seeds
        for domain, patterns in self.DOMAIN_PATTERN_SEEDS.items():
            for pattern in patterns:
                self.patterns[pattern.pattern_id] = pattern
            self.domain_profiles[domain] = DomainPatternProfile(
                domain=domain,
                patterns_discovered=[p.pattern_id for p in patterns],
                pattern_weights={p.pattern_id: p.base_confidence for p in patterns}
            )
        
        # Load learned patterns from storage
        self._load_learned_patterns()
    
    # Citation indicators that suggest legitimate statistics (not metric gaming)
    CITATION_INDICATORS = [
        r"\bsource\s*:",
        r"\bper\s+",
        r"\bfrom\s+",
        r"\baccording\s+to\b",
        r"\breference\s*:",
        r"\bcitation\s*:",
        r"\|\s*\w+\s*\|",  # Table format |Value|
        r"\.md\b",         # Markdown file reference
        r"\.py\b",         # Python file reference
        r"\([^)]*\.\w+\)", # File reference in parentheses
    ]
    
    def _has_nearby_citation(self, text: str, position: int, window: int = 100) -> bool:
        """
        Check if there's a citation indicator near the matched position.
        
        Phase 15 Enhancement: Reduces false positives on legitimately cited statistics.
        
        Args:
            text: Full text being analyzed
            position: Position of the match
            window: Characters before/after to check for citations
        
        Returns:
            True if citation indicator found nearby
        """
        # Get surrounding context
        start = max(0, position - window)
        end = min(len(text), position + window)
        context = text[start:end].lower()
        
        # Check for citation indicators
        for indicator in self.CITATION_INDICATORS:
            if re.search(indicator, context, re.IGNORECASE):
                return True
        
        return False
    
    def detect_patterns(self, text: str, domain: str = "general") -> List[PatternMatch]:
        """
        Detect patterns in text using both common and domain-specific patterns.
        
        Phase 15 Enhancement: Citation-aware detection for METRIC_GAMING patterns.
        Numbers/percentages near citations are not flagged as gaming.
        
        Args:
            text: Text to analyze
            domain: Domain context for domain-specific patterns
        
        Returns:
            List of pattern matches with confidence scores
        """
        matches = []
        text_lower = text.lower()
        
        # Check all patterns
        for pattern_id, pattern in self.patterns.items():
            # Skip domain-specific patterns for other domains
            if pattern.pattern_type == PatternType.DOMAIN_SPECIFIC:
                pattern_domain = pattern_id.split("_")[0]
                if pattern_domain != domain and domain != "general":
                    continue
            
            # Try regex match
            if pattern.regex:
                try:
                    for match in re.finditer(pattern.regex, text, re.IGNORECASE):
                        confidence = pattern.get_effectiveness(domain)
                        
                        # Phase 15 Enhancement: Citation-aware METRIC_GAMING detection
                        # If this is a metric gaming pattern and there's a nearby citation,
                        # significantly reduce confidence (legitimate cited statistics)
                        if pattern.category == PatternCategory.METRIC_GAMING:
                            if self._has_nearby_citation(text, match.start()):
                                confidence *= 0.3  # Reduce to 30% - likely legitimate
                                # Skip entirely if confidence too low
                                if confidence < 0.25:
                                    continue
                        
                        matches.append(PatternMatch(
                            pattern=pattern,
                            matched_text=match.group(),
                            confidence=confidence,
                            position=(match.start(), match.end()),
                            domain=domain,
                            context=text[max(0, match.start()-50):match.end()+50]
                        ))
                except re.error:
                    pass
            
            # Try keyword match
            for keyword in pattern.keywords:
                if keyword.lower() in text_lower:
                    # Find position
                    pos = text_lower.find(keyword.lower())
                    if pos >= 0:
                        confidence = pattern.get_effectiveness(domain) * 0.9  # Slightly lower for keyword
                        
                        # Phase 15 Enhancement: Citation-aware for keyword matches too
                        if pattern.category == PatternCategory.METRIC_GAMING:
                            if self._has_nearby_citation(text, pos):
                                confidence *= 0.3
                                if confidence < 0.25:
                                    continue
                        
                        matches.append(PatternMatch(
                            pattern=pattern,
                            matched_text=keyword,
                            confidence=confidence,
                            position=(pos, pos + len(keyword)),
                            domain=domain,
                            context=text[max(0, pos-50):pos+len(keyword)+50]
                        ))
        
        # Deduplicate overlapping matches (keep highest confidence)
        matches = self._deduplicate_matches(matches)
        
        return matches
    
    def _deduplicate_matches(self, matches: List[PatternMatch]) -> List[PatternMatch]:
        """Remove overlapping matches, keeping highest confidence."""
        if not matches:
            return []
        
        # Sort by confidence descending
        matches.sort(key=lambda m: m.confidence, reverse=True)
        
        kept = []
        used_positions: Set[Tuple[int, int]] = set()
        
        for match in matches:
            # Check for overlap with existing matches
            overlaps = False
            for start, end in used_positions:
                if not (match.position[1] < start or match.position[0] > end):
                    overlaps = True
                    break
            
            if not overlaps:
                kept.append(match)
                used_positions.add(match.position)
        
        return kept
    
    def record_pattern_outcome(self, pattern_id: str, was_correct: bool, domain: str = None):
        """
        Record whether a pattern detection was correct (true positive) or wrong (false positive).
        
        Args:
            pattern_id: ID of the pattern
            was_correct: True if the pattern correctly identified an issue
            domain: Domain context
        """
        if pattern_id in self.patterns:
            self.patterns[pattern_id].record_outcome(was_correct, domain)
            
            # Update domain profile
            if domain and domain in self.domain_profiles:
                profile = self.domain_profiles[domain]
                profile.total_audits += 1
                if was_correct:
                    profile.issues_caught += 1
                else:
                    profile.false_positives += 1
            
            # Save updated patterns
            self._save_learned_patterns()
    
    def learn_pattern_from_feedback(self, 
                                     text: str, 
                                     domain: str,
                                     issue_description: str,
                                     suggested_pattern: str = None) -> Optional[Pattern]:
        """
        Learn a new pattern from Multi-Track LLM feedback or user feedback.
        
        Args:
            text: The text that contained an issue
            domain: Domain context
            issue_description: Description of what was wrong
            suggested_pattern: Optional regex/keyword suggestion from LLM
        
        Returns:
            Newly created pattern if successful
        """
        # Extract keywords from issue description
        keywords = self._extract_keywords(issue_description)
        
        # Try to identify a pattern
        if suggested_pattern:
            regex = suggested_pattern
        else:
            # Try to create pattern from matched text
            regex = None
        
        # Create new pattern
        pattern_id = f"{domain}_learned_{len(self.patterns)}"
        new_pattern = Pattern(
            pattern_id=pattern_id,
            name=f"Learned: {issue_description[:50]}",
            regex=regex,
            keywords=keywords,
            category=PatternCategory.TGTBT,  # Default, can be refined
            pattern_type=PatternType.DOMAIN_SPECIFIC,
            base_confidence=0.5,  # Start with lower confidence for learned patterns
            source="learned"
        )
        
        # Add to patterns
        self.patterns[pattern_id] = new_pattern
        
        # Add to domain profile
        if domain not in self.domain_profiles:
            self.domain_profiles[domain] = DomainPatternProfile(
                domain=domain,
                patterns_discovered=[],
                pattern_weights={}
            )
        
        self.domain_profiles[domain].patterns_discovered.append(pattern_id)
        self.domain_profiles[domain].pattern_weights[pattern_id] = 0.5
        
        # Save
        self._save_learned_patterns()
        
        return new_pattern
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract potential keywords from text."""
        # Remove common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                     'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                     'through', 'during', 'before', 'after', 'above', 'below',
                     'between', 'under', 'again', 'further', 'then', 'once',
                     'that', 'this', 'these', 'those', 'and', 'but', 'or', 'nor',
                     'so', 'yet', 'both', 'either', 'neither', 'not', 'only'}
        
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Get most common (simple frequency)
        from collections import Counter
        word_counts = Counter(keywords)
        return [word for word, _ in word_counts.most_common(5)]
    
    def get_domain_patterns(self, domain: str) -> List[Pattern]:
        """Get all patterns applicable to a domain."""
        patterns = []
        
        # Common patterns
        for pattern in self.patterns.values():
            if pattern.pattern_type == PatternType.COMMON:
                patterns.append(pattern)
        
        # Domain-specific patterns
        if domain in self.domain_profiles:
            for pattern_id in self.domain_profiles[domain].patterns_discovered:
                if pattern_id in self.patterns:
                    patterns.append(self.patterns[pattern_id])
        
        return patterns
    
    def get_pattern_effectiveness_report(self, domain: str = None) -> Dict:
        """Get effectiveness report for patterns."""
        report = {
            "total_patterns": len(self.patterns),
            "common_patterns": len([p for p in self.patterns.values() if p.pattern_type == PatternType.COMMON]),
            "domain_patterns": len([p for p in self.patterns.values() if p.pattern_type == PatternType.DOMAIN_SPECIFIC]),
            "learned_patterns": len([p for p in self.patterns.values() if p.source == "learned"]),
            "patterns": []
        }
        
        for pattern in self.patterns.values():
            effectiveness = pattern.get_effectiveness(domain)
            report["patterns"].append({
                "id": pattern.pattern_id,
                "name": pattern.name,
                "category": pattern.category.value,
                "effectiveness": effectiveness,
                "total_triggers": pattern.total_triggers,
                "true_positives": pattern.true_positives,
                "false_positives": pattern.false_positives
            })
        
        # Sort by effectiveness
        report["patterns"].sort(key=lambda x: x["effectiveness"], reverse=True)
        
        return report
    
    def _load_learned_patterns(self):
        """Load learned patterns from storage."""
        patterns_file = self.storage_path / "learned_patterns.json"
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r') as f:
                    data = json.load(f)
                    
                # Load patterns
                for pattern_data in data.get("patterns", []):
                    pattern = Pattern.from_dict(pattern_data)
                    self.patterns[pattern.pattern_id] = pattern
                
                # Load domain profiles
                for domain, profile_data in data.get("domain_profiles", {}).items():
                    self.domain_profiles[domain] = DomainPatternProfile(
                        domain=profile_data["domain"],
                        patterns_discovered=profile_data["patterns_discovered"],
                        pattern_weights=profile_data["pattern_weights"],
                        total_audits=profile_data.get("total_audits", 0),
                        issues_caught=profile_data.get("issues_caught", 0),
                        false_positives=profile_data.get("false_positives", 0),
                        domain_keywords=profile_data.get("domain_keywords", {}),
                        high_stakes_indicators=profile_data.get("high_stakes_indicators", [])
                    )
            except Exception as e:
                print(f"[DomainPatternLearner] Error loading patterns: {e}")
    
    def _save_learned_patterns(self):
        """Save learned patterns to storage."""
        patterns_file = self.storage_path / "learned_patterns.json"
        try:
            data = {
                "patterns": [p.to_dict() for p in self.patterns.values() if p.source == "learned"],
                "domain_profiles": {d: p.to_dict() for d, p in self.domain_profiles.items()}
            }
            with open(patterns_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[DomainPatternLearner] Error saving patterns: {e}")

    # =========================================================================
    # Learning Interface
    # =========================================================================
    
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

