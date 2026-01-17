"""
BASE Neuro-Symbolic Reasoning Module
Formal logic verification and constraint satisfaction

This module enables BASE to:
1. Extract logical statements from natural language
2. Verify logical consistency and validity
3. Check constraint satisfaction
4. Detect contradictions and fallacies

Uses simplified symbolic logic (no external Z3 dependency required).

Patent Claims: R&D Invention 2
Proprietary IP - 100% owned by Invitas Inc.
"""

import re
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


class LogicalOperator(str, Enum):
    """Logical operators."""
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "implies"
    IFF = "iff"  # if and only if


class StatementType(str, Enum):
    """Types of logical statements."""
    PROPOSITION = "proposition"       # Simple statement
    CONDITIONAL = "conditional"       # If-then
    UNIVERSAL = "universal"           # For all X
    EXISTENTIAL = "existential"       # There exists X
    NEGATION = "negation"             # Not X
    CONJUNCTION = "conjunction"       # X and Y
    DISJUNCTION = "disjunction"       # X or Y


class VerificationResult(str, Enum):
    """Results of logical verification."""
    VALID = "valid"             # Logically sound
    INVALID = "invalid"         # Contains errors
    INCOMPLETE = "incomplete"   # Cannot verify
    CONTRADICTION = "contradiction"  # Self-contradictory


class FallacyType(str, Enum):
    """Types of logical fallacies."""
    AFFIRMING_CONSEQUENT = "affirming_consequent"
    DENYING_ANTECEDENT = "denying_antecedent"
    FALSE_DICHOTOMY = "false_dichotomy"
    HASTY_GENERALIZATION = "hasty_generalization"
    CIRCULAR_REASONING = "circular_reasoning"
    AD_HOMINEM = "ad_hominem"
    STRAW_MAN = "straw_man"
    BANDWAGON = "bandwagon"  # Appeal to popularity
    APPEAL_TO_AUTHORITY = "appeal_to_authority"
    # BASE-GUIDED: Added missing fallacy types
    APPEAL_TO_NATURE = "appeal_to_nature"
    SLIPPERY_SLOPE = "slippery_slope"


@dataclass
class LogicalStatement:
    """Represents an extracted logical statement."""
    statement_id: str
    statement_type: StatementType
    content: str                    # Natural language
    symbolic_form: str              # Symbolic representation
    variables: List[str]            # Logical variables
    confidence: float
    source_text: str


@dataclass
class Constraint:
    """Represents a constraint to be checked."""
    constraint_id: str
    description: str
    condition: str                  # The constraint condition
    variables: List[str]
    is_satisfied: Optional[bool] = None
    violation_reason: Optional[str] = None


@dataclass
class FallacyDetection:
    """Detection of a logical fallacy."""
    fallacy_type: FallacyType
    description: str
    evidence: str
    confidence: float
    suggestion: str                 # How to fix


@dataclass
class ConsistencyCheck:
    """Result of consistency checking."""
    is_consistent: bool
    contradictions: List[Tuple[str, str]]  # Pairs of contradicting statements
    explanation: str


@dataclass
class LogicalVerification:
    """Complete logical verification result."""
    # Extraction results
    statements: List[LogicalStatement]
    constraints: List[Constraint]
    
    # Verification results
    verification_result: VerificationResult
    validity_score: float           # 0-1 score
    
    # Consistency
    consistency: ConsistencyCheck
    
    # Fallacies
    fallacies_detected: List[FallacyDetection]
    fallacy_risk_score: float
    
    # Completeness
    inference_coverage: float       # How much can be verified
    unverifiable_claims: List[str]
    
    # Meta
    processing_time_ms: float
    timestamp: float
    warnings: List[str] = field(default_factory=list)


class NeuroSymbolicModule:
    """
    Neuro-Symbolic reasoning module for BASE.
    
    Implements:
    - Logical statement extraction from natural language
    - Formal verification (validity checking)
    - Constraint satisfaction checking
    - Contradiction detection
    - Fallacy identification
    
    Uses pattern-based symbolic logic (no external solver required).
    
    Integration:
    - Triggered by Query Router for logical/verification queries
    - Reports to Consistency Checker
    - Particularly important for legal, compliance, technical domains
    """
    
    # Logical connective patterns
    CONNECTIVE_PATTERNS = {
        LogicalOperator.AND: [r'\b(and|also|moreover|furthermore|additionally)\b'],
        LogicalOperator.OR: [r'\b(or|either|alternatively)\b'],
        LogicalOperator.NOT: [r'\b(not|never|no|none|neither)\b'],
        LogicalOperator.IMPLIES: [r'\b(if|then|implies|therefore|thus|hence|consequently)\b'],
        LogicalOperator.IFF: [r'\b(if and only if|iff|exactly when|equivalent to)\b'],
    }
    
    # Conditional patterns
    CONDITIONAL_PATTERNS = [
        r'if\s+(.+?)\s*,?\s*then\s+(.+)',
        r'when\s+(.+?)\s*,?\s*(.+)',
        r'(.+)\s+implies\s+(.+)',
        r'(.+)\s+therefore\s+(.+)',
        r'(.+)\s+so\s+(.+)',
    ]
    
    # Quantifier patterns
    QUANTIFIER_PATTERNS = {
        StatementType.UNIVERSAL: [
            r'\b(all|every|any|each|always|everywhere|everything)\b',
            r'\bfor all\b',
        ],
        StatementType.EXISTENTIAL: [
            r'\b(some|exists|there is|there are|at least one|sometimes)\b',
            r'\bthere exists\b',
        ],
    }
    
    # Fallacy patterns - ENHANCED with more patterns
    FALLACY_PATTERNS = {
        FallacyType.AFFIRMING_CONSEQUENT: [
            r'if.*then.*\. .* therefore',  # If A then B. B. Therefore A.
            r'because.*happened.*must mean',
            r'the .* is .* so .* must be',
        ],
        FallacyType.FALSE_DICHOTOMY: [
            r'\b(either.*or)\b',
            r'\b(only two|only choice|no other option|must be one)\b',
            r'\b(us or them|with us or against)\b',
            r'\b(you must|you have to).*(or)\b',
            r'\b(black and white|one or the other)\b',
        ],
        FallacyType.HASTY_GENERALIZATION: [
            r'\b(all.*because|everyone.*since|never.*based on)\b',
            r'\b(always happens|never works|everyone knows)\b',
        ],
        FallacyType.CIRCULAR_REASONING: [
            r'because.*is.*because',
            r'true because.*said.*true',
            r'proves.*because.*proves',
            # BASE SELF-IMPROVEMENT: Added based on failure analysis
            r'best.*because.*superior',
            r'better.*because.*better',
            r'superior.*because.*best',
        ],
        FallacyType.AD_HOMINEM: [
            r'\b(wrong because .* is)\b',
            r'\b(can\'t trust .* because .* is)\b',
            r'\b(\w+ is wrong|ignore \w+ because)\b',
            r'\b(what does .* know|you\'re just|typical \w+)\b',
            r'\b(not an expert|not qualified|what would .* know)\b',
        ],
        FallacyType.APPEAL_TO_AUTHORITY: [
            r'\b(expert says|authority claims|professor believes)\b',
            r'\b(doctors agree|scientists say|research shows)\b',
            r'\b(according to experts|trusted sources say)\b',
            # PHASE 1 FIX: More authority patterns (NS-A4)
            r'\b(scientists?\s+agree|experts?\s+agree|doctors?\s+agree)\b',
            r'\b(scientists?\s+recommend|doctors?\s+recommend|experts?\s+recommend)\b',
            r'\b(leading\s+(?:scientists?|experts?|doctors?))\b',
            r'\b(studies?\s+show|research\s+(?:shows|proves|confirms))\b',
            r'\b((?:top|renowned|leading)\s+(?:expert|scientist|professor))\b',
            r'\b(according\s+to\s+(?:science|research|studies))\b',
        ],
        FallacyType.STRAW_MAN: [
            r'\b(so you\'re saying|what you really mean|obviously you think)\b',
            r'\b(you want to|you believe that all)\b',
            # BASE-GUIDED FIX: Straw man patterns
            r'\bso they must\s+(?:want|believe|think|mean)\b',
            r'\bthey (?:want|believe).*(?:all|every|completely|totally)\b',
            r'\b(?:want|believe).*(?:ban all|eliminate all|destroy all)\b',
            r'\bleave.*(?:defenseless|helpless|vulnerable)\b',
        ],
        # BASE-GUIDED: New fallacy patterns
        FallacyType.APPEAL_TO_NATURE: [
            r'\bnatural\b.*\b(?:therefore|so|must be|is)\s+(?:safe|healthy|good|better)\b',
            r'\b(?:because|since)\s+(?:it\'?s?|this is)\s+natural\b',
            r'\bnatural\s+(?:means?|implies?|=)\s+(?:safe|good|healthy)\b',
            r'\borganic\b.*\b(?:therefore|so)\s+(?:better|healthier)\b',
        ],
        FallacyType.SLIPPERY_SLOPE: [
            r'\bif we (?:allow|permit|let).*(?:everyone|all|everything)\s+will\b',
            r'\b(?:soon|eventually|next)\s+(?:everyone|everything)\s+will\b',
            r'\b(?:entire|whole)\s+(?:system|thing|world)\s+will\s+(?:collapse|fall|fail)\b',
            r'\bwhere does it (?:end|stop)\b',
        ],
        # BASE SELF-IMPROVEMENT: Added bandwagon/appeal to popularity
        FallacyType.BANDWAGON: [
            r'\beveryone\s+(?:believes?|thinks?|knows?|says?)\b.*\b(?:so|must|therefore)\b',
            r'\b(?:so|must|therefore)\b.*\beveryone\s+(?:believes?|thinks?|knows?|says?)\b',
            r'\bmillions?\s+(?:of people|can\'t be wrong)\b',
            r'\bmost\s+people\s+(?:agree|believe|think)\b.*\b(?:so|therefore|must)\b',
            r'\bpopular\b.*\b(?:must be|therefore|so)\s+(?:right|true|correct)\b',
            r'\bif\s+(?:everyone|most people|many people)\s+(?:do|does|did)\s+it\b',
        ],
    }
    
    # Contradiction patterns - ENHANCED
    CONTRADICTION_PATTERNS = [
        (r'\b(is)\b', r'\b(is not|isn\'t|are not|aren\'t)\b'),
        (r'\b(all)\b', r'\b(none|no|not all|some .* not)\b'),
        (r'\b(always)\b', r'\b(never|not always|sometimes not)\b'),
        (r'\b(true)\b', r'\b(false|not true|untrue)\b'),
        (r'\b(can)\b', r'\b(cannot|can\'t|can not)\b'),
        (r'\b(must)\b', r'\b(must not|mustn\'t|need not)\b'),
        (r'\b(will)\b', r'\b(will not|won\'t)\b'),
        (r'\b(do)\b', r'\b(do not|don\'t)\b'),
        (r'\b(does)\b', r'\b(does not|doesn\'t)\b'),
        (r'\b(have)\b', r'\b(have not|haven\'t|don\'t have)\b'),
        (r'\b(possible)\b', r'\b(impossible|not possible)\b'),
        (r'\b(able)\b', r'\b(unable|not able)\b'),
    ]
    
    # ADDED: Syllogism patterns for universal statements
    UNIVERSAL_CLAIM_PATTERNS = [
        (r'all\s+(\w+)\s+(?:can|are|have|do)\s+(\w+)', 'universal_positive'),  # All X can Y
        (r'(\w+)\s+(?:cannot|can\'t|are not|don\'t)\s+(\w+)', 'particular_negative'),  # X cannot Y
        (r'(\w+)\s+(?:is|are)\s+(?:a\s+)?(\w+)', 'particular_positive'),  # X is Y
    ]
    
    # BASE Enhancement: Semantic contradiction patterns
    SEMANTIC_CONTRADICTION_PATTERNS = [
        (r'\balways\b.*\bnever\b|\bnever\b.*\balways\b', 'Always/never contradiction'),
        (r'\beveryone\b.*\bno\s+one\b|\bno\s+one\b.*\beveryone\b', 'Universal scope contradiction'),
        (r'\bimpossible\b.*\bguaranteed\b|\bguaranteed\b.*\bimpossible\b', 'Certainty contradiction'),
        (r'\bsimultaneously\b.*\b(?:and|but)\s+(?:not|never)\b', 'Simultaneous contradiction'),
        (r'\bfinished\b.*\bhasn\'t\s+started\b', 'Completion paradox'),
        (r'\b100%\b.*\b(?:and|but|while)\s+(?:0%|zero|none|nothing)\b', 'Percentage contradiction'),
        (r'\b100%\s+\w+\b.*\b(?:but|and)\s+0%\b', 'Success rate contradiction'),
        (r'\b\d+%\s+success\b.*\b0%\s+\w+\b', 'Effectiveness contradiction'),
        (r'\ball\b.*\b(?:and|but|while)\s+none\b|\bnone\b.*\b(?:and|but|while)\s+all\b', 'All/none contradiction'),
        (r'\bcompletely\b.*\bpartially\b|\bpartially\b.*\bcompletely\b', 'Degree contradiction'),
    ]
    
    def __init__(self,
                 min_statement_confidence: float = 0.50,
                 fallacy_threshold: float = 0.40):
        """
        Initialize Neuro-Symbolic module.
        
        Args:
            min_statement_confidence: Minimum confidence for statement extraction
            fallacy_threshold: Threshold for flagging fallacies
        """
        self.min_statement_confidence = min_statement_confidence
        self.fallacy_threshold = fallacy_threshold
        
        # Learning state
        self._outcomes: List[Dict[str, Any]] = []
        self._feedback: List[Dict[str, Any]] = []
        self._domain_adjustments: Dict[str, float] = {}
        self._total_verifications: int = 0
        self._verification_accuracy: List[bool] = []
    
    # ===== Learning Interface Methods =====
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict[str, Any]) -> None:
        """Record verification outcome for learning."""
        self._outcomes.append(outcome)
        if 'verification_correct' in outcome:
            self._verification_accuracy.append(outcome['verification_correct'])
    
    def record_feedback(self, feedback: Dict[str, Any]) -> None:
        """Record feedback on verification results."""
        self._feedback.append(feedback)
        domain = feedback.get('domain', 'general')
        if feedback.get('missed_fallacy', False):
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) - 0.05
        if feedback.get('false_positive_fallacy', False):
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) + 0.05
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt detection thresholds based on performance."""
        pass
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get domain-specific adjustment."""
        return self._domain_adjustments.get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        verif_acc = sum(self._verification_accuracy) / max(len(self._verification_accuracy), 1)
        
        return {
            'total_verifications': self._total_verifications,
            'verification_accuracy': verif_acc,
            'domain_adjustments': dict(self._domain_adjustments),
            'outcomes_recorded': len(self._outcomes),
            'feedback_recorded': len(self._feedback),
            'min_statement_confidence': self.min_statement_confidence,
            'fallacy_threshold': self.fallacy_threshold
        }
    
    def verify_logic(self, query: str = None, response: str = None) -> LogicalVerification:
        """Alias for verify() - standard interface compatibility."""
        return self.verify(query=query or "", response=response or "")
    
    def verify(self,
               query: str,
               response: str,
               constraints: List[str] = None,
               context: Dict[str, Any] = None) -> LogicalVerification:
        """
        Perform complete logical verification.
        
        Args:
            query: User query
            response: AI-generated response
            constraints: Optional list of constraints to check
            context: Additional context
        
        Returns:
            LogicalVerification with complete analysis
        """
        start_time = time.time()
        self._total_verifications += 1
        context = context or {}
        constraints = constraints or []
        warnings = []
        
        combined_text = f"{query} {response}"
        
        # 1. Extract logical statements
        statements = self._extract_statements(response)
        
        # 2. Parse constraints
        constraint_objects = self._parse_constraints(constraints, response)
        
        # 3. Check consistency (pass raw_text for better syllogism detection)
        consistency = self._check_consistency(statements, raw_text=response)
        
        # 4. Detect fallacies
        fallacies = self._detect_fallacies(response)
        fallacy_risk = len(fallacies) / 5  # Max 5 fallacy types
        
        if fallacies:
            warnings.append(f"Detected {len(fallacies)} logical fallacies")
        
        # 5. Verify logical validity
        validity_score, verification_result = self._verify_logic(
            statements, consistency, fallacies
        )
        
        # 6. Compute inference coverage
        coverage, unverifiable = self._compute_coverage(statements, response)
        
        if consistency.contradictions:
            verification_result = VerificationResult.CONTRADICTION
            warnings.append(f"Found {len(consistency.contradictions)} contradictions")
        
        processing_time = (time.time() - start_time) * 1000
        
        return LogicalVerification(
            statements=statements,
            constraints=constraint_objects,
            verification_result=verification_result,
            validity_score=validity_score,
            consistency=consistency,
            fallacies_detected=fallacies,
            fallacy_risk_score=min(fallacy_risk, 1.0),
            inference_coverage=coverage,
            unverifiable_claims=unverifiable,
            processing_time_ms=processing_time,
            timestamp=time.time(),
            warnings=warnings
        )
    
    def _extract_statements(self, text: str) -> List[LogicalStatement]:
        """Extract logical statements from text."""
        statements = []
        statement_id = 0
        
        # Split into sentences
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            # Determine statement type and extract
            statement = self._analyze_sentence(sentence, statement_id)
            if statement and statement.confidence >= self.min_statement_confidence:
                statements.append(statement)
                statement_id += 1
        
        return statements[:15]  # Limit to 15 statements
    
    def _analyze_sentence(self, sentence: str, sid: int) -> Optional[LogicalStatement]:
        """Analyze a sentence for logical content."""
        sentence_lower = sentence.lower()
        
        # Check for conditionals first (most important)
        for pattern in self.CONDITIONAL_PATTERNS:
            match = re.search(pattern, sentence_lower)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    antecedent = groups[0].strip()
                    consequent = groups[1].strip()
                    
                    return LogicalStatement(
                        statement_id=f"S{sid}",
                        statement_type=StatementType.CONDITIONAL,
                        content=sentence,
                        symbolic_form=f"({antecedent}) → ({consequent})",
                        variables=self._extract_variables(sentence),
                        confidence=0.8,
                        source_text=sentence
                    )
        
        # Check for quantifiers
        for stmt_type, patterns in self.QUANTIFIER_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, sentence_lower):
                    return LogicalStatement(
                        statement_id=f"S{sid}",
                        statement_type=stmt_type,
                        content=sentence,
                        symbolic_form=self._to_symbolic(sentence, stmt_type),
                        variables=self._extract_variables(sentence),
                        confidence=0.7,
                        source_text=sentence
                    )
        
        # Check for logical connectives
        for operator, patterns in self.CONNECTIVE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, sentence_lower):
                    stmt_type = self._operator_to_statement_type(operator)
                    return LogicalStatement(
                        statement_id=f"S{sid}",
                        statement_type=stmt_type,
                        content=sentence,
                        symbolic_form=self._to_symbolic(sentence, stmt_type),
                        variables=self._extract_variables(sentence),
                        confidence=0.6,
                        source_text=sentence
                    )
        
        # Default to proposition
        if len(sentence.split()) >= 5:
            return LogicalStatement(
                statement_id=f"S{sid}",
                statement_type=StatementType.PROPOSITION,
                content=sentence,
                symbolic_form=f"P{sid}",
                variables=self._extract_variables(sentence),
                confidence=0.5,
                source_text=sentence
            )
        
        return None
    
    def _operator_to_statement_type(self, operator: LogicalOperator) -> StatementType:
        """Convert operator to statement type."""
        mapping = {
            LogicalOperator.AND: StatementType.CONJUNCTION,
            LogicalOperator.OR: StatementType.DISJUNCTION,
            LogicalOperator.NOT: StatementType.NEGATION,
            LogicalOperator.IMPLIES: StatementType.CONDITIONAL,
            LogicalOperator.IFF: StatementType.CONDITIONAL,
        }
        return mapping.get(operator, StatementType.PROPOSITION)
    
    def _to_symbolic(self, sentence: str, stmt_type: StatementType) -> str:
        """Convert sentence to symbolic form."""
        sentence_clean = re.sub(r'\b(the|a|an|is|are|was|were)\b', '', sentence.lower())
        words = sentence_clean.split()[:5]  # First 5 significant words
        
        if stmt_type == StatementType.UNIVERSAL:
            return f"∀x: {' '.join(words)}(x)"
        elif stmt_type == StatementType.EXISTENTIAL:
            return f"∃x: {' '.join(words)}(x)"
        elif stmt_type == StatementType.NEGATION:
            return f"¬({' '.join(words)})"
        elif stmt_type == StatementType.CONJUNCTION:
            mid = len(words) // 2
            return f"({' '.join(words[:mid])}) ∧ ({' '.join(words[mid:])})"
        elif stmt_type == StatementType.DISJUNCTION:
            mid = len(words) // 2
            return f"({' '.join(words[:mid])}) ∨ ({' '.join(words[mid:])})"
        else:
            return f"P({' '.join(words)})"
    
    def _extract_variables(self, sentence: str) -> List[str]:
        """Extract logical variables (key entities) from sentence."""
        # Extract capitalized terms as potential variables
        variables = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence)
        return list(set(variables))[:5]
    
    def _parse_constraints(self,
                           constraints: List[str],
                           response: str) -> List[Constraint]:
        """Parse and check constraints."""
        constraint_objects = []
        
        for i, constraint_text in enumerate(constraints):
            constraint = Constraint(
                constraint_id=f"C{i}",
                description=constraint_text,
                condition=constraint_text,
                variables=self._extract_variables(constraint_text),
            )
            
            # Simple satisfaction check
            constraint.is_satisfied = self._check_constraint_satisfied(
                constraint_text, response
            )
            
            if not constraint.is_satisfied:
                constraint.violation_reason = "Constraint not satisfied in response"
            
            constraint_objects.append(constraint)
        
        return constraint_objects
    
    def _check_constraint_satisfied(self, constraint: str, response: str) -> bool:
        """Check if a constraint is satisfied by response."""
        # Simple keyword-based check
        constraint_keywords = set(
            w.lower() for w in re.findall(r'\b\w+\b', constraint)
            if len(w) > 3
        )
        response_lower = response.lower()
        
        # Check if constraint keywords appear in response
        matches = sum(1 for kw in constraint_keywords if kw in response_lower)
        return matches >= len(constraint_keywords) * 0.5
    
    def _check_consistency(self, statements: List[LogicalStatement], raw_text: str = "") -> ConsistencyCheck:
        """Check logical consistency of statements - ENHANCED with syllogism detection."""
        contradictions = []
        
        # Combine all statement text for analysis
        # Use raw_text if provided (more complete), otherwise combine statements
        if raw_text:
            all_text_lower = raw_text.lower()
        else:
            all_text = " ".join([s.content for s in statements])
            all_text_lower = all_text.lower()
        
        # Check pairs of statements for direct contradictions
        for i, stmt1 in enumerate(statements):
            for stmt2 in statements[i+1:]:
                if self._statements_contradict(stmt1, stmt2):
                    contradictions.append((stmt1.statement_id, stmt2.statement_id))
        
        # ENHANCED: Check for syllogistic contradictions
        syllogistic_contradictions = self._check_syllogistic_contradiction(all_text_lower)
        contradictions.extend(syllogistic_contradictions)
        
        # BASE Enhancement: Check for semantic contradictions
        semantic_contradictions = self._check_semantic_contradictions(all_text_lower)
        contradictions.extend(semantic_contradictions)
        
        is_consistent = len(contradictions) == 0
        
        if contradictions:
            explanation = f"Found {len(contradictions)} contradicting statement pairs"
        else:
            explanation = "No contradictions detected"
        
        return ConsistencyCheck(
            is_consistent=is_consistent,
            contradictions=contradictions,
            explanation=explanation
        )
    
    def _check_syllogistic_contradiction(self, text: str) -> List[Tuple[str, str]]:
        """
        Check for syllogistic contradictions like:
        'All birds can fly' + 'Penguins are birds' + 'Penguins cannot fly'
        
        This is the classic Barbara syllogism contradiction.
        Also checks for All/Some contradictions and temporal paradoxes.
        """
        contradictions = []
        
        # Check All/Some quantifier contradictions first
        all_some_contradictions = self._check_all_some_contradiction(text)
        contradictions.extend(all_some_contradictions)
        
        # Check temporal contradictions
        temporal_contradictions = self._check_temporal_contradiction(text)
        contradictions.extend(temporal_contradictions)
        
        # Pattern 1: All X are/can Y + Z is/are X + Z cannot/are not Y
        # Extract universal claims (All X can Y / All X are Y)
        universal_claims = []
        universal_patterns = [
            r'all\s+(\w+)\s+(?:can|are|have|do)\s+(\w+)',  # All birds can fly
            r'all\s+(\w+)\s+are\s+(\w+)',  # All birds are flyers
            r'every\s+(\w+)\s+(?:can|is|has|does)\s+(\w+)',  # Every bird can fly
        ]
        for pattern in universal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                universal_claims.append((match[0].lower(), match[1].lower()))
        
        # Extract membership claims (X is/are Y)
        membership_claims = []
        membership_patterns = [
            r'(\w+)\s+(?:is|are)\s+(?:a\s+)?(\w+)',  # Penguins are birds
        ]
        for pattern in membership_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                membership_claims.append((match[0].lower(), match[1].lower()))
        
        # Extract negative claims (X cannot/don't Y)
        negative_claims = []
        negative_patterns = [
            r'(\w+)\s+(?:cannot|can\'t|can not|don\'t|do not|are not|aren\'t)\s+(\w+)',
        ]
        for pattern in negative_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                negative_claims.append((match[0].lower(), match[1].lower()))
        
        # Check for syllogistic contradictions
        # If: All X can Y, Z is X, Z cannot Y → Contradiction!
        for (category, property_name) in universal_claims:
            for (instance, member_of) in membership_claims:
                # Check if instance is member of category
                if member_of == category or category.startswith(member_of) or member_of.startswith(category):
                    # Check if instance has negative claim about property
                    for (neg_instance, neg_property) in negative_claims:
                        if neg_instance == instance or instance.startswith(neg_instance) or neg_instance.startswith(instance):
                            # Check if negative property matches the universal property
                            if neg_property == property_name or property_name.startswith(neg_property) or neg_property.startswith(property_name):
                                contradictions.append(
                                    (f"Universal: All {category} {property_name}",
                                     f"Particular: {instance} cannot {neg_property}")
                                )
        
        # Also check direct contradictions like "X can Y" and "X cannot Y"
        positive_patterns = [
            r'(\w+)\s+(?:can|do|does|is|are|has|have)\s+(\w+)',
        ]
        positive_claims = []
        for pattern in positive_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                positive_claims.append((match[0].lower(), match[1].lower()))
        
        for (pos_subj, pos_pred) in positive_claims:
            for (neg_subj, neg_pred) in negative_claims:
                if pos_subj == neg_subj and pos_pred == neg_pred:
                    contradictions.append(
                        (f"Positive: {pos_subj} {pos_pred}",
                         f"Negative: {neg_subj} cannot {neg_pred}")
                    )
        
        return contradictions
    
    def _check_all_some_contradiction(self, text: str) -> List[Tuple[str, str]]:
        """
        Check for All/Some quantifier contradictions like:
        'All students passed' + 'Some students failed'
        """
        contradictions = []
        
        # Find "All X verb Y" patterns
        all_patterns = [
            r'all\s+(\w+)\s+(\w+)',  # All students passed
            r'every\s+(\w+)\s+(\w+)',  # Every student passed
            r'no\s+(\w+)\s+(\w+)',  # No students failed
        ]
        
        # Find "Some X verb Y" patterns
        some_patterns = [
            r'some\s+(\w+)\s+(\w+)',  # Some students failed
            r'a few\s+(\w+)\s+(\w+)',
            r'several\s+(\w+)\s+(\w+)',
        ]
        
        all_claims = []
        some_claims = []
        
        for pattern in all_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()
                if len(groups) >= 2:
                    all_claims.append((groups[0].lower(), groups[1].lower()))
        
        for pattern in some_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()
                if len(groups) >= 2:
                    some_claims.append((groups[0].lower(), groups[1].lower()))
        
        # Check for contradictions: "All X passed" vs "Some X failed"
        opposite_verbs = {
            'passed': 'failed',
            'succeeded': 'failed',
            'won': 'lost',
            'lived': 'died',
            'arrived': 'left',
        }
        
        for (all_subj, all_verb) in all_claims:
            for (some_subj, some_verb) in some_claims:
                # Same subject
                if all_subj == some_subj or all_subj.rstrip('s') == some_subj.rstrip('s'):
                    # Opposite verbs
                    if opposite_verbs.get(all_verb) == some_verb or opposite_verbs.get(some_verb) == all_verb:
                        contradictions.append(
                            (f"Universal: All {all_subj} {all_verb}",
                             f"Existential: Some {some_subj} {some_verb}")
                        )
        
        return contradictions
    
    def _check_temporal_contradiction(self, text: str) -> List[Tuple[str, str]]:
        """
        Check for temporal paradoxes like:
        'completed before it started' or 'finished yesterday but began today'
        """
        contradictions = []
        
        # Patterns for temporal paradoxes
        paradox_patterns = [
            # X before Y where X logically requires Y first
            r'(\w+)\s+before\s+(?:it\s+)?(\w+)',
            r'(\w+)\s+yesterday.*?(\w+)\s+today',
            r'(\w+)\s+after\s+(?:it\s+)?(\w+)',
        ]
        
        # Temporal ordering requirements
        temporal_order = {
            ('completed', 'started'): 'paradox',
            ('finished', 'began'): 'paradox',
            ('ended', 'started'): 'paradox',
            ('arrived', 'left'): 'paradox',
            ('born', 'died'): 'requires_first',
        }
        
        for pattern in paradox_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()
                if len(groups) >= 2:
                    verb1 = groups[0].lower()
                    verb2 = groups[1].lower()
                    
                    # Check for known paradoxes
                    if (verb1, verb2) in temporal_order:
                        if temporal_order[(verb1, verb2)] == 'paradox':
                            contradictions.append(
                                (f"Temporal: {verb1} before {verb2}",
                                 "Paradox: Logical sequence violated")
                            )
        
        # Also check for explicit temporal contradictions
        if 'before' in text.lower() and 'after' in text.lower():
            # Simple heuristic for "X before Y... X after Y"
            pass  # Could enhance with more sophisticated parsing
        
        return contradictions
    
    def _check_semantic_contradictions(self, text: str) -> List[Tuple[str, str]]:
        """
        BASE Enhancement: Check for semantic contradictions using patterns.
        These are higher-level logical contradictions that pattern matching can catch.
        """
        contradictions = []
        
        for pattern, description in self.SEMANTIC_CONTRADICTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                # Extract the matching text for evidence
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    contradictions.append(
                        (f"Semantic: {description}",
                         f"Evidence: '{match.group(0)[:50]}...'")
                    )
        
        return contradictions
    
    def _statements_contradict(self,
                                stmt1: LogicalStatement,
                                stmt2: LogicalStatement) -> bool:
        """Check if two statements contradict each other."""
        text1 = stmt1.content.lower()
        text2 = stmt2.content.lower()
        
        # Check for contradiction patterns
        for pos_pattern, neg_pattern in self.CONTRADICTION_PATTERNS:
            # Check if one has positive and other has negative
            has_pos_1 = bool(re.search(pos_pattern, text1))
            has_neg_1 = bool(re.search(neg_pattern, text1))
            has_pos_2 = bool(re.search(pos_pattern, text2))
            has_neg_2 = bool(re.search(neg_pattern, text2))
            
            # Check if same subject with opposite predicate
            if (has_pos_1 and has_neg_2) or (has_neg_1 and has_pos_2):
                # Check for common subject
                common_vars = set(stmt1.variables) & set(stmt2.variables)
                if common_vars:
                    return True
        
        return False
    
    def _detect_fallacies(self, text: str) -> List[FallacyDetection]:
        """Detect logical fallacies in text."""
        fallacies = []
        text_lower = text.lower()
        
        fallacy_info = {
            FallacyType.AFFIRMING_CONSEQUENT: (
                "Affirming the Consequent",
                "Concluding A from 'if A then B' and B"
            ),
            FallacyType.FALSE_DICHOTOMY: (
                "False Dichotomy",
                "Presenting only two options when more exist"
            ),
            FallacyType.HASTY_GENERALIZATION: (
                "Hasty Generalization",
                "Drawing broad conclusions from limited examples"
            ),
            FallacyType.CIRCULAR_REASONING: (
                "Circular Reasoning",
                "Using conclusion as a premise"
            ),
            FallacyType.AD_HOMINEM: (
                "Ad Hominem",
                "Attacking the person instead of the argument"
            ),
            FallacyType.APPEAL_TO_AUTHORITY: (
                "Appeal to Authority",
                "Using authority as sole evidence"
            ),
        }
        
        for fallacy_type, patterns in self.FALLACY_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    info = fallacy_info.get(fallacy_type, ("Unknown", "Unknown fallacy"))
                    
                    fallacy = FallacyDetection(
                        fallacy_type=fallacy_type,
                        description=info[0],
                        evidence=match.group()[:100],
                        confidence=0.6,
                        suggestion=f"Revise to avoid {info[1].lower()}"
                    )
                    fallacies.append(fallacy)
                    break  # One detection per fallacy type
        
        return fallacies
    
    def _verify_logic(self,
                      statements: List[LogicalStatement],
                      consistency: ConsistencyCheck,
                      fallacies: List[FallacyDetection]) -> Tuple[float, VerificationResult]:
        """Compute overall logical validity."""
        if not statements:
            return 0.5, VerificationResult.INCOMPLETE
        
        # Start with base score
        score = 0.7
        
        # Penalize for inconsistency
        if not consistency.is_consistent:
            score -= 0.3
        
        # Penalize for fallacies
        score -= len(fallacies) * 0.1
        
        # Boost for high-confidence statements
        avg_confidence = sum(s.confidence for s in statements) / len(statements)
        score = score * 0.7 + avg_confidence * 0.3
        
        score = max(0.0, min(1.0, score))
        
        # Determine result
        if not consistency.is_consistent:
            result = VerificationResult.CONTRADICTION
        elif score >= 0.7:
            result = VerificationResult.VALID
        elif score >= 0.4:
            result = VerificationResult.INCOMPLETE
        else:
            result = VerificationResult.INVALID
        
        return score, result
    
    def _compute_coverage(self,
                           statements: List[LogicalStatement],
                           text: str) -> Tuple[float, List[str]]:
        """Compute how much of the text can be logically verified."""
        # Count sentences
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 10]
        
        if not sentences:
            return 0.0, []
        
        # Count covered sentences
        covered = set()
        for stmt in statements:
            for i, sent in enumerate(sentences):
                if stmt.source_text.lower() in sent.lower() or sent.lower() in stmt.source_text.lower():
                    covered.add(i)
        
        coverage = len(covered) / len(sentences)
        
        # Find unverifiable
        unverifiable = [
            sentences[i][:80] + "..." 
            for i in range(len(sentences)) 
            if i not in covered
        ][:5]
        
        return coverage, unverifiable

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


# Self-test
if __name__ == "__main__":
    module = NeuroSymbolicModule()
    
    print("=" * 60)
    print("NEURO-SYMBOLIC REASONING MODULE TEST")
    print("=" * 60)
    
    # Test 1: Valid logic
    query = "Is this argument valid?"
    response = """
    If all humans are mortal, and Socrates is human, then Socrates is mortal.
    All humans are mortal. Socrates is human. Therefore, Socrates is mortal.
    """
    
    result = module.verify(query, response)
    
    print(f"\nTest 1: Valid Syllogism")
    print(f"Statements Extracted: {len(result.statements)}")
    for stmt in result.statements[:3]:
        print(f"  - {stmt.statement_type.value}: {stmt.symbolic_form}")
    
    print(f"\nVerification: {result.verification_result.value}")
    print(f"Validity Score: {result.validity_score:.2f}")
    print(f"Consistent: {result.consistency.is_consistent}")
    print(f"Fallacies: {len(result.fallacies_detected)}")
    
    # Test 2: Fallacy detection
    print("\n" + "=" * 60)
    print("FALLACY DETECTION TEST")
    print("=" * 60)
    
    fallacious_text = """
    You must either support this policy or you hate the country.
    Everyone knows this is true because experts say so.
    You can't trust John's argument because he's not a scientist.
    """
    
    result2 = module.verify("Is this valid?", fallacious_text)
    
    print(f"\nFallacies Detected: {len(result2.fallacies_detected)}")
    for fallacy in result2.fallacies_detected:
        print(f"  - {fallacy.fallacy_type.value}: {fallacy.description}")
        print(f"    Suggestion: {fallacy.suggestion}")
    
    print(f"\nVerification: {result2.verification_result.value}")
    print(f"Fallacy Risk Score: {result2.fallacy_risk_score:.2f}")
    
    # Test 3: Contradiction detection
    print("\n" + "=" * 60)
    print("CONTRADICTION DETECTION TEST")
    print("=" * 60)
    
    contradictory_text = """
    All cats are mammals. Dogs are not mammals.
    Cats can fly. Cats cannot fly.
    """
    
    result3 = module.verify("Check for contradictions", contradictory_text)
    
    print(f"\nConsistent: {result3.consistency.is_consistent}")
    print(f"Contradictions: {len(result3.consistency.contradictions)}")
    print(f"Explanation: {result3.consistency.explanation}")
    print(f"Processing Time: {result3.processing_time_ms:.1f}ms")

