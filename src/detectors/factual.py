"""
BASE Cognitive Governance Engine v16.2
Factual Detector - Entailment and Contradiction

Per PPA 2, Invention 1 (Must-Pass Predicates):
- Verifies response actually answers the query
- Detects factual contradictions with source documents
- Checks claim-by-claim entailment
- Identifies unsupported assertions

Mode determined by BASEConfig:
- FULL: NLI model for entailment (CrossEncoder, ~90% accuracy)
- LITE: Rule-based + word overlap (~70% accuracy)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import re
import json
from pathlib import Path

# Import config for mode detection
try:
    from core.config import get_config
except ImportError:
    get_config = None


@dataclass
class FactualClaim:
    """A single factual claim extracted from text."""
    text: str
    claim_type: str  # 'numerical', 'assertion', 'comparison', 'temporal'
    confidence: float = 0.5
    entities: List[str] = field(default_factory=list)
    numbers: List[Tuple[str, str]] = field(default_factory=list)  # (value, unit)


@dataclass
class EntailmentResult:
    """Result of checking if source entails claim."""
    claim: str
    entailed: bool
    confidence: float
    supporting_text: Optional[str] = None
    contradiction: Optional[Dict] = None
    label: str = "neutral"  # 'entailment', 'contradiction', 'neutral'


@dataclass
class FactualAnalysis:
    """Complete factual analysis result."""
    score: float  # 0-1, overall factual accuracy
    
    # Query-Response alignment
    query_answered: bool
    query_coverage: float
    
    # Claim analysis
    total_claims: int
    entailed_claims: int
    contradicted_claims: int
    neutral_claims: int
    
    # Detailed results
    claim_results: List[EntailmentResult] = field(default_factory=list)
    contradictions: List[Dict] = field(default_factory=list)
    unsupported_assertions: List[str] = field(default_factory=list)
    
    # Additional signals
    has_citations: bool = False
    citation_accuracy: float = 0.0
    speculation_level: float = 0.0  # 0-1, how much speculation
    
    def to_dict(self) -> Dict:
        return {
            'score': self.score,
            'query_answered': self.query_answered,
            'query_coverage': self.query_coverage,
            'claims': {
                'total': self.total_claims,
                'entailed': self.entailed_claims,
                'contradicted': self.contradicted_claims,
                'neutral': self.neutral_claims
            },
            'contradictions': self.contradictions[:5],
            'unsupported': self.unsupported_assertions[:5],
            'speculation_level': self.speculation_level,
            'has_citations': self.has_citations
        }


class FactualDetector:
    """
    Factual accuracy detector.
    
    Performs:
    1. Query-response alignment (does response answer the query?)
    2. Claim extraction from response
    3. Entailment checking against source documents
    4. Contradiction detection
    5. Speculation detection
    """
    
    # Claim extraction patterns
    CLAIM_PATTERNS = {
        'numerical': [
            r'(?:is|are|was|were|equals?|costs?|contains?)\s+(?:about\s+)?(\d+(?:\.\d+)?)\s*(\w+)?',
            r'(\d+(?:\.\d+)?)\s*(\w+)?\s+(?:is|are|was|were)',
        ],
        'assertion': [
            r'^([A-Z][^.!?]*(?:is|are|was|were|has|have|will|can)[^.!?]*)[.!?]',
            r'(?:therefore|thus|hence|consequently)\s*,?\s*([^.!?]+)[.!?]',
        ],
        'comparison': [
            r'(\w+)\s+(?:is|are)\s+(?:more|less|greater|smaller|higher|lower|better|worse)\s+than\s+(\w+)',
            r'(?:compared\s+to|versus|vs\.?)\s+([^,]+)',
        ],
        'temporal': [
            r'(?:in|on|at|during|since|until|before|after)\s+(\d{4}|\d{1,2}/\d{1,2}(?:/\d{2,4})?)',
            r'(\d+)\s+(?:years?|months?|days?|hours?)\s+(?:ago|from\s+now)',
        ]
    }
    
    # Speculation indicators
    SPECULATION_PATTERNS = [
        r'\b(?:might|may|could|possibly|perhaps|probably|likely)\b',
        r'\b(?:I\s+think|I\s+believe|in\s+my\s+opinion)\b',
        r'\b(?:it\s+(?:seems?|appears?))\b',
        r'\b(?:reportedly|allegedly|supposedly)\b',
    ]
    
    # Certainty indicators
    CERTAINTY_PATTERNS = [
        r'\b(?:definitely|certainly|absolutely|undoubtedly)\b',
        r'\b(?:is\s+(?:true|false|correct|incorrect))\b',
        r'\b(?:always|never|every|none)\b',
    ]
    
    # Domain-specific factual thresholds (learned)
    DOMAIN_THRESHOLDS = {
        'general': 0.6,
        'medical': 0.8,  # Strictest for medical facts
        'financial': 0.75,
        'legal': 0.75,
        'coding': 0.7,
        'technical': 0.65
    }
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def __init__(self, 
                 use_nli: bool = None,  # None = auto from config
                 nli_model: str = None,
                 learning_path: Path = None):
        
        # Get config for mode determination
        config = get_config() if get_config else None
        
        # Determine if we should use NLI
        if use_nli is None:
            self.use_nli = config.use_nli if config else False
        else:
            self.use_nli = use_nli
        
        # Get model name from config or parameter
        self.nli_model_name = nli_model or (
            config.nli_model if config else "cross-encoder/nli-deberta-v3-small"
        )
        
        # Use temp directory if no path provided (fixes read-only filesystem issues)
        if learning_path is None:
            import tempfile
            temp_dir = Path(tempfile.mkdtemp(prefix="base_factual_"))
            learning_path = temp_dir / "factual_learning.json"
        self.learning_path = learning_path
        
        # Phase 2: Adaptive Learning Components
        self._claim_type_accuracy: Dict[str, List[bool]] = {}  # claim_type -> accuracy history
        self._domain_adjustments: Dict[str, float] = {}  # domain -> threshold adjustment
        self._learning_rate = 0.1
        
        # NLI model (only loaded if in FULL mode)
        self.nli_model = None
        if self.use_nli:
            try:
                from sentence_transformers import CrossEncoder
                self.nli_model = CrossEncoder(self.nli_model_name)
                print(f"[FactualDetector] FULL mode: Loaded {self.nli_model_name}")
            except ImportError:
                print("[FactualDetector] LITE mode: sentence-transformers not available, using rule-based")
                self.use_nli = False
            except Exception as e:
                print(f"[FactualDetector] LITE mode: Failed to load model ({e}), using rule-based")
                self.use_nli = False
        else:
            print("[FactualDetector] LITE mode: Using rule-based entailment")
        
        # Learned patterns
        self.learned_claim_patterns: List[str] = []
        self._load_learning()
    
    def analyze(self, 
                query: str = None,
                response: str = None,
                documents: List[Dict] = None) -> FactualAnalysis:
        """
        Perform complete factual analysis.
        
        Args:
            query: Original user query
            response: AI-generated response
            documents: Source documents
        
        Returns:
            FactualAnalysis with detailed results
        """
        # Handle None inputs
        query = query or ""
        response = response or ""
        documents = documents or []
        
        doc_text = " ".join([d.get('content', '') for d in documents])
        
        # 1. Check if response answers the query
        query_answered, query_coverage = self._check_query_answered(query, response)
        
        # 2. Extract claims from response
        claims = self._extract_claims(response)
        
        # 3. Check each claim against documents
        claim_results = [self._check_entailment(c, doc_text, documents) for c in claims]
        
        # 4. Identify contradictions
        contradictions = [r.contradiction for r in claim_results if r.contradiction]
        
        # 5. Count by label
        entailed = sum(1 for r in claim_results if r.label == 'entailment')
        contradicted = sum(1 for r in claim_results if r.label == 'contradiction')
        neutral = sum(1 for r in claim_results if r.label == 'neutral')
        
        # 6. Find unsupported assertions
        unsupported = [r.claim for r in claim_results 
                      if r.label == 'neutral' and r.confidence < 0.3]
        
        # 7. Check for citations
        has_citations = bool(re.search(r'\[\d+\]|\(\d{4}\)|according\s+to|source:', response, re.I))
        citation_accuracy = self._check_citations(response, documents) if has_citations else 0.0
        
        # 8. Detect speculation level
        speculation_level = self._compute_speculation_level(response)
        
        # Compute overall score
        if claim_results:
            entailment_ratio = entailed / len(claim_results)
            contradiction_penalty = contradicted / len(claim_results) * 0.5
            
            score = (
                0.3 * (1.0 if query_answered else 0.5) +
                0.4 * entailment_ratio +
                0.2 * (1.0 - contradiction_penalty) +
                0.1 * (1.0 - speculation_level)
            )
        else:
            score = 0.5 if query_answered else 0.3
        
        return FactualAnalysis(
            score=score,
            query_answered=query_answered,
            query_coverage=query_coverage,
            total_claims=len(claims),
            entailed_claims=entailed,
            contradicted_claims=contradicted,
            neutral_claims=neutral,
            claim_results=claim_results,
            contradictions=contradictions,
            unsupported_assertions=unsupported,
            has_citations=has_citations,
            citation_accuracy=citation_accuracy,
            speculation_level=speculation_level
        )
    
    def _check_query_answered(self, query: str, response: str) -> Tuple[bool, float]:
        """Check if response actually answers the query."""
        # Extract query keywords
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        
        # Remove common words
        common = {'what', 'where', 'when', 'how', 'why', 'who', 'which', 'the', 
                 'and', 'for', 'with', 'can', 'does', 'about', 'this', 'that'}
        query_words -= common
        
        # Check response coverage
        response_words = set(re.findall(r'\b\w{3,}\b', response.lower()))
        
        if not query_words:
            return True, 1.0
        
        coverage = len(query_words & response_words) / len(query_words)
        
        # Also check for refusal patterns
        refusal_patterns = [
            r"I\s+(?:cannot|can't|am\s+unable)",
            r"(?:beyond|outside)\s+(?:my|the)\s+(?:scope|ability)",
            r"I\s+don't\s+have\s+(?:access|information)",
        ]
        
        is_refusal = any(re.search(p, response, re.I) for p in refusal_patterns)
        
        # Refusal is acceptable if justified
        if is_refusal:
            return True, coverage  # Justified refusal counts as answered
        
        answered = coverage >= 0.3
        return answered, coverage
    
    def _extract_claims(self, response: str) -> List[FactualClaim]:
        """Extract factual claims from response."""
        claims = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', response)
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue
            
            # Skip questions and meta-statements
            if sent.endswith('?') or sent.lower().startswith(('i ', 'let me ', 'here')):
                continue
            
            # Determine claim type
            claim_type = self._classify_claim(sent)
            
            # Extract entities and numbers
            entities = self._extract_entities(sent)
            numbers = self._extract_numbers(sent)
            
            # Only add if it looks like a factual statement
            if claim_type != 'unknown' or numbers or entities:
                claims.append(FactualClaim(
                    text=sent[:200],
                    claim_type=claim_type,
                    confidence=0.7 if numbers else 0.5,
                    entities=entities,
                    numbers=numbers
                ))
        
        return claims[:15]  # Limit to 15 claims
    
    def _classify_claim(self, text: str) -> str:
        """Classify the type of claim."""
        for claim_type, patterns in self.CLAIM_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.I):
                    return claim_type
        
        # Check for assertion patterns
        if re.match(r'^[A-Z].*(?:is|are|was|were|has|have)', text):
            return 'assertion'
        
        return 'unknown'
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        entities = []
        
        # Capitalized phrases (names, places, organizations)
        caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        entities.extend(caps)
        
        return list(set(entities))[:5]
    
    def _extract_numbers(self, text: str) -> List[Tuple[str, str]]:
        """Extract numbers with units from text."""
        numbers = []
        
        pattern = r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(%|percent|dollars?|usd|\$|€|£|mg|kg|ml|l|g|years?|months?|days?|hours?|minutes?)?'
        
        for match in re.finditer(pattern, text, re.I):
            value = match.group(1).replace(',', '')
            unit = match.group(2) or ''
            numbers.append((value, unit.lower()))
        
        return numbers
    
    def _check_entailment(self, 
                          claim: FactualClaim,
                          doc_text: str,
                          documents: List[Dict]) -> EntailmentResult:
        """Check if source documents entail the claim."""
        
        # Use NLI model if available
        if self.use_nli and self.nli_model:
            return self._nli_entailment(claim, documents)
        
        # Otherwise use rule-based approach
        return self._rule_based_entailment(claim, doc_text)
    
    def _nli_entailment(self, 
                        claim: FactualClaim, 
                        documents: List[Dict]) -> EntailmentResult:
        """Use NLI model for entailment checking."""
        best_result = EntailmentResult(
            claim=claim.text,
            entailed=False,
            confidence=0.0,
            label='neutral'
        )
        
        try:
            for doc in documents:
                doc_text = doc.get('content', '')
                if not doc_text:
                    continue
                
                # NLI expects (premise, hypothesis) pairs
                scores = self.nli_model.predict([(doc_text[:512], claim.text)])
                
                # Scores are typically [contradiction, entailment, neutral]
                if hasattr(scores, '__len__') and len(scores) >= 3:
                    if scores[1] > best_result.confidence:  # Entailment score
                        best_result = EntailmentResult(
                            claim=claim.text,
                            entailed=scores[1] > 0.5,
                            confidence=float(scores[1]),
                            label='entailment' if scores[1] > 0.5 else (
                                'contradiction' if scores[0] > 0.5 else 'neutral'
                            )
                        )
        except Exception as e:
            print(f"NLI error: {e}")
            return self._rule_based_entailment(claim, ' '.join([d.get('content', '') for d in documents]))
        
        return best_result
    
    def _rule_based_entailment(self, 
                               claim: FactualClaim, 
                               doc_text: str) -> EntailmentResult:
        """Rule-based entailment checking."""
        # Check for numerical contradictions first
        contradiction = self._check_numerical_contradiction(claim, doc_text)
        if contradiction:
            return EntailmentResult(
                claim=claim.text,
                entailed=False,
                confidence=0.8,
                contradiction=contradiction,
                label='contradiction'
            )
        
        # Compute word overlap
        claim_words = set(re.findall(r'\b\w{4,}\b', claim.text.lower()))
        doc_words = set(re.findall(r'\b\w{4,}\b', doc_text.lower()))
        
        if not claim_words:
            return EntailmentResult(
                claim=claim.text,
                entailed=False,
                confidence=0.3,
                label='neutral'
            )
        
        overlap = len(claim_words & doc_words) / len(claim_words)
        
        # Check entity overlap
        entity_overlap = 0.0
        if claim.entities:
            claim_entities_lower = {e.lower() for e in claim.entities}
            doc_entities = set(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', doc_text))
            doc_entities_lower = {e.lower() for e in doc_entities}
            entity_overlap = len(claim_entities_lower & doc_entities_lower) / len(claim.entities)
        
        # Check number overlap
        number_overlap = 0.0
        if claim.numbers:
            claim_nums = {n[0] for n in claim.numbers}
            doc_nums = set(re.findall(r'\b(\d+(?:\.\d+)?)\b', doc_text))
            number_overlap = len(claim_nums & doc_nums) / len(claim.numbers)
        
        # Combined confidence
        confidence = 0.4 * overlap + 0.3 * entity_overlap + 0.3 * number_overlap
        
        # Find supporting text
        supporting = self._find_supporting_sentence(claim.text, doc_text)
        
        if confidence > 0.5:
            label = 'entailment'
            entailed = True
        elif confidence < 0.2:
            label = 'neutral'
            entailed = False
        else:
            label = 'neutral'
            entailed = False
        
        return EntailmentResult(
            claim=claim.text,
            entailed=entailed,
            confidence=confidence,
            supporting_text=supporting,
            label=label
        )
    
    def _check_numerical_contradiction(self, 
                                       claim: FactualClaim, 
                                       doc_text: str) -> Optional[Dict]:
        """Check for numerical contradictions."""
        if not claim.numbers:
            return None
        
        doc_numbers = self._extract_numbers(doc_text)
        
        for c_num, c_unit in claim.numbers:
            for d_num, d_unit in doc_numbers:
                # Same unit (or both without unit)
                if c_unit == d_unit or (not c_unit and not d_unit):
                    # Check context similarity
                    c_ctx = self._get_number_context(c_num, claim.text)
                    d_ctx = self._get_number_context(d_num, doc_text)
                    
                    ctx_similarity = self._context_overlap(c_ctx, d_ctx)
                    
                    if ctx_similarity > 0.4:
                        try:
                            c_val = float(c_num)
                            d_val = float(d_num)
                            
                            if c_val != 0 and d_val != 0:
                                ratio = c_val / d_val
                                
                                if ratio < 0.5 or ratio > 2.0:
                                    return {
                                        'type': 'numerical',
                                        'claim_value': f"{c_num} {c_unit}".strip(),
                                        'document_value': f"{d_num} {d_unit}".strip(),
                                        'ratio': ratio,
                                        'severity': 'high' if ratio < 0.3 or ratio > 3.0 else 'medium'
                                    }
                        except:
                            pass
        
        return None
    
    def _get_number_context(self, number: str, text: str) -> str:
        """Get context around a number in text."""
        pattern = rf'\b.{{0,50}}{re.escape(number)}.{{0,50}}\b'
        match = re.search(pattern, text, re.I)
        return match.group() if match else ""
    
    def _context_overlap(self, ctx1: str, ctx2: str) -> float:
        """Compute context overlap between two strings."""
        words1 = set(re.findall(r'\b\w{4,}\b', ctx1.lower()))
        words2 = set(re.findall(r'\b\w{4,}\b', ctx2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        return len(words1 & words2) / max(len(words1), len(words2))
    
    def _find_supporting_sentence(self, claim: str, doc_text: str) -> Optional[str]:
        """Find the best supporting sentence in document."""
        claim_words = set(re.findall(r'\b\w{4,}\b', claim.lower()))
        
        if not claim_words:
            return None
        
        best_sent = None
        best_overlap = 0.0
        
        for sent in re.split(r'[.!?]+', doc_text):
            sent = sent.strip()
            if len(sent) < 10:
                continue
            
            sent_words = set(re.findall(r'\b\w{4,}\b', sent.lower()))
            if not sent_words:
                continue
            
            overlap = len(claim_words & sent_words) / len(claim_words)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_sent = sent
        
        return best_sent[:200] if best_sent and best_overlap > 0.3 else None
    
    def _check_citations(self, response: str, documents: List[Dict]) -> float:
        """Check accuracy of citations in response."""
        # Extract citations
        citations = re.findall(r'\[(\d+)\]', response)
        
        if not citations:
            return 0.0
        
        # Check if citation numbers are valid
        valid_citations = sum(1 for c in citations if int(c) <= len(documents))
        
        return valid_citations / len(citations)
    
    def _compute_speculation_level(self, response: str) -> float:
        """Compute how much speculation is in the response."""
        # Count speculation indicators
        speculation_count = sum(
            len(re.findall(p, response, re.I)) 
            for p in self.SPECULATION_PATTERNS
        )
        
        # Count certainty indicators
        certainty_count = sum(
            len(re.findall(p, response, re.I)) 
            for p in self.CERTAINTY_PATTERNS
        )
        
        # Normalize by response length
        word_count = len(response.split())
        if word_count == 0:
            return 0.0
        
        speculation_rate = speculation_count / (word_count / 50)  # Per 50 words
        certainty_rate = certainty_count / (word_count / 50)
        
        # Combine: more speculation or over-certainty both reduce score
        level = min(1.0, speculation_rate * 0.2 + (certainty_rate * 0.1 if certainty_rate > 3 else 0))
        
        return level
    
    def _load_learning(self):
        """Load learned patterns."""
        if self.learning_path.exists():
            try:
                with open(self.learning_path) as f:
                    data = json.load(f)
                    self.learned_claim_patterns = data.get('claim_patterns', [])
            except:
                pass
    
    def learn_pattern(self, pattern: str, was_useful: bool):
        """Learn a new claim extraction pattern."""
        if was_useful and pattern not in self.learned_claim_patterns:
            self.learned_claim_patterns.append(pattern)
            
            self.learning_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.learning_path, 'w') as f:
                json.dump({'claim_patterns': self.learned_claim_patterns}, f)
    
    def record_outcome(self, result: 'FactualAnalysis', was_correct: bool, domain: str = 'general'):
        """
        Learn from factual analysis outcome to improve future accuracy.
        
        Phase 2 Enhancement: Adaptive Learning for PPA3 Claims
        
        Args:
            result: The factual analysis result that was evaluated
            was_correct: Whether the factual verification was correct
            domain: Domain context for domain-specific learning
        """
        # Track claim type accuracy
        for claim in result.claims_extracted:
            claim_type = claim.get('type', 'factual')
            if claim_type not in self._claim_type_accuracy:
                self._claim_type_accuracy[claim_type] = []
            self._claim_type_accuracy[claim_type].append(was_correct)
            # Keep only last 100 outcomes per claim type
            if len(self._claim_type_accuracy[claim_type]) > 100:
                self._claim_type_accuracy[claim_type] = self._claim_type_accuracy[claim_type][-100:]
        
        # Update domain adjustments
        if not was_correct and domain != 'general':
            current_adj = self._domain_adjustments.get(domain, 0.0)
            if result.score < 0.5:
                # Was too strict
                self._domain_adjustments[domain] = current_adj - self._learning_rate
            else:
                # Was too lenient
                self._domain_adjustments[domain] = current_adj + self._learning_rate
    
    def get_claim_type_accuracy(self, claim_type: str) -> float:
        """Get learned accuracy rate for a claim type."""
        history = self._claim_type_accuracy.get(claim_type, [])
        if not history:
            return 0.5  # Unknown - assume neutral
        return sum(history) / len(history)
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get learned threshold adjustment for a domain."""
        return self._domain_adjustments.get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get statistics about adaptive learning state."""
        return {
            'domains_adjusted': len(self._domain_adjustments),
            'domain_adjustments': dict(self._domain_adjustments),
            'claim_types_tracked': len(self._claim_type_accuracy),
            'claim_type_accuracy': {
                k: sum(v)/len(v) if v else 0.5 
                for k, v in self._claim_type_accuracy.items()
            },
            'learning_rate': self._learning_rate
        }

    # ========================================
    # LEARNING INTERFACE (Auto-added)
    # ========================================

    def record_feedback(self, result, was_accurate):
        """Record feedback for learning adjustment."""
        self.record_outcome({'result': str(result)}, {}, was_accurate)

    def adapt_thresholds(self, threshold_name, current_value, direction='decrease'):
        """Adapt thresholds based on learning history."""
        adjustment = 0.05 if direction == 'increase' else -0.05
        return max(0.1, min(0.95, current_value + adjustment))



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
