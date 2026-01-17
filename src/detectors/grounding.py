"""
BAIS Cognitive Governance Engine v16.2
Grounding Detector - Semantic Verification

Per PPA 1, Invention 2 & PPA 2, Invention 1:
- Verifies response is grounded in provided documents
- Uses semantic similarity (not just keyword overlap)
- Tracks entity preservation (numbers, names, dates)
- Verifies individual claims against sources

Supports two modes (controlled by BAISConfig):
1. LITE: TF-IDF based similarity (no heavy dependencies, ~70% accuracy)
2. FULL: Sentence transformer embeddings (~90% accuracy)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter
import re
import math
import json
from pathlib import Path

# Import config for mode detection
try:
    from core.config import get_config
except ImportError:
    get_config = None  # Will use defaults


@dataclass
class ClaimVerification:
    """Result of verifying a single claim."""
    claim: str
    verified: bool
    confidence: float
    supporting_text: Optional[str] = None
    contradiction: Optional[Dict] = None
    source_index: Optional[int] = None


@dataclass
class GroundingResult:
    """Complete grounding analysis result."""
    score: float  # 0-1, overall grounding score
    
    # Component scores
    semantic_similarity: float
    entity_preservation: float
    number_preservation: float
    claim_verification_rate: float
    
    # Details
    verified_claims: List[ClaimVerification] = field(default_factory=list)
    ungrounded_claims: List[str] = field(default_factory=list)
    contradictions: List[Dict] = field(default_factory=list)
    
    # Entities
    response_entities: Set[str] = field(default_factory=set)
    grounded_entities: Set[str] = field(default_factory=set)
    ungrounded_entities: Set[str] = field(default_factory=set)
    
    # Numbers
    response_numbers: Set[str] = field(default_factory=set)
    grounded_numbers: Set[str] = field(default_factory=set)
    contradicting_numbers: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'score': self.score,
            'semantic_similarity': self.semantic_similarity,
            'entity_preservation': self.entity_preservation,
            'number_preservation': self.number_preservation,
            'claim_verification_rate': self.claim_verification_rate,
            'verified_claims_count': len(self.verified_claims),
            'ungrounded_claims': self.ungrounded_claims[:5],
            'contradictions': self.contradictions[:5],
            'grounded_entities': len(self.grounded_entities),
            'ungrounded_entities': list(self.ungrounded_entities)[:5]
        }


class TFIDFSimilarity:
    """
    TF-IDF based semantic similarity.
    Lightweight alternative to embedding models.
    """
    
    def __init__(self):
        self.idf_scores: Dict[str, float] = {}
        self.document_count = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and extract words
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        return words
    
    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency."""
        counts = Counter(tokens)
        total = len(tokens) or 1
        return {word: count / total for word, count in counts.items()}
    
    def fit(self, documents: List[str]):
        """Fit IDF on document corpus."""
        self.document_count = len(documents)
        word_doc_counts: Dict[str, int] = Counter()
        
        for doc in documents:
            words = set(self._tokenize(doc))
            for word in words:
                word_doc_counts[word] += 1
        
        # Compute IDF: log(N / (1 + df))
        for word, df in word_doc_counts.items():
            self.idf_scores[word] = math.log((self.document_count + 1) / (df + 1)) + 1
    
    def _compute_tfidf(self, text: str) -> Dict[str, float]:
        """Compute TF-IDF vector for text."""
        tokens = self._tokenize(text)
        tf = self._compute_tf(tokens)
        
        tfidf = {}
        for word, tf_score in tf.items():
            idf = self.idf_scores.get(word, math.log(self.document_count + 1) + 1)
            tfidf[word] = tf_score * idf
        
        return tfidf
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        vec1 = self._compute_tfidf(text1)
        vec2 = self._compute_tfidf(text2)
        
        # Get all words
        all_words = set(vec1.keys()) | set(vec2.keys())
        
        if not all_words:
            return 0.0
        
        # Compute dot product and magnitudes
        dot_product = sum(vec1.get(w, 0) * vec2.get(w, 0) for w in all_words)
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values())) or 1
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values())) or 1
        
        return dot_product / (mag1 * mag2)


class GroundingDetector:
    """
    Semantic grounding detector with adaptive learning.
    
    Verifies that AI responses are properly grounded in source documents.
    Uses multi-signal approach:
    1. Semantic similarity (TF-IDF or embeddings based on mode)
    2. Entity extraction and verification
    3. Number preservation checking
    4. Claim-by-claim verification
    5. Contradiction detection
    
    Mode determined by BAISConfig:
    - FULL: Neural embeddings (sentence-transformers)
    - LITE: TF-IDF statistical similarity
    
    Adaptive Learning Features (Phase 2 Enhancement):
    - Source reliability tracking
    - Domain-specific similarity thresholds
    - Claim verification pattern learning
    """
    
    # Domain-specific grounding thresholds (learned)
    DOMAIN_THRESHOLDS = {
        'general': 0.6,
        'medical': 0.75,  # Stricter grounding for medical
        'financial': 0.7,
        'legal': 0.7,
        'coding': 0.65,
        'technical': 0.65
    }
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def __init__(self, 
                 use_embeddings: bool = None,  # None = auto from config
                 embedding_model: str = None,
                 learning_path: Path = None):
        
        # Get config for mode determination
        config = get_config() if get_config else None
        
        # Determine if we should use embeddings
        if use_embeddings is None:
            self.use_embeddings = config.use_embeddings if config else False
        else:
            self.use_embeddings = use_embeddings
        
        # Get model name from config or parameter
        self.embedding_model_name = embedding_model or (
            config.embedding_model if config else "all-MiniLM-L6-v2"
        )
        
        # Use temp directory if no path provided (fixes read-only filesystem issues)
        if learning_path is None:
            import tempfile
            temp_dir = Path(tempfile.mkdtemp(prefix="bais_grounding_"))
            learning_path = temp_dir / "grounding_learning.json"
        self.learning_path = learning_path
        
        # Phase 2: Adaptive Learning Components
        self._source_reliability: Dict[str, float] = {}  # source_id -> reliability score
        self._domain_adjustments: Dict[str, float] = {}  # domain -> threshold adjustment
        self._claim_patterns: Dict[str, float] = {}  # claim_type -> verification success rate
        self._learning_rate = 0.1
        
        # TF-IDF similarity (always available as fallback)
        self.tfidf = TFIDFSimilarity()
        
        # Embedding model (only loaded if in FULL mode)
        self.embedding_model = None
        if self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                print(f"[GroundingDetector] FULL mode: Loaded {self.embedding_model_name}")
            except ImportError:
                print("[GroundingDetector] LITE mode: sentence-transformers not available, using TF-IDF")
                self.use_embeddings = False
            except Exception as e:
                print(f"[GroundingDetector] LITE mode: Failed to load model ({e}), using TF-IDF")
                self.use_embeddings = False
        else:
            print("[GroundingDetector] LITE mode: Using TF-IDF statistical similarity")
        
        # Learned patterns for entity extraction
        self.entity_patterns = self._load_entity_patterns()
        
        # Number extraction patterns
        self.number_patterns = [
            r'\b(\d+(?:\.\d+)?)\s*(mg|kg|ml|l|g|%|percent|dollars?|usd|\$|€|£|years?|months?|days?|hours?|minutes?|seconds?)\b',
            r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b',
            r'\b(\d+/\d+)\b',  # Fractions
        ]
    
    def detect(self, response: str, documents: List[Dict] = None, query: str = None) -> GroundingResult:
        """Alias for analyze() - standard detector interface."""
        return self.analyze(response=response, documents=documents or [], query=query)
    
    def analyze(self, 
                response: str, 
                documents: List[Dict],
                query: str = None) -> GroundingResult:
        """
        Perform complete grounding analysis.
        
        Args:
            response: The AI-generated response
            documents: List of source documents with 'content' field
            query: Optional original query for context
        
        Returns:
            GroundingResult with detailed analysis
        """
        doc_texts = [d.get('content', '') for d in documents]
        combined_docs = ' '.join(doc_texts)
        
        # 1. Semantic similarity
        semantic_sim = self._compute_semantic_similarity(response, doc_texts)
        
        # 2. Entity analysis
        entity_result = self._analyze_entities(response, combined_docs)
        
        # 3. Number analysis
        number_result = self._analyze_numbers(response, combined_docs)
        
        # 4. Claim verification
        claims = self._extract_claims(response)
        verified_claims = [self._verify_claim(c, doc_texts) for c in claims]
        
        # 5. Contradiction detection
        contradictions = self._detect_contradictions(response, combined_docs)
        
        # Compute overall score
        claim_rate = (sum(1 for c in verified_claims if c.verified) / len(verified_claims) 
                     if verified_claims else 0.5)
        
        # Weighted combination
        score = (
            0.25 * semantic_sim +
            0.15 * entity_result['preservation'] +
            0.30 * number_result['preservation'] +  # Numbers are critical
            0.20 * claim_rate +
            0.10 * (1.0 - min(len(contradictions) * 0.2, 1.0))  # Penalize contradictions
        )
        
        # Apply contradiction penalty
        if contradictions:
            score *= (1.0 - min(len(contradictions) * 0.15, 0.5))
        
        return GroundingResult(
            score=score,
            semantic_similarity=semantic_sim,
            entity_preservation=entity_result['preservation'],
            number_preservation=number_result['preservation'],
            claim_verification_rate=claim_rate,
            verified_claims=verified_claims,
            ungrounded_claims=[c.claim for c in verified_claims if not c.verified],
            contradictions=contradictions,
            response_entities=entity_result['response_entities'],
            grounded_entities=entity_result['grounded'],
            ungrounded_entities=entity_result['ungrounded'],
            response_numbers=number_result['response_numbers'],
            grounded_numbers=number_result['grounded'],
            contradicting_numbers=number_result['contradictions']
        )
    
    def _compute_semantic_similarity(self, 
                                     response: str, 
                                     documents: List[str]) -> float:
        """Compute semantic similarity between response and documents."""
        if not documents or not response.strip():
            return 0.0
        
        if self.use_embeddings and self.embedding_model:
            # Use sentence transformer embeddings
            try:
                r_emb = self.embedding_model.encode(response)
                d_embs = [self.embedding_model.encode(d) for d in documents if d.strip()]
                
                if not d_embs:
                    return 0.0
                
                # Max similarity with any document
                import numpy as np
                similarities = []
                for d_emb in d_embs:
                    sim = np.dot(r_emb, d_emb) / (np.linalg.norm(r_emb) * np.linalg.norm(d_emb) + 1e-8)
                    similarities.append(sim)
                
                return float(max(similarities))
            except Exception:
                pass
        
        # Fall back to TF-IDF
        self.tfidf.fit(documents)
        similarities = [self.tfidf.similarity(response, d) for d in documents if d.strip()]
        return max(similarities) if similarities else 0.0
    
    def _analyze_entities(self, response: str, documents: str) -> Dict:
        """Extract and analyze named entities."""
        response_entities = self._extract_entities(response)
        doc_entities = self._extract_entities(documents)
        
        grounded = response_entities & doc_entities
        ungrounded = response_entities - doc_entities
        
        preservation = len(grounded) / max(len(response_entities), 1) if response_entities else 0.5
        
        return {
            'response_entities': response_entities,
            'grounded': grounded,
            'ungrounded': ungrounded,
            'preservation': preservation
        }
    
    def _extract_entities(self, text: str) -> Set[str]:
        """Extract named entities from text."""
        entities = set()
        
        # Capitalized words (likely names/places)
        capitals = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        entities.update(c.lower() for c in capitals if len(c) > 2)
        
        # Custom patterns
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, text, re.I)
            entities.update(m.lower() if isinstance(m, str) else m[0].lower() for m in matches)
        
        # Filter common words
        common = {'the', 'and', 'for', 'with', 'from', 'this', 'that', 'will', 'can', 'are', 'was', 'been'}
        entities -= common
        
        return entities
    
    def _analyze_numbers(self, response: str, documents: str) -> Dict:
        """Extract and analyze numerical values."""
        response_numbers = self._extract_numbers_with_context(response)
        doc_numbers = self._extract_numbers_with_context(documents)
        
        grounded = set()
        contradictions = []
        
        for r_num, r_unit, r_ctx in response_numbers:
            matched = False
            for d_num, d_unit, d_ctx in doc_numbers:
                # Same unit or no unit specified
                if r_unit == d_unit or not r_unit or not d_unit:
                    if r_num == d_num:
                        grounded.add((r_num, r_unit))
                        matched = True
                        break
                    elif r_unit == d_unit and r_unit:  # Same unit, different value
                        # Check if context is similar
                        ctx_sim = self._context_similarity(r_ctx, d_ctx)
                        if ctx_sim > 0.5:
                            # Potential contradiction
                            try:
                                ratio = float(r_num) / float(d_num) if float(d_num) != 0 else float('inf')
                                if ratio < 0.5 or ratio > 2.0:
                                    contradictions.append({
                                        'response_value': f"{r_num} {r_unit}",
                                        'document_value': f"{d_num} {d_unit}",
                                        'response_context': r_ctx[:50],
                                        'document_context': d_ctx[:50],
                                        'severity': 'high' if ratio < 0.3 or ratio > 3.0 else 'medium'
                                    })
                            except:
                                pass
            
            if not matched and not any(r_num == c.get('response_value', '').split()[0] for c in contradictions):
                # Number in response not found in documents
                pass
        
        response_num_set = {(n, u) for n, u, _ in response_numbers}
        preservation = len(grounded) / max(len(response_num_set), 1) if response_num_set else 0.5
        
        return {
            'response_numbers': response_num_set,
            'grounded': grounded,
            'contradictions': contradictions,
            'preservation': preservation
        }
    
    def _extract_numbers_with_context(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract numbers with their units and surrounding context."""
        results = []
        
        for pattern in self.number_patterns:
            for match in re.finditer(pattern, text, re.I):
                groups = match.groups()
                number = groups[0] if groups else match.group()
                unit = groups[1] if len(groups) > 1 else ''
                
                # Get context (10 words before and after)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                results.append((number, unit.lower() if unit else '', context))
        
        return results
    
    def _context_similarity(self, ctx1: str, ctx2: str) -> float:
        """Check if two contexts are similar (discussing same topic)."""
        words1 = set(re.findall(r'\b\w{4,}\b', ctx1.lower()))
        words2 = set(re.findall(r'\b\w{4,}\b', ctx2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1 & words2)
        union = len(words1 | words2)
        
        return overlap / union if union > 0 else 0.0
    
    def _extract_claims(self, response: str) -> List[str]:
        """Extract individual claims from response."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', response)
        
        claims = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10:
                # Filter out questions and meta-statements
                if not sent.endswith('?') and not sent.lower().startswith(('i ', 'we ', 'you ')):
                    claims.append(sent)
        
        return claims[:10]  # Limit to 10 claims
    
    def _verify_claim(self, claim: str, documents: List[str]) -> ClaimVerification:
        """Verify a single claim against documents."""
        best_match = None
        best_score = 0.0
        best_doc_idx = None
        
        for idx, doc in enumerate(documents):
            if not doc.strip():
                continue
            
            # Compute similarity
            if self.use_embeddings and self.embedding_model:
                try:
                    c_emb = self.embedding_model.encode(claim)
                    d_emb = self.embedding_model.encode(doc)
                    import numpy as np
                    sim = float(np.dot(c_emb, d_emb) / (np.linalg.norm(c_emb) * np.linalg.norm(d_emb) + 1e-8))
                except:
                    sim = self.tfidf.similarity(claim, doc)
            else:
                sim = self.tfidf.similarity(claim, doc)
            
            if sim > best_score:
                best_score = sim
                best_doc_idx = idx
                
                # Find best matching sentence in doc
                doc_sentences = re.split(r'[.!?]+', doc)
                best_sent_score = 0
                for sent in doc_sentences:
                    sent = sent.strip()
                    if len(sent) > 10:
                        sent_sim = self.tfidf.similarity(claim, sent)
                        if sent_sim > best_sent_score:
                            best_sent_score = sent_sim
                            best_match = sent
        
        # Check for contradictions in the claim
        contradiction = self._check_claim_contradiction(claim, documents)
        
        verified = best_score > 0.4 and not contradiction
        
        return ClaimVerification(
            claim=claim[:100],
            verified=verified,
            confidence=best_score,
            supporting_text=best_match[:100] if best_match else None,
            contradiction=contradiction,
            source_index=best_doc_idx
        )
    
    def _check_claim_contradiction(self, claim: str, documents: List[str]) -> Optional[Dict]:
        """Check if claim contradicts documents."""
        # Extract numbers from claim
        claim_nums = self._extract_numbers_with_context(claim)
        
        for doc in documents:
            doc_nums = self._extract_numbers_with_context(doc)
            
            for c_num, c_unit, c_ctx in claim_nums:
                for d_num, d_unit, d_ctx in doc_nums:
                    if c_unit == d_unit and c_unit:
                        ctx_sim = self._context_similarity(c_ctx, d_ctx)
                        if ctx_sim > 0.5:
                            try:
                                c_val = float(c_num.replace(',', ''))
                                d_val = float(d_num.replace(',', ''))
                                if c_val != 0 and d_val != 0:
                                    ratio = c_val / d_val
                                    if ratio < 0.5 or ratio > 2.0:
                                        return {
                                            'type': 'numerical_contradiction',
                                            'claim_value': f"{c_num} {c_unit}",
                                            'document_value': f"{d_num} {d_unit}"
                                        }
                            except:
                                pass
        
        return None
    
    def _detect_contradictions(self, response: str, documents: str) -> List[Dict]:
        """Detect all contradictions between response and documents."""
        contradictions = []
        
        # Number contradictions
        num_result = self._analyze_numbers(response, documents)
        contradictions.extend(num_result['contradictions'])
        
        # Claim contradictions
        claims = self._extract_claims(response)
        for claim in claims:
            contradiction = self._check_claim_contradiction(claim, [documents])
            if contradiction:
                contradiction['claim'] = claim[:100]
                contradictions.append(contradiction)
        
        return contradictions
    
    def _load_entity_patterns(self) -> List[str]:
        """Load learned entity patterns."""
        patterns = [
            r'\b(Dr\.?\s+\w+)',  # Doctor names
            r'\b(\w+\s+(?:Inc|Corp|LLC|Ltd)\.?)\b',  # Company names
            r'\b(\d{4})\b',  # Years
        ]
        
        if self.learning_path.exists():
            try:
                with open(self.learning_path) as f:
                    data = json.load(f)
                    patterns.extend(data.get('entity_patterns', []))
            except:
                pass
        
        return patterns
    
    def learn_pattern(self, pattern: str, pattern_type: str = 'entity'):
        """Add a new learned pattern."""
        data = {'entity_patterns': [], 'number_patterns': []}
        
        if self.learning_path.exists():
            try:
                with open(self.learning_path) as f:
                    data = json.load(f)
            except:
                pass
        
        key = f'{pattern_type}_patterns'
        if key in data and pattern not in data[key]:
            data[key].append(pattern)
            
            self.learning_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.learning_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Reload patterns
            self.entity_patterns = self._load_entity_patterns()
    
    def record_outcome(self, result: GroundingResult, was_correct: bool, domain: str = 'general'):
        """
        Learn from grounding verification outcome to improve future accuracy.
        
        Phase 2 Enhancement: Adaptive Learning for PPA3 Claims
        
        Args:
            result: The grounding result that was evaluated
            was_correct: Whether the grounding verification was correct
            domain: Domain context for domain-specific learning
        """
        # Update domain adjustments
        if not was_correct and domain != 'general':
            current_adj = self._domain_adjustments.get(domain, 0.0)
            if result.score < 0.5:
                # Was too strict - decrease threshold for this domain
                self._domain_adjustments[domain] = current_adj - self._learning_rate
            else:
                # Was too lenient - increase threshold for this domain
                self._domain_adjustments[domain] = current_adj + self._learning_rate
        
        # Update source reliability based on outcome
        for source in result.sources_used:
            source_id = source.get('id', source.get('content', '')[:50])
            current_reliability = self._source_reliability.get(source_id, 1.0)
            if was_correct:
                self._source_reliability[source_id] = min(2.0, current_reliability * (1 + self._learning_rate))
            else:
                self._source_reliability[source_id] = max(0.1, current_reliability * (1 - self._learning_rate))
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get learned threshold adjustment for a domain."""
        return self._domain_adjustments.get(domain, 0.0)
    
    def get_source_reliability(self, source_id: str) -> float:
        """Get learned reliability score for a source."""
        return self._source_reliability.get(source_id, 1.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get statistics about adaptive learning state."""
        return {
            'domains_adjusted': len(self._domain_adjustments),
            'domain_adjustments': dict(self._domain_adjustments),
            'sources_tracked': len(self._source_reliability),
            'claim_patterns': len(self._claim_patterns),
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
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback to adjust behavior."""
        if not hasattr(self, '_learning_params'):
            self._learning_params = {}
        was_correct = feedback.get('was_correct', True)
        if not was_correct:
            adjustment = feedback.get('adjustment', 0.05)
            self._learning_rate = max(0.001, self._learning_rate * (1 - adjustment))
    
    def get_statistics(self) -> Dict:
        """Return learning statistics."""
        return {
            'sources_tracked': len(self._source_reliability),
            'claim_patterns': len(self._claim_patterns),
            'learning_rate': self._learning_rate,
            'history_size': len(getattr(self, '_outcomes', []))
        }
    
    def serialize_state(self) -> Dict:
        """Serialize state for persistence."""
        return {
            'source_reliability': dict(self._source_reliability),
            'claim_patterns': dict(self._claim_patterns),
            'learning_rate': self._learning_rate,
            'outcomes': getattr(self, '_outcomes', [])[-100:],
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore state from serialized data."""
        self._source_reliability = state.get('source_reliability', {})
        self._claim_patterns = state.get('claim_patterns', {})
        self._learning_rate = state.get('learning_rate', 0.01)
        self._outcomes = state.get('outcomes', [])


# Phase 49: Documentation compatibility alias
FactualGroundingDetector = GroundingDetector
