"""
BASE RAG Quality Scoring Pathway
Migrated from Onyx Governance - Enhancement 1

Document quality scoring for RAG systems.

Mathematical Formulation:
Q_rag = α_r * R + α_c * C + α_d * D + α_t * T

Where:
- R = Relevance score (document-query similarity)
- C = Coverage score (query term coverage)
- D = Diversity score (source variety)
- T = Recency score (document freshness)
- α = Adaptive weights (sum to 1.0)

Patent Claims: Enhancement-1
Proprietary IP - 100% owned by Invitas Inc.
"""

import re
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class RAGQualityScore:
    """RAG quality metrics."""
    overall_score: float
    relevance_score: float
    coverage_score: float
    diversity_score: float
    recency_score: float
    document_count: int
    average_document_score: float
    response_alignment: float
    timestamp: float
    processing_time_ms: float = 0.0
    
    # Component weights used
    weights: Dict[str, float] = None


class RAGQualityPathway:
    """
    RAG Quality Scoring Pathway for BASE.
    
    Implements:
    - Document relevance scoring
    - Query coverage analysis
    - Source diversity measurement
    - Document recency assessment
    - Response-document alignment
    
    Supports adaptive weight learning via BASE OCO learner.
    """
    
    def __init__(self,
                 relevance_weight: float = 0.40,
                 coverage_weight: float = 0.30,
                 diversity_weight: float = 0.15,
                 recency_weight: float = 0.15):
        """
        Initialize RAG Quality Pathway.
        
        Args:
            relevance_weight: Weight for relevance score (α_r)
            coverage_weight: Weight for coverage score (α_c)
            diversity_weight: Weight for diversity score (α_d)
            recency_weight: Weight for recency score (α_t)
        """
        self.alpha_r = relevance_weight
        self.alpha_c = coverage_weight
        self.alpha_d = diversity_weight
        self.alpha_t = recency_weight
        
        # Normalize weights to sum to 1.0
        self._normalize_weights()
        
        # Thresholds
        self.min_quality_threshold = 0.40
        self.good_quality_threshold = 0.70
    
    def _normalize_weights(self):
        """Ensure weights sum to 1.0."""
        total = self.alpha_r + self.alpha_c + self.alpha_d + self.alpha_t
        if abs(total - 1.0) > 1e-6:
            self.alpha_r /= total
            self.alpha_c /= total
            self.alpha_d /= total
            self.alpha_t /= total
    
    def score(self,
              query: str,
              documents: List[Dict[str, Any]],
              response: str = None) -> RAGQualityScore:
        """
        Score RAG quality.
        
        Args:
            query: User query
            documents: Retrieved documents
            response: Generated response (optional, for alignment check)
        
        Returns:
            RAGQualityScore with all metrics
        """
        start_time = time.time()
        
        if not documents:
            return RAGQualityScore(
                overall_score=0.0,
                relevance_score=0.0,
                coverage_score=0.0,
                diversity_score=0.0,
                recency_score=0.0,
                document_count=0,
                average_document_score=0.0,
                response_alignment=0.0,
                timestamp=time.time(),
                processing_time_ms=(time.time() - start_time) * 1000,
                weights=self._get_weights()
            )
        
        # Compute individual scores
        relevance = self._compute_relevance(documents)
        coverage = self._compute_coverage(query, documents)
        diversity = self._compute_diversity(documents)
        recency = self._compute_recency(documents)
        
        # Compute response alignment if response provided
        response_alignment = 0.0
        if response:
            response_alignment = self._compute_response_alignment(response, documents)
        
        # Overall score: Q_rag = α_r*R + α_c*C + α_d*D + α_t*T
        overall = (
            self.alpha_r * relevance +
            self.alpha_c * coverage +
            self.alpha_d * diversity +
            self.alpha_t * recency
        )
        
        # Average document score
        doc_scores = [d.get('score', 0.0) for d in documents]
        avg_doc_score = sum(doc_scores) / len(doc_scores) if doc_scores else 0.0
        
        processing_time = (time.time() - start_time) * 1000
        
        return RAGQualityScore(
            overall_score=overall,
            relevance_score=relevance,
            coverage_score=coverage,
            diversity_score=diversity,
            recency_score=recency,
            document_count=len(documents),
            average_document_score=avg_doc_score,
            response_alignment=response_alignment,
            timestamp=time.time(),
            processing_time_ms=processing_time,
            weights=self._get_weights()
        )
    
    def _get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        return {
            'relevance': self.alpha_r,
            'coverage': self.alpha_c,
            'diversity': self.alpha_d,
            'recency': self.alpha_t
        }
    
    def _compute_relevance(self, documents: List[Dict[str, Any]]) -> float:
        """Compute relevance from document retrieval scores."""
        if not documents:
            return 0.0
        
        scores = []
        for doc in documents:
            score = doc.get('score', 0.0)
            # Normalize score to [0, 1]
            if score > 1.0:
                score = min(score / 100.0, 1.0)  # Handle percentage scores
            scores.append(score)
        
        # Weighted average (higher weight to top documents)
        if len(scores) <= 3:
            return sum(scores) / len(scores) if scores else 0.0
        
        # Weight: 1.0, 0.8, 0.6, 0.4, ...
        weights = [max(1.0 - 0.2 * i, 0.2) for i in range(len(scores))]
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        weight_total = sum(weights)
        
        return weighted_sum / weight_total if weight_total > 0 else 0.0
    
    def _compute_coverage(self, query: str, documents: List[Dict[str, Any]]) -> float:
        """Compute how well documents cover query terms."""
        if not query or not documents:
            return 0.0
        
        # Extract query terms (remove stopwords)
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how', 'why',
            'when', 'where', 'which', 'who', 'to', 'of', 'in', 'for', 'on', 'with'
        }
        query_terms = set(
            w.lower() for w in re.findall(r'\b\w+\b', query)
            if w.lower() not in stopwords and len(w) > 2
        )
        
        if not query_terms:
            return 0.5  # Default for very simple queries
        
        # Check coverage across documents
        covered_terms = set()
        for doc in documents[:10]:
            content = doc.get('content', '').lower()
            for term in query_terms:
                if term in content:
                    covered_terms.add(term)
        
        coverage = len(covered_terms) / len(query_terms)
        return min(coverage, 1.0)
    
    def _compute_diversity(self, documents: List[Dict[str, Any]]) -> float:
        """Compute document source diversity."""
        if len(documents) < 2:
            return 0.5  # Default for single document
        
        # Extract sources
        sources = set()
        for doc in documents:
            source = doc.get('source') or doc.get('source_type') or doc.get('document_id', '')
            if source:
                sources.add(str(source))
        
        # Diversity = unique sources / total documents
        source_diversity = len(sources) / len(documents) if documents else 0.0
        
        # Also check content diversity (using simple word overlap)
        content_diversity = self._compute_content_diversity(documents)
        
        # Combined diversity
        return (source_diversity * 0.5 + content_diversity * 0.5)
    
    def _compute_content_diversity(self, documents: List[Dict[str, Any]]) -> float:
        """Compute content diversity using word overlap."""
        if len(documents) < 2:
            return 0.5
        
        # Get word sets for each document
        word_sets = []
        for doc in documents[:5]:
            content = doc.get('content', '').lower()
            words = set(re.findall(r'\b\w+\b', content))
            word_sets.append(words)
        
        if len(word_sets) < 2:
            return 0.5
        
        # Compute average pairwise dissimilarity
        dissimilarities = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                if word_sets[i] and word_sets[j]:
                    intersection = len(word_sets[i] & word_sets[j])
                    union = len(word_sets[i] | word_sets[j])
                    jaccard = intersection / union if union > 0 else 0
                    dissimilarities.append(1 - jaccard)  # Dissimilarity
        
        return sum(dissimilarities) / len(dissimilarities) if dissimilarities else 0.5
    
    def _compute_recency(self, documents: List[Dict[str, Any]]) -> float:
        """Compute document recency score."""
        if not documents:
            return 0.5
        
        current_time = time.time()
        recency_scores = []
        
        for doc in documents:
            # Try to get timestamp
            timestamp = doc.get('updated_at') or doc.get('created_at') or doc.get('timestamp')
            
            if timestamp:
                # Convert to seconds if needed
                if isinstance(timestamp, str):
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        timestamp = dt.timestamp()
                    except:
                        timestamp = None
                elif hasattr(timestamp, 'timestamp'):
                    timestamp = timestamp.timestamp()
                
                if timestamp:
                    # Recency: decay over 365 days
                    age_days = (current_time - timestamp) / (24 * 3600)
                    recency = max(0.0, 1.0 - age_days / 365.0)
                    recency_scores.append(recency)
                else:
                    recency_scores.append(0.5)  # Unknown
            else:
                recency_scores.append(0.5)  # Unknown
        
        return sum(recency_scores) / len(recency_scores) if recency_scores else 0.5
    
    def _compute_response_alignment(self, 
                                    response: str,
                                    documents: List[Dict[str, Any]]) -> float:
        """Compute how well response aligns with documents."""
        if not response or not documents:
            return 0.0
        
        response_words = set(re.findall(r'\b\w+\b', response.lower()))
        
        # Check alignment with each document
        alignments = []
        for doc in documents[:5]:
            content = doc.get('content', '').lower()
            doc_words = set(re.findall(r'\b\w+\b', content))
            
            if response_words and doc_words:
                overlap = len(response_words & doc_words)
                alignment = overlap / len(response_words)
                alignments.append(min(alignment, 1.0))
        
        return max(alignments) if alignments else 0.0
    
    def update_weights(self, 
                       relevance: float = None,
                       coverage: float = None,
                       diversity: float = None,
                       recency: float = None):
        """
        Update weights (for adaptive learning integration).
        
        Can be called by BASE OCO learner to adjust weights based on outcomes.
        """
        if relevance is not None:
            self.alpha_r = relevance
        if coverage is not None:
            self.alpha_c = coverage
        if diversity is not None:
            self.alpha_d = diversity
        if recency is not None:
            self.alpha_t = recency
        
        self._normalize_weights()

    # Learning Interface
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        if not hasattr(self, '_outcomes'): self._outcomes = []
        self._outcomes.append(outcome)
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        pass
    
    def get_statistics(self) -> Dict:
        return {'outcomes': len(getattr(self, '_outcomes', []))}
    
    def serialize_state(self) -> Dict:
        return {'outcomes': getattr(self, '_outcomes', [])[-100:]}
    
    def deserialize_state(self, state: Dict) -> None:
        self._outcomes = state.get('outcomes', [])


# Self-test
if __name__ == "__main__":
    pathway = RAGQualityPathway()
    
    query = "What are Tesla's recent financial results?"
    documents = [
        {"content": "Tesla reported Q4 2023 revenue of $25.17 billion, up 3% YoY.", "score": 0.92},
        {"content": "The company's full-year 2023 revenue reached $96.77 billion.", "score": 0.88},
        {"content": "Tesla's automotive gross margin was 18.2% in Q4 2023.", "score": 0.85}
    ]
    response = "Tesla's Q4 2023 revenue was $25.17 billion, with full-year revenue of $96.77 billion."
    
    result = pathway.score(query, documents, response)
    
    print(f"Overall Score: {result.overall_score:.3f}")
    print(f"  Relevance: {result.relevance_score:.3f} (weight: {result.weights['relevance']:.2f})")
    print(f"  Coverage: {result.coverage_score:.3f} (weight: {result.weights['coverage']:.2f})")
    print(f"  Diversity: {result.diversity_score:.3f} (weight: {result.weights['diversity']:.2f})")
    print(f"  Recency: {result.recency_score:.3f} (weight: {result.weights['recency']:.2f})")
    print(f"Response Alignment: {result.response_alignment:.3f}")
    print(f"Documents: {result.document_count}")
    print(f"Processing Time: {result.processing_time_ms:.1f}ms")






