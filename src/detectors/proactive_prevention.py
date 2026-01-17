"""
BAIS Proactive Hallucination Prevention
Migrated from Onyx Governance - Enhancement 5

Pre-emptive risk assessment BEFORE response generation.
Catches potential hallucinations early to prevent propagation.

Mathematical Formulation:
Allow if: E > threshold_E AND G > threshold_G
Where:
- E = Evidence sufficiency score
- G = Document grounding score

Patent Claims: Enhancement-5
Proprietary IP - 100% owned by Invitas Inc.
"""

import re
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class RiskLevel(str, Enum):
    """Risk level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskAssessment:
    """Risk assessment result."""
    risk_level: RiskLevel
    risk_score: float  # 0.0 (low) to 1.0 (critical)
    evidence_score: float
    grounding_score: float
    risk_factors: List[str]
    recommendations: List[str]
    allow_generation: bool
    require_human_review: bool
    constraints: List[str]  # Constraints to pass to LLM
    timestamp: float
    processing_time_ms: float = 0.0


class ProactiveHallucinationPrevention:
    """
    Proactive Hallucination Prevention for BAIS.
    
    Runs BEFORE response generation to:
    1. Assess query risk
    2. Check evidence sufficiency
    3. Validate document grounding
    4. Generate constraints for LLM
    
    Can trigger early exit for critical risk queries.
    """
    
    # High-risk query patterns
    HIGH_RISK_PATTERNS = [
        r'\b(?:medical|health|diagnosis|treatment|dosage|prescription)\b',
        r'\b(?:legal|law|lawsuit|court|attorney|liability)\b',
        r'\b(?:financial|investment|stock|trading|loan|mortgage)\b',
        r'\b(?:suicide|self-harm|violence|weapon)\b',
    ]
    
    # Insufficient evidence indicators
    INSUFFICIENT_EVIDENCE_INDICATORS = [
        'no documents', 'empty', 'not found', 'unavailable',
        'insufficient', 'limited', 'unclear'
    ]
    
    def __init__(self,
                 evidence_threshold: float = 0.50,
                 grounding_threshold: float = 0.40,
                 high_risk_multiplier: float = 1.5):
        """
        Initialize Proactive Prevention.
        
        Args:
            evidence_threshold: Minimum evidence score to allow generation
            grounding_threshold: Minimum grounding score to allow generation
            high_risk_multiplier: Multiplier for thresholds in high-risk domains
        """
        self.evidence_threshold = evidence_threshold
        self.grounding_threshold = grounding_threshold
        self.high_risk_multiplier = high_risk_multiplier
    
    def assess(self,
               query: str,
               documents: List[Dict[str, Any]],
               context: Dict[str, Any] = None) -> RiskAssessment:
        """
        Assess query risk BEFORE generation.
        
        Args:
            query: User query
            documents: Retrieved documents
            context: Additional context (domain, etc.)
        
        Returns:
            RiskAssessment with risk level and recommendations
        """
        start_time = time.time()
        context = context or {}
        
        risk_factors = []
        recommendations = []
        constraints = []
        
        # Detect high-risk patterns
        is_high_risk = self._detect_high_risk(query)
        if is_high_risk:
            risk_factors.append("Query contains high-risk domain terms")
        
        # Compute evidence sufficiency
        evidence_score = self._compute_evidence_score(query, documents)
        
        # Compute grounding score
        grounding_score = self._compute_grounding_score(query, documents)
        
        # Adjust thresholds for high-risk queries
        evidence_threshold = self.evidence_threshold
        grounding_threshold = self.grounding_threshold
        
        if is_high_risk:
            evidence_threshold *= self.high_risk_multiplier
            grounding_threshold *= self.high_risk_multiplier
            risk_factors.append(f"Elevated thresholds for high-risk query")
        
        # Check evidence sufficiency
        if evidence_score < evidence_threshold:
            risk_factors.append(f"Insufficient evidence ({evidence_score:.2f} < {evidence_threshold:.2f})")
            recommendations.append("Retrieve more relevant documents")
            constraints.append("Only make claims directly supported by provided documents")
        
        # Check grounding
        if grounding_score < grounding_threshold:
            risk_factors.append(f"Weak grounding ({grounding_score:.2f} < {grounding_threshold:.2f})")
            recommendations.append("Verify claims against documents")
            constraints.append("Acknowledge uncertainty when documents don't fully support claims")
        
        # Check document count
        if len(documents) == 0:
            risk_factors.append("No documents available")
            recommendations.append("Cannot generate without evidence")
            constraints.append("Do not make factual claims without evidence")
        elif len(documents) < 3:
            risk_factors.append(f"Limited documents ({len(documents)})")
            recommendations.append("Consider retrieving more documents")
        
        # Compute overall risk
        risk_score = self._compute_risk_score(
            evidence_score, grounding_score, 
            len(risk_factors), is_high_risk, len(documents)
        )
        
        risk_level = self._determine_risk_level(risk_score)
        
        # Determine if generation should be allowed
        allow_generation = risk_level != RiskLevel.CRITICAL
        require_human_review = risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        
        # Add risk-level specific recommendations
        if risk_level == RiskLevel.CRITICAL:
            recommendations.append("Query requires human review before response")
            constraints.append("Do not generate response - await human review")
        elif risk_level == RiskLevel.HIGH:
            recommendations.append("Response should be reviewed by human")
            constraints.append("Clearly state limitations and uncertainties")
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.append("Verify key facts before finalizing")
            constraints.append("Include appropriate caveats")
        
        processing_time = (time.time() - start_time) * 1000
        
        return RiskAssessment(
            risk_level=risk_level,
            risk_score=risk_score,
            evidence_score=evidence_score,
            grounding_score=grounding_score,
            risk_factors=risk_factors,
            recommendations=recommendations,
            allow_generation=allow_generation,
            require_human_review=require_human_review,
            constraints=constraints,
            timestamp=time.time(),
            processing_time_ms=processing_time
        )
    
    def _detect_high_risk(self, query: str) -> bool:
        """Detect if query is in a high-risk domain."""
        query_lower = query.lower()
        for pattern in self.HIGH_RISK_PATTERNS:
            if re.search(pattern, query_lower):
                return True
        return False
    
    def _compute_evidence_score(self,
                                 query: str,
                                 documents: List[Dict[str, Any]]) -> float:
        """
        Compute evidence sufficiency score.
        
        E = (1/|D|) * Σ relevance(doc_i) * coverage(doc_i)
        """
        if not documents:
            return 0.0
        
        query_terms = set(
            w.lower() for w in re.findall(r'\b\w+\b', query)
            if len(w) > 3
        )
        
        scores = []
        for doc in documents:
            content = doc.get('content', '').lower()
            doc_score = doc.get('score', 0.5)
            
            # Coverage: how many query terms appear in document
            if query_terms:
                covered = sum(1 for t in query_terms if t in content)
                coverage = covered / len(query_terms)
            else:
                coverage = 0.5
            
            # Combined score
            combined = doc_score * 0.6 + coverage * 0.4
            scores.append(combined)
        
        # Average with higher weight to top documents
        if len(scores) <= 2:
            return sum(scores) / len(scores) if scores else 0.0
        
        weights = [1.0 - 0.15 * i for i in range(len(scores))]
        weights = [max(w, 0.3) for w in weights]
        
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        return weighted_sum / sum(weights)
    
    def _compute_grounding_score(self,
                                  query: str,
                                  documents: List[Dict[str, Any]]) -> float:
        """
        Compute document grounding score.
        
        G = max(relevance(doc_i)) for all documents
        """
        if not documents:
            return 0.0
        
        max_score = 0.0
        query_lower = query.lower()
        
        for doc in documents:
            content = doc.get('content', '').lower()
            doc_score = doc.get('score', 0.5)
            
            # Check for query term presence
            query_words = set(query_lower.split())
            doc_words = set(content.split())
            
            if query_words:
                overlap = len(query_words & doc_words) / len(query_words)
            else:
                overlap = 0.5
            
            # Combined grounding
            grounding = doc_score * 0.5 + overlap * 0.5
            max_score = max(max_score, grounding)
        
        return min(max_score, 1.0)
    
    def _compute_risk_score(self,
                            evidence: float,
                            grounding: float,
                            num_factors: int,
                            is_high_risk: bool,
                            num_docs: int) -> float:
        """Compute overall risk score (0=low, 1=critical)."""
        # Base risk from evidence and grounding
        base_risk = 1.0 - (evidence * 0.5 + grounding * 0.5)
        
        # Penalty for risk factors
        factor_penalty = min(num_factors * 0.1, 0.3)
        
        # Penalty for high-risk domain
        domain_penalty = 0.2 if is_high_risk else 0.0
        
        # Penalty for few documents
        doc_penalty = 0.3 if num_docs == 0 else (0.1 if num_docs < 3 else 0.0)
        
        risk_score = base_risk + factor_penalty + domain_penalty + doc_penalty
        return min(risk_score, 1.0)
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from score."""
        if risk_score >= 0.8:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

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
    prevention = ProactiveHallucinationPrevention()
    
    # Test 1: Low risk
    result = prevention.assess(
        query="What is Python?",
        documents=[
            {"content": "Python is a programming language created by Guido van Rossum.", "score": 0.9},
            {"content": "Python is known for its simple syntax and readability.", "score": 0.85}
        ]
    )
    print(f"\nTest 1 (Low Risk):")
    print(f"  Risk Level: {result.risk_level.value}")
    print(f"  Risk Score: {result.risk_score:.2f}")
    print(f"  Allow Generation: {result.allow_generation}")
    
    # Test 2: High risk (medical)
    result = prevention.assess(
        query="What is the correct dosage of ibuprofen for pain?",
        documents=[
            {"content": "Ibuprofen is a nonsteroidal anti-inflammatory drug.", "score": 0.6}
        ]
    )
    print(f"\nTest 2 (Medical - High Risk):")
    print(f"  Risk Level: {result.risk_level.value}")
    print(f"  Risk Score: {result.risk_score:.2f}")
    print(f"  Risk Factors: {result.risk_factors}")
    print(f"  Require Human Review: {result.require_human_review}")
    
    # Test 3: Critical (no documents)
    result = prevention.assess(
        query="What are the legal implications of this contract?",
        documents=[]
    )
    print(f"\nTest 3 (Legal - No Docs):")
    print(f"  Risk Level: {result.risk_level.value}")
    print(f"  Risk Score: {result.risk_score:.2f}")
    print(f"  Allow Generation: {result.allow_generation}")
    print(f"  Constraints: {result.constraints}")






