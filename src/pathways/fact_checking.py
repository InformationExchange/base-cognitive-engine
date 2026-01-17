"""
BASE Fact-Checking Pathway
Migrated from Onyx Governance - Enhancement 2

Claim extraction and verification against source documents.

Patent Claims: Enhancement-2
Proprietary IP - 100% owned by Invitas Inc.
"""

import re
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Claim:
    """Extracted claim from response."""
    claim_id: str
    text: str
    claim_type: str  # factual, numerical, temporal, causal
    confidence: float
    start_pos: int = 0
    end_pos: int = 0


@dataclass
class ClaimVerification:
    """Verification result for a single claim."""
    claim: Claim
    verified: bool
    confidence: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    verification_method: str
    reason: Optional[str] = None


@dataclass
class FactCheckResult:
    """Overall fact-checking result."""
    verified: bool
    coverage: float  # verified_claims / total_claims
    verified_claims: int
    total_claims: int
    claim_verifications: List[ClaimVerification]
    overall_confidence: float
    unverified_claims: List[str]
    contradicted_claims: List[str]
    timestamp: float
    processing_time_ms: float = 0.0


class FactCheckingPathway:
    """
    Fact-Checking Governance Pathway for BASE.
    
    Implements:
    - Claim extraction from responses
    - Verification against source documents
    - Coverage computation
    - Contradiction detection
    
    Mathematical Formulation:
    Coverage = |verified_claims| / |total_claims|
    """
    
    # Claim patterns
    NUMERICAL_PATTERN = r'\b\d+(?:\.\d+)?(?:\s*(?:%|percent|million|billion|thousand))?\b'
    TEMPORAL_PATTERN = r'\b(?:19|20)\d{2}\b|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}'
    FACTUAL_PATTERN = r'[^.!?]*(?:is|are|was|were|has|have|had)\s+[^.!?]*[.!?]'
    DEFINITIVE_WORDS = ['always', 'never', 'all', 'none', 'every', 'must', 'definitely', 'certainly', 'proven', 'confirmed']
    
    def __init__(self,
                 verification_threshold: float = 0.70,
                 coverage_threshold: float = 0.60):
        """
        Initialize Fact-Checking Pathway.
        
        Args:
            verification_threshold: Confidence needed for claim verification
            coverage_threshold: Minimum coverage to consider verified
        """
        self.verification_threshold = verification_threshold
        self.coverage_threshold = coverage_threshold
    
    def verify(self,
               response: str,
               documents: List[Dict[str, Any]],
               context: Dict[str, Any] = None) -> FactCheckResult:
        """
        Verify facts in response against documents.
        
        Args:
            response: AI-generated response
            documents: Source documents
            context: Additional context (optional)
        
        Returns:
            FactCheckResult with verification results
        """
        start_time = time.time()
        
        # Extract claims
        claims = self._extract_claims(response)
        
        if not claims:
            return FactCheckResult(
                verified=True,
                coverage=1.0,
                verified_claims=0,
                total_claims=0,
                claim_verifications=[],
                overall_confidence=1.0,
                unverified_claims=[],
                contradicted_claims=[],
                timestamp=time.time(),
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Verify each claim
        verifications = []
        for claim in claims:
            verification = self._verify_claim(claim, documents)
            verifications.append(verification)
        
        # Compute metrics
        verified_count = sum(1 for v in verifications if v.verified)
        total_count = len(verifications)
        coverage = verified_count / total_count if total_count > 0 else 1.0
        
        overall_confidence = (
            sum(v.confidence for v in verifications) / total_count
            if total_count > 0 else 1.0
        )
        
        # Collect unverified and contradicted
        unverified = [v.claim.text[:100] for v in verifications 
                      if not v.verified and not v.contradicting_evidence]
        contradicted = [v.claim.text[:100] for v in verifications 
                        if v.contradicting_evidence]
        
        # Overall verification
        verified = coverage >= self.coverage_threshold and not contradicted
        
        processing_time = (time.time() - start_time) * 1000
        
        return FactCheckResult(
            verified=verified,
            coverage=coverage,
            verified_claims=verified_count,
            total_claims=total_count,
            claim_verifications=verifications,
            overall_confidence=overall_confidence,
            unverified_claims=unverified,
            contradicted_claims=contradicted,
            timestamp=time.time(),
            processing_time_ms=processing_time
        )
    
    def _extract_claims(self, response: str) -> List[Claim]:
        """Extract verifiable claims from response."""
        claims = []
        claim_id = 0
        
        # Numerical claims (statistics, percentages)
        for match in re.finditer(self.NUMERICAL_PATTERN, response):
            # Get surrounding context
            start = max(0, match.start() - 50)
            end = min(len(response), match.end() + 50)
            context = response[start:end]
            
            claim = Claim(
                claim_id=f"claim_{claim_id}",
                text=context.strip(),
                claim_type="numerical",
                confidence=0.8,
                start_pos=match.start(),
                end_pos=match.end()
            )
            claims.append(claim)
            claim_id += 1
        
        # Temporal claims (dates, years)
        for match in re.finditer(self.TEMPORAL_PATTERN, response, re.IGNORECASE):
            start = max(0, match.start() - 50)
            end = min(len(response), match.end() + 50)
            context = response[start:end]
            
            claim = Claim(
                claim_id=f"claim_{claim_id}",
                text=context.strip(),
                claim_type="temporal",
                confidence=0.85,
                start_pos=match.start(),
                end_pos=match.end()
            )
            claims.append(claim)
            claim_id += 1
        
        # Factual statements
        for match in re.finditer(self.FACTUAL_PATTERN, response):
            text = match.group().strip()
            if 20 < len(text) < 200:  # Reasonable length
                # Check for definitive language (higher priority)
                has_definitive = any(w in text.lower() for w in self.DEFINITIVE_WORDS)
                
                claim = Claim(
                    claim_id=f"claim_{claim_id}",
                    text=text,
                    claim_type="factual",
                    confidence=0.9 if has_definitive else 0.6,
                    start_pos=match.start(),
                    end_pos=match.end()
                )
                claims.append(claim)
                claim_id += 1
        
        # Deduplicate overlapping claims
        claims = self._deduplicate_claims(claims)
        
        return claims[:15]  # Limit to 15 claims
    
    def _deduplicate_claims(self, claims: List[Claim]) -> List[Claim]:
        """Remove overlapping claims, keeping highest confidence."""
        if not claims:
            return claims
        
        # Sort by confidence descending
        sorted_claims = sorted(claims, key=lambda c: c.confidence, reverse=True)
        
        kept = []
        used_ranges = []
        
        for claim in sorted_claims:
            # Check overlap with kept claims
            overlaps = False
            for start, end in used_ranges:
                if not (claim.end_pos < start or claim.start_pos > end):
                    overlaps = True
                    break
            
            if not overlaps:
                kept.append(claim)
                used_ranges.append((claim.start_pos, claim.end_pos))
        
        return kept
    
    def _verify_claim(self, 
                      claim: Claim,
                      documents: List[Dict[str, Any]]) -> ClaimVerification:
        """Verify a single claim against documents."""
        if not documents:
            return ClaimVerification(
                claim=claim,
                verified=False,
                confidence=0.0,
                supporting_evidence=[],
                contradicting_evidence=[],
                verification_method="no_documents",
                reason="No documents available for verification"
            )
        
        supporting = []
        contradicting = []
        
        claim_lower = claim.text.lower()
        claim_keywords = self._extract_keywords(claim.text)
        
        for i, doc in enumerate(documents[:10]):  # Check top 10 docs
            content = doc.get('content', '')
            if not content:
                continue
            
            content_lower = content.lower()
            
            # Check for support
            if self._text_supports(claim_lower, content_lower, claim_keywords):
                supporting.append(f"doc_{i}: {content[:100]}...")
            
            # Check for contradiction
            elif self._text_contradicts(claim_lower, content_lower, claim_keywords):
                contradicting.append(f"doc_{i}: {content[:100]}...")
        
        # Verification decision
        verified = len(supporting) > 0 and len(contradicting) == 0
        
        # Confidence based on evidence
        if verified:
            confidence = min(len(supporting) / 3.0, 1.0) * claim.confidence
        elif contradicting:
            confidence = 0.0
        else:
            confidence = 0.3  # No evidence either way
        
        reason = None
        if not verified:
            if contradicting:
                reason = f"Found {len(contradicting)} contradicting sources"
            else:
                reason = "No supporting evidence found"
        
        return ClaimVerification(
            claim=claim,
            verified=verified,
            confidence=confidence,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            verification_method="document_search",
            reason=reason
        )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key terms from text."""
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'has', 'have', 'had',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
            'be', 'this', 'that', 'it', 'and', 'or', 'but', 'if'
        }
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w not in stopwords and len(w) > 3][:10]
    
    def _text_supports(self, 
                       claim: str, 
                       document: str,
                       keywords: List[str]) -> bool:
        """Check if document supports claim."""
        # Check keyword overlap
        keyword_hits = sum(1 for kw in keywords if kw in document)
        keyword_ratio = keyword_hits / len(keywords) if keywords else 0
        
        # Check for direct text match
        claim_words = set(claim.split())
        doc_words = set(document.split())
        word_overlap = len(claim_words & doc_words) / len(claim_words) if claim_words else 0
        
        return keyword_ratio > 0.5 or word_overlap > 0.4
    
    def _text_contradicts(self,
                          claim: str,
                          document: str,
                          keywords: List[str]) -> bool:
        """Check if document contradicts claim."""
        negation_words = ['not', 'no', 'never', 'none', 'neither', 'cannot', 
                          "doesn't", "don't", "isn't", "aren't", "wasn't", "weren't",
                          'false', 'incorrect', 'wrong', 'untrue']
        
        # Check if document has negation near keywords
        for kw in keywords[:5]:
            if kw in document:
                # Find keyword position
                pos = document.find(kw)
                context = document[max(0, pos-30):pos+30]
                
                # Check for negation in context
                for neg in negation_words:
                    if neg in context:
                        return True
        
        return False

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
    pathway = FactCheckingPathway()
    
    response = "Tesla was founded in 2003 and has over 100,000 employees. The company's revenue in 2023 was $96.77 billion."
    documents = [
        {"content": "Tesla, Inc. was incorporated in July 2003 by Martin Eberhard and Marc Tarpenning."},
        {"content": "Tesla reported full-year 2023 revenue of $96.77 billion. The company employs approximately 140,000 people globally."}
    ]
    
    result = pathway.verify(response, documents)
    
    print(f"Verified: {result.verified}")
    print(f"Coverage: {result.coverage:.2%}")
    print(f"Claims: {result.verified_claims}/{result.total_claims}")
    print(f"Unverified: {result.unverified_claims}")
    print(f"Contradicted: {result.contradicted_claims}")
    print(f"Processing Time: {result.processing_time_ms:.1f}ms")






