"""
BASE Hybrid Proof Validator
===========================

Phase 17: Integrates pattern-based evidence demand with LLM contextual validation.

This module bridges:
1. EvidenceDemandLoop (pattern-based claim extraction)
2. LLMProofValidator (context-aware validation)
3. MultiTrackChallenger (multi-LLM verification)

Patent Alignment:
- NOVEL-3+: Enhanced Proof-Based Verification
- GAP-1+: Enhanced Evidence Demand
- NOVEL-22: LLM Challenger
- NOVEL-23: Multi-Track Challenger
- PPA1-Inv20: Human-Machine Hybrid Arbitration

Brain Layer: Layer 6 (Evidence/Basal Ganglia)

Key Insight: Pattern-based detection is FAST but lacks context.
LLM-based validation is CONTEXTUAL but slower.
Hybrid approach: Pattern first, LLM for ambiguous cases.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import asyncio

# Import from local modules
try:
    from .evidence_demand import (
        EvidenceDemandLoop, ExtractedClaim, VerificationResult,
        VerificationStatus, ClaimType
    )
    from .llm_proof_validator import (
        LLMProofValidator, ProofValidationResult, ClaimContext,
        ProofRequirement
    )
except ImportError:
    # Handle relative import issues
    from evidence_demand import (
        EvidenceDemandLoop, ExtractedClaim, VerificationResult,
        VerificationStatus, ClaimType
    )
    from llm_proof_validator import (
        LLMProofValidator, ProofValidationResult, ClaimContext,
        ProofRequirement
    )


class ValidationMode(Enum):
    """How to validate claims."""
    PATTERN_ONLY = "pattern"      # Fast, deterministic
    LLM_ONLY = "llm"              # Contextual, slower
    HYBRID = "hybrid"             # Pattern first, LLM for uncertain


@dataclass
class HybridValidationResult:
    """Combined result from pattern and LLM validation."""
    claim: ExtractedClaim
    
    # Pattern-based results
    pattern_result: Optional[VerificationResult]
    pattern_confidence: float
    
    # LLM-based results
    llm_result: Optional[ProofValidationResult]
    llm_used: bool
    llm_reason: str
    
    # Combined decision
    final_status: VerificationStatus
    final_confidence: float
    should_flag: bool
    recommendations: List[str]
    
    # Learning data
    mode_used: ValidationMode
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "claim_id": self.claim.claim_id,
            "claim_text": self.claim.claim_text[:100],
            "final_status": self.final_status.value,
            "final_confidence": self.final_confidence,
            "should_flag": self.should_flag,
            "mode_used": self.mode_used.value,
            "llm_used": self.llm_used,
            "recommendations": self.recommendations[:3]
        }


@dataclass
class HybridLearningRecord:
    """Record for learning which mode works best."""
    claim_type: ClaimType
    context: ClaimContext
    mode_used: ValidationMode
    was_correct: bool
    pattern_confidence: float
    llm_confidence: float
    domain: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class HybridProofValidator:
    """
    Hybrid validation combining pattern-based and LLM-based approaches.
    
    Strategy:
    1. Pattern-based extraction: Fast, catch obvious issues
    2. Confidence assessment: Is pattern result clear?
    3. LLM validation: Only for uncertain cases (saves API calls)
    4. Combined decision: Weight pattern + LLM based on context
    5. Learning: Track which mode works for which claim types
    """
    
    # Thresholds for triggering LLM validation
    HIGH_CONFIDENCE_THRESHOLD = 0.85   # Pattern result is clear
    LOW_CONFIDENCE_THRESHOLD = 0.40    # Pattern result is uncertain
    
    # Claims that ALWAYS need LLM validation
    HIGH_STAKES_CLAIMS = [
        ClaimType.TEST_RESULT,      # "All tests pass"
        ClaimType.QUALITY,          # "Production ready"
    ]
    
    def __init__(
        self,
        evidence_demand: EvidenceDemandLoop = None,
        llm_validator: LLMProofValidator = None,
        default_mode: ValidationMode = ValidationMode.HYBRID
    ):
        self.evidence_demand = evidence_demand or EvidenceDemandLoop()
        self.llm_validator = llm_validator or LLMProofValidator()
        self.default_mode = default_mode
        
        # Learning storage
        self.learning_records: List[HybridLearningRecord] = []
        self.mode_effectiveness: Dict[str, Dict[str, float]] = {}
    
    async def validate_claims(
        self,
        response: str,
        evidence: List[str] = None,
        context_text: str = "",
        mode: ValidationMode = None,
        domain: str = "general"
    ) -> List[HybridValidationResult]:
        """
        Validate all claims in a response using hybrid approach.
        
        Args:
            response: The LLM response to validate
            evidence: List of evidence items provided
            context_text: Additional context for the claims
            mode: Validation mode (default: HYBRID)
            domain: Domain context (medical, legal, etc.)
        
        Returns:
            List of HybridValidationResult for each claim found
        """
        mode = mode or self.default_mode
        evidence = evidence or []
        results = []
        
        # Step 1: Pattern-based claim extraction
        claims = self.evidence_demand.extract_claims(response)
        
        for claim in claims:
            result = await self._validate_single_claim(
                claim, evidence, context_text, response, mode, domain
            )
            results.append(result)
        
        return results
    
    async def _validate_single_claim(
        self,
        claim: ExtractedClaim,
        evidence: List[str],
        context_text: str,
        full_response: str,
        mode: ValidationMode,
        domain: str
    ) -> HybridValidationResult:
        """Validate a single claim using appropriate mode."""
        
        # Step 1: Pattern-based verification (always run)
        pattern_result = self.evidence_demand.verify_claim(claim, evidence)
        pattern_confidence = self._calculate_pattern_confidence(pattern_result)
        
        # Step 2: Determine if LLM validation is needed
        llm_result = None
        llm_used = False
        llm_reason = ""
        
        if mode == ValidationMode.LLM_ONLY:
            llm_used = True
            llm_reason = "LLM_ONLY mode"
        elif mode == ValidationMode.HYBRID:
            # Decision logic for hybrid mode
            needs_llm, reason = self._needs_llm_validation(
                claim, pattern_result, pattern_confidence
            )
            if needs_llm:
                llm_used = True
                llm_reason = reason
        
        # Step 3: Run LLM validation if needed
        if llm_used:
            llm_result = await self.llm_validator.validate_claim(
                claim=claim.claim_text,
                evidence=evidence,
                surrounding_text=context_text or full_response[:500],
                use_llm=True,
                domain=domain
            )
        
        # Step 4: Combine results
        final_status, final_confidence, should_flag, recommendations = \
            self._combine_results(
                claim, pattern_result, pattern_confidence,
                llm_result, mode
            )
        
        return HybridValidationResult(
            claim=claim,
            pattern_result=pattern_result,
            pattern_confidence=pattern_confidence,
            llm_result=llm_result,
            llm_used=llm_used,
            llm_reason=llm_reason,
            final_status=final_status,
            final_confidence=final_confidence,
            should_flag=should_flag,
            recommendations=recommendations,
            mode_used=mode
        )
    
    def _calculate_pattern_confidence(
        self, result: VerificationResult
    ) -> float:
        """Calculate confidence in pattern-based result."""
        if result.status == VerificationStatus.VERIFIED:
            return 0.9 if result.evidence_count >= 3 else 0.7
        elif result.status == VerificationStatus.PARTIALLY_VERIFIED:
            return 0.5
        elif result.status == VerificationStatus.CONTRADICTED:
            return 0.85  # High confidence in contradiction
        elif result.status == VerificationStatus.UNVERIFIED:
            return 0.6  # Moderate confidence - no evidence found
        else:
            return 0.3  # Low confidence
    
    def _needs_llm_validation(
        self,
        claim: ExtractedClaim,
        pattern_result: VerificationResult,
        pattern_confidence: float
    ) -> Tuple[bool, str]:
        """Determine if LLM validation is needed for this claim."""
        
        # High-stakes claims always need LLM
        if claim.claim_type in self.HIGH_STAKES_CLAIMS:
            return True, f"High-stakes claim type: {claim.claim_type.value}"
        
        # Low pattern confidence needs LLM
        if pattern_confidence < self.LOW_CONFIDENCE_THRESHOLD:
            return True, f"Low pattern confidence: {pattern_confidence:.2f}"
        
        # Partial verification is ambiguous
        if pattern_result.status == VerificationStatus.PARTIALLY_VERIFIED:
            return True, "Partial verification - context needed"
        
        # Clear cases don't need LLM
        if pattern_confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            return False, f"High pattern confidence: {pattern_confidence:.2f}"
        
        # Default: use LLM for uncertain middle ground
        if self.LOW_CONFIDENCE_THRESHOLD <= pattern_confidence < self.HIGH_CONFIDENCE_THRESHOLD:
            return True, f"Uncertain confidence: {pattern_confidence:.2f}"
        
        return False, "Pattern result sufficient"
    
    def _combine_results(
        self,
        claim: ExtractedClaim,
        pattern_result: VerificationResult,
        pattern_confidence: float,
        llm_result: Optional[ProofValidationResult],
        mode: ValidationMode
    ) -> Tuple[VerificationStatus, float, bool, List[str]]:
        """
        Combine pattern and LLM results into final decision.
        
        Returns (status, confidence, should_flag, recommendations)
        """
        recommendations = []
        
        if not llm_result:
            # Pattern-only decision
            return (
                pattern_result.status,
                pattern_confidence,
                pattern_result.status in [
                    VerificationStatus.UNVERIFIED,
                    VerificationStatus.CONTRADICTED
                ],
                pattern_result.recommendations if hasattr(pattern_result, 'recommendations') else []
            )
        
        # Hybrid decision: weight pattern and LLM results
        pattern_weight = 0.4  # Pattern is fast but less contextual
        llm_weight = 0.6      # LLM understands context better
        
        # Map statuses to numeric scores
        pattern_score = {
            VerificationStatus.VERIFIED: 1.0,
            VerificationStatus.PARTIALLY_VERIFIED: 0.5,
            VerificationStatus.PENDING: 0.3,
            VerificationStatus.UNVERIFIED: 0.1,
            VerificationStatus.CONTRADICTED: 0.0
        }.get(pattern_result.status, 0.5)
        
        llm_score = llm_result.sufficiency_score if llm_result.is_sufficient else 0.3
        
        # Combined score
        combined_score = (
            pattern_score * pattern_weight * pattern_confidence +
            llm_score * llm_weight * llm_result.context_confidence
        )
        
        # Determine final status
        if combined_score >= 0.7:
            final_status = VerificationStatus.VERIFIED
        elif combined_score >= 0.4:
            final_status = VerificationStatus.PARTIALLY_VERIFIED
        else:
            final_status = VerificationStatus.UNVERIFIED
        
        # Should flag?
        should_flag = llm_result.should_flag or combined_score < 0.4
        
        # Combine recommendations
        if llm_result.missing_evidence:
            recommendations.extend(llm_result.missing_evidence)
        if hasattr(pattern_result, 'recommendations'):
            recommendations.extend(pattern_result.recommendations)
        
        # Add context-specific recommendations
        if llm_result.context == ClaimContext.PLANNING and not llm_result.should_flag:
            recommendations.append("Planning context - estimates acceptable")
        elif llm_result.context == ClaimContext.AUDIT and not llm_result.is_sufficient:
            recommendations.append("Audit context requires verified evidence")
        
        return (final_status, combined_score, should_flag, recommendations)
    
    def record_outcome(
        self,
        result: HybridValidationResult,
        was_correct: bool,
        domain: str = "general"
    ):
        """Record validation outcome for learning."""
        record = HybridLearningRecord(
            claim_type=result.claim.claim_type,
            context=result.llm_result.context if result.llm_result else ClaimContext.GENERAL,
            mode_used=result.mode_used,
            was_correct=was_correct,
            pattern_confidence=result.pattern_confidence,
            llm_confidence=result.llm_result.context_confidence if result.llm_result else 0.0,
            domain=domain
        )
        self.learning_records.append(record)
        
        # Update mode effectiveness
        mode_key = result.mode_used.value
        claim_key = result.claim.claim_type.value
        
        if mode_key not in self.mode_effectiveness:
            self.mode_effectiveness[mode_key] = {}
        
        if claim_key not in self.mode_effectiveness[mode_key]:
            self.mode_effectiveness[mode_key][claim_key] = {"correct": 0, "total": 0}
        
        self.mode_effectiveness[mode_key][claim_key]["total"] += 1
        if was_correct:
            self.mode_effectiveness[mode_key][claim_key]["correct"] += 1
        
        # Also record to sub-validators for their learning
        if result.llm_result:
            self.llm_validator.record_outcome(
                result.llm_result,
                was_correct,
                result.llm_result.context if was_correct else None
            )
    
    def get_mode_recommendations(self, claim_type: ClaimType) -> ValidationMode:
        """Get recommended validation mode for a claim type based on learning."""
        claim_key = claim_type.value
        
        best_mode = self.default_mode
        best_accuracy = 0.0
        
        for mode_key, claims in self.mode_effectiveness.items():
            if claim_key in claims:
                stats = claims[claim_key]
                accuracy = stats["correct"] / max(stats["total"], 1)
                if accuracy > best_accuracy and stats["total"] >= 5:
                    best_accuracy = accuracy
                    best_mode = ValidationMode(mode_key)
        
        return best_mode
    
    def validate(
        self,
        claim: str,
        evidence: List[str] = None,
        domain: str = "general"
    ) -> Dict[str, Any]:
        """
        Simple synchronous validation interface.
        
        Args:
            claim: The claim text to validate
            evidence: List of evidence items
            domain: Domain context
        
        Returns:
            Dict with validation results
        """
        import asyncio
        evidence = evidence or []
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(
                self.validate_claims(
                    response=claim,
                    evidence=evidence,
                    domain=domain
                )
            )
            
            if results:
                r = results[0]
                return {
                    "valid": r.final_status == VerificationStatus.VERIFIED,
                    "confidence": r.final_confidence,
                    "status": r.final_status.value,
                    "should_flag": r.should_flag,
                    "recommendations": r.recommendations,
                    "mode_used": r.mode_used.value
                }
            else:
                # No claims extracted - assume valid
                return {
                    "valid": True,
                    "confidence": 0.5,
                    "status": "no_claims",
                    "should_flag": False,
                    "recommendations": [],
                    "mode_used": "pattern"
                }
        except Exception as e:
            return {
                "valid": False,
                "confidence": 0.0,
                "status": "error",
                "should_flag": True,
                "recommendations": [f"Validation error: {str(e)}"],
                "mode_used": "error"
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total_validations = len(self.learning_records)
        llm_used_count = sum(1 for r in self.learning_records if r.llm_confidence > 0)
        correct_count = sum(1 for r in self.learning_records if r.was_correct)
        
        return {
            "total_validations": total_validations,
            "llm_used_count": llm_used_count,
            "llm_usage_rate": llm_used_count / max(total_validations, 1),
            "correct_count": correct_count,
            "accuracy": correct_count / max(total_validations, 1),
            "mode_effectiveness": self.mode_effectiveness,
            "api_calls_saved": total_validations - llm_used_count
        }

    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard wrapper for non-standard record_outcome."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback."""
        pass  # Implement specific learning logic


    def serialize_state(self) -> Dict:
        """Serialize state."""
        return {'outcomes': getattr(self, '_outcomes', [])[-100:]}


    def deserialize_state(self, state: Dict) -> None:
        """Restore state."""
        self._outcomes = state.get('outcomes', [])


# Module-level instance
hybrid_proof_validator = HybridProofValidator()

