"""
BASE LLM-Derived Proof Validator
================================

Phase 17: Context-aware proof validation using LLMs.

Instead of pattern-matching "100%" claims, this module:
1. Uses LLM to understand the CONTEXT of the claim
2. Determines what proof is APPROPRIATE for that context
3. Validates whether provided evidence satisfies the requirement
4. LEARNS from outcomes to improve future validation

Patent Alignment:
- NOVEL-3: Evidence Demand (enhanced with LLM context)
- GAP-1: Evidence Demand Loop (context-aware)
- PPA1-Inv20: Human-Machine Hybrid Arbitration
- NOVEL-22: LLM Challenger
- NOVEL-28: Intelligent Dimension Selection

Brain Layer: Layer 6 (Evidence/Basal Ganglia)

Key Insight: "100% complete" in a PLANNING doc should be handled 
differently than "100% complete" in a FINAL REPORT.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json
import os
import asyncio
import httpx


class ClaimContext(Enum):
    """Context types for claims - determines required proof level."""
    PLANNING = "planning"           # Future-oriented, estimates acceptable
    PROGRESS_UPDATE = "progress"    # Intermediate, directional claims OK
    FINAL_REPORT = "final"          # Requires concrete evidence
    TECHNICAL_DOC = "technical"     # Requires verifiable details
    AUDIT = "audit"                 # Highest standard - enumerated proof
    GENERAL = "general"             # Default context


class ProofRequirement(Enum):
    """What type of proof is required for the context."""
    NONE = "none"                   # Claim doesn't need proof (opinion)
    DIRECTIONAL = "directional"    # Trends/improvements OK without exact numbers
    ENUMERATED = "enumerated"      # Must list items to support count claims
    CITED = "cited"                # Must have citations/sources
    EXECUTABLE = "executable"      # Must be runnable/testable
    VERIFIED = "verified"          # Must have external verification


@dataclass
class ProofValidationResult:
    """Result of LLM-based proof validation."""
    claim: str
    context: ClaimContext
    required_proof: ProofRequirement
    provided_evidence: List[str]
    
    # LLM assessment
    is_sufficient: bool
    sufficiency_score: float      # 0.0 - 1.0
    llm_reasoning: str
    missing_evidence: List[str]
    
    # Learning data
    context_confidence: float     # How confident was context detection
    should_flag: bool             # Final decision: flag or accept
    recommendation: str
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    llm_model: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "claim": self.claim[:100],
            "context": self.context.value,
            "required_proof": self.required_proof.value,
            "is_sufficient": self.is_sufficient,
            "sufficiency_score": self.sufficiency_score,
            "should_flag": self.should_flag,
            "missing_evidence": self.missing_evidence[:3],
            "recommendation": self.recommendation
        }


@dataclass
class ContextLearning:
    """Learning record for context detection improvement."""
    claim_pattern: str
    detected_context: ClaimContext
    actual_context: Optional[ClaimContext]  # From feedback
    was_correct: bool
    domain: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class LLMProofValidator:
    """
    Use LLM to determine context-appropriate proof requirements.
    
    This replaces rigid pattern matching with contextual understanding:
    - Planning doc with "100% complete roadmap" → OK (it's a plan)
    - Final report with "100% complete" → Needs enumeration
    - Audit doc with "100% complete" → Needs detailed proof
    """
    
    # Context indicators for heuristic pre-check
    CONTEXT_INDICATORS = {
        ClaimContext.PLANNING: [
            'roadmap', 'plan', 'timeline', 'phase', 'milestone', 'estimate',
            'proposed', 'will be', 'expected', 'target', 'goal'
        ],
        ClaimContext.PROGRESS_UPDATE: [
            'update', 'progress', 'status', 'current', 'ongoing', 'in progress'
        ],
        ClaimContext.FINAL_REPORT: [
            'complete', 'finished', 'done', 'delivered', 'production-ready',
            'final', 'shipped', 'released'
        ],
        ClaimContext.TECHNICAL_DOC: [
            'implementation', 'architecture', 'module', 'class', 'function',
            'api', 'endpoint', 'schema'
        ],
        ClaimContext.AUDIT: [
            'audit', 'verification', 'compliance', 'certified', 'validated',
            'verified', 'tested', 'evidence'
        ]
    }
    
    # Proof requirements by context
    PROOF_REQUIREMENTS = {
        ClaimContext.PLANNING: ProofRequirement.NONE,
        ClaimContext.PROGRESS_UPDATE: ProofRequirement.DIRECTIONAL,
        ClaimContext.FINAL_REPORT: ProofRequirement.ENUMERATED,
        ClaimContext.TECHNICAL_DOC: ProofRequirement.EXECUTABLE,
        ClaimContext.AUDIT: ProofRequirement.VERIFIED,
        ClaimContext.GENERAL: ProofRequirement.DIRECTIONAL,
    }
    
    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or "learning_data/proof_validation"
        self.learnings: List[ContextLearning] = []
        self.context_accuracy: Dict[str, float] = {}  # Per-context accuracy
        self._llm_client = None
        self._load_learnings()
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard wrapper for non-standard record_outcome."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def _load_learnings(self):
        """Load previous learnings from disk."""
        try:
            path = f"{self.storage_path}/context_learnings.json"
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                    self.context_accuracy = data.get("accuracy", {})
        except Exception:
            pass
    
    def _save_learnings(self):
        """Save learnings to disk."""
        try:
            os.makedirs(self.storage_path, exist_ok=True)
            path = f"{self.storage_path}/context_learnings.json"
            with open(path, "w") as f:
                json.dump({
                    "accuracy": self.context_accuracy,
                    "total_validations": len(self.learnings)
                }, f, indent=2)
        except Exception:
            pass
    
    def _heuristic_context(self, text: str, claim: str) -> Tuple[ClaimContext, float]:
        """
        Fast heuristic context detection (no LLM call).
        Returns (context, confidence).
        """
        text_lower = text.lower()
        claim_lower = claim.lower()
        
        scores = {}
        for context, indicators in self.CONTEXT_INDICATORS.items():
            score = sum(1 for ind in indicators if ind in text_lower or ind in claim_lower)
            scores[context] = score
        
        if not scores or max(scores.values()) == 0:
            return ClaimContext.GENERAL, 0.5
        
        best_context = max(scores, key=scores.get)
        best_score = scores[best_context]
        total_indicators = len(self.CONTEXT_INDICATORS[best_context])
        confidence = min(best_score / max(total_indicators * 0.3, 1), 1.0)
        
        return best_context, confidence
    
    async def _llm_context_validation(
        self, 
        claim: str, 
        text: str, 
        heuristic_context: ClaimContext
    ) -> Tuple[ClaimContext, float, str]:
        """
        Use LLM to validate/refine context detection.
        Returns (context, confidence, reasoning).
        """
        api_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
        if not api_key:
            # Fallback to heuristic if no API key
            return heuristic_context, 0.7, "Heuristic-only (no LLM API key)"
        
        prompt = f"""Analyze this claim and its surrounding text to determine the CONTEXT.

CLAIM: "{claim}"

SURROUNDING TEXT: "{text[:500]}"

CONTEXT OPTIONS:
1. PLANNING - Future-oriented, estimates, roadmaps (proof: not required)
2. PROGRESS_UPDATE - Intermediate status, ongoing work (proof: directional trends OK)
3. FINAL_REPORT - Completed work claims (proof: enumerated list required)
4. TECHNICAL_DOC - Implementation details (proof: must be verifiable)
5. AUDIT - Compliance/verification claims (proof: documented evidence required)
6. GENERAL - Default context

HEURISTIC DETECTED: {heuristic_context.value}

Respond in JSON format:
{{
  "context": "one of the options above",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "agrees_with_heuristic": true/false
}}
"""
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": "grok-2-1212",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    # Parse JSON from response
                    import re
                    json_match = re.search(r'\{[^{}]+\}', content, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group())
                        context_str = parsed.get("context", "GENERAL").upper()
                        context = ClaimContext[context_str] if context_str in ClaimContext.__members__ else heuristic_context
                        confidence = float(parsed.get("confidence", 0.7))
                        reasoning = parsed.get("reasoning", "LLM analysis")
                        return context, confidence, reasoning
        except Exception as e:
            pass
        
        return heuristic_context, 0.6, f"LLM validation failed, using heuristic"
    
    async def validate_claim(
        self,
        claim: str,
        evidence: List[str],
        surrounding_text: str = "",
        use_llm: bool = True,
        domain: str = "general"
    ) -> ProofValidationResult:
        """
        Validate a claim with context-aware proof requirements.
        
        Args:
            claim: The claim to validate (e.g., "100% complete")
            evidence: List of evidence items provided
            surrounding_text: Context around the claim
            use_llm: Whether to use LLM for context validation
            domain: Domain context (medical, legal, etc.)
        
        Returns:
            ProofValidationResult with context-aware assessment
        """
        # Step 1: Detect context (heuristic first)
        heuristic_context, heuristic_conf = self._heuristic_context(
            surrounding_text or claim, claim
        )
        
        # Step 2: LLM validation if enabled and confidence is low
        if use_llm and heuristic_conf < 0.8:
            context, confidence, reasoning = await self._llm_context_validation(
                claim, surrounding_text or claim, heuristic_context
            )
        else:
            context, confidence, reasoning = heuristic_context, heuristic_conf, "Heuristic detection"
        
        # Step 3: Determine proof requirement for context
        required_proof = self.PROOF_REQUIREMENTS.get(context, ProofRequirement.DIRECTIONAL)
        
        # Step 4: Validate evidence against requirement
        is_sufficient, sufficiency_score, missing = self._validate_evidence(
            claim, evidence, required_proof
        )
        
        # Step 5: Make final decision
        should_flag = not is_sufficient and context not in [ClaimContext.PLANNING]
        
        # Generate recommendation
        if is_sufficient:
            recommendation = "Claim has appropriate evidence for context"
        elif context == ClaimContext.PLANNING:
            recommendation = "Planning context - estimates acceptable without enumeration"
        else:
            recommendation = f"Missing {required_proof.value} proof: {', '.join(missing[:3])}"
        
        result = ProofValidationResult(
            claim=claim,
            context=context,
            required_proof=required_proof,
            provided_evidence=evidence,
            is_sufficient=is_sufficient,
            sufficiency_score=sufficiency_score,
            llm_reasoning=reasoning,
            missing_evidence=missing,
            context_confidence=confidence,
            should_flag=should_flag,
            recommendation=recommendation,
            llm_model="grok-2-1212" if use_llm else "heuristic"
        )
        
        return result
    
    def _validate_evidence(
        self,
        claim: str,
        evidence: List[str],
        required: ProofRequirement
    ) -> Tuple[bool, float, List[str]]:
        """
        Validate evidence against proof requirement.
        Returns (is_sufficient, score, missing_items).
        """
        missing = []
        
        if required == ProofRequirement.NONE:
            return True, 1.0, []
        
        if required == ProofRequirement.DIRECTIONAL:
            # Just need some evidence of direction
            if evidence and len(evidence) >= 1:
                return True, 0.8, []
            missing.append("At least one supporting point")
            return False, 0.3, missing
        
        if required == ProofRequirement.ENUMERATED:
            # Extract numbers from claim
            import re
            numbers = re.findall(r'\d+', claim)
            if numbers:
                claimed_count = int(numbers[0])
                if len(evidence) >= claimed_count * 0.5:  # 50% threshold
                    return True, len(evidence) / claimed_count, []
                missing.append(f"Enumerated list (claimed {claimed_count}, provided {len(evidence)})")
                return False, len(evidence) / max(claimed_count, 1), missing
            # No number claim - just need some enumeration
            if len(evidence) >= 3:
                return True, 0.7, []
            missing.append("Enumerated list of items")
            return False, 0.4, missing
        
        if required == ProofRequirement.CITED:
            # Look for citation-like patterns in evidence
            citation_patterns = ['source:', 'ref:', 'see:', 'http', '.md', '.py']
            has_citations = any(
                any(p in e.lower() for p in citation_patterns)
                for e in evidence
            )
            if has_citations:
                return True, 0.8, []
            missing.append("Source citations")
            return False, 0.3, missing
        
        if required == ProofRequirement.EXECUTABLE:
            # Look for code/file references
            code_patterns = ['.py', '.js', '.ts', 'def ', 'class ', 'function']
            has_code = any(
                any(p in e for p in code_patterns)
                for e in evidence
            )
            if has_code:
                return True, 0.7, []
            missing.append("Executable code or file references")
            return False, 0.3, missing
        
        if required == ProofRequirement.VERIFIED:
            # Highest standard - need multiple types of evidence
            checks = [
                len(evidence) >= 3,
                any('verified' in e.lower() or 'tested' in e.lower() for e in evidence),
            ]
            score = sum(checks) / len(checks)
            if score >= 0.5:
                return True, score, []
            missing.append("Verified evidence (test results, audit trail)")
            return False, score, missing
        
        return True, 0.5, []
    
    def record_outcome(
        self,
        result: ProofValidationResult,
        was_correct: bool,
        actual_context: ClaimContext = None
    ):
        """
        Record outcome for learning.
        
        Args:
            result: The validation result
            was_correct: Whether the validation was correct
            actual_context: The actual context (if different from detected)
        """
        learning = ContextLearning(
            claim_pattern=result.claim[:50],
            detected_context=result.context,
            actual_context=actual_context or result.context,
            was_correct=was_correct,
            domain="general"
        )
        self.learnings.append(learning)
        
        # Update accuracy tracking
        context_key = result.context.value
        if context_key not in self.context_accuracy:
            self.context_accuracy[context_key] = {"correct": 0, "total": 0}
        
        self.context_accuracy[context_key]["total"] += 1
        if was_correct:
            self.context_accuracy[context_key]["correct"] += 1
        
        self._save_learnings()
    
    def get_context_accuracy(self) -> Dict[str, float]:
        """Get accuracy rates per context type."""
        return {
            ctx: stats["correct"] / max(stats["total"], 1)
            for ctx, stats in self.context_accuracy.items()
        }

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


# Module-level instance
llm_proof_validator = LLMProofValidator()

