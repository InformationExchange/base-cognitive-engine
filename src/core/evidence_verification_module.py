"""
BASE Evidence Verification Module (NOVEL-53)

Actually VERIFIES evidence is real, not just checks format.
Uses multi-track AI for independent verification.

Key Capabilities:
1. Multi-LLM verification of claims/citations
2. Automatic track suggestion based on complexity/importance
3. User override for track count and LLM selection
4. Latest model checking and auto-update
5. Consensus-based verification decisions

Brain Layer: 8 (Verification & Validation)
Patent Alignment: Novel invention extending BASE governance
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum
from datetime import datetime
import logging
import asyncio

logger = logging.getLogger(__name__)


class VerificationType(Enum):
    """Types of verification requests."""
    CITATION = "citation"           # Verify a citation/reference exists
    CODE = "code"                   # Verify code segment is correct
    MEDICAL = "medical"             # Verify medical claim/diagnosis
    LEGAL = "legal"                 # Verify legal precedent/statute
    FACTUAL = "factual"             # Verify factual claim
    CALCULATION = "calculation"     # Verify numerical calculation
    EXISTENCE = "existence"         # Verify something exists (file, API, etc.)


class VerificationComplexity(Enum):
    """Complexity levels for track suggestion."""
    TRIVIAL = 1         # Simple fact check - 1 track sufficient
    LOW = 2             # Basic verification - 2 tracks
    MEDIUM = 3          # Moderate complexity - 3 tracks
    HIGH = 4            # Complex/important - 4+ tracks
    CRITICAL = 5        # Life/safety critical - maximum tracks


class VerificationResult(Enum):
    """Outcome of verification."""
    VERIFIED = "verified"           # All tracks agree it's valid
    LIKELY_VALID = "likely_valid"   # Majority agree valid
    UNCERTAIN = "uncertain"         # No consensus
    LIKELY_FAKE = "likely_fake"     # Majority agree fake/wrong
    FAKE = "fake"                   # All tracks agree it's fake/wrong
    ERROR = "error"                 # Verification failed


@dataclass
class VerificationRequest:
    """Request for verification."""
    content: str                        # The content to verify
    verification_type: VerificationType # What kind of verification
    context: str = ""                   # Additional context
    domain: str = "general"             # Industry domain
    importance: str = "normal"          # low/normal/high/critical
    user_track_override: Optional[int] = None  # User can specify track count
    user_llm_override: Optional[List[str]] = None  # User can specify LLMs


@dataclass
class TrackResult:
    """Result from a single verification track."""
    llm_name: str
    model_version: str
    verdict: str                    # "valid", "invalid", "uncertain"
    confidence: float
    reasoning: str
    evidence_found: List[str]       # What evidence the LLM found
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class VerificationResponse:
    """Complete verification response."""
    request: VerificationRequest
    result: VerificationResult
    confidence: float
    tracks_used: int
    track_results: List[TrackResult]
    consensus_reasoning: str
    recommendations: List[str]
    verification_time_ms: float


class EvidenceVerificationModule:
    """
    NOVEL-53: Actually verify evidence using multi-track AI.
    
    Not just format checking - sends to multiple LLMs for independent
    verification and uses consensus to determine validity.
    
    Integrates with:
    - MultiTrackChallenger (existing)
    - MultiTrackOrchestrator (existing)
    - LLMRegistry (existing)
    - SemanticModeSelector (existing)
    """
    
    # Default track counts by complexity
    DEFAULT_TRACKS = {
        VerificationComplexity.TRIVIAL: 1,
        VerificationComplexity.LOW: 2,
        VerificationComplexity.MEDIUM: 3,
        VerificationComplexity.HIGH: 4,
        VerificationComplexity.CRITICAL: 5
    }
    
    # Importance to complexity mapping
    IMPORTANCE_COMPLEXITY = {
        "low": VerificationComplexity.TRIVIAL,
        "normal": VerificationComplexity.LOW,
        "high": VerificationComplexity.HIGH,
        "critical": VerificationComplexity.CRITICAL
    }
    
    # Domain-specific complexity boosts
    DOMAIN_COMPLEXITY_BOOST = {
        "healthcare": 2,    # Always add 2 tracks for medical
        "legal": 1,         # Add 1 track for legal
        "financial": 1,     # Add 1 track for financial
        "vibe_coding": 0,   # Code can be tested directly
        "general": 0
    }
    
    def __init__(
        self,
        llm_registry: Any = None,
        multi_track_challenger: Any = None,
        semantic_selector: Any = None,
        max_tracks: int = 5,
        auto_update_models: bool = True
    ):
        self.llm_registry = llm_registry
        self.multi_track_challenger = multi_track_challenger
        self.semantic_selector = semantic_selector
        self.max_tracks = max_tracks
        self.auto_update_models = auto_update_models
        
        # Learning state
        self._outcomes: List[Dict] = []
        self._verification_history: List[VerificationResponse] = []
        self._domain_accuracy: Dict[str, List[bool]] = {}
        
        # Available LLMs (will be populated from registry)
        self._available_llms: List[Dict] = []
        self._last_model_check: Optional[datetime] = None
        
        logger.info("[EvidenceVerification] Module initialized")
        
        # Check for latest models on init
        if auto_update_models:
            self._refresh_available_models()
    
    def _refresh_available_models(self) -> None:
        """
        Check LLMRegistry for latest available models.
        Only uses the most advanced models from each provider.
        """
        if self.llm_registry is None:
            # Default fallback
            self._available_llms = [
                {"provider": "grok", "model": "grok-3", "tier": "advanced"},
                {"provider": "openai", "model": "gpt-4o", "tier": "advanced"},
                {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "tier": "advanced"},
                {"provider": "google", "model": "gemini-2.0-flash", "tier": "advanced"},
            ]
            return
        
        try:
            # Get models from registry
            all_models = self.llm_registry.get_available_models()
            
            # Filter to only advanced/latest models per provider
            advanced_models = []
            providers_seen = set()
            
            for model in all_models:
                provider = model.get("provider", "unknown")
                if provider not in providers_seen:
                    # Take first (assumed best) model per provider
                    advanced_models.append(model)
                    providers_seen.add(provider)
            
            self._available_llms = advanced_models
            self._last_model_check = datetime.now()
            
            logger.info(f"[EvidenceVerification] Refreshed models: {len(advanced_models)} providers")
            
        except Exception as e:
            logger.warning(f"[EvidenceVerification] Model refresh failed: {e}")
    
    def suggest_tracks(
        self,
        request: VerificationRequest
    ) -> Tuple[int, List[str], str]:
        """
        Suggest number of tracks and which LLMs based on:
        - Verification type
        - Domain
        - Importance
        - Semantic complexity
        
        Returns: (track_count, llm_list, reasoning)
        """
        # Start with base complexity from importance
        base_complexity = self.IMPORTANCE_COMPLEXITY.get(
            request.importance, 
            VerificationComplexity.LOW
        )
        
        # Add domain-specific boost
        domain_boost = self.DOMAIN_COMPLEXITY_BOOST.get(request.domain, 0)
        
        # Calculate tracks
        base_tracks = self.DEFAULT_TRACKS[base_complexity]
        total_tracks = min(base_tracks + domain_boost, self.max_tracks)
        
        # Use semantic selector if available for additional analysis
        if self.semantic_selector:
            try:
                semantic_analysis = self.semantic_selector.analyze(request.content)
                if semantic_analysis.get("high_stakes", False):
                    total_tracks = min(total_tracks + 1, self.max_tracks)
            except:
                pass
        
        # Build reasoning
        reasoning = f"Base: {base_tracks} tracks ({request.importance} importance)"
        if domain_boost:
            reasoning += f" + {domain_boost} for {request.domain} domain"
        reasoning += f" = {total_tracks} tracks"
        
        # Select LLMs (diverse providers)
        llm_list = self._select_diverse_llms(total_tracks)
        
        return total_tracks, llm_list, reasoning
    
    def _select_diverse_llms(self, count: int) -> List[str]:
        """Select diverse LLMs from different providers."""
        if not self._available_llms:
            self._refresh_available_models()
        
        selected = []
        for llm in self._available_llms[:count]:
            selected.append(f"{llm['provider']}/{llm['model']}")
        
        return selected
    
    async def verify(
        self,
        request: VerificationRequest
    ) -> VerificationResponse:
        """
        Actually verify evidence using multi-track AI.
        
        1. Determine track count (auto or user override)
        2. Select LLMs (auto or user override)
        3. Query each LLM independently
        4. Build consensus
        5. Return verification result
        """
        import time
        start_time = time.time()
        
        # Determine tracks
        if request.user_track_override:
            track_count = min(request.user_track_override, self.max_tracks)
            _, auto_llms, _ = self.suggest_tracks(request)
            llms = auto_llms[:track_count]
            reasoning = f"User override: {track_count} tracks"
        else:
            track_count, llms, reasoning = self.suggest_tracks(request)
        
        # User can also override LLMs
        if request.user_llm_override:
            llms = request.user_llm_override[:track_count]
        
        # Query each LLM
        track_results = await self._query_tracks(request, llms)
        
        # Build consensus
        result, confidence, consensus_reasoning = self._build_consensus(track_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(result, track_results)
        
        # Build response
        response = VerificationResponse(
            request=request,
            result=result,
            confidence=confidence,
            tracks_used=len(track_results),
            track_results=track_results,
            consensus_reasoning=consensus_reasoning,
            recommendations=recommendations,
            verification_time_ms=(time.time() - start_time) * 1000
        )
        
        # Record for learning
        self._verification_history.append(response)
        
        return response
    
    async def _query_tracks(
        self,
        request: VerificationRequest,
        llms: List[str]
    ) -> List[TrackResult]:
        """Query each LLM track for verification."""
        results = []
        
        # Build verification prompt based on type
        prompt = self._build_verification_prompt(request)
        
        for llm_spec in llms:
            try:
                # Parse provider/model
                parts = llm_spec.split("/")
                provider = parts[0] if len(parts) > 0 else "unknown"
                model = parts[1] if len(parts) > 1 else llm_spec
                
                # Query LLM (use multi_track_challenger if available)
                if self.multi_track_challenger:
                    response = await self._query_via_challenger(provider, model, prompt)
                else:
                    response = await self._query_direct(provider, model, prompt)
                
                # Parse response
                verdict, confidence, reasoning, evidence = self._parse_llm_response(response)
                
                results.append(TrackResult(
                    llm_name=provider,
                    model_version=model,
                    verdict=verdict,
                    confidence=confidence,
                    reasoning=reasoning,
                    evidence_found=evidence
                ))
                
            except Exception as e:
                logger.warning(f"[EvidenceVerification] Track {llm_spec} failed: {e}")
                results.append(TrackResult(
                    llm_name=llm_spec,
                    model_version="unknown",
                    verdict="error",
                    confidence=0.0,
                    reasoning=f"Query failed: {str(e)}",
                    evidence_found=[]
                ))
        
        return results
    
    def _build_verification_prompt(self, request: VerificationRequest) -> str:
        """Build prompt for verification based on type."""
        type_instructions = {
            VerificationType.CITATION: """
Verify if this citation/reference is REAL and ACCURATE:
- Does this source actually exist?
- Is the quoted content accurate?
- Are the details (author, date, page) correct?
""",
            VerificationType.CODE: """
Verify if this code segment is CORRECT:
- Does it compile/run without errors?
- Does it do what it claims to do?
- Are there any bugs or issues?
""",
            VerificationType.MEDICAL: """
Verify this medical claim/diagnosis:
- Is this medically accurate?
- Does it align with clinical guidelines?
- Are there any safety concerns?
WARNING: This is for verification only, not medical advice.
""",
            VerificationType.LEGAL: """
Verify this legal claim/precedent:
- Does this case/statute actually exist?
- Is the interpretation correct?
- Is it applicable in the stated jurisdiction?
""",
            VerificationType.FACTUAL: """
Verify this factual claim:
- Is this statement accurate?
- What evidence supports or contradicts it?
- Rate your confidence in the claim.
""",
            VerificationType.CALCULATION: """
Verify this calculation:
- Is the math correct?
- Are the formulas applied correctly?
- What is the correct result?
""",
            VerificationType.EXISTENCE: """
Verify this exists:
- Can you confirm this exists?
- What evidence shows it exists or doesn't?
- If it doesn't exist, what might have been intended?
"""
        }
        
        instructions = type_instructions.get(
            request.verification_type, 
            type_instructions[VerificationType.FACTUAL]
        )
        
        prompt = f"""
VERIFICATION REQUEST
====================
Type: {request.verification_type.value}
Domain: {request.domain}

CONTENT TO VERIFY:
{request.content}

{f"ADDITIONAL CONTEXT: {request.context}" if request.context else ""}

INSTRUCTIONS:
{instructions}

Respond in this format:
VERDICT: [valid/invalid/uncertain]
CONFIDENCE: [0.0-1.0]
REASONING: [Your detailed reasoning]
EVIDENCE: [List any evidence you found]
"""
        return prompt
    
    async def _query_via_challenger(
        self,
        provider: str,
        model: str,
        prompt: str
    ) -> str:
        """Query via MultiTrackChallenger."""
        # Use existing challenger infrastructure
        try:
            result = await self.multi_track_challenger.challenge_single(
                provider=provider,
                model=model,
                prompt=prompt
            )
            return result.get("response", "")
        except:
            return await self._query_direct(provider, model, prompt)
    
    async def _query_direct(
        self,
        provider: str,
        model: str,
        prompt: str
    ) -> str:
        """Direct query to LLM (fallback)."""
        # This would use the LLM registry for actual calls
        # For now, return a simulated response structure
        return f"""
VERDICT: uncertain
CONFIDENCE: 0.5
REASONING: Unable to verify - no direct LLM connection available
EVIDENCE: []
"""
    
    def _parse_llm_response(
        self,
        response: str
    ) -> Tuple[str, float, str, List[str]]:
        """Parse LLM response into structured data."""
        import re
        
        # Extract verdict
        verdict_match = re.search(r'VERDICT:\s*(\w+)', response, re.IGNORECASE)
        verdict = verdict_match.group(1).lower() if verdict_match else "uncertain"
        
        # Extract confidence
        conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response, re.IGNORECASE)
        confidence = float(conf_match.group(1)) if conf_match else 0.5
        
        # Extract reasoning
        reason_match = re.search(r'REASONING:\s*(.+?)(?=EVIDENCE:|$)', response, re.IGNORECASE | re.DOTALL)
        reasoning = reason_match.group(1).strip() if reason_match else "No reasoning provided"
        
        # Extract evidence
        evidence_match = re.search(r'EVIDENCE:\s*(.+?)$', response, re.IGNORECASE | re.DOTALL)
        evidence_text = evidence_match.group(1).strip() if evidence_match else ""
        evidence = [e.strip() for e in evidence_text.split('\n') if e.strip() and e.strip() != '[]']
        
        return verdict, confidence, reasoning, evidence
    
    def _build_consensus(
        self,
        results: List[TrackResult]
    ) -> Tuple[VerificationResult, float, str]:
        """Build consensus from track results."""
        if not results:
            return VerificationResult.ERROR, 0.0, "No tracks returned results"
        
        # Count verdicts
        valid_count = sum(1 for r in results if r.verdict == "valid")
        invalid_count = sum(1 for r in results if r.verdict == "invalid")
        uncertain_count = sum(1 for r in results if r.verdict == "uncertain")
        error_count = sum(1 for r in results if r.verdict == "error")
        
        total = len(results)
        working_total = total - error_count
        
        if working_total == 0:
            return VerificationResult.ERROR, 0.0, "All tracks failed"
        
        # Calculate confidence (average of non-error results)
        avg_confidence = sum(r.confidence for r in results if r.verdict != "error") / working_total
        
        # Determine result
        valid_ratio = valid_count / working_total
        invalid_ratio = invalid_count / working_total
        
        if valid_ratio == 1.0:
            result = VerificationResult.VERIFIED
            reasoning = f"All {valid_count} tracks agree: VALID"
        elif valid_ratio >= 0.75:
            result = VerificationResult.LIKELY_VALID
            reasoning = f"{valid_count}/{working_total} tracks say valid"
        elif invalid_ratio == 1.0:
            result = VerificationResult.FAKE
            reasoning = f"All {invalid_count} tracks agree: INVALID/FAKE"
        elif invalid_ratio >= 0.75:
            result = VerificationResult.LIKELY_FAKE
            reasoning = f"{invalid_count}/{working_total} tracks say invalid"
        else:
            result = VerificationResult.UNCERTAIN
            reasoning = f"No consensus: {valid_count} valid, {invalid_count} invalid, {uncertain_count} uncertain"
        
        return result, avg_confidence, reasoning
    
    def _generate_recommendations(
        self,
        result: VerificationResult,
        track_results: List[TrackResult]
    ) -> List[str]:
        """Generate recommendations based on verification result."""
        recommendations = []
        
        if result == VerificationResult.VERIFIED:
            recommendations.append("Evidence appears valid - proceed with confidence")
        
        elif result == VerificationResult.LIKELY_VALID:
            recommendations.append("Most LLMs agree it's valid, but verify independently for critical uses")
        
        elif result == VerificationResult.UNCERTAIN:
            recommendations.append("No consensus reached - recommend human review")
            recommendations.append("Consider adding more verification tracks")
        
        elif result == VerificationResult.LIKELY_FAKE:
            recommendations.append("Most LLMs flagged this as invalid - investigate before using")
            recommendations.append("Check original sources manually")
        
        elif result == VerificationResult.FAKE:
            recommendations.append("All LLMs agree this is invalid/fake - DO NOT USE")
            recommendations.append("Request corrected information from the source")
        
        # Add track-specific insights
        for track in track_results:
            if track.evidence_found:
                recommendations.append(f"{track.llm_name} found: {track.evidence_found[0][:50]}...")
        
        return recommendations
    
    # ===== Convenience Methods =====
    
    async def verify_citation(self, citation: str, context: str = "") -> VerificationResponse:
        """Convenience: Verify a citation."""
        return await self.verify(VerificationRequest(
            content=citation,
            verification_type=VerificationType.CITATION,
            context=context
        ))
    
    async def verify_code(self, code: str, language: str = "") -> VerificationResponse:
        """Convenience: Verify code segment."""
        return await self.verify(VerificationRequest(
            content=code,
            verification_type=VerificationType.CODE,
            context=f"Language: {language}" if language else "",
            domain="vibe_coding"
        ))
    
    async def verify_medical(self, claim: str, context: str = "") -> VerificationResponse:
        """Convenience: Verify medical claim."""
        return await self.verify(VerificationRequest(
            content=claim,
            verification_type=VerificationType.MEDICAL,
            context=context,
            domain="healthcare",
            importance="critical"  # Medical is always critical
        ))
    
    async def second_opinion(
        self,
        content: str,
        verification_type: VerificationType,
        original_source: str = "",
        num_tracks: int = 3
    ) -> VerificationResponse:
        """Get a second opinion on any content."""
        return await self.verify(VerificationRequest(
            content=content,
            verification_type=verification_type,
            context=f"Original source: {original_source}" if original_source else "",
            user_track_override=num_tracks
        ))
    
    # ===== Learning Interface =====
    
    def record_outcome(self, outcome: Dict) -> None:
        """Record verification outcome for learning."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)
        
        # Track domain accuracy
        domain = outcome.get('domain', 'general')
        was_correct = outcome.get('was_correct', True)
        
        if domain not in self._domain_accuracy:
            self._domain_accuracy[domain] = []
        self._domain_accuracy[domain].append(was_correct)
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback to improve verification."""
        # Adjust track suggestions based on feedback
        domain = feedback.get('domain', 'general')
        tracks_were_sufficient = feedback.get('tracks_sufficient', True)
        
        if not tracks_were_sufficient:
            # Boost tracks for this domain
            current = self.DOMAIN_COMPLEXITY_BOOST.get(domain, 0)
            self.DOMAIN_COMPLEXITY_BOOST[domain] = min(current + 1, 3)
    
    def get_statistics(self) -> Dict:
        """Return verification statistics."""
        return {
            'total_verifications': len(self._verification_history),
            'outcomes_recorded': len(getattr(self, '_outcomes', [])),
            'domain_accuracy': {
                d: sum(a)/len(a) if a else 0 
                for d, a in self._domain_accuracy.items()
            },
            'available_llms': len(self._available_llms)
        }
    
    def serialize_state(self) -> Dict:
        """Serialize state for persistence."""
        return {
            'outcomes': getattr(self, '_outcomes', [])[-100:],
            'domain_boosts': self.DOMAIN_COMPLEXITY_BOOST,
            'last_model_check': self._last_model_check.isoformat() if self._last_model_check else None
        }
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore state from serialized data."""
        self._outcomes = state.get('outcomes', [])
        if state.get('domain_boosts'):
            self.DOMAIN_COMPLEXITY_BOOST.update(state['domain_boosts'])


# =============================================================================
# FACTORY
# =============================================================================

def create_verification_module(
    llm_registry: Any = None,
    multi_track: Any = None,
    semantic_selector: Any = None
) -> EvidenceVerificationModule:
    """Create a verification module with optional integrations."""
    return EvidenceVerificationModule(
        llm_registry=llm_registry,
        multi_track_challenger=multi_track,
        semantic_selector=semantic_selector
    )


if __name__ == "__main__":
    import asyncio
    
    print("=" * 70)
    print("EVIDENCE VERIFICATION MODULE DEMO")
    print("=" * 70)
    
    module = EvidenceVerificationModule()
    
    # Test track suggestion
    print("\n[TEST 1] Track Suggestion")
    request = VerificationRequest(
        content="According to Smith et al. (2023), the treatment is 95% effective",
        verification_type=VerificationType.CITATION,
        domain="healthcare",
        importance="critical"
    )
    
    tracks, llms, reasoning = module.suggest_tracks(request)
    print(f"  Suggested: {tracks} tracks")
    print(f"  LLMs: {llms}")
    print(f"  Reasoning: {reasoning}")
    
    # Test with user override
    print("\n[TEST 2] User Override")
    request2 = VerificationRequest(
        content="SELECT * FROM users WHERE id = 1",
        verification_type=VerificationType.CODE,
        domain="vibe_coding",
        importance="normal",
        user_track_override=2,
        user_llm_override=["openai/gpt-4o", "anthropic/claude-sonnet-4-20250514"]
    )
    
    tracks2, llms2, reasoning2 = module.suggest_tracks(request2)
    print(f"  Auto suggest: {tracks2} tracks, {llms2}")
    print(f"  User override: 2 tracks, {request2.user_llm_override}")
    
    print("\n" + "=" * 70)
    print("MODULE READY")
    print("=" * 70)

