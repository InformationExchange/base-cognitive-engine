"""
BASE Multi-Track LLM Challenger - Parallel Analysis Across Multiple LLMs

Enables A/B/C/...N track analysis where multiple LLMs challenge a response
in parallel, then results are aggregated for consensus decision.

Key Features:
1. Selectable LLMs per track (Grok, Claude, GPT, Gemini, Mistral, etc.)
2. Parallel execution for speed
3. Consensus aggregation (majority vote, weighted average, unanimous)
4. Track-specific prompts and challenge types
5. Learning which tracks perform best for which domains

Patent Alignment:
- NOVEL-8: Cross-LLM Governance Orchestration
- NOVEL-19: LLM Registry (Multi-Provider Management)
- PPA1-Inv20: Human-Machine Hybrid Arbitration (Bayesian consensus)
- PPA1-Inv23: AI Common Sense via Multi-Source Triangulation
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from pathlib import Path
import json
import asyncio
import hashlib
import aiohttp


class LLMProvider(Enum):
    """Supported LLM providers for challenger tracks."""
    VERTEX = "vertex"       # Google Vertex AI (PRIMARY)
    GROK = "grok"           # xAI Grok
    CLAUDE = "claude"       # Anthropic Claude
    GPT4 = "gpt4"           # OpenAI GPT-4
    GPT35 = "gpt35"         # OpenAI GPT-3.5
    GEMINI = "gemini"       # Google Gemini (public API)
    MISTRAL = "mistral"     # Mistral AI
    LLAMA = "llama"         # Meta Llama (via API)
    COHERE = "cohere"       # Cohere
    LOCAL = "local"         # Local model (Ollama, etc.)


class ConsensusMethod(Enum):
    """Methods for aggregating multi-track results."""
    MAJORITY_VOTE = "majority"      # Most common decision wins
    WEIGHTED_AVERAGE = "weighted"   # Weighted by track confidence
    UNANIMOUS = "unanimous"         # All must agree
    STRICTEST = "strictest"         # Most conservative decision
    BAYESIAN = "bayesian"           # Bayesian consensus (PPA1-Inv20)


class OutputMode(Enum):
    """
    Multi-track output modes - determines how results are presented to users.
    
    CONSENSUS: Traditional mode - aggregate tracks into single decision
    PARALLEL_PATHS: Present each track as an alternate path for user selection
    AUDIT_TRAIL: Post-evaluation mode - all tracks shown as audit record
    USER_SELECTIVE: Allow users to pick elements from different tracks
    """
    CONSENSUS = "consensus"           # Single aggregated decision (default)
    PARALLEL_PATHS = "parallel"       # Each track as alternate execution path
    AUDIT_TRAIL = "audit"             # All tracks as audit record
    USER_SELECTIVE = "selective"      # User picks elements from tracks


@dataclass
class TrackConfig:
    """Configuration for a single challenger track."""
    track_id: str
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    weight: float = 1.0                    # Weight in consensus
    enabled: bool = True
    max_tokens: int = 2000
    temperature: float = 0.3               # Lower = more deterministic
    timeout_seconds: float = 30.0
    challenge_types: List[str] = field(default_factory=list)
    custom_system_prompt: Optional[str] = None
    
    def __post_init__(self):
        if not self.challenge_types:
            # Default includes citation_verification for factual domains
            self.challenge_types = ["evidence_demand", "devils_advocate", "completeness", "citation_verification"]


@dataclass
class SelectableElement:
    """
    An element that users can select from track results.
    Enables cherry-picking of specific findings, guidance, or decisions.
    
    Patent Alignment:
    - PPA1-Inv20: Human-Machine Hybrid Arbitration
    - NOVEL-23: Multi-Track Challenger with user selection
    """
    element_id: str
    track_id: str
    provider: LLMProvider
    element_type: str              # "issue", "gap", "guidance", "decision", "risk"
    content: str
    confidence: float
    selected: bool = False         # User selection state
    
    def to_dict(self) -> Dict:
        return {
            "id": self.element_id,
            "track": self.track_id,
            "provider": self.provider.value,
            "type": self.element_type,
            "content": self.content,
            "confidence": self.confidence,
            "selected": self.selected
        }


@dataclass
class TrackResult:
    """Result from a single challenger track."""
    track_id: str
    provider: LLMProvider
    model_name: str
    
    # Analysis results
    issues_found: List[str]
    evidence_gaps: List[str]
    risk_factors: List[str]
    confidence_score: float        # 0.0-1.0
    
    # Decision
    should_accept: bool
    should_remediate: bool
    remediation_guidance: List[str]
    
    # Metadata
    execution_time_ms: float
    raw_response: str
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "track_id": self.track_id,
            "provider": self.provider.value,
            "model": self.model_name,
            "issues": len(self.issues_found),
            "evidence_gaps": len(self.evidence_gaps),
            "confidence": self.confidence_score,
            "accept": self.should_accept,
            "time_ms": self.execution_time_ms
        }
    
    def to_selectable_elements(self) -> List[SelectableElement]:
        """Convert track result to selectable elements for USER_SELECTIVE mode."""
        elements = []
        
        # Issues as selectable elements
        for i, issue in enumerate(self.issues_found):
            elements.append(SelectableElement(
                element_id=f"{self.track_id}_issue_{i}",
                track_id=self.track_id,
                provider=self.provider,
                element_type="issue",
                content=issue,
                confidence=self.confidence_score
            ))
        
        # Evidence gaps
        for i, gap in enumerate(self.evidence_gaps):
            elements.append(SelectableElement(
                element_id=f"{self.track_id}_gap_{i}",
                track_id=self.track_id,
                provider=self.provider,
                element_type="gap",
                content=gap,
                confidence=self.confidence_score
            ))
        
        # Guidance
        for i, guidance in enumerate(self.remediation_guidance):
            elements.append(SelectableElement(
                element_id=f"{self.track_id}_guidance_{i}",
                track_id=self.track_id,
                provider=self.provider,
                element_type="guidance",
                content=guidance,
                confidence=self.confidence_score
            ))
        
        # Risk factors
        for i, risk in enumerate(self.risk_factors):
            elements.append(SelectableElement(
                element_id=f"{self.track_id}_risk_{i}",
                track_id=self.track_id,
                provider=self.provider,
                element_type="risk",
                content=risk,
                confidence=self.confidence_score
            ))
        
        # Decision as element
        elements.append(SelectableElement(
            element_id=f"{self.track_id}_decision",
            track_id=self.track_id,
            provider=self.provider,
            element_type="decision",
            content=f"{'ACCEPT' if self.should_accept else 'REJECT'}: {self.raw_response[:200]}...",
            confidence=self.confidence_score
        ))
        
        return elements


@dataclass
class MultiTrackVerdict:
    """
    Aggregated verdict from all tracks with multiple output modes.
    
    Output Modes:
    - CONSENSUS: Single aggregated decision (traditional)
    - PARALLEL_PATHS: Each track as alternate execution path
    - AUDIT_TRAIL: All tracks shown as audit record
    - USER_SELECTIVE: User picks elements from tracks
    """
    # Input
    claim: str
    domain: str
    
    # Track results
    track_results: List[TrackResult]
    tracks_executed: int
    tracks_succeeded: int
    tracks_failed: int
    
    # Consensus (for CONSENSUS mode)
    consensus_method: ConsensusMethod
    consensus_accept: bool
    consensus_confidence: float
    consensus_remediate: bool
    
    # Aggregated findings
    all_issues: List[str]
    all_evidence_gaps: List[str]
    unified_guidance: List[str]
    
    # Disagreements (valuable for learning)
    track_disagreements: List[Dict]
    
    # Performance
    total_time_ms: float
    
    # Inventions
    inventions_used: List[str]
    
    # Output mode control (default: CONSENSUS for backward compatibility)
    output_mode: OutputMode = OutputMode.CONSENSUS
    
    # Selectable elements for USER_SELECTIVE mode
    selectable_elements: List[SelectableElement] = field(default_factory=list)
    
    # User selections (populated after user interaction)
    user_selections: List[str] = field(default_factory=list)
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        return f"""
MULTI-TRACK VERDICT
==================
Tracks: {self.tracks_succeeded}/{self.tracks_executed} succeeded
Mode: {self.output_mode.value}
Consensus ({self.consensus_method.value}): {'ACCEPT' if self.consensus_accept else 'REJECT'}
Confidence: {self.consensus_confidence:.2f}
Issues found: {len(self.all_issues)}
Evidence gaps: {len(self.all_evidence_gaps)}
Remediation needed: {self.consensus_remediate}
Time: {self.total_time_ms:.0f}ms
        """.strip()
    
    def get_parallel_paths(self) -> List[Dict]:
        """
        Get each track as an alternate execution path.
        Used in PARALLEL_PATHS mode for users to select entire paths.
        """
        paths = []
        for result in self.track_results:
            if result.error is None:
                paths.append({
                    "track_id": result.track_id,
                    "provider": result.provider.value,
                    "model": result.model_name,
                    "decision": "ACCEPT" if result.should_accept else "REJECT",
                    "confidence": result.confidence_score,
                    "issues": result.issues_found,
                    "evidence_gaps": result.evidence_gaps,
                    "guidance": result.remediation_guidance,
                    "risks": result.risk_factors,
                    "raw_analysis": result.raw_response
                })
        return paths
    
    def get_audit_trail(self) -> Dict:
        """
        Get complete audit trail for post-evaluation review.
        Used in AUDIT_TRAIL mode for compliance and analysis.
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "claim_analyzed": self.claim,
            "domain": self.domain,
            "tracks": [
                {
                    "track_id": r.track_id,
                    "provider": r.provider.value,
                    "model": r.model_name,
                    "execution_time_ms": r.execution_time_ms,
                    "decision": "ACCEPT" if r.should_accept else "REJECT",
                    "confidence": r.confidence_score,
                    "issues": r.issues_found,
                    "evidence_gaps": r.evidence_gaps,
                    "guidance": r.remediation_guidance,
                    "error": r.error
                }
                for r in self.track_results
            ],
            "consensus": {
                "method": self.consensus_method.value,
                "accept": self.consensus_accept,
                "confidence": self.consensus_confidence,
                "remediate": self.consensus_remediate
            },
            "disagreements": self.track_disagreements,
            "inventions_used": self.inventions_used
        }
    
    def get_selectable_elements_by_type(self, element_type: str = None) -> List[SelectableElement]:
        """
        Get selectable elements, optionally filtered by type.
        Used in USER_SELECTIVE mode.
        
        Args:
            element_type: Filter by "issue", "gap", "guidance", "decision", "risk"
        """
        if element_type:
            return [e for e in self.selectable_elements if e.element_type == element_type]
        return self.selectable_elements
    
    def select_elements(self, element_ids: List[str]) -> Dict:
        """
        User selects specific elements from different tracks.
        Returns a composite result from selected elements.
        """
        selected = []
        for element in self.selectable_elements:
            if element.element_id in element_ids:
                element.selected = True
                selected.append(element)
        
        self.user_selections = element_ids
        
        # Build composite from selections
        composite = {
            "selected_issues": [e.content for e in selected if e.element_type == "issue"],
            "selected_gaps": [e.content for e in selected if e.element_type == "gap"],
            "selected_guidance": [e.content for e in selected if e.element_type == "guidance"],
            "selected_risks": [e.content for e in selected if e.element_type == "risk"],
            "selected_decisions": [e.content for e in selected if e.element_type == "decision"],
            "sources": list(set(e.track_id for e in selected)),
            "average_confidence": sum(e.confidence for e in selected) / len(selected) if selected else 0.0
        }
        
        return composite
    
    def select_entire_path(self, track_id: str) -> Dict:
        """
        User selects an entire track's path.
        Returns that track's complete result.
        """
        for result in self.track_results:
            if result.track_id == track_id:
                return {
                    "track_id": track_id,
                    "provider": result.provider.value,
                    "decision": "ACCEPT" if result.should_accept else "REJECT",
                    "issues": result.issues_found,
                    "evidence_gaps": result.evidence_gaps,
                    "guidance": result.remediation_guidance,
                    "confidence": result.confidence_score,
                    "analysis": result.raw_response
                }
        return {"error": f"Track {track_id} not found"}

    # Learning Interface
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        if not hasattr(self, '_outcomes'): self._outcomes = []
        getattr(self, '_outcomes', {}).append(outcome)
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        pass
    
    def get_statistics(self) -> Dict:
        if not hasattr(self, '_total_challenges'): self._total_challenges = 0
        return {'outcomes': len(getattr(self, '_outcomes', []))}
    
    def serialize_state(self) -> Dict:
        return {'outcomes': len(getattr(self, '_outcomes', []))[-100:]}
    
    def deserialize_state(self, state: Dict) -> None:
        self._outcomes = state.get('outcomes', [])


class MultiTrackChallenger:
    """
    Orchestrates multiple LLM challenger tracks for robust analysis.
    
    Each track is an independent LLM that challenges the response.
    Results are aggregated using configurable consensus methods.
    """
    
    # Default API endpoints
    API_ENDPOINTS = {
        LLMProvider.GROK: "https://api.x.ai/v1/chat/completions",
        LLMProvider.CLAUDE: "https://api.anthropic.com/v1/messages",
        LLMProvider.GPT4: "https://api.openai.com/v1/chat/completions",
        LLMProvider.GPT35: "https://api.openai.com/v1/chat/completions",
        LLMProvider.GEMINI: "https://generativelanguage.googleapis.com/v1beta/models",
        LLMProvider.MISTRAL: "https://api.mistral.ai/v1/chat/completions",
        LLMProvider.COHERE: "https://api.cohere.ai/v1/chat",
    }
    
    # Default models per provider (updated to latest versions)
    # Primary provider for multi-track (Vertex AI)
    PRIMARY_PROVIDER = LLMProvider.VERTEX
    
    # Provider priority order for multi-track analysis
    PROVIDER_PRIORITY = [
        LLMProvider.VERTEX,   # Primary - Google Vertex AI (enterprise)
        LLMProvider.GPT4,     # Secondary - OpenAI
        LLMProvider.GROK,     # Tertiary - xAI
        LLMProvider.GEMINI,   # Fallback - Google public API
    ]
    
    DEFAULT_MODELS = {
        LLMProvider.VERTEX: "gemini-3-pro-preview",  # PRIMARY - Latest Gemini 3 via Vertex
        LLMProvider.GROK: "grok-4-1-fast-reasoning",  # Latest Grok with reasoning
        LLMProvider.CLAUDE: "claude-sonnet-4-20250514",  # Latest Claude Sonnet
        LLMProvider.GPT4: "gpt-4o",  # Latest GPT-4
        LLMProvider.GPT35: "gpt-3.5-turbo",
        LLMProvider.GEMINI: "gemini-3-pro-preview",  # Latest Gemini 3
        LLMProvider.MISTRAL: "mistral-large-latest",
        LLMProvider.COHERE: "command-r-plus",
    }
    
    @classmethod
    def get_latest_model_for_provider(cls, provider: LLMProvider) -> str:
        """Get the latest available model for a provider from registry."""
        try:
            from core.llm_registry import LLMRegistry, LLMProvider as RegistryProvider
            registry = LLMRegistry()
            
            # Map our provider enum to registry enum
            provider_map = {
                LLMProvider.GROK: RegistryProvider.GROK,
                LLMProvider.CLAUDE: RegistryProvider.ANTHROPIC,
                LLMProvider.GPT4: RegistryProvider.OPENAI,
            }
            
            if provider in provider_map:
                latest = registry.get_latest_model(provider_map[provider])
                if latest:
                    return latest.model_id
        except Exception as e:
            pass  # Fall back to default
        
        return cls.DEFAULT_MODELS.get(provider, "unknown")
    
    # Challenge prompts
    CHALLENGE_PROMPTS = {
        "evidence_demand": """You are a rigorous evidence auditor. Analyze this response for a {domain} context.

CLAIM: {claim}
RESPONSE: {response}

YOUR TASK: Identify what EVIDENCE is needed to verify claims.

For each assertion:
1. What evidence would prove it?
2. Is that evidence present?
3. If missing, rate severity (1-5)

Respond in JSON:
{{"evidence_gaps": [{{"claim": "...", "evidence_needed": "...", "present": false, "severity": 4}}], "overall_score": 0-100, "accept": true/false}}""",

        "devils_advocate": """You are a devil's advocate. Find ways this could be WRONG.

CLAIM: {claim}
RESPONSE: {response}
DOMAIN: {domain}

YOUR TASK: Actively seek flaws, errors, and problems.

Consider:
1. What assumptions might be false?
2. What scenarios would make this incorrect?
3. What logical fallacies are present?

Respond in JSON:
{{"flaws": [{{"issue": "...", "severity": 1-5, "likelihood": 1-5}}], "risk_score": 0-100, "concerns": ["..."]}}""",

        "completeness": """You are a completeness auditor for {domain}.

CLAIM: {claim}
RESPONSE: {response}

YOUR TASK: Is this ACTUALLY complete or just claimed complete?

Check for:
- TODO/FIXME/placeholder markers
- Missing error handling
- Unimplemented functions
- Missing edge cases

Respond in JSON:
{{"incomplete_items": [{{"item": "...", "importance": 1-5}}], "actually_complete": true/false, "completion_percent": 0-100}}""",

        "safety": """You are a safety auditor for {domain} content.

CLAIM: {claim}
RESPONSE: {response}

YOUR TASK: Identify safety risks.

For {domain}, check:
- Medical: dangerous advice, missing disclaimers
- Legal: unauthorized practice, missing qualifications
- Financial: unrealistic promises, missing risk warnings
- Technical: security vulnerabilities, data exposure

Respond in JSON:
{{"safety_issues": [{{"issue": "...", "severity": 1-5}}], "safe": true/false, "required_disclaimers": ["..."]}}""",

        "citation_verification": """You are a citation verification expert. Your job is to VERIFY that sources/citations mentioned are REAL and ACCURATE.

CLAIM: {claim}
RESPONSE: {response}
DOMAIN: {domain}

YOUR TASK - THREE STEPS:

STEP 1: EXTRACT CITATIONS
List all sources, studies, papers, statistics, or factual claims referenced.
Examples: "Smith et al. 2019", "JAMA study", "FDA approved", "25% reduction"

STEP 2: VERIFY CITATIONS EXIST
For each citation, determine:
- Does this source/study/paper actually exist?
- Is the author/journal/organization real?
- Is the date/year plausible?

STEP 3: VERIFY CITATIONS SUPPORT CLAIMS
For each citation, determine:
- Does the cited source actually say what is claimed?
- Is the statistic/finding accurately represented?
- Is context missing that changes the meaning?

Respond in JSON:
{{
    "citations_found": [
        {{
            "citation": "exact citation text",
            "type": "study|statistic|expert|regulation|other",
            "exists": true/false,
            "exists_confidence": 0.0-1.0,
            "supports_claim": true/false,
            "supports_confidence": 0.0-1.0,
            "issues": ["any problems with this citation"]
        }}
    ],
    "uncited_claims": ["claims made without any citation"],
    "fabricated_citations": ["citations that appear to be made up"],
    "misrepresented_citations": ["citations that don't actually support the claim"],
    "overall_citation_score": 0-100,
    "verified": true/false
}}

IMPORTANT: Do NOT just accept statements. VERIFY with your knowledge. If you cannot verify a citation exists, mark exists_confidence as low."""
    }
    
    def __init__(self, 
                 default_tracks: List[TrackConfig] = None,
                 consensus_method: ConsensusMethod = ConsensusMethod.WEIGHTED_AVERAGE,
                 storage_path: Path = None,
                 auto_discover_models: bool = True):
        """
        Initialize the multi-track challenger.
        
        Args:
            default_tracks: Default track configurations
            consensus_method: How to aggregate track results
            storage_path: Where to store learning data
            auto_discover_models: Auto-discover latest models from provider APIs
        """
        self.tracks: Dict[str, TrackConfig] = {}
        self.consensus_method = consensus_method
        self.storage_path = storage_path or Path("multi_track_data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.auto_discover_models = auto_discover_models
        
        # Model discovery cache
        self._discovered_models: Dict[LLMProvider, List[str]] = {}
        self._discovery_timestamp: Optional[datetime] = None
        
        # Auto-discover models on init
        if auto_discover_models:
            self._refresh_model_discovery()
        
        # Add default tracks if provided
        if default_tracks:
            for track in default_tracks:
                self.add_track(track)
        
        # Track performance history for learning
        self.track_performance: Dict[str, Dict] = {}
        self._load_performance()
        
        # Learning interface state
        self._outcomes: List[Dict] = []
        self._feedback: List[Dict] = []
        self._domain_adjustments: Dict[str, float] = {}
    
    # ===== Learning Interface Methods =====
    
    def record_outcome(self, outcome: Dict) -> None:
        """Record challenge outcome for learning."""
        getattr(self, '_outcomes', {}).append(outcome)
        # Update track performance based on outcome
        for track_id, result in outcome.get('track_results', {}).items():
            if track_id not in self.track_performance:
                self.track_performance[track_id] = {'correct': 0, 'total': 0}
            self.track_performance[track_id]['total'] += 1
            if result.get('found_real_issue', False):
                self.track_performance[track_id]['correct'] += 1
    
    def record_feedback(self, feedback: Dict) -> None:
        """Record feedback on multi-track challenges."""
        getattr(self, '_feedback', {}).append(feedback)
        domain = feedback.get('domain', 'general')
        if feedback.get('consensus_wrong', False):
            self._domain_adjustments[domain] = getattr(self, '_domain_adjustments', {}).get(domain, 0.0) - 0.05
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt consensus thresholds."""
        pass
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get domain adjustment."""
        return getattr(self, '_domain_adjustments', {}).get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics."""
        return {
            'track_performance': dict(self.track_performance),
            'domain_adjustments': dict(self._domain_adjustments),
            'outcomes_recorded': len(self._outcomes),
            'feedback_recorded': len(self._feedback),
            'tracks_configured': len(self.tracks)
        }
    
    def _refresh_model_discovery(self):
        """Refresh the list of available models from provider APIs."""
        try:
            from core.llm_registry import LLMRegistry, LLMProvider as RegistryProvider
            
            registry = LLMRegistry()
            
            # Map our enum to registry enum
            provider_map = {
                LLMProvider.GROK: RegistryProvider.GROK,
                LLMProvider.GPT4: RegistryProvider.OPENAI,
            }
            
            for our_provider, registry_provider in provider_map.items():
                discovered = registry.discover_models(registry_provider)
                if discovered:
                    self._discovered_models[our_provider] = discovered
                    print(f"[MultiTrack] Discovered {len(discovered)} models for {our_provider.value}")
                    
                    # Update default model to best reasoning model
                    best = registry.get_best_reasoning_model(registry_provider)
                    if best:
                        self.DEFAULT_MODELS[our_provider] = best
                        print(f"[MultiTrack] Set {our_provider.value} default to: {best}")
            
            self._discovery_timestamp = datetime.now()
            
        except Exception as e:
            print(f"[MultiTrack] Model discovery failed: {e}")
    
    def get_best_model(self, provider: LLMProvider, prefer_reasoning: bool = True) -> str:
        """
        Get the best available model for a provider.
        
        Args:
            provider: LLM provider to get model for
            prefer_reasoning: If True, prioritize reasoning-capable models
        
        Returns:
            Best available model ID
        """
        # Check if we have discovered models
        if provider in self._discovered_models and self._discovered_models[provider]:
            models = self._discovered_models[provider]
            
            if prefer_reasoning:
                # Prefer reasoning models (exclude non-reasoning variants)
                reasoning = [m for m in models if 'reasoning' in m.lower() and 'non-reasoning' not in m.lower()]
                if reasoning:
                    # Sort to get the latest version (highest version number)
                    reasoning.sort(reverse=True)
                    return reasoning[0]
            
            # Fall back to any latest model
            return models[0]
        
        # Fall back to default
        return self.DEFAULT_MODELS.get(provider, "unknown")
    
    def add_track(self, config: TrackConfig) -> None:
        """Add a challenger track."""
        self.tracks[config.track_id] = config
        if config.track_id not in self.track_performance:
            self.track_performance[config.track_id] = {
                "total_runs": 0,
                "correct_decisions": 0,
                "avg_time_ms": 0,
                "domains": {}
            }
    
    def remove_track(self, track_id: str) -> bool:
        """Remove a challenger track."""
        if track_id in self.tracks:
            del self.tracks[track_id]
            return True
        return False
    
    def configure_track(self, track_id: str, **kwargs) -> bool:
        """Update track configuration."""
        if track_id not in self.tracks:
            return False
        
        track = self.tracks[track_id]
        for key, value in kwargs.items():
            if hasattr(track, key):
                setattr(track, key, value)
        return True
    
    def get_active_tracks(self) -> List[TrackConfig]:
        """Get all enabled tracks."""
        return [t for t in self.tracks.values() if t.enabled]
    
    def create_preset_tracks(self, 
                             providers: List[LLMProvider],
                             api_keys: Dict[LLMProvider, str]) -> List[TrackConfig]:
        """
        Create tracks for specified providers with API keys.
        
        Args:
            providers: Which LLM providers to use
            api_keys: API keys for each provider
        
        Returns:
            List of configured tracks
        """
        tracks = []
        for i, provider in enumerate(providers):
            track = TrackConfig(
                track_id=f"track_{provider.value}_{i+1}",
                provider=provider,
                model_name=self.DEFAULT_MODELS.get(provider, "unknown"),
                api_key=api_keys.get(provider),
                weight=1.0,
                enabled=provider in api_keys
            )
            tracks.append(track)
            self.add_track(track)
        return tracks
    
    async def execute_track(self,
                            track: TrackConfig,
                            claim: str,
                            response: str,
                            domain: str) -> TrackResult:
        """
        Execute a single challenger track.
        
        Args:
            track: Track configuration
            claim: The claim being analyzed
            response: The full response
            domain: Domain context
        
        Returns:
            TrackResult with analysis
        """
        import time
        start_time = time.time()
        
        try:
            # Build challenge prompts
            challenges = []
            for challenge_type in track.challenge_types:
                if challenge_type in self.CHALLENGE_PROMPTS:
                    prompt = self.CHALLENGE_PROMPTS[challenge_type].format(
                        claim=claim,
                        response=response,
                        domain=domain
                    )
                    challenges.append((challenge_type, prompt))
            
            # Execute challenges via LLM API
            all_issues = []
            all_evidence_gaps = []
            all_risk_factors = []
            all_guidance = []
            raw_responses = []
            
            for challenge_type, prompt in challenges:
                result = await self._call_llm(track, prompt)
                raw_responses.append(f"[{challenge_type}]: {result}")
                
                # Parse result
                parsed = self._parse_llm_result(result, challenge_type)
                all_issues.extend(parsed.get("issues", []))
                all_evidence_gaps.extend(parsed.get("evidence_gaps", []))
                all_risk_factors.extend(parsed.get("risks", []))
                all_guidance.extend(parsed.get("guidance", []))
            
            # Calculate confidence
            issue_penalty = min(len(all_issues) * 0.1, 0.4)
            evidence_penalty = min(len(all_evidence_gaps) * 0.1, 0.3)
            base_confidence = 0.7
            confidence = max(0.0, base_confidence - issue_penalty - evidence_penalty)
            
            execution_time = (time.time() - start_time) * 1000
            
            return TrackResult(
                track_id=track.track_id,
                provider=track.provider,
                model_name=track.model_name,
                issues_found=all_issues,
                evidence_gaps=all_evidence_gaps,
                risk_factors=all_risk_factors,
                confidence_score=confidence,
                should_accept=confidence >= 0.6 and len(all_issues) < 3,
                should_remediate=len(all_issues) > 0 or len(all_evidence_gaps) > 0,
                remediation_guidance=all_guidance,
                execution_time_ms=execution_time,
                raw_response="\n".join(raw_responses)
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return TrackResult(
                track_id=track.track_id,
                provider=track.provider,
                model_name=track.model_name,
                issues_found=[],
                evidence_gaps=[],
                risk_factors=[],
                confidence_score=0.5,
                should_accept=False,
                should_remediate=True,
                remediation_guidance=["Track execution failed - manual review recommended"],
                execution_time_ms=execution_time,
                raw_response="",
                error=str(e)
            )
    
    async def challenge_parallel(self,
                                  claim: str,
                                  response: str,
                                  domain: str = "general",
                                  tracks: List[str] = None,
                                  output_mode: OutputMode = OutputMode.CONSENSUS) -> MultiTrackVerdict:
        """
        Run all tracks in parallel with configurable output modes.
        
        Args:
            claim: The claim being analyzed
            response: The full response
            domain: Domain context
            tracks: Specific track IDs to use (default: all enabled)
            output_mode: How to present results:
                - CONSENSUS: Single aggregated decision (default)
                - PARALLEL_PATHS: Each track as alternate execution path
                - AUDIT_TRAIL: All tracks as audit record for review
                - USER_SELECTIVE: Allow users to pick elements from tracks
        
        Returns:
            MultiTrackVerdict with results based on output_mode
        
        Usage Examples:
            # Traditional consensus decision
            verdict = await challenger.challenge_parallel(claim, response, domain)
            
            # Get alternate paths for user selection
            verdict = await challenger.challenge_parallel(
                claim, response, domain, output_mode=OutputMode.PARALLEL_PATHS
            )
            paths = verdict.get_parallel_paths()
            
            # Get audit trail for compliance
            verdict = await challenger.challenge_parallel(
                claim, response, domain, output_mode=OutputMode.AUDIT_TRAIL
            )
            audit = verdict.get_audit_trail()
            
            # Allow user to pick elements
            verdict = await challenger.challenge_parallel(
                claim, response, domain, output_mode=OutputMode.USER_SELECTIVE
            )
            issues = verdict.get_selectable_elements_by_type("issue")
            composite = verdict.select_elements(["track_1_issue_0", "track_2_guidance_1"])
        """
        import time
        start_time = time.time()
        
        # Get tracks to execute
        if tracks:
            active_tracks = [self.tracks[t] for t in tracks if t in self.tracks and self.tracks[t].enabled]
        else:
            active_tracks = self.get_active_tracks()
        
        if not active_tracks:
            # Fallback: create default pattern-based track
            return await self._fallback_analysis(claim, response, domain)
        
        # Execute all tracks in parallel
        tasks = [
            self.execute_track(track, claim, response, domain)
            for track in active_tracks
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        track_results = []
        for result in results:
            if isinstance(result, TrackResult):
                track_results.append(result)
            elif isinstance(result, Exception):
                print(f"[MultiTrack] Track failed with exception: {result}")
        
        # Calculate consensus
        verdict = self._calculate_consensus(
            track_results=track_results,
            claim=claim,
            domain=domain,
            total_time_ms=(time.time() - start_time) * 1000,
            output_mode=output_mode
        )
        
        # Update performance tracking
        self._update_performance(track_results, domain)
        
        return verdict
    
    def _calculate_consensus(self,
                              track_results: List[TrackResult],
                              claim: str,
                              domain: str,
                              total_time_ms: float,
                              output_mode: OutputMode = OutputMode.CONSENSUS) -> MultiTrackVerdict:
        """
        Calculate consensus from track results with configurable output mode.
        
        Args:
            track_results: Results from all executed tracks
            claim: The analyzed claim
            domain: Domain context
            total_time_ms: Total execution time
            output_mode: How to present results
        """
        
        if not track_results:
            return MultiTrackVerdict(
                claim=claim,
                domain=domain,
                track_results=[],
                tracks_executed=0,
                tracks_succeeded=0,
                tracks_failed=0,
                consensus_method=self.consensus_method,
                consensus_accept=False,
                consensus_confidence=0.0,
                consensus_remediate=True,
                all_issues=[],
                all_evidence_gaps=[],
                unified_guidance=["No tracks executed - manual review required"],
                track_disagreements=[],
                total_time_ms=total_time_ms,
                inventions_used=["NOVEL-8: Cross-LLM Orchestration"],
                output_mode=output_mode,
                selectable_elements=[]
            )
        
        # Aggregate all findings
        all_issues = []
        all_evidence_gaps = []
        all_guidance = []
        
        for result in track_results:
            all_issues.extend(result.issues_found)
            all_evidence_gaps.extend(result.evidence_gaps)
            all_guidance.extend(result.remediation_guidance)
        
        # Deduplicate
        all_issues = list(set(all_issues))
        all_evidence_gaps = list(set(all_evidence_gaps))
        all_guidance = list(set(all_guidance))
        
        # Calculate consensus based on method
        succeeded = [r for r in track_results if r.error is None]
        failed = [r for r in track_results if r.error is not None]
        
        if self.consensus_method == ConsensusMethod.MAJORITY_VOTE:
            accept_votes = sum(1 for r in succeeded if r.should_accept)
            consensus_accept = accept_votes > len(succeeded) / 2
            consensus_confidence = accept_votes / len(succeeded) if succeeded else 0.0
            
        elif self.consensus_method == ConsensusMethod.WEIGHTED_AVERAGE:
            total_weight = sum(self.tracks[r.track_id].weight for r in succeeded if r.track_id in self.tracks)
            if total_weight > 0:
                weighted_confidence = sum(
                    r.confidence_score * self.tracks[r.track_id].weight 
                    for r in succeeded if r.track_id in self.tracks
                ) / total_weight
                consensus_confidence = weighted_confidence
                consensus_accept = weighted_confidence >= 0.6
            else:
                consensus_confidence = 0.5
                consensus_accept = False
                
        elif self.consensus_method == ConsensusMethod.UNANIMOUS:
            consensus_accept = all(r.should_accept for r in succeeded) if succeeded else False
            consensus_confidence = min(r.confidence_score for r in succeeded) if succeeded else 0.0
            
        elif self.consensus_method == ConsensusMethod.STRICTEST:
            consensus_accept = all(r.should_accept for r in succeeded) if succeeded else False
            consensus_confidence = min(r.confidence_score for r in succeeded) if succeeded else 0.0
            
        elif self.consensus_method == ConsensusMethod.BAYESIAN:
            # Bayesian consensus: P(accept|evidence) using prior and likelihoods
            prior = 0.5  # Neutral prior
            for r in succeeded:
                likelihood = r.confidence_score if r.should_accept else (1 - r.confidence_score)
                prior = (prior * likelihood) / (prior * likelihood + (1 - prior) * (1 - likelihood) + 0.001)
            consensus_confidence = prior
            consensus_accept = prior >= 0.6
        
        else:
            # Default to majority
            accept_votes = sum(1 for r in succeeded if r.should_accept)
            consensus_accept = accept_votes > len(succeeded) / 2
            consensus_confidence = accept_votes / len(succeeded) if succeeded else 0.0
        
        # Find disagreements (valuable for learning)
        disagreements = []
        if succeeded:
            accept_tracks = [r.track_id for r in succeeded if r.should_accept]
            reject_tracks = [r.track_id for r in succeeded if not r.should_accept]
            if accept_tracks and reject_tracks:
                disagreements.append({
                    "type": "accept_reject_split",
                    "accept_tracks": accept_tracks,
                    "reject_tracks": reject_tracks
                })
        
        consensus_remediate = any(r.should_remediate for r in succeeded) or len(all_issues) > 0
        
        # Build selectable elements for USER_SELECTIVE mode
        selectable_elements = []
        if output_mode == OutputMode.USER_SELECTIVE:
            for result in succeeded:
                selectable_elements.extend(result.to_selectable_elements())
        
        return MultiTrackVerdict(
            claim=claim,
            domain=domain,
            track_results=track_results,
            tracks_executed=len(track_results),
            tracks_succeeded=len(succeeded),
            tracks_failed=len(failed),
            consensus_method=self.consensus_method,
            consensus_accept=consensus_accept,
            consensus_confidence=consensus_confidence,
            consensus_remediate=consensus_remediate,
            all_issues=all_issues,
            all_evidence_gaps=all_evidence_gaps,
            unified_guidance=all_guidance[:10],
            track_disagreements=disagreements,
            total_time_ms=total_time_ms,
            inventions_used=[
                "NOVEL-8: Cross-LLM Governance Orchestration",
                "NOVEL-19: LLM Registry (Multi-Provider)",
                "PPA1-Inv20: Human-Machine Hybrid Arbitration",
                "PPA1-Inv23: AI Common Sense via Triangulation",
                "NOVEL-23+: Multi-Track with User-Selective Output"
            ],
            output_mode=output_mode,
            selectable_elements=selectable_elements
        )
    
    async def _call_llm(self, track: TrackConfig, prompt: str) -> str:
        """Call the LLM API for a track."""
        
        if not track.api_key:
            return self._pattern_fallback(prompt)
        
        try:
            async with aiohttp.ClientSession() as session:
                # Special handling for Gemini API (different auth pattern)
                if track.provider == LLMProvider.GEMINI:
                    return await self._call_gemini(session, track, prompt)
                
                headers = self._get_headers(track)
                payload = self._get_payload(track, prompt)
                endpoint = self.API_ENDPOINTS.get(track.provider)
                
                if not endpoint:
                    return self._pattern_fallback(prompt)
                
                async with session.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=track.timeout_seconds)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._extract_content(track.provider, data)
                    else:
                        error_text = await response.text()
                        print(f"[MultiTrack] API error {response.status}: {error_text[:100]}")
                        return self._pattern_fallback(prompt)
                        
        except Exception as e:
            print(f"[MultiTrack] API call failed: {e}")
            return self._pattern_fallback(prompt)
    
    async def _call_gemini(self, session: aiohttp.ClientSession, track: TrackConfig, prompt: str) -> str:
        """
        Call Gemini API with correct auth format.
        
        Gemini uses API key in URL query parameter, not header.
        Endpoint format: {base}/models/{model}:generateContent?key=API_KEY
        """
        try:
            # Build Gemini-specific endpoint with API key in URL
            base_url = "https://generativelanguage.googleapis.com/v1beta"
            model = track.model_name or "gemini-2.0-flash"
            endpoint = f"{base_url}/models/{model}:generateContent?key={track.api_key}"
            
            # Gemini payload format
            system = track.custom_system_prompt or "You are a rigorous adversarial analyzer. Find problems, don't confirm quality. Output valid JSON."
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": f"{system}\n\n{prompt}"}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": track.temperature,
                    "maxOutputTokens": track.max_tokens
                }
            }
            
            headers = {"Content-Type": "application/json"}
            
            async with session.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=track.timeout_seconds)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    # Extract content from Gemini response format
                    candidates = data.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])
                        if parts:
                            return parts[0].get("text", "")
                    return self._pattern_fallback(prompt)
                else:
                    error_text = await response.text()
                    print(f"[MultiTrack/Gemini] API error {response.status}: {error_text[:100]}")
                    return self._pattern_fallback(prompt)
                    
        except Exception as e:
            print(f"[MultiTrack/Gemini] API call failed: {e}")
            return self._pattern_fallback(prompt)
    
    def _get_headers(self, track: TrackConfig) -> Dict[str, str]:
        """Get API headers for provider."""
        if track.provider == LLMProvider.CLAUDE:
            return {
                "Content-Type": "application/json",
                "x-api-key": track.api_key,
                "anthropic-version": "2023-06-01"
            }
        elif track.provider in [LLMProvider.GPT4, LLMProvider.GPT35]:
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {track.api_key}"
            }
        elif track.provider == LLMProvider.GROK:
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {track.api_key}"
            }
        else:
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {track.api_key}"
            }
    
    def _get_payload(self, track: TrackConfig, prompt: str) -> Dict:
        """Get API payload for provider."""
        system = track.custom_system_prompt or "You are a rigorous adversarial analyzer. Find problems, don't confirm quality. Output valid JSON."
        
        if track.provider == LLMProvider.CLAUDE:
            return {
                "model": track.model_name,
                "max_tokens": track.max_tokens,
                "messages": [
                    {"role": "user", "content": f"{system}\n\n{prompt}"}
                ]
            }
        else:  # OpenAI-compatible format (GPT, Grok, Mistral)
            return {
                "model": track.model_name,
                "max_tokens": track.max_tokens,
                "temperature": track.temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ]
            }
    
    def _extract_content(self, provider: LLMProvider, response: Dict) -> str:
        """Extract content from API response."""
        if provider == LLMProvider.CLAUDE:
            return response.get("content", [{}])[0].get("text", "")
        else:  # OpenAI-compatible
            return response.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    def _pattern_fallback(self, prompt: str) -> str:
        """Pattern-based fallback when API unavailable."""
        # Simple pattern detection
        issues = []
        evidence_gaps = []
        
        prompt_lower = prompt.lower()
        
        if "100%" in prompt_lower or "zero" in prompt_lower:
            issues.append("Absolute claim detected")
        if "todo" in prompt_lower or "fixme" in prompt_lower:
            issues.append("Placeholder markers found")
        if "complete" in prompt_lower or "finished" in prompt_lower:
            evidence_gaps.append("Completion claim needs verification")
        if "always" in prompt_lower or "never" in prompt_lower:
            issues.append("Absolute language detected")
        
        return json.dumps({
            "issues": issues,
            "evidence_gaps": evidence_gaps,
            "confidence": 0.5,
            "source": "pattern_fallback"
        })
    
    def _parse_llm_result(self, result: str, challenge_type: str) -> Dict:
        """Parse LLM result into structured format."""
        parsed = {
            "issues": [],
            "evidence_gaps": [],
            "risks": [],
            "guidance": []
        }
        
        try:
            # Try to parse as JSON
            data = json.loads(result)
            
            # Extract based on challenge type
            if challenge_type == "evidence_demand":
                for gap in data.get("evidence_gaps", []):
                    if not gap.get("present", True):
                        parsed["evidence_gaps"].append(gap.get("evidence_needed", "Unknown"))
                if not data.get("accept", True):
                    parsed["guidance"].append("Evidence gaps detected - demand verification")
                    
            elif challenge_type == "devils_advocate":
                for flaw in data.get("flaws", []):
                    parsed["issues"].append(flaw.get("issue", "Unknown flaw"))
                parsed["risks"].extend(data.get("concerns", []))
                
            elif challenge_type == "completeness":
                for item in data.get("incomplete_items", []):
                    parsed["issues"].append(f"Incomplete: {item.get('item', 'Unknown')}")
                if not data.get("actually_complete", True):
                    parsed["guidance"].append("Not actually complete - finish implementation")
                    
            elif challenge_type == "safety":
                for issue in data.get("safety_issues", []):
                    parsed["issues"].append(f"Safety: {issue.get('issue', 'Unknown')}")
                for disclaimer in data.get("required_disclaimers", []):
                    parsed["guidance"].append(f"Add disclaimer: {disclaimer}")
                    
            elif challenge_type == "citation_verification":
                # Process fabricated citations - these are serious
                for fabricated in data.get("fabricated_citations", []):
                    parsed["issues"].append(f"FABRICATED CITATION: {fabricated}")
                    parsed["risks"].append("Citation appears to be made up")
                
                # Process misrepresented citations
                for misrep in data.get("misrepresented_citations", []):
                    parsed["issues"].append(f"MISREPRESENTED: {misrep}")
                
                # Process uncited claims
                for uncited in data.get("uncited_claims", []):
                    parsed["evidence_gaps"].append(f"Uncited: {uncited}")
                
                # Process individual citations
                for citation in data.get("citations_found", []):
                    if not citation.get("exists", True):
                        parsed["issues"].append(f"Unverified citation: {citation.get('citation', 'Unknown')}")
                    elif citation.get("exists_confidence", 1.0) < 0.5:
                        parsed["evidence_gaps"].append(f"Low confidence: {citation.get('citation', 'Unknown')}")
                    if not citation.get("supports_claim", True):
                        parsed["issues"].append(f"Citation doesn't support claim: {citation.get('citation', 'Unknown')}")
                    for issue in citation.get("issues", []):
                        parsed["risks"].append(issue)
                
                if not data.get("verified", True):
                    parsed["guidance"].append("Citations need verification - provide verifiable sources")
                
                # Adjust score based on citation quality
                citation_score = data.get("overall_citation_score", 50)
                if citation_score < 50:
                    parsed["score"] = min(parsed.get("score", 100), citation_score)
                    
        except json.JSONDecodeError:
            # Fallback: extract issues from text
            if "issue" in result.lower() or "problem" in result.lower():
                parsed["issues"].append("Issue detected in response")
            if "evidence" in result.lower() or "proof" in result.lower():
                parsed["evidence_gaps"].append("Evidence may be needed")
        
        return parsed
    
    async def _fallback_analysis(self, claim: str, response: str, domain: str) -> MultiTrackVerdict:
        """Fallback when no tracks available."""
        import time
        start = time.time()
        
        # Pattern-based analysis
        issues = []
        evidence_gaps = []
        
        response_lower = response.lower()
        
        if "100%" in response_lower or "zero bugs" in response_lower:
            issues.append("Absolute perfection claim")
        if "todo" in response_lower or "fixme" in response_lower:
            issues.append("Contains placeholder markers")
        if "complete" in response_lower and "test" not in response_lower:
            evidence_gaps.append("Completion claim without test evidence")
        
        confidence = 0.6 - (len(issues) * 0.1) - (len(evidence_gaps) * 0.1)
        
        return MultiTrackVerdict(
            claim=claim,
            domain=domain,
            track_results=[],
            tracks_executed=0,
            tracks_succeeded=0,
            tracks_failed=0,
            consensus_method=ConsensusMethod.MAJORITY_VOTE,
            consensus_accept=confidence >= 0.5 and len(issues) < 2,
            consensus_confidence=max(0.0, confidence),
            consensus_remediate=len(issues) > 0,
            all_issues=issues,
            all_evidence_gaps=evidence_gaps,
            unified_guidance=["Pattern-based analysis only - add LLM tracks for deeper analysis"],
            track_disagreements=[],
            total_time_ms=(time.time() - start) * 1000,
            inventions_used=["Pattern-based fallback"]
        )
    
    def _update_performance(self, results: List[TrackResult], domain: str):
        """Update track performance metrics."""
        for result in results:
            if result.track_id in self.track_performance:
                perf = self.track_performance[result.track_id]
                perf["total_runs"] += 1
                # Update rolling average time
                n = perf["total_runs"]
                perf["avg_time_ms"] = ((n - 1) * perf["avg_time_ms"] + result.execution_time_ms) / n
                # Track domain performance
                if domain not in perf["domains"]:
                    perf["domains"][domain] = {"runs": 0, "issues_found": 0}
                perf["domains"][domain]["runs"] += 1
                perf["domains"][domain]["issues_found"] += len(result.issues_found)
        
        self._save_performance()
    
    def _load_performance(self):
        """Load performance history from disk."""
        perf_file = self.storage_path / "track_performance.json"
        if perf_file.exists():
            try:
                with open(perf_file, 'r') as f:
                    self.track_performance = json.load(f)
            except Exception:
                pass
    
    def _save_performance(self):
        """Save performance history to disk."""
        perf_file = self.storage_path / "track_performance.json"
        try:
            with open(perf_file, 'w') as f:
                json.dump(self.track_performance, f, indent=2)
        except Exception:
            pass
    
    def get_best_tracks_for_domain(self, domain: str, n: int = 3) -> List[str]:
        """Get the best performing tracks for a domain."""
        domain_scores = []
        
        for track_id, perf in self.track_performance.items():
            if domain in perf.get("domains", {}):
                domain_perf = perf["domains"][domain]
                # Score based on issues found per run (higher = better at finding issues)
                if domain_perf["runs"] > 0:
                    score = domain_perf["issues_found"] / domain_perf["runs"]
                    domain_scores.append((track_id, score))
        
        # Sort by score descending
        domain_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [t[0] for t in domain_scores[:n]]
    
    # =========================================================================
    # Learning Interface (5/5 methods)
    # =========================================================================
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback about track performance."""
        track_id = feedback.get('track_id')
        if track_id and track_id in self.track_performance:
            if feedback.get('was_correct', True):
                self.track_performance[track_id]["improvements"] = self.track_performance[track_id].get("improvements", 0) + 1
            else:
                self.track_performance[track_id]["failures"] = self.track_performance[track_id].get("failures", 0) + 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return multi-track statistics."""
        return {
            'total_comparisons': getattr(self, '_total_comparisons', 0),
            'improvements_found': getattr(self, '_improvements_found', 0),
            'agreement_total': getattr(self, '_agreement_total', 0),
            'track_performance': dict(self.track_performance),
            'outcomes_recorded': len(self._outcomes)
        }
    
    def serialize_state(self) -> Dict:
        """Serialize state for persistence."""
        return {
            'track_performance': dict(self.track_performance),
            'outcomes': getattr(self, '_outcomes', [])[-100:],
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore state from serialized data."""
        self.track_performance = state.get('track_performance', {})
        self._outcomes = state.get('outcomes', [])


def create_multi_track_challenger(
    providers: List[LLMProvider] = None,
    api_keys: Dict[LLMProvider, str] = None,
    consensus: ConsensusMethod = ConsensusMethod.WEIGHTED_AVERAGE
) -> MultiTrackChallenger:
    """
    Factory function to create multi-track challenger.
    
    Args:
        providers: LLM providers to use
        api_keys: API keys for providers
        consensus: Consensus method
    
    Returns:
        Configured MultiTrackChallenger
    """
    challenger = MultiTrackChallenger(consensus_method=consensus)
    
    if providers and api_keys:
        challenger.create_preset_tracks(providers, api_keys)
    
    return challenger
