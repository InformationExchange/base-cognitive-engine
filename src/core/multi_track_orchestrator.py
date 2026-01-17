"""
BAIS Cognitive Governance Engine - Multi-Track Orchestrator
Phase 3: A/B/N LLM comparison with consensus and user selection

Features:
- User configures which LLMs to use
- BAIS suggests track count based on complexity/risk
- Multiple LLMs evaluate in parallel
- Consensus algorithm for best output
- User can select preferred track or accept consensus

Patent: NOVEL-43 (Multi-Track Orchestrator)
"""

import logging
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class TrackSelectionMode(Enum):
    """How final track is selected."""
    CONSENSUS = "consensus"         # Algorithmic consensus
    USER_SELECT = "user_select"     # User picks
    HIGHEST_SCORE = "highest_score" # Highest BAIS score
    LOWEST_ISSUES = "lowest_issues" # Fewest issues detected


class ComplexityLevel(Enum):
    """Query complexity levels."""
    LOW = "low"           # Simple, well-understood
    MEDIUM = "medium"     # Moderate complexity
    HIGH = "high"         # Complex, multi-step
    CRITICAL = "critical" # High stakes, requires verification


class ConsolidationStrategy(Enum):
    """How to consolidate multiple track outputs."""
    BEST_SINGLE = "best_single"         # Pick best one
    MERGE_BEST_PARTS = "merge_best_parts" # Combine best parts
    WEIGHTED_AVERAGE = "weighted_average" # Score-weighted merge


@dataclass
class TrackResult:
    """Result from a single LLM track."""
    track_id: str
    llm_provider: str
    response: str
    
    # BAIS evaluation
    bais_score: float
    issues_detected: List[Dict[str, Any]]
    evidence_strength: str
    
    # Performance metrics
    latency_ms: float
    tokens_used: int
    
    # Confidence
    confidence: float
    reasoning: Optional[str] = None
    
    # Metadata
    timestamp: str = ""


@dataclass
class ConsensusResult:
    """Result from consensus algorithm."""
    recommended_response: str
    recommendation_reason: str
    strategy_used: ConsolidationStrategy
    
    # Agreement metrics
    agreement_score: float  # 0-1, how much tracks agree
    divergence_areas: List[str]
    
    # Source tracks
    contributing_tracks: List[str]
    primary_track: str
    
    # Confidence
    consensus_confidence: float


@dataclass
class TrackSuggestion:
    """BAIS suggestion for track configuration."""
    suggested_track_count: int
    suggested_llms: List[str]
    reason: str
    complexity: ComplexityLevel
    risk_level: str


# =============================================================================
# Complexity Analyzer
# =============================================================================

class ComplexityAnalyzer:
    """Analyzes query complexity to suggest track configuration."""
    
    def __init__(self):
        # High-risk domain keywords
        self._high_risk_domains = {
            'medical', 'health', 'diagnosis', 'treatment', 'drug',
            'legal', 'law', 'contract', 'liability', 'court',
            'financial', 'investment', 'tax', 'audit', 'compliance',
            'security', 'encryption', 'authentication', 'vulnerability'
        }
        
        # Complexity indicators
        self._complexity_indicators = {
            'high': ['architecture', 'design', 'system', 'integrate', 'migrate', 'refactor'],
            'medium': ['implement', 'create', 'build', 'modify', 'update'],
            'low': ['fix', 'typo', 'rename', 'format', 'comment']
        }
    
    def analyze(self, query: str, context: Optional[Dict] = None) -> TrackSuggestion:
        """Analyze query and suggest track configuration."""
        query_lower = query.lower()
        
        # Check for high-risk domains
        risk_level = "low"
        for domain in self._high_risk_domains:
            if domain in query_lower:
                risk_level = "high"
                break
        
        # Determine complexity
        complexity = ComplexityLevel.MEDIUM
        for level, indicators in self._complexity_indicators.items():
            if any(ind in query_lower for ind in indicators):
                complexity = ComplexityLevel[level.upper()]
                break
        
        # Additional complexity factors
        if len(query) > 500:
            complexity = ComplexityLevel.HIGH
        
        if context and context.get('domain') in ['medical', 'financial', 'legal']:
            risk_level = "high"
            complexity = ComplexityLevel.CRITICAL
        
        # Suggest track count
        if complexity == ComplexityLevel.CRITICAL or risk_level == "high":
            suggested_count = 3
            suggested_llms = ["grok", "openai", "anthropic"]
            reason = f"High complexity ({complexity.value}) or high risk ({risk_level}) - recommend 3 tracks for verification"
        elif complexity == ComplexityLevel.HIGH:
            suggested_count = 2
            suggested_llms = ["grok", "openai"]
            reason = f"High complexity - recommend 2 tracks for comparison"
        elif complexity == ComplexityLevel.MEDIUM:
            suggested_count = 1
            suggested_llms = ["grok"]
            reason = f"Medium complexity - single track sufficient"
        else:
            suggested_count = 1
            suggested_llms = ["grok"]
            reason = f"Low complexity - single track optimal for speed"
        
        return TrackSuggestion(
            suggested_track_count=suggested_count,
            suggested_llms=suggested_llms,
            reason=reason,
            complexity=complexity,
            risk_level=risk_level
        )

    # =========================================================================
    # Learning Interface
    # =========================================================================
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        """Record operation outcome for learning."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback to adjust behavior."""
        if not hasattr(self, '_learning_params'):
            self._learning_params = {}
        was_correct = feedback.get('was_correct', True)
        if not was_correct:
            for key in self._learning_params:
                self._learning_params[key] *= 0.95
    
    def get_statistics(self) -> Dict:
        """Return learning statistics."""
        return {
            'outcomes': len(getattr(self, '_outcomes', [])),
            'params': getattr(self, '_learning_params', {})
        }
    
    def serialize_state(self) -> Dict:
        """Serialize state for persistence."""
        return {
            'outcomes': getattr(self, '_outcomes', [])[-100:],
            'params': getattr(self, '_learning_params', {}),
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore state from serialized data."""
        self._outcomes = state.get('outcomes', [])
        self._learning_params = state.get('params', {})


# =============================================================================
# Consensus Engine
# =============================================================================

class ConsensusEngine:
    """Generates consensus from multiple track results."""
    
    def __init__(self):
        self._consensus_history: List[ConsensusResult] = []
    
    def generate_consensus(
        self,
        track_results: List[TrackResult],
        strategy: ConsolidationStrategy = ConsolidationStrategy.BEST_SINGLE
    ) -> ConsensusResult:
        """Generate consensus from track results."""
        if not track_results:
            return ConsensusResult(
                recommended_response="No tracks available",
                recommendation_reason="No LLM tracks were executed",
                strategy_used=strategy,
                agreement_score=0.0,
                divergence_areas=[],
                contributing_tracks=[],
                primary_track="none",
                consensus_confidence=0.0
            )
        
        if len(track_results) == 1:
            track = track_results[0]
            return ConsensusResult(
                recommended_response=track.response,
                recommendation_reason=f"Single track from {track.llm_provider}",
                strategy_used=ConsolidationStrategy.BEST_SINGLE,
                agreement_score=1.0,
                divergence_areas=[],
                contributing_tracks=[track.track_id],
                primary_track=track.track_id,
                consensus_confidence=track.confidence
            )
        
        # Multiple tracks - apply strategy
        if strategy == ConsolidationStrategy.BEST_SINGLE:
            return self._best_single_consensus(track_results)
        elif strategy == ConsolidationStrategy.WEIGHTED_AVERAGE:
            return self._weighted_consensus(track_results)
        else:
            return self._best_single_consensus(track_results)
    
    def _best_single_consensus(self, tracks: List[TrackResult]) -> ConsensusResult:
        """Select best single track based on score and issues."""
        # Score = BAIS score - (issue penalty)
        scored_tracks = []
        for track in tracks:
            issue_penalty = len(track.issues_detected) * 0.05
            effective_score = track.bais_score - issue_penalty
            scored_tracks.append((track, effective_score))
        
        # Sort by effective score
        scored_tracks.sort(key=lambda x: x[1], reverse=True)
        best_track, best_score = scored_tracks[0]
        
        # Calculate agreement
        scores = [t.bais_score for t in tracks]
        score_variance = self._calculate_variance(scores)
        agreement_score = max(0, 1 - score_variance)
        
        # Find divergence areas
        divergence = self._find_divergence(tracks)
        
        return ConsensusResult(
            recommended_response=best_track.response,
            recommendation_reason=f"Best score from {best_track.llm_provider} (score: {best_score:.2f})",
            strategy_used=ConsolidationStrategy.BEST_SINGLE,
            agreement_score=agreement_score,
            divergence_areas=divergence,
            contributing_tracks=[best_track.track_id],
            primary_track=best_track.track_id,
            consensus_confidence=best_track.confidence * agreement_score
        )
    
    def _weighted_consensus(self, tracks: List[TrackResult]) -> ConsensusResult:
        """Generate weighted consensus (for compatible outputs)."""
        # For text outputs, this falls back to best single
        # In future, could merge structured data
        return self._best_single_consensus(tracks)
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _find_divergence(self, tracks: List[TrackResult]) -> List[str]:
        """Find areas where tracks diverge."""
        divergence = []
        
        # Check if issue counts differ significantly
        issue_counts = [len(t.issues_detected) for t in tracks]
        if max(issue_counts) - min(issue_counts) > 2:
            divergence.append(f"Issue detection varies: {min(issue_counts)}-{max(issue_counts)}")
        
        # Check if scores differ significantly
        scores = [t.bais_score for t in tracks]
        if max(scores) - min(scores) > 0.2:
            divergence.append(f"Scores vary: {min(scores):.2f}-{max(scores):.2f}")
        
        # Check response length variance
        lengths = [len(t.response) for t in tracks]
        if max(lengths) > min(lengths) * 2:
            divergence.append("Response lengths vary significantly")
        
        return divergence

    # =========================================================================
    # Learning Interface
    # =========================================================================
    
    def record_outcome(self, outcome: Dict) -> None:
        """Record operation outcome for learning."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback to adjust behavior."""
        if not hasattr(self, '_learning_params'):
            self._learning_params = {}
        was_correct = feedback.get('was_correct', True)
        if not was_correct:
            for key in self._learning_params:
                self._learning_params[key] *= 0.95
    
    def get_statistics(self) -> Dict:
        """Return learning statistics."""
        return {
            'outcomes': len(getattr(self, '_outcomes', [])),
            'params': getattr(self, '_learning_params', {})
        }
    
    def serialize_state(self) -> Dict:
        """Serialize state for persistence."""
        return {
            'outcomes': getattr(self, '_outcomes', [])[-100:],
            'params': getattr(self, '_learning_params', {}),
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore state from serialized data."""
        self._outcomes = state.get('outcomes', [])
        self._learning_params = state.get('params', {})


# =============================================================================
# Multi-Track Orchestrator
# =============================================================================

class MultiTrackOrchestrator:
    """
    Orchestrates multiple LLM tracks for A/B/N comparison.
    
    Features:
    - User configures LLMs
    - BAIS suggests based on complexity
    - Parallel execution
    - Consensus generation
    - User selection option
    """
    
    def __init__(
        self,
        available_llms: Optional[List[str]] = None,
        max_parallel: int = 3,
        bais_evaluator: Optional[Callable] = None
    ):
        self.available_llms = available_llms or ["grok", "openai", "anthropic", "google"]
        self.max_parallel = max_parallel
        self.bais_evaluator = bais_evaluator
        
        self.complexity_analyzer = ComplexityAnalyzer()
        self.consensus_engine = ConsensusEngine()
        
        # Statistics
        self._stats = {
            'total_orchestrations': 0,
            'tracks_executed': 0,
            'consensus_used': 0,
            'user_selections': 0,
            'by_llm': {llm: 0 for llm in self.available_llms}
        }
    
    def suggest_tracks(
        self, 
        query: str, 
        context: Optional[Dict] = None
    ) -> TrackSuggestion:
        """Suggest track configuration based on query complexity."""
        return self.complexity_analyzer.analyze(query, context)
    
    def execute_tracks(
        self,
        query: str,
        llms: List[str],
        llm_provider: Callable[[str, str], str],
        context: Optional[Dict] = None
    ) -> List[TrackResult]:
        """Execute multiple tracks in parallel."""
        self._stats['total_orchestrations'] += 1
        
        # Filter to available LLMs
        llms_to_use = [llm for llm in llms if llm in self.available_llms]
        
        if not llms_to_use:
            logger.warning("[MultiTrack] No valid LLMs specified, using grok")
            llms_to_use = ["grok"]
        
        results = []
        
        # Execute in parallel using threads
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = {}
            for llm in llms_to_use:
                future = executor.submit(
                    self._execute_single_track,
                    query, llm, llm_provider, context
                )
                futures[future] = llm
            
            for future in as_completed(futures):
                llm = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    self._stats['tracks_executed'] += 1
                    self._stats['by_llm'][llm] += 1
                except Exception as e:
                    logger.error(f"[MultiTrack] Track {llm} failed: {e}")
        
        return results
    
    def _execute_single_track(
        self,
        query: str,
        llm: str,
        llm_provider: Callable,
        context: Optional[Dict]
    ) -> TrackResult:
        """Execute a single LLM track."""
        start_time = datetime.utcnow()
        track_id = f"{llm}-{hashlib.md5(query[:50].encode()).hexdigest()[:8]}"
        
        try:
            # Get LLM response
            response = llm_provider(query, llm)
            
            # Evaluate with BAIS if evaluator provided
            if self.bais_evaluator:
                evaluation = self.bais_evaluator(response)
                bais_score = evaluation.get('score', 0.5)
                issues = evaluation.get('issues', [])
                evidence = evaluation.get('evidence_strength', 'medium')
                confidence = evaluation.get('confidence', 0.5)
            else:
                # Default evaluation
                bais_score = 0.5
                issues = []
                evidence = 'medium'
                confidence = 0.5
            
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return TrackResult(
                track_id=track_id,
                llm_provider=llm,
                response=response,
                bais_score=bais_score,
                issues_detected=issues,
                evidence_strength=evidence,
                latency_ms=latency,
                tokens_used=len(response.split()),  # Approximate
                confidence=confidence,
                timestamp=start_time.isoformat()
            )
            
        except Exception as e:
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            return TrackResult(
                track_id=track_id,
                llm_provider=llm,
                response=f"ERROR: {str(e)}",
                bais_score=0.0,
                issues_detected=[{'type': 'execution_error', 'detail': str(e)}],
                evidence_strength='none',
                latency_ms=latency,
                tokens_used=0,
                confidence=0.0,
                timestamp=start_time.isoformat()
            )
    
    def get_consensus(
        self,
        track_results: List[TrackResult],
        strategy: ConsolidationStrategy = ConsolidationStrategy.BEST_SINGLE
    ) -> ConsensusResult:
        """Get consensus from track results."""
        self._stats['consensus_used'] += 1
        return self.consensus_engine.generate_consensus(track_results, strategy)
    
    def user_select_track(
        self,
        track_results: List[TrackResult],
        selected_track_id: str
    ) -> Optional[TrackResult]:
        """User selects a specific track."""
        self._stats['user_selections'] += 1
        for track in track_results:
            if track.track_id == selected_track_id:
                return track
        return None
    
    def orchestrate(
        self,
        query: str,
        llm_provider: Callable[[str, str], str],
        user_llm_selection: Optional[List[str]] = None,
        auto_select_best: bool = True,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Full orchestration: suggest, execute, consensus.
        
        Args:
            query: User query
            llm_provider: Function(query, llm_name) -> response
            user_llm_selection: User's LLM preferences (overrides suggestion)
            auto_select_best: Auto-select consensus or return all for user
            context: Additional context
        
        Returns:
            {
                'suggestion': TrackSuggestion,
                'track_results': List[TrackResult],
                'consensus': ConsensusResult (if auto_select_best),
                'user_selection_required': bool,
                'recommended_response': str
            }
        """
        # Get suggestion
        suggestion = self.suggest_tracks(query, context)
        
        # Determine LLMs to use
        llms_to_use = user_llm_selection or suggestion.suggested_llms
        
        # Execute tracks
        track_results = self.execute_tracks(query, llms_to_use, llm_provider, context)
        
        # Generate result
        result = {
            'suggestion': {
                'suggested_count': suggestion.suggested_track_count,
                'suggested_llms': suggestion.suggested_llms,
                'reason': suggestion.reason,
                'complexity': suggestion.complexity.value,
                'risk_level': suggestion.risk_level
            },
            'track_results': [
                {
                    'track_id': t.track_id,
                    'llm': t.llm_provider,
                    'bais_score': t.bais_score,
                    'issues_count': len(t.issues_detected),
                    'latency_ms': t.latency_ms,
                    'confidence': t.confidence,
                    'response_preview': t.response[:100]
                }
                for t in track_results
            ]
        }
        
        if auto_select_best:
            consensus = self.get_consensus(track_results)
            result['consensus'] = {
                'recommended_response': consensus.recommended_response,
                'reason': consensus.recommendation_reason,
                'agreement_score': consensus.agreement_score,
                'divergence_areas': consensus.divergence_areas,
                'primary_track': consensus.primary_track,
                'confidence': consensus.consensus_confidence
            }
            result['recommended_response'] = consensus.recommended_response
            result['user_selection_required'] = False
        else:
            result['user_selection_required'] = True
            result['available_tracks'] = [t.track_id for t in track_results]
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestration statistics."""
        return {
            **self._stats,
            'learning_params': getattr(self, '_learning_params', {}),
            'history_size': len(getattr(self, '_outcome_history', []))
        }
    
    # =========================================================================
    # Learning Interface (5/5 methods)
    # =========================================================================
    
    def record_outcome(self, outcome: Dict[str, Any]) -> None:
        """Record orchestration outcome for learning."""
        if not hasattr(self, '_outcome_history'):
            self._outcome_history: List[Dict] = []
        
        self._stats['total_orchestrations'] += 1
        if outcome.get('consensus_used', False):
            self._stats['consensus_used'] += 1
        
        self._outcome_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            **outcome
        })
        if len(self._outcome_history) > 1000:
            self._outcome_history = self._outcome_history[-500:]
    
    def learn_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """Learn from feedback to improve track selection."""
        if not hasattr(self, '_learning_params'):
            self._learning_params = {'track_preference': {}, 'consensus_weight': 0.5}
        
        was_correct = feedback.get('was_correct', True)
        selected_track = feedback.get('selected_track')
        
        if selected_track and was_correct:
            # Boost preference for this track's LLM
            llm = feedback.get('llm_provider', 'unknown')
            current = self._learning_params['track_preference'].get(llm, 0.5)
            self._learning_params['track_preference'][llm] = min(1.0, current + 0.05)
        elif not was_correct and selected_track:
            llm = feedback.get('llm_provider', 'unknown')
            current = self._learning_params['track_preference'].get(llm, 0.5)
            self._learning_params['track_preference'][llm] = max(0.1, current - 0.05)
    
    def serialize_state(self) -> Dict[str, Any]:
        """Serialize state for persistence."""
        return {
            'outcome_history': getattr(self, '_outcome_history', [])[-100:],
            'learning_params': getattr(self, '_learning_params', {}),
            'stats': self._stats.copy(),
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict[str, Any]) -> None:
        """Restore state from serialized data."""
        self._outcome_history = state.get('outcome_history', [])
        self._learning_params = state.get('learning_params', {'track_preference': {}, 'consensus_weight': 0.5})
        self._stats.update(state.get('stats', {}))


# =============================================================================
# Factory
# =============================================================================

def create_multi_track_orchestrator(
    available_llms: Optional[List[str]] = None,
    **kwargs
) -> MultiTrackOrchestrator:
    """Factory function."""
    return MultiTrackOrchestrator(available_llms=available_llms, **kwargs)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MULTI-TRACK ORCHESTRATOR TEST")
    print("=" * 80)
    
    orchestrator = MultiTrackOrchestrator()
    
    # Mock LLM provider
    def mock_llm_provider(query: str, llm: str) -> str:
        responses = {
            "grok": f"[Grok] Response to: {query[:30]}...",
            "openai": f"[OpenAI] Response to: {query[:30]}...",
            "anthropic": f"[Anthropic] Response to: {query[:30]}..."
        }
        return responses.get(llm, f"[{llm}] Response")
    
    # Test 1: Simple query
    print("\n[1] Testing simple query (low complexity)...")
    result1 = orchestrator.orchestrate(
        query="Fix the typo in this code",
        llm_provider=mock_llm_provider,
        auto_select_best=True
    )
    print(f"    Suggested tracks: {result1['suggestion']['suggested_count']}")
    print(f"    Reason: {result1['suggestion']['reason']}")
    print(f"    Tracks executed: {len(result1['track_results'])}")
    
    # Test 2: Complex medical query
    print("\n[2] Testing complex medical query (high risk)...")
    result2 = orchestrator.orchestrate(
        query="Design a diagnostic system for detecting cardiac conditions from ECG data",
        llm_provider=mock_llm_provider,
        auto_select_best=True
    )
    print(f"    Suggested tracks: {result2['suggestion']['suggested_count']}")
    print(f"    Risk level: {result2['suggestion']['risk_level']}")
    print(f"    Complexity: {result2['suggestion']['complexity']}")
    print(f"    Tracks executed: {len(result2['track_results'])}")
    if 'consensus' in result2:
        print(f"    Agreement score: {result2['consensus']['agreement_score']:.2f}")
    
    # Test 3: User override
    print("\n[3] Testing user LLM override...")
    result3 = orchestrator.orchestrate(
        query="Build a simple REST API",
        llm_provider=mock_llm_provider,
        user_llm_selection=["grok", "openai"],  # User overrides
        auto_select_best=False  # User wants to select
    )
    print(f"    User selection required: {result3['user_selection_required']}")
    print(f"    Available tracks: {result3.get('available_tracks', [])}")
    
    # Statistics
    print("\n[4] Statistics:")
    stats = orchestrator.get_statistics()
    for key, value in stats.items():
        print(f"    {key}: {value}")
    
    print("\n" + "=" * 80)
    print("✓ MULTI-TRACK ORCHESTRATOR TEST COMPLETE")
    print("=" * 80)

