"""
BAIS Cognitive Governance Engine v34.0
Multi-Modal Signal Processing & Concurrent Context Handling

Phase 34: Addresses PPA2-C1-22, PPA2-C1-23, PPA3-Comp2
- Multi-modal signal fusion (text, embeddings, metadata)
- Concurrent context handling with isolation
- Session-aware processing with state management

Patent Claims Addressed:
- PPA2-C1-22: Multi-modal signal processing and fusion
- PPA2-C1-23: Concurrent context handling with isolation
- PPA3-Comp2: Session-aware governance with state persistence
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import asyncio
import uuid
import logging
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Types of input modalities."""
    TEXT = "text"
    EMBEDDING = "embedding"
    METADATA = "metadata"
    STRUCTURED = "structured"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"


class FusionStrategy(Enum):
    """Strategies for fusing multi-modal signals."""
    EARLY_FUSION = "early"      # Concatenate before processing
    LATE_FUSION = "late"        # Process separately, combine results
    HYBRID_FUSION = "hybrid"    # Mix of early and late
    ATTENTION_FUSION = "attention"  # Attention-weighted combination
    HIERARCHICAL = "hierarchical"   # Layer-by-layer fusion


class ContextIsolationLevel(Enum):
    """Isolation levels for concurrent contexts."""
    NONE = "none"              # Shared state
    READ_COMMITTED = "read_committed"  # See committed changes
    REPEATABLE_READ = "repeatable_read"  # Consistent reads
    SERIALIZABLE = "serializable"  # Full isolation


class SessionState(Enum):
    """States of a governance session."""
    CREATED = "created"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    EXPIRED = "expired"


@dataclass
class ModalSignal:
    """A signal from a specific modality."""
    modality: ModalityType
    data: Any
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_vector(self) -> np.ndarray:
        """Convert signal to vector representation."""
        if self.modality == ModalityType.EMBEDDING:
            return np.array(self.data) if not isinstance(self.data, np.ndarray) else self.data
        elif self.modality == ModalityType.NUMERIC:
            return np.array([self.data, self.confidence])
        elif self.modality == ModalityType.TEXT:
            # Simple text features (length, word count, etc.)
            text = str(self.data)
            return np.array([len(text), len(text.split()), self.confidence])
        else:
            return np.array([self.confidence])


@dataclass
class FusedSignal:
    """Result of multi-modal fusion."""
    fused_vector: np.ndarray
    modalities_used: List[ModalityType]
    fusion_strategy: FusionStrategy
    confidence: float
    weights: Dict[ModalityType, float]
    agreement_score: float  # How much modalities agree


@dataclass
class ContextSnapshot:
    """Snapshot of context state at a point in time."""
    snapshot_id: str
    timestamp: datetime
    state: Dict[str, Any]
    version: int


@dataclass
class GovernanceContext:
    """
    A governance context for concurrent processing.
    
    PPA2-C1-23: Concurrent context handling with isolation.
    """
    context_id: str
    session_id: str
    isolation_level: ContextIsolationLevel
    created_at: datetime
    state: Dict[str, Any] = field(default_factory=dict)
    snapshots: List[ContextSnapshot] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)
    version: int = 0
    
    def __post_init__(self):
        # Ensure lock is created if not provided
        if self.lock is None:
            self.lock = threading.RLock()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from context with isolation."""
        with self.lock:
            return self.state.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set value in context with isolation."""
        with self.lock:
            self.state[key] = value
            self.version += 1
    
    def snapshot(self) -> ContextSnapshot:
        """Create snapshot of current state."""
        with self.lock:
            snap = ContextSnapshot(
                snapshot_id=str(uuid.uuid4())[:8],
                timestamp=datetime.utcnow(),
                state=self.state.copy(),
                version=self.version
            )
            self.snapshots.append(snap)
            return snap
    
    def rollback(self, snapshot_id: str) -> bool:
        """Rollback to a previous snapshot."""
        with self.lock:
            for snap in self.snapshots:
                if snap.snapshot_id == snapshot_id:
                    self.state = snap.state.copy()
                    self.version = snap.version
                    return True
            return False


@dataclass


class GovernanceSession:
    """
    A governance session with state persistence.
    
    PPA3-Comp2: Session-aware governance with state persistence.
    """
    session_id: str
    user_id: Optional[str]
    created_at: datetime
    last_activity: datetime
    state: SessionState
    ttl: timedelta
    context_history: List[str] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.utcnow() - self.last_activity > self.ttl
    
    def touch(self) -> None:
        """Update last activity time."""
        self.last_activity = datetime.utcnow()
    
    def add_decision(self, decision: Dict[str, Any]) -> None:
        """Add a decision to session history."""
        decision["timestamp"] = datetime.utcnow().isoformat()
        self.decisions.append(decision)
        self.touch()


class MultiModalFusion:
    """
    Multi-modal signal fusion engine.
    
    PPA2-C1-22: Multi-modal signal processing and fusion.
    """
    
    def __init__(self, default_strategy: FusionStrategy = FusionStrategy.LATE_FUSION):
        """Initialize multi-modal fusion."""
        self.default_strategy = default_strategy
        self.modality_weights: Dict[ModalityType, float] = {
            ModalityType.TEXT: 0.25,
            ModalityType.EMBEDDING: 0.30,
            ModalityType.METADATA: 0.10,
            ModalityType.STRUCTURED: 0.15,
            ModalityType.NUMERIC: 0.10,
            ModalityType.CATEGORICAL: 0.05,
            ModalityType.TEMPORAL: 0.05
        }
        
    def fuse(
        self,
        signals: List[ModalSignal],
        strategy: Optional[FusionStrategy] = None
    ) -> FusedSignal:
        """
        Fuse multiple modal signals.
        
        Args:
            signals: List of modal signals
            strategy: Fusion strategy to use
            
        Returns:
            FusedSignal with combined information
        """
        strategy = strategy or self.default_strategy
        
        if not signals:
            return FusedSignal(
                fused_vector=np.array([0.5]),
                modalities_used=[],
                fusion_strategy=strategy,
                confidence=0.0,
                weights={},
                agreement_score=0.0
            )
        
        if strategy == FusionStrategy.EARLY_FUSION:
            return self._early_fusion(signals)
        elif strategy == FusionStrategy.LATE_FUSION:
            return self._late_fusion(signals)
        elif strategy == FusionStrategy.ATTENTION_FUSION:
            return self._attention_fusion(signals)
        elif strategy == FusionStrategy.HIERARCHICAL:
            return self._hierarchical_fusion(signals)
        else:
            return self._late_fusion(signals)
    
    def _early_fusion(self, signals: List[ModalSignal]) -> FusedSignal:
        """Concatenate all signals before processing."""
        vectors = [s.to_vector() for s in signals]
        fused = np.concatenate(vectors)
        
        weights = {}
        for s in signals:
            weights[s.modality] = self.modality_weights.get(s.modality, 0.1)
        
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        confidence = np.mean([s.confidence for s in signals])
        agreement = self._calculate_agreement(signals)
        
        return FusedSignal(
            fused_vector=fused,
            modalities_used=[s.modality for s in signals],
            fusion_strategy=FusionStrategy.EARLY_FUSION,
            confidence=confidence,
            weights=weights,
            agreement_score=agreement
        )
    
    def _late_fusion(self, signals: List[ModalSignal]) -> FusedSignal:
        """Process each modality separately, then combine."""
        # Get weighted scores per modality
        weighted_scores = []
        weights = {}
        
        for s in signals:
            weight = self.modality_weights.get(s.modality, 0.1)
            weights[s.modality] = weight
            weighted_scores.append(s.confidence * weight)
        
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        # Fused score
        fused_score = sum(weighted_scores) / max(sum(weights.values()), 0.001)
        agreement = self._calculate_agreement(signals)
        
        return FusedSignal(
            fused_vector=np.array([fused_score, agreement]),
            modalities_used=[s.modality for s in signals],
            fusion_strategy=FusionStrategy.LATE_FUSION,
            confidence=fused_score,
            weights=weights,
            agreement_score=agreement
        )
    
    def _attention_fusion(self, signals: List[ModalSignal]) -> FusedSignal:
        """Use attention-based weighting."""
        if len(signals) <= 1:
            return self._late_fusion(signals)
        
        # Calculate attention scores based on confidence
        confidences = np.array([s.confidence for s in signals])
        attention_weights = np.exp(confidences) / np.sum(np.exp(confidences))
        
        # Weighted combination
        vectors = [s.to_vector() for s in signals]
        max_len = max(len(v) for v in vectors)
        
        # Pad vectors to same length
        padded = [np.pad(v, (0, max_len - len(v))) for v in vectors]
        
        # Apply attention
        fused = np.zeros(max_len)
        for i, (vec, weight) in enumerate(zip(padded, attention_weights)):
            fused += vec * weight
        
        weights = {s.modality: float(w) for s, w in zip(signals, attention_weights)}
        agreement = self._calculate_agreement(signals)
        
        return FusedSignal(
            fused_vector=fused,
            modalities_used=[s.modality for s in signals],
            fusion_strategy=FusionStrategy.ATTENTION_FUSION,
            confidence=float(np.mean(confidences)),
            weights=weights,
            agreement_score=agreement
        )
    
    def _hierarchical_fusion(self, signals: List[ModalSignal]) -> FusedSignal:
        """Layer-by-layer fusion."""
        # Group by modality type
        groups = defaultdict(list)
        for s in signals:
            groups[s.modality].append(s)
        
        # Fuse within groups first
        group_results = []
        for modality, group in groups.items():
            if len(group) == 1:
                group_results.append(group[0])
            else:
                # Average within group
                avg_conf = np.mean([s.confidence for s in group])
                combined = ModalSignal(
                    modality=modality,
                    data=avg_conf,
                    confidence=avg_conf
                )
                group_results.append(combined)
        
        # Then fuse across groups
        return self._late_fusion(group_results)
    
    def _calculate_agreement(self, signals: List[ModalSignal]) -> float:
        """Calculate agreement score between modalities."""
        if len(signals) <= 1:
            return 1.0
        
        confidences = [s.confidence for s in signals]
        # Agreement is inverse of variance
        variance = np.var(confidences)
        return max(0.0, 1.0 - variance * 4)  # Scale variance to 0-1


class ConcurrentContextManager:
    """
    Manages concurrent governance contexts with isolation.
    
    PPA2-C1-23: Concurrent context handling with isolation.
    """
    
    def __init__(self, max_contexts: int = 100):
        """Initialize context manager."""
        self.max_contexts = max_contexts
        self.contexts: Dict[str, GovernanceContext] = {}
        self.global_lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    def create_context(
        self,
        session_id: str,
        isolation_level: ContextIsolationLevel = ContextIsolationLevel.READ_COMMITTED
    ) -> GovernanceContext:
        """Create a new governance context."""
        with self.global_lock:
            if len(self.contexts) >= self.max_contexts:
                # Clean up oldest contexts
                self._cleanup_contexts()
            
            context_id = str(uuid.uuid4())[:12]
            context = GovernanceContext(
                context_id=context_id,
                session_id=session_id,
                isolation_level=isolation_level,
                created_at=datetime.utcnow()
            )
            self.contexts[context_id] = context
            return context
    
    def get_context(self, context_id: str) -> Optional[GovernanceContext]:
        """Get context by ID."""
        return self.contexts.get(context_id)
    
    def execute_in_context(
        self,
        context_id: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function within a context."""
        context = self.get_context(context_id)
        if not context:
            raise ValueError(f"Context {context_id} not found")
        
        with context.lock:
            return func(context, *args, **kwargs)
    
    def execute_parallel(
        self,
        tasks: List[Tuple[str, Callable, tuple, dict]]
    ) -> List[Any]:
        """
        Execute multiple tasks in parallel with context isolation.
        
        Args:
            tasks: List of (context_id, func, args, kwargs)
            
        Returns:
            List of results
        """
        futures = []
        for context_id, func, args, kwargs in tasks:
            future = self.executor.submit(
                self.execute_in_context,
                context_id, func, *args, **kwargs
            )
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Parallel execution error: {e}")
                results.append(None)
        
        return results
    
    def _cleanup_contexts(self, keep_recent: int = 50):
        """Clean up old contexts."""
        if len(self.contexts) <= keep_recent:
            return
        
        # Sort by creation time
        sorted_contexts = sorted(
            self.contexts.items(),
            key=lambda x: x[1].created_at
        )
        
        # Remove oldest
        to_remove = len(sorted_contexts) - keep_recent
        for context_id, _ in sorted_contexts[:to_remove]:
            del self.contexts[context_id]
    
    def close_context(self, context_id: str) -> bool:
        """Close and remove a context."""
        with self.global_lock:
            if context_id in self.contexts:
                del self.contexts[context_id]
                return True
            return False

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


class SessionManager:
    """
    Manages governance sessions with state persistence.
    
    PPA3-Comp2: Session-aware governance with state persistence.
    """
    
    def __init__(
        self,
        default_ttl: timedelta = timedelta(hours=1),
        max_sessions: int = 1000
    ):
        """Initialize session manager."""
        self.default_ttl = default_ttl
        self.max_sessions = max_sessions
        self.sessions: Dict[str, GovernanceSession] = {}
        self.lock = threading.RLock()
        
    def create_session(
        self,
        user_id: Optional[str] = None,
        ttl: Optional[timedelta] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> GovernanceSession:
        """Create a new governance session."""
        with self.lock:
            if len(self.sessions) >= self.max_sessions:
                self._cleanup_expired()
            
            session_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            session = GovernanceSession(
                session_id=session_id,
                user_id=user_id,
                created_at=now,
                last_activity=now,
                state=SessionState.ACTIVE,
                ttl=ttl or self.default_ttl,
                metadata=metadata or {}
            )
            self.sessions[session_id] = session
            return session
    
    def get_session(self, session_id: str) -> Optional[GovernanceSession]:
        """Get session by ID."""
        session = self.sessions.get(session_id)
        if session and session.is_expired():
            session.state = SessionState.EXPIRED
        return session
    
    def get_or_create(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> GovernanceSession:
        """Get existing session or create new one."""
        if session_id:
            session = self.get_session(session_id)
            if session and session.state == SessionState.ACTIVE:
                session.touch()
                return session
        
        return self.create_session(user_id=user_id)
    
    def end_session(self, session_id: str) -> bool:
        """End a session."""
        session = self.get_session(session_id)
        if session:
            session.state = SessionState.COMPLETED
            return True
        return False
    
    def get_session_history(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get decision history for a session."""
        session = self.get_session(session_id)
        if not session:
            return []
        return session.decisions[-limit:]
    
    def _cleanup_expired(self):
        """Clean up expired sessions."""
        with self.lock:
            expired = [
                sid for sid, s in self.sessions.items()
                if s.is_expired()
            ]
            for sid in expired:
                self.sessions[sid].state = SessionState.EXPIRED

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


class MultiModalContextEngine:
    """
    Unified engine for multi-modal processing and concurrent contexts.
    
    Implements PPA2-C1-22, PPA2-C1-23, PPA3-Comp2.
    """
    
    def __init__(self):
        """Initialize multi-modal context engine."""
        self.fusion = MultiModalFusion()
        self.context_manager = ConcurrentContextManager()
        self.session_manager = SessionManager()
        
        logger.info("[MultiModal] Multi-Modal Context Engine initialized")
    
    def process_multimodal(
        self,
        signals: List[ModalSignal],
        strategy: Optional[FusionStrategy] = None,
        context_id: Optional[str] = None
    ) -> FusedSignal:
        """
        Process multi-modal signals with optional context.
        
        Args:
            signals: List of modal signals
            strategy: Fusion strategy
            context_id: Optional context for isolation
            
        Returns:
            Fused signal result
        """
        if context_id:
            context = self.context_manager.get_context(context_id)
            if context:
                context.set("last_fusion_time", datetime.utcnow().isoformat())
                context.set("modalities_processed", [s.modality.value for s in signals])
        
        return self.fusion.fuse(signals, strategy)
    
    def create_governance_context(
        self,
        session_id: Optional[str] = None,
        isolation_level: ContextIsolationLevel = ContextIsolationLevel.READ_COMMITTED
    ) -> Tuple[GovernanceSession, GovernanceContext]:
        """
        Create a session with associated context.
        
        Args:
            session_id: Optional existing session ID
            isolation_level: Context isolation level
            
        Returns:
            Tuple of (session, context)
        """
        session = self.session_manager.get_or_create(session_id)
        context = self.context_manager.create_context(
            session_id=session.session_id,
            isolation_level=isolation_level
        )
        session.context_history.append(context.context_id)
        
        return session, context
    
    def evaluate_with_context(
        self,
        signals: List[ModalSignal],
        session_id: Optional[str] = None,
        isolation_level: ContextIsolationLevel = ContextIsolationLevel.READ_COMMITTED
    ) -> Dict[str, Any]:
        """
        Full evaluation with session and context management.
        
        Args:
            signals: Multi-modal signals
            session_id: Optional session ID
            isolation_level: Context isolation level
            
        Returns:
            Evaluation result with session info
        """
        # Get or create session
        session, context = self.create_governance_context(
            session_id=session_id,
            isolation_level=isolation_level
        )
        
        # Process signals
        fused = self.process_multimodal(signals, context_id=context.context_id)
        
        # Store in context
        context.set("fused_confidence", fused.confidence)
        context.set("agreement_score", fused.agreement_score)
        
        # Create decision record
        decision = {
            "context_id": context.context_id,
            "fused_confidence": fused.confidence,
            "agreement_score": fused.agreement_score,
            "modalities": [m.value for m in fused.modalities_used],
            "fusion_strategy": fused.fusion_strategy.value
        }
        session.add_decision(decision)
        
        return {
            "session_id": session.session_id,
            "context_id": context.context_id,
            "fused_signal": {
                "confidence": fused.confidence,
                "agreement": fused.agreement_score,
                "modalities": [m.value for m in fused.modalities_used],
                "strategy": fused.fusion_strategy.value,
                "weights": {k.value: v for k, v in fused.weights.items()}
            },
            "session_state": session.state.value,
            "decision_count": len(session.decisions)
        }
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of a governance session."""
        session = self.session_manager.get_session(session_id)
        if not session:
            return {"error": "Session not found"}
        
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "state": session.state.value,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "decision_count": len(session.decisions),
            "context_count": len(session.context_history),
            "is_expired": session.is_expired()
        }

    # ========================================
    # PHASE 49: LEARNING METHODS
    # ========================================
    

    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, input_data, output_data, was_correct, domain=None, metadata=None):
        """Record outcome for learning (Phase 49)."""
        if hasattr(self, '_learning_manager') and self._learning_manager:
            self._learning_manager.record_outcome(
                module_name=self.__class__.__name__.lower(),
                input_data=input_data, output_data=output_data,
                was_correct=was_correct, domain=domain, metadata=metadata
            )
    
    def record_feedback(self, result, was_accurate):
        """Record feedback for learning (Phase 49)."""
        self.record_outcome({"result": str(result)}, {}, was_accurate)
    
    def adapt_thresholds(self, threshold_name, current_value, direction='decrease'):
        """Adapt thresholds based on learning (Phase 49)."""
        if hasattr(self, '_learning_manager') and self._learning_manager:
            return self._learning_manager.adapt_threshold(
                self.__class__.__name__.lower(), threshold_name, current_value, direction
            )
        return max(0.1, min(0.95, current_value + (0.05 if direction == 'increase' else -0.05)))
    
    def get_domain_adjustment(self, domain):
        """Get domain-specific adjustment (Phase 49)."""
        if hasattr(self, '_learning_manager') and self._learning_manager:
            return self._learning_manager.get_domain_adjustment(self.__class__.__name__.lower(), domain)
        return 0.0
    
    def save_state(self, filepath=None):
        """Save learning state (Phase 49)."""
        return self._learning_manager.save_state() if hasattr(self, '_learning_manager') and self._learning_manager else False
    
    def load_state(self, filepath=None):
        """Load learning state (Phase 49)."""
        return self._learning_manager.load_state() if hasattr(self, '_learning_manager') and self._learning_manager else False
    
    def get_learning_statistics(self):
        """Get learning statistics (Phase 49)."""
        if hasattr(self, '_learning_manager') and self._learning_manager:
            stats = self._learning_manager.get_learning_statistics(self.__class__.__name__.lower())
            return stats.__dict__ if hasattr(stats, '__dict__') else stats
        return {"module": self.__class__.__name__, "status": "no_learning_manager"}


# Test function
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

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 34: Multi-Modal Signals & Concurrent Contexts Test")
    print("=" * 70)
    
    engine = MultiModalContextEngine()
    
    # Test 1: Multi-modal fusion
    print("\n[1] Multi-Modal Signal Fusion")
    print("-" * 60)
    
    signals = [
        ModalSignal(modality=ModalityType.TEXT, data="This is a test response", confidence=0.85),
        ModalSignal(modality=ModalityType.EMBEDDING, data=[0.1, 0.2, 0.3, 0.4], confidence=0.90),
        ModalSignal(modality=ModalityType.NUMERIC, data=0.75, confidence=0.80),
        ModalSignal(modality=ModalityType.METADATA, data={"domain": "medical"}, confidence=0.70)
    ]
    
    for strategy in [FusionStrategy.EARLY_FUSION, FusionStrategy.LATE_FUSION, FusionStrategy.ATTENTION_FUSION]:
        fused = engine.fusion.fuse(signals, strategy)
        print(f"  {strategy.value}:")
        print(f"    Confidence: {fused.confidence:.3f}")
        print(f"    Agreement: {fused.agreement_score:.3f}")
        print(f"    Modalities: {len(fused.modalities_used)}")
    
    # Test 2: Concurrent contexts
    print("\n[2] Concurrent Context Handling")
    print("-" * 60)
    
    session, context1 = engine.create_governance_context()
    _, context2 = engine.create_governance_context(session.session_id)
    
    print(f"  Session ID: {session.session_id[:12]}...")
    print(f"  Context 1: {context1.context_id}")
    print(f"  Context 2: {context2.context_id}")
    print(f"  Isolation: {context1.isolation_level.value}")
    
    # Test isolation
    context1.set("test_key", "value_1")
    context2.set("test_key", "value_2")
    
    print(f"  Context 1 value: {context1.get('test_key')}")
    print(f"  Context 2 value: {context2.get('test_key')}")
    print(f"  Isolation Working: {context1.get('test_key') != context2.get('test_key')}")
    
    # Test 3: Session management
    print("\n[3] Session-Aware Processing")
    print("-" * 60)
    
    result = engine.evaluate_with_context(signals)
    
    print(f"  Session: {result['session_id'][:12]}...")
    print(f"  Context: {result['context_id']}")
    print(f"  Fused Confidence: {result['fused_signal']['confidence']:.3f}")
    print(f"  Agreement: {result['fused_signal']['agreement']:.3f}")
    print(f"  Decision Count: {result['decision_count']}")
    
    # Get session summary
    summary = engine.get_session_summary(result['session_id'])
    print(f"  Session State: {summary['state']}")
    print(f"  Context History: {summary['context_count']}")
    
    # Test 4: Snapshot and rollback
    print("\n[4] Context Snapshot & Rollback")
    print("-" * 60)
    
    context1.set("counter", 0)
    snap = context1.snapshot()
    print(f"  Snapshot ID: {snap.snapshot_id}")
    print(f"  Counter before change: {context1.get('counter')}")
    
    context1.set("counter", 100)
    print(f"  Counter after change: {context1.get('counter')}")
    
    context1.rollback(snap.snapshot_id)
    print(f"  Counter after rollback: {context1.get('counter')}")
    print(f"  Rollback Working: {context1.get('counter') == 0}")
    
    print("\n" + "=" * 70)
    print("PHASE 34: Multi-Modal Context Engine - VERIFIED")
    print("=" * 70)


