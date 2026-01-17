"""
BASE Cognitive Governance Engine v16.0
Outcome Memory - Persistent Decision Database

Stores all governance decisions and their outcomes for:
1. Learning from past decisions
2. Finding similar cases
3. Tracking accuracy over time
4. Providing audit trail
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
import numpy as np
from contextlib import contextmanager
import threading


@dataclass
class DecisionRecord:
    """Complete record of a governance decision."""
    id: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Query info
    query: str = ""
    query_embedding: Optional[bytes] = None
    domain: str = "general"
    
    # Context
    context_features: Dict = field(default_factory=dict)
    risk_level: float = 0.0
    complexity: float = 0.0
    
    # Signals
    grounding_score: float = 0.0
    temporal_score: float = 0.0
    behavioral_score: float = 0.0
    factual_score: float = 0.0
    fused_score: float = 0.0
    
    # Decision
    accuracy: float = 0.0
    threshold_used: float = 50.0
    was_accepted: bool = True
    pathway: str = "VERIFIED"
    
    # Outcome (may be updated later)
    was_correct: Optional[bool] = None
    feedback: Optional[str] = None
    feedback_timestamp: Optional[datetime] = None
    
    # Metadata
    response_length: int = 0
    processing_time_ms: float = 0.0
    algorithm_used: str = "oco"
    version: str = "16.0.0"
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'query': self.query[:500],  # Truncate for display
            'domain': self.domain,
            'context_features': self.context_features,
            'risk_level': self.risk_level,
            'complexity': self.complexity,
            'grounding_score': self.grounding_score,
            'temporal_score': self.temporal_score,
            'behavioral_score': self.behavioral_score,
            'factual_score': self.factual_score,
            'fused_score': self.fused_score,
            'accuracy': self.accuracy,
            'threshold_used': self.threshold_used,
            'was_accepted': self.was_accepted,
            'pathway': self.pathway,
            'was_correct': self.was_correct,
            'feedback': self.feedback,
            'processing_time_ms': self.processing_time_ms,
            'algorithm_used': self.algorithm_used
        }


class OutcomeMemory:
    """
    SQLite-backed persistent storage for governance decisions.
    
    Features:
    - Full decision history
    - Feedback integration
    - Similar case retrieval
    - Performance analytics
    - Thread-safe operations
    """
    
    def __init__(self, db_path: Path = None, data_dir: Path = None):
        # Accept either db_path or data_dir for compatibility
        # Use temp directory if no path provided (fixes read-only filesystem issues)
        if db_path is None and data_dir is None:
            import tempfile
            temp_dir = Path(tempfile.mkdtemp(prefix="base_outcomes_"))
            db_path = temp_dir / "outcomes.db"
        self.db_path = db_path or (data_dir / "outcomes.db" if data_dir else None)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._local = threading.local()
        self._init_database()
    
    @contextmanager
    def _get_connection(self):
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
        
        try:
            yield self._local.conn
        except Exception:
            self._local.conn.rollback()
            raise
    
    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                -- Main decisions table
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    
                    -- Query info
                    query TEXT,
                    query_embedding BLOB,
                    domain TEXT,
                    
                    -- Context
                    context_features TEXT,  -- JSON
                    risk_level REAL,
                    complexity REAL,
                    
                    -- Signals
                    grounding_score REAL,
                    temporal_score REAL,
                    behavioral_score REAL,
                    factual_score REAL,
                    fused_score REAL,
                    
                    -- Decision
                    accuracy REAL,
                    threshold_used REAL,
                    was_accepted INTEGER,
                    pathway TEXT,
                    
                    -- Outcome
                    was_correct INTEGER,
                    feedback TEXT,
                    feedback_timestamp TEXT,
                    
                    -- Metadata
                    response_length INTEGER,
                    processing_time_ms REAL,
                    algorithm_used TEXT,
                    version TEXT
                );
                
                -- Indices for common queries
                CREATE INDEX IF NOT EXISTS idx_timestamp ON decisions(timestamp);
                CREATE INDEX IF NOT EXISTS idx_domain ON decisions(domain);
                CREATE INDEX IF NOT EXISTS idx_was_correct ON decisions(was_correct);
                CREATE INDEX IF NOT EXISTS idx_was_accepted ON decisions(was_accepted);
                CREATE INDEX IF NOT EXISTS idx_accuracy ON decisions(accuracy);
                
                -- Aggregated statistics (for fast queries)
                CREATE TABLE IF NOT EXISTS statistics (
                    date TEXT PRIMARY KEY,
                    total_decisions INTEGER,
                    correct_decisions INTEGER,
                    false_positives INTEGER,
                    false_negatives INTEGER,
                    avg_accuracy REAL,
                    avg_threshold REAL,
                    domains TEXT,  -- JSON
                    pathways TEXT  -- JSON
                );
                
                -- Learning events (threshold updates, algorithm changes)
                CREATE TABLE IF NOT EXISTS learning_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT,  -- 'threshold_update', 'algorithm_switch', etc.
                    domain TEXT,
                    old_value REAL,
                    new_value REAL,
                    trigger TEXT,  -- What caused this update
                    metadata TEXT  -- JSON
                );
                
                CREATE INDEX IF NOT EXISTS idx_learning_timestamp ON learning_events(timestamp);
            """)
            conn.commit()
    
    def record_decision(self, record: DecisionRecord) -> int:
        """
        Record a new governance decision.
        
        Returns: The ID of the inserted record
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO decisions (
                    timestamp, query, query_embedding, domain,
                    context_features, risk_level, complexity,
                    grounding_score, temporal_score, behavioral_score, factual_score, fused_score,
                    accuracy, threshold_used, was_accepted, pathway,
                    was_correct, feedback, feedback_timestamp,
                    response_length, processing_time_ms, algorithm_used, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.timestamp.isoformat(),
                record.query,
                record.query_embedding,
                record.domain,
                json.dumps(record.context_features),
                record.risk_level,
                record.complexity,
                record.grounding_score,
                record.temporal_score,
                record.behavioral_score,
                record.factual_score,
                record.fused_score,
                record.accuracy,
                record.threshold_used,
                1 if record.was_accepted else 0,
                record.pathway,
                None if record.was_correct is None else (1 if record.was_correct else 0),
                record.feedback,
                record.feedback_timestamp.isoformat() if record.feedback_timestamp else None,
                record.response_length,
                record.processing_time_ms,
                record.algorithm_used,
                record.version
            ))
            conn.commit()
            return cursor.lastrowid
    
    def record_feedback(self, decision_id: int, was_correct: bool, feedback: str = None) -> bool:
        """
        Record feedback on a past decision.
        
        This is how the system learns from outcomes.
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                UPDATE decisions 
                SET was_correct = ?, feedback = ?, feedback_timestamp = ?
                WHERE id = ?
            """, (
                1 if was_correct else 0,
                feedback,
                datetime.utcnow().isoformat(),
                decision_id
            ))
            conn.commit()
            return cursor.rowcount > 0
    
    def get_decision(self, decision_id: int) -> Optional[DecisionRecord]:
        """Get a specific decision by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM decisions WHERE id = ?", (decision_id,)
            ).fetchone()
            
            if row:
                return self._row_to_record(row)
        return None
    
    def get_recent_decisions(self, 
                            limit: int = 100,
                            domain: str = None,
                            only_with_feedback: bool = False) -> List[DecisionRecord]:
        """Get recent decisions with optional filters."""
        query = "SELECT * FROM decisions WHERE 1=1"
        params = []
        
        if domain:
            query += " AND domain = ?"
            params.append(domain)
        
        if only_with_feedback:
            query += " AND was_correct IS NOT NULL"
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_record(row) for row in rows]
    
    def get_decisions_for_learning(self, 
                                   domain: str = None,
                                   since: datetime = None,
                                   limit: int = 1000) -> List[DecisionRecord]:
        """
        Get decisions with feedback for learning updates.
        
        Only returns decisions where we know the outcome.
        """
        query = "SELECT * FROM decisions WHERE was_correct IS NOT NULL"
        params = []
        
        if domain:
            query += " AND domain = ?"
            params.append(domain)
        
        if since:
            query += " AND timestamp > ?"
            params.append(since.isoformat())
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_record(row) for row in rows]
    
    def find_similar_cases(self, 
                          query_embedding: bytes,
                          domain: str = None,
                          limit: int = 10) -> List[Tuple[DecisionRecord, float]]:
        """
        Find similar past cases using embedding similarity.
        
        Note: For production, consider using a vector database like FAISS.
        This implementation does brute-force comparison.
        """
        # Get candidates
        query = """
            SELECT * FROM decisions 
            WHERE query_embedding IS NOT NULL
        """
        params = []
        
        if domain:
            query += " AND domain = ?"
            params.append(domain)
        
        query += " ORDER BY timestamp DESC LIMIT 500"
        
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        
        if not rows:
            return []
        
        # Compute similarities
        target_emb = np.frombuffer(query_embedding, dtype=np.float32)
        
        similarities = []
        for row in rows:
            if row['query_embedding']:
                candidate_emb = np.frombuffer(row['query_embedding'], dtype=np.float32)
                if len(candidate_emb) == len(target_emb):
                    sim = np.dot(target_emb, candidate_emb) / (
                        np.linalg.norm(target_emb) * np.linalg.norm(candidate_emb) + 1e-8
                    )
                    similarities.append((self._row_to_record(row), float(sim)))
        
        # Sort by similarity and return top
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]
    
    def get_accuracy_by_domain(self, 
                               since: datetime = None) -> Dict[str, Dict[str, float]]:
        """Get accuracy statistics broken down by domain."""
        query = """
            SELECT 
                domain,
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                SUM(CASE WHEN was_accepted = 1 AND was_correct = 0 THEN 1 ELSE 0 END) as false_positive,
                SUM(CASE WHEN was_accepted = 0 AND was_correct = 1 THEN 1 ELSE 0 END) as false_negative,
                AVG(accuracy) as avg_accuracy,
                AVG(threshold_used) as avg_threshold
            FROM decisions
            WHERE was_correct IS NOT NULL
        """
        params = []
        
        if since:
            query += " AND timestamp > ?"
            params.append(since.isoformat())
        
        query += " GROUP BY domain"
        
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        
        result = {}
        for row in rows:
            total = row['total']
            result[row['domain']] = {
                'total': total,
                'correct': row['correct'],
                'accuracy_rate': row['correct'] / total if total > 0 else 0,
                'false_positive_rate': row['false_positive'] / total if total > 0 else 0,
                'false_negative_rate': row['false_negative'] / total if total > 0 else 0,
                'avg_accuracy_score': row['avg_accuracy'],
                'avg_threshold': row['avg_threshold']
            }
        
        return result
    
    def get_accuracy_trend(self, 
                          domain: str = None,
                          days: int = 30,
                          granularity: str = 'day') -> List[Dict]:
        """Get accuracy trend over time."""
        if granularity == 'day':
            date_format = '%Y-%m-%d'
            interval = 1
        elif granularity == 'hour':
            date_format = '%Y-%m-%d %H:00'
            interval = 1/24
        else:
            date_format = '%Y-%m-%d'
            interval = 1
        
        since = datetime.utcnow() - timedelta(days=days)
        
        query = f"""
            SELECT 
                strftime('{date_format}', timestamp) as period,
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                AVG(accuracy) as avg_accuracy
            FROM decisions
            WHERE was_correct IS NOT NULL
            AND timestamp > ?
        """
        params = [since.isoformat()]
        
        if domain:
            query += " AND domain = ?"
            params.append(domain)
        
        query += f" GROUP BY strftime('{date_format}', timestamp) ORDER BY period"
        
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        
        return [
            {
                'period': row['period'],
                'total': row['total'],
                'correct': row['correct'],
                'accuracy_rate': row['correct'] / row['total'] if row['total'] > 0 else 0,
                'avg_accuracy_score': row['avg_accuracy']
            }
            for row in rows
        ]
    
    def record_learning_event(self,
                             event_type: str,
                             domain: str,
                             old_value: float,
                             new_value: float,
                             trigger: str,
                             metadata: Dict = None):
        """Record a learning event (threshold update, algorithm change, etc.)."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO learning_events 
                (timestamp, event_type, domain, old_value, new_value, trigger, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow().isoformat(),
                event_type,
                domain,
                old_value,
                new_value,
                trigger,
                json.dumps(metadata or {})
            ))
            conn.commit()
    
    def get_learning_history(self, 
                            domain: str = None,
                            event_type: str = None,
                            limit: int = 100) -> List[Dict]:
        """Get history of learning events."""
        query = "SELECT * FROM learning_events WHERE 1=1"
        params = []
        
        if domain:
            query += " AND domain = ?"
            params.append(domain)
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        
        return [
            {
                'id': row['id'],
                'timestamp': row['timestamp'],
                'event_type': row['event_type'],
                'domain': row['domain'],
                'old_value': row['old_value'],
                'new_value': row['new_value'],
                'trigger': row['trigger'],
                'metadata': json.loads(row['metadata']) if row['metadata'] else {}
            }
            for row in rows
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        with self._get_connection() as conn:
            # Overall stats
            overall = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN was_correct IS NOT NULL THEN 1 ELSE 0 END) as with_feedback,
                    SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                    AVG(accuracy) as avg_accuracy,
                    AVG(processing_time_ms) as avg_processing_time
                FROM decisions
            """).fetchone()
            
            # By domain
            by_domain = conn.execute("""
                SELECT domain, COUNT(*) as count
                FROM decisions
                GROUP BY domain
            """).fetchall()
            
            # By pathway
            by_pathway = conn.execute("""
                SELECT pathway, COUNT(*) as count
                FROM decisions
                GROUP BY pathway
            """).fetchall()
            
            # Recent trends
            recent = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct
                FROM decisions
                WHERE timestamp > datetime('now', '-7 days')
                AND was_correct IS NOT NULL
            """).fetchone()
        
        total = overall['total'] or 0
        with_feedback = overall['with_feedback'] or 0
        
        return {
            'total_decisions': total,
            'decisions_with_feedback': with_feedback,
            'feedback_rate': with_feedback / total if total > 0 else 0,
            'correct_decisions': overall['correct'] or 0,
            'accuracy_rate': (overall['correct'] or 0) / with_feedback if with_feedback > 0 else 0,
            'avg_accuracy_score': overall['avg_accuracy'] or 0,
            'avg_processing_time_ms': overall['avg_processing_time'] or 0,
            'by_domain': {row['domain']: row['count'] for row in by_domain},
            'by_pathway': {row['pathway']: row['count'] for row in by_pathway},
            'recent_7d': {
                'total': recent['total'],
                'correct': recent['correct'] or 0,
                'accuracy_rate': (recent['correct'] or 0) / recent['total'] if recent['total'] > 0 else 0
            }
        }
    
    def _row_to_record(self, row: sqlite3.Row) -> DecisionRecord:
        """Convert database row to DecisionRecord."""
        return DecisionRecord(
            id=row['id'],
            timestamp=datetime.fromisoformat(row['timestamp']) if row['timestamp'] else None,
            query=row['query'] or "",
            query_embedding=row['query_embedding'],
            domain=row['domain'] or "general",
            context_features=json.loads(row['context_features']) if row['context_features'] else {},
            risk_level=row['risk_level'] or 0.0,
            complexity=row['complexity'] or 0.0,
            grounding_score=row['grounding_score'] or 0.0,
            temporal_score=row['temporal_score'] or 0.0,
            behavioral_score=row['behavioral_score'] or 0.0,
            factual_score=row['factual_score'] or 0.0,
            fused_score=row['fused_score'] or 0.0,
            accuracy=row['accuracy'] or 0.0,
            threshold_used=row['threshold_used'] or 50.0,
            was_accepted=bool(row['was_accepted']),
            pathway=row['pathway'] or "VERIFIED",
            was_correct=None if row['was_correct'] is None else bool(row['was_correct']),
            feedback=row['feedback'],
            feedback_timestamp=datetime.fromisoformat(row['feedback_timestamp']) if row['feedback_timestamp'] else None,
            response_length=row['response_length'] or 0,
            processing_time_ms=row['processing_time_ms'] or 0.0,
            algorithm_used=row['algorithm_used'] or "oco",
            version=row['version'] or "16.0.0"
        )
    
    def cleanup_old_records(self, keep_days: int = 90):
        """Remove old records to manage database size."""
        cutoff = datetime.utcnow() - timedelta(days=keep_days)
        
        with self._get_connection() as conn:
            # Keep records with feedback longer
            conn.execute("""
                DELETE FROM decisions 
                WHERE timestamp < ? 
                AND was_correct IS NULL
            """, (cutoff.isoformat(),))
            
            # Very old records with feedback
            very_old = datetime.utcnow() - timedelta(days=keep_days * 2)
            conn.execute("""
                DELETE FROM decisions 
                WHERE timestamp < ?
            """, (very_old.isoformat(),))
            
            conn.execute("VACUUM")
            conn.commit()
    
    # ===== Learning Interface Methods =====
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict[str, Any]) -> None:
        """Record decision outcome for learning."""
        record = DecisionRecord(
            query=outcome.get('query', ''),
            domain=outcome.get('domain', 'general'),
            accuracy=outcome.get('accuracy', 0.0),
            was_accepted=outcome.get('was_accepted', True),
            was_correct=outcome.get('was_correct')
        )
        self.store(record)
    
    def record_feedback(self, feedback: Dict[str, Any]) -> None:
        """Record feedback on decisions."""
        decision_id = feedback.get('decision_id')
        if decision_id:
            self.update_outcome(
                decision_id,
                feedback.get('was_correct', True),
                feedback.get('feedback_text', '')
            )
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt thresholds based on stored outcomes."""
        pass
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get domain-specific adjustment based on historical performance."""
        recent = self.get_recent_by_domain(domain, hours=168)  # 7 days
        if not recent:
            return 0.0
        
        correct = sum(1 for r in recent if r.was_correct)
        total_with_feedback = sum(1 for r in recent if r.was_correct is not None)
        
        if total_with_feedback < 5:
            return 0.0
        
        accuracy = correct / total_with_feedback
        return (accuracy - 0.5) * 0.2  # Scale to -0.1 to 0.1
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning statistics from outcome memory."""
        stats = self.get_statistics()
        stats['type'] = 'outcome_memory'
        return stats

    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback."""
        pass  # Implement specific learning logic


    def serialize_state(self) -> Dict:
        """Serialize state."""
        return {'outcomes': getattr(self, '_outcomes', [])[-100:]}


    def deserialize_state(self, state: Dict) -> None:
        """Restore state."""
        self._outcomes = state.get('outcomes', [])

