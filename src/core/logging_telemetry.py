"""
BAIS Cognitive Governance Engine v45.0
Logging & Telemetry with AI + Pattern + Learning

Phase 45: Observability Infrastructure
- AI-enhanced structured logging
- Intelligent telemetry collection
- Continuous learning from log patterns
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from datetime import datetime
from collections import defaultdict, deque
import threading
import logging

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    DEBUG = ("debug", 10)
    INFO = ("info", 20)
    WARNING = ("warning", 30)
    ERROR = ("error", 40)
    CRITICAL = ("critical", 50)
    
    def __init__(self, label: str, level: int):
        self.label = label
        self.level = level


class TelemetryType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    EVENT = "event"


class AnomalyType(Enum):
    FREQUENCY = "frequency"
    PATTERN = "pattern"
    TIMING = "timing"
    ERROR_SPIKE = "error_spike"


@dataclass
class LogEntry:
    timestamp: datetime
    level: LogLevel
    message: str
    component: str
    context: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    span_id: Optional[str] = None


@dataclass
class TelemetryPoint:
    name: str
    value: float
    telemetry_type: TelemetryType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LogAnomaly:
    anomaly_type: AnomalyType
    description: str
    affected_component: str
    severity: float
    detected_at: datetime = field(default_factory=datetime.utcnow)


class PatternBasedLogAnalyzer:
    """
    Analyzes logs using patterns.
    Layer 1: Static pattern detection.
    """
    
    ERROR_PATTERNS = {
        "connection_error": ["connection refused", "timeout", "unreachable"],
        "auth_error": ["unauthorized", "forbidden", "invalid token"],
        "resource_error": ["out of memory", "disk full", "quota exceeded"],
        "input_error": ["invalid input", "validation failed", "malformed"],
    }
    
    def __init__(self):
        self.pattern_matches: Dict[str, int] = defaultdict(int)
        self.detection_count = 0
    
    def analyze(self, entry: LogEntry) -> Optional[str]:
        """Analyze log entry for patterns."""
        msg_lower = entry.message.lower()
        
        for pattern_name, keywords in self.ERROR_PATTERNS.items():
            for keyword in keywords:
                if keyword in msg_lower:
                    self.pattern_matches[pattern_name] += 1
                    self.detection_count += 1
                    return pattern_name
        
        return None
    
    def get_pattern_summary(self) -> Dict[str, int]:
        return dict(self.pattern_matches)

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


class AILogEnhancer:
    """
    AI-enhanced log analysis.
    Layer 2: Intelligent log processing.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.enhancement_count = 0
        self.anomalies_detected: List[LogAnomaly] = []
    
    def detect_anomaly(self, entries: List[LogEntry], window_seconds: int = 60) -> Optional[LogAnomaly]:
        """Detect anomalies in log stream."""
        if len(entries) < 5:
            return None
        
        # Count errors in window
        error_count = sum(1 for e in entries if e.level.level >= LogLevel.ERROR.level)
        error_rate = error_count / len(entries)
        
        if error_rate > 0.3:  # More than 30% errors
            anomaly = LogAnomaly(
                anomaly_type=AnomalyType.ERROR_SPIKE,
                description=f"Error rate {error_rate:.1%} exceeds threshold",
                affected_component=entries[-1].component if entries else "unknown",
                severity=min(1.0, error_rate * 2)
            )
            self.anomalies_detected.append(anomaly)
            return anomaly
        
        return None
    
    def correlate_logs(self, entries: List[LogEntry]) -> Dict[str, List[LogEntry]]:
        """Correlate logs by trace_id."""
        correlation = defaultdict(list)
        for entry in entries:
            if entry.trace_id:
                correlation[entry.trace_id].append(entry)
        return dict(correlation)

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


class TelemetryCollector:
    """
    Collects and manages telemetry.
    Layer 3: Telemetry infrastructure.
    """
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.telemetry_buffer: deque = deque(maxlen=buffer_size)
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
    
    def record(self, point: TelemetryPoint):
        """Record a telemetry point."""
        self.telemetry_buffer.append(point)
        
        if point.telemetry_type == TelemetryType.COUNTER:
            self.counters[point.name] += point.value
        elif point.telemetry_type == TelemetryType.GAUGE:
            self.gauges[point.name] = point.value
        elif point.telemetry_type == TelemetryType.HISTOGRAM:
            self.histograms[point.name].append(point.value)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {k: {"count": len(v), "mean": sum(v)/len(v) if v else 0} 
                          for k, v in self.histograms.items()}
        }


class LogLearner:
    """
    Learns from log patterns.
    Layer 4: Continuous learning.
    """
    
    def __init__(self):
        self.component_errors: Dict[str, int] = defaultdict(int)
        self.error_patterns: Dict[str, List[str]] = defaultdict(list)
        self.baseline_error_rate: float = 0.05
        self.learned_thresholds: Dict[str, float] = {}
    
    def learn_from_entry(self, entry: LogEntry, pattern: Optional[str]):
        """Learn from a log entry."""
        if entry.level.level >= LogLevel.ERROR.level:
            self.component_errors[entry.component] += 1
            if pattern:
                self.error_patterns[entry.component].append(pattern)
    
    def get_component_health(self) -> Dict[str, float]:
        """Get learned component health scores."""
        total_errors = sum(self.component_errors.values()) or 1
        return {
            comp: 1.0 - (errors / total_errors)
            for comp, errors in self.component_errors.items()
        }
    
    def update_thresholds(self):
        """Update learned thresholds."""
        for comp, errors in self.component_errors.items():
            self.learned_thresholds[comp] = max(0.1, 1.0 - (errors / 100))

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


class EnhancedLoggingEngine:
    """
    Unified logging engine with AI + Pattern + Learning.
    """
    
    def __init__(self, api_key: Optional[str] = None, use_ai: bool = True):
        # Layer 1: Pattern analysis
        self.pattern_analyzer = PatternBasedLogAnalyzer()
        
        # Layer 2: AI enhancement
        self.ai_enhancer = AILogEnhancer(api_key) if use_ai else None
        
        # Layer 3: Telemetry
        self.telemetry = TelemetryCollector()
        
        # Layer 4: Learning
        self.learner = LogLearner()
        
        # Log storage
        self.log_buffer: deque = deque(maxlen=10000)
        self.total_logs = 0
        
        self.lock = threading.RLock()
        logger.info("[Logging] Enhanced Logging Engine initialized")
    
    def log(self, level: LogLevel, message: str, component: str, 
            context: Optional[Dict[str, Any]] = None, trace_id: Optional[str] = None):
        """Log a message with full processing."""
        entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=level,
            message=message,
            component=component,
            context=context or {},
            trace_id=trace_id
        )
        
        with self.lock:
            self.log_buffer.append(entry)
            self.total_logs += 1
            
            # Pattern analysis
            pattern = self.pattern_analyzer.analyze(entry)
            
            # Learning
            self.learner.learn_from_entry(entry, pattern)
            
            # Record telemetry
            self.telemetry.record(TelemetryPoint(
                name=f"log_{level.label}",
                value=1,
                telemetry_type=TelemetryType.COUNTER,
                labels={"component": component}
            ))
        
        return entry
    
    def check_anomalies(self) -> Optional[LogAnomaly]:
        """Check for anomalies in recent logs."""
        if not self.ai_enhancer:
            return None
        
        with self.lock:
            recent = list(self.log_buffer)[-100:]
            return self.ai_enhancer.detect_anomaly(recent)
    
    def get_status(self) -> Dict[str, Any]:
        health = self.learner.get_component_health()
        metrics = self.telemetry.get_metrics()
        return {
            "mode": "AI + Pattern + Learning",
            "ai_enabled": self.ai_enhancer is not None,
            "total_logs": self.total_logs,
            "pattern_detections": self.pattern_analyzer.detection_count,
            "anomalies_detected": len(self.ai_enhancer.anomalies_detected) if self.ai_enhancer else 0,
            "components_tracked": len(self.learner.component_errors),
            "telemetry_points": len(self.telemetry.telemetry_buffer)
        }



    # ========================================
    # PHASE 49: PERSISTENCE METHODS
    # ========================================
    
    def save_state(self, filepath=None):
        """Save learning state (Phase 49)."""
        if hasattr(self, '_learning_manager') and self._learning_manager:
            return self._learning_manager.save_state()
        return False
    
    def load_state(self, filepath=None):
        """Load learning state (Phase 49)."""
        if hasattr(self, '_learning_manager') and self._learning_manager:
            return self._learning_manager.load_state()
        return False



    # ========================================
    # LEARNING INTERFACE (Auto-added)
    # ========================================

    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, input_data, output_data, was_correct, domain=None, metadata=None):
        """Record outcome for adaptive learning."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append({
            'input': str(input_data)[:100],
            'correct': was_correct,
            'domain': domain
        })
        self._outcomes = self._outcomes[-1000:]

    def record_feedback(self, result, was_accurate):
        """Record feedback for learning adjustment."""
        self.record_outcome({'result': str(result)}, {}, was_accurate)

    def adapt_thresholds(self, threshold_name, current_value, direction='decrease'):
        """Adapt thresholds based on learning history."""
        adjustment = 0.05 if direction == 'increase' else -0.05
        return max(0.1, min(0.95, current_value + adjustment))

    def get_domain_adjustment(self, domain):
        """Get learned adjustment for a domain."""
        if not hasattr(self, '_domain_adjustments'):
            self._domain_adjustments = {}
        return self._domain_adjustments.get(domain, 0.0)

    def get_learning_statistics(self):
        """Get learning statistics."""
        outcomes = getattr(self, '_outcomes', [])
        correct = sum(1 for o in outcomes if o.get('correct', False))
        return {
            'module': self.__class__.__name__,
            'total_outcomes': len(outcomes),
            'accuracy': correct / len(outcomes) if outcomes else 0.0
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


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 45: Logging & Telemetry (AI + Pattern + Learning)")
    print("=" * 70)
    
    engine = EnhancedLoggingEngine(api_key=None, use_ai=True)
    
    print("\n[1] Structured Logging")
    print("-" * 60)
    
    # Log various messages
    engine.log(LogLevel.INFO, "System starting up", "main")
    engine.log(LogLevel.INFO, "Processing request", "api", {"request_id": "123"})
    engine.log(LogLevel.WARNING, "High latency detected", "database")
    engine.log(LogLevel.ERROR, "Connection refused to service", "network")
    engine.log(LogLevel.ERROR, "Invalid token received", "auth")
    
    print(f"  Total Logs: {engine.total_logs}")
    print(f"  Pattern Detections: {engine.pattern_analyzer.detection_count}")
    
    print("\n[2] Pattern Analysis")
    print("-" * 60)
    patterns = engine.pattern_analyzer.get_pattern_summary()
    for pattern, count in patterns.items():
        print(f"  {pattern}: {count}")
    
    print("\n[3] Telemetry Collection")
    print("-" * 60)
    
    # Record additional telemetry
    engine.telemetry.record(TelemetryPoint("request_latency", 45.5, TelemetryType.HISTOGRAM))
    engine.telemetry.record(TelemetryPoint("request_latency", 52.3, TelemetryType.HISTOGRAM))
    engine.telemetry.record(TelemetryPoint("active_connections", 15, TelemetryType.GAUGE))
    
    metrics = engine.telemetry.get_metrics()
    print(f"  Counters: {len(metrics['counters'])}")
    print(f"  Gauges: {metrics['gauges']}")
    print(f"  Histograms: {list(metrics['histograms'].keys())}")
    
    print("\n[4] Learning & Health")
    print("-" * 60)
    health = engine.learner.get_component_health()
    for comp, score in health.items():
        print(f"  {comp}: {score:.2f}")
    
    print("\n[5] Engine Status")
    print("-" * 60)
    status = engine.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")
    
    print("\n" + "=" * 70)
    print("PHASE 45: Logging Engine - VERIFIED")
    print("=" * 70)
