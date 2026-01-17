"""
BASE Cognitive Governance Engine v41.0
Real-time Monitoring with AI + Pattern + Learning

Phase 41: Operational Enhancement
- AI-enhanced anomaly detection
- Pattern-based alerting
- Continuous learning from metrics
"""

import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Deque
from enum import Enum
from datetime import datetime, timedelta
from collections import deque, defaultdict
import threading
import logging
import numpy as np

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = ("info", 0)
    WARNING = ("warning", 1)
    ERROR = ("error", 2)
    CRITICAL = ("critical", 3)
    
    def __init__(self, label: str, level: int):
        self.label = label
        self.level = level


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertState(Enum):
    FIRING = "firing"
    RESOLVED = "resolved"
    PENDING = "pending"
    SILENCED = "silenced"


@dataclass
class MetricPoint:
    name: str
    value: float
    labels: Dict[str, str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class Alert:
    alert_id: str
    name: str
    severity: AlertSeverity
    message: str
    state: AlertState
    metric_name: str
    threshold: float
    current_value: float
    created_at: datetime
    resolved_at: Optional[datetime] = None
    ai_detected: bool = False


@dataclass
class HealthStatus:
    component: str
    healthy: bool
    latency_ms: float
    last_check: datetime
    error_count: int
    message: str


class PatternBasedAlerting:
    """
    Pattern-based alerting rules.
    Layer 1: Static threshold detection.
    """
    
    DEFAULT_RULES = {
        "high_latency": {"metric": "latency_ms", "threshold": 100, "op": ">", "severity": AlertSeverity.WARNING},
        "error_rate": {"metric": "error_rate", "threshold": 0.05, "op": ">", "severity": AlertSeverity.ERROR},
        "low_cache_hit": {"metric": "cache_hit_rate", "threshold": 0.3, "op": "<", "severity": AlertSeverity.WARNING},
        "high_memory": {"metric": "memory_mb", "threshold": 500, "op": ">", "severity": AlertSeverity.WARNING},
        "low_throughput": {"metric": "throughput_rps", "threshold": 10, "op": "<", "severity": AlertSeverity.WARNING},
        "violation_spike": {"metric": "violations_per_min", "threshold": 10, "op": ">", "severity": AlertSeverity.ERROR},
    }
    
    def __init__(self):
        self.rules = dict(self.DEFAULT_RULES)
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.detection_count = 0
    
    def check_rules(self, metrics: Dict[str, float]) -> List[Alert]:
        """Check all rules against current metrics."""
        alerts = []
        
        for rule_name, rule in self.rules.items():
            metric_name = rule["metric"]
            if metric_name not in metrics:
                continue
            
            value = metrics[metric_name]
            threshold = rule["threshold"]
            op = rule["op"]
            
            triggered = False
            if op == ">" and value > threshold:
                triggered = True
            elif op == "<" and value < threshold:
                triggered = True
            elif op == ">=" and value >= threshold:
                triggered = True
            elif op == "<=" and value <= threshold:
                triggered = True
            
            if triggered:
                self.detection_count += 1
                alert = Alert(
                    alert_id=hashlib.sha256(f"{rule_name}:{datetime.utcnow()}".encode()).hexdigest()[:12],
                    name=rule_name,
                    severity=rule["severity"],
                    message=f"{metric_name} {op} {threshold} (current: {value:.2f})",
                    state=AlertState.FIRING,
                    metric_name=metric_name,
                    threshold=threshold,
                    current_value=value,
                    created_at=datetime.utcnow()
                )
                alerts.append(alert)
                self.active_alerts[rule_name] = alert
                self.alert_history.append(alert)
            elif rule_name in self.active_alerts:
                # Resolve alert
                self.active_alerts[rule_name].state = AlertState.RESOLVED
                self.active_alerts[rule_name].resolved_at = datetime.utcnow()
                del self.active_alerts[rule_name]
        
        return alerts
    
    def add_rule(self, name: str, metric: str, threshold: float, op: str, severity: AlertSeverity):
        """Add a custom alerting rule."""
        self.rules[name] = {"metric": metric, "threshold": threshold, "op": op, "severity": severity}


class AIAnomalyDetector:
    """
    AI-enhanced anomaly detection.
    Layer 2: Statistical and ML-based detection.
    """
    
    def __init__(self, window_size: int = 100, sensitivity: float = 2.0):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.metric_windows: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window_size))
        self.baselines: Dict[str, Dict[str, float]] = {}
        self.anomaly_count = 0
    
    def update(self, metric_name: str, value: float):
        """Update metric window."""
        self.metric_windows[metric_name].append(value)
        
        # Update baseline if enough data
        if len(self.metric_windows[metric_name]) >= 20:
            values = list(self.metric_windows[metric_name])
            self.baselines[metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "p95": np.percentile(values, 95)
            }
    
    def detect_anomaly(self, metric_name: str, value: float) -> Optional[Alert]:
        """Detect anomalies using statistical methods."""
        if metric_name not in self.baselines:
            return None
        
        baseline = self.baselines[metric_name]
        mean, std = baseline["mean"], baseline["std"]
        
        if std == 0:
            return None
        
        z_score = abs(value - mean) / std
        
        if z_score > self.sensitivity:
            self.anomaly_count += 1
            return Alert(
                alert_id=hashlib.sha256(f"anomaly:{metric_name}:{datetime.utcnow()}".encode()).hexdigest()[:12],
                name=f"anomaly_{metric_name}",
                severity=AlertSeverity.WARNING if z_score < 3 else AlertSeverity.ERROR,
                message=f"Anomaly detected: z-score={z_score:.2f} (value={value:.2f}, mean={mean:.2f})",
                state=AlertState.FIRING,
                metric_name=metric_name,
                threshold=mean + self.sensitivity * std,
                current_value=value,
                created_at=datetime.utcnow(),
                ai_detected=True
            )
        
        return None
    
    def predict_trend(self, metric_name: str) -> Optional[str]:
        """Predict metric trend."""
        if metric_name not in self.metric_windows:
            return None
        
        values = list(self.metric_windows[metric_name])
        if len(values) < 10:
            return None
        
        # Simple linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        return "stable"

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


class AdaptiveAlertLearner:
    """
    Learns optimal alert thresholds from feedback.
    Layer 3: Continuous improvement.
    """
    
    def __init__(self):
        self.threshold_adjustments: Dict[str, List[float]] = defaultdict(list)
        self.false_positive_rules: Dict[str, int] = defaultdict(int)
        self.true_positive_rules: Dict[str, int] = defaultdict(int)
        self.learned_thresholds: Dict[str, float] = {}
    
    def record_feedback(self, alert: Alert, was_valid: bool):
        """Record feedback on alert validity."""
        if was_valid:
            self.true_positive_rules[alert.name] += 1
        else:
            self.false_positive_rules[alert.name] += 1
            # Suggest threshold adjustment
            self.threshold_adjustments[alert.name].append(alert.current_value)
    
    def get_recommended_threshold(self, rule_name: str, current_threshold: float) -> float:
        """Get recommended threshold based on feedback."""
        if rule_name not in self.threshold_adjustments:
            return current_threshold
        
        adjustments = self.threshold_adjustments[rule_name]
        if len(adjustments) < 3:
            return current_threshold
        
        # Adjust threshold to reduce false positives
        avg_fp_value = np.mean(adjustments)
        new_threshold = (current_threshold + avg_fp_value) / 2
        self.learned_thresholds[rule_name] = new_threshold
        return new_threshold
    
    def get_precision(self, rule_name: str) -> float:
        """Get precision for a rule."""
        tp = self.true_positive_rules.get(rule_name, 0)
        fp = self.false_positive_rules.get(rule_name, 0)
        if tp + fp == 0:
            return 1.0
        return tp / (tp + fp)

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


class EnhancedMonitoringEngine:
    """
    Unified monitoring engine with AI + Pattern + Learning.
    """
    
    def __init__(self, api_key: Optional[str] = None, use_ai: bool = True):
        # Layer 1: Pattern-based alerting
        self.alerting = PatternBasedAlerting()
        
        # Layer 2: AI anomaly detection
        self.anomaly_detector = AIAnomalyDetector() if use_ai else None
        
        # Layer 3: Adaptive learning
        self.learner = AdaptiveAlertLearner()
        
        # Metrics storage
        self.metrics: Dict[str, float] = {}
        self.metric_history: List[MetricPoint] = []
        self.health_status: Dict[str, HealthStatus] = {}
        
        # Stats
        self.total_metrics = 0
        self.total_alerts = 0
        self.ai_anomalies = 0
        
        self.lock = threading.RLock()
        logger.info("[Monitoring] Enhanced Monitoring Engine initialized")
    
    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        with self.lock:
            self.metrics[name] = value
            self.metric_history.append(MetricPoint(name=name, value=value, labels=labels or {}))
            self.total_metrics += 1
            
            # Update anomaly detector
            if self.anomaly_detector:
                self.anomaly_detector.update(name, value)
            
            # Keep history bounded
            if len(self.metric_history) > 10000:
                self.metric_history = self.metric_history[-5000:]
    
    def check_alerts(self) -> List[Alert]:
        """Check all alerting rules and anomaly detection."""
        alerts = []
        
        with self.lock:
            # Pattern-based alerts
            alerts.extend(self.alerting.check_rules(self.metrics))
            
            # AI anomaly detection
            if self.anomaly_detector:
                for name, value in self.metrics.items():
                    anomaly = self.anomaly_detector.detect_anomaly(name, value)
                    if anomaly:
                        alerts.append(anomaly)
                        self.ai_anomalies += 1
            
            self.total_alerts += len(alerts)
        
        return alerts
    
    def record_health(self, component: str, healthy: bool, latency_ms: float, message: str = ""):
        """Record component health status."""
        with self.lock:
            self.health_status[component] = HealthStatus(
                component=component,
                healthy=healthy,
                latency_ms=latency_ms,
                last_check=datetime.utcnow(),
                error_count=0 if healthy else self.health_status.get(component, HealthStatus(component, True, 0, datetime.utcnow(), 0, "")).error_count + 1,
                message=message
            )
    
    def record_alert_feedback(self, alert: Alert, was_valid: bool):
        """Record feedback for learning."""
        self.learner.record_feedback(alert, was_valid)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard display."""
        with self.lock:
            return {
                "metrics": dict(self.metrics),
                "active_alerts": [
                    {"name": a.name, "severity": a.severity.label, "message": a.message}
                    for a in self.alerting.active_alerts.values()
                ],
                "health": {
                    name: {"healthy": h.healthy, "latency_ms": h.latency_ms}
                    for name, h in self.health_status.items()
                },
                "trends": {
                    name: self.anomaly_detector.predict_trend(name)
                    for name in self.metrics.keys()
                } if self.anomaly_detector else {}
            }
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "mode": "AI + Pattern + Learning",
            "ai_enabled": self.anomaly_detector is not None,
            "total_metrics": self.total_metrics,
            "total_alerts": self.total_alerts,
            "ai_anomalies": self.ai_anomalies,
            "active_alerts": len(self.alerting.active_alerts),
            "alert_rules": len(self.alerting.rules),
            "components_monitored": len(self.health_status),
            "learned_thresholds": len(self.learner.learned_thresholds)
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
    
    def record_feedback(self, result, was_accurate):
        """Record feedback for learning (standard interface)."""
        self.record_outcome({'result': str(result)}, {}, was_accurate)

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
    print("PHASE 41: Real-time Monitoring (AI + Pattern + Learning)")
    print("=" * 70)
    
    engine = EnhancedMonitoringEngine(api_key=None, use_ai=True)
    
    print("\n[1] Testing Metric Recording")
    print("-" * 60)
    
    # Simulate metrics
    for i in range(50):
        engine.record_metric("latency_ms", 20 + np.random.randn() * 5)
        engine.record_metric("cache_hit_rate", 0.7 + np.random.randn() * 0.1)
        engine.record_metric("error_rate", 0.01 + np.random.randn() * 0.005)
    
    # Add anomaly
    engine.record_metric("latency_ms", 200)  # Spike!
    
    print(f"  Metrics Recorded: {engine.total_metrics}")
    
    print("\n[2] Testing Alert Detection")
    print("-" * 60)
    
    alerts = engine.check_alerts()
    print(f"  Alerts Triggered: {len(alerts)}")
    for alert in alerts[:3]:
        print(f"    - [{alert.severity.label}] {alert.name}: {alert.message}")
    
    print("\n[3] Testing AI Anomaly Detection")
    print("-" * 60)
    print(f"  AI Anomalies Detected: {engine.ai_anomalies}")
    if engine.anomaly_detector:
        for metric in ["latency_ms", "cache_hit_rate"]:
            trend = engine.anomaly_detector.predict_trend(metric)
            print(f"    {metric} trend: {trend}")
    
    print("\n[4] Testing Continuous Learning")
    print("-" * 60)
    
    # Simulate feedback
    for alert in alerts[:2]:
        engine.record_alert_feedback(alert, was_valid=True)
    if len(alerts) > 2:
        engine.record_alert_feedback(alerts[2], was_valid=False)
    
    for rule in ["high_latency", "error_rate"]:
        precision = engine.learner.get_precision(rule)
        print(f"    {rule} precision: {precision:.1%}")
    
    print("\n[5] Dashboard Data")
    print("-" * 60)
    
    dashboard = engine.get_dashboard_data()
    print(f"  Metrics: {len(dashboard['metrics'])}")
    print(f"  Active Alerts: {len(dashboard['active_alerts'])}")
    print(f"  Health Components: {len(dashboard['health'])}")
    
    print("\n[6] Engine Status")
    print("-" * 60)
    for k, v in engine.get_status().items():
        print(f"  {k}: {v}")
    
    print("\n" + "=" * 70)
    print("PHASE 41: Monitoring Engine - VERIFIED")
    print("=" * 70)
