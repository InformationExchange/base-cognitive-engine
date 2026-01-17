"""
BAIS Performance Metrics Module

Tracks per-invention and per-layer performance metrics.
Enables continuous improvement through data-driven optimization.

Key Features:
1. Per-invention latency tracking (67 inventions)
2. Per-layer accuracy aggregation (10 brain layers)
3. Domain-specific performance tracking
4. A/B test win rate correlation
5. Learning loop feedback integration

Patent Alignment:
- PPA1-Inv12: Adaptive Difficulty Adjustment (performance-based adaptation)
- NOVEL-6: Adaptive Signal Weighting (metrics-driven weighting)
- PPA2-Inv27: OCO Threshold Adapter (metrics feedback)
- NOVEL-30: Dimensional Learning (effectiveness tracking)

Phase 16A Enhancement: Foundation for continuous A/B testing and learning.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from enum import Enum
import json
import time
import threading
from collections import defaultdict


class BrainLayer(Enum):
    """BAIS 10 Brain-Like Layers"""
    PERCEPTION = "L1_perception"           # Sensory Cortex
    BEHAVIORAL = "L2_behavioral"           # Limbic System
    REASONING = "L3_reasoning"             # Prefrontal Cortex
    MEMORY = "L4_memory"                   # Hippocampus
    SELF_AWARENESS = "L5_self_awareness"   # Anterior Cingulate
    EVIDENCE = "L6_evidence"               # Basal Ganglia
    CHALLENGE = "L7_challenge"             # Amygdala
    IMPROVEMENT = "L8_improvement"         # Cerebellum
    ORCHESTRATION = "L9_orchestration"     # Thalamus
    OUTPUT = "L10_output"                  # Motor Cortex


# Invention to Layer mapping (all 67 inventions)
INVENTION_LAYER_MAP = {
    # Layer 1: Perception (8 inventions)
    "PPA1-Inv1": BrainLayer.PERCEPTION,
    "PPA1-Inv3": BrainLayer.PERCEPTION,
    "PPA1-Inv4": BrainLayer.PERCEPTION,
    "PPA1-Inv10": BrainLayer.PERCEPTION,
    "UP1": BrainLayer.PERCEPTION,
    "UP2": BrainLayer.PERCEPTION,
    "NOVEL-9": BrainLayer.PERCEPTION,
    "NOVEL-28": BrainLayer.PERCEPTION,
    
    # Layer 2: Behavioral (11 inventions)
    "PPA2-Inv2": BrainLayer.BEHAVIORAL,
    "PPA2-Inv4": BrainLayer.BEHAVIORAL,
    "PPA3-Inv2": BrainLayer.BEHAVIORAL,
    "NOVEL-1": BrainLayer.BEHAVIORAL,
    "NOVEL-21": BrainLayer.BEHAVIORAL,
    "PPA2-Big5": BrainLayer.BEHAVIORAL,
    "PPA1-Inv2": BrainLayer.BEHAVIORAL,
    "PPA1-Inv11": BrainLayer.BEHAVIORAL,
    "PPA1-Inv14": BrainLayer.BEHAVIORAL,
    "NOVEL-14": BrainLayer.BEHAVIORAL,
    "NOVEL-15": BrainLayer.BEHAVIORAL,
    
    # Layer 3: Reasoning (12 inventions)
    "PPA1-Inv5": BrainLayer.REASONING,
    "PPA1-Inv7": BrainLayer.REASONING,
    "UP3": BrainLayer.REASONING,
    "NOVEL-16": BrainLayer.REASONING,
    "PPA1-Inv8": BrainLayer.REASONING,
    "PPA1-Inv13": BrainLayer.REASONING,
    "PPA1-Inv16": BrainLayer.REASONING,
    "PPA2-Inv3": BrainLayer.REASONING,
    "PPA3-Inv3": BrainLayer.REASONING,
    "NOVEL-4": BrainLayer.REASONING,
    "NOVEL-5": BrainLayer.REASONING,
    "NOVEL-29": BrainLayer.REASONING,
    
    # Layer 4: Memory (6 inventions)
    "PPA1-Inv12": BrainLayer.MEMORY,
    "PPA1-Inv22": BrainLayer.MEMORY,
    "PPA2-Inv27": BrainLayer.MEMORY,
    "NOVEL-18": BrainLayer.MEMORY,
    "NOVEL-30": BrainLayer.MEMORY,
    "LLMAwareLearning": BrainLayer.MEMORY,
    
    # Layer 5: Self-Awareness (4 inventions)
    "NOVEL-17": BrainLayer.SELF_AWARENESS,
    "PPA1-Inv17": BrainLayer.SELF_AWARENESS,
    "NOVEL-7": BrainLayer.SELF_AWARENESS,
    "PPA1-Inv11": BrainLayer.SELF_AWARENESS,
    
    # Layer 6: Evidence (4 inventions)
    "NOVEL-3": BrainLayer.EVIDENCE,
    "GAP-1": BrainLayer.EVIDENCE,
    "PPA1-Inv6": BrainLayer.EVIDENCE,
    "UP4": BrainLayer.EVIDENCE,
    
    # Layer 7: Challenge (4 inventions)
    "NOVEL-22": BrainLayer.CHALLENGE,
    "NOVEL-23": BrainLayer.CHALLENGE,
    "PPA1-Inv23": BrainLayer.CHALLENGE,
    "NOVEL-8": BrainLayer.CHALLENGE,
    
    # Layer 8: Improvement (5 inventions)
    "NOVEL-20": BrainLayer.IMPROVEMENT,
    "UP5": BrainLayer.IMPROVEMENT,
    "PPA1-Inv15": BrainLayer.IMPROVEMENT,
    "PPA3-Inv1": BrainLayer.IMPROVEMENT,
    "PPA3-Inv2": BrainLayer.IMPROVEMENT,
    
    # Layer 9: Orchestration (8 inventions)
    "PPA1-Inv7": BrainLayer.ORCHESTRATION,
    "PPA1-Inv9": BrainLayer.ORCHESTRATION,
    "NOVEL-6": BrainLayer.ORCHESTRATION,
    "NOVEL-28": BrainLayer.ORCHESTRATION,
    "NOVEL-11": BrainLayer.ORCHESTRATION,
    "PPA2-Inv26": BrainLayer.ORCHESTRATION,
    "NOVEL-12": BrainLayer.ORCHESTRATION,
    "NOVEL-19": BrainLayer.ORCHESTRATION,
    
    # Layer 10: Output (5 inventions)
    "PPA1-Inv19": BrainLayer.OUTPUT,
    "PPA1-Inv25": BrainLayer.OUTPUT,
    "NOVEL-10": BrainLayer.OUTPUT,
    "PPA1-Inv18": BrainLayer.OUTPUT,
    "PPA1-Inv20": BrainLayer.OUTPUT,
}


@dataclass
class InventionMetrics:
    """Performance metrics for a single invention."""
    invention_id: str
    layer: BrainLayer
    
    # Timing metrics
    total_calls: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    
    # Accuracy metrics
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    
    # A/B test metrics
    ab_wins: int = 0
    ab_losses: int = 0
    ab_ties: int = 0
    
    # Domain performance
    domain_accuracy: Dict[str, float] = field(default_factory=dict)
    domain_calls: Dict[str, int] = field(default_factory=dict)
    
    # Timestamps
    first_call: Optional[str] = None
    last_call: Optional[str] = None
    
    @property
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(1, self.total_calls)
    
    @property
    def accuracy(self) -> float:
        total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total
    
    @property
    def precision(self) -> float:
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    @property
    def recall(self) -> float:
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    @property
    def f1_score(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)
    
    @property
    def ab_win_rate(self) -> float:
        total = self.ab_wins + self.ab_losses + self.ab_ties
        if total == 0:
            return 0.0
        return self.ab_wins / total
    
    def record_call(self, latency_ms: float, domain: str = "general"):
        """Record a call to this invention."""
        self.total_calls += 1
        self.total_latency_ms += latency_ms
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        
        now = datetime.now().isoformat()
        if self.first_call is None:
            self.first_call = now
        self.last_call = now
        
        self.domain_calls[domain] = self.domain_calls.get(domain, 0) + 1
    
    def record_outcome(self, was_correct: bool, domain: str = "general"):
        """Record outcome for accuracy tracking."""
        # Simple binary for now - can expand to full confusion matrix
        if was_correct:
            self.true_positives += 1
        else:
            self.false_positives += 1
        
        # Update domain accuracy
        current = self.domain_accuracy.get(domain, 0.5)
        alpha = 0.1
        self.domain_accuracy[domain] = alpha * (1.0 if was_correct else 0.0) + (1 - alpha) * current
    
    def record_ab_result(self, result: str):
        """Record A/B test result: 'win', 'loss', or 'tie'."""
        if result == 'win':
            self.ab_wins += 1
        elif result == 'loss':
            self.ab_losses += 1
        else:
            self.ab_ties += 1
    
    def to_dict(self) -> Dict:
        return {
            "invention_id": self.invention_id,
            "layer": self.layer.value,
            "total_calls": self.total_calls,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2) if self.min_latency_ms != float('inf') else None,
            "max_latency_ms": round(self.max_latency_ms, 2),
            "accuracy": round(self.accuracy, 3),
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "f1_score": round(self.f1_score, 3),
            "ab_win_rate": round(self.ab_win_rate, 3),
            "ab_record": f"{self.ab_wins}W-{self.ab_losses}L-{self.ab_ties}T",
            "domain_accuracy": {k: round(v, 3) for k, v in self.domain_accuracy.items()},
            "first_call": self.first_call,
            "last_call": self.last_call
        }


@dataclass
class LayerMetrics:
    """Aggregated metrics for a brain layer."""
    layer: BrainLayer
    invention_ids: List[str] = field(default_factory=list)
    
    # Aggregated timing
    total_calls: int = 0
    total_latency_ms: float = 0.0
    
    # Aggregated accuracy
    weighted_accuracy: float = 0.0
    
    # A/B metrics
    total_ab_wins: int = 0
    total_ab_tests: int = 0
    
    # Bottleneck detection
    slowest_invention: Optional[str] = None
    slowest_latency_ms: float = 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(1, self.total_calls)
    
    @property
    def ab_win_rate(self) -> float:
        if self.total_ab_tests == 0:
            return 0.0
        return self.total_ab_wins / self.total_ab_tests
    
    def to_dict(self) -> Dict:
        return {
            "layer": self.layer.value,
            "layer_name": self.layer.name,
            "invention_count": len(self.invention_ids),
            "total_calls": self.total_calls,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "weighted_accuracy": round(self.weighted_accuracy, 3),
            "ab_win_rate": round(self.ab_win_rate, 3),
            "bottleneck": {
                "invention": self.slowest_invention,
                "latency_ms": round(self.slowest_latency_ms, 2)
            } if self.slowest_invention else None
        }


class PerformanceTracker:
    """
    Central performance tracking for BAIS.
    
    Tracks:
    - Per-invention metrics (67 inventions)
    - Per-layer aggregations (10 layers)
    - Domain-specific performance
    - A/B test correlations
    """
    
    def __init__(self, storage_path: Path = None):
        """Initialize the performance tracker."""
        self.storage_path = storage_path or Path("performance_data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Metrics storage
        self.invention_metrics: Dict[str, InventionMetrics] = {}
        self.layer_metrics: Dict[BrainLayer, LayerMetrics] = {}
        
        # Session tracking
        self.session_start = datetime.now()
        self.session_evaluations = 0
        
        # Initialize layer metrics
        for layer in BrainLayer:
            self.layer_metrics[layer] = LayerMetrics(layer=layer)
        
        # Populate invention-layer relationships
        for inv_id, layer in INVENTION_LAYER_MAP.items():
            self.layer_metrics[layer].invention_ids.append(inv_id)
        
        # Load persisted metrics
        self._load_metrics()
    
    def start_invention_timer(self, invention_id: str) -> float:
        """Start timing an invention call. Returns start time."""
        return time.perf_counter() * 1000  # milliseconds
    
    def record_decision(
        self,
        invention_id: str,
        was_correct: bool,
        confidence: float,
        latency_ms: float,
        domain: str = "general"
    ):
        """
        Simple interface to record a decision outcome.
        
        Args:
            invention_id: Identifier for the invention/detector
            was_correct: Whether the decision was correct
            confidence: Confidence score (0-1)
            latency_ms: Processing time in milliseconds
            domain: Domain context
        """
        # Record call timing
        with self._lock:
            if invention_id not in self.invention_metrics:
                layer = INVENTION_LAYER_MAP.get(invention_id, BrainLayer.ORCHESTRATION)
                self.invention_metrics[invention_id] = InventionMetrics(
                    invention_id=invention_id,
                    layer=layer
                )
            
            metrics = self.invention_metrics[invention_id]
            metrics.record_call(latency_ms, domain)
            metrics.record_outcome(was_correct, domain)
            
            # Update layer
            layer = metrics.layer
            self.layer_metrics[layer].total_calls += 1
    
    def record_invention_call(
        self,
        invention_id: str,
        start_time_ms: float,
        domain: str = "general"
    ):
        """Record a completed invention call with timing."""
        latency_ms = time.perf_counter() * 1000 - start_time_ms
        
        with self._lock:
            # Get or create invention metrics
            if invention_id not in self.invention_metrics:
                layer = INVENTION_LAYER_MAP.get(invention_id, BrainLayer.ORCHESTRATION)
                self.invention_metrics[invention_id] = InventionMetrics(
                    invention_id=invention_id,
                    layer=layer
                )
            
            metrics = self.invention_metrics[invention_id]
            metrics.record_call(latency_ms, domain)
            
            # Update layer aggregates
            layer = metrics.layer
            self.layer_metrics[layer].total_calls += 1
            self.layer_metrics[layer].total_latency_ms += latency_ms
            
            # Track bottleneck
            if metrics.avg_latency_ms > self.layer_metrics[layer].slowest_latency_ms:
                self.layer_metrics[layer].slowest_invention = invention_id
                self.layer_metrics[layer].slowest_latency_ms = metrics.avg_latency_ms
    
    def record_invention_outcome(
        self,
        invention_id: str,
        was_correct: bool,
        domain: str = "general"
    ):
        """Record the outcome of an invention's detection."""
        with self._lock:
            if invention_id in self.invention_metrics:
                self.invention_metrics[invention_id].record_outcome(was_correct, domain)
                
                # Update layer weighted accuracy
                layer = self.invention_metrics[invention_id].layer
                self._update_layer_accuracy(layer)
    
    def record_ab_result(
        self,
        invention_id: str,
        result: str  # 'win', 'loss', 'tie'
    ):
        """Record A/B test result for an invention."""
        with self._lock:
            if invention_id in self.invention_metrics:
                self.invention_metrics[invention_id].record_ab_result(result)
                
                # Update layer A/B metrics
                layer = self.invention_metrics[invention_id].layer
                self.layer_metrics[layer].total_ab_tests += 1
                if result == 'win':
                    self.layer_metrics[layer].total_ab_wins += 1
    
    def _update_layer_accuracy(self, layer: BrainLayer):
        """Update weighted accuracy for a layer based on its inventions."""
        layer_metrics = self.layer_metrics[layer]
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for inv_id in layer_metrics.invention_ids:
            if inv_id in self.invention_metrics:
                inv_metrics = self.invention_metrics[inv_id]
                weight = inv_metrics.total_calls  # Weight by usage
                if weight > 0:
                    weighted_sum += inv_metrics.accuracy * weight
                    total_weight += weight
        
        if total_weight > 0:
            layer_metrics.weighted_accuracy = weighted_sum / total_weight
    
    def get_invention_report(self, invention_id: str = None) -> Dict:
        """Get metrics report for one or all inventions."""
        with self._lock:
            if invention_id:
                if invention_id in self.invention_metrics:
                    return self.invention_metrics[invention_id].to_dict()
                return {"error": f"Invention {invention_id} not found"}
            
            return {
                inv_id: metrics.to_dict()
                for inv_id, metrics in self.invention_metrics.items()
            }
    
    def get_layer_report(self, layer: BrainLayer = None) -> Dict:
        """Get metrics report for one or all layers."""
        with self._lock:
            if layer:
                return self.layer_metrics[layer].to_dict()
            
            return {
                layer.value: metrics.to_dict()
                for layer, metrics in self.layer_metrics.items()
            }
    
    def get_bottleneck_report(self) -> Dict:
        """Identify performance bottlenecks across all layers."""
        with self._lock:
            bottlenecks = []
            
            for layer, metrics in self.layer_metrics.items():
                if metrics.slowest_invention:
                    bottlenecks.append({
                        "layer": layer.value,
                        "invention": metrics.slowest_invention,
                        "avg_latency_ms": round(metrics.slowest_latency_ms, 2),
                        "total_calls": metrics.total_calls
                    })
            
            # Sort by latency
            bottlenecks.sort(key=lambda x: x["avg_latency_ms"], reverse=True)
            
            return {
                "top_bottlenecks": bottlenecks[:5],
                "recommendations": self._generate_recommendations(bottlenecks)
            }
    
    def _generate_recommendations(self, bottlenecks: List[Dict]) -> List[str]:
        """Generate optimization recommendations based on bottlenecks."""
        recommendations = []
        
        for b in bottlenecks[:3]:
            if b["avg_latency_ms"] > 100:
                recommendations.append(
                    f"Consider caching for {b['invention']} (Layer {b['layer']}) - "
                    f"avg {b['avg_latency_ms']}ms over {b['total_calls']} calls"
                )
            elif b["avg_latency_ms"] > 50:
                recommendations.append(
                    f"Monitor {b['invention']} for optimization opportunities"
                )
        
        return recommendations
    
    def get_domain_report(self, domain: str = None) -> Dict:
        """Get performance breakdown by domain."""
        with self._lock:
            if domain:
                return self._get_single_domain_report(domain)
            
            # Aggregate across all domains
            all_domains = set()
            for metrics in self.invention_metrics.values():
                all_domains.update(metrics.domain_calls.keys())
            
            return {
                d: self._get_single_domain_report(d)
                for d in all_domains
            }
    
    def _get_single_domain_report(self, domain: str) -> Dict:
        """Get performance for a single domain."""
        domain_stats = {
            "total_calls": 0,
            "avg_accuracy": 0.0,
            "top_inventions": [],
            "inventions_used": 0
        }
        
        accuracies = []
        for inv_id, metrics in self.invention_metrics.items():
            if domain in metrics.domain_calls:
                domain_stats["total_calls"] += metrics.domain_calls[domain]
                domain_stats["inventions_used"] += 1
                if domain in metrics.domain_accuracy:
                    accuracies.append(metrics.domain_accuracy[domain])
        
        if accuracies:
            domain_stats["avg_accuracy"] = round(sum(accuracies) / len(accuracies), 3)
        
        return domain_stats
    
    def get_summary(self) -> Dict:
        """Get quick summary of performance metrics."""
        with self._lock:
            return {
                "total_inventions": len(self.invention_metrics),
                "total_calls": sum(m.total_calls for m in self.invention_metrics.values()),
                "overall_accuracy": self._calculate_overall_accuracy(),
                "session_evaluations": self.session_evaluations,
                "duration_minutes": (datetime.now() - self.session_start).seconds / 60
            }
    
    def get_comprehensive_report(self) -> Dict:
        """Get complete performance report."""
        with self._lock:
            return {
                "session": {
                    "start": self.session_start.isoformat(),
                    "duration_minutes": (datetime.now() - self.session_start).seconds / 60,
                    "evaluations": self.session_evaluations
                },
                "summary": {
                    "total_inventions_tracked": len(self.invention_metrics),
                    "total_calls": sum(m.total_calls for m in self.invention_metrics.values()),
                    "overall_accuracy": self._calculate_overall_accuracy(),
                    "overall_ab_win_rate": self._calculate_overall_ab_win_rate()
                },
                "layers": self.get_layer_report(),
                "bottlenecks": self.get_bottleneck_report(),
                "domains": self.get_domain_report()
            }
    
    def _calculate_overall_accuracy(self) -> float:
        """Calculate overall weighted accuracy."""
        total_weight = 0.0
        weighted_sum = 0.0
        
        for metrics in self.invention_metrics.values():
            weight = metrics.total_calls
            if weight > 0:
                weighted_sum += metrics.accuracy * weight
                total_weight += weight
        
        return round(weighted_sum / max(1, total_weight), 3)
    
    def _calculate_overall_ab_win_rate(self) -> float:
        """Calculate overall A/B test win rate."""
        total_wins = sum(m.ab_wins for m in self.invention_metrics.values())
        total_tests = sum(m.ab_wins + m.ab_losses + m.ab_ties for m in self.invention_metrics.values())
        
        return round(total_wins / max(1, total_tests), 3)
    
    def _load_metrics(self):
        """Load persisted metrics from storage."""
        metrics_file = self.storage_path / "performance_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                # TODO: Deserialize metrics
            except Exception as e:
                print(f"[PerformanceTracker] Load error: {e}")
    
    def save_metrics(self):
        """Persist metrics to storage."""
        metrics_file = self.storage_path / "performance_metrics.json"
        try:
            with open(metrics_file, 'w') as f:
                json.dump(self.get_comprehensive_report(), f, indent=2)
        except Exception as e:
            print(f"[PerformanceTracker] Save error: {e}")

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


# Singleton for easy access
performance_tracker = PerformanceTracker()


# Context manager for timing invention calls
class InventionTimer:
    """Context manager for timing invention execution."""
    
    def __init__(self, invention_id: str, domain: str = "general"):
        self.invention_id = invention_id
        self.domain = domain
        self.start_time = None
    
    def __enter__(self):
        self.start_time = performance_tracker.start_invention_timer(self.invention_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            performance_tracker.record_invention_call(
                self.invention_id,
                self.start_time,
                self.domain
            )
        return False  # Don't suppress exceptions

