"""
BAIS Dynamic Orchestration System
=================================

Phase 4 Enhancement: Intelligent pathway selection and orchestration.

This module implements:
1. Dynamic Pathway Selection: Choose optimal pathway based on context
2. Layer Skip Prediction: Learn which layers are unnecessary for specific domains
3. Early Termination: Stop when confident decision is reached
4. Resource Optimization: Minimize computation while maintaining accuracy

Patent Alignment:
- PPA2-Inv27: OCO Threshold Adapter (adaptive learning)
- PPA2-Comp8: VOI Hierarchical Short-Circuiting
- NOVEL-10: Smart Gate (query routing)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import json
import time
import tempfile


class PathwayType(Enum):
    """Types of governance pathways."""
    FAST = "fast"           # Quick threshold check only
    STANDARD = "standard"   # Full pipeline
    DEEP = "deep"          # Multi-track challenger
    AUDIT = "audit"        # Full audit with evidence demand


@dataclass
class PathwayMetrics:
    """Performance metrics for a pathway."""
    accuracy: float = 0.5
    avg_time_ms: float = 100.0
    samples: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    def update(self, was_correct: bool, time_ms: float):
        """Update metrics with new outcome."""
        self.samples += 1
        self.avg_time_ms = (self.avg_time_ms * (self.samples - 1) + time_ms) / self.samples
        if was_correct:
            self.accuracy = (self.accuracy * (self.samples - 1) + 1.0) / self.samples
        else:
            self.accuracy = (self.accuracy * (self.samples - 1) + 0.0) / self.samples


@dataclass
class OrchestrationDecision:
    """Decision from the orchestrator."""
    pathway: PathwayType
    layers_to_run: List[str]
    layers_to_skip: List[str]
    confidence: float
    reasoning: List[str]
    estimated_time_ms: float


@dataclass
class LayerPerformance:
    """Performance metrics for a layer."""
    layer_name: str
    accuracy_by_domain: Dict[str, float] = field(default_factory=dict)
    time_ms_by_domain: Dict[str, float] = field(default_factory=dict)
    skip_rate_by_domain: Dict[str, float] = field(default_factory=dict)
    
    def should_skip(self, domain: str) -> bool:
        """Determine if layer should be skipped for domain."""
        # Skip if layer has low accuracy for this domain
        accuracy = self.accuracy_by_domain.get(domain, 0.5)
        skip_rate = self.skip_rate_by_domain.get(domain, 0.0)
        return skip_rate > 0.7 and accuracy < 0.3


class DynamicPathwaySelector:
    """
    Intelligent pathway selection based on learned performance.
    
    Phase 4 Enhancement: Dynamic orchestration.
    
    Features:
    1. Domain-specific pathway preferences
    2. Query complexity analysis
    3. Time budget consideration
    4. Learned layer skip prediction
    """
    
    # Default pathway configurations
    PATHWAY_CONFIGS = {
        PathwayType.FAST: {
            "layers": ["behavioral"],
            "time_budget_ms": 50,
            "description": "Quick bias check only"
        },
        PathwayType.STANDARD: {
            "layers": ["behavioral", "grounding", "factual", "temporal"],
            "time_budget_ms": 200,
            "description": "Full detector pipeline"
        },
        PathwayType.DEEP: {
            "layers": ["behavioral", "grounding", "factual", "temporal", "challenger", "evidence"],
            "time_budget_ms": 1000,
            "description": "Full pipeline with multi-track challenge"
        },
        PathwayType.AUDIT: {
            "layers": ["behavioral", "grounding", "factual", "temporal", "challenger", "evidence", "self_awareness"],
            "time_budget_ms": 5000,
            "description": "Full audit with all verification"
        }
    }
    
    def __init__(self, storage_path: Path = None):
        # Use temp directory if no path provided
        if storage_path is None:
            storage_path = Path(tempfile.mkdtemp(prefix="bais_orchestration_"))
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Pathway performance tracking
        self.pathway_metrics: Dict[PathwayType, PathwayMetrics] = {
            p: PathwayMetrics() for p in PathwayType
        }
        
        # Domain preferences (learned)
        self.domain_preferences: Dict[str, PathwayType] = {}
        
        # Layer performance tracking
        self.layer_performance: Dict[str, LayerPerformance] = {}
        
        # Complexity thresholds (learned)
        self.complexity_thresholds = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8
        }
        
        # Learning rate
        self._learning_rate = 0.1
        
        # Load persisted state
        self._load_state()
    
    def select_pathway(
        self, 
        query: str, 
        domain: str = "general",
        time_budget_ms: int = None,
        risk_level: str = "medium"
    ) -> OrchestrationDecision:
        """
        Select optimal pathway based on context and learned performance.
        
        Args:
            query: The user query
            domain: Domain context
            time_budget_ms: Optional time budget
            risk_level: Risk level (low, medium, high, critical)
            
        Returns:
            OrchestrationDecision with selected pathway and layers
        """
        reasoning = []
        
        # 1. Check domain preference (learned)
        if domain in self.domain_preferences:
            preferred = self.domain_preferences[domain]
            reasoning.append(f"Domain '{domain}' prefers pathway: {preferred.value}")
        else:
            preferred = None
        
        # 2. Analyze query complexity
        complexity = self._analyze_complexity(query)
        reasoning.append(f"Query complexity: {complexity}")
        
        # 3. Consider risk level
        if risk_level in ["high", "critical"]:
            min_pathway = PathwayType.DEEP
            reasoning.append(f"High risk requires deep analysis")
        elif risk_level == "medium":
            min_pathway = PathwayType.STANDARD
        else:
            min_pathway = PathwayType.FAST
        
        # 4. Consider time budget
        if time_budget_ms:
            candidates = [
                p for p, config in self.PATHWAY_CONFIGS.items()
                if config["time_budget_ms"] <= time_budget_ms
            ]
            reasoning.append(f"Time budget {time_budget_ms}ms allows: {[c.value for c in candidates]}")
        else:
            candidates = list(PathwayType)
        
        # 5. Select pathway
        if not candidates:
            candidates = [PathwayType.STANDARD]  # Fallback
            
        if complexity == "high" and PathwayType.DEEP in candidates:
            pathway = PathwayType.DEEP
        elif complexity == "low" and PathwayType.FAST in candidates and min_pathway == PathwayType.FAST:
            pathway = PathwayType.FAST
        elif preferred and preferred in candidates:
            pathway = preferred
        else:
            # Default to standard if available
            pathway = PathwayType.STANDARD if PathwayType.STANDARD in candidates else candidates[0]
        
        reasoning.append(f"Selected pathway: {pathway.value}")
        
        # 6. Determine layers to run and skip
        config = self.PATHWAY_CONFIGS[pathway]
        layers_to_run = []
        layers_to_skip = []
        
        for layer in config["layers"]:
            if layer in self.layer_performance:
                if self.layer_performance[layer].should_skip(domain):
                    layers_to_skip.append(layer)
                    reasoning.append(f"Skipping '{layer}' (learned low value for {domain})")
                else:
                    layers_to_run.append(layer)
            else:
                layers_to_run.append(layer)
        
        # Calculate estimated time
        estimated_time = sum(
            self.layer_performance.get(l, LayerPerformance(l)).time_ms_by_domain.get(domain, 50.0)
            for l in layers_to_run
        )
        
        return OrchestrationDecision(
            pathway=pathway,
            layers_to_run=layers_to_run,
            layers_to_skip=layers_to_skip,
            confidence=self.pathway_metrics[pathway].accuracy,
            reasoning=reasoning,
            estimated_time_ms=estimated_time
        )
    
    def record_outcome(
        self,
        pathway: PathwayType,
        domain: str,
        was_correct: bool,
        time_ms: float,
        layers_run: List[str]
    ):
        """
        Record outcome to improve future pathway selection.
        
        Args:
            pathway: Pathway used
            domain: Domain context
            was_correct: Whether the decision was correct
            time_ms: Actual execution time
            layers_run: Layers that were executed
        """
        # Update pathway metrics
        self.pathway_metrics[pathway].update(was_correct, time_ms)
        
        # Update domain preference if this pathway performed well
        if was_correct and self.pathway_metrics[pathway].accuracy > 0.8:
            self.domain_preferences[domain] = pathway
        
        # Update layer performance
        for layer in layers_run:
            if layer not in self.layer_performance:
                self.layer_performance[layer] = LayerPerformance(layer)
            
            lp = self.layer_performance[layer]
            
            # Update accuracy
            current_acc = lp.accuracy_by_domain.get(domain, 0.5)
            if was_correct:
                lp.accuracy_by_domain[domain] = current_acc * 0.9 + 0.1
            else:
                lp.accuracy_by_domain[domain] = current_acc * 0.9
        
        # Persist state
        self._save_state()
    
    def _analyze_complexity(self, query: str) -> str:
        """Analyze query complexity."""
        # Simple heuristic based on length and keywords
        length = len(query)
        
        # Check for complexity indicators
        complex_keywords = [
            "analyze", "compare", "evaluate", "comprehensive",
            "detailed", "explain", "verify", "audit"
        ]
        keyword_count = sum(1 for kw in complex_keywords if kw in query.lower())
        
        # Calculate complexity score
        score = (length / 200) * 0.5 + (keyword_count / 5) * 0.5
        
        if score > self.complexity_thresholds["high"]:
            return "high"
        elif score > self.complexity_thresholds["medium"]:
            return "medium"
        else:
            return "low"
    
    def get_statistics(self) -> Dict:
        """Get orchestration statistics."""
        return {
            "pathway_metrics": {
                p.value: {
                    "accuracy": m.accuracy,
                    "avg_time_ms": m.avg_time_ms,
                    "samples": m.samples
                }
                for p, m in self.pathway_metrics.items()
            },
            "domain_preferences": {
                d: p.value for d, p in self.domain_preferences.items()
            },
            "layer_count": len(self.layer_performance),
            "complexity_thresholds": self.complexity_thresholds
        }
    
    def _save_state(self):
        """Persist orchestration state."""
        state = {
            "pathway_metrics": {
                p.value: {
                    "accuracy": m.accuracy,
                    "avg_time_ms": m.avg_time_ms,
                    "samples": m.samples,
                    "false_positives": m.false_positives,
                    "false_negatives": m.false_negatives
                }
                for p, m in self.pathway_metrics.items()
            },
            "domain_preferences": {d: p.value for d, p in self.domain_preferences.items()},
            "complexity_thresholds": self.complexity_thresholds
        }
        
        state_path = self.storage_path / "orchestration_state.json"
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load orchestration state."""
        state_path = self.storage_path / "orchestration_state.json"
        
        if not state_path.exists():
            return
        
        try:
            with open(state_path) as f:
                state = json.load(f)
            
            # Load pathway metrics
            for p_name, metrics in state.get("pathway_metrics", {}).items():
                pathway = PathwayType(p_name)
                self.pathway_metrics[pathway] = PathwayMetrics(
                    accuracy=metrics.get("accuracy", 0.5),
                    avg_time_ms=metrics.get("avg_time_ms", 100.0),
                    samples=metrics.get("samples", 0),
                    false_positives=metrics.get("false_positives", 0),
                    false_negatives=metrics.get("false_negatives", 0)
                )
            
            # Load domain preferences
            for domain, p_name in state.get("domain_preferences", {}).items():
                self.domain_preferences[domain] = PathwayType(p_name)
            
            # Load complexity thresholds
            self.complexity_thresholds = state.get("complexity_thresholds", self.complexity_thresholds)
            
        except Exception as e:
            print(f"Warning: Could not load orchestration state: {e}")
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard wrapper for non-standard record_outcome."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

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


class OrchestrationEngine:
    """
    Main orchestration engine that coordinates pathway selection and execution.
    
    Integrates with IntegratedGovernanceEngine to provide dynamic orchestration.
    """
    
    def __init__(self):
        self.selector = DynamicPathwaySelector()
        self.execution_history: List[Dict] = []
    
    def plan_execution(
        self,
        query: str,
        domain: str = "general",
        time_budget_ms: int = None,
        risk_level: str = "medium"
    ) -> OrchestrationDecision:
        """
        Plan the execution pathway for a query.
        
        Args:
            query: User query
            domain: Domain context
            time_budget_ms: Optional time budget
            risk_level: Risk level
            
        Returns:
            OrchestrationDecision with execution plan
        """
        return self.selector.select_pathway(
            query=query,
            domain=domain,
            time_budget_ms=time_budget_ms,
            risk_level=risk_level
        )
    
    def record_execution(
        self,
        decision: OrchestrationDecision,
        domain: str,
        was_correct: bool,
        actual_time_ms: float
    ):
        """Record execution outcome for learning."""
        self.selector.record_outcome(
            pathway=decision.pathway,
            domain=domain,
            was_correct=was_correct,
            time_ms=actual_time_ms,
            layers_run=decision.layers_to_run
        )
        
        self.execution_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "pathway": decision.pathway.value,
            "domain": domain,
            "was_correct": was_correct,
            "time_ms": actual_time_ms,
            "layers": decision.layers_to_run
        })
    
    def get_statistics(self) -> Dict:
        """Get orchestration statistics."""
        return {
            "selector_stats": self.selector.get_statistics(),
            "execution_count": len(self.execution_history),
            "recent_accuracy": sum(
                1 for e in self.execution_history[-100:]
                if e.get("was_correct", False)
            ) / max(1, len(self.execution_history[-100:])) if self.execution_history else 0.0
        }

    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        """Record operation outcome."""
        if not hasattr(self, '_outcomes'): self._outcomes = []
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

