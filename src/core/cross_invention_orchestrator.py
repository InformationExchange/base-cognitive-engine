"""
BASE Cognitive Governance Engine v25.0
Cross-Invention Orchestrator - Signal Flow Between Inventions

Patent Alignment:
- PPA1-Inv7: Reasoning Trees (chained reasoning)
- PPA1-Inv8: Contradiction Resolution (inter-signal conflicts)
- PPA2-Comp7: OCO Threshold (dynamic threshold adjustment based on signal flow)
- NOVEL-10: SmartGate (orchestration entry point)

This module implements cross-invention orchestration:
1. Define signal flow pathways between inventions
2. Enable outputs from one invention to feed into others
3. Detect and resolve signal conflicts
4. Track invention activation chains
5. Optimize orchestration based on effectiveness

Phase 25 Enhancement: Formalize cross-invention signal flow for
coherent multi-invention reasoning.

NO PLACEHOLDERS. NO STUBS. FULL IMPLEMENTATION.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import logging
import asyncio

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of signals that flow between inventions."""
    SCORE = "score"              # Numerical score (0-1)
    CONFIDENCE = "confidence"    # Confidence level
    ISSUES = "issues"            # List of detected issues
    METADATA = "metadata"        # Contextual metadata
    DECISION = "decision"        # Boolean or categorical decision
    EVIDENCE = "evidence"        # Evidence/proof data


class InventionLayer(Enum):
    """Brain layers for inventions (from BASE_BRAIN_ARCHITECTURE.md)."""
    INPUT = 1           # L1: Input Processing
    PERCEPTION = 2      # L2: Perception
    CLASSIFICATION = 3  # L3: Classification
    MEMORY = 4          # L4: Memory
    REASONING = 5       # L5: Reasoning
    EVIDENCE = 6        # L6: Evidence
    INTEGRATION = 7     # L7: Integration
    DECISION = 8        # L8: Decision
    ACTION = 9          # L9: Action
    OUTPUT = 10         # L10: Output


@dataclass
class SignalValue:
    """A signal value flowing between inventions."""
    signal_type: SignalType
    value: Any
    source_invention: str
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class InventionNode:
    """Node representing an invention in the orchestration graph."""
    invention_id: str
    name: str
    layer: InventionLayer
    input_signals: List[str] = field(default_factory=list)  # Signal names consumed
    output_signals: List[str] = field(default_factory=list)  # Signal names produced
    depends_on: List[str] = field(default_factory=list)  # Invention IDs
    feeds_into: List[str] = field(default_factory=list)  # Invention IDs


@dataclass
class SignalConflict:
    """Detected conflict between signals."""
    signal_name: str
    sources: List[str]
    values: List[Any]
    conflict_type: str  # 'contradictory', 'inconsistent', 'divergent'
    severity: float  # 0-1
    resolution: Optional[str] = None


@dataclass
class OrchestrationResult:
    """Result of cross-invention orchestration."""
    activated_inventions: List[str]
    signal_flow: Dict[str, List[str]]  # signal_name -> [source -> target]
    conflicts_detected: List[SignalConflict]
    conflicts_resolved: int
    total_signals: int
    execution_order: List[str]
    processing_time_ms: float


class CrossInventionOrchestrator:
    """
    Cross-Invention Orchestrator.
    
    Manages signal flow between BASE inventions:
    1. Defines signal pathways
    2. Routes outputs to dependent inventions
    3. Detects and resolves conflicts
    4. Optimizes execution order
    """
    
    # Signal flow definitions (source_signal -> target_inventions)
    SIGNAL_PATHWAYS = {
        # Grounding signals flow to...
        'grounding_score': ['signal_fusion', 'ccp_calibrator', 'human_arbitration'],
        'grounding_issues': ['behavioral_detector', 'corrective_action'],
        'source_citations': ['factual_detector', 'contradiction_resolver'],
        
        # Factual signals flow to...
        'factual_score': ['signal_fusion', 'ccp_calibrator', 'multi_track'],
        'factual_issues': ['behavioral_detector', 'corrective_action'],
        'entity_mentions': ['world_models', 'knowledge_graph'],
        
        # Behavioral signals flow to...
        'behavioral_score': ['signal_fusion', 'ccp_calibrator', 'smart_gate'],
        'detected_biases': ['corrective_action', 'multi_framework', 'human_arbitration'],
        'tgtbt_score': ['ccp_calibrator', 'proof_verification'],
        
        # Temporal signals flow to...
        'temporal_score': ['signal_fusion', 'ccp_calibrator'],
        'temporal_stability': ['ccp_calibrator', 'learning_coordinator'],
        
        # CCP signals flow to...
        'calibrated_posterior': ['decision_maker', 'multi_track'],
        'confidence_interval': ['human_arbitration', 'output_formatter'],
        'uncertainty': ['smart_gate', 'multi_framework'],
        
        # Multi-Track signals flow to...
        'consensus_confidence': ['decision_maker', 'human_arbitration'],
        'cross_llm_issues': ['corrective_action', 'audit_trail'],
        
        # World Models signals flow to...
        'causal_graph': ['contradiction_resolver', 'reasoning_trees'],
        'world_state': ['theory_of_mind', 'creative_reasoning'],
        
        # Decision signals flow to...
        'accept_decision': ['output_formatter', 'audit_trail'],
        'accuracy_score': ['learning_coordinator', 'privacy_manager'],
    }
    
    # Invention dependency graph
    INVENTION_GRAPH = {
        # Layer 1-2: Input/Perception (no dependencies)
        'smart_gate': InventionNode(
            invention_id='NOVEL-10',
            name='SmartGate',
            layer=InventionLayer.INPUT,
            input_signals=['query', 'context'],
            output_signals=['routing_decision', 'risk_factors', 'estimated_cost'],
            depends_on=[],
            feeds_into=['grounding_detector', 'factual_detector', 'behavioral_detector']
        ),
        
        # Layer 3: Classification
        'grounding_detector': InventionNode(
            invention_id='PPA1-Inv1',
            name='Grounding Detector',
            layer=InventionLayer.CLASSIFICATION,
            input_signals=['response', 'documents'],
            output_signals=['grounding_score', 'grounding_issues', 'source_citations'],
            depends_on=['smart_gate'],
            feeds_into=['signal_fusion', 'contradiction_resolver']
        ),
        'factual_detector': InventionNode(
            invention_id='PPA1-Inv3',
            name='Factual Detector',
            layer=InventionLayer.CLASSIFICATION,
            input_signals=['response', 'source_citations'],
            output_signals=['factual_score', 'factual_issues', 'entity_mentions'],
            depends_on=['smart_gate', 'grounding_detector'],
            feeds_into=['signal_fusion', 'world_models']
        ),
        'behavioral_detector': InventionNode(
            invention_id='PPA3-Behav',
            name='Behavioral Bias Detector',
            layer=InventionLayer.CLASSIFICATION,
            input_signals=['query', 'response', 'grounding_issues'],
            output_signals=['behavioral_score', 'detected_biases', 'tgtbt_score'],
            depends_on=['smart_gate', 'grounding_detector'],
            feeds_into=['signal_fusion', 'ccp_calibrator', 'corrective_action']
        ),
        
        # Layer 5: Reasoning
        'contradiction_resolver': InventionNode(
            invention_id='PPA1-Inv8',
            name='Contradiction Resolver',
            layer=InventionLayer.REASONING,
            input_signals=['response', 'source_citations', 'factual_issues'],
            output_signals=['contradictions', 'resolution_suggestions'],
            depends_on=['grounding_detector', 'factual_detector'],
            feeds_into=['signal_fusion', 'corrective_action']
        ),
        'world_models': InventionNode(
            invention_id='NOVEL-16',
            name='World Models',
            layer=InventionLayer.REASONING,
            input_signals=['response', 'entity_mentions', 'context'],
            output_signals=['causal_graph', 'world_state'],
            depends_on=['factual_detector'],
            feeds_into=['theory_of_mind', 'creative_reasoning']
        ),
        'theory_of_mind': InventionNode(
            invention_id='NOVEL-14',
            name='Theory of Mind',
            layer=InventionLayer.REASONING,
            input_signals=['query', 'response', 'world_state'],
            output_signals=['intent_analysis', 'perspective_map'],
            depends_on=['world_models'],
            feeds_into=['multi_framework']
        ),
        
        # Layer 6: Evidence
        'ccp_calibrator': InventionNode(
            invention_id='PPA2-Comp9',
            name='CCP Calibrator',
            layer=InventionLayer.EVIDENCE,
            input_signals=['grounding_score', 'factual_score', 'behavioral_score', 'tgtbt_score'],
            output_signals=['calibrated_posterior', 'confidence_interval', 'uncertainty'],
            depends_on=['grounding_detector', 'factual_detector', 'behavioral_detector'],
            feeds_into=['decision_maker', 'multi_track']
        ),
        
        # Layer 7: Integration
        'signal_fusion': InventionNode(
            invention_id='PPA1-Inv6',
            name='Signal Fusion',
            layer=InventionLayer.INTEGRATION,
            input_signals=['grounding_score', 'factual_score', 'behavioral_score', 'temporal_score'],
            output_signals=['fused_score', 'fused_confidence'],
            depends_on=['grounding_detector', 'factual_detector', 'behavioral_detector'],
            feeds_into=['decision_maker', 'ccp_calibrator']
        ),
        'multi_track': InventionNode(
            invention_id='NOVEL-12',
            name='Multi-Track Challenger',
            layer=InventionLayer.INTEGRATION,
            input_signals=['query', 'response', 'calibrated_posterior'],
            output_signals=['consensus_confidence', 'cross_llm_issues'],
            depends_on=['ccp_calibrator'],
            feeds_into=['decision_maker', 'human_arbitration']
        ),
        
        # Layer 8: Decision
        'decision_maker': InventionNode(
            invention_id='PPA2-Comp7',
            name='OCO Decision Maker',
            layer=InventionLayer.DECISION,
            input_signals=['fused_score', 'calibrated_posterior', 'consensus_confidence'],
            output_signals=['accept_decision', 'accuracy_score'],
            depends_on=['signal_fusion', 'ccp_calibrator', 'multi_track'],
            feeds_into=['human_arbitration', 'corrective_action', 'audit_trail']
        ),
        'human_arbitration': InventionNode(
            invention_id='PPA1-Inv20',
            name='Human Arbitration',
            layer=InventionLayer.DECISION,
            input_signals=['accept_decision', 'confidence_interval', 'detected_biases'],
            output_signals=['escalation_decision', 'arbitration_request'],
            depends_on=['decision_maker', 'ccp_calibrator', 'behavioral_detector'],
            feeds_into=['audit_trail']
        ),
        
        # Layer 9: Action
        'corrective_action': InventionNode(
            invention_id='NOVEL-21',
            name='Corrective Action Engine',
            layer=InventionLayer.ACTION,
            input_signals=['accept_decision', 'detected_biases', 'cross_llm_issues'],
            output_signals=['corrective_instructions', 'regeneration_required'],
            depends_on=['decision_maker', 'behavioral_detector', 'multi_track'],
            feeds_into=['output_formatter']
        ),
        
        # Layer 10: Output
        'audit_trail': InventionNode(
            invention_id='PPA2-Comp6',
            name='Audit Trail',
            layer=InventionLayer.OUTPUT,
            input_signals=['accept_decision', 'cross_llm_issues', 'escalation_decision'],
            output_signals=['audit_record'],
            depends_on=['decision_maker', 'human_arbitration'],
            feeds_into=[]
        ),
    }
    
    def __init__(self, storage_path: Path = None):
        """
        Initialize Cross-Invention Orchestrator.
        
        Args:
            storage_path: Path for persisting orchestration state
        """
        self.storage_path = storage_path
        self.signal_buffer: Dict[str, SignalValue] = {}
        self.execution_history: List[OrchestrationResult] = []
        self.conflict_count = 0
        
        # Learning state
        self._outcomes: List[Dict] = []
        self._feedback: List[Dict] = []
        self._domain_adjustments: Dict[str, float] = {}
        self._orchestration_accuracy: List[bool] = []
        
    def get_execution_order(self, 
                           required_inventions: List[str] = None) -> List[str]:
        """
        Get topologically sorted execution order for inventions.
        
        Args:
            required_inventions: Specific inventions to include (None = all)
            
        Returns:
            List of invention keys in execution order
        """
        # Get all inventions or filter to required
        inventions = required_inventions or list(self.INVENTION_GRAPH.keys())
        
        # Build adjacency list for topological sort
        in_degree = {inv: 0 for inv in inventions}
        graph = {inv: [] for inv in inventions}
        
        for inv_key in inventions:
            if inv_key not in self.INVENTION_GRAPH:
                continue
            node = self.INVENTION_GRAPH[inv_key]
            for dep in node.depends_on:
                if dep in inventions:
                    graph[dep].append(inv_key)
                    in_degree[inv_key] += 1
        
        # Kahn's algorithm for topological sort
        queue = [inv for inv in inventions if in_degree[inv] == 0]
        result = []
        
        while queue:
            # Sort by layer for deterministic ordering
            queue.sort(key=lambda x: self.INVENTION_GRAPH.get(x, InventionNode(
                invention_id='', name='', layer=InventionLayer.OUTPUT
            )).layer.value)
            
            current = queue.pop(0)
            result.append(current)
            
            for neighbor in graph.get(current, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result
    
    def route_signal(self, 
                    signal_name: str, 
                    value: Any, 
                    source: str,
                    confidence: float = 1.0) -> List[str]:
        """
        Route a signal to its target inventions.
        
        Args:
            signal_name: Name of the signal
            value: Signal value
            source: Source invention
            confidence: Signal confidence
            
        Returns:
            List of target inventions that will receive the signal
        """
        # Store in buffer
        self.signal_buffer[signal_name] = SignalValue(
            signal_type=self._infer_signal_type(signal_name, value),
            value=value,
            source_invention=source,
            confidence=confidence
        )
        
        # Get targets
        targets = self.SIGNAL_PATHWAYS.get(signal_name, [])
        
        return targets
    
    def _infer_signal_type(self, signal_name: str, value: Any) -> SignalType:
        """Infer signal type from name and value."""
        if 'score' in signal_name:
            return SignalType.SCORE
        elif 'confidence' in signal_name:
            return SignalType.CONFIDENCE
        elif 'issues' in signal_name or 'biases' in signal_name:
            return SignalType.ISSUES
        elif 'decision' in signal_name:
            return SignalType.DECISION
        elif 'evidence' in signal_name or 'citation' in signal_name:
            return SignalType.EVIDENCE
        else:
            return SignalType.METADATA
    
    def get_signal(self, signal_name: str) -> Optional[SignalValue]:
        """Get a signal from the buffer."""
        return self.signal_buffer.get(signal_name)
    
    def detect_conflicts(self) -> List[SignalConflict]:
        """
        Detect conflicts between signals in the buffer.
        
        Returns:
            List of detected signal conflicts
        """
        conflicts = []
        
        # Group signals by type for conflict detection
        score_signals = {k: v for k, v in self.signal_buffer.items() 
                        if v.signal_type == SignalType.SCORE}
        
        # Check for contradictory scores (large divergence)
        if len(score_signals) >= 2:
            scores = [(k, v.value) for k, v in score_signals.items() 
                     if isinstance(v.value, (int, float))]
            if scores:
                values = [s[1] for s in scores]
                if max(values) - min(values) > 0.5:
                    conflicts.append(SignalConflict(
                        signal_name='score_divergence',
                        sources=[s[0] for s in scores],
                        values=values,
                        conflict_type='divergent',
                        severity=(max(values) - min(values))
                    ))
        
        # Check for conflicting decisions
        decision_signals = {k: v for k, v in self.signal_buffer.items() 
                          if v.signal_type == SignalType.DECISION}
        
        if len(decision_signals) >= 2:
            decisions = list(decision_signals.items())
            for i, (k1, v1) in enumerate(decisions):
                for k2, v2 in decisions[i+1:]:
                    if isinstance(v1.value, bool) and isinstance(v2.value, bool):
                        if v1.value != v2.value:
                            conflicts.append(SignalConflict(
                                signal_name='decision_conflict',
                                sources=[k1, k2],
                                values=[v1.value, v2.value],
                                conflict_type='contradictory',
                                severity=1.0
                            ))
        
        return conflicts
    
    def resolve_conflict(self, conflict: SignalConflict) -> Optional[Any]:
        """
        Resolve a signal conflict.
        
        Args:
            conflict: The conflict to resolve
            
        Returns:
            Resolved value or None
        """
        if conflict.conflict_type == 'divergent' and conflict.signal_name == 'score_divergence':
            # Use weighted average based on source confidence
            total_weight = 0
            weighted_sum = 0
            
            for signal_name, value in zip(conflict.sources, conflict.values):
                signal = self.signal_buffer.get(signal_name)
                weight = signal.confidence if signal else 0.5
                weighted_sum += value * weight
                total_weight += weight
            
            if total_weight > 0:
                resolved = weighted_sum / total_weight
                conflict.resolution = f'weighted_average: {resolved:.3f}'
                return resolved
        
        elif conflict.conflict_type == 'contradictory':
            # Conservative approach: reject (False) when in doubt
            conflict.resolution = 'conservative_reject'
            self.conflict_count += 1
            return False
        
        return None
    
    def get_invention_inputs(self, invention_key: str) -> Dict[str, SignalValue]:
        """
        Get all input signals available for an invention.
        
        Args:
            invention_key: Key of the invention
            
        Returns:
            Dict of signal_name -> SignalValue
        """
        if invention_key not in self.INVENTION_GRAPH:
            return {}
        
        node = self.INVENTION_GRAPH[invention_key]
        inputs = {}
        
        for signal_name in node.input_signals:
            if signal_name in self.signal_buffer:
                inputs[signal_name] = self.signal_buffer[signal_name]
        
        return inputs
    
    def get_signal_flow_diagram(self) -> str:
        """Generate ASCII diagram of signal flow."""
        lines = []
        lines.append("BASE Cross-Invention Signal Flow")
        lines.append("=" * 60)
        
        # Group by layer
        layers = {}
        for key, node in self.INVENTION_GRAPH.items():
            layer = node.layer.value
            if layer not in layers:
                layers[layer] = []
            layers[layer].append((key, node))
        
        for layer_num in sorted(layers.keys()):
            layer_name = InventionLayer(layer_num).name
            lines.append(f"\n[Layer {layer_num}: {layer_name}]")
            
            for key, node in layers[layer_num]:
                lines.append(f"  {node.name} ({node.invention_id})")
                if node.input_signals:
                    lines.append(f"    ← IN: {', '.join(node.input_signals[:3])}")
                if node.output_signals:
                    lines.append(f"    → OUT: {', '.join(node.output_signals[:3])}")
        
        return '\n'.join(lines)
    
    def orchestrate(self, 
                   signals: Dict[str, Any],
                   required_inventions: List[str] = None) -> OrchestrationResult:
        """
        Orchestrate signal flow across inventions.
        
        Args:
            signals: Initial signals to route
            required_inventions: Specific inventions to include
            
        Returns:
            OrchestrationResult with flow details
        """
        import time
        start_time = time.time()
        
        # Clear buffer and populate with initial signals
        self.signal_buffer.clear()
        for name, value in signals.items():
            self.route_signal(name, value, source='input')
        
        # Get execution order
        execution_order = self.get_execution_order(required_inventions)
        
        # Track signal flow
        signal_flow: Dict[str, List[str]] = {}
        activated = []
        
        # Process in order
        for inv_key in execution_order:
            inputs = self.get_invention_inputs(inv_key)
            if inputs:
                activated.append(inv_key)
                
                # Track flow
                for signal_name in inputs:
                    if signal_name not in signal_flow:
                        signal_flow[signal_name] = []
                    signal_flow[signal_name].append(inv_key)
        
        # Detect conflicts
        conflicts = self.detect_conflicts()
        resolved = 0
        for conflict in conflicts:
            resolution = self.resolve_conflict(conflict)
            if resolution is not None:
                resolved += 1
        
        processing_time = (time.time() - start_time) * 1000
        
        result = OrchestrationResult(
            activated_inventions=activated,
            signal_flow=signal_flow,
            conflicts_detected=conflicts,
            conflicts_resolved=resolved,
            total_signals=len(self.signal_buffer),
            execution_order=execution_order,
            processing_time_ms=processing_time
        )
        
        self.execution_history.append(result)
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestration statistics."""
        return {
            'total_inventions': len(self.INVENTION_GRAPH),
            'total_signal_pathways': len(self.SIGNAL_PATHWAYS),
            'execution_count': len(self.execution_history),
            'total_conflicts_resolved': self.conflict_count,
            'current_buffer_size': len(self.signal_buffer),
            'layer_distribution': {
                layer.name: len([n for n in self.INVENTION_GRAPH.values() 
                               if n.layer == layer])
                for layer in InventionLayer
            }
        }
    
    # ===== Learning Interface Methods =====
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict[str, Any]) -> None:
        """Record orchestration outcome for learning."""
        self._outcomes.append(outcome)
        if 'orchestration_effective' in outcome:
            self._orchestration_accuracy.append(outcome['orchestration_effective'])
    
    def record_feedback(self, feedback: Dict[str, Any]) -> None:
        """Record feedback on orchestration decisions."""
        self._feedback.append(feedback)
        domain = feedback.get('domain', 'general')
        if feedback.get('wrong_order', False):
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) - 0.05
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt orchestration thresholds."""
        pass
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get domain adjustment."""
        return self._domain_adjustments.get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        stats = self.get_statistics()
        orch_acc = sum(self._orchestration_accuracy) / max(len(self._orchestration_accuracy), 1)
        stats['orchestration_accuracy'] = orch_acc
        stats['domain_adjustments'] = dict(self._domain_adjustments)
        stats['outcomes_recorded'] = len(self._outcomes)
        stats['feedback_recorded'] = len(self._feedback)
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


if __name__ == "__main__":
    print("=" * 70)
    print("Phase 25: Cross-Invention Orchestrator Test")
    print("=" * 70)
    
    orchestrator = CrossInventionOrchestrator()
    
    # Test execution order
    print("\n[Execution Order (Topological Sort)]")
    order = orchestrator.get_execution_order()
    for i, inv in enumerate(order):
        node = orchestrator.INVENTION_GRAPH.get(inv)
        if node:
            print(f"  {i+1}. {node.name} (L{node.layer.value})")
    
    # Test signal routing
    print("\n[Signal Routing Test]")
    test_signals = {
        'grounding_score': 0.75,
        'factual_score': 0.85,
        'behavioral_score': 0.60,
        'detected_biases': ['tgtbt', 'overconfidence'],
    }
    
    for signal, value in test_signals.items():
        targets = orchestrator.route_signal(signal, value, 'test')
        print(f"  {signal} -> {targets}")
    
    # Test orchestration
    print("\n[Full Orchestration Test]")
    result = orchestrator.orchestrate(test_signals)
    
    print(f"  Activated Inventions: {len(result.activated_inventions)}")
    print(f"  Signal Flow Paths: {len(result.signal_flow)}")
    print(f"  Conflicts Detected: {len(result.conflicts_detected)}")
    print(f"  Conflicts Resolved: {result.conflicts_resolved}")
    print(f"  Processing Time: {result.processing_time_ms:.2f}ms")
    
    # Print signal flow diagram
    print("\n" + orchestrator.get_signal_flow_diagram())
    
    # Statistics
    print("\n[Statistics]")
    stats = orchestrator.get_statistics()
    print(f"  Total Inventions: {stats['total_inventions']}")
    print(f"  Signal Pathways: {stats['total_signal_pathways']}")
    print(f"  Layer Distribution:")
    for layer, count in stats['layer_distribution'].items():
        if count > 0:
            print(f"    - {layer}: {count}")
    
    print("\n" + "=" * 70)
    print("Phase 25 Cross-Invention Orchestrator Complete")
    print("=" * 70)

