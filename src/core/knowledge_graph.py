"""
BASE Bias-Aware Knowledge Graph (PPA1-Inv6)

Implements a knowledge graph with bias awareness:
1. Entity extraction and linking
2. Relationship mapping
3. Bias tracking in knowledge nodes
4. Temporal knowledge decay

Patent Alignment:
- PPA1-Inv6: Bias-Aware Knowledge Graph
- Brain Layer: 3 (Limbic System)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from enum import Enum
import hashlib


class EntityType(Enum):
    """Types of knowledge entities."""
    CONCEPT = "concept"
    PERSON = "person"
    ORGANIZATION = "organization"
    EVENT = "event"
    CLAIM = "claim"
    OPINION = "opinion"
    FACT = "fact"


class RelationType(Enum):
    """Types of relationships."""
    IS_A = "is_a"
    HAS_PROPERTY = "has_property"
    RELATED_TO = "related_to"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    CAUSES = "causes"
    BIASED_TOWARD = "biased_toward"


@dataclass
class KnowledgeNode:
    """A node in the knowledge graph."""
    node_id: str
    entity_type: EntityType
    content: str
    bias_scores: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = "unknown"
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0


@dataclass
class KnowledgeEdge:
    """An edge (relationship) in the knowledge graph."""
    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 1.0
    bias_indicator: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class KnowledgeQueryResult:
    """Result of querying the knowledge graph."""
    nodes: List[KnowledgeNode]
    edges: List[KnowledgeEdge]
    bias_warnings: List[str]
    confidence: float


class BiasAwareKnowledgeGraph:
    """
    Bias-Aware Knowledge Graph for BASE.
    
    Implements PPA1-Inv6: Bias-Aware Knowledge Graph
    Brain Layer: 3 (Limbic System)
    
    Capabilities:
    1. Store and retrieve knowledge entities
    2. Track relationships with bias awareness
    3. Apply temporal decay to stale knowledge
    4. Detect potential bias in knowledge paths
    """
    
    def __init__(self, decay_rate: float = 0.01):
        """
        Initialize the knowledge graph.
        
        Args:
            decay_rate: Rate at which confidence decays over time
        """
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []
        self.decay_rate = decay_rate
        
        # Indexes for efficient lookup
        self._content_index: Dict[str, str] = {}  # content hash -> node_id
        self._type_index: Dict[EntityType, Set[str]] = {}
        
        # Learning state
        self._outcomes: List[Dict] = []
        self._feedback: List[Dict] = []
        self._domain_adjustments: Dict[str, float] = {}
        self._total_queries: int = 0
        self._query_accuracy: List[bool] = []
    
    def add_node(self, content: str, entity_type: EntityType,
                 bias_scores: Dict[str, float] = None,
                 source: str = "unknown") -> str:
        """
        Add a knowledge node.
        
        Args:
            content: Content of the node
            entity_type: Type of entity
            bias_scores: Optional bias scores by dimension
            source: Source of the knowledge
            
        Returns:
            Node ID
        """
        # Generate unique ID
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        node_id = f"{entity_type.value}_{content_hash}"
        
        # Check for duplicates
        if content_hash in self._content_index:
            existing_id = self._content_index[content_hash]
            existing = self.nodes[existing_id]
            existing.access_count += 1
            existing.last_accessed = datetime.now()
            return existing_id
        
        # Create node
        node = KnowledgeNode(
            node_id=node_id,
            entity_type=entity_type,
            content=content,
            bias_scores=bias_scores or {},
            source=source
        )
        
        self.nodes[node_id] = node
        self._content_index[content_hash] = node_id
        
        # Update type index
        if entity_type not in self._type_index:
            self._type_index[entity_type] = set()
        self._type_index[entity_type].add(node_id)
        
        return node_id
    
    def add_edge(self, source_id: str, target_id: str,
                 relation_type: RelationType, weight: float = 1.0,
                 bias_indicator: str = None) -> bool:
        """
        Add an edge between nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation_type: Type of relationship
            weight: Edge weight
            bias_indicator: Optional bias indicator
            
        Returns:
            True if edge was added
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            return False
        
        edge = KnowledgeEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            bias_indicator=bias_indicator
        )
        
        self.edges.append(edge)
        return True
    
    def query(self, query: str, entity_types: List[EntityType] = None,
              max_hops: int = 2) -> KnowledgeQueryResult:
        """
        Query the knowledge graph.
        
        Args:
            query: Query string
            entity_types: Filter by entity types
            max_hops: Maximum relationship hops
            
        Returns:
            KnowledgeQueryResult with matching nodes and edges
        """
        self._total_queries += 1
        query_lower = query.lower()
        
        # Find matching nodes
        matching_nodes = []
        for node in self.nodes.values():
            if entity_types and node.entity_type not in entity_types:
                continue
            if any(term in node.content.lower() for term in query_lower.split()):
                matching_nodes.append(node)
                node.access_count += 1
                node.last_accessed = datetime.now()
        
        # Find related edges
        node_ids = {n.node_id for n in matching_nodes}
        related_edges = []
        
        for hop in range(max_hops):
            new_ids = set()
            for edge in self.edges:
                if edge.source_id in node_ids:
                    related_edges.append(edge)
                    new_ids.add(edge.target_id)
                elif edge.target_id in node_ids:
                    related_edges.append(edge)
                    new_ids.add(edge.source_id)
            node_ids.update(new_ids)
        
        # Collect bias warnings
        bias_warnings = self._check_bias_warnings(matching_nodes, related_edges)
        
        # Calculate confidence with decay
        confidence = self._calculate_confidence(matching_nodes)
        
        return KnowledgeQueryResult(
            nodes=matching_nodes,
            edges=related_edges,
            bias_warnings=bias_warnings,
            confidence=confidence
        )
    
    def _check_bias_warnings(self, nodes: List[KnowledgeNode],
                              edges: List[KnowledgeEdge]) -> List[str]:
        """Check for bias warnings in knowledge path."""
        warnings = []
        
        # Check node bias scores
        for node in nodes:
            for dimension, score in node.bias_scores.items():
                if score > 0.7:
                    warnings.append(f"High {dimension} bias ({score:.2f}) in: {node.content[:50]}")
        
        # Check edge bias indicators
        for edge in edges:
            if edge.bias_indicator:
                warnings.append(f"Relationship has bias indicator: {edge.bias_indicator}")
            if edge.relation_type == RelationType.BIASED_TOWARD:
                warnings.append(f"Biased relationship detected in knowledge path")
        
        return warnings
    
    def _calculate_confidence(self, nodes: List[KnowledgeNode]) -> float:
        """Calculate confidence with temporal decay."""
        if not nodes:
            return 0.0
        
        now = datetime.now()
        total_confidence = 0.0
        
        for node in nodes:
            # Apply temporal decay
            age_hours = (now - node.created_at).total_seconds() / 3600
            decay_factor = max(0.1, 1.0 - self.decay_rate * age_hours)
            total_confidence += node.confidence * decay_factor
        
        return total_confidence / len(nodes)
    
    def get_bias_summary(self) -> Dict[str, Any]:
        """Get summary of biases in the knowledge graph."""
        bias_counts: Dict[str, List[float]] = {}
        
        for node in self.nodes.values():
            for dimension, score in node.bias_scores.items():
                if dimension not in bias_counts:
                    bias_counts[dimension] = []
                bias_counts[dimension].append(score)
        
        summary = {}
        for dimension, scores in bias_counts.items():
            summary[dimension] = {
                'count': len(scores),
                'avg_score': sum(scores) / len(scores),
                'max_score': max(scores),
                'min_score': min(scores)
            }
        
        return summary
    
    # ===== Learning Interface Methods =====
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        """Record query outcome for learning."""
        self._outcomes.append(outcome)
        if 'query_useful' in outcome:
            self._query_accuracy.append(outcome['query_useful'])
    
    def record_feedback(self, feedback: Dict) -> None:
        """Record feedback on knowledge graph results."""
        self._feedback.append(feedback)
        domain = feedback.get('domain', 'general')
        if feedback.get('result_inaccurate', False):
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) - 0.05
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt thresholds."""
        pass
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get domain adjustment."""
        return self._domain_adjustments.get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics."""
        query_acc = sum(self._query_accuracy) / max(len(self._query_accuracy), 1)
        
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'total_queries': self._total_queries,
            'query_accuracy': query_acc,
            'bias_summary': self.get_bias_summary(),
            'domain_adjustments': dict(self._domain_adjustments),
            'outcomes_recorded': len(self._outcomes),
            'feedback_recorded': len(self._feedback)
        }


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
    kg = BiasAwareKnowledgeGraph()
    
    # Add some test nodes
    ai_id = kg.add_node("Artificial Intelligence", EntityType.CONCEPT, 
                        bias_scores={'gender': 0.3, 'cultural': 0.2})
    ml_id = kg.add_node("Machine Learning", EntityType.CONCEPT,
                        bias_scores={'technical': 0.1})
    claim_id = kg.add_node("AI will replace all jobs", EntityType.OPINION,
                           bias_scores={'sensationalism': 0.8})
    
    # Add edges
    kg.add_edge(ml_id, ai_id, RelationType.IS_A)
    kg.add_edge(claim_id, ai_id, RelationType.RELATED_TO, bias_indicator="unverified claim")
    
    # Query
    result = kg.query("artificial intelligence")
    
    print("=" * 60)
    print("BIAS-AWARE KNOWLEDGE GRAPH TEST")
    print("=" * 60)
    print(f"Query: 'artificial intelligence'")
    print(f"Matching nodes: {len(result.nodes)}")
    print(f"Related edges: {len(result.edges)}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"\nBias Warnings: {result.bias_warnings}")
    print(f"\nLearning stats: {kg.get_learning_statistics()}")

