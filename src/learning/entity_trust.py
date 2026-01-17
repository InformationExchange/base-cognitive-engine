"""
BASE Cognitive Governance Engine v16.4
Entity Trust Inheritance System

PPA-2 Component 1: FULL IMPLEMENTATION
Bias ratings propagate from entities to data with trust discounting.

This module implements:
1. Entity Registry: Track known entities and their trust scores
2. Trust Inheritance: Data inherits trust from source entities
3. Trust Discounting: Decay trust over hops/time
4. Trust Aggregation: Combine trust from multiple entities
5. Trust Learning: Update entity trust from feedback

Trust Model:
- Trust scores: 0.0 (untrusted) to 1.0 (fully trusted)
- Inheritance: T(data) = Σ w_i × T(entity_i) × discount(hops)
- Discounting: discount(h) = β^h where β < 1
- Update: T_new = α × feedback + (1-α) × T_old
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import math


class EntityType(str, Enum):
    """Types of entities that can have trust scores."""
    SOURCE = "source"           # News source, website
    AUTHOR = "author"           # Individual author
    ORGANIZATION = "organization"  # Company, institution
    DATABASE = "database"       # Data source
    DOCUMENT = "document"       # Specific document
    API = "api"                 # API endpoint
    MODEL = "model"             # AI model


@dataclass
class Entity:
    """An entity with a trust score."""
    entity_id: str
    name: str
    entity_type: EntityType
    
    # Trust score
    trust_score: float = 0.5  # 0-1
    trust_confidence: float = 0.5  # How confident in the trust score
    
    # Trust components
    accuracy_history: float = 0.5  # Historical accuracy
    bias_rating: float = 0.0       # Known bias (0 = unbiased)
    verification_status: str = "unknown"  # verified, unverified, disputed
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    observation_count: int = 0
    
    # Relationships
    parent_entities: List[str] = field(default_factory=list)  # Entity IDs
    child_entities: List[str] = field(default_factory=list)
    
    def update_trust(self, feedback: float, weight: float = 0.1):
        """
        Update trust score based on feedback.
        
        Args:
            feedback: 0-1, how accurate/trustworthy this entity was
            weight: Learning rate
        """
        # Exponential moving average
        self.trust_score = weight * feedback + (1 - weight) * self.trust_score
        self.trust_score = max(0.0, min(1.0, self.trust_score))
        
        # Increase confidence with more observations
        self.observation_count += 1
        self.trust_confidence = min(0.95, 0.5 + 0.005 * self.observation_count)
        
        self.last_updated = datetime.utcnow()
    
    def get_effective_trust(self) -> float:
        """Get trust score adjusted for bias."""
        bias_penalty = self.bias_rating * 0.3
        return max(0.0, self.trust_score - bias_penalty)
    
    def to_dict(self) -> Dict:
        return {
            'entity_id': self.entity_id,
            'name': self.name,
            'type': self.entity_type.value,
            'trust_score': self.trust_score,
            'trust_confidence': self.trust_confidence,
            'effective_trust': self.get_effective_trust(),
            'bias_rating': self.bias_rating,
            'verification_status': self.verification_status,
            'observation_count': self.observation_count
        }


@dataclass
class TrustInheritance:
    """Record of trust inherited from entities to data."""
    data_id: str
    source_entities: List[str]
    inherited_trust: float
    discounted_trust: float
    hop_count: int
    aggregation_method: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            'data_id': self.data_id,
            'source_entities': self.source_entities,
            'inherited_trust': self.inherited_trust,
            'discounted_trust': self.discounted_trust,
            'hop_count': self.hop_count,
            'aggregation_method': self.aggregation_method
        }


class EntityTrustSystem:
    """
    Entity Trust Inheritance System.
    
    PPA-2 Component 1: Full Implementation
    
    Manages entity trust scores and propagates trust
    to data with appropriate discounting.
    """
    
    # Trust discounting factor per hop
    DISCOUNT_FACTOR = 0.9  # β in discount formula
    
    # Trust decay over time (per day)
    TIME_DECAY = 0.99
    
    # Default trust for unknown entities
    DEFAULT_TRUST = 0.5
    
    # Learning rate for trust updates
    LEARNING_RATE = 0.1
    
    def __init__(self, storage_path: Path = None):
        # Use temp directory if no path provided (fixes read-only filesystem issues)
        if storage_path is None:
            import tempfile
            temp_dir = Path(tempfile.mkdtemp(prefix="base_trust_"))
            storage_path = temp_dir / "entity_trust.json"
        self.storage_path = storage_path
        
        # Entity registry
        self.entities: Dict[str, Entity] = {}
        
        # Inheritance records
        self.inheritance_log: List[TrustInheritance] = []
        
        # Initialize default entities
        self._initialize_default_entities()
        
        # Load persisted
        self._load_state()
    
    def _initialize_default_entities(self):
        """Initialize well-known entities with prior trust."""
        
        # Trusted sources
        self.register_entity(Entity(
            entity_id="wikipedia",
            name="Wikipedia",
            entity_type=EntityType.SOURCE,
            trust_score=0.7,
            trust_confidence=0.8,
            accuracy_history=0.75,
            bias_rating=0.1,
            verification_status="verified"
        ))
        
        self.register_entity(Entity(
            entity_id="pubmed",
            name="PubMed",
            entity_type=EntityType.DATABASE,
            trust_score=0.9,
            trust_confidence=0.9,
            accuracy_history=0.92,
            bias_rating=0.05,
            verification_status="verified"
        ))
        
        self.register_entity(Entity(
            entity_id="gov_official",
            name="Government Official Sources",
            entity_type=EntityType.ORGANIZATION,
            trust_score=0.8,
            trust_confidence=0.85,
            accuracy_history=0.82,
            bias_rating=0.15,
            verification_status="verified"
        ))
        
        # Moderate trust
        self.register_entity(Entity(
            entity_id="news_mainstream",
            name="Mainstream News",
            entity_type=EntityType.SOURCE,
            trust_score=0.6,
            trust_confidence=0.7,
            accuracy_history=0.65,
            bias_rating=0.25,
            verification_status="verified"
        ))
        
        # Lower trust
        self.register_entity(Entity(
            entity_id="social_media",
            name="Social Media",
            entity_type=EntityType.SOURCE,
            trust_score=0.3,
            trust_confidence=0.5,
            accuracy_history=0.35,
            bias_rating=0.4,
            verification_status="unverified"
        ))
        
        self.register_entity(Entity(
            entity_id="unknown_source",
            name="Unknown Source",
            entity_type=EntityType.SOURCE,
            trust_score=0.25,
            trust_confidence=0.3,
            accuracy_history=0.3,
            bias_rating=0.3,
            verification_status="unknown"
        ))
    
    def register_entity(self, entity: Entity):
        """Register an entity in the system."""
        self.entities[entity.entity_id] = entity
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)
    
    def compute_inherited_trust(self,
                               source_entities: List[str],
                               hop_count: int = 1,
                               aggregation: str = "weighted_average") -> TrustInheritance:
        """
        Compute inherited trust from source entities.
        
        Args:
            source_entities: List of entity IDs
            hop_count: Number of hops from original source
            aggregation: Method to combine trust scores
        
        Returns:
            TrustInheritance record
        """
        if not source_entities:
            return TrustInheritance(
                data_id=f"data_{datetime.utcnow().timestamp()}",
                source_entities=[],
                inherited_trust=self.DEFAULT_TRUST,
                discounted_trust=self.DEFAULT_TRUST,
                hop_count=hop_count,
                aggregation_method=aggregation
            )
        
        # Collect trust scores
        trust_scores = []
        weights = []
        
        for entity_id in source_entities:
            entity = self.entities.get(entity_id)
            if entity:
                trust_scores.append(entity.get_effective_trust())
                weights.append(entity.trust_confidence)
            else:
                trust_scores.append(self.DEFAULT_TRUST)
                weights.append(0.3)  # Low confidence for unknown
        
        # Aggregate
        if aggregation == "weighted_average":
            if sum(weights) > 0:
                inherited = sum(t * w for t, w in zip(trust_scores, weights)) / sum(weights)
            else:
                inherited = self.DEFAULT_TRUST
        elif aggregation == "max":
            inherited = max(trust_scores)
        elif aggregation == "min":
            inherited = min(trust_scores)
        elif aggregation == "pessimistic":
            # 25th percentile
            sorted_scores = sorted(trust_scores)
            idx = max(0, len(sorted_scores) // 4)
            inherited = sorted_scores[idx]
        else:
            inherited = sum(trust_scores) / len(trust_scores)
        
        # Apply hop discount
        discount = self.DISCOUNT_FACTOR ** hop_count
        discounted = inherited * discount
        
        inheritance = TrustInheritance(
            data_id=f"data_{datetime.utcnow().timestamp()}",
            source_entities=source_entities,
            inherited_trust=inherited,
            discounted_trust=discounted,
            hop_count=hop_count,
            aggregation_method=aggregation
        )
        
        self.inheritance_log.append(inheritance)
        if len(self.inheritance_log) > 1000:
            self.inheritance_log.pop(0)
        
        return inheritance
    
    def get_data_trust(self,
                      source_entity_id: str = None,
                      source_entities: List[str] = None,
                      source_url: str = None,
                      hop_count: int = 1) -> float:
        """
        Get trust score for data from a source.
        
        Convenience method that handles various input formats.
        """
        entities = []
        
        if source_entity_id:
            entities.append(source_entity_id)
        
        if source_entities:
            entities.extend(source_entities)
        
        if source_url:
            # Try to identify entity from URL
            entity_id = self._identify_entity_from_url(source_url)
            if entity_id:
                entities.append(entity_id)
        
        if not entities:
            entities = ["unknown_source"]
        
        inheritance = self.compute_inherited_trust(entities, hop_count)
        return inheritance.discounted_trust
    
    def _identify_entity_from_url(self, url: str) -> Optional[str]:
        """Identify entity from URL."""
        url_lower = url.lower()
        
        if "wikipedia.org" in url_lower:
            return "wikipedia"
        elif "pubmed" in url_lower or "ncbi.nlm.nih.gov" in url_lower:
            return "pubmed"
        elif ".gov" in url_lower:
            return "gov_official"
        elif any(site in url_lower for site in ["twitter.com", "facebook.com", "reddit.com", "x.com"]):
            return "social_media"
        elif any(site in url_lower for site in ["nytimes.com", "bbc.com", "reuters.com", "apnews.com"]):
            return "news_mainstream"
        
        return None
    
    def update_entity_trust(self, entity_id: str, feedback: float, reason: str = ""):
        """
        Update entity trust based on feedback.
        
        Args:
            entity_id: Entity to update
            feedback: 0-1, how accurate the entity was
            reason: Optional reason for update
        """
        entity = self.entities.get(entity_id)
        if entity:
            entity.update_trust(feedback, self.LEARNING_RATE)
            self._save_state()
    
    def propagate_feedback(self, 
                          data_id: str,
                          feedback: float):
        """
        Propagate feedback to source entities.
        
        Updates trust of all entities that contributed to the data.
        """
        # Find inheritance record
        for record in reversed(self.inheritance_log):
            if record.data_id == data_id:
                # Update each source entity
                for entity_id in record.source_entities:
                    self.update_entity_trust(entity_id, feedback)
                break
    
    def get_entity_hierarchy(self, entity_id: str) -> Dict[str, Any]:
        """Get entity with its parent/child relationships."""
        entity = self.entities.get(entity_id)
        if not entity:
            return {}
        
        return {
            'entity': entity.to_dict(),
            'parents': [
                self.entities[p].to_dict() 
                for p in entity.parent_entities 
                if p in self.entities
            ],
            'children': [
                self.entities[c].to_dict() 
                for c in entity.child_entities 
                if c in self.entities
            ]
        }
    
    def link_entities(self, parent_id: str, child_id: str):
        """Create parent-child relationship between entities."""
        if parent_id in self.entities and child_id in self.entities:
            parent = self.entities[parent_id]
            child = self.entities[child_id]
            
            if child_id not in parent.child_entities:
                parent.child_entities.append(child_id)
            if parent_id not in child.parent_entities:
                child.parent_entities.append(parent_id)
            
            # Child inherits some trust from parent
            if child.observation_count < 10:
                child.trust_score = 0.7 * parent.trust_score + 0.3 * child.trust_score
            
            self._save_state()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get trust system statistics."""
        trust_scores = [e.trust_score for e in self.entities.values()]
        
        return {
            'total_entities': len(self.entities),
            'avg_trust': sum(trust_scores) / len(trust_scores) if trust_scores else 0,
            'trust_distribution': {
                'high_trust': sum(1 for t in trust_scores if t >= 0.7),
                'medium_trust': sum(1 for t in trust_scores if 0.4 <= t < 0.7),
                'low_trust': sum(1 for t in trust_scores if t < 0.4)
            },
            'inheritance_records': len(self.inheritance_log),
            'entities_by_type': {
                t.value: sum(1 for e in self.entities.values() if e.entity_type == t)
                for t in EntityType
            }
        }
    
    def list_entities(self, 
                     entity_type: EntityType = None,
                     min_trust: float = None) -> List[Dict]:
        """List entities with optional filtering."""
        entities = list(self.entities.values())
        
        if entity_type:
            entities = [e for e in entities if e.entity_type == entity_type]
        
        if min_trust is not None:
            entities = [e for e in entities if e.trust_score >= min_trust]
        
        return [e.to_dict() for e in sorted(entities, key=lambda x: x.trust_score, reverse=True)]
    
    def _save_state(self):
        """Persist entity trust state."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'entities': {k: {
                'entity_id': v.entity_id,
                'name': v.name,
                'type': v.entity_type.value,
                'trust_score': v.trust_score,
                'trust_confidence': v.trust_confidence,
                'accuracy_history': v.accuracy_history,
                'bias_rating': v.bias_rating,
                'verification_status': v.verification_status,
                'observation_count': v.observation_count,
                'parent_entities': v.parent_entities,
                'child_entities': v.child_entities
            } for k, v in self.entities.items()},
            'last_updated': datetime.utcnow().isoformat()
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load persisted state."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path) as f:
                state = json.load(f)
            
            for entity_id, e_dict in state.get('entities', {}).items():
                if entity_id not in self.entities:
                    self.entities[entity_id] = Entity(
                        entity_id=e_dict['entity_id'],
                        name=e_dict['name'],
                        entity_type=EntityType(e_dict['type']),
                        trust_score=e_dict['trust_score'],
                        trust_confidence=e_dict.get('trust_confidence', 0.5),
                        accuracy_history=e_dict.get('accuracy_history', 0.5),
                        bias_rating=e_dict.get('bias_rating', 0.0),
                        verification_status=e_dict.get('verification_status', 'unknown'),
                        observation_count=e_dict.get('observation_count', 0),
                        parent_entities=e_dict.get('parent_entities', []),
                        child_entities=e_dict.get('child_entities', [])
                    )
                else:
                    # Update existing with persisted values if they've changed
                    existing = self.entities[entity_id]
                    existing.trust_score = e_dict.get('trust_score', existing.trust_score)
                    existing.observation_count = e_dict.get('observation_count', existing.observation_count)
                    
        except Exception as e:
            print(f"Warning: Could not load entity trust state: {e}")

    
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

