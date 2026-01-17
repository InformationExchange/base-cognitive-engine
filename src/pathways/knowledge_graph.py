"""
BAIS Knowledge Graph Pathway
Migrated from Onyx Governance - Enhancement 4

Entity/relationship extraction and verification against source documents.

Patent Claims: PPA1-Inv6, Enhancement-4
Proprietary IP - 100% owned by Invitas Inc.
"""

import re
import time
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field


@dataclass
class KnowledgeEntity:
    """Knowledge graph entity."""
    entity_id: str
    entity_type: str  # PERSON, ORG, LOCATION, CONCEPT, etc.
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class KnowledgeRelationship:
    """Knowledge graph relationship."""
    source_entity: str
    target_entity: str
    relationship_type: str  # IS_A, HAS, PART_OF, CAUSES, etc.
    confidence: float = 0.7


@dataclass
class KnowledgeAlignment:
    """Knowledge alignment result."""
    aligned: bool
    alignment_score: float
    entities_found: int
    relationships_found: int
    contradictions: List[str]
    missing_entities: List[str]
    entity_coverage: float
    reason: Optional[str]
    timestamp: float
    processing_time_ms: float = 0.0


class KnowledgeGraphPathway:
    """
    Knowledge Graph Governance Pathway for BAIS.
    
    Implements:
    - Entity extraction from responses (NER-based)
    - Relationship extraction (pattern-based)
    - Knowledge graph alignment verification
    - Contradiction detection
    
    Integrates with BAIS Module Registry for self-registration.
    """
    
    # Entity type patterns
    ENTITY_PATTERNS = {
        'PERSON': [
            r'\b(Dr\.|Mr\.|Ms\.|Mrs\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:,?\s+(?:Jr\.|Sr\.|III|IV))?\b',
        ],
        'ORG': [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc\.|Corp\.|LLC|Ltd\.|Company|Group)\b',
            r'\b(?:The\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:University|Institute|Foundation)\b',
        ],
        'LOCATION': [
            r'\b(?:in|at|from)\s+[A-Z][a-z]+(?:,\s*[A-Z][a-z]+)?\b',
        ],
        'DATE': [
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{4}\b',
        ],
    }
    
    # Relationship patterns
    RELATIONSHIP_PATTERNS = [
        (r'(\w+)\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*(\w+)', 'IS_A'),
        (r'(\w+)\s+(?:has|have|had)\s+(\w+)', 'HAS'),
        (r'(\w+)\s+(?:causes?|leads?\s+to|results?\s+in)\s+(\w+)', 'CAUSES'),
        (r'(\w+)\s+(?:belongs?\s+to|is\s+part\s+of)\s+(\w+)', 'PART_OF'),
        (r'(\w+)\s+(?:works?\s+(?:at|for)|employed\s+by)\s+(\w+)', 'WORKS_FOR'),
        (r'(\w+)\s+(?:located\s+in|based\s+in)\s+(\w+)', 'LOCATED_IN'),
    ]
    
    def __init__(self, 
                 alignment_threshold: float = 0.60,
                 entity_coverage_threshold: float = 0.50):
        """
        Initialize Knowledge Graph Pathway.
        
        Args:
            alignment_threshold: Minimum alignment score to consider aligned
            entity_coverage_threshold: Minimum entity coverage required
        """
        self.alignment_threshold = alignment_threshold
        self.entity_coverage_threshold = entity_coverage_threshold
        
        # Try to use spaCy for better NER (if available)
        self.nlp = None
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except:
            pass  # Fallback to pattern-based extraction
    
    def align(self,
              response: str,
              query: str,
              documents: List[Dict[str, Any]]) -> KnowledgeAlignment:
        """
        Align response entities/relationships with document knowledge.
        
        Args:
            response: AI-generated response
            query: Original query
            documents: Source documents
        
        Returns:
            KnowledgeAlignment with verification results
        """
        start_time = time.time()
        
        # Extract entities from response
        response_entities = self._extract_entities(response)
        
        # Extract entities from documents
        doc_text = " ".join(d.get('content', '') for d in documents[:5])
        document_entities = self._extract_entities(doc_text)
        
        # Extract relationships
        response_relationships = self._extract_relationships(response, response_entities)
        document_relationships = self._extract_relationships(doc_text, document_entities)
        
        # Compute alignment score
        alignment_score, entity_coverage = self._compute_alignment(
            response_entities, document_entities,
            response_relationships, document_relationships
        )
        
        # Detect contradictions
        contradictions = self._detect_contradictions(
            response_entities, document_entities,
            response_relationships, document_relationships
        )
        
        # Find missing entities
        missing_entities = self._find_missing_entities(
            response_entities, document_entities
        )
        
        # Determine alignment
        aligned = (
            alignment_score >= self.alignment_threshold and
            len(contradictions) == 0
        )
        
        reason = None
        if not aligned:
            if alignment_score < self.alignment_threshold:
                reason = f"Alignment {alignment_score:.2f} < threshold {self.alignment_threshold}"
            elif contradictions:
                reason = f"Found {len(contradictions)} contradictions"
        
        processing_time = (time.time() - start_time) * 1000
        
        return KnowledgeAlignment(
            aligned=aligned,
            alignment_score=alignment_score,
            entities_found=len(response_entities),
            relationships_found=len(response_relationships),
            contradictions=contradictions,
            missing_entities=missing_entities,
            entity_coverage=entity_coverage,
            reason=reason,
            timestamp=time.time(),
            processing_time_ms=processing_time
        )
    
    def _extract_entities(self, text: str) -> List[KnowledgeEntity]:
        """Extract named entities from text."""
        entities = []
        
        # Try spaCy first (more accurate)
        if self.nlp:
            try:
                doc = self.nlp(text[:10000])  # Limit text length
                for ent in doc.ents:
                    entity = KnowledgeEntity(
                        entity_id=f"ent_{hash(ent.text) % 100000}",
                        entity_type=ent.label_,
                        name=ent.text,
                        confidence=0.9
                    )
                    entities.append(entity)
                return entities[:30]  # Limit
            except:
                pass  # Fallback to patterns
        
        # Pattern-based extraction
        seen_names = set()
        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    name = match.group().strip()
                    if name and name.lower() not in seen_names and len(name) > 2:
                        seen_names.add(name.lower())
                        entity = KnowledgeEntity(
                            entity_id=f"ent_{hash(name) % 100000}",
                            entity_type=entity_type,
                            name=name,
                            confidence=0.7
                        )
                        entities.append(entity)
        
        # Also extract capitalized phrases (generic entities)
        caps_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        for match in re.finditer(caps_pattern, text):
            name = match.group()
            if name.lower() not in seen_names and len(name) > 3:
                seen_names.add(name.lower())
                entity = KnowledgeEntity(
                    entity_id=f"ent_{hash(name) % 100000}",
                    entity_type='UNKNOWN',
                    name=name,
                    confidence=0.5
                )
                entities.append(entity)
        
        return entities[:30]  # Limit to 30 entities
    
    def _extract_relationships(self, 
                               text: str, 
                               entities: List[KnowledgeEntity]) -> List[KnowledgeRelationship]:
        """Extract relationships between entities."""
        relationships = []
        entity_names = {e.name.lower(): e for e in entities}
        
        for pattern, rel_type in self.RELATIONSHIP_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()
                if len(groups) >= 2:
                    source_name = groups[0].lower()
                    target_name = groups[-1].lower()
                    
                    # Find matching entities
                    source_entity = self._find_entity(source_name, entity_names)
                    target_entity = self._find_entity(target_name, entity_names)
                    
                    if source_entity and target_entity and source_entity != target_entity:
                        rel = KnowledgeRelationship(
                            source_entity=source_entity.entity_id,
                            target_entity=target_entity.entity_id,
                            relationship_type=rel_type,
                            confidence=0.7
                        )
                        relationships.append(rel)
        
        return relationships[:15]  # Limit to 15 relationships
    
    def _find_entity(self, 
                     name: str, 
                     entity_names: Dict[str, KnowledgeEntity]) -> Optional[KnowledgeEntity]:
        """Find entity by name (fuzzy match)."""
        # Exact match
        if name in entity_names:
            return entity_names[name]
        
        # Partial match
        for ent_name, entity in entity_names.items():
            if name in ent_name or ent_name in name:
                return entity
        
        return None
    
    def _compute_alignment(self,
                           response_entities: List[KnowledgeEntity],
                           document_entities: List[KnowledgeEntity],
                           response_rels: List[KnowledgeRelationship],
                           document_rels: List[KnowledgeRelationship]) -> tuple:
        """Compute alignment score and entity coverage."""
        if not response_entities:
            return 0.0, 0.0
        
        # Entity alignment
        response_names = {e.name.lower() for e in response_entities}
        document_names = {e.name.lower() for e in document_entities}
        
        entity_overlap = len(response_names & document_names)
        entity_coverage = entity_overlap / len(response_names) if response_names else 0.0
        
        # Relationship alignment
        rel_alignment = 0.5  # Default
        if response_rels and document_rels:
            response_rel_pairs = {(r.source_entity, r.target_entity) for r in response_rels}
            document_rel_pairs = {(r.source_entity, r.target_entity) for r in document_rels}
            
            rel_overlap = len(response_rel_pairs & document_rel_pairs)
            rel_alignment = rel_overlap / len(response_rel_pairs) if response_rel_pairs else 0.0
        
        # Combined score
        alignment_score = entity_coverage * 0.7 + rel_alignment * 0.3
        
        return alignment_score, entity_coverage
    
    def _detect_contradictions(self,
                               response_entities: List[KnowledgeEntity],
                               document_entities: List[KnowledgeEntity],
                               response_rels: List[KnowledgeRelationship],
                               document_rels: List[KnowledgeRelationship]) -> List[str]:
        """Detect contradictions between response and documents."""
        contradictions = []
        
        # Build relationship dicts
        response_rel_dict = {
            (r.source_entity, r.target_entity): r.relationship_type
            for r in response_rels
        }
        document_rel_dict = {
            (r.source_entity, r.target_entity): r.relationship_type
            for r in document_rels
        }
        
        # Check for conflicting relationship types
        for key, rel_type in response_rel_dict.items():
            if key in document_rel_dict:
                doc_rel_type = document_rel_dict[key]
                if rel_type != doc_rel_type:
                    contradictions.append(
                        f"Relationship conflict: {key[0]}->{key[1]} "
                        f"({rel_type} vs {doc_rel_type})"
                    )
        
        return contradictions[:5]  # Limit
    
    def _find_missing_entities(self,
                               response_entities: List[KnowledgeEntity],
                               document_entities: List[KnowledgeEntity]) -> List[str]:
        """Find important entities in documents missing from response."""
        response_names = {e.name.lower() for e in response_entities}
        
        # Find high-confidence document entities not in response
        missing = []
        for entity in document_entities:
            if entity.name.lower() not in response_names:
                if entity.confidence > 0.7:  # Only important entities
                    missing.append(entity.name)
        
        return missing[:10]  # Limit to 10

    # Learning Interface
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        if not hasattr(self, '_outcomes'): self._outcomes = []
        self._outcomes.append(outcome)
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        pass
    
    def get_statistics(self) -> Dict:
        return {'outcomes': len(getattr(self, '_outcomes', []))}
    
    def serialize_state(self) -> Dict:
        return {'outcomes': getattr(self, '_outcomes', [])[-100:]}
    
    def deserialize_state(self, state: Dict) -> None:
        self._outcomes = state.get('outcomes', [])


# Self-test
if __name__ == "__main__":
    pathway = KnowledgeGraphPathway()
    
    response = "Tesla Inc., led by CEO Elon Musk, is headquartered in Austin, Texas. The company was founded in 2003."
    query = "Tell me about Tesla"
    documents = [
        {"content": "Tesla Inc. is an American electric vehicle company. Elon Musk serves as CEO. The headquarters moved to Austin, Texas in 2021."},
        {"content": "Tesla was incorporated in 2003 by Martin Eberhard and Marc Tarpenning."}
    ]
    
    result = pathway.align(response, query, documents)
    
    print(f"Aligned: {result.aligned}")
    print(f"Alignment Score: {result.alignment_score:.3f}")
    print(f"Entities Found: {result.entities_found}")
    print(f"Relationships Found: {result.relationships_found}")
    print(f"Contradictions: {result.contradictions}")
    print(f"Processing Time: {result.processing_time_ms:.1f}ms")






