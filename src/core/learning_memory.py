"""
BASE Learning Memory System

Persistent memory that enables learning across sessions.
Not just pattern matching - actual accumulated knowledge.

Patent Alignment: Novel Invention - Cross-Session Learning
"""

import json
import os
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional, Any, Set
from pathlib import Path
from enum import Enum


class MemoryType(Enum):
    """Types of memories"""
    CORRECTION = "correction"       # A correction that was made
    SUCCESS = "success"             # A successful improvement
    FAILURE = "failure"             # A failed attempt
    PATTERN = "pattern"             # A learned pattern
    DOMAIN_RULE = "domain_rule"     # Domain-specific rule
    USER_PREFERENCE = "user_preference"  # User preference learned


class MemoryPriority(Enum):
    """Priority of memories"""
    CRITICAL = "critical"     # Must always apply
    HIGH = "high"             # Apply in most cases
    MEDIUM = "medium"         # Apply when relevant
    LOW = "low"               # Apply if no conflicts


@dataclass
class Memory:
    """A single memory entry"""
    id: str
    memory_type: MemoryType
    priority: MemoryPriority
    domain: str
    trigger: str  # What triggers this memory
    content: str  # The actual learning
    evidence: List[str]  # Supporting evidence
    confidence: float
    times_applied: int = 0
    times_successful: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: Optional[str] = None
    
    def effectiveness(self) -> float:
        """Calculate effectiveness rate"""
        if self.times_applied == 0:
            return 0.5  # Neutral for unused
        return self.times_successful / self.times_applied


@dataclass
class MemorySearchResult:
    """Result of memory search"""
    memories: List[Memory]
    relevance_scores: Dict[str, float]
    total_found: int


@dataclass
class LearningOutcome:
    """Outcome of applying memories"""
    memories_applied: List[str]  # Memory IDs
    outcome: str  # 'success', 'partial', 'failure'
    details: str
    improvement_score: float


class LearningMemory:
    """
    Persistent learning memory system.
    
    Key capabilities:
    1. Store corrections and improvements
    2. Recall relevant memories for new queries
    3. Track effectiveness of memories
    4. Adapt based on outcomes
    """
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def __init__(self, storage_path: str = "learning_data/memory"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.memories: Dict[str, Memory] = {}
        self.index: Dict[str, List[str]] = {}  # domain -> memory_ids
        
        self._load_memories()
    
    def _load_memories(self):
        """Load memories from disk"""
        memory_file = self.storage_path / "memories.json"
        if memory_file.exists():
            try:
                with open(memory_file, 'r') as f:
                    data = json.load(f)
                    for m_data in data.get('memories', []):
                        memory = Memory(
                            id=m_data['id'],
                            memory_type=MemoryType(m_data['memory_type']),
                            priority=MemoryPriority(m_data['priority']),
                            domain=m_data['domain'],
                            trigger=m_data['trigger'],
                            content=m_data['content'],
                            evidence=m_data.get('evidence', []),
                            confidence=m_data.get('confidence', 0.5),
                            times_applied=m_data.get('times_applied', 0),
                            times_successful=m_data.get('times_successful', 0),
                            created_at=m_data.get('created_at', datetime.now().isoformat()),
                            last_used=m_data.get('last_used')
                        )
                        self.memories[memory.id] = memory
                        
                        # Index by domain
                        if memory.domain not in self.index:
                            self.index[memory.domain] = []
                        self.index[memory.domain].append(memory.id)
            except Exception as e:
                print(f"Warning: Could not load memories: {e}")
    
    def _save_memories(self):
        """Save memories to disk"""
        memory_file = self.storage_path / "memories.json"
        try:
            data = {
                'memories': [
                    {
                        'id': m.id,
                        'memory_type': m.memory_type.value,
                        'priority': m.priority.value,
                        'domain': m.domain,
                        'trigger': m.trigger,
                        'content': m.content,
                        'evidence': m.evidence,
                        'confidence': m.confidence,
                        'times_applied': m.times_applied,
                        'times_successful': m.times_successful,
                        'created_at': m.created_at,
                        'last_used': m.last_used
                    }
                    for m in self.memories.values()
                ],
                'saved_at': datetime.now().isoformat()
            }
            with open(memory_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save memories: {e}")
    
    def store(self, 
              memory_type: MemoryType,
              domain: str,
              trigger: str,
              content: str,
              evidence: List[str] = None,
              priority: MemoryPriority = MemoryPriority.MEDIUM,
              confidence: float = 0.7) -> Memory:
        """
        Store a new memory.
        
        Args:
            memory_type: Type of memory
            domain: Domain (medical, financial, etc.)
            trigger: What triggers this memory
            content: The actual learning
            evidence: Supporting evidence
            priority: Priority level
            confidence: Initial confidence
            
        Returns:
            The created Memory
        """
        # Generate ID
        memory_id = hashlib.sha256(
            f"{domain}:{trigger}:{content}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        memory = Memory(
            id=memory_id,
            memory_type=memory_type,
            priority=priority,
            domain=domain,
            trigger=trigger,
            content=content,
            evidence=evidence or [],
            confidence=confidence
        )
        
        self.memories[memory_id] = memory
        
        # Update index
        if domain not in self.index:
            self.index[domain] = []
        self.index[domain].append(memory_id)
        
        self._save_memories()
        
        return memory
    
    def recall(self, 
               query: str,
               domain: str = 'general',
               min_confidence: float = 0.3) -> MemorySearchResult:
        """
        Recall relevant memories for a query.
        
        Args:
            query: The query/context
            domain: Domain to search in
            min_confidence: Minimum confidence threshold
            
        Returns:
            MemorySearchResult with relevant memories
        """
        results = []
        scores = {}
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Search in specific domain and general
        domains_to_search = [domain, 'general']
        if domain != 'general':
            domains_to_search.append(domain)
        
        for search_domain in domains_to_search:
            if search_domain not in self.index:
                continue
            
            for memory_id in self.index[search_domain]:
                memory = self.memories.get(memory_id)
                if not memory or memory.confidence < min_confidence:
                    continue
                
                # Calculate relevance score
                relevance = self._calculate_relevance(memory, query_lower, query_words)
                
                if relevance > 0.2:  # Minimum relevance threshold
                    results.append(memory)
                    scores[memory_id] = relevance
        
        # Sort by relevance * confidence * priority
        def sort_key(m: Memory) -> float:
            priority_mult = {
                MemoryPriority.CRITICAL: 2.0,
                MemoryPriority.HIGH: 1.5,
                MemoryPriority.MEDIUM: 1.0,
                MemoryPriority.LOW: 0.7
            }
            return scores[m.id] * m.confidence * priority_mult.get(m.priority, 1.0)
        
        results.sort(key=sort_key, reverse=True)
        
        return MemorySearchResult(
            memories=results[:10],  # Top 10
            relevance_scores=scores,
            total_found=len(results)
        )
    
    def _calculate_relevance(self, memory: Memory, 
                              query_lower: str, query_words: Set[str]) -> float:
        """Calculate relevance of memory to query"""
        trigger_lower = memory.trigger.lower()
        trigger_words = set(trigger_lower.split())
        
        # Word overlap
        overlap = len(query_words & trigger_words)
        max_words = max(len(query_words), len(trigger_words))
        word_score = overlap / max_words if max_words > 0 else 0
        
        # Substring matching
        substring_score = 0
        if trigger_lower in query_lower:
            substring_score = 0.5
        elif any(w in query_lower for w in trigger_words if len(w) > 4):
            substring_score = 0.3
        
        # Effectiveness bonus
        effectiveness_bonus = memory.effectiveness() * 0.2
        
        return word_score * 0.5 + substring_score * 0.3 + effectiveness_bonus
    
    def record_outcome(self, memory_ids: List[str], 
                       success: bool, details: str = "") -> None:
        """
        Record outcome of applying memories.
        
        Args:
            memory_ids: IDs of memories that were applied
            success: Whether the application was successful
            details: Additional details
        """
        for memory_id in memory_ids:
            if memory_id in self.memories:
                memory = self.memories[memory_id]
                memory.times_applied += 1
                if success:
                    memory.times_successful += 1
                memory.last_used = datetime.now().isoformat()
                
                # Adjust confidence based on outcome
                if success:
                    memory.confidence = min(1.0, memory.confidence + 0.05)
                else:
                    memory.confidence = max(0.1, memory.confidence - 0.1)
        
        self._save_memories()
    
    def prune(self, min_effectiveness: float = 0.3, 
              min_applications: int = 3) -> int:
        """
        Prune ineffective memories.
        
        Args:
            min_effectiveness: Minimum effectiveness to keep
            min_applications: Minimum applications before pruning
            
        Returns:
            Number of memories pruned
        """
        to_prune = []
        
        for memory_id, memory in self.memories.items():
            if (memory.times_applied >= min_applications and 
                memory.effectiveness() < min_effectiveness):
                to_prune.append(memory_id)
        
        for memory_id in to_prune:
            memory = self.memories.pop(memory_id)
            if memory.domain in self.index:
                self.index[memory.domain] = [
                    m for m in self.index[memory.domain] if m != memory_id
                ]
        
        if to_prune:
            self._save_memories()
        
        return len(to_prune)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        by_domain = {}
        by_type = {}
        total_effectiveness = 0
        effective_count = 0
        
        for memory in self.memories.values():
            by_domain[memory.domain] = by_domain.get(memory.domain, 0) + 1
            by_type[memory.memory_type.value] = by_type.get(memory.memory_type.value, 0) + 1
            
            if memory.times_applied > 0:
                total_effectiveness += memory.effectiveness()
                effective_count += 1
        
        return {
            'total_memories': len(self.memories),
            'by_domain': by_domain,
            'by_type': by_type,
            'average_effectiveness': total_effectiveness / effective_count if effective_count > 0 else 0,
            'total_applications': sum(m.times_applied for m in self.memories.values()),
            'total_successes': sum(m.times_successful for m in self.memories.values())
        }
    
    def export_learnings(self) -> List[Dict[str, Any]]:
        """Export learned patterns for review"""
        learnings = []
        
        for memory in self.memories.values():
            if memory.times_applied > 0 and memory.effectiveness() > 0.5:
                learnings.append({
                    'domain': memory.domain,
                    'trigger': memory.trigger,
                    'learning': memory.content,
                    'effectiveness': memory.effectiveness(),
                    'applications': memory.times_applied
                })
        
        return sorted(learnings, key=lambda x: x['effectiveness'], reverse=True)

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


# Pre-seed with foundational learnings
def seed_foundational_memories(memory: LearningMemory):
    """Seed memory with foundational learnings from BASE development"""
    
    foundational = [
        # Medical domain
        {
            'domain': 'medical',
            'trigger': 'medical advice recommendation',
            'content': 'Always add disclaimer to consult healthcare provider',
            'evidence': ['Medical advice requires professional verification'],
            'priority': MemoryPriority.CRITICAL
        },
        {
            'domain': 'medical',
            'trigger': 'treatment recommendation certainty',
            'content': 'Replace absolute certainty with hedged language',
            'evidence': ['Medical outcomes vary by individual'],
            'priority': MemoryPriority.HIGH
        },
        
        # Financial domain
        {
            'domain': 'financial',
            'trigger': 'investment recommendation',
            'content': 'Add risk warnings and professional advice disclaimer',
            'evidence': ['Financial advice regulated, outcomes uncertain'],
            'priority': MemoryPriority.CRITICAL
        },
        {
            'domain': 'financial',
            'trigger': 'guaranteed returns claim',
            'content': 'Flag and remove guarantee language - no investment is guaranteed',
            'evidence': ['No legitimate investment guarantees returns'],
            'priority': MemoryPriority.CRITICAL
        },
        
        # Reasoning
        {
            'domain': 'general',
            'trigger': 'bandwagon argument',
            'content': 'Popularity does not equal validity - flag bandwagon fallacy',
            'evidence': ['Everyone doing X is not evidence X is correct'],
            'priority': MemoryPriority.HIGH
        },
        {
            'domain': 'general',
            'trigger': 'appeal to authority without citation',
            'content': 'Require specific citations for authority claims',
            'evidence': ['Vague authority claims are manipulation'],
            'priority': MemoryPriority.HIGH
        },
        
        # BASE self-improvement learnings
        {
            'domain': 'development',
            'trigger': 'declaring completion prematurely',
            'content': 'Verify completion against objective criteria before declaring done',
            'evidence': ['Mission drift occurred from premature completion claims'],
            'priority': MemoryPriority.CRITICAL
        },
        {
            'domain': 'development',
            'trigger': 'blocking instead of improving',
            'content': 'Goal is to IMPROVE outputs, not block/reject them',
            'evidence': ['Mission drift: blocking is not improvement'],
            'priority': MemoryPriority.CRITICAL
        },
        {
            'domain': 'development',
            'trigger': 'simulated testing',
            'content': 'Use real data and actual testing, not simulations',
            'evidence': ['Simulated tests missed real issues'],
            'priority': MemoryPriority.HIGH
        }
    ]
    
    for learning in foundational:
        # Check if similar memory exists
        existing = memory.recall(learning['trigger'], learning['domain'])
        if not any(m.content == learning['content'] for m in existing.memories):
            memory.store(
                memory_type=MemoryType.PATTERN,
                domain=learning['domain'],
                trigger=learning['trigger'],
                content=learning['content'],
                evidence=learning.get('evidence', []),
                priority=learning.get('priority', MemoryPriority.MEDIUM),
                confidence=0.9  # High confidence for foundational
            )


def test_learning_memory():
    """Test the learning memory system"""
    import tempfile
    
    # Use temp directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = LearningMemory(storage_path=tmpdir)
        
        print("=" * 70)
        print("LEARNING MEMORY SYSTEM TEST")
        print("=" * 70)
        
        # Seed foundational memories
        seed_foundational_memories(memory)
        print(f"\nSeeded {len(memory.memories)} foundational memories")
        
        # Test recall
        print("\n--- Testing Recall ---")
        
        queries = [
            ("I recommend this investment for guaranteed returns", "financial"),
            ("Everyone is using this treatment so it must work", "medical"),
            ("The system is complete and all tests pass", "development")
        ]
        
        for query, domain in queries:
            print(f"\nQuery: {query[:50]}...")
            print(f"Domain: {domain}")
            
            result = memory.recall(query, domain)
            print(f"Found {result.total_found} relevant memories")
            
            for mem in result.memories[:3]:
                print(f"  - [{mem.priority.value}] {mem.content[:60]}...")
                print(f"    Relevance: {result.relevance_scores[mem.id]:.2f}")
        
        # Test storing new memory
        print("\n--- Testing Store ---")
        new_mem = memory.store(
            memory_type=MemoryType.CORRECTION,
            domain='test',
            trigger='test scenario',
            content='This is a test correction',
            evidence=['Test evidence'],
            confidence=0.8
        )
        print(f"Stored new memory: {new_mem.id}")
        
        # Test outcome recording
        print("\n--- Testing Outcome Recording ---")
        memory.record_outcome([new_mem.id], success=True)
        print(f"After success: confidence={new_mem.confidence:.2f}, effectiveness={new_mem.effectiveness():.2f}")
        
        memory.record_outcome([new_mem.id], success=False)
        print(f"After failure: confidence={new_mem.confidence:.2f}, effectiveness={new_mem.effectiveness():.2f}")
        
        # Print stats
        print("\n--- Memory Stats ---")
        stats = memory.get_stats()
        print(f"Total memories: {stats['total_memories']}")
        print(f"By domain: {stats['by_domain']}")
        print(f"By type: {stats['by_type']}")
        
        # Export learnings
        print("\n--- Exported Learnings ---")
        learnings = memory.export_learnings()
        for learning in learnings[:3]:
            print(f"  [{learning['domain']}] {learning['learning'][:50]}...")


if __name__ == "__main__":
    test_learning_memory()

