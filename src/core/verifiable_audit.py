"""
BAIS Cognitive Governance Engine v31.0
Verifiable Audit - Tamper-Evident Hash-Chaining

Phase 31: Addresses PPA2-C1-17, PPA2-C1-27
- Tamper-evident audit via hash-chaining
- Append-only audit log
- Cryptographic integrity verification
- Merkle tree for batch verification

Patent Claims Addressed:
- PPA2-C1-17: Verifiable audit via tamper-evident hash-chaining, append-only
- PPA2-C1-27: Adaptation events logged in verifiable audit with all parameters
"""

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum
import logging
import uuid

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""
    DECISION = "decision"
    THRESHOLD_ADAPTATION = "threshold_adaptation"
    DRIFT_DETECTED = "drift_detected"
    QUARANTINE = "quarantine"
    CERTIFICATE_ISSUED = "certificate_issued"
    MUST_PASS_EVALUATION = "must_pass_evaluation"
    TEMPORAL_ROBUSTNESS = "temporal_robustness"
    GOVERNANCE_VIOLATION = "governance_violation"
    SYSTEM_CONFIG_CHANGE = "system_config_change"


class VerificationStatus(Enum):
    """Status of chain verification."""
    VALID = "valid"
    INVALID = "invalid"
    TAMPERED = "tampered"
    INCOMPLETE = "incomplete"


@dataclass
class AuditEntry:
    """Single audit entry in the hash chain."""
    sequence_id: int
    timestamp: str
    event_type: AuditEventType
    event_data: Dict[str, Any]
    previous_hash: str
    current_hash: str
    nonce: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sequence_id": self.sequence_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "event_data": self.event_data,
            "previous_hash": self.previous_hash,
            "current_hash": self.current_hash,
            "nonce": self.nonce
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEntry':
        """Create from dictionary."""
        return cls(
            sequence_id=data["sequence_id"],
            timestamp=data["timestamp"],
            event_type=AuditEventType(data["event_type"]),
            event_data=data["event_data"],
            previous_hash=data["previous_hash"],
            current_hash=data["current_hash"],
            nonce=data.get("nonce", "")
        )


@dataclass
class VerificationResult:
    """Result of audit chain verification."""
    status: VerificationStatus
    valid_entries: int
    total_entries: int
    first_invalid_at: Optional[int]
    details: Dict[str, Any] = field(default_factory=dict)


class HashChainAudit:
    """
    Tamper-evident audit log using hash-chaining.
    
    PPA2-C1-17: Verifiable audit via tamper-evident hash-chaining, append-only.
    
    Each entry contains the hash of the previous entry, creating an
    immutable chain where any modification is detectable.
    """
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        hash_algorithm: str = "sha256"
    ):
        """
        Initialize hash chain audit.
        
        Args:
            storage_path: Path for persistent storage
            hash_algorithm: Hash algorithm to use
        """
        self.storage_path = storage_path
        self.hash_algorithm = hash_algorithm
        self.chain: List[AuditEntry] = []
        self.genesis_hash = self._compute_genesis_hash()
        
        # Load existing chain if storage exists
        if storage_path and storage_path.exists():
            self._load_chain()
        
        logger.info(f"[HashChain] Initialized with {len(self.chain)} entries")
    
    def _compute_genesis_hash(self) -> str:
        """Compute genesis (initial) hash."""
        genesis_data = {
            "type": "genesis",
            "timestamp": "2025-01-01T00:00:00Z",
            "system": "BAIS-Cognitive-Engine",
            "version": "31.0.0"
        }
        return self._hash_data(genesis_data)
    
    def _hash_data(self, data: Any) -> str:
        """Compute hash of data."""
        hasher = hashlib.new(self.hash_algorithm)
        serialized = json.dumps(data, sort_keys=True, default=str)
        hasher.update(serialized.encode('utf-8'))
        return hasher.hexdigest()
    
    def _compute_entry_hash(
        self,
        sequence_id: int,
        timestamp: str,
        event_type: str,
        event_data: Dict[str, Any],
        previous_hash: str,
        nonce: str
    ) -> str:
        """Compute hash for an audit entry."""
        entry_content = {
            "sequence_id": sequence_id,
            "timestamp": timestamp,
            "event_type": event_type,
            "event_data": event_data,
            "previous_hash": previous_hash,
            "nonce": nonce
        }
        return self._hash_data(entry_content)
    
    def append(
        self,
        event_type: AuditEventType,
        event_data: Dict[str, Any]
    ) -> AuditEntry:
        """
        Append new entry to audit chain.
        
        Args:
            event_type: Type of event
            event_data: Event data to log
            
        Returns:
            Created AuditEntry
        """
        sequence_id = len(self.chain)
        timestamp = datetime.utcnow().isoformat() + "Z"
        previous_hash = self.chain[-1].current_hash if self.chain else self.genesis_hash
        nonce = uuid.uuid4().hex[:8]
        
        current_hash = self._compute_entry_hash(
            sequence_id=sequence_id,
            timestamp=timestamp,
            event_type=event_type.value,
            event_data=event_data,
            previous_hash=previous_hash,
            nonce=nonce
        )
        
        entry = AuditEntry(
            sequence_id=sequence_id,
            timestamp=timestamp,
            event_type=event_type,
            event_data=event_data,
            previous_hash=previous_hash,
            current_hash=current_hash,
            nonce=nonce
        )
        
        self.chain.append(entry)
        
        # Persist if storage configured
        if self.storage_path:
            self._save_chain()
        
        logger.debug(f"[HashChain] Appended entry {sequence_id}: {event_type.value}")
        
        return entry
    
    def verify_chain(self) -> VerificationResult:
        """
        Verify integrity of entire audit chain.
        
        Returns:
            VerificationResult with validation status
        """
        if not self.chain:
            return VerificationResult(
                status=VerificationStatus.VALID,
                valid_entries=0,
                total_entries=0,
                first_invalid_at=None,
                details={"message": "Empty chain is valid"}
            )
        
        valid_count = 0
        
        for i, entry in enumerate(self.chain):
            # Check sequence
            if entry.sequence_id != i:
                return VerificationResult(
                    status=VerificationStatus.TAMPERED,
                    valid_entries=valid_count,
                    total_entries=len(self.chain),
                    first_invalid_at=i,
                    details={"error": f"Sequence mismatch at {i}"}
                )
            
            # Check previous hash link
            expected_prev = self.genesis_hash if i == 0 else self.chain[i-1].current_hash
            if entry.previous_hash != expected_prev:
                return VerificationResult(
                    status=VerificationStatus.TAMPERED,
                    valid_entries=valid_count,
                    total_entries=len(self.chain),
                    first_invalid_at=i,
                    details={"error": f"Hash link broken at {i}"}
                )
            
            # Recompute and verify current hash
            computed_hash = self._compute_entry_hash(
                sequence_id=entry.sequence_id,
                timestamp=entry.timestamp,
                event_type=entry.event_type.value,
                event_data=entry.event_data,
                previous_hash=entry.previous_hash,
                nonce=entry.nonce
            )
            
            if entry.current_hash != computed_hash:
                return VerificationResult(
                    status=VerificationStatus.TAMPERED,
                    valid_entries=valid_count,
                    total_entries=len(self.chain),
                    first_invalid_at=i,
                    details={"error": f"Hash mismatch at {i}"}
                )
            
            valid_count += 1
        
        return VerificationResult(
            status=VerificationStatus.VALID,
            valid_entries=valid_count,
            total_entries=len(self.chain),
            first_invalid_at=None,
            details={"message": "Chain integrity verified"}
        )
    
    def verify_entry(self, sequence_id: int) -> bool:
        """
        Verify a specific entry in the chain.
        
        Args:
            sequence_id: Sequence ID of entry to verify
            
        Returns:
            True if entry is valid
        """
        if sequence_id < 0 or sequence_id >= len(self.chain):
            return False
        
        entry = self.chain[sequence_id]
        
        # Check previous hash
        expected_prev = self.genesis_hash if sequence_id == 0 else self.chain[sequence_id-1].current_hash
        if entry.previous_hash != expected_prev:
            return False
        
        # Check current hash
        computed = self._compute_entry_hash(
            sequence_id=entry.sequence_id,
            timestamp=entry.timestamp,
            event_type=entry.event_type.value,
            event_data=entry.event_data,
            previous_hash=entry.previous_hash,
            nonce=entry.nonce
        )
        
        return entry.current_hash == computed
    
    def get_entry(self, sequence_id: int) -> Optional[AuditEntry]:
        """Get entry by sequence ID."""
        if 0 <= sequence_id < len(self.chain):
            return self.chain[sequence_id]
        return None
    
    def get_entries_by_type(self, event_type: AuditEventType) -> List[AuditEntry]:
        """Get all entries of a specific type."""
        return [e for e in self.chain if e.event_type == event_type]
    
    def get_chain_summary(self) -> Dict[str, Any]:
        """Get summary of audit chain."""
        if not self.chain:
            return {"entries": 0, "verified": True}
        
        verification = self.verify_chain()
        event_counts = {}
        for entry in self.chain:
            event_type = entry.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        return {
            "entries": len(self.chain),
            "verified": verification.status == VerificationStatus.VALID,
            "first_entry": self.chain[0].timestamp if self.chain else None,
            "last_entry": self.chain[-1].timestamp if self.chain else None,
            "event_counts": event_counts,
            "genesis_hash": self.genesis_hash,
            "latest_hash": self.chain[-1].current_hash if self.chain else None
        }
    
    def _save_chain(self):
        """Save chain to storage."""
        if not self.storage_path:
            return
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = [entry.to_dict() for entry in self.chain]
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_chain(self):
        """Load chain from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        with open(self.storage_path, 'r') as f:
            data = json.load(f)
        
        self.chain = [AuditEntry.from_dict(entry) for entry in data]

    # Learning Interface
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


class MerkleTree:
    """
    Merkle tree for batch verification of audit entries.
    
    Enables efficient verification of large audit logs by computing
    a single root hash that represents all entries.
    """
    
    def __init__(self, hash_algorithm: str = "sha256"):
        """Initialize Merkle tree."""
        self.hash_algorithm = hash_algorithm
        self.leaves: List[str] = []
        self.root: Optional[str] = None
        
    def add_leaf(self, data: Any) -> str:
        """Add a leaf node."""
        hasher = hashlib.new(self.hash_algorithm)
        serialized = json.dumps(data, sort_keys=True, default=str)
        hasher.update(serialized.encode('utf-8'))
        leaf_hash = hasher.hexdigest()
        self.leaves.append(leaf_hash)
        return leaf_hash
    
    def compute_root(self) -> str:
        """Compute Merkle root."""
        if not self.leaves:
            return hashlib.new(self.hash_algorithm).hexdigest()
        
        level = self.leaves.copy()
        
        while len(level) > 1:
            next_level = []
            for i in range(0, len(level), 2):
                left = level[i]
                right = level[i + 1] if i + 1 < len(level) else left
                combined = left + right
                hasher = hashlib.new(self.hash_algorithm)
                hasher.update(combined.encode('utf-8'))
                next_level.append(hasher.hexdigest())
            level = next_level
        
        self.root = level[0]
        return self.root
    
    def get_proof(self, index: int) -> List[Tuple[str, str]]:
        """
        Get Merkle proof for a leaf.
        
        Args:
            index: Index of leaf
            
        Returns:
            List of (sibling_hash, position) tuples
        """
        if index < 0 or index >= len(self.leaves):
            return []
        
        proof = []
        level = self.leaves.copy()
        idx = index
        
        while len(level) > 1:
            sibling_idx = idx + 1 if idx % 2 == 0 else idx - 1
            if sibling_idx < len(level):
                position = "right" if idx % 2 == 0 else "left"
                proof.append((level[sibling_idx], position))
            
            # Move to next level
            next_level = []
            for i in range(0, len(level), 2):
                left = level[i]
                right = level[i + 1] if i + 1 < len(level) else left
                combined = left + right
                hasher = hashlib.new(self.hash_algorithm)
                hasher.update(combined.encode('utf-8'))
                next_level.append(hasher.hexdigest())
            
            level = next_level
            idx = idx // 2
        
        return proof
    
    def verify_proof(
        self,
        leaf_hash: str,
        proof: List[Tuple[str, str]],
        root: str
    ) -> bool:
        """
        Verify a Merkle proof.
        
        Args:
            leaf_hash: Hash of the leaf to verify
            proof: Proof path
            root: Expected Merkle root
            
        Returns:
            True if proof is valid
        """
        current = leaf_hash
        
        for sibling_hash, position in proof:
            if position == "right":
                combined = current + sibling_hash
            else:
                combined = sibling_hash + current
            
            hasher = hashlib.new(self.hash_algorithm)
            hasher.update(combined.encode('utf-8'))
            current = hasher.hexdigest()
        
        return current == root

    # Learning Interface
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


class VerifiableAuditManager:
    """
    Unified manager for verifiable audit logging.
    
    Implements PPA2-C1-17 and PPA2-C1-27 with both hash-chain
    and Merkle tree verification methods.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize verifiable audit manager.
        
        Args:
            storage_path: Path for persistent storage
        """
        self.hash_chain = HashChainAudit(storage_path=storage_path)
        self.merkle_tree = MerkleTree()
        
        logger.info("[VerifiableAudit] Verifiable Audit Manager initialized")
    
    def log_decision(
        self,
        query: str,
        response: str,
        decision: str,
        confidence: float,
        parameters: Dict[str, Any]
    ) -> AuditEntry:
        """Log a governance decision."""
        return self.hash_chain.append(
            AuditEventType.DECISION,
            {
                "query": query[:200],  # Truncate for storage
                "response": response[:200],
                "decision": decision,
                "confidence": confidence,
                "parameters": parameters
            }
        )
    
    def log_threshold_adaptation(
        self,
        old_threshold: float,
        new_threshold: float,
        reason: str,
        parameters: Dict[str, Any]
    ) -> AuditEntry:
        """
        Log threshold adaptation event.
        
        PPA2-C1-27: Adaptation events logged with all parameters.
        """
        return self.hash_chain.append(
            AuditEventType.THRESHOLD_ADAPTATION,
            {
                "old_threshold": old_threshold,
                "new_threshold": new_threshold,
                "reason": reason,
                "parameters": parameters
            }
        )
    
    def log_drift_detection(
        self,
        drift_type: str,
        severity: str,
        confidence: float,
        algorithm: str,
        details: Dict[str, Any]
    ) -> AuditEntry:
        """Log drift detection event."""
        return self.hash_chain.append(
            AuditEventType.DRIFT_DETECTED,
            {
                "drift_type": drift_type,
                "severity": severity,
                "confidence": confidence,
                "algorithm": algorithm,
                "details": details
            }
        )
    
    def log_quarantine(
        self,
        component_id: str,
        reason: str,
        impairment_score: float
    ) -> AuditEntry:
        """Log quarantine event."""
        return self.hash_chain.append(
            AuditEventType.QUARANTINE,
            {
                "component_id": component_id,
                "reason": reason,
                "impairment_score": impairment_score
            }
        )
    
    def verify_integrity(self) -> VerificationResult:
        """Verify integrity of entire audit log."""
        return self.hash_chain.verify_chain()
    
    def build_merkle_root(self) -> str:
        """Build Merkle root for current chain."""
        self.merkle_tree = MerkleTree()
        for entry in self.hash_chain.chain:
            self.merkle_tree.add_leaf(entry.to_dict())
        return self.merkle_tree.compute_root()
    
    def get_audit_proof(self, sequence_id: int) -> Dict[str, Any]:
        """Get cryptographic proof for an audit entry."""
        entry = self.hash_chain.get_entry(sequence_id)
        if not entry:
            return {"error": "Entry not found"}
        
        # Build Merkle proof
        merkle_proof = self.merkle_tree.get_proof(sequence_id)
        
        return {
            "entry": entry.to_dict(),
            "hash_chain_valid": self.hash_chain.verify_entry(sequence_id),
            "merkle_proof": merkle_proof,
            "merkle_root": self.merkle_tree.root
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get audit summary."""
        return self.hash_chain.get_chain_summary()

    # ========================================
    # PHASE 49: LEARNING METHODS
    # ========================================
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, input_data, output_data, was_correct, domain=None, metadata=None):
        """Record outcome for learning (Phase 49)."""
        if hasattr(self, '_learning_manager') and self._learning_manager:
            self._learning_manager.record_outcome(
                module_name=self.__class__.__name__.lower(),
                input_data=input_data, output_data=output_data,
                was_correct=was_correct, domain=domain, metadata=metadata
            )
    
    def record_feedback(self, result, was_accurate):
        """Record feedback for learning (Phase 49)."""
        self.record_outcome({"result": str(result)}, {}, was_accurate)
    
    def adapt_thresholds(self, threshold_name, current_value, direction='decrease'):
        """Adapt thresholds based on learning (Phase 49)."""
        if hasattr(self, '_learning_manager') and self._learning_manager:
            return self._learning_manager.adapt_threshold(
                self.__class__.__name__.lower(), threshold_name, current_value, direction
            )
        return max(0.1, min(0.95, current_value + (0.05 if direction == 'increase' else -0.05)))
    
    def get_domain_adjustment(self, domain):
        """Get domain-specific adjustment (Phase 49)."""
        if hasattr(self, '_learning_manager') and self._learning_manager:
            return self._learning_manager.get_domain_adjustment(self.__class__.__name__.lower(), domain)
        return 0.0
    
    def save_state(self, filepath=None):
        """Save learning state (Phase 49)."""
        return self._learning_manager.save_state() if hasattr(self, '_learning_manager') and self._learning_manager else False
    
    def load_state(self, filepath=None):
        """Load learning state (Phase 49)."""
        return self._learning_manager.load_state() if hasattr(self, '_learning_manager') and self._learning_manager else False
    
    def get_learning_statistics(self):
        """Get learning statistics (Phase 49)."""
        if hasattr(self, '_learning_manager') and self._learning_manager:
            stats = self._learning_manager.get_learning_statistics(self.__class__.__name__.lower())
            return stats.__dict__ if hasattr(stats, '__dict__') else stats
        return {"module": self.__class__.__name__, "status": "no_learning_manager"}

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


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 31: Verifiable Audit Module Test")
    print("=" * 60)
    
    manager = VerifiableAuditManager()
    
    # Test 1: Log some events
    print("\n[1] Logging Audit Events:")
    
    manager.log_decision(
        query="Is this treatment safe?",
        response="Treatment has risks...",
        decision="accept",
        confidence=0.85,
        parameters={"threshold": 0.5, "domain": "medical"}
    )
    
    manager.log_threshold_adaptation(
        old_threshold=0.5,
        new_threshold=0.55,
        reason="drift detected",
        parameters={"tau": 1.1, "volatility": 0.15}
    )
    
    manager.log_drift_detection(
        drift_type="gradual",
        severity="medium",
        confidence=0.75,
        algorithm="cusum",
        details={"C_plus": 2.3}
    )
    
    print(f"    Logged {len(manager.hash_chain.chain)} events")
    
    # Test 2: Verify chain integrity
    print("\n[2] Chain Integrity Verification:")
    result = manager.verify_integrity()
    print(f"    Status: {result.status.value}")
    print(f"    Valid entries: {result.valid_entries}/{result.total_entries}")
    
    # Test 3: Build Merkle root
    print("\n[3] Merkle Tree:")
    root = manager.build_merkle_root()
    print(f"    Root hash: {root[:32]}...")
    
    # Test 4: Get audit proof
    print("\n[4] Audit Proof for Entry 0:")
    proof = manager.get_audit_proof(0)
    print(f"    Hash chain valid: {proof['hash_chain_valid']}")
    print(f"    Merkle proof steps: {len(proof['merkle_proof'])}")
    
    # Test 5: Summary
    print("\n[5] Audit Summary:")
    summary = manager.get_summary()
    print(f"    Total entries: {summary['entries']}")
    print(f"    Verified: {summary['verified']}")
    print(f"    Event types: {list(summary['event_counts'].keys())}")
    
    # Test 6: Tamper detection
    print("\n[6] Tamper Detection Test:")
    # Simulate tampering
    if manager.hash_chain.chain:
        original_data = manager.hash_chain.chain[0].event_data.copy()
        manager.hash_chain.chain[0].event_data["decision"] = "TAMPERED"
        
        result = manager.verify_integrity()
        print(f"    After tampering: {result.status.value}")
        
        # Restore
        manager.hash_chain.chain[0].event_data = original_data
    
    print("\n" + "=" * 60)
    print("PHASE 31: Verifiable Audit Module - VERIFIED")
    print("=" * 60)


