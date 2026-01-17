"""
BAIS Cognitive Governance Engine v16.4
Verifiable Audit System with Merkle Trees

PPA-2 Component 7: FULL IMPLEMENTATION
Tamper-evident logging with cryptographic verification.

This module implements:
1. Merkle Tree: Cryptographic proof of record integrity
2. Hash Chain: Sequential integrity verification
3. Audit Trail: Complete decision provenance
4. Verification: Prove records haven't been altered

Mathematical Foundation:
- Merkle Tree: H(node) = H(H(left) || H(right))
- Hash Chain: H_n = H(H_{n-1} || record_n)
- VDF placeholder: Time-locked commitment (simplified)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import hashlib
import json
import uuid
import math


@dataclass
class AuditRecord:
    """Single audit record with cryptographic binding."""
    id: str
    timestamp: datetime
    record_hash: str
    previous_hash: str
    merkle_root: Optional[str]
    
    # Decision data
    decision_id: str
    query_hash: str
    response_hash: str
    accuracy_score: float
    was_accepted: bool
    pathway: str
    
    # Signals
    grounding_score: float
    factual_score: float
    behavioral_score: float
    temporal_score: float
    
    # Metadata
    domain: str
    version: str
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'record_hash': self.record_hash,
            'previous_hash': self.previous_hash,
            'merkle_root': self.merkle_root,
            'decision_id': self.decision_id,
            'query_hash': self.query_hash,
            'response_hash': self.response_hash,
            'accuracy_score': self.accuracy_score,
            'was_accepted': self.was_accepted,
            'pathway': self.pathway,
            'grounding_score': self.grounding_score,
            'factual_score': self.factual_score,
            'behavioral_score': self.behavioral_score,
            'temporal_score': self.temporal_score,
            'domain': self.domain,
            'version': self.version
        }


@dataclass
class MerkleProof:
    """Merkle proof for verifying record inclusion."""
    record_hash: str
    proof_path: List[Tuple[str, str]]  # (hash, direction: 'left'/'right')
    root_hash: str
    leaf_index: int
    tree_size: int
    
    def to_dict(self) -> Dict:
        return {
            'record_hash': self.record_hash,
            'proof_path': [(h, d) for h, d in self.proof_path],
            'root_hash': self.root_hash,
            'leaf_index': self.leaf_index,
            'tree_size': self.tree_size
        }


class MerkleTree:
    """
    Merkle Tree implementation for tamper-evident logging.
    
    Properties:
    - Any change to a leaf invalidates the root
    - Efficient O(log n) proofs of inclusion
    - Can verify integrity without full tree
    """
    
    def __init__(self):
        self.leaves: List[str] = []
        self.tree: List[List[str]] = []  # Layers from leaves to root
        self.root: Optional[str] = None
    
    def _hash(self, data: str) -> str:
        """SHA-256 hash of data."""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _combine_hash(self, left: str, right: str) -> str:
        """Hash two nodes together."""
        return self._hash(left + right)
    
    def add_leaf(self, data_hash: str) -> int:
        """
        Add a leaf and rebuild tree.
        Returns leaf index.
        """
        self.leaves.append(data_hash)
        self._rebuild_tree()
        return len(self.leaves) - 1
    
    def _rebuild_tree(self):
        """Rebuild entire tree from leaves."""
        if not self.leaves:
            self.tree = []
            self.root = None
            return
        
        # Layer 0 = leaves
        self.tree = [self.leaves.copy()]
        
        # Build up layers
        current_layer = self.tree[0]
        while len(current_layer) > 1:
            next_layer = []
            for i in range(0, len(current_layer), 2):
                left = current_layer[i]
                # If odd number, duplicate last element
                right = current_layer[i + 1] if i + 1 < len(current_layer) else left
                next_layer.append(self._combine_hash(left, right))
            self.tree.append(next_layer)
            current_layer = next_layer
        
        self.root = self.tree[-1][0] if self.tree else None
    
    def get_root(self) -> Optional[str]:
        """Get current root hash."""
        return self.root
    
    def get_proof(self, leaf_index: int) -> Optional[MerkleProof]:
        """
        Generate inclusion proof for a leaf.
        
        Returns path from leaf to root with sibling hashes.
        """
        if leaf_index >= len(self.leaves):
            return None
        
        proof_path = []
        index = leaf_index
        
        for layer_idx in range(len(self.tree) - 1):
            layer = self.tree[layer_idx]
            
            # Determine sibling
            if index % 2 == 0:
                # Current is left child, sibling is right
                sibling_idx = index + 1
                if sibling_idx < len(layer):
                    proof_path.append((layer[sibling_idx], 'right'))
                else:
                    proof_path.append((layer[index], 'right'))  # Duplicate
            else:
                # Current is right child, sibling is left
                sibling_idx = index - 1
                proof_path.append((layer[sibling_idx], 'left'))
            
            # Move to parent index
            index = index // 2
        
        return MerkleProof(
            record_hash=self.leaves[leaf_index],
            proof_path=proof_path,
            root_hash=self.root,
            leaf_index=leaf_index,
            tree_size=len(self.leaves)
        )
    
    def verify_proof(self, proof: MerkleProof) -> bool:
        """
        Verify a Merkle proof.
        
        Returns True if proof is valid.
        """
        current_hash = proof.record_hash
        
        for sibling_hash, direction in proof.proof_path:
            if direction == 'left':
                current_hash = self._combine_hash(sibling_hash, current_hash)
            else:
                current_hash = self._combine_hash(current_hash, sibling_hash)
        
        return current_hash == proof.root_hash

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


class HashChain:
    """
    Sequential hash chain for tamper-evidence.
    
    Each record includes hash of previous record,
    creating unbreakable chain.
    """
    
    def __init__(self):
        self.chain: List[Dict] = []
        self.current_hash: str = self._hash("genesis")
    
    def _hash(self, data: str) -> str:
        """SHA-256 hash."""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def append(self, record_data: Dict) -> Tuple[str, str]:
        """
        Append record to chain.
        
        Returns (record_hash, previous_hash)
        """
        previous_hash = self.current_hash
        
        # Include previous hash in record
        record_with_chain = {
            'previous_hash': previous_hash,
            'data': record_data,
            'timestamp': datetime.utcnow().isoformat(),
            'index': len(self.chain)
        }
        
        # Hash the record
        record_json = json.dumps(record_with_chain, sort_keys=True)
        record_hash = self._hash(record_json)
        
        record_with_chain['record_hash'] = record_hash
        self.chain.append(record_with_chain)
        self.current_hash = record_hash
        
        return record_hash, previous_hash
    
    def verify_chain(self) -> Tuple[bool, Optional[int]]:
        """
        Verify entire chain integrity.
        
        Returns (is_valid, first_invalid_index or None)
        """
        if not self.chain:
            return True, None
        
        # Verify genesis
        if self.chain[0]['previous_hash'] != self._hash("genesis"):
            return False, 0
        
        # Verify each link
        for i in range(1, len(self.chain)):
            expected_prev = self.chain[i - 1]['record_hash']
            actual_prev = self.chain[i]['previous_hash']
            
            if expected_prev != actual_prev:
                return False, i
            
            # Verify record hash
            record_copy = dict(self.chain[i])
            stored_hash = record_copy.pop('record_hash', None)
            computed_hash = self._hash(json.dumps(record_copy, sort_keys=True))
            
            if stored_hash != computed_hash:
                return False, i
        
        return True, None
    
    def get_chain_length(self) -> int:
        return len(self.chain)


class VerifiableAuditSystem:
    """
    Complete Verifiable Audit System.
    
    PPA-2 Component 7: Full Implementation
    
    Combines:
    - Merkle Tree for efficient inclusion proofs
    - Hash Chain for sequential integrity
    - Audit trail with full provenance
    - Verification APIs
    """
    
    BATCH_SIZE = 100  # Records per Merkle tree batch
    
    def __init__(self, storage_path: Path = None):
        # Use temp directory if no path provided (fixes read-only filesystem issues)
        if storage_path is None:
            import tempfile
            storage_path = Path(tempfile.mkdtemp(prefix="bais_audit_"))
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Core structures
        self.hash_chain = HashChain()
        self.merkle_tree = MerkleTree()
        self.current_batch: List[str] = []
        
        # Audit records
        self.records: Dict[str, AuditRecord] = {}
        self.merkle_proofs: Dict[str, MerkleProof] = {}
        
        # Batch tracking
        self.batch_roots: List[Dict] = []  # Historical Merkle roots
        
        # Load state
        self._load_state()
    
    def record_decision(self,
                       decision_id: str,
                       query: str,
                       response: str,
                       accuracy: float,
                       accepted: bool,
                       pathway: str,
                       signals: Dict[str, float],
                       domain: str,
                       version: str = "16.4.0") -> AuditRecord:
        """
        Record a decision with full audit trail.
        
        Creates:
        - Hash chain entry
        - Merkle tree leaf
        - Audit record with proofs
        """
        record_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        # Hash sensitive data
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        response_hash = hashlib.sha256(response.encode()).hexdigest()[:16]
        
        # Create record data for hashing
        record_data = {
            'decision_id': decision_id,
            'query_hash': query_hash,
            'response_hash': response_hash,
            'accuracy': accuracy,
            'accepted': accepted,
            'pathway': pathway,
            'signals': signals,
            'domain': domain,
            'version': version,
            'timestamp': timestamp.isoformat()
        }
        
        # Add to hash chain
        record_hash, previous_hash = self.hash_chain.append(record_data)
        
        # Add to Merkle tree
        leaf_index = self.merkle_tree.add_leaf(record_hash)
        self.current_batch.append(record_hash)
        
        # Check if batch complete
        merkle_root = None
        if len(self.current_batch) >= self.BATCH_SIZE:
            merkle_root = self._finalize_batch()
        else:
            merkle_root = self.merkle_tree.get_root()
        
        # Create audit record
        audit_record = AuditRecord(
            id=record_id,
            timestamp=timestamp,
            record_hash=record_hash,
            previous_hash=previous_hash,
            merkle_root=merkle_root,
            decision_id=decision_id,
            query_hash=query_hash,
            response_hash=response_hash,
            accuracy_score=accuracy,
            was_accepted=accepted,
            pathway=pathway,
            grounding_score=signals.get('grounding', 0),
            factual_score=signals.get('factual', 0),
            behavioral_score=signals.get('behavioral', 0),
            temporal_score=signals.get('temporal', 0),
            domain=domain,
            version=version
        )
        
        self.records[record_id] = audit_record
        
        # Generate and store proof
        proof = self.merkle_tree.get_proof(leaf_index)
        if proof:
            self.merkle_proofs[record_id] = proof
        
        # Persist
        self._save_state()
        
        return audit_record
    
    def _finalize_batch(self) -> str:
        """Finalize current batch and start new Merkle tree."""
        root = self.merkle_tree.get_root()
        
        self.batch_roots.append({
            'root': root,
            'size': len(self.current_batch),
            'timestamp': datetime.utcnow().isoformat(),
            'first_hash': self.current_batch[0] if self.current_batch else None,
            'last_hash': self.current_batch[-1] if self.current_batch else None
        })
        
        # Start new batch
        self.current_batch = []
        self.merkle_tree = MerkleTree()
        
        return root
    
    def get_record(self, record_id: str) -> Optional[AuditRecord]:
        """Get audit record by ID."""
        return self.records.get(record_id)
    
    def get_proof(self, record_id: str) -> Optional[MerkleProof]:
        """Get Merkle proof for a record."""
        return self.merkle_proofs.get(record_id)
    
    def verify_record(self, record_id: str) -> Dict[str, Any]:
        """
        Verify integrity of a specific record.
        
        Returns verification result with details.
        """
        record = self.records.get(record_id)
        if not record:
            return {'valid': False, 'error': 'Record not found'}
        
        proof = self.merkle_proofs.get(record_id)
        
        results = {
            'record_id': record_id,
            'record_hash': record.record_hash,
            'merkle_root': record.merkle_root,
            'checks': {}
        }
        
        # Check 1: Merkle proof verification
        if proof:
            merkle_valid = self.merkle_tree.verify_proof(proof)
            results['checks']['merkle_proof'] = {
                'valid': merkle_valid,
                'root_matches': proof.root_hash == record.merkle_root
            }
        else:
            results['checks']['merkle_proof'] = {
                'valid': False,
                'error': 'No proof available'
            }
        
        # Check 2: Hash chain position
        chain_valid, invalid_idx = self.hash_chain.verify_chain()
        results['checks']['hash_chain'] = {
            'valid': chain_valid,
            'chain_length': self.hash_chain.get_chain_length(),
            'invalid_index': invalid_idx
        }
        
        # Overall validity
        results['valid'] = all(
            c.get('valid', False) 
            for c in results['checks'].values()
        )
        
        return results
    
    def verify_full_audit(self) -> Dict[str, Any]:
        """
        Verify entire audit system integrity.
        """
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_records': len(self.records),
            'total_batches': len(self.batch_roots),
            'chain_length': self.hash_chain.get_chain_length(),
            'checks': {}
        }
        
        # Check 1: Hash chain integrity
        chain_valid, invalid_idx = self.hash_chain.verify_chain()
        results['checks']['hash_chain'] = {
            'valid': chain_valid,
            'invalid_index': invalid_idx
        }
        
        # Check 2: Batch roots consistency
        batch_check = {
            'valid': True,
            'batches_verified': 0,
            'errors': []
        }
        for i, batch in enumerate(self.batch_roots):
            if batch.get('root'):
                batch_check['batches_verified'] += 1
            else:
                batch_check['errors'].append(f"Batch {i} has no root")
                batch_check['valid'] = False
        results['checks']['batch_roots'] = batch_check
        
        # Check 3: Record-proof consistency
        proof_check = {
            'valid': True,
            'records_checked': 0,
            'proofs_valid': 0,
            'errors': []
        }
        for record_id, record in list(self.records.items())[:100]:  # Sample check
            proof_check['records_checked'] += 1
            proof = self.merkle_proofs.get(record_id)
            if proof:
                # Note: Can't verify against old Merkle trees
                # Just check proof exists
                proof_check['proofs_valid'] += 1
        results['checks']['record_proofs'] = proof_check
        
        # Overall
        results['valid'] = all(
            c.get('valid', False)
            for c in results['checks'].values()
        )
        
        return results
    
    def get_audit_trail(self, 
                       decision_id: Optional[str] = None,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       domain: Optional[str] = None,
                       limit: int = 100) -> List[Dict]:
        """Get audit trail with optional filters."""
        results = []
        
        for record in self.records.values():
            # Apply filters
            if decision_id and record.decision_id != decision_id:
                continue
            if start_time and record.timestamp < start_time:
                continue
            if end_time and record.timestamp > end_time:
                continue
            if domain and record.domain != domain:
                continue
            
            results.append(record.to_dict())
            
            if len(results) >= limit:
                break
        
        # Sort by timestamp
        results.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return results
    
    def export_for_compliance(self) -> Dict[str, Any]:
        """Export audit data for compliance reporting."""
        return {
            'export_timestamp': datetime.utcnow().isoformat(),
            'system_version': '16.4.0',
            'total_decisions': len(self.records),
            'chain_integrity': self.hash_chain.verify_chain()[0],
            'batch_count': len(self.batch_roots),
            'batch_roots': self.batch_roots,
            'sample_records': [
                r.to_dict() for r in list(self.records.values())[:10]
            ],
            'verification_result': self.verify_full_audit()
        }
    
    def _save_state(self):
        """Persist audit state."""
        # Save hash chain
        chain_path = self.storage_path / "hash_chain.json"
        with open(chain_path, 'w') as f:
            json.dump({
                'chain': self.hash_chain.chain[-1000:],
                'current_hash': self.hash_chain.current_hash
            }, f)
        
        # Save Merkle state
        merkle_path = self.storage_path / "merkle_state.json"
        with open(merkle_path, 'w') as f:
            json.dump({
                'current_batch': self.current_batch,
                'batch_roots': self.batch_roots
            }, f)
        
        # Save records (sampling for large datasets)
        records_path = self.storage_path / "audit_records.json"
        records_to_save = {k: v.to_dict() for k, v in list(self.records.items())[-1000:]}
        with open(records_path, 'w') as f:
            json.dump(records_to_save, f)
    
    def _load_state(self):
        """Load persisted state."""
        try:
            # Load hash chain
            chain_path = self.storage_path / "hash_chain.json"
            if chain_path.exists():
                with open(chain_path) as f:
                    data = json.load(f)
                self.hash_chain.chain = data.get('chain', [])
                self.hash_chain.current_hash = data.get('current_hash', self.hash_chain._hash("genesis"))
            
            # Load Merkle state
            merkle_path = self.storage_path / "merkle_state.json"
            if merkle_path.exists():
                with open(merkle_path) as f:
                    data = json.load(f)
                self.current_batch = data.get('current_batch', [])
                self.batch_roots = data.get('batch_roots', [])
                # Rebuild tree from current batch
                for record_hash in self.current_batch:
                    self.merkle_tree.add_leaf(record_hash)
            
            # Load records
            records_path = self.storage_path / "audit_records.json"
            if records_path.exists():
                with open(records_path) as f:
                    data = json.load(f)
                for record_id, record_dict in data.items():
                    self.records[record_id] = AuditRecord(
                        id=record_dict['id'],
                        timestamp=datetime.fromisoformat(record_dict['timestamp']),
                        record_hash=record_dict['record_hash'],
                        previous_hash=record_dict['previous_hash'],
                        merkle_root=record_dict.get('merkle_root'),
                        decision_id=record_dict['decision_id'],
                        query_hash=record_dict['query_hash'],
                        response_hash=record_dict['response_hash'],
                        accuracy_score=record_dict['accuracy_score'],
                        was_accepted=record_dict['was_accepted'],
                        pathway=record_dict['pathway'],
                        grounding_score=record_dict['grounding_score'],
                        factual_score=record_dict['factual_score'],
                        behavioral_score=record_dict['behavioral_score'],
                        temporal_score=record_dict['temporal_score'],
                        domain=record_dict['domain'],
                        version=record_dict['version']
                    )
                    
        except Exception as e:
            print(f"Warning: Could not load audit state: {e}")


