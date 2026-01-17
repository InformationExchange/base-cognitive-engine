"""
AUDIT TRAIL & CASE MANAGEMENT (NOVEL-33)
=========================================
Persistent storage of all governance decisions with case IDs.

Features:
- Case ID generation
- Transaction logging
- Audit record storage (JSON/SQLite)
- Query and retrieval
- Statistics and reporting

Created: December 26, 2025
Patent: NOVEL-33 - Governance Audit Trail System
"""

import json
import uuid
import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum

# Import clinical status if available
try:
    from .clinical_status_classifier import ClinicalStatus
except ImportError:
    class ClinicalStatus(Enum):
        TRULY_WORKING = "truly_working"
        INCOMPLETE = "incomplete"
        STUBBED = "stubbed"
        SIMULATED = "simulated"
        FALLBACK = "fallback"
        FAILOVER = "failover"
        UNKNOWN = "unknown"


class AuditAction(Enum):
    """Types of audit actions."""
    EVALUATE = "evaluate"
    VERIFY_COMPLETION = "verify_completion"
    CHECK_QUERY = "check_query"
    IMPROVE_RESPONSE = "improve_response"
    AB_TEST = "ab_test"
    PROOF_ANALYSIS = "proof_analysis"


class AuditDecision(Enum):
    """Possible governance decisions."""
    ACCEPT = "accept"
    REJECT = "reject"
    ENHANCE = "enhance"
    BLOCK = "block"
    WARNING = "warning"


@dataclass
class AuditRecord:
    """Complete audit record for a governance decision."""
    case_id: str
    transaction_id: str
    timestamp: str
    action: str
    query: str
    response: str
    decision: str
    confidence: float
    clinical_status: str
    llm_reasoning: str
    proof_analysis: Dict[str, Any]
    gaps_identified: List[str]
    warnings: List[str]
    inventions_used: List[str]
    brain_layers_activated: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditRecord':
        return cls(**data)


@dataclass
class Case:
    """A case groups related audit records."""
    case_id: str
    created: str
    status: str  # open, closed, escalated
    domain: str
    description: str
    related_cases: List[str] = field(default_factory=list)
    audit_records: List[str] = field(default_factory=list)  # transaction IDs
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Case':
        return cls(**data)


class AuditTrailManager:
    """
    Manages the audit trail for all governance decisions.
    
    Provides persistent storage, querying, and statistics.
    """
    
    def __init__(self, storage_dir: str = None):
        """
        Initialize audit trail manager.
        
        Args:
            storage_dir: Directory for audit storage. Defaults to ./audit_data
        """
        if storage_dir is None:
            storage_dir = os.path.join(os.path.dirname(__file__), "..", "..", "audit_data")
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage files
        self.records_file = self.storage_dir / "audit_records.json"
        self.cases_file = self.storage_dir / "cases.json"
        self.transactions_file = self.storage_dir / "transactions.json"
        
        # In-memory cache
        self._records: Dict[str, AuditRecord] = {}
        self._cases: Dict[str, Case] = {}
        self._transactions: List[Dict] = []
        
        # Load existing data
        self._load_data()
    
    def _load_data(self):
        """Load existing audit data from storage."""
        # Load records
        if self.records_file.exists():
            try:
                with open(self.records_file, 'r') as f:
                    data = json.load(f)
                    self._records = {
                        k: AuditRecord.from_dict(v) 
                        for k, v in data.items()
                    }
            except (json.JSONDecodeError, KeyError):
                self._records = {}
        
        # Load cases
        if self.cases_file.exists():
            try:
                with open(self.cases_file, 'r') as f:
                    data = json.load(f)
                    self._cases = {
                        k: Case.from_dict(v)
                        for k, v in data.items()
                    }
            except (json.JSONDecodeError, KeyError):
                self._cases = {}
        
        # Load transactions
        if self.transactions_file.exists():
            try:
                with open(self.transactions_file, 'r') as f:
                    self._transactions = json.load(f)
            except json.JSONDecodeError:
                self._transactions = []
    
    def _save_data(self):
        """Persist audit data to storage."""
        # Save records
        with open(self.records_file, 'w') as f:
            json.dump(
                {k: v.to_dict() for k, v in self._records.items()},
                f,
                indent=2
            )
        
        # Save cases
        with open(self.cases_file, 'w') as f:
            json.dump(
                {k: v.to_dict() for k, v in self._cases.items()},
                f,
                indent=2
            )
        
        # Save transactions (append-only conceptually, but we write all)
        with open(self.transactions_file, 'w') as f:
            json.dump(self._transactions, f, indent=2)
    
    def generate_case_id(self) -> str:
        """Generate a unique case ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d")
        unique = uuid.uuid4().hex[:8].upper()
        return f"CASE-{timestamp}-{unique}"
    
    def generate_transaction_id(self) -> str:
        """Generate a unique transaction ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        unique = uuid.uuid4().hex[:6].upper()
        return f"TX-{timestamp}-{unique}"
    
    def create_case(
        self,
        domain: str,
        description: str,
        tags: List[str] = None,
        related_cases: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> Case:
        """
        Create a new case.
        
        Args:
            domain: Domain of the case (medical, legal, coding, etc.)
            description: Brief description
            tags: Optional tags for categorization
            related_cases: Optional related case IDs
            metadata: Optional additional metadata
            
        Returns:
            Created Case object
        """
        case = Case(
            case_id=self.generate_case_id(),
            created=datetime.utcnow().isoformat(),
            status="open",
            domain=domain,
            description=description,
            related_cases=related_cases or [],
            audit_records=[],
            tags=tags or [],
            metadata=metadata or {}
        )
        
        self._cases[case.case_id] = case
        self._save_data()
        
        return case
    
    def record_audit(
        self,
        case_id: str,
        action: AuditAction,
        query: str,
        response: str,
        decision: AuditDecision,
        confidence: float,
        clinical_status: ClinicalStatus = None,
        llm_reasoning: str = "",
        proof_analysis: Dict[str, Any] = None,
        gaps_identified: List[str] = None,
        warnings: List[str] = None,
        inventions_used: List[str] = None,
        brain_layers_activated: List[int] = None,
        metadata: Dict[str, Any] = None
    ) -> AuditRecord:
        """
        Record an audit event.
        
        Args:
            case_id: The case this audit belongs to
            action: Type of action performed
            query: Original query/request
            response: LLM response audited
            decision: Governance decision
            confidence: Confidence in the decision (0-1)
            clinical_status: Clinical status classification
            llm_reasoning: LLM's reasoning for proof
            proof_analysis: Detailed proof analysis results
            gaps_identified: List of identified gaps
            warnings: List of warnings generated
            inventions_used: List of invention IDs exercised
            brain_layers_activated: List of brain layers activated
            metadata: Additional metadata
            
        Returns:
            Created AuditRecord
        """
        transaction_id = self.generate_transaction_id()
        
        record = AuditRecord(
            case_id=case_id,
            transaction_id=transaction_id,
            timestamp=datetime.utcnow().isoformat(),
            action=action.value if isinstance(action, AuditAction) else action,
            query=query[:1000],  # Truncate for storage
            response=response[:2000],  # Truncate for storage
            decision=decision.value if isinstance(decision, AuditDecision) else decision,
            confidence=confidence,
            clinical_status=clinical_status.value if clinical_status else "unknown",
            llm_reasoning=llm_reasoning,
            proof_analysis=proof_analysis or {},
            gaps_identified=gaps_identified or [],
            warnings=warnings or [],
            inventions_used=inventions_used or [],
            brain_layers_activated=brain_layers_activated or [],
            metadata=metadata or {}
        )
        
        # Store record
        self._records[transaction_id] = record
        
        # Update case
        if case_id in self._cases:
            self._cases[case_id].audit_records.append(transaction_id)
        
        # Log transaction
        self._transactions.append({
            "transaction_id": transaction_id,
            "case_id": case_id,
            "action": record.action,
            "decision": record.decision,
            "timestamp": record.timestamp
        })
        
        # Persist
        self._save_data()
        
        return record
    
    def get_record(self, transaction_id: str) -> Optional[AuditRecord]:
        """Get an audit record by transaction ID."""
        return self._records.get(transaction_id)
    
    def get_case(self, case_id: str) -> Optional[Case]:
        """Get a case by case ID."""
        return self._cases.get(case_id)
    
    def get_case_records(self, case_id: str) -> List[AuditRecord]:
        """Get all audit records for a case."""
        case = self._cases.get(case_id)
        if not case:
            return []
        
        return [
            self._records[tx_id]
            for tx_id in case.audit_records
            if tx_id in self._records
        ]
    
    def query_records(
        self,
        action: AuditAction = None,
        decision: AuditDecision = None,
        clinical_status: ClinicalStatus = None,
        domain: str = None,
        since: datetime = None,
        limit: int = 100
    ) -> List[AuditRecord]:
        """
        Query audit records with filters.
        
        Args:
            action: Filter by action type
            decision: Filter by decision
            clinical_status: Filter by clinical status
            domain: Filter by domain (requires case lookup)
            since: Filter by timestamp (records after this time)
            limit: Maximum records to return
            
        Returns:
            List of matching AuditRecords
        """
        results = []
        
        for record in self._records.values():
            # Apply filters
            if action and record.action != action.value:
                continue
            if decision and record.decision != decision.value:
                continue
            if clinical_status and record.clinical_status != clinical_status.value:
                continue
            if since:
                record_time = datetime.fromisoformat(record.timestamp)
                if record_time < since:
                    continue
            if domain:
                case = self._cases.get(record.case_id)
                if not case or case.domain != domain:
                    continue
            
            results.append(record)
            
            if len(results) >= limit:
                break
        
        # Sort by timestamp descending
        results.sort(key=lambda r: r.timestamp, reverse=True)
        return results[:limit]
    
    def close_case(self, case_id: str, resolution: str = None):
        """Close a case."""
        case = self._cases.get(case_id)
        if case:
            case.status = "closed"
            if resolution:
                case.metadata["resolution"] = resolution
            case.metadata["closed_at"] = datetime.utcnow().isoformat()
            self._save_data()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit trail statistics."""
        if not self._records:
            return {
                "total_records": 0,
                "total_cases": 0,
                "total_transactions": 0
            }
        
        # Count by decision
        decisions = {}
        for record in self._records.values():
            decisions[record.decision] = decisions.get(record.decision, 0) + 1
        
        # Count by clinical status
        statuses = {}
        for record in self._records.values():
            statuses[record.clinical_status] = statuses.get(record.clinical_status, 0) + 1
        
        # Count by action
        actions = {}
        for record in self._records.values():
            actions[record.action] = actions.get(record.action, 0) + 1
        
        # Case statistics
        open_cases = sum(1 for c in self._cases.values() if c.status == "open")
        closed_cases = sum(1 for c in self._cases.values() if c.status == "closed")
        
        return {
            "total_records": len(self._records),
            "total_cases": len(self._cases),
            "total_transactions": len(self._transactions),
            "by_decision": decisions,
            "by_clinical_status": statuses,
            "by_action": actions,
            "open_cases": open_cases,
            "closed_cases": closed_cases,
            "average_confidence": sum(r.confidence for r in self._records.values()) / len(self._records) if self._records else 0
        }
    
    def generate_report(self, case_id: str = None) -> str:
        """
        Generate a clinical report for a case or all recent activity.
        
        Args:
            case_id: Optional case ID. If None, reports on recent activity.
            
        Returns:
            Formatted clinical report string
        """
        if case_id:
            case = self._cases.get(case_id)
            if not case:
                return f"Case {case_id} not found."
            
            records = self.get_case_records(case_id)
            title = f"CLINICAL GOVERNANCE REPORT - CASE {case_id}"
        else:
            records = list(self._records.values())[-10:]  # Last 10
            title = "CLINICAL GOVERNANCE REPORT - RECENT ACTIVITY"
        
        report = [
            "=" * 70,
            title,
            "=" * 70,
            f"Generated: {datetime.utcnow().isoformat()}",
            ""
        ]
        
        if case_id and case:
            report.extend([
                f"Domain: {case.domain}",
                f"Status: {case.status}",
                f"Created: {case.created}",
                f"Description: {case.description}",
                ""
            ])
        
        report.append("-" * 70)
        report.append("AUDIT RECORDS")
        report.append("-" * 70)
        
        for record in records:
            report.extend([
                "",
                f"Transaction: {record.transaction_id}",
                f"Timestamp: {record.timestamp}",
                f"Action: {record.action}",
                f"Decision: {record.decision}",
                f"Confidence: {record.confidence:.2%}",
                f"Clinical Status: {record.clinical_status}",
                "",
                f"Query: {record.query[:200]}...",
                "",
                f"Gaps Identified: {len(record.gaps_identified)}",
                *[f"  - {gap}" for gap in record.gaps_identified[:5]],
                "",
                f"Warnings: {len(record.warnings)}",
                *[f"  - {w}" for w in record.warnings[:5]],
                "",
                f"Inventions Used: {', '.join(record.inventions_used[:5]) or 'None recorded'}",
                "",
                f"LLM Reasoning:",
                f"  {record.llm_reasoning[:500] if record.llm_reasoning else 'None'}",
                "",
                "-" * 40
            ])
        
        # Summary statistics
        stats = self.get_statistics()
        report.extend([
            "",
            "=" * 70,
            "SUMMARY STATISTICS",
            "=" * 70,
            f"Total Records: {stats['total_records']}",
            f"Total Cases: {stats['total_cases']}",
            f"Open Cases: {stats.get('open_cases', 0)}",
            f"Average Confidence: {stats.get('average_confidence', 0):.2%}",
            "",
            "By Decision:",
            *[f"  {k}: {v}" for k, v in stats.get('by_decision', {}).items()],
            "",
            "By Clinical Status:",
            *[f"  {k}: {v}" for k, v in stats.get('by_clinical_status', {}).items()],
        ])
        
        return "\n".join(report)

    
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

