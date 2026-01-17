"""
BASE Cognitive Governance Engine - Governance Output
Phase 7: Unified output structure across all modes

Provides consistent output format for:
- AUDIT_ONLY: Evaluation report
- AUDIT_AND_REMEDIATE: Evaluation + blocking/approval status
- DIRECT_ASSISTANCE: Enhanced response

Platform-agnostic serialization for API, Cursor, CLI, etc.

Patent: NOVEL-47 (Unified Governance Output)

Learning Interface: 5/5 methods implemented
- record_outcome(): Track output generation results
- learn_from_feedback(): Adjust formatting preferences
- get_statistics(): Return output generation metrics
- serialize_state(): Save learning state
- deserialize_state(): Restore learning state
"""

import logging
import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# =============================================================================
# Output Status Enums
# =============================================================================

class OutputStatus(Enum):
    """Overall output status."""
    SUCCESS = "success"           # Request completed successfully
    BLOCKED = "blocked"           # Request blocked, needs action
    ENHANCED = "enhanced"         # Response was enhanced
    PENDING_APPROVAL = "pending_approval"  # Awaiting approval
    FAILED = "failed"             # Request failed
    PARTIAL = "partial"           # Partially completed


class ActionRequired(Enum):
    """Actions that may be required from user."""
    NONE = "none"
    REVIEW = "review"             # Review issues but can proceed
    APPROVE = "approve"           # Must approve to proceed
    FIX = "fix"                   # Must fix issues to proceed
    SELECT = "select"             # Must select from options
    CLARIFY = "clarify"           # Must provide clarification


# =============================================================================
# Output Sections
# =============================================================================

@dataclass
class IssueReport:
    """Report on a single issue."""
    issue_id: str
    issue_type: str
    severity: str
    description: str
    location: Optional[str] = None
    suggested_fix: Optional[str] = None
    auto_fixable: bool = False


@dataclass
class EvidenceReport:
    """Report on evidence quality."""
    strength: str  # none, weak, medium, strong, verified
    score: float
    verified_items: List[str]
    failed_items: List[str]
    summary: str


@dataclass
class EnhancementReport:
    """Report on enhancements applied."""
    total_enhancements: int
    enhancement_types: List[str]
    confidence_before: float
    confidence_after: float
    changes_summary: List[str]


@dataclass
class MultiTrackReport:
    """Report on multi-track results."""
    enabled: bool
    tracks_executed: int
    track_summaries: List[Dict[str, Any]]
    consensus_used: bool
    selected_track: Optional[str]
    agreement_score: float


@dataclass
class ApprovalReport:
    """Report on approval status."""
    required: bool
    status: str  # pending, approved, rejected, timeout
    approval_id: Optional[str] = None
    timeout_at: Optional[str] = None
    action_items: List[str] = field(default_factory=list)


# =============================================================================
# Base Governance Output
# =============================================================================

@dataclass
class GovernanceOutput:
    """
    Unified output structure for all BASE governance modes.
    
    This is the standard output format that all BASE operations
    should return, regardless of mode or platform.
    """
    
    # Core identification
    output_id: str
    mode: str  # audit_only, audit_and_remediate, direct_assistance
    timestamp: str
    
    # Status
    status: OutputStatus
    action_required: ActionRequired
    
    # The response
    original_response: str
    final_response: str
    response_was_modified: bool
    
    # Reports
    issues: List[IssueReport]
    evidence: EvidenceReport
    enhancements: Optional[EnhancementReport]
    multi_track: Optional[MultiTrackReport]
    approval: Optional[ApprovalReport]
    
    # Metadata
    processing_time_ms: float
    base_version: str = "2.0"
    
    # For user
    user_message: str = ""
    next_steps: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'output_id': self.output_id,
            'mode': self.mode,
            'timestamp': self.timestamp,
            'status': self.status.value,
            'action_required': self.action_required.value,
            'original_response': self.original_response,
            'final_response': self.final_response,
            'response_was_modified': self.response_was_modified,
            'issues': [asdict(i) for i in self.issues],
            'evidence': asdict(self.evidence),
            'enhancements': asdict(self.enhancements) if self.enhancements else None,
            'multi_track': asdict(self.multi_track) if self.multi_track else None,
            'approval': asdict(self.approval) if self.approval else None,
            'processing_time_ms': self.processing_time_ms,
            'base_version': self.base_version,
            'user_message': self.user_message,
            'next_steps': self.next_steps
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_cursor_format(self) -> str:
        """Format for Cursor IDE display."""
        lines = []
        
        # Header
        lines.append(f"## BASE Governance Report [{self.mode.upper()}]")
        lines.append(f"Status: **{self.status.value.upper()}**")
        
        if self.action_required != ActionRequired.NONE:
            lines.append(f"⚠️ Action Required: **{self.action_required.value.upper()}**")
        
        # Issues
        if self.issues:
            lines.append(f"\n### Issues Detected ({len(self.issues)})")
            for issue in self.issues:
                severity_emoji = {'low': '💡', 'medium': '⚠️', 'high': '🔴', 'critical': '🚨'}.get(issue.severity, '•')
                lines.append(f"{severity_emoji} **{issue.issue_type}**: {issue.description}")
                if issue.suggested_fix:
                    lines.append(f"  → Fix: {issue.suggested_fix}")
        
        # Evidence
        lines.append(f"\n### Evidence Quality")
        lines.append(f"Strength: **{self.evidence.strength.upper()}** (score: {self.evidence.score:.2f})")
        
        # Enhancements
        if self.enhancements and self.enhancements.total_enhancements > 0:
            lines.append(f"\n### Enhancements Applied ({self.enhancements.total_enhancements})")
            for change in self.enhancements.changes_summary:
                lines.append(f"✓ {change}")
        
        # Next steps
        if self.next_steps:
            lines.append(f"\n### Next Steps")
            for i, step in enumerate(self.next_steps, 1):
                lines.append(f"{i}. {step}")
        
        # User message
        if self.user_message:
            lines.append(f"\n---\n{self.user_message}")
        
        return "\n".join(lines)
    
    def to_cli_format(self) -> str:
        """Format for CLI display."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"BASE Governance Report [{self.mode.upper()}]")
        lines.append("=" * 60)
        lines.append(f"Status: {self.status.value}")
        lines.append(f"Action Required: {self.action_required.value}")
        lines.append(f"Evidence: {self.evidence.strength} ({self.evidence.score:.2f})")
        lines.append(f"Issues: {len(self.issues)}")
        if self.enhancements:
            lines.append(f"Enhancements: {self.enhancements.total_enhancements}")
        lines.append("-" * 60)
        
        if self.next_steps:
            lines.append("Next Steps:")
            for step in self.next_steps:
                lines.append(f"  → {step}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    # =========================================================================
    # Factory Methods
    # =========================================================================
    
    @classmethod
    def create_empty(cls) -> 'GovernanceOutput':
        """
        Factory method to create an empty GovernanceOutput with defaults.
        Resolves instantiation errors when all parameters aren't known.
        """
        return cls(
            output_id=f"BASE-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:17]}",
            mode="audit_only",
            timestamp=datetime.utcnow().isoformat(),
            status=OutputStatus.SUCCESS,
            action_required=ActionRequired.NONE,
            original_response="",
            final_response="",
            response_was_modified=False,
            issues=[],
            evidence=EvidenceReport(
                strength="unknown",
                score=0.0,
                verified_items=[],
                failed_items=[],
                summary="Not evaluated"
            ),
            enhancements=None,
            multi_track=None,
            approval=None,
            processing_time_ms=0.0,
            base_version="2.0",
            user_message="",
            next_steps=[]
        )


# =============================================================================
# Governance Output Manager (with Learning Interface)
# =============================================================================

class GovernanceOutputManager:
    """
    Manager for GovernanceOutput creation with learning capabilities.
    
    Implements the standard BASE learning interface (5/5 methods).
    """
    
    def __init__(self):
        # Learning state
        self._outcome_history: List[Dict[str, Any]] = []
        self._learning_params: Dict[str, Any] = {
            'preferred_format': 'cursor',
            'verbosity_level': 'normal',
            'include_evidence': True,
            'include_next_steps': True
        }
        self._stats: Dict[str, int] = {
            'total_outputs': 0,
            'audit_only': 0,
            'audit_and_remediate': 0,
            'direct_assistance': 0,
            'blocked_outputs': 0,
            'enhanced_outputs': 0,
            'feedback_received': 0,
            'format_preferences_learned': 0
        }
    
    def create_output(self, mode: str = "audit_only") -> GovernanceOutput:
        """Create a new GovernanceOutput and track it."""
        output = GovernanceOutput.create_empty()
        output.mode = mode
        
        self._stats['total_outputs'] += 1
        self._stats[mode] = self._stats.get(mode, 0) + 1
        
        return output
    
    # =========================================================================
    # Learning Interface (5/5 methods)
    # =========================================================================
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict[str, Any]) -> None:
        """
        Record an output generation outcome for learning.
        
        Args:
            outcome: Dictionary containing:
                - output_id: ID of the output
                - mode: Mode used
                - was_useful: Whether the output was useful to user
                - format_preference: User's format preference if expressed
        """
        self._stats['total_outputs'] += 1
        
        mode = outcome.get('mode', 'unknown')
        self._stats[mode] = self._stats.get(mode, 0) + 1
        
        if outcome.get('status') == 'blocked':
            self._stats['blocked_outputs'] += 1
        if outcome.get('status') == 'enhanced':
            self._stats['enhanced_outputs'] += 1
        
        self._outcome_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            **outcome
        })
        
        # Keep bounded
        if len(self._outcome_history) > 1000:
            self._outcome_history = self._outcome_history[-500:]
    
    def learn_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Learn from user feedback to improve output formatting.
        
        Args:
            feedback: Dictionary containing:
                - format_preference: 'cursor', 'cli', 'json'
                - verbosity: 'minimal', 'normal', 'verbose'
                - include_evidence: bool
                - include_next_steps: bool
                - was_helpful: bool
        """
        self._stats['feedback_received'] += 1
        
        # Update format preferences
        if 'format_preference' in feedback:
            self._learning_params['preferred_format'] = feedback['format_preference']
            self._stats['format_preferences_learned'] += 1
        
        if 'verbosity' in feedback:
            self._learning_params['verbosity_level'] = feedback['verbosity']
        
        if 'include_evidence' in feedback:
            self._learning_params['include_evidence'] = feedback['include_evidence']
        
        if 'include_next_steps' in feedback:
            self._learning_params['include_next_steps'] = feedback['include_next_steps']
        
        logger.debug(f"[GovernanceOutputManager] Learned from feedback: {feedback}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return output generation statistics."""
        total = self._stats['total_outputs']
        return {
            **self._stats,
            'success_rate': (total - self._stats['blocked_outputs']) / max(1, total),
            'enhancement_rate': self._stats['enhanced_outputs'] / max(1, total),
            'learning_params': self._learning_params.copy(),
            'history_size': len(self._outcome_history)
        }
    
    def serialize_state(self) -> Dict[str, Any]:
        """Serialize state for persistence."""
        return {
            'outcome_history': self._outcome_history[-100:],
            'learning_params': self._learning_params.copy(),
            'stats': self._stats.copy(),
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict[str, Any]) -> None:
        """Restore state from serialized data."""
        self._outcome_history = state.get('outcome_history', [])
        self._learning_params.update(state.get('learning_params', {}))
        self._stats.update(state.get('stats', {}))
        logger.info(f"[GovernanceOutputManager] Restored state with {len(self._outcome_history)} history items")


# Singleton instance for easy access
_output_manager: Optional[GovernanceOutputManager] = None

def get_output_manager() -> GovernanceOutputManager:
    """Get the singleton output manager instance."""
    global _output_manager
    if _output_manager is None:
        _output_manager = GovernanceOutputManager()
    return _output_manager


# =============================================================================
# Output Builder
# =============================================================================

class GovernanceOutputBuilder:
    """Builder pattern for constructing governance outputs."""
    
    def __init__(self):
        self._output_id = f"BASE-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:17]}"
        self._mode = "audit_only"
        self._status = OutputStatus.SUCCESS
        self._action_required = ActionRequired.NONE
        self._original_response = ""
        self._final_response = ""
        self._issues: List[IssueReport] = []
        self._evidence: Optional[EvidenceReport] = None
        self._enhancements: Optional[EnhancementReport] = None
        self._multi_track: Optional[MultiTrackReport] = None
        self._approval: Optional[ApprovalReport] = None
        self._user_message = ""
        self._next_steps: List[str] = []
        self._start_time = datetime.utcnow()
    
    def set_mode(self, mode: str) -> 'GovernanceOutputBuilder':
        self._mode = mode
        return self
    
    def set_status(self, status: OutputStatus, action: ActionRequired = ActionRequired.NONE) -> 'GovernanceOutputBuilder':
        self._status = status
        self._action_required = action
        return self
    
    def set_response(self, original: str, final: str) -> 'GovernanceOutputBuilder':
        self._original_response = original
        self._final_response = final
        return self
    
    def add_issue(
        self,
        issue_type: str,
        severity: str,
        description: str,
        suggested_fix: Optional[str] = None,
        auto_fixable: bool = False
    ) -> 'GovernanceOutputBuilder':
        self._issues.append(IssueReport(
            issue_id=f"ISS-{len(self._issues)+1}",
            issue_type=issue_type,
            severity=severity,
            description=description,
            suggested_fix=suggested_fix,
            auto_fixable=auto_fixable
        ))
        return self
    
    def set_evidence(
        self,
        strength: str,
        score: float,
        verified: List[str] = None,
        failed: List[str] = None
    ) -> 'GovernanceOutputBuilder':
        self._evidence = EvidenceReport(
            strength=strength,
            score=score,
            verified_items=verified or [],
            failed_items=failed or [],
            summary=f"Evidence strength: {strength} ({score:.2f})"
        )
        return self
    
    def set_enhancements(
        self,
        total: int,
        types: List[str],
        before: float,
        after: float,
        changes: List[str]
    ) -> 'GovernanceOutputBuilder':
        self._enhancements = EnhancementReport(
            total_enhancements=total,
            enhancement_types=types,
            confidence_before=before,
            confidence_after=after,
            changes_summary=changes
        )
        return self
    
    def set_multi_track(
        self,
        enabled: bool,
        tracks: int = 0,
        summaries: List[Dict] = None,
        consensus: bool = False,
        selected: Optional[str] = None,
        agreement: float = 0.0
    ) -> 'GovernanceOutputBuilder':
        self._multi_track = MultiTrackReport(
            enabled=enabled,
            tracks_executed=tracks,
            track_summaries=summaries or [],
            consensus_used=consensus,
            selected_track=selected,
            agreement_score=agreement
        )
        return self
    
    def set_approval(
        self,
        required: bool,
        status: str = "not_required",
        approval_id: Optional[str] = None,
        action_items: List[str] = None
    ) -> 'GovernanceOutputBuilder':
        self._approval = ApprovalReport(
            required=required,
            status=status,
            approval_id=approval_id,
            action_items=action_items or []
        )
        return self
    
    def set_user_message(self, message: str) -> 'GovernanceOutputBuilder':
        self._user_message = message
        return self
    
    def add_next_step(self, step: str) -> 'GovernanceOutputBuilder':
        self._next_steps.append(step)
        return self
    
    def build(self) -> GovernanceOutput:
        """Build the final output."""
        processing_time = (datetime.utcnow() - self._start_time).total_seconds() * 1000
        
        # Default evidence if not set
        if self._evidence is None:
            self._evidence = EvidenceReport(
                strength="unknown",
                score=0.0,
                verified_items=[],
                failed_items=[],
                summary="Evidence not evaluated"
            )
        
        return GovernanceOutput(
            output_id=self._output_id,
            mode=self._mode,
            timestamp=self._start_time.isoformat(),
            status=self._status,
            action_required=self._action_required,
            original_response=self._original_response,
            final_response=self._final_response,
            response_was_modified=self._original_response != self._final_response,
            issues=self._issues,
            evidence=self._evidence,
            enhancements=self._enhancements,
            multi_track=self._multi_track,
            approval=self._approval,
            processing_time_ms=processing_time,
            user_message=self._user_message,
            next_steps=self._next_steps
        )

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


# =============================================================================
# Output Factory
# =============================================================================

def create_audit_only_output(
    response: str,
    issues: List[Dict],
    evidence_strength: str,
    evidence_score: float
) -> GovernanceOutput:
    """Factory for AUDIT_ONLY output."""
    builder = GovernanceOutputBuilder()
    builder.set_mode("audit_only")
    builder.set_response(response, response)
    builder.set_evidence(evidence_strength, evidence_score)
    builder.set_status(OutputStatus.SUCCESS)
    
    for issue in issues:
        builder.add_issue(
            issue_type=issue.get('type', 'unknown'),
            severity=issue.get('severity', 'low'),
            description=issue.get('description', ''),
            suggested_fix=issue.get('fix')
        )
    
    if issues:
        builder.set_user_message(f"Audit complete. {len(issues)} issues found for review.")
    else:
        builder.set_user_message("Audit complete. No significant issues found.")
    
    return builder.build()


def create_remediation_output(
    response: str,
    blocked: bool,
    issues: List[Dict],
    evidence_strength: str,
    evidence_score: float,
    approval_required: bool = False
) -> GovernanceOutput:
    """Factory for AUDIT_AND_REMEDIATE output."""
    builder = GovernanceOutputBuilder()
    builder.set_mode("audit_and_remediate")
    builder.set_response(response, response)
    builder.set_evidence(evidence_strength, evidence_score)
    
    if blocked:
        builder.set_status(OutputStatus.BLOCKED, ActionRequired.FIX)
        builder.set_user_message("Request blocked. Please fix the issues below.")
    elif approval_required:
        builder.set_status(OutputStatus.PENDING_APPROVAL, ActionRequired.APPROVE)
        builder.set_user_message("Approval required to proceed.")
    else:
        builder.set_status(OutputStatus.SUCCESS)
        builder.set_user_message("Request approved.")
    
    for issue in issues:
        builder.add_issue(
            issue_type=issue.get('type', 'unknown'),
            severity=issue.get('severity', 'low'),
            description=issue.get('description', ''),
            suggested_fix=issue.get('fix')
        )
        if issue.get('fix'):
            builder.add_next_step(issue.get('fix'))
    
    return builder.build()


def create_assistance_output(
    original_response: str,
    enhanced_response: str,
    enhancements: List[Dict],
    confidence_before: float,
    confidence_after: float
) -> GovernanceOutput:
    """Factory for DIRECT_ASSISTANCE output."""
    builder = GovernanceOutputBuilder()
    builder.set_mode("direct_assistance")
    builder.set_response(original_response, enhanced_response)
    builder.set_evidence("verified", confidence_after)
    builder.set_status(OutputStatus.ENHANCED)
    
    enhancement_types = list(set(e.get('type', 'unknown') for e in enhancements))
    changes = [e.get('description', '') for e in enhancements if e.get('description')]
    
    builder.set_enhancements(
        total=len(enhancements),
        types=enhancement_types,
        before=confidence_before,
        after=confidence_after,
        changes=changes
    )
    
    if enhancements:
        builder.set_user_message(f"Response enhanced with {len(enhancements)} improvements.")
    else:
        builder.set_user_message("Response delivered (no enhancements needed).")
    
    return builder.build()


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("GOVERNANCE OUTPUT TEST")
    print("=" * 80)
    
    # Test 1: AUDIT_ONLY output
    print("\n[1] Testing AUDIT_ONLY output...")
    output1 = create_audit_only_output(
        response="Here is my implementation...",
        issues=[
            {'type': 'weak_evidence', 'severity': 'medium', 'description': 'No test results provided'},
            {'type': 'overconfidence', 'severity': 'low', 'description': 'Claims 100% success'}
        ],
        evidence_strength="weak",
        evidence_score=0.3
    )
    print(f"    Status: {output1.status.value}")
    print(f"    Issues: {len(output1.issues)}")
    print(f"    User message: {output1.user_message}")
    
    # Test 2: AUDIT_AND_REMEDIATE blocked output
    print("\n[2] Testing AUDIT_AND_REMEDIATE (blocked)...")
    output2 = create_remediation_output(
        response="Task complete",
        blocked=True,
        issues=[
            {'type': 'missing_implementation', 'severity': 'critical', 'description': 'Class not found', 'fix': 'Implement the missing class'}
        ],
        evidence_strength="none",
        evidence_score=0.0
    )
    print(f"    Status: {output2.status.value}")
    print(f"    Action required: {output2.action_required.value}")
    print(f"    Next steps: {output2.next_steps}")
    
    # Test 3: DIRECT_ASSISTANCE output
    print("\n[3] Testing DIRECT_ASSISTANCE output...")
    output3 = create_assistance_output(
        original_response="This is 100% correct.",
        enhanced_response="This is very likely correct.",
        enhancements=[
            {'type': 'overconfidence_removal', 'description': 'Replaced "100%" with "very likely"'}
        ],
        confidence_before=0.4,
        confidence_after=0.7
    )
    print(f"    Status: {output3.status.value}")
    print(f"    Modified: {output3.response_was_modified}")
    print(f"    Enhancements: {output3.enhancements.total_enhancements}")
    print(f"    Confidence boost: {output3.enhancements.confidence_after - output3.enhancements.confidence_before:.2f}")
    
    # Test 4: Format outputs
    print("\n[4] Testing output formats...")
    print("\n    --- Cursor Format ---")
    print(output2.to_cursor_format()[:300] + "...")
    print("\n    --- CLI Format ---")
    print(output2.to_cli_format())
    
    # Test 5: JSON serialization
    print("\n[5] Testing JSON serialization...")
    json_output = output1.to_json()
    print(f"    JSON length: {len(json_output)} chars")
    print(f"    Parsed successfully: {json.loads(json_output)['status'] == 'success'}")
    
    print("\n" + "=" * 80)
    print("✓ GOVERNANCE OUTPUT TEST COMPLETE")
    print("=" * 80)

