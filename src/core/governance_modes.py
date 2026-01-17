"""
BAIS Cognitive Governance Engine - Governance Modes
Phase 1 & 2: Mode Controller and Evidence Classification

Three operational modes:
1. AUDIT_ONLY: Evaluate, score, continue (never block) - for compliance review
2. AUDIT_AND_REMEDIATE: Evaluate, block if critical, remediate with approval
3. DIRECT_ASSISTANCE: Real-time enhancement before user sees output

Platform-agnostic design - works with Cursor, VS Code, API, CLI, etc.

Patent: NOVEL-42 (Governance Mode Controller)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# =============================================================================
# Core Enums
# =============================================================================

class BAISMode(Enum):
    """BAIS operational modes."""
    AUDIT_ONLY = "audit_only"
    AUDIT_AND_REMEDIATE = "audit_and_remediate"
    DIRECT_ASSISTANCE = "direct_assistance"


class ApprovalMode(Enum):
    """How approval is obtained."""
    CURSOR_PROMPT = "cursor_prompt"      # Cursor IDE integration
    VSCODE_PROMPT = "vscode_prompt"      # VS Code integration
    API_CALLBACK = "api_callback"        # Webhook callback
    CLI_PROMPT = "cli_prompt"            # Command line prompt
    SLACK = "slack"                      # Slack integration
    AUTO = "auto"                        # Auto-approve (for testing)
    BOTH = "both"                        # Try IDE first, fall back to API


class EvidenceStrength(Enum):
    """Strength of evidence provided."""
    NONE = "none"           # No evidence at all
    WEAK = "weak"           # Description only (0.0-0.3)
    MEDIUM = "medium"       # File exists, imports work (0.3-0.6)
    STRONG = "strong"       # Instantiation succeeds (0.6-0.8)
    VERIFIED = "verified"   # Execution output confirmed (0.8-1.0)
    
    @property
    def score_range(self) -> tuple:
        ranges = {
            'none': (0.0, 0.0),
            'weak': (0.0, 0.3),
            'medium': (0.3, 0.6),
            'strong': (0.6, 0.8),
            'verified': (0.8, 1.0)
        }
        return ranges.get(self.value, (0.0, 0.0))


class IssueSeverity(Enum):
    """Severity of detected issues."""
    INFO = "info"           # Informational only
    LOW = "low"             # Minor issue
    MEDIUM = "medium"       # Moderate issue
    HIGH = "high"           # Significant issue
    CRITICAL = "critical"   # Blocking issue


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GovernanceConfig:
    """Configuration for BAIS governance."""
    
    # Mode settings
    mode: BAISMode = BAISMode.AUDIT_ONLY
    approval_mode: ApprovalMode = ApprovalMode.API_CALLBACK
    
    # Evidence settings
    require_execution_proof: bool = True
    minimum_evidence_strength: EvidenceStrength = EvidenceStrength.MEDIUM
    
    # Blocking behavior
    block_on_critical: bool = True
    block_on_weak_evidence: bool = False  # Only in AUDIT_AND_REMEDIATE
    continue_on_medium_evidence: bool = True
    
    # Remediation settings
    remediation_auto: bool = False  # Auto-execute approved remediations
    max_remediation_attempts: int = 5
    
    # Multi-track settings
    multi_track_enabled: bool = False
    multi_track_llms: List[str] = field(default_factory=lambda: ["grok"])
    multi_track_suggestion_enabled: bool = True
    auto_select_best_track: bool = True  # For DIRECT_ASSISTANCE
    
    # Learning settings
    learning_enabled: bool = True
    user_label_discount: float = 0.7  # Skepticism for user feedback
    
    # Approval settings
    approval_timeout_seconds: int = 300  # 5 minutes for IDE
    api_callback_timeout_seconds: int = 86400  # 24 hours for API
    api_callback_url: Optional[str] = None
    
    # Platform detection
    detect_platform: bool = True
    force_platform: Optional[str] = None  # Override detected platform


@dataclass
class Issue:
    """A detected issue."""
    issue_id: str
    issue_type: str
    severity: IssueSeverity
    description: str
    evidence: str
    remediation_suggestion: Optional[str] = None
    auto_fixable: bool = False


@dataclass
class EvidenceReport:
    """Report on evidence quality."""
    strength: EvidenceStrength
    score: float
    details: List[str]
    execution_output: Optional[str] = None
    verified_items: List[str] = field(default_factory=list)
    failed_items: List[str] = field(default_factory=list)


# =============================================================================
# Evidence Classifier
# =============================================================================

class EvidenceClassifier:
    """
    Classifies evidence strength based on what was actually verified.
    
    NOT: "They said it works"
    YES: "Execution proved it works"
    """
    
    def __init__(self, config: Optional[GovernanceConfig] = None):
        self.config = config or GovernanceConfig()
        
        # Patterns that indicate different evidence levels
        self._execution_patterns = [
            "executed", "returned", "output:", "result:", 
            "passed", "succeeded", "verified", "confirmed"
        ]
        self._instantiation_patterns = [
            "instantiated", "created instance", "initialized",
            "constructor called"
        ]
        self._import_patterns = [
            "imported", "import success", "module loaded", 
            "found in", "exists in"
        ]
        self._file_patterns = [
            "file exists", "file created", "file found",
            "path exists"
        ]
        self._description_patterns = [
            "implemented", "added", "created", "will be",
            "should work", "designed to", "intended to"
        ]
        self._outcomes = []

    def classify(self, evidence: List[str]) -> EvidenceReport:
        """Classify evidence strength from provided evidence list."""
        if not evidence:
            return EvidenceReport(
                strength=EvidenceStrength.NONE,
                score=0.0,
                details=["No evidence provided"]
            )
        
        scores = []
        details = []
        verified_items = []
        failed_items = []
        
        for e in evidence:
            e_lower = e.lower()
            score, detail = self._score_single_evidence(e_lower)
            scores.append(score)
            details.append(detail)
            
            if score >= 0.8:
                verified_items.append(e[:50])
            elif score < 0.3:
                failed_items.append(e[:50])
        
        # Calculate overall score (weighted toward highest evidence)
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        
        # Weight: 60% max, 40% average (reward strong evidence)
        final_score = max_score * 0.6 + avg_score * 0.4
        
        # Map to strength
        strength = self._score_to_strength(final_score)
        
        return EvidenceReport(
            strength=strength,
            score=final_score,
            details=details,
            verified_items=verified_items,
            failed_items=failed_items
        )
    
    def _score_single_evidence(self, evidence: str) -> tuple:
        """Score a single piece of evidence."""
        # Check for execution evidence (highest)
        if any(p in evidence for p in self._execution_patterns):
            return 0.9, "Execution evidence"
        
        # Check for instantiation evidence
        if any(p in evidence for p in self._instantiation_patterns):
            return 0.75, "Instantiation evidence"
        
        # Check for import evidence
        if any(p in evidence for p in self._import_patterns):
            return 0.5, "Import evidence"
        
        # Check for file evidence
        if any(p in evidence for p in self._file_patterns):
            return 0.4, "File existence evidence"
        
        # Check for description only
        if any(p in evidence for p in self._description_patterns):
            return 0.2, "Description only"
        
        # Unknown
        return 0.1, "Weak/Unknown evidence"
    
    def _score_to_strength(self, score: float) -> EvidenceStrength:
        """Map score to evidence strength."""
        if score >= 0.8:
            return EvidenceStrength.VERIFIED
        elif score >= 0.6:
            return EvidenceStrength.STRONG
        elif score >= 0.3:
            return EvidenceStrength.MEDIUM
        elif score > 0:
            return EvidenceStrength.WEAK
        else:
            return EvidenceStrength.NONE
    
    def meets_requirements(
        self, 
        report: EvidenceReport, 
        mode: BAISMode
    ) -> tuple:
        """
        Check if evidence meets requirements for the given mode.
        
        Returns (meets_requirements: bool, reason: str)
        """
        min_strength = self.config.minimum_evidence_strength
        
        # AUDIT_ONLY: Always continues, just reports
        if mode == BAISMode.AUDIT_ONLY:
            return True, f"AUDIT_ONLY mode - evidence: {report.strength.value}"
        
        # DIRECT_ASSISTANCE: Enhances regardless, doesn't block
        if mode == BAISMode.DIRECT_ASSISTANCE:
            return True, f"DIRECT_ASSISTANCE mode - will enhance"
        
        # AUDIT_AND_REMEDIATE: May block based on config
        if mode == BAISMode.AUDIT_AND_REMEDIATE:
            if report.strength == EvidenceStrength.NONE:
                return False, "No evidence provided - BLOCKED"
            
            if report.strength == EvidenceStrength.WEAK:
                if self.config.block_on_weak_evidence:
                    return False, "Weak evidence only - BLOCKED"
                else:
                    return True, "Weak evidence - WARN but continue"
            
            strength_order = [
                EvidenceStrength.NONE,
                EvidenceStrength.WEAK, 
                EvidenceStrength.MEDIUM,
                EvidenceStrength.STRONG,
                EvidenceStrength.VERIFIED
            ]
            
            if strength_order.index(report.strength) >= strength_order.index(min_strength):
                return True, f"Evidence ({report.strength.value}) meets minimum ({min_strength.value})"
            else:
                return False, f"Evidence ({report.strength.value}) below minimum ({min_strength.value})"
        
        return True, "Unknown mode - allowing"
    
    # =========================================================================
    # Learning Interface (5/5 methods)
    # =========================================================================
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict[str, Any]) -> None:
        """Record classification outcome for learning."""
        if not hasattr(self, '_outcome_history'):
            self._outcome_history: List[Dict] = []
        if not hasattr(self, '_stats'):
            self._stats = {'total': 0, 'verified': 0, 'weak': 0}
        
        self._stats['total'] += 1
        strength = outcome.get('strength', 'unknown')
        if strength in ['verified', 'strong']:
            self._stats['verified'] += 1
        elif strength in ['none', 'weak']:
            self._stats['weak'] += 1
        
        self._outcome_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            **outcome
        })
        if len(self._outcome_history) > 1000:
            self._outcome_history = self._outcome_history[-500:]
    
    def learn_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """Learn from feedback to improve classification."""
        if not hasattr(self, '_learning_params'):
            self._learning_params = {'score_adjustment': 0.0}
        
        was_correct = feedback.get('was_correct', True)
        if not was_correct:
            if feedback.get('evidence_was_stronger', False):
                self._learning_params['score_adjustment'] += 0.05
            elif feedback.get('evidence_was_weaker', False):
                self._learning_params['score_adjustment'] -= 0.05
        
        self._learning_params['score_adjustment'] = max(-0.2, min(0.2, 
            self._learning_params['score_adjustment']))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return classification statistics."""
        stats = getattr(self, '_stats', {'total': 0, 'verified': 0, 'weak': 0})
        return {
            **stats,
            'verified_rate': stats['verified'] / max(1, stats['total']),
            'weak_rate': stats['weak'] / max(1, stats['total']),
            'learning_params': getattr(self, '_learning_params', {}),
            'history_size': len(getattr(self, '_outcome_history', []))
        }
    
    def serialize_state(self) -> Dict[str, Any]:
        """Serialize state for persistence."""
        return {
            'outcome_history': getattr(self, '_outcome_history', [])[-100:],
            'learning_params': getattr(self, '_learning_params', {}),
            'stats': getattr(self, '_stats', {}),
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict[str, Any]) -> None:
        """Restore state from serialized data."""
        self._outcome_history = state.get('outcome_history', [])
        self._learning_params = state.get('learning_params', {'score_adjustment': 0.0})
        self._stats = state.get('stats', {'total': 0, 'verified': 0, 'weak': 0})


# =============================================================================
# Approval Interface (Platform-Agnostic)
# =============================================================================

class ApprovalInterface(ABC):
    """Abstract interface for approval providers."""
    
    @abstractmethod
    def request_approval(
        self,
        approval_id: str,
        action_items: List[str],
        remediation_prompts: List[str],
        timeout_seconds: int
    ) -> Dict[str, Any]:
        """
        Request approval for remediation.
        
        Returns:
            {
                'approved': bool,
                'response': 'APPROVE' | 'APPROVE_WITH_CHANGES' | 'REJECT' | 'TIMEOUT',
                'user_changes': Optional[str],
                'selected_track': Optional[str]
            }
        """
        pass
    
    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Name of the platform."""
        pass


class APICallbackApproval(ApprovalInterface):
    """Approval via API callback (webhook)."""
    
    def __init__(self, callback_url: str):
        self.callback_url = callback_url
        self._pending_approvals: Dict[str, Dict] = {}
    
    @property
    def platform_name(self) -> str:
        return "api_callback"
    
    def request_approval(
        self,
        approval_id: str,
        action_items: List[str],
        remediation_prompts: List[str],
        timeout_seconds: int
    ) -> Dict[str, Any]:
        """Request approval via API callback."""
        # Store pending approval
        self._pending_approvals[approval_id] = {
            'action_items': action_items,
            'remediation_prompts': remediation_prompts,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'pending'
        }
        
        # In real implementation, would POST to callback_url
        # For now, return pending status
        logger.info(f"[APICallback] Approval {approval_id} requested via {self.callback_url}")
        
        return {
            'approved': None,  # Pending
            'response': 'PENDING',
            'approval_id': approval_id,
            'callback_url': self.callback_url
        }
    
    def receive_callback(self, approval_id: str, decision: Dict) -> bool:
        """Receive callback with approval decision."""
        if approval_id in self._pending_approvals:
            self._pending_approvals[approval_id]['status'] = decision.get('response', 'UNKNOWN')
            self._pending_approvals[approval_id]['approved'] = decision.get('approved', False)
            return True
        return False


class CLIApproval(ApprovalInterface):
    """Approval via command line prompt."""
    
    @property
    def platform_name(self) -> str:
        return "cli"
    
    def request_approval(
        self,
        approval_id: str,
        action_items: List[str],
        remediation_prompts: List[str],
        timeout_seconds: int
    ) -> Dict[str, Any]:
        """Request approval via CLI."""
        print("\n" + "=" * 60)
        print("BAIS APPROVAL REQUIRED")
        print("=" * 60)
        print(f"\nApproval ID: {approval_id}")
        print("\nAction Items:")
        for i, item in enumerate(action_items, 1):
            print(f"  {i}. {item}")
        print("\nRemediation Prompts:")
        for i, prompt in enumerate(remediation_prompts, 1):
            print(f"  {i}. {prompt[:100]}...")
        print("\n" + "-" * 60)
        
        # In real implementation, would wait for input
        # For now, auto-approve for testing
        return {
            'approved': True,
            'response': 'APPROVE',
            'user_changes': None
        }


class AutoApproval(ApprovalInterface):
    """Auto-approval for testing."""
    
    @property
    def platform_name(self) -> str:
        return "auto"
    
    def request_approval(
        self,
        approval_id: str,
        action_items: List[str],
        remediation_prompts: List[str],
        timeout_seconds: int
    ) -> Dict[str, Any]:
        """Auto-approve everything (for testing)."""
        logger.warning(f"[AutoApproval] Auto-approving {approval_id}")
        return {
            'approved': True,
            'response': 'APPROVE',
            'user_changes': None
        }


# =============================================================================
# Mode Controller
# =============================================================================

class GovernanceModeController:
    """
    Central controller for BAIS governance modes.
    
    Handles:
    - Mode selection and configuration
    - Evidence evaluation
    - Approval flow
    - Mode-specific behavior
    """
    
    def __init__(self, config: Optional[GovernanceConfig] = None):
        self.config = config or GovernanceConfig()
        self.evidence_classifier = EvidenceClassifier(self.config)
        
        # Initialize approval provider based on config
        self._approval_provider = self._create_approval_provider()
        
        # Statistics
        self._stats = {
            'evaluations': 0,
            'blocked': 0,
            'approved': 0,
            'enhanced': 0,
            'by_mode': {m.value: 0 for m in BAISMode}
        }
    
    def _create_approval_provider(self) -> ApprovalInterface:
        """Create approval provider based on config."""
        mode = self.config.approval_mode
        
        if mode == ApprovalMode.API_CALLBACK:
            return APICallbackApproval(self.config.api_callback_url or "http://localhost:8000/bais/approve")
        elif mode == ApprovalMode.CLI_PROMPT:
            return CLIApproval()
        elif mode == ApprovalMode.AUTO:
            return AutoApproval()
        else:
            # Default to API callback
            return APICallbackApproval(self.config.api_callback_url or "http://localhost:8000/bais/approve")
    
    def set_mode(self, mode: BAISMode) -> None:
        """Set the governance mode."""
        self.config.mode = mode
        logger.info(f"[ModeController] Mode set to: {mode.value}")
    
    def evaluate(
        self,
        response: str,
        evidence: List[str],
        issues: List[Issue]
    ) -> Dict[str, Any]:
        """
        Evaluate a response based on current mode.
        
        Returns mode-appropriate result.
        """
        self._stats['evaluations'] += 1
        self._stats['by_mode'][self.config.mode.value] += 1
        
        # Classify evidence
        evidence_report = self.evidence_classifier.classify(evidence)
        
        # Check if evidence meets requirements
        meets_req, reason = self.evidence_classifier.meets_requirements(
            evidence_report, self.config.mode
        )
        
        # Determine critical issues
        critical_issues = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
        high_issues = [i for i in issues if i.severity == IssueSeverity.HIGH]
        
        # Mode-specific handling
        if self.config.mode == BAISMode.AUDIT_ONLY:
            return self._handle_audit_only(
                response, evidence_report, issues, meets_req, reason
            )
        
        elif self.config.mode == BAISMode.AUDIT_AND_REMEDIATE:
            return self._handle_audit_remediate(
                response, evidence_report, issues, meets_req, reason,
                critical_issues, high_issues
            )
        
        elif self.config.mode == BAISMode.DIRECT_ASSISTANCE:
            return self._handle_direct_assistance(
                response, evidence_report, issues, meets_req, reason
            )
        
        return {'error': 'Unknown mode'}
    
    def _handle_audit_only(
        self,
        response: str,
        evidence_report: EvidenceReport,
        issues: List[Issue],
        meets_req: bool,
        reason: str
    ) -> Dict[str, Any]:
        """AUDIT_ONLY: Report issues, never block."""
        action_items = []
        for issue in issues:
            if issue.remediation_suggestion:
                action_items.append(f"{issue.severity.value.upper()}: {issue.remediation_suggestion}")
        
        return {
            'mode': BAISMode.AUDIT_ONLY.value,
            'blocked': False,
            'continue_execution': True,
            'evidence_strength': evidence_report.strength.value,
            'evidence_score': evidence_report.score,
            'issues': [{'type': i.issue_type, 'severity': i.severity.value, 'desc': i.description} for i in issues],
            'action_items': action_items,
            'audit_notes': reason,
            'original_response': response
        }
    
    def _handle_audit_remediate(
        self,
        response: str,
        evidence_report: EvidenceReport,
        issues: List[Issue],
        meets_req: bool,
        reason: str,
        critical_issues: List[Issue],
        high_issues: List[Issue]
    ) -> Dict[str, Any]:
        """AUDIT_AND_REMEDIATE: May block, request approval."""
        
        # Determine if should block
        should_block = False
        block_reason = None
        
        if critical_issues and self.config.block_on_critical:
            should_block = True
            block_reason = f"Critical issues found: {len(critical_issues)}"
        elif not meets_req:
            should_block = True
            block_reason = reason
        
        if should_block:
            self._stats['blocked'] += 1
            
            # Generate remediation prompts
            remediation_prompts = []
            for issue in issues:
                if issue.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH]:
                    if issue.remediation_suggestion:
                        remediation_prompts.append(issue.remediation_suggestion)
            
            action_items = [i.description for i in critical_issues + high_issues]
            
            return {
                'mode': BAISMode.AUDIT_AND_REMEDIATE.value,
                'blocked': True,
                'continue_execution': False,
                'block_reason': block_reason,
                'evidence_strength': evidence_report.strength.value,
                'evidence_score': evidence_report.score,
                'issues': [{'type': i.issue_type, 'severity': i.severity.value, 'desc': i.description} for i in issues],
                'action_items': action_items,
                'remediation_prompts': remediation_prompts,
                'awaiting_approval': True,
                'original_response': response
            }
        else:
            self._stats['approved'] += 1
            return {
                'mode': BAISMode.AUDIT_AND_REMEDIATE.value,
                'blocked': False,
                'continue_execution': True,
                'evidence_strength': evidence_report.strength.value,
                'evidence_score': evidence_report.score,
                'issues': [{'type': i.issue_type, 'severity': i.severity.value, 'desc': i.description} for i in issues],
                'original_response': response
            }
    
    def _handle_direct_assistance(
        self,
        response: str,
        evidence_report: EvidenceReport,
        issues: List[Issue],
        meets_req: bool,
        reason: str
    ) -> Dict[str, Any]:
        """DIRECT_ASSISTANCE: Enhance in real-time."""
        self._stats['enhanced'] += 1
        
        # Determine enhancement level needed
        if evidence_report.strength in [EvidenceStrength.NONE, EvidenceStrength.WEAK]:
            enhancement_level = "major"
        elif evidence_report.strength == EvidenceStrength.MEDIUM:
            enhancement_level = "moderate"
        else:
            enhancement_level = "minor"
        
        changes_to_make = []
        for issue in issues:
            if issue.auto_fixable and issue.remediation_suggestion:
                changes_to_make.append(issue.remediation_suggestion)
        
        return {
            'mode': BAISMode.DIRECT_ASSISTANCE.value,
            'blocked': False,
            'continue_execution': True,
            'enhancement_level': enhancement_level,
            'evidence_strength': evidence_report.strength.value,
            'evidence_score': evidence_report.score,
            'changes_to_make': changes_to_make,
            'issues': [{'type': i.issue_type, 'severity': i.severity.value, 'desc': i.description} for i in issues],
            'original_response': response,
            'needs_enhancement': len(changes_to_make) > 0 or enhancement_level in ["major", "moderate"]
        }
    
    def request_approval(
        self,
        action_items: List[str],
        remediation_prompts: List[str]
    ) -> Dict[str, Any]:
        """Request approval using configured provider."""
        approval_id = f"BAIS-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        timeout = (
            self.config.approval_timeout_seconds 
            if self.config.approval_mode != ApprovalMode.API_CALLBACK 
            else self.config.api_callback_timeout_seconds
        )
        
        return self._approval_provider.request_approval(
            approval_id=approval_id,
            action_items=action_items,
            remediation_prompts=remediation_prompts,
            timeout_seconds=timeout
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get mode controller statistics."""
        return {
            **self._stats,
            'learning_params': getattr(self, '_learning_params', {}),
            'history_size': len(getattr(self, '_outcome_history', []))
        }
    
    # =========================================================================
    # Learning Interface (5/5 methods)
    # =========================================================================
    
    def record_outcome(self, outcome: Dict[str, Any]) -> None:
        """Record governance outcome for learning."""
        if not hasattr(self, '_outcome_history'):
            self._outcome_history: List[Dict] = []
        
        self._stats['total_evaluations'] = self._stats.get('total_evaluations', 0) + 1
        mode = outcome.get('mode', 'unknown')
        if mode == 'audit_only':
            pass  # Already tracked in _handle_audit_only
        elif mode == 'audit_and_remediate':
            pass  # Already tracked
        
        self._outcome_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            **outcome
        })
        if len(self._outcome_history) > 1000:
            self._outcome_history = self._outcome_history[-500:]
    
    def learn_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """Learn from feedback to improve mode selection."""
        if not hasattr(self, '_learning_params'):
            self._learning_params = {'block_threshold_adjustment': 0.0, 'enhancement_threshold': 0.5}
        
        was_correct = feedback.get('was_correct', True)
        if not was_correct:
            if feedback.get('should_have_blocked', False):
                self._learning_params['block_threshold_adjustment'] -= 0.05
            elif feedback.get('should_not_have_blocked', False):
                self._learning_params['block_threshold_adjustment'] += 0.05
        
        self._learning_params['block_threshold_adjustment'] = max(-0.3, min(0.3, 
            self._learning_params['block_threshold_adjustment']))
    
    def serialize_state(self) -> Dict[str, Any]:
        """Serialize state for persistence."""
        return {
            'outcome_history': getattr(self, '_outcome_history', [])[-100:],
            'learning_params': getattr(self, '_learning_params', {}),
            'stats': self._stats.copy(),
            'config': {
                'mode': self.config.mode.value,
                'approval_mode': self.config.approval_mode.value
            },
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict[str, Any]) -> None:
        """Restore state from serialized data."""
        self._outcome_history = state.get('outcome_history', [])
        self._learning_params = state.get('learning_params', {})
        self._stats.update(state.get('stats', {}))


# =============================================================================
# Factory
# =============================================================================

def create_mode_controller(
    mode: BAISMode = BAISMode.AUDIT_ONLY,
    **kwargs
) -> GovernanceModeController:
    """Factory function to create mode controller."""
    config = GovernanceConfig(mode=mode, **kwargs)
    return GovernanceModeController(config)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("GOVERNANCE MODES TEST")
    print("=" * 80)
    
    # Test AUDIT_ONLY mode
    print("\n[1] Testing AUDIT_ONLY mode...")
    controller = create_mode_controller(mode=BAISMode.AUDIT_ONLY)
    
    result = controller.evaluate(
        response="I implemented the class",
        evidence=["Class was implemented", "Should work correctly"],
        issues=[
            Issue(
                issue_id="1",
                issue_type="weak_evidence",
                severity=IssueSeverity.MEDIUM,
                description="Only descriptions provided",
                evidence="No execution proof",
                remediation_suggestion="Run tests to verify"
            )
        ]
    )
    
    print(f"    Mode: {result['mode']}")
    print(f"    Blocked: {result['blocked']}")
    print(f"    Continue: {result['continue_execution']}")
    print(f"    Evidence: {result['evidence_strength']} ({result['evidence_score']:.2f})")
    
    # Test AUDIT_AND_REMEDIATE mode
    print("\n[2] Testing AUDIT_AND_REMEDIATE mode with critical issue...")
    controller.set_mode(BAISMode.AUDIT_AND_REMEDIATE)
    
    result2 = controller.evaluate(
        response="I implemented the class",
        evidence=["Class was implemented"],
        issues=[
            Issue(
                issue_id="2",
                issue_type="missing_implementation",
                severity=IssueSeverity.CRITICAL,
                description="Required method missing",
                evidence="Execution failed",
                remediation_suggestion="Implement method_x()"
            )
        ]
    )
    
    print(f"    Mode: {result2['mode']}")
    print(f"    Blocked: {result2['blocked']}")
    print(f"    Block Reason: {result2.get('block_reason', 'N/A')}")
    print(f"    Awaiting Approval: {result2.get('awaiting_approval', False)}")
    
    # Test DIRECT_ASSISTANCE mode
    print("\n[3] Testing DIRECT_ASSISTANCE mode...")
    controller.set_mode(BAISMode.DIRECT_ASSISTANCE)
    
    result3 = controller.evaluate(
        response="Here is the code",
        evidence=["Code executed successfully", "All tests passed"],
        issues=[
            Issue(
                issue_id="3",
                issue_type="style",
                severity=IssueSeverity.LOW,
                description="Code style could be improved",
                evidence="Linting warnings",
                remediation_suggestion="Apply formatting",
                auto_fixable=True
            )
        ]
    )
    
    print(f"    Mode: {result3['mode']}")
    print(f"    Enhancement Level: {result3['enhancement_level']}")
    print(f"    Evidence: {result3['evidence_strength']} ({result3['evidence_score']:.2f})")
    print(f"    Changes to Make: {result3['changes_to_make']}")
    
    # Statistics
    print("\n[4] Statistics:")
    stats = controller.get_statistics()
    for key, value in stats.items():
        print(f"    {key}: {value}")
    
    print("\n" + "=" * 80)
    print("✓ GOVERNANCE MODES TEST COMPLETE")
    print("=" * 80)

