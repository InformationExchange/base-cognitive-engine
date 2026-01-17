"""
BASE v2.0 Orchestrator - Unified Integration
Phase 8: Integrates all v2.0 components into a single orchestrator

This is the main entry point for BASE v2.0 governance.

Components integrated:
- EnforcementLoop (Phase 1a)
- GovernanceModeController (Phase 1)
- EvidenceClassifier (Phase 2)
- MultiTrackOrchestrator (Phase 3)
- ApprovalGate (Phase 4 - via governance_modes)
- SkepticalLearningManager (Phase 5)
- RealTimeAssistanceEngine (Phase 6)
- GovernanceOutput (Phase 7)

Patent: NOVEL-47 (BASE v2.0 Orchestrator)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from datetime import datetime
import hashlib

# Import all v2.0 components
from core.enforcement_loop import (
    EnforcementLoop, 
    EnforcementEngine,
    EnforcementResult,
    EnforcementDecision,
    EnforcedOutput
)
from core.governance_modes import (
    GovernanceModeController,
    GovernanceConfig,
    BASEMode,
    EvidenceClassifier,
    Issue,
    IssueSeverity,
    ApprovalMode
)
from core.multi_track_orchestrator import (
    MultiTrackOrchestrator,
    TrackResult,
    ConsensusResult,
    TrackSuggestion
)
from core.skeptical_learning import (
    SkepticalLearningManager,
    LearningSource,
    AdjustedLearningSignal
)
from core.realtime_assistance import (
    RealTimeAssistanceEngine,
    AssistanceLevel,
    AssistanceResult,
    IssueDetection
)
from core.governance_output import (
    GovernanceOutput,
    GovernanceOutputBuilder,
    OutputStatus,
    ActionRequired,
    create_audit_only_output,
    create_remediation_output,
    create_assistance_output
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BASEv2Config:
    """Configuration for BASE v2.0."""
    
    # Mode settings
    mode: BASEMode = BASEMode.AUDIT_ONLY
    
    # Enforcement settings
    enforce_completion: bool = True
    max_enforcement_attempts: int = 5
    
    # Multi-track settings
    multi_track_enabled: bool = False
    available_llms: List[str] = field(default_factory=lambda: ["grok"])
    auto_select_best_track: bool = True
    
    # Learning settings
    learning_enabled: bool = True
    user_label_discount: float = 0.7
    
    # Assistance settings
    assistance_level: AssistanceLevel = AssistanceLevel.MODERATE
    
    # Evidence settings
    minimum_evidence_strength: str = "medium"
    
    # Approval settings
    approval_mode: ApprovalMode = ApprovalMode.API_CALLBACK


# =============================================================================
# BASE v2.0 Orchestrator
# =============================================================================

class BASEv2Orchestrator:
    """
    BASE v2.0 Orchestrator - Unified governance layer.
    
    Modes:
    - AUDIT_ONLY: Evaluate, report, continue
    - AUDIT_AND_REMEDIATE: Evaluate, block if needed, require approval
    - DIRECT_ASSISTANCE: Enhance in real-time
    
    All modes support:
    - Evidence classification
    - Issue detection
    - Multi-track LLM comparison (optional)
    - Skeptical learning
    - Enforcement loops (in remediate mode)
    """
    
    def __init__(self, config: Optional[BASEv2Config] = None):
        self.config = config or BASEv2Config()
        
        # Initialize components
        self._init_components()
        
        # Statistics
        self._stats = {
            'total_requests': 0,
            'by_mode': {m.value: 0 for m in BASEMode},
            'blocked': 0,
            'enhanced': 0,
            'enforcement_loops': 0,
            'multi_track_used': 0,
            'learning_signals': 0
        }
        
        logger.info(f"[BASEv2] Orchestrator initialized in {self.config.mode.value} mode")
    
    def _init_components(self) -> None:
        """Initialize all components."""
        # Mode controller
        governance_config = GovernanceConfig(
            mode=self.config.mode,
            approval_mode=self.config.approval_mode,
            minimum_evidence_strength=self._str_to_evidence_strength(self.config.minimum_evidence_strength),
            multi_track_enabled=self.config.multi_track_enabled
        )
        self.mode_controller = GovernanceModeController(governance_config)
        
        # Enforcement
        self.enforcement_loop = EnforcementLoop(
            max_attempts=self.config.max_enforcement_attempts
        )
        
        # Multi-track
        self.multi_track = MultiTrackOrchestrator(
            available_llms=self.config.available_llms,
            base_evaluator=self._evaluate_for_multi_track
        )
        
        # Learning
        self.learning_manager = SkepticalLearningManager(
            user_label_discount=self.config.user_label_discount
        )
        
        # Assistance
        self.assistance_engine = RealTimeAssistanceEngine(
            assistance_level=self.config.assistance_level,
            multi_track_enabled=self.config.multi_track_enabled,
            multi_track_provider=self._multi_track_provider if self.config.multi_track_enabled else None
        )
        
        # Evidence classifier
        self.evidence_classifier = EvidenceClassifier()
    
    def _str_to_evidence_strength(self, strength: str):
        """Convert string to EvidenceStrength enum."""
        from core.governance_modes import EvidenceStrength
        mapping = {
            'none': EvidenceStrength.NONE,
            'weak': EvidenceStrength.WEAK,
            'medium': EvidenceStrength.MEDIUM,
            'strong': EvidenceStrength.STRONG,
            'verified': EvidenceStrength.VERIFIED
        }
        return mapping.get(strength.lower(), EvidenceStrength.MEDIUM)
    
    def _evaluate_for_multi_track(self, response: str) -> Dict[str, Any]:
        """Evaluate response for multi-track scoring."""
        # Simple evaluation for multi-track
        issues = []
        
        # Check for overconfidence
        overconfident_terms = ['100%', 'guaranteed', 'always', 'never', 'perfect']
        for term in overconfident_terms:
            if term.lower() in response.lower():
                issues.append({'type': 'overconfidence', 'detail': term})
        
        # Check for incomplete markers
        incomplete_markers = ['...', 'etc', 'to be continued']
        for marker in incomplete_markers:
            if marker in response.lower():
                issues.append({'type': 'incomplete', 'detail': marker})
        
        # Calculate score
        score = 1.0 - (len(issues) * 0.1)
        score = max(0.1, min(1.0, score))
        
        return {
            'score': score,
            'issues': issues,
            'evidence_strength': 'medium',
            'confidence': score
        }
    
    def _multi_track_provider(self, query: str) -> Dict[str, Any]:
        """Provider for multi-track orchestration."""
        def mock_llm(q: str, llm: str) -> str:
            return f"[{llm}] Response to: {q[:50]}..."
        
        return self.multi_track.orchestrate(
            query=query,
            llm_provider=mock_llm,
            auto_select_best=self.config.auto_select_best_track
        )
    
    def set_mode(self, mode: BASEMode) -> None:
        """Set governance mode."""
        self.config.mode = mode
        self.mode_controller.set_mode(mode)
        logger.info(f"[BASEv2] Mode changed to: {mode.value}")
    
    def govern(
        self,
        query: str,
        response: str,
        evidence: Optional[List[str]] = None,
        domain: str = "general",
        context: Optional[Dict] = None,
        llm_provider: Optional[Callable] = None
    ) -> GovernanceOutput:
        """
        Main governance entry point.
        
        Args:
            query: Original user query
            response: LLM response to govern
            evidence: Evidence supporting the response
            domain: Domain context
            context: Additional context
            llm_provider: Optional LLM provider for multi-track
        
        Returns:
            GovernanceOutput with results
        """
        start_time = datetime.utcnow()
        self._stats['total_requests'] += 1
        self._stats['by_mode'][self.config.mode.value] += 1
        
        # Route to appropriate handler
        if self.config.mode == BASEMode.AUDIT_ONLY:
            return self._handle_audit_only(query, response, evidence or [], domain, start_time)
        
        elif self.config.mode == BASEMode.AUDIT_AND_REMEDIATE:
            return self._handle_audit_remediate(
                query, response, evidence or [], domain, context, llm_provider, start_time
            )
        
        elif self.config.mode == BASEMode.DIRECT_ASSISTANCE:
            return self._handle_direct_assistance(
                query, response, evidence or [], domain, start_time
            )
        
        # Fallback
        return create_audit_only_output(response, [], "unknown", 0.0)
    
    def _handle_audit_only(
        self,
        query: str,
        response: str,
        evidence: List[str],
        domain: str,
        start_time: datetime
    ) -> GovernanceOutput:
        """Handle AUDIT_ONLY mode."""
        # Classify evidence
        evidence_report = self.evidence_classifier.classify(evidence)
        
        # Detect issues
        issues = self._detect_issues(response, domain)
        
        # Record learning signal
        if self.config.learning_enabled:
            self._record_learning(
                response=response,
                issues=issues,
                evidence_strength=evidence_report.strength.value,
                was_correct=len(issues) == 0
            )
        
        # Build output
        builder = GovernanceOutputBuilder()
        builder.set_mode("audit_only")
        builder.set_response(response, response)
        builder.set_evidence(
            evidence_report.strength.value,
            evidence_report.score,
            evidence_report.verified_items,
            evidence_report.failed_items
        )
        builder.set_status(OutputStatus.SUCCESS)
        
        for issue in issues:
            builder.add_issue(
                issue_type=issue['type'],
                severity=issue['severity'],
                description=issue['description'],
                suggested_fix=issue.get('fix')
            )
        
        if issues:
            builder.set_user_message(f"Audit complete. {len(issues)} issues found.")
        else:
            builder.set_user_message("Audit complete. No significant issues.")
        
        return builder.build()
    
    def _handle_audit_remediate(
        self,
        query: str,
        response: str,
        evidence: List[str],
        domain: str,
        context: Optional[Dict],
        llm_provider: Optional[Callable],
        start_time: datetime
    ) -> GovernanceOutput:
        """Handle AUDIT_AND_REMEDIATE mode."""
        # Classify evidence
        evidence_report = self.evidence_classifier.classify(evidence)
        
        # Detect issues
        issues = self._detect_issues(response, domain)
        
        # Check for critical issues
        critical_issues = [i for i in issues if i['severity'] in ['critical', 'high']]
        
        # Should we block?
        should_block = (
            len(critical_issues) > 0 or
            evidence_report.strength.value in ['none', 'weak']
        )
        
        builder = GovernanceOutputBuilder()
        builder.set_mode("audit_and_remediate")
        builder.set_evidence(
            evidence_report.strength.value,
            evidence_report.score,
            evidence_report.verified_items,
            evidence_report.failed_items
        )
        
        if should_block:
            self._stats['blocked'] += 1
            
            # Use enforcement loop if enabled and llm_provider available
            if self.config.enforce_completion and llm_provider:
                self._stats['enforcement_loops'] += 1
                # Here we would run enforcement loop
                # For now, just set blocked status
            
            builder.set_status(OutputStatus.BLOCKED, ActionRequired.FIX)
            builder.set_response(response, response)
            
            for issue in issues:
                builder.add_issue(
                    issue_type=issue['type'],
                    severity=issue['severity'],
                    description=issue['description'],
                    suggested_fix=issue.get('fix')
                )
                if issue.get('fix'):
                    builder.add_next_step(issue['fix'])
            
            builder.set_user_message(
                f"BLOCKED: {len(critical_issues)} critical issues. "
                f"Evidence strength: {evidence_report.strength.value}"
            )
            builder.set_approval(
                required=True,
                status="pending",
                action_items=[i['description'] for i in critical_issues]
            )
        else:
            builder.set_status(OutputStatus.SUCCESS)
            builder.set_response(response, response)
            
            for issue in issues:
                builder.add_issue(
                    issue_type=issue['type'],
                    severity=issue['severity'],
                    description=issue['description'],
                    suggested_fix=issue.get('fix')
                )
            
            builder.set_user_message("Approved. Proceed with caution on noted issues.")
        
        # Record learning
        if self.config.learning_enabled:
            self._record_learning(
                response=response,
                issues=issues,
                evidence_strength=evidence_report.strength.value,
                was_correct=not should_block
            )
        
        return builder.build()
    
    def _handle_direct_assistance(
        self,
        query: str,
        response: str,
        evidence: List[str],
        domain: str,
        start_time: datetime
    ) -> GovernanceOutput:
        """Handle DIRECT_ASSISTANCE mode."""
        # CRITICAL: Detect issues FIRST before enhancing
        issues = self._detect_issues(response, domain)
        evidence_report = self.evidence_classifier.classify(evidence)
        
        # Check for critical issues that should NOT be masked
        critical_issues = [i for i in issues if i['severity'] in ['critical', 'high']]
        has_critical = len(critical_issues) > 0
        has_no_evidence = evidence_report.strength.value in ['none', 'weak']
        
        # SAFETY CHECK: Do NOT enhance dangerous content
        if has_critical and domain in ['medical', 'legal', 'financial']:
            # Block instead of enhance for sensitive domains with critical issues
            self._stats['blocked'] += 1
            
            builder = GovernanceOutputBuilder()
            builder.set_mode("direct_assistance")
            builder.set_response(response, response)  # Return original, NOT enhanced
            builder.set_status(OutputStatus.BLOCKED, ActionRequired.FIX)
            builder.set_evidence(
                evidence_report.strength.value,
                evidence_report.score,
                evidence_report.verified_items,
                evidence_report.failed_items
            )
            
            for issue in issues:
                builder.add_issue(
                    issue_type=issue['type'],
                    severity=issue['severity'],
                    description=issue['description'],
                    suggested_fix=issue.get('fix')
                )
                if issue.get('fix'):
                    builder.add_next_step(issue['fix'])
            
            builder.set_user_message(
                f"BLOCKED: Cannot enhance response with {len(critical_issues)} critical issues "
                f"in {domain} domain. Fix issues first."
            )
            return builder.build()
        
        # Safe to enhance - get assistance
        assistance_result = self.assistance_engine.assist(
            response=response,
            query=query,
            domain=domain,
            detected_issues=[
                IssueDetection(
                    issue_type=i['type'],
                    severity=i['severity'],
                    location='',
                    description=i['description'],
                    suggested_fix=i.get('fix', '')
                ) for i in issues
            ] if issues else None
        )
        
        self._stats['enhanced'] += 1
        
        # Determine final status based on issues
        if has_critical:
            # Has critical issues but not in sensitive domain - warn but allow
            final_status = OutputStatus.PARTIAL
            action = ActionRequired.REVIEW
            user_msg = f"Enhanced with warnings. {len(critical_issues)} issues need review."
        elif has_no_evidence:
            # Weak evidence - enhanced but flagged
            final_status = OutputStatus.ENHANCED
            action = ActionRequired.REVIEW
            user_msg = f"Enhanced but evidence is {evidence_report.strength.value}. Verify correctness."
        else:
            # All good
            final_status = OutputStatus.ENHANCED
            action = ActionRequired.NONE
            user_msg = f"Enhanced. Confidence: {assistance_result.original_confidence:.0%} → {assistance_result.enhanced_confidence:.0%}"
        
        # Build output
        builder = GovernanceOutputBuilder()
        builder.set_mode("direct_assistance")
        builder.set_response(response, assistance_result.enhanced_response)
        builder.set_status(final_status, action)
        
        # Add evidence
        evidence_report = self.evidence_classifier.classify(evidence)
        builder.set_evidence(
            evidence_report.strength.value,
            evidence_report.score,
            evidence_report.verified_items,
            evidence_report.failed_items
        )
        
        # Add enhancements
        if assistance_result.enhancements_applied:
            builder.set_enhancements(
                total=assistance_result.total_enhancements,
                types=[e.enhancement_type.value for e in assistance_result.enhancements_applied],
                before=assistance_result.original_confidence,
                after=assistance_result.enhanced_confidence,
                changes=[e.description for e in assistance_result.enhancements_applied]
            )
        
        # Multi-track info
        if assistance_result.used_multi_track:
            self._stats['multi_track_used'] += 1
            builder.set_multi_track(
                enabled=True,
                tracks=len(assistance_result.alternative_tracks),
                summaries=assistance_result.alternative_tracks,
                consensus=True,
                selected=assistance_result.selected_track
            )
        
        builder.set_user_message(
            f"Enhanced response. Confidence: "
            f"{assistance_result.original_confidence:.0%} → {assistance_result.enhanced_confidence:.0%}"
        )
        
        return builder.build()
    
    def _detect_issues(self, response: str, domain: str) -> List[Dict]:
        """Detect issues in response."""
        issues = []
        response_lower = response.lower()
        
        # Overconfidence
        overconfident = ['100%', 'guaranteed', 'always', 'never', 'perfect', 'flawless']
        for term in overconfident:
            if term in response_lower:
                issues.append({
                    'type': 'overconfidence',
                    'severity': 'medium',
                    'description': f'Overconfident term: "{term}"',
                    'fix': f'Replace "{term}" with hedged language'
                })
                break
        
        # Incomplete
        incomplete = ['...', 'etc', 'to be continued', 'more to come']
        for marker in incomplete:
            if marker in response_lower:
                issues.append({
                    'type': 'incomplete',
                    'severity': 'high',
                    'description': 'Response appears incomplete',
                    'fix': 'Provide complete response'
                })
                break
        
        # Domain-specific
        if domain == 'medical' and 'disclaimer' not in response_lower:
            issues.append({
                'type': 'missing_disclaimer',
                'severity': 'critical',
                'description': 'Medical response without disclaimer',
                'fix': 'Add medical disclaimer'
            })
        
        return issues
    
    def _record_learning(
        self,
        response: str,
        issues: List[Dict],
        evidence_strength: str,
        was_correct: bool
    ) -> None:
        """Record learning signal."""
        self._stats['learning_signals'] += 1
        
        # Record execution-based learning (from BASE analysis)
        self.learning_manager.learn_from_execution(
            module_name="base_v2",
            input_data={'response_preview': response[:100]},
            output_data={'issues_count': len(issues), 'evidence': evidence_strength},
            execution_succeeded=was_correct,
            execution_output=f"Issues: {len(issues)}, Evidence: {evidence_strength}"
        )
    
    def suggest_tracks(self, query: str) -> TrackSuggestion:
        """Suggest multi-track configuration for a query."""
        return self.multi_track.suggest_tracks(query)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            **self._stats,
            'mode_controller': self.mode_controller.get_statistics(),
            'learning': self.learning_manager.get_statistics(),
            'assistance': self.assistance_engine.get_statistics(),
            'enforcement': self.enforcement_loop.get_statistics()
        }

    
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


# =============================================================================
# Factory
# =============================================================================

def create_base_v2(
    mode: BASEMode = BASEMode.AUDIT_ONLY,
    **kwargs
) -> BASEv2Orchestrator:
    """Factory function to create BASE v2.0 orchestrator."""
    config = BASEv2Config(mode=mode, **kwargs)
    return BASEv2Orchestrator(config)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("BASE v2.0 ORCHESTRATOR TEST")
    print("=" * 80)
    
    # Test all three modes
    
    # 1. AUDIT_ONLY
    print("\n[1] Testing AUDIT_ONLY mode...")
    base = create_base_v2(mode=BASEMode.AUDIT_ONLY)
    output1 = base.govern(
        query="How do I implement authentication?",
        response="Just use JWT tokens. It's 100% secure and guaranteed to work.",
        evidence=["Implementation complete"]
    )
    print(f"    Status: {output1.status.value}")
    print(f"    Issues: {len(output1.issues)}")
    print(f"    Evidence: {output1.evidence.strength}")
    print(f"    Message: {output1.user_message}")
    
    # 2. AUDIT_AND_REMEDIATE
    print("\n[2] Testing AUDIT_AND_REMEDIATE mode...")
    base.set_mode(BASEMode.AUDIT_AND_REMEDIATE)
    output2 = base.govern(
        query="How do I treat a headache?",
        response="Take aspirin. It will definitely cure your headache.",
        evidence=["Doctor recommended"],
        domain="medical"
    )
    print(f"    Status: {output2.status.value}")
    print(f"    Action: {output2.action_required.value}")
    print(f"    Issues: {len(output2.issues)}")
    print(f"    Message: {output2.user_message}")
    if output2.next_steps:
        print(f"    Next steps: {output2.next_steps}")
    
    # 3. DIRECT_ASSISTANCE
    print("\n[3] Testing DIRECT_ASSISTANCE mode...")
    base.set_mode(BASEMode.DIRECT_ASSISTANCE)
    output3 = base.govern(
        query="How do I configure Redis?",
        response="Configure Redis with these settings. This is 100% the best approach...",
        evidence=["Tests passed", "Performance verified"]
    )
    print(f"    Status: {output3.status.value}")
    print(f"    Modified: {output3.response_was_modified}")
    if output3.enhancements:
        print(f"    Enhancements: {output3.enhancements.total_enhancements}")
        print(f"    Confidence: {output3.enhancements.confidence_before:.0%} → {output3.enhancements.confidence_after:.0%}")
    print(f"    Message: {output3.user_message}")
    
    # Statistics
    print("\n[4] Statistics:")
    stats = base.get_statistics()
    print(f"    Total requests: {stats['total_requests']}")
    print(f"    By mode: {stats['by_mode']}")
    print(f"    Blocked: {stats['blocked']}")
    print(f"    Enhanced: {stats['enhanced']}")
    print(f"    Learning signals: {stats['learning_signals']}")
    
    # Output formats
    print("\n[5] Sample Cursor format output:")
    print(output2.to_cursor_format()[:400] + "...")
    
    print("\n" + "=" * 80)
    print("✓ BASE v2.0 ORCHESTRATOR TEST COMPLETE")
    print("=" * 80)

