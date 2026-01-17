"""
BAIS Governance Rules
Codified lessons learned from self-improvement cycles

These rules are used by BAIS to govern its own development and testing,
ensuring the same mistakes are not repeated.

ALL 10 RULES IMPLEMENTED:
- Rule 1: Integration > Existence
- Rule 2: Define Success Upfront
- Rule 3: Question Uniformity
- Rule 4: Match Methodology to Claim Type
- Rule 5: Trace Data Flow
- Rule 6: Check for Empty Structures
- Rule 7: Push Past Incremental
- Rule 8: Test Methodology Must Match Claim Type
- Rule 9: Suspicious Uniformity = Wrong Test
- Rule 10: Learning Direction Matters
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Set
import re


class ClaimType(Enum):
    """Types of claims requiring different test methodologies."""
    CONTENT = "content"           # Content detection - A/B testing
    ALGORITHMIC = "algorithmic"   # Mathematical - Unit tests
    BEHAVIORAL = "behavioral"     # Scenarios - Scenario tests
    API = "api"                   # Endpoints - Functional tests
    INTEGRATION = "integration"   # Component connectivity


class TestMethodology(Enum):
    """Appropriate test methodology for each claim type."""
    AB_TESTING = "ab_testing"
    UNIT_TESTING = "unit_testing"
    SCENARIO_TESTING = "scenario_testing"
    FUNCTIONAL_TESTING = "functional_testing"
    INTEGRATION_TESTING = "integration_testing"


class RuleSeverity(Enum):
    """Severity levels for governance violations."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class GovernanceViolation:
    """A detected governance rule violation."""
    rule_id: str
    rule_name: str
    description: str
    evidence: str
    recommendation: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL


@dataclass
class DataFlowStep:
    """A step in the data flow trace."""
    name: str
    receives_input: bool
    produces_output: bool
    connected_to_next: bool


class BAISGovernanceRules:
    """
    Codified governance rules for BAIS self-improvement.
    
    These rules encode lessons learned to prevent repeated mistakes.
    ALL 10 RULES IMPLEMENTED.
    """
    
    # Rule mappings
    CLAIM_TYPE_METHODOLOGY = {
        ClaimType.CONTENT: TestMethodology.AB_TESTING,
        ClaimType.ALGORITHMIC: TestMethodology.UNIT_TESTING,
        ClaimType.BEHAVIORAL: TestMethodology.SCENARIO_TESTING,
        ClaimType.API: TestMethodology.FUNCTIONAL_TESTING,
        ClaimType.INTEGRATION: TestMethodology.INTEGRATION_TESTING,
    }
    
    # Claim ID patterns to claim types
    CLAIM_PATTERNS = {
        r'PPA2-C\d': ClaimType.ALGORITHMIC,       # PPA2 Components
        r'PPA2-Inv2[678]': ClaimType.ALGORITHMIC, # Core Gate, Adaptive Controller, Cognitive Window
        r'PPA3-Behav': ClaimType.BEHAVIORAL,      # Behavioral claims
        r'UTIL': ClaimType.API,                   # Utility/API claims
        r'NOVEL-3': ClaimType.INTEGRATION,        # Audit integration
        r'NOVEL-9': ClaimType.CONTENT,            # Query analyzer
        r'PPA1-Content': ClaimType.CONTENT,       # Content detection
    }
    
    # Data flow expected steps
    EXPECTED_DATA_FLOW = [
        'input', 'processing', 'detection', 'learning', 'adaptation', 'output'
    ]
    
    def __init__(self):
        self.violations: List[GovernanceViolation] = []
    
    @property
    def rules(self) -> List[Dict]:
        """Return list of all governance rules with descriptions."""
        return [
            {"id": "RULE-1", "name": "Integration > Existence", "desc": "Components must be connected, not just exist"},
            {"id": "RULE-2", "name": "Define Success Upfront", "desc": "Clear success criteria before testing"},
            {"id": "RULE-3", "name": "Question Uniformity", "desc": "100% pass rates indicate test issues"},
            {"id": "RULE-4", "name": "Match Methodology", "desc": "Test type must match claim type"},
            {"id": "RULE-5", "name": "Trace Data Flow", "desc": "Verify data actually flows through components"},
            {"id": "RULE-6", "name": "Check Empty Structures", "desc": "Detect placeholder implementations"},
            {"id": "RULE-7", "name": "Push Past Incremental", "desc": "Real improvements, not just hedging"},
            {"id": "RULE-8", "name": "Test Match Claim", "desc": "Methodology must validate the claim"},
            {"id": "RULE-9", "name": "Suspicious Uniformity", "desc": "Uniform results indicate wrong test"},
            {"id": "RULE-10", "name": "Learning Direction", "desc": "Verify learning improves metrics"},
        ]
    
    # =========================================================================
    # RULE 1: Integration > Existence
    # =========================================================================
    def check_rule_1_integration(self, 
                                  component_name: str,
                                  exists: bool,
                                  is_called: bool,
                                  data_flows: bool) -> Optional[GovernanceViolation]:
        """
        RULE 1: Don't just verify components exist - verify they're connected.
        
        A component that exists but is never called is NOT implemented.
        """
        if exists and not is_called:
            return GovernanceViolation(
                rule_id="RULE-1",
                rule_name="Integration > Existence",
                description=f"Component '{component_name}' exists but is not called in production flow",
                evidence=f"exists={exists}, is_called={is_called}, data_flows={data_flows}",
                recommendation="Verify component is actually invoked during execution",
                severity="HIGH"
            )
        if exists and is_called and not data_flows:
            return GovernanceViolation(
                rule_id="RULE-1",
                rule_name="Integration > Existence",
                description=f"Component '{component_name}' is called but data doesn't flow through",
                evidence=f"exists={exists}, is_called={is_called}, data_flows={data_flows}",
                recommendation="Check for empty data structures (e.g., calibration_scores = [])",
                severity="CRITICAL"
            )
        return None
    
    # =========================================================================
    # RULE 2: Define Success Upfront
    # =========================================================================
    def check_rule_2_success_criteria(self,
                                       current_value: float,
                                       target_value: float,
                                       improvement_from: float) -> Optional[GovernanceViolation]:
        """
        RULE 2: Define success as meeting target, not just "any improvement".
        
        13.1% → 14.6% is NOT success if target is 70%.
        """
        if current_value < target_value and current_value > improvement_from:
            gap = target_value - current_value
            return GovernanceViolation(
                rule_id="RULE-2",
                rule_name="Define Success Upfront",
                description=f"Current {current_value}% is improved but still {gap:.1f}% below target {target_value}%",
                evidence=f"improved_from={improvement_from}%, current={current_value}%, target={target_value}%",
                recommendation=f"Do not declare success. Need {gap:.1f}% more improvement.",
                severity="MEDIUM"
            )
        return None
    
    # =========================================================================
    # RULE 3: Question Uniformity
    # =========================================================================
    def check_rule_3_question_uniformity(self,
                                          test_results: List[Dict[str, Any]],
                                          uniformity_threshold: float = 0.6) -> Optional[GovernanceViolation]:
        """
        RULE 3: If many test results are identical, question the test.
        
        62 claims at 50% = "Wrong test methodology", not "All tested"
        """
        if not test_results:
            return None
        
        # Count identical scores
        score_counts: Dict[float, int] = {}
        for result in test_results:
            score = result.get('score', 0)
            score_counts[score] = score_counts.get(score, 0) + 1
        
        # Find most common score
        if not score_counts:
            return None
            
        most_common_score = max(score_counts, key=score_counts.get)
        most_common_count = score_counts[most_common_score]
        uniformity_ratio = most_common_count / len(test_results)
        
        if uniformity_ratio >= uniformity_threshold:
            return GovernanceViolation(
                rule_id="RULE-3",
                rule_name="Question Uniformity",
                description=f"{most_common_count}/{len(test_results)} results ({uniformity_ratio*100:.0f}%) have identical score {most_common_score}",
                evidence=f"score_distribution={score_counts}",
                recommendation="Question: Is the test actually exercising the claims? Consider different methodology.",
                severity="HIGH"
            )
        return None
    
    # =========================================================================
    # RULE 4: Match Methodology to Claim Type
    # =========================================================================
    def check_rule_4_methodology_match(self,
                                        claim_type: ClaimType,
                                        methodology_used: TestMethodology) -> Optional[GovernanceViolation]:
        """
        RULE 4: Match methodology to claim type.
        
        Content Detection Claims → A/B Testing
        Mathematical Claims → Unit Tests with Formula Verification
        Behavioral Claims → Scenario-Based Tests
        API Claims → Functional Endpoint Tests
        """
        expected = self.CLAIM_TYPE_METHODOLOGY.get(claim_type)
        
        if methodology_used != expected:
            return GovernanceViolation(
                rule_id="RULE-4",
                rule_name="Match Methodology to Claim Type",
                description=f"{claim_type.value} claim tested with {methodology_used.value} instead of {expected.value}",
                evidence=f"claim_type={claim_type.value}, used={methodology_used.value}, expected={expected.value}",
                recommendation=f"Use {expected.value} methodology for {claim_type.value} claims",
                severity="HIGH"
            )
        return None
    
    # =========================================================================
    # RULE 5: Trace Data Flow
    # =========================================================================
    def check_rule_5_data_flow(self,
                                flow_steps: List[DataFlowStep]) -> Optional[GovernanceViolation]:
        """
        RULE 5: Trace complete data flow from input to adaptation.
        
        Follow: Input → Processing → Detection → Learning → Adaptation → Output
        If any link is broken → NOT COMPLETE
        """
        if not flow_steps:
            return GovernanceViolation(
                rule_id="RULE-5",
                rule_name="Trace Data Flow",
                description="No data flow steps provided",
                evidence="flow_steps=[]",
                recommendation="Define complete data flow: Input → Processing → Detection → Learning → Adaptation → Output",
                severity="CRITICAL"
            )
        
        broken_links = []
        for i, step in enumerate(flow_steps):
            if not step.receives_input and i > 0:
                broken_links.append(f"{step.name} doesn't receive input from previous step")
            if not step.produces_output and i < len(flow_steps) - 1:
                broken_links.append(f"{step.name} doesn't produce output for next step")
            if not step.connected_to_next and i < len(flow_steps) - 1:
                broken_links.append(f"{step.name} not connected to next step")
        
        if broken_links:
            return GovernanceViolation(
                rule_id="RULE-5",
                rule_name="Trace Data Flow",
                description=f"Data flow has {len(broken_links)} broken link(s)",
                evidence="; ".join(broken_links[:3]),  # Show first 3
                recommendation="Fix broken links in data flow chain",
                severity="CRITICAL"
            )
        return None
    
    # =========================================================================
    # RULE 6: Check for Empty Structures
    # =========================================================================
    def check_rule_6_empty_structures(self,
                                       structures: Dict[str, Any]) -> Optional[GovernanceViolation]:
        """
        RULE 6: Check for empty data structures that indicate incomplete integration.
        
        calibration_scores = []  → RED FLAG
        audit_records = 0        → RED FLAG
        thresholds = {}         → RED FLAG
        """
        empty_structures = []
        
        for name, value in structures.items():
            if value is None:
                empty_structures.append(f"{name}=None")
            elif isinstance(value, (list, dict, set)) and len(value) == 0:
                empty_structures.append(f"{name}=empty")
            elif isinstance(value, (int, float)) and value == 0:
                # Only flag if it's a count that should be non-zero
                if any(indicator in name.lower() for indicator in ['count', 'records', 'points', 'entries']):
                    empty_structures.append(f"{name}=0")
        
        if empty_structures:
            return GovernanceViolation(
                rule_id="RULE-6",
                rule_name="Check for Empty Structures",
                description=f"Found {len(empty_structures)} empty/zero data structure(s)",
                evidence=", ".join(empty_structures[:5]),  # Show first 5
                recommendation="Empty structures indicate incomplete integration - verify data is flowing",
                severity="HIGH"
            )
        return None
    
    # =========================================================================
    # RULE 7: Push Past Incremental
    # =========================================================================
    def check_rule_7_push_past_incremental(self,
                                            current: float,
                                            previous: float,
                                            target: float,
                                            minimum_meaningful_improvement: float = 5.0) -> Optional[GovernanceViolation]:
        """
        RULE 7: Don't celebrate incremental improvements far from target.
        
        DON'T: 13.1% → 14.6% = "Good improvement!"
        DO:    Target is 70%. Current is 14.6%. Need 55.4% more.
        """
        improvement = current - previous
        gap_to_target = target - current
        
        # If we improved but are still far from target
        if 0 < improvement < minimum_meaningful_improvement and gap_to_target > 20:
            return GovernanceViolation(
                rule_id="RULE-7",
                rule_name="Push Past Incremental",
                description=f"Improvement of {improvement:.1f}% is incremental. Still {gap_to_target:.1f}% from target.",
                evidence=f"previous={previous}%, current={current}%, target={target}%, improvement={improvement:.1f}%",
                recommendation=f"Don't celebrate incremental. Push for substantial improvement toward {target}%",
                severity="MEDIUM"
            )
        return None
    
    # =========================================================================
    # RULE 8: Test Methodology Must Match Claim Type (Specific Version)
    # =========================================================================
    def check_rule_8_methodology(self,
                                  claim_id: str,
                                  methodology_used: TestMethodology) -> Optional[GovernanceViolation]:
        """
        RULE 8: Test methodology must match claim type.
        
        - Content claims → A/B testing
        - Algorithmic claims → Unit tests with formula verification
        - Behavioral claims → Scenario-based tests
        - API claims → Functional endpoint tests
        """
        claim_type = self._get_claim_type(claim_id)
        correct_methodology = self.CLAIM_TYPE_METHODOLOGY.get(claim_type)
        
        if methodology_used != correct_methodology:
            return GovernanceViolation(
                rule_id="RULE-8",
                rule_name="Test Methodology Must Match Claim Type",
                description=f"Claim '{claim_id}' is {claim_type.value} but tested with {methodology_used.value}",
                evidence=f"claim_type={claim_type.value}, used={methodology_used.value}, correct={correct_methodology.value}",
                recommendation=f"Use {correct_methodology.value} for {claim_type.value} claims",
                severity="HIGH"
            )
        return None
    
    def _get_claim_type(self, claim_id: str) -> ClaimType:
        """Determine claim type from claim ID."""
        for pattern, claim_type in self.CLAIM_PATTERNS.items():
            if re.match(pattern, claim_id):
                return claim_type
        return ClaimType.CONTENT  # Default
    
    # =========================================================================
    # RULE 9: Suspicious Uniformity = Wrong Test
    # =========================================================================
    def check_rule_9_uniformity(self,
                                 scores: List[float],
                                 threshold: float = 0.8) -> Optional[GovernanceViolation]:
        """
        RULE 9: If many claims have identical scores, test methodology is wrong.
        
        62 claims at exactly 50% = test isn't exercising claims.
        This is a signal to change methodology, not accept results.
        """
        if not scores:
            return None
        
        # Count score frequencies
        score_counts = {}
        for s in scores:
            score_counts[s] = score_counts.get(s, 0) + 1
        
        # Find most common score
        most_common_score = max(score_counts, key=score_counts.get)
        most_common_count = score_counts[most_common_score]
        uniformity_ratio = most_common_count / len(scores)
        
        if uniformity_ratio >= threshold:
            return GovernanceViolation(
                rule_id="RULE-9",
                rule_name="Suspicious Uniformity = Wrong Test",
                description=f"{most_common_count}/{len(scores)} claims ({uniformity_ratio*100:.0f}%) have identical score {most_common_score}%",
                evidence=f"score_distribution={score_counts}",
                recommendation="Change test methodology - tests aren't exercising claim functionality",
                severity="CRITICAL"
            )
        return None
    
    # =========================================================================
    # RULE 10: Learning Direction Matters
    # =========================================================================
    def check_rule_10_learning_direction(self,
                                          score: float,
                                          threshold_change: float) -> Optional[GovernanceViolation]:
        """
        RULE 10: Learning direction must be correct.
        
        - Lower score (< 50%) = missed detection = LOWER threshold to detect more
        - Baseline (50%) = no detection = slightly LOWER threshold
        - Higher score (≥ 70%) = good detection = maintain/raise threshold
        
        50% baseline is NOT success - it's absence of detection.
        """
        if score < 50 and threshold_change > 0:
            return GovernanceViolation(
                rule_id="RULE-10",
                rule_name="Learning Direction Matters",
                description=f"Score {score}% (failed) but threshold RAISED by {threshold_change}",
                evidence=f"score={score}%, threshold_change=+{threshold_change}",
                recommendation="Failed detection should LOWER threshold to be more sensitive",
                severity="CRITICAL"
            )
        
        if 50 <= score < 70 and threshold_change > 0.1:
            return GovernanceViolation(
                rule_id="RULE-10",
                rule_name="Learning Direction Matters",
                description=f"Score {score}% (baseline/partial) but threshold RAISED significantly by {threshold_change}",
                evidence=f"score={score}%, threshold_change=+{threshold_change}",
                recommendation="Baseline (50%) is absence of detection - should lower threshold",
                severity="HIGH"
            )
        
        return None
    
    # =========================================================================
    # Run All Governance Checks
    # =========================================================================
    def run_governance_check(self, context: Dict[str, Any]) -> List[GovernanceViolation]:
        """
        Run all governance checks on the given context.
        
        Context should contain:
        - components: List of {name, exists, is_called, data_flows}
        - metrics: {current, target, improved_from, previous}
        - claims: List of {claim_id, methodology, score}
        - scores: List of all scores
        - learning: List of {score, threshold_change}
        - test_results: List of test result dicts
        - data_flow: List of DataFlowStep
        - structures: Dict of data structures to check for emptiness
        """
        violations = []
        
        # Rule 1: Integration
        for comp in context.get('components', []):
            v = self.check_rule_1_integration(
                comp['name'], comp['exists'], comp['is_called'], comp['data_flows']
            )
            if v:
                violations.append(v)
        
        # Rule 2: Success criteria
        metrics = context.get('metrics', {})
        if metrics and 'current' in metrics and 'target' in metrics:
            v = self.check_rule_2_success_criteria(
                metrics.get('current', 0),
                metrics.get('target', 70),
                metrics.get('improved_from', 0)
            )
            if v:
                violations.append(v)
        
        # Rule 3: Question uniformity
        test_results = context.get('test_results', [])
        if test_results:
            v = self.check_rule_3_question_uniformity(test_results)
            if v:
                violations.append(v)
        
        # Rule 4 is covered by Rule 8 (specific claim ID version)
        
        # Rule 5: Data flow
        data_flow = context.get('data_flow', [])
        if data_flow:
            v = self.check_rule_5_data_flow(data_flow)
            if v:
                violations.append(v)
        
        # Rule 6: Empty structures
        structures = context.get('structures', {})
        if structures:
            v = self.check_rule_6_empty_structures(structures)
            if v:
                violations.append(v)
        
        # Rule 7: Push past incremental
        if metrics and 'previous' in metrics:
            v = self.check_rule_7_push_past_incremental(
                metrics.get('current', 0),
                metrics.get('previous', 0),
                metrics.get('target', 70)
            )
            if v:
                violations.append(v)
        
        # Rule 8: Methodology match
        for claim in context.get('claims', []):
            v = self.check_rule_8_methodology(
                claim['claim_id'],
                TestMethodology(claim['methodology'])
            )
            if v:
                violations.append(v)
        
        # Rule 9: Uniformity
        scores = context.get('scores', [])
        if scores:
            v = self.check_rule_9_uniformity(scores)
            if v:
                violations.append(v)
        
        # Rule 10: Learning direction
        for learning in context.get('learning', []):
            v = self.check_rule_10_learning_direction(
                learning['score'],
                learning['threshold_change']
            )
            if v:
                violations.append(v)
        
        self.violations = violations
        return violations
    
    def get_violation_summary(self) -> str:
        """Get summary of all violations."""
        if not self.violations:
            return "✅ No governance violations detected"
        
        summary = f"⚠️ {len(self.violations)} governance violations detected:\n"
        for v in self.violations:
            summary += f"\n  [{v.severity}] {v.rule_id}: {v.rule_name}\n"
            summary += f"      {v.description}\n"
            summary += f"      Recommendation: {v.recommendation}\n"
        
        return summary
    
    def get_rules_summary(self) -> str:
        """Get summary of all 10 rules."""
        return """
BAIS Governance Rules (All 10 Implemented):

RULE 1: Integration > Existence
   Components must be called in production flow, not just exist.

RULE 2: Define Success Upfront
   Success = meeting target, not "any improvement".

RULE 3: Question Uniformity
   Many identical results = wrong methodology.

RULE 4: Match Methodology to Claim Type
   Content→A/B, Algorithmic→Unit, Behavioral→Scenario, API→Functional

RULE 5: Trace Data Flow
   Input → Processing → Detection → Learning → Adaptation → Output

RULE 6: Check for Empty Structures
   Empty lists, zero counts = incomplete integration.

RULE 7: Push Past Incremental
   Don't celebrate small gains far from target.

RULE 8: Test Methodology Must Match Claim Type
   (Specific claim ID version of Rule 4)

RULE 9: Suspicious Uniformity = Wrong Test
   High score uniformity = test not exercising claims.

RULE 10: Learning Direction Matters
   50% baseline should LOWER threshold, not raise it.
"""

    # ===== Learning Interface Methods =====
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)
    
    def record_outcome(self, outcome: Dict) -> None:
        """Record outcome for learning."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback."""
        if not hasattr(self, '_feedback'):
            self._feedback = []
        self._feedback.append(feedback)
    
    def get_statistics(self) -> Dict:
        """Return learning statistics."""
        return {'outcomes': len(getattr(self, '_outcomes', []))}
    
    def serialize_state(self) -> Dict:
        """Serialize state for persistence."""
        return {'outcomes': getattr(self, '_outcomes', [])[-100:]}
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore state from serialized data."""
        self._outcomes = state.get('outcomes', [])


def test_governance_rules():
    """Test the governance rules system."""
    print("=" * 70)
    print("TESTING ALL 10 BAIS GOVERNANCE RULES")
    print("=" * 70)
    
    rules = BAISGovernanceRules()
    print(rules.get_rules_summary())
    
    # Test context with violations
    context = {
        'components': [
            {'name': 'OCOLearner', 'exists': True, 'is_called': True, 'data_flows': True},
            {'name': 'OldModule', 'exists': True, 'is_called': False, 'data_flows': False},
        ],
        'metrics': {
            'current': 14.6,
            'previous': 13.1,
            'target': 70,
            'improved_from': 13.1
        },
        'claims': [
            {'claim_id': 'PPA2-C1-1', 'methodology': 'ab_testing'},  # Wrong!
            {'claim_id': 'PPA3-Behav-1', 'methodology': 'scenario_testing'},  # Correct
        ],
        'scores': [50, 50, 50, 50, 50, 75, 50, 50, 50, 50],  # Suspicious uniformity
        'learning': [
            {'score': 35, 'threshold_change': 0.5},  # Wrong direction!
            {'score': 75, 'threshold_change': 0.1},  # Correct
        ],
        'test_results': [
            {'score': 50}, {'score': 50}, {'score': 50}, {'score': 50},
            {'score': 50}, {'score': 50}, {'score': 50}, {'score': 50},
        ],
        'structures': {
            'calibration_scores': [],
            'audit_records_count': 0,
            'thresholds': {'general': 50},
        },
        'data_flow': [
            DataFlowStep('input', True, True, True),
            DataFlowStep('processing', True, True, False),  # Broken link!
            DataFlowStep('output', False, True, True),
        ],
    }
    
    violations = rules.run_governance_check(context)
    print(rules.get_violation_summary())
    
    print(f"\nTotal violations: {len(violations)}")
    return violations


if __name__ == "__main__":
    test_governance_rules()
