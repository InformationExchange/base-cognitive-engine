"""
BAIS Mission Alignment Checker

Prevents drift from stated objectives during complex operations.
This is the component that would have caught "blocking instead of improving".

Patent Alignment: Novel Invention - Real-Time Mission Governance
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
import re
from datetime import datetime


class DriftType(Enum):
    """Types of mission drift"""
    NONE = "none"
    SCOPE_CREEP = "scope_creep"          # Adding features not requested
    OBJECTIVE_SHIFT = "objective_shift"   # Changed what we're trying to achieve
    METHOD_DRIFT = "method_drift"         # Wrong approach for the goal
    QUALITY_DEGRADATION = "quality_degradation"  # Cutting corners
    OVER_ENGINEERING = "over_engineering"  # Adding unnecessary complexity


class AlignmentStatus(Enum):
    """Overall alignment status"""
    ALIGNED = "aligned"
    WARNING = "warning"
    DRIFT_DETECTED = "drift_detected"
    CRITICAL_DRIFT = "critical_drift"


@dataclass
class MissionObjective:
    """A specific mission objective"""
    id: str
    description: str
    success_criteria: List[str]
    anti_patterns: List[str]  # What NOT to do
    priority: int = 1
    completed: bool = False


@dataclass
class DriftIndicator:
    """Evidence of potential drift"""
    drift_type: DriftType
    description: str
    evidence: str
    severity: float  # 0-1
    recommendation: str


@dataclass
class AlignmentCheckResult:
    """Result of alignment check"""
    status: AlignmentStatus
    drift_indicators: List[DriftIndicator]
    alignment_score: float  # 0-1
    objectives_status: Dict[str, bool]  # objective_id -> on_track
    correction_guidance: Optional[str]
    reasoning_trace: List[str]


@dataclass
class MissionContext:
    """Complete mission context"""
    primary_objective: str
    secondary_objectives: List[str]
    constraints: List[str]
    anti_patterns: List[str]
    success_definition: str
    created_at: datetime = field(default_factory=datetime.now)


class MissionAlignmentChecker:
    """
    Checks if current work aligns with stated mission.
    
    Key capabilities:
    1. Track objectives and their status
    2. Detect drift from objectives
    3. Provide correction guidance
    4. Maintain reasoning trace
    """
    
    # Common drift patterns
    DRIFT_PATTERNS = {
        'blocking_instead_of_improving': {
            'patterns': [
                r'reject(?:ed|ing|s)?\s+(?:the\s+)?(?:query|request|input)',
                r'block(?:ed|ing|s)?\s+(?:the\s+)?(?:query|request|input)',
                r'deny(?:ied|ing|s)?\s+access',
                r'flag(?:ged|ging|s)?\s+(?:as\s+)?(?:unsafe|dangerous|harmful)',
                r'prevent(?:ed|ing|s)?\s+(?:the\s+)?(?:action|request)',
            ],
            'drift_type': DriftType.OBJECTIVE_SHIFT,
            'description': "System is blocking/rejecting instead of improving"
        },
        'simulating_instead_of_real': {
            'patterns': [
                r'simulat(?:ed|ing|e)',
                r'mock(?:ed|ing)?(?:\s+data)?',
                r'fake(?:d|ing)?\s+(?:response|result|output)',
                r'construct(?:ed|ing)?\s+(?:input|test)',
                r'hypothetical',
            ],
            'drift_type': DriftType.METHOD_DRIFT,
            'description': "Using simulated data instead of real testing"
        },
        'premature_completion': {
            'patterns': [
                r'(?:all\s+)?(?:tests?\s+)?pass(?:ed|ing)?',
                r'100%\s+(?:complete|success|pass)',
                r'(?:fully\s+)?implement(?:ed|ation)?(?:\s+complete)?',
                r'done|complete|finished',
            ],
            'drift_type': DriftType.QUALITY_DEGRADATION,
            'description': "Declaring completion without verification"
        },
        'scope_expansion': {
            'patterns': [
                r'also\s+(?:add(?:ed|ing)?|implement(?:ed|ing)?)',
                r'bonus\s+feature',
                r'extra\s+(?:feature|capability)',
                r'while\s+(?:we\'re|I\'m)\s+at\s+it',
                r'might\s+as\s+well',
            ],
            'drift_type': DriftType.SCOPE_CREEP,
            'description': "Adding features not part of original objective"
        },
        'over_abstraction': {
            'patterns': [
                r'(?:generic|abstract|flexible)\s+(?:framework|architecture|design)',
                r'future[\s-]?proof',
                r'(?:extensible|scalable)\s+(?:for|to)',
                r'could\s+be\s+(?:reused|extended)',
            ],
            'drift_type': DriftType.OVER_ENGINEERING,
            'description': "Creating unnecessary abstractions"
        }
    }
    
    # Success criteria for common objectives
    OBJECTIVE_TEMPLATES = {
        'improve_llm_output': MissionObjective(
            id='improve_llm_output',
            description="Improve the quality of LLM outputs",
            success_criteria=[
                'Response is more accurate than original',
                'Factual errors are corrected',
                'Reasoning is sound',
                'Appropriate caveats are added',
                'User gets actionable, safe guidance'
            ],
            anti_patterns=[
                'Blocking the response entirely',
                'Adding generic disclaimers without addressing issues',
                'Rejecting valid queries',
                'Over-cautious non-answers'
            ],
            priority=1
        ),
        'clinical_accuracy': MissionObjective(
            id='clinical_accuracy',
            description="Ensure clinical/scientific accuracy",
            success_criteria=[
                'Claims are evidence-based',
                'Uncertainty is acknowledged',
                'Limitations are stated',
                'No optimistic exaggeration'
            ],
            anti_patterns=[
                'Simulated test data',
                'Constructed inputs',
                'Optimistic reporting',
                'Cherry-picked results'
            ],
            priority=1
        )
    }
    
    def __init__(self):
        self.active_mission: Optional[MissionContext] = None
        self.objectives: Dict[str, MissionObjective] = {}
        self.reasoning_trace: List[str] = []
        self.check_history: List[AlignmentCheckResult] = []
    
    def set_mission(self, mission: MissionContext) -> None:
        """Set the active mission context"""
        self.active_mission = mission
        self.reasoning_trace = []
        self._log(f"Mission set: {mission.primary_objective}")
        
        # Add default objectives based on mission
        if 'improve' in mission.primary_objective.lower():
            self.add_objective(self.OBJECTIVE_TEMPLATES['improve_llm_output'])
        if 'clinical' in mission.primary_objective.lower() or 'accurate' in mission.primary_objective.lower():
            self.add_objective(self.OBJECTIVE_TEMPLATES['clinical_accuracy'])
    
    def add_objective(self, objective: MissionObjective) -> None:
        """Add an objective to track"""
        self.objectives[objective.id] = objective
        self._log(f"Added objective: {objective.id}")
    
    def check_alignment(self, 
                        current_output: str,
                        context: Optional[str] = None) -> AlignmentCheckResult:
        """
        Check if current output aligns with mission.
        
        Args:
            current_output: The text/output to check
            context: Additional context about what's being done
            
        Returns:
            AlignmentCheckResult with status and guidance
        """
        self._log(f"Checking alignment for output ({len(current_output)} chars)")
        
        drift_indicators = []
        
        # Check against drift patterns
        for pattern_name, pattern_info in self.DRIFT_PATTERNS.items():
            for pattern in pattern_info['patterns']:
                matches = re.findall(pattern, current_output, re.IGNORECASE)
                if matches:
                    severity = min(len(matches) * 0.2, 1.0)
                    drift_indicators.append(DriftIndicator(
                        drift_type=pattern_info['drift_type'],
                        description=pattern_info['description'],
                        evidence=matches[0] if matches else "",
                        severity=severity,
                        recommendation=self._get_recommendation(pattern_info['drift_type'])
                    ))
                    self._log(f"Drift detected: {pattern_name} (severity: {severity:.2f})")
        
        # Check against mission anti-patterns if set
        if self.active_mission:
            for anti_pattern in self.active_mission.anti_patterns:
                if anti_pattern.lower() in current_output.lower():
                    drift_indicators.append(DriftIndicator(
                        drift_type=DriftType.OBJECTIVE_SHIFT,
                        description=f"Anti-pattern detected: {anti_pattern}",
                        evidence=anti_pattern,
                        severity=0.8,
                        recommendation=f"Avoid: {anti_pattern}"
                    ))
        
        # Check objective-specific anti-patterns
        for obj_id, obj in self.objectives.items():
            for anti in obj.anti_patterns:
                if anti.lower() in current_output.lower():
                    drift_indicators.append(DriftIndicator(
                        drift_type=DriftType.OBJECTIVE_SHIFT,
                        description=f"Violates objective '{obj_id}': {anti}",
                        evidence=anti,
                        severity=0.7,
                        recommendation=f"Align with objective: {obj.description}"
                    ))
        
        # Calculate alignment score
        alignment_score = self._calculate_alignment_score(drift_indicators)
        
        # Determine status
        status = self._determine_status(alignment_score, drift_indicators)
        
        # Get objectives status
        objectives_status = self._check_objectives_status(current_output)
        
        # Generate correction guidance if needed
        correction_guidance = None
        if status in [AlignmentStatus.DRIFT_DETECTED, AlignmentStatus.CRITICAL_DRIFT]:
            correction_guidance = self._generate_correction_guidance(drift_indicators)
        
        result = AlignmentCheckResult(
            status=status,
            drift_indicators=drift_indicators,
            alignment_score=alignment_score,
            objectives_status=objectives_status,
            correction_guidance=correction_guidance,
            reasoning_trace=self.reasoning_trace.copy()
        )
        
        self.check_history.append(result)
        self._log(f"Alignment check complete: {status.value} (score: {alignment_score:.2f})")
        
        return result
    
    def _log(self, message: str):
        """Log reasoning step"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.reasoning_trace.append(f"[{timestamp}] {message}")
    
    def _calculate_alignment_score(self, indicators: List[DriftIndicator]) -> float:
        """Calculate overall alignment score"""
        if not indicators:
            return 1.0
        
        total_severity = sum(i.severity for i in indicators)
        # More drift = lower score
        score = max(0, 1.0 - (total_severity / 3.0))  # Cap at 3 severe issues = 0
        return score
    
    def _determine_status(self, score: float, 
                          indicators: List[DriftIndicator]) -> AlignmentStatus:
        """Determine alignment status from score and indicators"""
        
        # Check for critical drift types
        critical_types = [DriftType.OBJECTIVE_SHIFT]
        has_critical = any(i.drift_type in critical_types and i.severity > 0.7 
                         for i in indicators)
        
        if has_critical or score < 0.3:
            return AlignmentStatus.CRITICAL_DRIFT
        elif score < 0.6:
            return AlignmentStatus.DRIFT_DETECTED
        elif score < 0.8:
            return AlignmentStatus.WARNING
        else:
            return AlignmentStatus.ALIGNED
    
    def _check_objectives_status(self, output: str) -> Dict[str, bool]:
        """Check status of each objective"""
        status = {}
        
        for obj_id, obj in self.objectives.items():
            # Check if any anti-patterns violated
            violated = any(anti.lower() in output.lower() for anti in obj.anti_patterns)
            status[obj_id] = not violated
        
        return status
    
    def _get_recommendation(self, drift_type: DriftType) -> str:
        """Get recommendation for drift type"""
        recommendations = {
            DriftType.OBJECTIVE_SHIFT: "Return to primary objective. Focus on IMPROVING output, not blocking.",
            DriftType.METHOD_DRIFT: "Use real data and actual testing, not simulations.",
            DriftType.QUALITY_DEGRADATION: "Verify completion with objective criteria before declaring done.",
            DriftType.SCOPE_CREEP: "Focus on requested features only. Document extras for later.",
            DriftType.OVER_ENGINEERING: "Keep it simple. Only build what's needed now.",
        }
        return recommendations.get(drift_type, "Review mission objectives.")
    
    def _generate_correction_guidance(self, 
                                       indicators: List[DriftIndicator]) -> str:
        """Generate specific correction guidance"""
        guidance_parts = ["MISSION ALIGNMENT CORRECTION NEEDED:\n"]
        
        # Group by drift type
        by_type: Dict[DriftType, List[DriftIndicator]] = {}
        for ind in indicators:
            if ind.drift_type not in by_type:
                by_type[ind.drift_type] = []
            by_type[ind.drift_type].append(ind)
        
        for drift_type, inds in by_type.items():
            guidance_parts.append(f"\n{drift_type.value.upper()}:")
            for ind in inds:
                guidance_parts.append(f"  - {ind.description}")
                guidance_parts.append(f"    Evidence: '{ind.evidence}'")
                guidance_parts.append(f"    Action: {ind.recommendation}")
        
        if self.active_mission:
            guidance_parts.append(f"\n\nREMINDER - Primary Mission: {self.active_mission.primary_objective}")
            guidance_parts.append(f"Success Definition: {self.active_mission.success_definition}")
        
        return "\n".join(guidance_parts)
    
    def get_mission_summary(self) -> str:
        """Get summary of current mission status"""
        if not self.active_mission:
            return "No active mission set."
        
        lines = [
            "=" * 50,
            "MISSION STATUS SUMMARY",
            "=" * 50,
            f"Primary: {self.active_mission.primary_objective}",
            f"Created: {self.active_mission.created_at.strftime('%Y-%m-%d %H:%M')}",
            "",
            "Objectives:"
        ]
        
        for obj_id, obj in self.objectives.items():
            status = "✓" if obj.completed else "○"
            lines.append(f"  {status} {obj.description}")
        
        if self.check_history:
            latest = self.check_history[-1]
            lines.extend([
                "",
                f"Latest Check: {latest.status.value}",
                f"Alignment Score: {latest.alignment_score:.2f}",
            ])
        
        return "\n".join(lines)

    # Learning Interface Methods
    
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



def test_mission_alignment():
    """Test the mission alignment checker"""
    checker = MissionAlignmentChecker()
    
    # Set up mission
    mission = MissionContext(
        primary_objective="Improve LLM output quality for medical advice",
        secondary_objectives=["Ensure safety", "Maintain accuracy"],
        constraints=["Must not block valid queries", "Must provide actionable guidance"],
        anti_patterns=["Rejecting queries", "Generic disclaimers only", "Simulated testing"],
        success_definition="User receives accurate, safe, actionable medical guidance"
    )
    
    checker.set_mission(mission)
    
    print("=" * 70)
    print("MISSION ALIGNMENT CHECKER TEST")
    print("=" * 70)
    
    # Test cases
    test_outputs = [
        # Good - aligned
        "The response has been improved to include appropriate medical disclaimers "
        "and more accurate dosage information.",
        
        # Bad - blocking instead of improving
        "The query has been rejected as potentially harmful. "
        "Access denied to medical information.",
        
        # Bad - simulated testing
        "Using mock data and simulated responses, all tests passed. "
        "100% complete with constructed inputs.",
        
        # Good - genuine improvement
        "Enhanced the response to clarify that the user should consult their doctor "
        "for personalized advice. Added evidence-based information about the condition."
    ]
    
    for i, output in enumerate(test_outputs):
        print(f"\n--- Test {i+1} ---")
        print(f"Output: {output[:80]}...")
        
        result = checker.check_alignment(output)
        
        print(f"Status: {result.status.value}")
        print(f"Alignment Score: {result.alignment_score:.2f}")
        
        if result.drift_indicators:
            print("Drift Detected:")
            for ind in result.drift_indicators:
                print(f"  - {ind.drift_type.value}: {ind.description}")
        
        if result.correction_guidance:
            print(f"\nCorrection Guidance:\n{result.correction_guidance[:200]}...")
    
    print("\n" + checker.get_mission_summary())

    # =========================================================================
    # Learning Interface
    # =========================================================================
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        """Record operation outcome for learning."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback to adjust behavior."""
        if not hasattr(self, '_learning_params'):
            self._learning_params = {}
        was_correct = feedback.get('was_correct', True)
        if not was_correct:
            for key in self._learning_params:
                self._learning_params[key] *= 0.95
    
    def get_statistics(self) -> Dict:
        """Return learning statistics."""
        return {
            'outcomes': len(getattr(self, '_outcomes', [])),
            'params': getattr(self, '_learning_params', {})
        }
    
    def serialize_state(self) -> Dict:
        """Serialize state for persistence."""
        return {
            'outcomes': getattr(self, '_outcomes', [])[-100:],
            'params': getattr(self, '_learning_params', {}),
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore state from serialized data."""
        self._outcomes = state.get('outcomes', [])
        self._learning_params = state.get('params', {})


if __name__ == "__main__":
    test_mission_alignment()






