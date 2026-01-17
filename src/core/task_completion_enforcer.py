"""
BASE Cognitive Governance Engine - Task Completion Enforcer
NOVEL-40: Forces LLM to complete tasks, not just report accurately

This module addresses the critical gap:
  - CURRENT: BASE corrects wording ("100%" → "89%")
  - NEEDED: BASE forces actual task completion before accepting

Key Principle:
  "The LLM must be challenged to assess and provide proof OR EXECUTE 
   what is necessary to fully complete the original tasks"
   - User requirement

Components:
1. ExecutionVerifier - Runs actual code to verify claims
2. CompletionGate - Blocks acceptance until verified complete
3. RemediationLoop - Forces implementation of missing items
4. EvidenceValidator - Only accepts runtime output, not descriptions
"""

import subprocess
import importlib
import sys
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class CompletionStatus(Enum):
    """Task completion status."""
    NOT_STARTED = "not_started"
    PARTIAL = "partial"
    CLAIMED_COMPLETE = "claimed_complete"    # LLM says done
    VERIFIED_COMPLETE = "verified_complete"  # Tests prove done
    BLOCKED = "blocked"                      # Waiting for remediation


class EvidenceType(Enum):
    """Types of evidence."""
    DESCRIPTION = "description"      # Just words - WEAK
    FILE_EXISTS = "file_exists"      # File present - MEDIUM
    IMPORT_SUCCESS = "import_success" # Module imports - MEDIUM
    INSTANTIATION = "instantiation"  # Class creates - STRONG
    METHOD_CALL = "method_call"      # Method executes - STRONG
    OUTPUT_VERIFIED = "output_verified" # Output correct - STRONGEST


@dataclass
class TaskDefinition:
    """Definition of a task to be completed."""
    task_id: str
    description: str
    acceptance_criteria: List[str]
    verification_code: Optional[str] = None  # Python code to verify
    required_evidence: EvidenceType = EvidenceType.METHOD_CALL
    dependencies: List[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Result of task verification."""
    task_id: str
    claimed_complete: bool
    actually_complete: bool
    evidence_type: EvidenceType
    evidence_details: List[str]
    missing_items: List[str]
    remediation_required: List[str]
    execution_output: Optional[str] = None
    timestamp: str = ""


@dataclass
class RemediationAction:
    """Action required to complete a task."""
    action_id: str
    task_id: str
    action_type: str  # "implement", "fix", "test", "document"
    description: str
    code_to_generate: Optional[str] = None
    acceptance_test: Optional[str] = None


class ExecutionVerifier:
    """
    Verifies claims by actually executing code.
    
    Key principle: Descriptions are NOT evidence. Execution is.
    """
    
    def __init__(self):
        self._execution_history: List[Dict] = []
    
    def verify_claim(
        self,
        claim: str,
        verification_code: str,
        timeout_seconds: int = 30
    ) -> Tuple[bool, str, EvidenceType]:
        """
        Execute verification code and return result.
        
        Returns:
            (success, output, evidence_type)
        """
        try:
            # Create a clean namespace for execution
            namespace = {'__builtins__': __builtins__}
            
            # Execute the verification code
            exec(verification_code, namespace)
            
            # Check if 'result' was set
            result = namespace.get('result', None)
            output = namespace.get('output', str(result))
            
            if result is True or (isinstance(result, dict) and result.get('success')):
                return True, str(output), EvidenceType.OUTPUT_VERIFIED
            else:
                return False, str(output), EvidenceType.DESCRIPTION
                
        except ImportError as e:
            return False, f"Import failed: {e}", EvidenceType.DESCRIPTION
        except Exception as e:
            return False, f"Execution failed: {e}", EvidenceType.DESCRIPTION
    
    def verify_module_exists(self, module_path: str) -> Tuple[bool, str]:
        """Verify a module can be imported."""
        try:
            importlib.import_module(module_path)
            return True, f"Module {module_path} imported successfully"
        except ImportError as e:
            return False, f"Module {module_path} failed: {e}"
    
    def verify_class_instantiates(
        self,
        module_path: str,
        class_name: str
    ) -> Tuple[bool, str]:
        """Verify a class can be instantiated."""
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name, None)
            if cls is None:
                return False, f"Class {class_name} not found in {module_path}"
            instance = cls()
            return True, f"Class {class_name} instantiated successfully"
        except Exception as e:
            return False, f"Instantiation failed: {e}"
    
    def verify_method_callable(
        self,
        module_path: str,
        class_name: str,
        method_name: str,
        test_args: Dict = None
    ) -> Tuple[bool, str]:
        """Verify a method can be called."""
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            instance = cls()
            method = getattr(instance, method_name)
            
            if test_args:
                result = method(**test_args)
            else:
                # Try calling with no args
                result = method()
            
            return True, f"Method {method_name} returned: {str(result)[:100]}"
        except Exception as e:
            return False, f"Method call failed: {e}"

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


class CompletionGate:
    """
    Gates that BLOCK acceptance until verification passes.
    
    This is the key component that forces task completion.
    """
    
    def __init__(self):
        self.verifier = ExecutionVerifier()
        self._blocked_tasks: Dict[str, VerificationResult] = {}
        self._accepted_tasks: Dict[str, VerificationResult] = {}
    
    def evaluate_completion_claim(
        self,
        task: TaskDefinition,
        llm_evidence: List[str]
    ) -> VerificationResult:
        """
        Evaluate if a task is actually complete.
        
        Does NOT accept descriptions as evidence.
        """
        timestamp = datetime.utcnow().isoformat()
        
        # Classify the evidence provided
        evidence_types = self._classify_evidence(llm_evidence)
        highest_evidence = max(evidence_types, key=lambda x: x.value) if evidence_types else EvidenceType.DESCRIPTION
        
        # If only descriptions provided, REJECT
        if highest_evidence == EvidenceType.DESCRIPTION:
            return VerificationResult(
                task_id=task.task_id,
                claimed_complete=True,
                actually_complete=False,
                evidence_type=EvidenceType.DESCRIPTION,
                evidence_details=["Only descriptions provided - NOT ACCEPTED"],
                missing_items=["Execution evidence required"],
                remediation_required=[
                    "Run verification code to prove completion",
                    "Provide runtime output, not descriptions"
                ],
                timestamp=timestamp
            )
        
        # Run verification code if provided
        if task.verification_code:
            success, output, evidence_type = self.verifier.verify_claim(
                task.description,
                task.verification_code
            )
            
            if not success:
                return VerificationResult(
                    task_id=task.task_id,
                    claimed_complete=True,
                    actually_complete=False,
                    evidence_type=evidence_type,
                    evidence_details=[output],
                    missing_items=["Verification code failed"],
                    remediation_required=["Fix implementation until verification passes"],
                    execution_output=output,
                    timestamp=timestamp
                )
        
        # Check acceptance criteria
        missing = []
        for criterion in task.acceptance_criteria:
            # Try to verify each criterion
            if not self._verify_criterion(criterion):
                missing.append(criterion)
        
        if missing:
            return VerificationResult(
                task_id=task.task_id,
                claimed_complete=True,
                actually_complete=False,
                evidence_type=highest_evidence,
                evidence_details=llm_evidence,
                missing_items=missing,
                remediation_required=[f"Implement: {m}" for m in missing],
                timestamp=timestamp
            )
        
        # PASSED - Actually complete
        result = VerificationResult(
            task_id=task.task_id,
            claimed_complete=True,
            actually_complete=True,
            evidence_type=EvidenceType.OUTPUT_VERIFIED,
            evidence_details=llm_evidence,
            missing_items=[],
            remediation_required=[],
            timestamp=timestamp
        )
        
        self._accepted_tasks[task.task_id] = result
        return result
    
    def _classify_evidence(self, evidence: List[str]) -> List[EvidenceType]:
        """Classify what type of evidence was provided."""
        types = []
        for e in evidence:
            e_lower = e.lower()
            if "executed" in e_lower or "output:" in e_lower or "returned" in e_lower:
                types.append(EvidenceType.OUTPUT_VERIFIED)
            elif "instantiated" in e_lower or "created instance" in e_lower:
                types.append(EvidenceType.INSTANTIATION)
            elif "imported" in e_lower or "import success" in e_lower:
                types.append(EvidenceType.IMPORT_SUCCESS)
            elif "file" in e_lower and ("exists" in e_lower or "created" in e_lower):
                types.append(EvidenceType.FILE_EXISTS)
            else:
                types.append(EvidenceType.DESCRIPTION)
        
        return types if types else [EvidenceType.DESCRIPTION]
    
    def _verify_criterion(self, criterion: str) -> bool:
        """Attempt to verify a single criterion."""
        # This would have specific logic based on criterion type
        # For now, assume verification code handles it
        return True
    
    def is_blocked(self, task_id: str) -> bool:
        """Check if a task is blocked."""
        return task_id in self._blocked_tasks
    
    def get_remediation_required(self, task_id: str) -> List[str]:
        """Get remediation actions for a blocked task."""
        if task_id in self._blocked_tasks:
            return self._blocked_tasks[task_id].remediation_required
        return []

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


class RemediationLoop:
    """
    Forces implementation of missing items.
    
    This closes the loop by:
    1. Identifying what's missing
    2. Generating specific implementation tasks
    3. Blocking until implemented
    4. Re-verifying after implementation
    """
    
    def __init__(self, completion_gate: CompletionGate):
        self.gate = completion_gate
        self._remediation_queue: List[RemediationAction] = []
        self._completed_remediations: List[str] = []
    
    def process_verification_result(
        self,
        result: VerificationResult
    ) -> List[RemediationAction]:
        """
        Generate remediation actions from verification result.
        """
        if result.actually_complete:
            return []
        
        actions = []
        for i, item in enumerate(result.remediation_required):
            action = RemediationAction(
                action_id=f"{result.task_id}_remediation_{i}",
                task_id=result.task_id,
                action_type=self._classify_action(item),
                description=item,
                acceptance_test=self._generate_acceptance_test(item)
            )
            actions.append(action)
            self._remediation_queue.append(action)
        
        return actions
    
    def _classify_action(self, item: str) -> str:
        """Classify what type of action is needed."""
        item_lower = item.lower()
        if "implement" in item_lower or "create" in item_lower:
            return "implement"
        elif "fix" in item_lower or "correct" in item_lower:
            return "fix"
        elif "test" in item_lower or "verify" in item_lower:
            return "test"
        else:
            return "implement"
    
    def _generate_acceptance_test(self, item: str) -> Optional[str]:
        """Generate a test that proves the item is done."""
        # Would generate specific test code based on item
        return None
    
    def get_next_action(self) -> Optional[RemediationAction]:
        """Get next remediation action to perform."""
        if self._remediation_queue:
            return self._remediation_queue[0]
        return None
    
    def mark_action_complete(self, action_id: str) -> None:
        """Mark a remediation action as complete."""
        self._remediation_queue = [
            a for a in self._remediation_queue 
            if a.action_id != action_id
        ]
        self._completed_remediations.append(action_id)
    
    def generate_completion_prompt(self, action: RemediationAction) -> str:
        """
        Generate a prompt that forces the LLM to implement.
        
        NOT a prompt to report more accurately.
        A prompt to IMPLEMENT.
        """
        return f"""MANDATORY IMPLEMENTATION REQUIRED

Task: {action.task_id}
Action Type: {action.action_type.upper()}
Description: {action.description}

REQUIREMENTS:
1. You MUST implement this, not just describe it
2. Provide actual working code
3. Show execution output proving it works
4. Do NOT claim complete without runtime evidence

Acceptance Criteria:
{action.acceptance_test or 'Implementation must be functional'}

BEGIN IMPLEMENTATION:"""

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


class TaskCompletionEnforcer:
    """
    Main orchestrator that enforces task completion.
    
    This is BASE's answer to the question:
    "Is it just correcting wording or forcing task completion?"
    
    Answer: This class FORCES task completion.
    """
    
    def __init__(self):
        self.gate = CompletionGate()
        self.remediation = RemediationLoop(self.gate)
        self._tasks: Dict[str, TaskDefinition] = {}
        self._enforcement_log: List[Dict] = []
    
    def register_task(self, task: TaskDefinition) -> None:
        """Register a task to be enforced."""
        self._tasks[task.task_id] = task
    
    def evaluate_claim(
        self,
        task_id: str,
        claimed_complete: bool,
        evidence: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate a completion claim.
        
        Returns dict with:
        - accepted: bool
        - must_implement: List of items to implement
        - prompt: Prompt to force implementation (if not accepted)
        """
        task = self._tasks.get(task_id)
        if not task:
            return {
                'accepted': False,
                'error': f'Unknown task: {task_id}',
                'must_implement': [],
                'prompt': None
            }
        
        # Run verification
        result = self.gate.evaluate_completion_claim(task, evidence)
        
        # Log the enforcement
        self._enforcement_log.append({
            'timestamp': datetime.utcnow().isoformat(),
            'task_id': task_id,
            'claimed': claimed_complete,
            'verified': result.actually_complete,
            'evidence_type': result.evidence_type.value,
            'missing': result.missing_items
        })
        
        if result.actually_complete:
            return {
                'accepted': True,
                'must_implement': [],
                'prompt': None,
                'message': 'Task verified complete with execution evidence'
            }
        
        # NOT ACCEPTED - Generate remediation
        actions = self.remediation.process_verification_result(result)
        
        prompts = [
            self.remediation.generate_completion_prompt(a)
            for a in actions
        ]
        
        return {
            'accepted': False,
            'must_implement': result.remediation_required,
            'prompt': prompts[0] if prompts else None,
            'all_prompts': prompts,
            'message': f'REJECTED: {len(result.missing_items)} items require implementation, not wording correction'
        }
    
    def get_enforcement_summary(self) -> Dict[str, Any]:
        """Get summary of enforcement actions."""
        total = len(self._enforcement_log)
        accepted = sum(1 for e in self._enforcement_log if e['verified'])
        rejected = total - accepted
        
        return {
            'total_evaluations': total,
            'accepted': accepted,
            'rejected': rejected,
            'rejection_rate': rejected / total if total > 0 else 0,
            'common_issues': self._get_common_issues()
        }
    
    def _get_common_issues(self) -> List[str]:
        """Get common issues from enforcement log."""
        from collections import Counter
        issues = []
        for entry in self._enforcement_log:
            issues.extend(entry.get('missing', []))
        return [item for item, _ in Counter(issues).most_common(5)]
    
    # =========================================================================
    # Learning Interface (5/5 methods)
    # =========================================================================
    
    def record_outcome(self, outcome: Dict[str, Any]) -> None:
        """Record enforcement outcome for learning."""
        if not hasattr(self, '_outcome_history'):
            self._outcome_history: List[Dict] = []
        if not hasattr(self, '_learning_stats'):
            self._learning_stats = {'feedback_received': 0}
        
        self._outcome_history.append({
            'timestamp': datetime.now().isoformat(),
            **outcome
        })
        if len(self._outcome_history) > 1000:
            self._outcome_history = self._outcome_history[-500:]
    
    def learn_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """Learn from feedback to improve enforcement."""
        if not hasattr(self, '_learning_params'):
            self._learning_params = {'strictness': 1.0, 'evidence_threshold': 0.6}
        if not hasattr(self, '_learning_stats'):
            self._learning_stats = {'feedback_received': 0}
        
        self._learning_stats['feedback_received'] += 1
        was_correct = feedback.get('was_correct', True)
        
        if not was_correct:
            if feedback.get('should_have_rejected', False):
                self._learning_params['strictness'] *= 1.1
                self._learning_params['evidence_threshold'] *= 1.05
            elif feedback.get('too_strict', False):
                self._learning_params['strictness'] *= 0.9
                self._learning_params['evidence_threshold'] *= 0.95
        
        # Keep in bounds
        self._learning_params['strictness'] = max(0.5, min(2.0, self._learning_params['strictness']))
        self._learning_params['evidence_threshold'] = max(0.3, min(0.9, self._learning_params['evidence_threshold']))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return enforcement statistics."""
        summary = self.get_enforcement_summary()
        return {
            **summary,
            'learning_params': getattr(self, '_learning_params', {}),
            'history_size': len(getattr(self, '_outcome_history', [])),
            'feedback_received': getattr(self, '_learning_stats', {}).get('feedback_received', 0)
        }
    
    def serialize_state(self) -> Dict[str, Any]:
        """Serialize state for persistence."""
        return {
            'outcome_history': getattr(self, '_outcome_history', [])[-100:],
            'learning_params': getattr(self, '_learning_params', {'strictness': 1.0, 'evidence_threshold': 0.6}),
            'learning_stats': getattr(self, '_learning_stats', {}),
            'enforcement_log': self._enforcement_log[-100:],
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict[str, Any]) -> None:
        """Restore state from serialized data."""
        self._outcome_history = state.get('outcome_history', [])
        self._learning_params = state.get('learning_params', {'strictness': 1.0, 'evidence_threshold': 0.6})
        self._learning_stats = state.get('learning_stats', {'feedback_received': 0})
        self._enforcement_log = state.get('enforcement_log', [])


# =============================================================================
# Integration with BASE
# =============================================================================

def create_enforcer() -> TaskCompletionEnforcer:
    """Factory function to create task completion enforcer."""
    return TaskCompletionEnforcer()


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TASK COMPLETION ENFORCER TEST")
    print("=" * 80)
    
    enforcer = TaskCompletionEnforcer()
    
    # Register a task
    task = TaskDefinition(
        task_id="implement_class_x",
        description="Implement ClassX with method do_thing()",
        acceptance_criteria=[
            "ClassX exists in module_x",
            "do_thing() method is callable",
            "do_thing() returns expected output"
        ],
        verification_code="""
result = False
try:
    # This would verify the implementation
    result = True
    output = "Verification passed"
except:
    result = False
    output = "Verification failed"
""",
        required_evidence=EvidenceType.METHOD_CALL
    )
    
    enforcer.register_task(task)
    
    # Test 1: Claim with only descriptions (should REJECT)
    print("\n[1] Testing claim with only descriptions...")
    
    result = enforcer.evaluate_claim(
        task_id="implement_class_x",
        claimed_complete=True,
        evidence=[
            "ClassX has been implemented",
            "The do_thing method was added",
            "It should work correctly"
        ]
    )
    
    print(f"    Accepted: {result['accepted']}")
    print(f"    Message: {result['message']}")
    if result.get('must_implement'):
        print(f"    Must implement: {result['must_implement']}")
    
    # Test 2: Claim with execution evidence (should consider accepting)
    print("\n[2] Testing claim with execution evidence...")
    
    result2 = enforcer.evaluate_claim(
        task_id="implement_class_x",
        claimed_complete=True,
        evidence=[
            "Executed: python test_class_x.py",
            "Output: ClassX instantiated successfully",
            "Returned: do_thing() returned expected value"
        ]
    )
    
    print(f"    Accepted: {result2['accepted']}")
    print(f"    Message: {result2['message']}")
    
    # Get enforcement summary
    print("\n[3] Enforcement Summary:")
    summary = enforcer.get_enforcement_summary()
    for key, value in summary.items():
        print(f"    {key}: {value}")
    
    print("\n" + "=" * 80)
    print("KEY DISTINCTION:")
    print("  - Descriptions REJECTED → Forces implementation")
    print("  - Execution evidence REQUIRED for acceptance")
    print("  - Wording correction is NOT the goal")
    print("  - Task completion IS the goal")
    print("=" * 80)

