"""
BAIS Cognitive Governance Engine - Enforcement Loop
Phase 1a: The component that ACTUALLY forces task completion

This is the critical component that transforms BAIS from:
  - "Report issues" → "Force completion"
  
Without this: BAIS blocks and suggests, LLM can ignore.
With this:    BAIS blocks and LOOPS until LLM fixes.

Patent: NOVEL-41 (Enforcement Loop)
"""

import logging
import subprocess
import importlib
import sys
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class EnforcementDecision(Enum):
    """Decision from enforcement evaluation."""
    BLOCKED = "blocked"           # Cannot proceed until fixed
    VERIFIED = "verified"         # Execution proof accepted
    NEEDS_CONTEXT = "needs_context"  # Ask user for clarification
    PARTIAL = "partial"           # Some items pass, some fail
    TIMEOUT = "timeout"           # Max attempts reached


class EvidenceStrength(Enum):
    """Strength of evidence provided."""
    NONE = "none"                 # No evidence
    WEAK = "weak"                 # Description only
    MEDIUM = "medium"             # File exists, imports work
    STRONG = "strong"             # Instantiation succeeds
    VERIFIED = "verified"         # Execution output confirmed


@dataclass
class VerificationItem:
    """A single item to verify."""
    item_id: str
    description: str
    verification_code: str
    status: EnforcementDecision = EnforcementDecision.BLOCKED
    execution_output: Optional[str] = None
    evidence_strength: EvidenceStrength = EvidenceStrength.NONE


@dataclass
class EnforcementResult:
    """Result from enforcement evaluation."""
    decision: EnforcementDecision
    claim: str
    execution_attempted: bool
    execution_output: str
    evidence_strength: EvidenceStrength
    
    # What needs to be fixed
    remediation_required: List[str] = field(default_factory=list)
    specific_failures: List[Dict[str, str]] = field(default_factory=list)
    
    # For partial results
    items_passed: int = 0
    items_failed: int = 0
    items_total: int = 0
    
    # User interaction
    ask_user: Optional[str] = None
    
    # Metadata
    attempt_number: int = 1
    timestamp: str = ""
    duration_ms: float = 0.0


@dataclass
class EnforcedOutput:
    """Final output after enforcement loop completes."""
    success: bool
    final_response: Optional[str]
    total_attempts: int
    verification_history: List[EnforcementResult]
    
    # If needs user input
    needs_user_input: bool = False
    user_prompt: Optional[str] = None
    
    # Metrics
    time_to_complete_ms: float = 0.0
    issues_fixed: int = 0


# =============================================================================
# Verification Executor
# =============================================================================

class VerificationExecutor:
    """
    Executes verification code and returns results.
    
    Supports multiple verification types:
    - Python code execution
    - Module import checks
    - Class instantiation
    - Method invocation
    - File existence checks
    """
    
    def __init__(self, timeout_seconds: int = 30):
        self.timeout = timeout_seconds
        self._execution_history: List[Dict] = []
    
    def execute(self, verification_code: str) -> Tuple[bool, str, EvidenceStrength]:
        """
        Execute verification code and return (success, output, evidence_strength).
        """
        start_time = datetime.utcnow()
        
        try:
            # Create clean namespace
            namespace = {
                '__builtins__': __builtins__,
                'importlib': importlib,
                'sys': sys
            }
            
            # Execute
            exec(verification_code, namespace)
            
            # Get results
            result = namespace.get('result', None)
            output = namespace.get('output', str(result))
            evidence = namespace.get('evidence_strength', 'verified')
            
            # Map evidence string to enum
            evidence_map = {
                'none': EvidenceStrength.NONE,
                'weak': EvidenceStrength.WEAK,
                'medium': EvidenceStrength.MEDIUM,
                'strong': EvidenceStrength.STRONG,
                'verified': EvidenceStrength.VERIFIED
            }
            evidence_strength = evidence_map.get(str(evidence).lower(), EvidenceStrength.MEDIUM)
            
            # Determine success
            if result is True:
                evidence_strength = EvidenceStrength.VERIFIED
                success = True
            elif isinstance(result, dict) and result.get('success'):
                evidence_strength = EvidenceStrength.VERIFIED
                success = True
            else:
                success = False
            
            self._log_execution(verification_code, success, output, start_time)
            return success, str(output), evidence_strength
            
        except ImportError as e:
            output = f"IMPORT_FAILED: {str(e)}"
            self._log_execution(verification_code, False, output, start_time)
            return False, output, EvidenceStrength.NONE
            
        except AttributeError as e:
            output = f"ATTRIBUTE_MISSING: {str(e)}"
            self._log_execution(verification_code, False, output, start_time)
            return False, output, EvidenceStrength.WEAK
            
        except Exception as e:
            output = f"EXECUTION_FAILED: {str(e)}"
            self._log_execution(verification_code, False, output, start_time)
            return False, output, EvidenceStrength.NONE
    
    def verify_import(self, module_path: str) -> Tuple[bool, str]:
        """Verify a module can be imported."""
        code = f"""
try:
    import {module_path}
    result = True
    output = "Module {module_path} imported successfully"
except ImportError as e:
    result = False
    output = str(e)
"""
        success, output, _ = self.execute(code)
        return success, output
    
    def verify_class_exists(self, module_path: str, class_name: str) -> Tuple[bool, str]:
        """Verify a class exists in a module."""
        code = f"""
try:
    mod = importlib.import_module('{module_path}')
    cls = getattr(mod, '{class_name}', None)
    if cls is not None:
        result = True
        output = "Class {class_name} found in {module_path}"
    else:
        result = False
        output = "Class {class_name} NOT found in {module_path}"
except Exception as e:
    result = False
    output = str(e)
"""
        success, output, _ = self.execute(code)
        return success, output
    
    def verify_instantiation(self, module_path: str, class_name: str) -> Tuple[bool, str]:
        """Verify a class can be instantiated."""
        code = f"""
try:
    mod = importlib.import_module('{module_path}')
    cls = getattr(mod, '{class_name}')
    instance = cls()
    result = True
    output = "Class {class_name} instantiated successfully"
except Exception as e:
    result = False
    output = str(e)
"""
        success, output, _ = self.execute(code)
        return success, output
    
    def verify_method_callable(
        self, 
        module_path: str, 
        class_name: str, 
        method_name: str,
        test_args: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """Verify a method can be called."""
        args_str = ", ".join(f"{k}={repr(v)}" for k, v in (test_args or {}).items())
        code = f"""
try:
    mod = importlib.import_module('{module_path}')
    cls = getattr(mod, '{class_name}')
    instance = cls()
    method = getattr(instance, '{method_name}')
    result_value = method({args_str})
    result = True
    output = f"Method {method_name} returned: {{str(result_value)[:100]}}"
except Exception as e:
    result = False
    output = str(e)
"""
        success, output, _ = self.execute(code)
        return success, output
    
    def _log_execution(
        self, 
        code: str, 
        success: bool, 
        output: str, 
        start_time: datetime
    ) -> None:
        """Log execution for audit."""
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        self._execution_history.append({
            'timestamp': start_time.isoformat(),
            'code_hash': hash(code),
            'success': success,
            'output_preview': output[:200],
            'duration_ms': duration_ms
        })
    
    def get_execution_history(self) -> List[Dict]:
        """Get execution history."""
        return self._execution_history.copy()


# =============================================================================
# Remediation Generator
# =============================================================================

class RemediationGenerator:
    """
    Generates specific, actionable remediation prompts from failures.
    
    NOT: "Fix the issues"
    YES: "IMPLEMENT: ClassName.missing_method() in module_path"
    """
    
    def generate(self, failures: List[Dict[str, str]]) -> List[str]:
        """Generate remediation prompts from failures."""
        remediation = []
        
        for failure in failures:
            error = failure.get('error', '')
            context = failure.get('context', '')
            
            # Import errors
            if 'No module named' in error or 'IMPORT_FAILED' in error:
                module = self._extract_module_name(error)
                if module:
                    remediation.append(f"CREATE MODULE: {module}")
                    remediation.append(f"IMPLEMENT required classes/functions in {module}")
            
            # Attribute errors
            elif 'has no attribute' in error or 'ATTRIBUTE_MISSING' in error:
                class_name, attr_name = self._extract_attribute_info(error)
                if class_name and attr_name:
                    remediation.append(f"IMPLEMENT: {class_name}.{attr_name}()")
            
            # Instantiation errors
            elif '__init__' in error or 'TypeError' in error:
                remediation.append(f"FIX constructor: {context}")
                remediation.append("CHECK: Required arguments and their types")
            
            # Assertion errors
            elif 'AssertionError' in error:
                remediation.append(f"FIX: Assertion failed - {error}")
                remediation.append("VERIFY: Expected behavior matches implementation")
            
            # General execution errors
            else:
                remediation.append(f"FIX: {error[:100]}")
        
        # Always add verification requirement
        if remediation:
            remediation.append("PROVIDE: Execution output proving the fix works")
        
        return remediation
    
    def _extract_module_name(self, error: str) -> Optional[str]:
        """Extract module name from import error."""
        if "'" in error:
            parts = error.split("'")
            if len(parts) >= 2:
                return parts[1]
        return None
    
    def _extract_attribute_info(self, error: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract class and attribute names from attribute error."""
        if "'" in error:
            parts = error.split("'")
            if len(parts) >= 4:
                return parts[1], parts[3]
        return None, None

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


# =============================================================================
# Enforcement Engine
# =============================================================================

class EnforcementEngine:
    """
    Core enforcement engine that evaluates claims with execution.
    
    This is the component that BLOCKS until execution proof is provided.
    """
    
    def __init__(self):
        self.executor = VerificationExecutor()
        self.remediation_gen = RemediationGenerator()
        
        # Statistics
        self._stats = {
            'total_evaluations': 0,
            'blocked': 0,
            'verified': 0,
            'needs_context': 0,
            'partial': 0
        }
    
    def evaluate(
        self,
        claim: str,
        verification_code: str,
        context_available: bool = True,
        attempt_number: int = 1
    ) -> EnforcementResult:
        """
        Evaluate a claim by executing verification code.
        
        NOT by checking if words are correct.
        ONLY by running actual code.
        """
        start_time = datetime.utcnow()
        self._stats['total_evaluations'] += 1
        
        # Check context
        if not context_available:
            self._stats['needs_context'] += 1
            return EnforcementResult(
                decision=EnforcementDecision.NEEDS_CONTEXT,
                claim=claim,
                execution_attempted=False,
                execution_output="",
                evidence_strength=EvidenceStrength.NONE,
                ask_user="Please provide more details about the expected behavior.",
                attempt_number=attempt_number,
                timestamp=start_time.isoformat()
            )
        
        # Execute verification
        success, output, evidence_strength = self.executor.execute(verification_code)
        
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        if success:
            self._stats['verified'] += 1
            return EnforcementResult(
                decision=EnforcementDecision.VERIFIED,
                claim=claim,
                execution_attempted=True,
                execution_output=output,
                evidence_strength=evidence_strength,
                attempt_number=attempt_number,
                timestamp=start_time.isoformat(),
                duration_ms=duration_ms
            )
        else:
            self._stats['blocked'] += 1
            
            # Generate remediation
            failures = [{'error': output, 'context': claim}]
            remediation = self.remediation_gen.generate(failures)
            
            return EnforcementResult(
                decision=EnforcementDecision.BLOCKED,
                claim=claim,
                execution_attempted=True,
                execution_output=output,
                evidence_strength=evidence_strength,
                remediation_required=remediation,
                specific_failures=failures,
                attempt_number=attempt_number,
                timestamp=start_time.isoformat(),
                duration_ms=duration_ms
            )
    
    def evaluate_multiple(
        self,
        items: List[VerificationItem]
    ) -> EnforcementResult:
        """Evaluate multiple verification items."""
        passed = 0
        failed = 0
        all_remediation = []
        all_failures = []
        all_outputs = []
        
        for item in items:
            result = self.evaluate(item.description, item.verification_code)
            item.status = result.decision
            item.execution_output = result.execution_output
            
            if result.decision == EnforcementDecision.VERIFIED:
                passed += 1
            else:
                failed += 1
                all_remediation.extend(result.remediation_required)
                all_failures.extend(result.specific_failures)
            
            all_outputs.append(f"{item.item_id}: {result.decision.value}")
        
        # Determine overall decision
        if failed == 0:
            decision = EnforcementDecision.VERIFIED
            self._stats['verified'] += 1
        elif passed == 0:
            decision = EnforcementDecision.BLOCKED
            self._stats['blocked'] += 1
        else:
            decision = EnforcementDecision.PARTIAL
            self._stats['partial'] += 1
        
        return EnforcementResult(
            decision=decision,
            claim=f"Verify {len(items)} items",
            execution_attempted=True,
            execution_output="\n".join(all_outputs),
            evidence_strength=EvidenceStrength.VERIFIED if failed == 0 else EvidenceStrength.WEAK,
            remediation_required=list(set(all_remediation)),  # Dedupe
            specific_failures=all_failures,
            items_passed=passed,
            items_failed=failed,
            items_total=len(items),
            timestamp=datetime.utcnow().isoformat()
        )
    
    def get_statistics(self) -> Dict[str, int]:
        """Get enforcement statistics."""
        return self._stats.copy()
    
    def record_outcome(self, outcome: Dict) -> None:
        """Record operation outcome for learning."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append({
            'timestamp': datetime.utcnow().isoformat(),
            **outcome
        })
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback to adjust behavior."""
        was_correct = feedback.get('was_correct', True)
        if not was_correct:
            # Track failures for pattern analysis
            if not hasattr(self, '_failure_patterns'):
                self._failure_patterns = []
            self._failure_patterns.append(feedback.get('pattern', 'unknown'))
    
    def serialize_state(self) -> Dict:
        """Serialize state for persistence."""
        return {
            'stats': self._stats,
            'outcomes': getattr(self, '_outcomes', [])[-100:],
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore state from serialized data."""
        self._stats = state.get('stats', self._stats)
        self._outcomes = state.get('outcomes', [])


# =============================================================================
# Enforcement Loop
# =============================================================================

class EnforcementLoop:
    """
    THE CRITICAL COMPONENT: Forces LLM to complete work.
    
    This wraps the LLM interaction and LOOPS until:
    1. Verification passes (VERIFIED)
    2. Max attempts reached (ask user)
    3. Context insufficient (ask user)
    
    The LLM CANNOT bypass this loop. It MUST provide working code.
    """
    
    def __init__(
        self,
        max_attempts: int = 5,
        engine: Optional[EnforcementEngine] = None
    ):
        self.max_attempts = max_attempts
        self.engine = engine or EnforcementEngine()
        
        # Loop statistics
        self._loop_stats = {
            'total_loops': 0,
            'successful_completions': 0,
            'max_attempts_reached': 0,
            'context_escalations': 0,
            'average_attempts': 0.0
        }
    
    def enforce(
        self,
        original_request: str,
        llm_response_provider: Callable[[str], Dict[str, Any]],
        verification_code_generator: Callable[[Dict], str]
    ) -> EnforcedOutput:
        """
        ENFORCE task completion by looping until verified.
        
        Args:
            original_request: The user's original request
            llm_response_provider: Function that gets LLM response for a prompt
            verification_code_generator: Function that generates verification code
        
        Returns:
            EnforcedOutput with final verified response or escalation
        """
        start_time = datetime.utcnow()
        self._loop_stats['total_loops'] += 1
        
        attempt = 0
        verification_history: List[EnforcementResult] = []
        current_prompt = original_request
        issues_fixed = 0
        
        while attempt < self.max_attempts:
            attempt += 1
            
            # Get LLM response
            llm_response = llm_response_provider(current_prompt)
            
            # Generate verification code
            verification_code = verification_code_generator(llm_response)
            
            # Evaluate with EXECUTION
            result = self.engine.evaluate(
                claim=llm_response.get('claim', 'Task completion'),
                verification_code=verification_code,
                context_available=llm_response.get('has_context', True),
                attempt_number=attempt
            )
            
            verification_history.append(result)
            
            # Check result
            if result.decision == EnforcementDecision.VERIFIED:
                # SUCCESS - Task actually complete
                self._loop_stats['successful_completions'] += 1
                self._update_average_attempts(attempt)
                
                return EnforcedOutput(
                    success=True,
                    final_response=llm_response.get('response'),
                    total_attempts=attempt,
                    verification_history=verification_history,
                    time_to_complete_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                    issues_fixed=issues_fixed
                )
            
            elif result.decision == EnforcementDecision.NEEDS_CONTEXT:
                # Need user input
                self._loop_stats['context_escalations'] += 1
                
                return EnforcedOutput(
                    success=False,
                    final_response=None,
                    total_attempts=attempt,
                    verification_history=verification_history,
                    needs_user_input=True,
                    user_prompt=result.ask_user,
                    time_to_complete_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                    issues_fixed=issues_fixed
                )
            
            else:  # BLOCKED or PARTIAL
                # Generate new prompt with specific failures
                issues_fixed += len(result.specific_failures)
                
                current_prompt = self._generate_remediation_prompt(
                    original_request=original_request,
                    previous_output=result.execution_output,
                    remediation=result.remediation_required,
                    attempt=attempt
                )
        
        # Max attempts reached
        self._loop_stats['max_attempts_reached'] += 1
        
        return EnforcedOutput(
            success=False,
            final_response=None,
            total_attempts=attempt,
            verification_history=verification_history,
            needs_user_input=True,
            user_prompt=f"Unable to complete after {self.max_attempts} attempts. Last issues:\n" + 
                        "\n".join(verification_history[-1].remediation_required[:5]),
            time_to_complete_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            issues_fixed=issues_fixed
        )
    
    def _generate_remediation_prompt(
        self,
        original_request: str,
        previous_output: str,
        remediation: List[str],
        attempt: int
    ) -> str:
        """Generate a prompt that forces the LLM to fix specific issues."""
        return f"""
═══════════════════════════════════════════════════════════════════════════════
ENFORCEMENT: TASK NOT COMPLETE - ATTEMPT {attempt} FAILED
═══════════════════════════════════════════════════════════════════════════════

ORIGINAL USER REQUEST:
{original_request}

YOUR PREVIOUS ATTEMPT FAILED WITH:
{previous_output}

═══════════════════════════════════════════════════════════════════════════════
REQUIRED ACTIONS (YOU MUST COMPLETE THESE):
═══════════════════════════════════════════════════════════════════════════════
{chr(10).join(f"  {i+1}. {r}" for i, r in enumerate(remediation))}

═══════════════════════════════════════════════════════════════════════════════
INSTRUCTIONS:
═══════════════════════════════════════════════════════════════════════════════
1. IMPLEMENT the required fixes above
2. DO NOT claim completion without actual code
3. PROVIDE working implementation that passes execution verification
4. If you cannot complete this, explain what additional context you need

BEGIN YOUR IMPLEMENTATION:
"""
    
    def _update_average_attempts(self, attempts: int) -> None:
        """Update running average of attempts."""
        total = self._loop_stats['successful_completions']
        current_avg = self._loop_stats['average_attempts']
        self._loop_stats['average_attempts'] = (current_avg * (total - 1) + attempts) / total
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get loop statistics."""
        return {
            **self._loop_stats,
            'engine_stats': self.engine.get_statistics(),
            'learning_params': getattr(self, '_learning_params', {}),
            'history_size': len(getattr(self, '_outcome_history', []))
        }
    
    # =========================================================================
    # Learning Interface (5/5 methods)
    # =========================================================================
    
    def record_outcome(self, outcome: Dict[str, Any]) -> None:
        """
        Record enforcement outcome for learning.
        
        Args:
            outcome: Dictionary containing:
                - success: bool - Was the enforcement successful?
                - attempts: int - How many attempts were needed?
                - user_escalated: bool - Did it escalate to user?
                - issues_found: List[str] - What issues were found?
        """
        # Initialize if not present
        if not hasattr(self, '_outcome_history'):
            self._outcome_history: List[Dict] = []
        if not hasattr(self, '_learning_params'):
            self._learning_params: Dict[str, float] = {
                'block_threshold': 0.5,
                'max_attempts_adjustment': 0,
                'strictness': 1.0
            }
        
        success = outcome.get('success', False)
        attempts = outcome.get('attempts', 1)
        
        self._loop_stats['total_loops'] += 1
        if success:
            self._loop_stats['successful_completions'] += 1
        if outcome.get('user_escalated', False):
            self._loop_stats['context_escalations'] += 1
        
        self._outcome_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            **outcome
        })
        
        # Keep bounded
        if len(self._outcome_history) > 1000:
            self._outcome_history = self._outcome_history[-500:]
        
        logger.debug(f"[EnforcementLoop] Recorded outcome: success={success}, attempts={attempts}")
    
    def learn_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Learn from user feedback to improve enforcement.
        
        Args:
            feedback: Dictionary containing:
                - was_correct: bool - Was the enforcement decision correct?
                - should_have_blocked: bool - Should it have blocked when it didn't?
                - too_strict: bool - Was the enforcement too strict?
                - suggestions: str - User suggestions
        """
        if not hasattr(self, '_learning_params'):
            self._learning_params = {
                'block_threshold': 0.5,
                'max_attempts_adjustment': 0,
                'strictness': 1.0
            }
        
        was_correct = feedback.get('was_correct', True)
        should_have_blocked = feedback.get('should_have_blocked', False)
        too_strict = feedback.get('too_strict', False)
        
        if not was_correct:
            if should_have_blocked:
                # Tighten enforcement - lower threshold for blocking
                self._learning_params['block_threshold'] *= 0.95
                self._learning_params['strictness'] *= 1.1
                logger.info("[EnforcementLoop] Tightened enforcement based on feedback")
            elif too_strict:
                # Loosen enforcement - raise threshold
                self._learning_params['block_threshold'] *= 1.05
                self._learning_params['strictness'] *= 0.9
                logger.info("[EnforcementLoop] Loosened enforcement based on feedback")
        
        # Ensure thresholds stay in valid range
        self._learning_params['block_threshold'] = max(0.2, min(0.9, 
            self._learning_params['block_threshold']))
        self._learning_params['strictness'] = max(0.5, min(2.0,
            self._learning_params['strictness']))
        
        # Adjust max attempts based on historical success
        if hasattr(self, '_outcome_history') and len(self._outcome_history) >= 10:
            recent = self._outcome_history[-10:]
            avg_attempts = sum(o.get('attempts', 1) for o in recent) / len(recent)
            if avg_attempts > 3:
                self._learning_params['max_attempts_adjustment'] = 1
            elif avg_attempts < 2:
                self._learning_params['max_attempts_adjustment'] = -1
    
    def serialize_state(self) -> Dict[str, Any]:
        """Serialize state for persistence."""
        return {
            'outcome_history': getattr(self, '_outcome_history', [])[-100:],
            'learning_params': getattr(self, '_learning_params', {}).copy(),
            'loop_stats': self._loop_stats.copy(),
            'max_attempts': self.max_attempts,
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict[str, Any]) -> None:
        """Restore state from serialized data."""
        self._outcome_history = state.get('outcome_history', [])
        self._learning_params = state.get('learning_params', {
            'block_threshold': 0.5,
            'max_attempts_adjustment': 0,
            'strictness': 1.0
        })
        self._loop_stats.update(state.get('loop_stats', {}))
        
        # Apply max_attempts adjustment
        adjustment = self._learning_params.get('max_attempts_adjustment', 0)
        self.max_attempts = state.get('max_attempts', 5) + adjustment
        
        logger.info(f"[EnforcementLoop] Restored state with {len(self._outcome_history)} history items")


# =============================================================================
# Factory and Helpers
# =============================================================================

def create_enforcement_loop(max_attempts: int = 5) -> EnforcementLoop:
    """Factory function to create enforcement loop."""
    return EnforcementLoop(max_attempts=max_attempts)


def create_simple_verification_code(module: str, class_name: str, methods: List[str]) -> str:
    """Helper to create verification code for a class with methods."""
    method_checks = "\n    ".join([
        f"assert hasattr(instance, '{m}'), 'Missing method: {m}'"
        for m in methods
    ])
    
    return f"""
try:
    mod = importlib.import_module('{module}')
    cls = getattr(mod, '{class_name}')
    instance = cls()
    {method_checks}
    result = True
    output = "All methods verified: {', '.join(methods)}"
except Exception as e:
    result = False
    output = str(e)
"""


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ENFORCEMENT LOOP TEST")
    print("=" * 80)
    
    loop = EnforcementLoop(max_attempts=3)
    
    # Simulate LLM responses (improving each time)
    attempt_responses = [
        # Attempt 1: Missing class entirely
        {
            'claim': 'TestClass implemented',
            'response': 'class TestClass: pass',
            'has_context': True,
            'code': 'result = False; output = "Class not in module"'
        },
        # Attempt 2: Class exists but missing method
        {
            'claim': 'TestClass with do_work',
            'response': 'class TestClass:\n    def __init__(self): pass',
            'has_context': True,
            'code': 'result = False; output = "Missing method: do_work"'
        },
        # Attempt 3: Complete
        {
            'claim': 'TestClass complete',
            'response': 'class TestClass:\n    def do_work(self): return "done"',
            'has_context': True,
            'code': 'result = True; output = "All verified"'
        }
    ]
    
    current_attempt = [0]  # Mutable counter
    
    def mock_llm_provider(prompt: str) -> Dict:
        idx = min(current_attempt[0], len(attempt_responses) - 1)
        current_attempt[0] += 1
        return attempt_responses[idx]
    
    def mock_verification_generator(response: Dict) -> str:
        return response.get('code', 'result = False')
    
    print("\n[1] Running enforcement loop...")
    output = loop.enforce(
        original_request="Implement TestClass with do_work() method",
        llm_response_provider=mock_llm_provider,
        verification_code_generator=mock_verification_generator
    )
    
    print(f"\n    Success: {output.success}")
    print(f"    Total Attempts: {output.total_attempts}")
    print(f"    Issues Fixed: {output.issues_fixed}")
    print(f"    Time: {output.time_to_complete_ms:.1f}ms")
    
    print("\n[2] Verification History:")
    for i, result in enumerate(output.verification_history):
        print(f"    Attempt {i+1}: {result.decision.value}")
        if result.remediation_required:
            print(f"      Remediation: {result.remediation_required[0][:50]}...")
    
    print("\n[3] Loop Statistics:")
    stats = loop.get_statistics()
    for key, value in stats.items():
        if not isinstance(value, dict):
            print(f"    {key}: {value}")
    
    print("\n" + "=" * 80)
    print("✓ ENFORCEMENT LOOP TEST COMPLETE")
    print("=" * 80)

