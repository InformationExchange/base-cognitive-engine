"""
BAIS Corrective Action System

1. Detects incomplete claims or missing proof
2. Instructs LLM to provide proof or complete work
3. Verifies corrected output
4. Iterates until threshold met or max attempts
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Awaitable
from datetime import datetime
from enum import Enum

logger = logging.getLogger("BAIS.CorrectiveAction")


class ActionType(Enum):
    """Types of corrective actions."""
    REGENERATE = "regenerate"           # Re-query LLM with guidance
    ADD_DISCLAIMER = "add_disclaimer"   # Add safety/legal disclaimers
    REMOVE_CLAIMS = "remove_claims"     # Remove manipulation language
    FIX_LOGIC = "fix_logic"             # Correct logical fallacies
    COMPLETE_CODE = "complete_code"     # Complete unfinished code
    FIX_TESTS = "fix_tests"             # Complete tests
    CREATE_FILES = "create_files"       # Create claimed files
    CORRECT_CLAIM = "correct_claim"     # Retract false claim
    PROVIDE_PROOF = "provide_proof"     # Provide proof of completion


@dataclass
class CorrectiveAction:
    """A specific corrective action to take."""
    action_type: ActionType
    description: str
    guidance: str  # Specific instructions for the LLM
    priority: int  # 1=highest, 5=lowest
    evidence: str = ""  # What triggered this action
    
    def to_dict(self) -> Dict:
        return {
            "type": self.action_type.value,
            "description": self.description,
            "guidance": self.guidance,
            "priority": self.priority,
            "evidence": self.evidence
        }


@dataclass
class CorrectionResult:
    """Result of applying corrective actions."""
    success: bool
    original_response: str
    corrected_response: str
    actions_taken: List[CorrectiveAction]
    iterations: int
    final_score: float
    improvement: float  # How much better (percentage)
    audit_trail: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "corrected_response": self.corrected_response,
            "actions_count": len(self.actions_taken),
            "iterations": self.iterations,
            "final_score": self.final_score,
            "improvement": self.improvement,
            "actions": [a.to_dict() for a in self.actions_taken]
        }


class CorrectiveActionEngine:
    """
    Corrective action engine.
    
    Detects issues, instructs LLM to provide proof or complete work.
    """
    
    # Issue patterns mapped to corrective actions
    ISSUE_TO_ACTION = {
        "TGTBT": [
            CorrectiveAction(
                action_type=ActionType.PROVIDE_PROOF,
                description="Provide proof or report incomplete",
                guidance="""Completion claim detected. Required: proof or status update.
1. Provide evidence (test output, file contents, verification results)
2. If incomplete, report: STATUS: INCOMPLETE, REASON: [reason], REMAINING: [items]
""",
                priority=1
            )
        ],
        "LOGICAL_FALLACY": [
            CorrectiveAction(
                action_type=ActionType.FIX_LOGIC,
                description="Correct logical errors",
                guidance="""Fix logical fallacies:
- False dichotomy: Acknowledge other options exist
- Hasty generalization: Add qualifiers like "in some cases"
- Appeal to authority: Focus on evidence, not who said it
- Straw man: Address the actual argument
""",
                priority=1
            )
        ],
        "INCOMPLETE_CODE": [
            CorrectiveAction(
                action_type=ActionType.COMPLETE_CODE,
                description="Complete unfinished implementation",
                guidance="""Complete the code:
1. Implement all TODO comments
2. Remove placeholder/mock implementations
3. Add error handling
4. Ensure all functions have actual implementations
5. Remove 'pass' statements with real logic
""",
                priority=1
            )
        ],
        "INCOMPLETE_TEST": [
            CorrectiveAction(
                action_type=ActionType.FIX_TESTS,
                description="Complete test implementation",
                guidance="""Complete the tests:
1. Remove 'assertTrue(True)' placeholder assertions
2. Add actual test logic
3. Test edge cases
4. Verify both positive and negative scenarios
5. Add assertions that verify actual behavior
""",
                priority=1
            )
        ],
        "MANIPULATION": [
            CorrectiveAction(
                action_type=ActionType.REMOVE_CLAIMS,
                description="Remove manipulative language",
                guidance="""Remove manipulation:
- Remove urgency language ("act now", "limited time")
- Remove fear appeals ("you'll regret it")
- Remove false scarcity ("only X left")
- Replace with neutral, factual statements
""",
                priority=1
            )
        ],
        "MISSING_DISCLAIMER": [
            CorrectiveAction(
                action_type=ActionType.ADD_DISCLAIMER,
                description="Add appropriate disclaimers",
                guidance="Add domain-appropriate disclaimers about seeking professional advice (medical, legal, financial).",
                priority=2
            )
        ],
        "FILE_NOT_FOUND": [
            CorrectiveAction(
                action_type=ActionType.CREATE_FILES,
                description="Create files or update status",
                guidance="""Files claimed do not exist.
1. Create the files with implementation, OR
2. Report: STATUS: INCOMPLETE, FILES_MISSING: [list]
""",
                priority=1
            )
        ],
        "PROOF_GAP": [
            CorrectiveAction(
                action_type=ActionType.PROVIDE_PROOF,
                description="Provide evidence",
                guidance="""Claims lack verification.
Provide: file contents, test output, or verification results.
If unavailable, report: STATUS: INCOMPLETE, REASON: [reason]
""",
                priority=1
            )
        ],
        "FALSE_COMPLETION": [
            CorrectiveAction(
                action_type=ActionType.CORRECT_CLAIM,
                description="Update completion status",
                guidance="""Completion claim unverified.
Report actual status:
- COMPLETED: [items with evidence]
- REMAINING: [items not done]
""",
                priority=1
            )
        ]
    }
    
    # Code-specific patterns that indicate incompleteness
    CODE_INCOMPLETE_PATTERNS = [
        ("TODO", "Contains TODO comment - code incomplete"),
        ("FIXME", "Contains FIXME comment - code needs fixing"),
        ("pass  # ", "Contains placeholder 'pass' statement"),
        ("NotImplementedError", "Raises NotImplementedError - not implemented"),
        ("raise NotImplemented", "Function not implemented"),
        ("PLACEHOLDER", "Contains PLACEHOLDER marker"),
        ("SAMPLE_", "Contains sample/mock implementation"),
        ("assertTrue(True)", "Test has placeholder assertion"),
        ("assert True", "Test has placeholder assertion"),
        ("# stub", "Contains stub implementation"),
        ("mock_", "Contains mock that should be real"),
    ]
    
    def __init__(
        self, 
        llm_caller: Optional[Callable[[str], Awaitable[str]]] = None,
        evaluator: Optional[Callable[[str, str], Awaitable[Dict]]] = None,
        max_iterations: int = 3,
        min_score_threshold: float = 65.0
    ):
        """
        Initialize the corrective action engine.
        
        Args:
            llm_caller: Async function to call LLM for regeneration
            evaluator: Async function to evaluate response quality
            max_iterations: Maximum correction attempts
            min_score_threshold: Minimum acceptable score
        """
        self.llm_caller = llm_caller
        self.evaluator = evaluator
        self.max_iterations = max_iterations
        self.min_score_threshold = min_score_threshold
        
    async def correct(
        self,
        query: str,
        response: str,
        issues: List[str],
        original_score: float = 50.0
    ) -> CorrectionResult:
        """
        Apply corrective actions to improve a response.
        
        FLOW:
        1. Detect issue (e.g., TGTBT - overconfident claim)
        2. Demand PROOF from LLM
        3. Use LLM to validate proof
        4. If proof valid → Accept
        5. If proof invalid → Force LLM to DO THE WORK
        6. If cannot complete → Report reason, mark INCOMPLETE
        
        NO WORDING CHANGES. NO HEDGING. PROVE IT OR DO IT.
        
        Args:
            query: Original user query
            response: Response to correct
            issues: List of detected issues (strings)
            original_score: Initial quality score
            
        Returns:
            CorrectionResult with corrected response
        """
        audit_trail = []
        actions_taken = []
        current_response = response
        current_score = original_score
        
        # Step 1: Determine corrective actions needed
        actions = self._determine_actions(response, issues)
        
        audit_trail.append({
            "step": "analysis",
            "issues_count": len(issues),
            "actions_planned": len(actions),
            "timestamp": datetime.now().isoformat()
        })
        
        if not actions:
            return CorrectionResult(
                success=True,
                original_response=response,
                corrected_response=response,
                actions_taken=[],
                iterations=0,
                final_score=original_score,
                improvement=0.0,
                audit_trail=audit_trail
            )
        
        # Step 2: Apply corrections iteratively
        for iteration in range(self.max_iterations):
            # Build correction prompt
            correction_prompt = self._build_correction_prompt(
                query, current_response, actions
            )
            
            # Get corrected response from LLM
            if self.llm_caller:
                try:
                    corrected = await self.llm_caller(correction_prompt)
                    if corrected and corrected.strip():
                        current_response = corrected
                        actions_taken.extend(actions[:2])  # Record top 2 actions
                except Exception as e:
                    logger.warning(f"LLM correction failed: {e}")
                    corrected = self._apply_pattern_corrections(current_response, actions)
                    if corrected:
                        current_response = corrected
            else:
                # Pattern-based correction only
                corrected = self._apply_pattern_corrections(current_response, actions)
                if corrected:
                    current_response = corrected
                    actions_taken.extend(actions[:2])
            
            # Evaluate corrected response
            if self.evaluator:
                try:
                    eval_result = await self.evaluator(query, current_response)
                    raw_score = eval_result.get('accuracy', current_score)
                    # Ensure score is in 0-100 range
                    if raw_score <= 1.0:
                        raw_score *= 100  # Convert from 0-1 to 0-100
                    current_score = max(0, min(100, raw_score))  # Clamp to 0-100
                except Exception as e:
                    logger.warning(f"Evaluation failed: {e}")
            
            audit_trail.append({
                "step": f"iteration_{iteration + 1}",
                "score": current_score,
                "response_length": len(current_response),
                "timestamp": datetime.now().isoformat()
            })
            
            # Check if threshold met
            if current_score >= self.min_score_threshold:
                break
            
            # Re-analyze for remaining issues
            remaining_issues = self._detect_remaining_issues(current_response)
            if not remaining_issues:
                break
            
            actions = self._determine_actions(current_response, remaining_issues)
        
        # Calculate improvement as absolute percentage points gained
        improvement = current_score - original_score
        
        # Success ONLY if:
        # 1. Score meets minimum threshold (65%), OR
        # 2. Significant improvement (>=20 pts) AND score is reasonable (>30%)
        # This prevents claiming success when fundamental issues remain
        success = (
            current_score >= self.min_score_threshold or 
            (improvement >= 20.0 and current_score > 30.0)
        )
        
        return CorrectionResult(
            success=success,
            original_response=response,
            corrected_response=current_response,
            actions_taken=actions_taken,
            iterations=iteration + 1 if 'iteration' in dir() else 1,
            final_score=current_score,
            improvement=improvement,  # Now represents percentage points gained
            audit_trail=audit_trail
        )
    
    def _determine_actions(
        self, 
        response: str, 
        issues: List[str]
    ) -> List[CorrectiveAction]:
        """Determine which corrective actions to take."""
        actions = []
        missing_files = []
        
        # Map issues to actions
        for issue in issues:
            issue_upper = issue.upper()
            
            # Extract specific file names from FILE_NOT_FOUND errors
            if "FILE_NOT_FOUND" in issue_upper:
                # Extract file path from issue string like "FILE_NOT_FOUND: src/core/fake.py"
                parts = issue.split("FILE_NOT_FOUND:")
                if len(parts) > 1:
                    file_path = parts[1].split(" - ")[0].strip()
                    missing_files.append(file_path)
            
            for pattern, action_list in self.ISSUE_TO_ACTION.items():
                if pattern in issue_upper:
                    # Clone actions and add specific evidence
                    for action in action_list:
                        new_action = CorrectiveAction(
                            action_type=action.action_type,
                            description=action.description,
                            guidance=action.guidance,
                            priority=action.priority,
                            evidence=issue  # Capture the specific issue as evidence
                        )
                        actions.append(new_action)
        
        # If we found missing files, add a specific action with all file names
        if missing_files:
            actions.insert(0, CorrectiveAction(
                action_type=ActionType.CREATE_FILES,
                description=f"Create {len(missing_files)} missing files",
                guidance=f"""The following files were claimed to exist but DO NOT EXIST:
{chr(10).join('- ' + f for f in missing_files)}

You MUST either:
1. Provide the ACTUAL CONTENT of each file (so they can be created), OR
2. Acknowledge these files do not exist and need to be created

Do NOT claim files exist if they don't.""",
                priority=1,
                evidence=f"FILE_NOT_FOUND: {', '.join(missing_files)}"
            ))
        
        # Check for code-specific issues
        for pattern, description in self.CODE_INCOMPLETE_PATTERNS:
            if pattern in response:
                actions.append(CorrectiveAction(
                    action_type=ActionType.COMPLETE_CODE,
                    description=description,
                    guidance=f"Remove or replace '{pattern}' with actual implementation",
                    priority=1,
                    evidence=pattern
                ))
        
        # Sort by priority and deduplicate
        seen = set()
        unique_actions = []
        for action in sorted(actions, key=lambda a: a.priority):
            key = (action.action_type, action.description)
            if key not in seen:
                seen.add(key)
                unique_actions.append(action)
        
        return unique_actions[:5]  # Top 5 actions
    
    def _build_correction_prompt(
        self,
        query: str,
        response: str,
        actions: List[CorrectiveAction]
    ) -> str:
        """Build a prompt to guide LLM correction."""
        
        # Check if this is a proof-based issue
        has_file_issues = any(a.action_type in [ActionType.CREATE_FILES, ActionType.CORRECT_CLAIM] for a in actions)
        has_proof_issues = any(a.action_type == ActionType.PROVIDE_PROOF for a in actions)
        
        actions_text = "\n".join([
            f"{i+1}. [{a.action_type.value}] {a.description}\n   Guidance: {a.guidance}"
            for i, a in enumerate(actions[:5])
        ])
        
        # Extract missing files from evidence if present
        missing_files = [a.evidence for a in actions if a.evidence and 'FILE_NOT_FOUND' in str(a.evidence)]
        
        if has_file_issues or has_proof_issues:
            missing_files_text = "\n".join(f"  - {f}" for f in missing_files) if missing_files else ""
            
            return f"""CORRECTION REQUIRED

TASK: {query}

RESPONSE:
{response}

ISSUES:
{actions_text}
{f"MISSING FILES:{chr(10)}{missing_files_text}" if missing_files_text else ""}

REQUIRED ACTION (select one):

1. PROVIDE PROOF
   - File contents
   - Test output
   - Verification results

2. COMPLETE THE WORK
   - Execute the task
   - Provide deliverables

3. REPORT STATUS
   - STATUS: INCOMPLETE
   - REASON: [specific reason]
   - REMAINING: [specific items]

RESPONSE:"""
        else:
            # Standard correction prompt for non-proof issues
            return f"""Please correct the following response. Apply these specific corrections:

ORIGINAL QUERY: {query}

RESPONSE TO CORRECT:
{response}

REQUIRED CORRECTIONS:
{actions_text}

IMPORTANT:
- Apply ALL corrections listed above
- Maintain the helpful content while fixing issues
- Do not add new problems
- Be factually accurate and appropriately uncertain
- If you cannot verify a claim, hedge it or remove it

CORRECTED RESPONSE:"""
    
    def _apply_pattern_corrections(
        self,
        response: str,
        actions: List[CorrectiveAction]
    ) -> str:
        """Pattern-based corrections without LLM. Limited to manipulation removal."""
        corrected = response
        
        for action in actions:
            if action.action_type == ActionType.REMOVE_CLAIMS:
                manipulation_phrases = [
                    "act now", "limited time", "don't wait",
                    "last chance", "you'll regret", "or else"
                ]
                for phrase in manipulation_phrases:
                    corrected = corrected.replace(phrase, "")
                    corrected = corrected.replace(phrase.capitalize(), "")
        
        return corrected
    
    def _detect_remaining_issues(self, response: str) -> List[str]:
        """Detect issues remaining after correction."""
        issues = []
        
        # Check for TGTBT patterns
        tgtbt_patterns = ["100%", "fully", "guaranteed", "zero bugs", "always works"]
        for pattern in tgtbt_patterns:
            if pattern.lower() in response.lower():
                issues.append(f"TGTBT: Still contains '{pattern}'")
        
        # Check for incomplete code
        for pattern, desc in self.CODE_INCOMPLETE_PATTERNS:
            if pattern in response:
                issues.append(f"INCOMPLETE_CODE: {desc}")
        
        return issues

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


class CorrectiveActionTrigger:
    """
    Triggers corrective actions based on BAIS evaluation results.
    
    This component connects BAIS detection to actual correction,
    ensuring that identified issues are not just flagged but FIXED.
    """
    
    def __init__(
        self,
        corrective_engine: CorrectiveActionEngine,
        auto_correct: bool = True,
        min_score_for_auto: float = 60.0,
        max_corrections: int = 3
    ):
        self.engine = corrective_engine
        self.auto_correct = auto_correct
        self.min_score_for_auto = min_score_for_auto
        self.max_corrections = max_corrections
        self.correction_history = []
        
    async def evaluate_and_correct(
        self,
        query: str,
        response: str,
        evaluation_result: Dict
    ) -> Dict:
        """
        Evaluate a response and automatically correct if needed.
        
        Returns:
            Dict with:
            - original_response
            - final_response (corrected if needed)
            - was_corrected
            - correction_result (if corrected)
            - decision: 'approved', 'corrected', 'blocked'
        """
        accuracy = evaluation_result.get('accuracy', 100)
        issues = evaluation_result.get('issues', [])
        warnings = evaluation_result.get('warnings', [])
        
        # Determine if correction needed
        needs_correction = (
            accuracy < self.min_score_for_auto or
            len(issues) > 0 or
            any('TGTBT' in str(w) for w in warnings) or
            any('INCOMPLETE' in str(w) for w in warnings)
        )
        
        if not needs_correction:
            return {
                "original_response": response,
                "final_response": response,
                "was_corrected": False,
                "correction_result": None,
                "decision": "approved"
            }
        
        if not self.auto_correct:
            return {
                "original_response": response,
                "final_response": response,
                "was_corrected": False,
                "correction_result": None,
                "decision": "needs_review",
                "issues_found": issues + warnings
            }
        
        # Apply corrections
        all_issues = issues + [str(w) for w in warnings]
        correction_result = await self.engine.correct(
            query=query,
            response=response,
            issues=all_issues,
            original_score=accuracy
        )
        
        # Record in history
        self.correction_history.append({
            "timestamp": datetime.now().isoformat(),
            "original_score": accuracy,
            "final_score": correction_result.final_score,
            "success": correction_result.success,
            "iterations": correction_result.iterations
        })
        
        decision = "corrected" if correction_result.success else "blocked"
        
        return {
            "original_response": response,
            "final_response": correction_result.corrected_response,
            "was_corrected": True,
            "correction_result": correction_result.to_dict(),
            "decision": decision,
            "improvement": correction_result.improvement
        }
    
    def get_statistics(self) -> Dict:
        """Get correction statistics."""
        if not self.correction_history:
            return {"total_corrections": 0}
        
        total = len(self.correction_history)
        successful = sum(1 for h in self.correction_history if h['success'])
        avg_improvement = sum(
            h['final_score'] - h['original_score']
            for h in self.correction_history
        ) / total
        
        return {
            "total_corrections": total,
            "successful_corrections": successful,
            "success_rate": successful / total * 100,
            "avg_score_improvement": avg_improvement
        }

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

