"""
BAIS LLM Challenger - Adversarial Analysis for Claim Verification

Instead of using LLM as a passive judge ("Is this good?"), 
this module uses LLM as an active CHALLENGER:
- "Find problems with this"
- "What would make this claim false?"
- "What evidence would prove this?"

This adversarial approach overcomes LLM judge bias by forcing
active flaw-seeking rather than passive approval.

Patent Alignment:
- NOVEL-14: Neuro-Symbolic Reasoning (logical verification)
- NOVEL-13: Theory of Mind (understanding intent)
- PPA1-Inv23: AI Common Sense via Multi-Source Triangulation
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pathlib import Path
import json
import asyncio


class ChallengeType(Enum):
    """Types of challenges to pose to an LLM response."""
    EVIDENCE_DEMAND = "evidence_demand"   # What proof would verify this?
    DEVILS_ADVOCATE = "devils_advocate"   # How could this be wrong?
    EDGE_CASE = "edge_case"               # What edge cases might fail?
    CONTRADICTION = "contradiction"        # What contradicts this?
    COMPLETENESS = "completeness"          # What's missing?
    ASSUMPTION = "assumption"              # What assumptions are made?


@dataclass
class Challenge:
    """A specific challenge posed to an LLM response."""
    challenge_id: str
    challenge_type: ChallengeType
    prompt: str
    target_claim: str
    domain: str


@dataclass
class ChallengeResult:
    """Result of a challenge analysis."""
    challenge_id: str
    issues_found: List[str]
    evidence_demanded: List[str]
    risk_factors: List[str]
    confidence_reduction: float  # How much to reduce confidence
    recommendation: str
    llm_reasoning: str


@dataclass
class ChallengerVerdict:
    """Final verdict from the challenger."""
    claim: str
    challenges_executed: int
    total_issues: int
    total_evidence_gaps: int
    adjusted_confidence: float
    should_accept: bool
    should_remediate: bool
    remediation_guidance: List[str]
    inventions_used: List[str]


class LLMChallenger:
    """
    Uses LLM as an adversarial challenger rather than passive judge.
    
    Key insight: Asking "Is this good?" leads to confirmation bias.
    Asking "Find problems with this" forces active flaw-seeking.
    """
    
    # Challenge prompts for different types
    CHALLENGE_PROMPTS = {
        ChallengeType.EVIDENCE_DEMAND: """
You are a rigorous evidence auditor. Analyze this claim and response.

CLAIM: {claim}
RESPONSE: {response}
DOMAIN: {domain}

YOUR TASK: Identify what EVIDENCE would be needed to verify this claim is true.

For each assertion, specify:
1. What specific evidence would prove it?
2. What format should that evidence take?
3. Is that evidence present in the response?
4. If not present, rate severity (1-5) of this evidence gap

OUTPUT FORMAT (JSON):
{{
    "evidence_gaps": [
        {{"assertion": "...", "required_evidence": "...", "present": false, "severity": 4}}
    ],
    "overall_evidence_score": 0-100,
    "recommendation": "ACCEPT/DEMAND_EVIDENCE/REJECT"
}}
""",
        
        ChallengeType.DEVILS_ADVOCATE: """
You are a devil's advocate. Your job is to find ways this response could be WRONG.

CLAIM: {claim}
RESPONSE: {response}
DOMAIN: {domain}

YOUR TASK: Actively look for flaws, errors, and problems.

Consider:
1. What assumptions does this make that might be false?
2. What scenarios would make this claim incorrect?
3. What information is missing that could change the conclusion?
4. What logical fallacies might be present?
5. What biases might have influenced this response?

OUTPUT FORMAT (JSON):
{{
    "potential_flaws": [
        {{"flaw": "...", "severity": 1-5, "likelihood": 1-5}}
    ],
    "overall_risk_score": 0-100,
    "most_concerning_issue": "..."
}}
""",

        ChallengeType.COMPLETENESS: """
You are a completeness auditor for {domain} work.

CLAIM: {claim}
RESPONSE: {response}

YOUR TASK: Determine if this work is ACTUALLY COMPLETE or just claimed complete.

For CODING tasks, check:
- Are all functions implemented (not just declared)?
- Are there TODO/FIXME/placeholder markers?
- Is error handling present?
- Are edge cases handled?

For TESTING tasks, check:
- Are actual test results shown (not just "tests pass")?
- Is test coverage mentioned?
- Are specific test names/descriptions provided?

For RESEARCH tasks, check:
- Are sources cited?
- Is methodology explained?
- Are limitations acknowledged?

OUTPUT FORMAT (JSON):
{{
    "completeness_issues": [
        {{"missing_element": "...", "importance": 1-5}}
    ],
    "actually_complete": true/false,
    "completion_percentage": 0-100,
    "what_remains": ["..."]
}}
""",

        ChallengeType.EDGE_CASE: """
You are an edge case specialist for {domain}.

CLAIM: {claim}
RESPONSE: {response}

YOUR TASK: Identify edge cases that might break this solution.

Consider:
1. Boundary conditions (empty, null, max, min)
2. Error conditions (network failure, invalid input)
3. Concurrent/timing issues
4. Scale issues (1 item vs 1 million items)
5. Platform/environment differences

OUTPUT FORMAT (JSON):
{{
    "edge_cases": [
        {{"case": "...", "handled": true/false, "impact_if_unhandled": 1-5}}
    ],
    "unhandled_count": 0,
    "critical_unhandled": ["..."]
}}
"""
    }
    
    def __init__(self, llm_helper=None):
        """
        Initialize the challenger.
        
        Args:
            llm_helper: LLM helper for making API calls
        """
        self.llm_helper = llm_helper
        self.challenge_history: List[ChallengerVerdict] = []
        
        # Learning state
        self._outcomes: List[Dict] = []
        self._feedback: List[Dict] = []
        self._domain_adjustments: Dict[str, float] = {}
        self._challenge_effectiveness: Dict[str, float] = {}
    
    # ===== Learning Interface Methods =====
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        """Record challenge outcome for learning."""
        getattr(self, '_outcomes', {}).append(outcome)
        challenge_type = outcome.get('challenge_type', 'unknown')
        if outcome.get('found_issue', False):
            self._challenge_effectiveness[challenge_type] = getattr(self, '_challenge_effectiveness', {}).get(challenge_type, 0.5) + 0.05
    
    def record_feedback(self, feedback: Dict) -> None:
        """Record feedback on challenges."""
        getattr(self, '_feedback', {}).append(feedback)
        domain = feedback.get('domain', 'general')
        if feedback.get('challenge_too_strict', False):
            self._domain_adjustments[domain] = getattr(self, '_domain_adjustments', {}).get(domain, 0.0) - 0.05
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt challenge thresholds."""
        pass
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get domain adjustment."""
        return getattr(self, '_domain_adjustments', {}).get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics."""
        return {
            'challenge_history_count': len(self.challenge_history),
            'challenge_effectiveness': dict(self._challenge_effectiveness),
            'domain_adjustments': dict(self._domain_adjustments),
            'outcomes_recorded': len(self._outcomes),
            'feedback_recorded': len(self._feedback)
        }
    
    def create_challenges(self, 
                          claim: str, 
                          response: str, 
                          domain: str,
                          challenge_types: List[ChallengeType] = None) -> List[Challenge]:
        """
        Create challenges for a claim/response pair.
        
        Args:
            claim: The claim being made
            response: The full response
            domain: Domain context (medical, legal, technical, etc.)
            challenge_types: Which challenge types to use (default: all)
        """
        if challenge_types is None:
            # Default: use appropriate challenges based on domain
            if domain == 'technical':
                challenge_types = [
                    ChallengeType.COMPLETENESS,
                    ChallengeType.EDGE_CASE,
                    ChallengeType.EVIDENCE_DEMAND
                ]
            elif domain in ['medical', 'legal', 'financial']:
                challenge_types = [
                    ChallengeType.EVIDENCE_DEMAND,
                    ChallengeType.DEVILS_ADVOCATE,
                    ChallengeType.COMPLETENESS
                ]
            else:
                challenge_types = [
                    ChallengeType.EVIDENCE_DEMAND,
                    ChallengeType.DEVILS_ADVOCATE
                ]
        
        challenges = []
        for i, ctype in enumerate(challenge_types):
            prompt_template = self.CHALLENGE_PROMPTS.get(ctype, "")
            prompt = prompt_template.format(
                claim=claim,
                response=response,
                domain=domain
            )
            
            challenges.append(Challenge(
                challenge_id=f"CHAL-{i+1:03d}",
                challenge_type=ctype,
                prompt=prompt,
                target_claim=claim,
                domain=domain
            ))
        
        return challenges
    
    async def execute_challenge(self, challenge: Challenge) -> ChallengeResult:
        """
        Execute a single challenge using LLM.
        
        Args:
            challenge: The challenge to execute
        
        Returns:
            ChallengeResult with findings
        """
        if not self.llm_helper:
            # Fallback: pattern-based analysis without LLM
            return self._pattern_based_challenge(challenge)
        
        try:
            # Call LLM with the challenge prompt
            result = await self.llm_helper._call_llm(
                system_prompt="You are a rigorous adversarial analyzer. Your job is to find problems, not confirm quality. Output valid JSON only.",
                user_prompt=challenge.prompt
            )
            
            if result.success and result.result:
                return self._parse_llm_response(challenge, result.result)
            else:
                return self._pattern_based_challenge(challenge)
                
        except Exception as e:
            print(f"[LLMChallenger] Error executing challenge: {e}")
            return self._pattern_based_challenge(challenge)
    
    async def challenge_response(self,
                                  claim: str,
                                  response: str,
                                  domain: str = 'general',
                                  base_confidence: float = 0.5) -> ChallengerVerdict:
        """
        Run full challenge analysis on a response.
        
        Args:
            claim: The claim being made
            response: The full response to challenge
            domain: Domain context
            base_confidence: Starting confidence from threshold analysis
        
        Returns:
            ChallengerVerdict with comprehensive analysis
        """
        # Create appropriate challenges
        challenges = self.create_challenges(claim, response, domain)
        
        # Execute challenges (in parallel if possible)
        results = []
        for challenge in challenges:
            result = await self.execute_challenge(challenge)
            results.append(result)
        
        # Aggregate results
        total_issues = sum(len(r.issues_found) for r in results)
        total_evidence_gaps = sum(len(r.evidence_demanded) for r in results)
        total_confidence_reduction = sum(r.confidence_reduction for r in results)
        
        # Calculate adjusted confidence
        adjusted_confidence = base_confidence - total_confidence_reduction
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
        
        # Determine verdict
        should_accept = adjusted_confidence >= 0.6 and total_issues < 3
        should_remediate = total_issues > 0 or total_evidence_gaps > 0
        
        # Gather remediation guidance
        remediation_guidance = []
        for r in results:
            if r.evidence_demanded:
                remediation_guidance.extend([f"DEMAND: {e}" for e in r.evidence_demanded])
            if r.issues_found:
                remediation_guidance.extend([f"FIX: {i}" for i in r.issues_found])
        
        verdict = ChallengerVerdict(
            claim=claim,
            challenges_executed=len(challenges),
            total_issues=total_issues,
            total_evidence_gaps=total_evidence_gaps,
            adjusted_confidence=adjusted_confidence,
            should_accept=should_accept,
            should_remediate=should_remediate,
            remediation_guidance=remediation_guidance[:10],  # Top 10
            inventions_used=[
                "NOVEL-14: Neuro-Symbolic Reasoning",
                "NOVEL-13: Theory of Mind",
                "PPA1-Inv23: AI Common Sense",
                "LLMChallenger: Adversarial Analysis"
            ]
        )
        
        # Store in history
        self.challenge_history.append(verdict)
        
        return verdict
    
    def _pattern_based_challenge(self, challenge: Challenge) -> ChallengeResult:
        """
        Fallback pattern-based challenge when LLM not available.
        """
        issues = []
        evidence_demanded = []
        risk_factors = []
        confidence_reduction = 0.0
        
        response_lower = challenge.target_claim.lower()
        
        # Evidence demand patterns
        if challenge.challenge_type == ChallengeType.EVIDENCE_DEMAND:
            if '100%' in response_lower or 'all tests' in response_lower:
                evidence_demanded.append("Test execution output showing actual results")
                confidence_reduction += 0.15
            if 'complete' in response_lower or 'finished' in response_lower:
                evidence_demanded.append("File listing showing all claimed files exist")
                confidence_reduction += 0.1
            if any(p in response_lower for p in ['implemented', 'created', 'built']):
                evidence_demanded.append("Code content showing actual implementation")
                confidence_reduction += 0.1
        
        # Devil's advocate patterns
        elif challenge.challenge_type == ChallengeType.DEVILS_ADVOCATE:
            if 'always' in response_lower or 'never' in response_lower:
                issues.append("Absolute claim - real systems have exceptions")
                confidence_reduction += 0.1
            if 'perfect' in response_lower or 'zero' in response_lower:
                issues.append("Perfection claim - unlikely in complex systems")
                confidence_reduction += 0.15
            if 'guaranteed' in response_lower:
                issues.append("Guarantee without basis")
                risk_factors.append("Overconfidence")
                confidence_reduction += 0.2
        
        # Completeness patterns
        elif challenge.challenge_type == ChallengeType.COMPLETENESS:
            if 'todo' in response_lower or 'fixme' in response_lower:
                issues.append("Contains TODO/FIXME markers")
                confidence_reduction += 0.3
            if 'placeholder' in response_lower or 'pass' in response_lower:
                issues.append("Contains placeholder code")
                confidence_reduction += 0.25
            if 'notimplementederror' in response_lower.replace(' ', ''):
                issues.append("Raises NotImplementedError")
                confidence_reduction += 0.3
        
        # Edge case patterns
        elif challenge.challenge_type == ChallengeType.EDGE_CASE:
            if 'error handling' not in response_lower and 'exception' not in response_lower:
                issues.append("No error handling mentioned")
                confidence_reduction += 0.1
            if 'edge case' not in response_lower and 'boundary' not in response_lower:
                issues.append("No edge case handling mentioned")
                confidence_reduction += 0.1
        
        recommendation = "ACCEPT" if confidence_reduction < 0.2 else "INVESTIGATE" if confidence_reduction < 0.4 else "REJECT"
        
        return ChallengeResult(
            challenge_id=challenge.challenge_id,
            issues_found=issues,
            evidence_demanded=evidence_demanded,
            risk_factors=risk_factors,
            confidence_reduction=confidence_reduction,
            recommendation=recommendation,
            llm_reasoning="Pattern-based analysis (LLM not available)"
        )
    
    def _parse_llm_response(self, challenge: Challenge, llm_result: Dict) -> ChallengeResult:
        """
        Parse LLM response into structured ChallengeResult.
        """
        issues = []
        evidence_demanded = []
        risk_factors = []
        confidence_reduction = 0.0
        
        try:
            # Handle different response formats
            if isinstance(llm_result, dict):
                # Evidence demand response
                if 'evidence_gaps' in llm_result:
                    for gap in llm_result.get('evidence_gaps', []):
                        if not gap.get('present', True):
                            evidence_demanded.append(gap.get('required_evidence', 'Unknown evidence'))
                            severity = gap.get('severity', 3)
                            confidence_reduction += severity * 0.05
                
                # Devil's advocate response
                if 'potential_flaws' in llm_result:
                    for flaw in llm_result.get('potential_flaws', []):
                        issues.append(flaw.get('flaw', 'Unknown flaw'))
                        severity = flaw.get('severity', 3)
                        likelihood = flaw.get('likelihood', 3)
                        confidence_reduction += (severity * likelihood) * 0.02
                
                # Completeness response
                if 'completeness_issues' in llm_result:
                    for issue in llm_result.get('completeness_issues', []):
                        issues.append(issue.get('missing_element', 'Unknown missing element'))
                        importance = issue.get('importance', 3)
                        confidence_reduction += importance * 0.05
                
                # Edge case response
                if 'edge_cases' in llm_result:
                    unhandled = [e for e in llm_result.get('edge_cases', []) if not e.get('handled', True)]
                    for edge in unhandled:
                        issues.append(f"Unhandled edge case: {edge.get('case', 'Unknown')}")
                        impact = edge.get('impact_if_unhandled', 3)
                        confidence_reduction += impact * 0.05
            
            # Extract raw response for reasoning
            raw_response = llm_result.get('raw_response', str(llm_result))
            
        except Exception as e:
            issues.append(f"Error parsing LLM response: {str(e)}")
            confidence_reduction = 0.2
            raw_response = str(llm_result)
        
        confidence_reduction = min(confidence_reduction, 0.5)  # Cap at 50% reduction
        recommendation = "ACCEPT" if confidence_reduction < 0.15 else "INVESTIGATE" if confidence_reduction < 0.3 else "REJECT"
        
        return ChallengeResult(
            challenge_id=challenge.challenge_id,
            issues_found=issues,
            evidence_demanded=evidence_demanded,
            risk_factors=risk_factors,
            confidence_reduction=confidence_reduction,
            recommendation=recommendation,
            llm_reasoning=raw_response[:500] if raw_response else "No reasoning provided"
        )
    
    # =========================================================================
    # Learning Interface (5/5 methods)
    # =========================================================================
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback about challenge effectiveness."""
        challenge_type = feedback.get('challenge_type')
        if challenge_type and not feedback.get('was_correct', True):
            current = getattr(self, '_challenge_effectiveness', {}).get(challenge_type, 0.5)
            self._challenge_effectiveness[challenge_type] = max(0.1, current - 0.05)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return challenger statistics."""
        return {
            'total_challenges': getattr(self, '_total_challenges', 0),
            'blocked_count': getattr(self, '_blocked_count', 0),
            'improved_count': getattr(self, '_improved_count', 0),
            'challenge_effectiveness': dict(self._challenge_effectiveness),
            'outcomes_recorded': len(self._outcomes)
        }
    
    def serialize_state(self) -> Dict:
        """Serialize state for persistence."""
        return {
            'challenge_effectiveness': dict(self._challenge_effectiveness),
            'outcomes': getattr(self, '_outcomes', [])[-100:],
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore state from serialized data."""
        self._challenge_effectiveness = state.get('challenge_effectiveness', {})
        self._outcomes = state.get('outcomes', [])


def create_challenger(llm_helper=None) -> LLMChallenger:
    """Factory function to create LLM Challenger."""
    return LLMChallenger(llm_helper=llm_helper)
