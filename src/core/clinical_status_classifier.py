"""
CLINICAL STATUS CLASSIFIER (NOVEL-32)
=====================================
Categorizes LLM outputs into clinical status categories.

Categories:
- TRULY_WORKING: Code executes, tests pass, no placeholders
- INCOMPLETE: Partial implementation, missing components
- STUBBED: Placeholder code, pass statements, TODOs
- SIMULATED: Mock data, fake results, no real execution
- FALLBACK: Error handling triggered, degraded mode
- FAILOVER: Alternative path used, not primary

Created: December 26, 2025
Patent: NOVEL-32 - Clinical Status Classification System
"""

import re
import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


class ClinicalStatus(Enum):
    """Clinical status categories for governance outputs."""
    TRULY_WORKING = "truly_working"
    INCOMPLETE = "incomplete"
    STUBBED = "stubbed"
    SIMULATED = "simulated"
    FALLBACK = "fallback"
    FAILOVER = "failover"
    UNKNOWN = "unknown"


@dataclass
class StatusEvidence:
    """Evidence supporting a status classification."""
    category: ClinicalStatus
    confidence: float
    indicators: List[str]
    source_lines: List[str] = field(default_factory=list)


@dataclass
class ClinicalClassificationResult:
    """Complete result of clinical status classification."""
    primary_status: ClinicalStatus
    confidence: float
    evidence: List[StatusEvidence]
    breakdown: Dict[ClinicalStatus, int]
    llm_reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_status": self.primary_status.value,
            "confidence": self.confidence,
            "evidence": [
                {
                    "category": e.category.value,
                    "confidence": e.confidence,
                    "indicators": e.indicators
                }
                for e in self.evidence
            ],
            "breakdown": {k.value: v for k, v in self.breakdown.items()},
            "llm_reasoning": self.llm_reasoning,
            "timestamp": self.timestamp.isoformat()
        }


class ClinicalStatusClassifier:
    """
    Classifies LLM outputs into clinical status categories.
    
    Uses pattern detection + optional LLM verification for
    accurate status assignment.
    """
    
    # Pattern sets for each status
    STUBBED_PATTERNS = [
        (r'\bTODO\b', "TODO marker found"),
        (r'\bFIXME\b', "FIXME marker found"),
        (r'\bpass\s*$', "Empty pass statement"),
        (r'\bpass\s*#', "Pass with comment"),
        (r'\braise\s+NotImplementedError', "NotImplementedError"),
        (r'\.\.\.', "Ellipsis placeholder"),
        (r'\bplaceholder\b', "Placeholder text"),
        (r'\bstub\b', "Stub reference"),
        (r'#\s*TODO', "TODO comment"),
        (r'#\s*STUB', "STUB comment"),
        (r'return\s+None\s*#', "Return None placeholder"),
    ]
    
    SIMULATED_PATTERNS = [
        (r'\bmock\b', "Mock reference"),
        (r'\bfake\b', "Fake data reference"),
        (r'\bsimulated?\b', "Simulated data"),
        (r'\btest_data\b', "Test data reference"),
        (r'\bdummy\b', "Dummy data"),
        (r'\bsample\b', "Sample data"),
        (r'random\.\w+\(', "Random data generation"),
        (r'\bhardcoded\b', "Hardcoded values"),
        (r'="example', "Example string"),
        (r"='example", "Example string"),
    ]
    
    INCOMPLETE_PATTERNS = [
        (r'\bwill\s+be\b', "Future tense - not complete"),
        (r'\bwill\s+add\b', "Future addition"),
        (r'\bcoming\s+soon\b', "Coming soon"),
        (r'\bplanned\b', "Planned feature"),
        (r'\bfuture\b', "Future reference"),
        (r'\bnot\s+yet\b', "Not yet implemented"),
        (r'\bpartial\b', "Partial implementation"),
        (r'\bincomplete\b', "Incomplete reference"),
        (r'\bmissing\b', "Missing component"),
        (r'\bneeds?\s+to\s+be\b', "Needs to be done"),
    ]
    
    FALLBACK_PATTERNS = [
        (r'except\s*:', "Bare except clause"),
        (r'except\s+\w+\s*:\s*pass', "Exception swallowed"),
        (r'try:\s*\n\s*.*\n\s*except.*:\s*\n\s*return\s+None', "Try-except returning None"),
        (r'\.get\([^,]+,\s*None\)', "Dict get with None default"),
        (r'or\s+None', "Or None fallback"),
        (r'if\s+.*\s+else\s+None', "Conditional None"),
        (r'fallback', "Fallback reference"),
        (r'default\s*=', "Default value"),
    ]
    
    FAILOVER_PATTERNS = [
        (r'alternative', "Alternative reference"),
        (r'backup', "Backup reference"),
        (r'secondary', "Secondary path"),
        (r'failover', "Failover reference"),
        (r'retry', "Retry logic"),
        (r'attempt\s+\d+', "Multiple attempts"),
    ]
    
    TRULY_WORKING_INDICATORS = [
        (r'assert\s+', "Assert statement"),
        (r'assertEqual', "Test assertion"),
        (r'assertTrue', "Test assertion"),
        (r'\.verify\(', "Verification call"),
        (r'test.*passed', "Test passed"),
        (r'✓|✅|PASS', "Pass indicator"),
        (r'exit\s*code[:\s]+0', "Exit code 0"),
        (r'returned.*result', "Returned result"),
        (r'output:\s*\S+', "Has output"),
        (r'score:\s*\d+', "Has score"),
    ]
    
    def __init__(self, llm_api_key: Optional[str] = None):
        """Initialize classifier with optional LLM for verification."""
        self.api_key = llm_api_key or os.environ.get('GROK_API_KEY') or os.environ.get('XAI_API_KEY')
        self._classification_history: List[ClinicalClassificationResult] = []
    
    def classify(
        self,
        response: str,
        evidence: List[str] = None,
        context: Dict[str, Any] = None
    ) -> ClinicalClassificationResult:
        """
        Classify the clinical status of a response.
        
        Args:
            response: The LLM response to classify
            evidence: Optional list of evidence items
            context: Optional context (query, domain, etc.)
            
        Returns:
            ClinicalClassificationResult with status and evidence
        """
        all_evidence = []
        breakdown = {status: 0 for status in ClinicalStatus}
        
        # Combine response and evidence for analysis
        text_to_analyze = response
        if evidence:
            text_to_analyze += "\n" + "\n".join(evidence)
        
        # Check each pattern category
        stubbed_ev = self._check_patterns(text_to_analyze, self.STUBBED_PATTERNS, ClinicalStatus.STUBBED)
        if stubbed_ev:
            all_evidence.append(stubbed_ev)
            breakdown[ClinicalStatus.STUBBED] = len(stubbed_ev.indicators)
        
        simulated_ev = self._check_patterns(text_to_analyze, self.SIMULATED_PATTERNS, ClinicalStatus.SIMULATED)
        if simulated_ev:
            all_evidence.append(simulated_ev)
            breakdown[ClinicalStatus.SIMULATED] = len(simulated_ev.indicators)
        
        incomplete_ev = self._check_patterns(text_to_analyze, self.INCOMPLETE_PATTERNS, ClinicalStatus.INCOMPLETE)
        if incomplete_ev:
            all_evidence.append(incomplete_ev)
            breakdown[ClinicalStatus.INCOMPLETE] = len(incomplete_ev.indicators)
        
        fallback_ev = self._check_patterns(text_to_analyze, self.FALLBACK_PATTERNS, ClinicalStatus.FALLBACK)
        if fallback_ev:
            all_evidence.append(fallback_ev)
            breakdown[ClinicalStatus.FALLBACK] = len(fallback_ev.indicators)
        
        failover_ev = self._check_patterns(text_to_analyze, self.FAILOVER_PATTERNS, ClinicalStatus.FAILOVER)
        if failover_ev:
            all_evidence.append(failover_ev)
            breakdown[ClinicalStatus.FAILOVER] = len(failover_ev.indicators)
        
        working_ev = self._check_patterns(text_to_analyze, self.TRULY_WORKING_INDICATORS, ClinicalStatus.TRULY_WORKING)
        if working_ev:
            all_evidence.append(working_ev)
            breakdown[ClinicalStatus.TRULY_WORKING] = len(working_ev.indicators)
        
        # Determine primary status
        primary_status, confidence = self._determine_primary_status(breakdown, all_evidence)
        
        result = ClinicalClassificationResult(
            primary_status=primary_status,
            confidence=confidence,
            evidence=all_evidence,
            breakdown=breakdown
        )
        
        self._classification_history.append(result)
        return result
    
    def _check_patterns(
        self,
        text: str,
        patterns: List[tuple],
        status: ClinicalStatus
    ) -> Optional[StatusEvidence]:
        """Check text against a set of patterns."""
        indicators = []
        source_lines = []
        
        for pattern, description in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                indicators.append(description)
                # Get surrounding context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                source_lines.append(text[start:end].strip())
        
        if indicators:
            # Confidence based on number of indicators
            confidence = min(1.0, len(indicators) * 0.2)
            return StatusEvidence(
                category=status,
                confidence=confidence,
                indicators=list(set(indicators)),  # Deduplicate
                source_lines=source_lines[:5]  # Limit to 5 examples
            )
        return None
    
    def _determine_primary_status(
        self,
        breakdown: Dict[ClinicalStatus, int],
        evidence: List[StatusEvidence]
    ) -> tuple:
        """Determine the primary status based on breakdown."""
        
        # Priority order (worst to best)
        priority = [
            ClinicalStatus.STUBBED,      # Worst - placeholder code
            ClinicalStatus.SIMULATED,    # Bad - fake data
            ClinicalStatus.INCOMPLETE,   # Partial work
            ClinicalStatus.FALLBACK,     # Degraded
            ClinicalStatus.FAILOVER,     # Alternative path
            ClinicalStatus.TRULY_WORKING # Best
        ]
        
        # If stubbed or simulated indicators exist, that's the status
        # regardless of "working" indicators
        for status in priority[:-1]:  # All except TRULY_WORKING
            if breakdown.get(status, 0) > 0:
                ev = next((e for e in evidence if e.category == status), None)
                confidence = ev.confidence if ev else 0.5
                return status, confidence
        
        # If only working indicators and nothing bad
        if breakdown.get(ClinicalStatus.TRULY_WORKING, 0) > 0:
            ev = next((e for e in evidence if e.category == ClinicalStatus.TRULY_WORKING), None)
            confidence = ev.confidence if ev else 0.5
            return ClinicalStatus.TRULY_WORKING, confidence
        
        # No indicators found
        return ClinicalStatus.UNKNOWN, 0.0
    
    async def classify_with_llm(
        self,
        response: str,
        evidence: List[str] = None,
        context: Dict[str, Any] = None
    ) -> ClinicalClassificationResult:
        """
        Classify with LLM verification for higher accuracy.
        
        Uses LLM to reason about the actual status of the response.
        """
        # First do pattern-based classification
        pattern_result = self.classify(response, evidence, context)
        
        if not self.api_key:
            return pattern_result
        
        try:
            import httpx
            
            prompt = f"""Classify this LLM output into one of these clinical status categories:

CATEGORIES:
- TRULY_WORKING: Code actually executes, tests actually pass, no placeholders
- INCOMPLETE: Partial implementation, missing components, future work mentioned
- STUBBED: Placeholder code (TODO, pass, NotImplementedError, ellipsis)
- SIMULATED: Mock/fake/sample data, not real execution
- FALLBACK: Error handling triggered, degraded mode
- FAILOVER: Alternative/backup path used

RESPONSE TO CLASSIFY:
{response[:2000]}

EVIDENCE PROVIDED:
{chr(10).join(evidence[:5]) if evidence else "None"}

PATTERN ANALYSIS (pre-check):
{pattern_result.primary_status.value} with confidence {pattern_result.confidence:.2f}
Indicators: {[e.indicators for e in pattern_result.evidence]}

YOUR TASK:
1. Analyze the actual content, not just patterns
2. Determine if this represents real, working implementation or not
3. Be strict: presence of TODO, placeholder, or "will be" means NOT truly working

Respond in JSON:
{{
    "status": "truly_working|incomplete|stubbed|simulated|fallback|failover",
    "confidence": 0.0-1.0,
    "reasoning": "Explain why this status was assigned",
    "key_evidence": ["list", "of", "evidence"]
}}"""

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": "grok-3-latest",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1
                    }
                )
                
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"]
                    
                    # Parse JSON response
                    import json
                    json_match = re.search(r'\{[^{}]*"status"[^{}]*\}', content, re.DOTALL)
                    if json_match:
                        llm_result = json.loads(json_match.group())
                        
                        # Map string to enum
                        status_map = {s.value: s for s in ClinicalStatus}
                        llm_status = status_map.get(
                            llm_result.get("status", "").lower(),
                            ClinicalStatus.UNKNOWN
                        )
                        
                        # Create enhanced result
                        return ClinicalClassificationResult(
                            primary_status=llm_status,
                            confidence=llm_result.get("confidence", 0.5),
                            evidence=pattern_result.evidence,
                            breakdown=pattern_result.breakdown,
                            llm_reasoning=llm_result.get("reasoning", "")
                        )
        
        except Exception as e:
            pattern_result.llm_reasoning = f"LLM verification failed: {str(e)[:100]}"
        
        return pattern_result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get classification statistics."""
        if not self._classification_history:
            return {"total": 0}
        
        status_counts = {s.value: 0 for s in ClinicalStatus}
        for result in self._classification_history:
            status_counts[result.primary_status.value] += 1
        
        return {
            "total": len(self._classification_history),
            "by_status": status_counts,
            "average_confidence": sum(r.confidence for r in self._classification_history) / len(self._classification_history)
        }

