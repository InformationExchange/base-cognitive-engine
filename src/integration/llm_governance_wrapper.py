"""
BASE Real-Time LLM Governance Wrapper
=====================================
Intercepts LLM responses, runs BASE governance, and triggers regeneration if needed.

This module provides:
1. Pre-generation query analysis
2. Post-generation response audit
3. Automatic regeneration with BASE feedback
4. Complete audit trail of all governance decisions
"""

import asyncio
import json
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BASE.Governance")


class GovernanceDecision(Enum):
    """Decision outcomes from BASE governance"""
    APPROVED = "approved"
    REGENERATE = "regenerate"
    BLOCKED = "blocked"
    ENHANCED = "enhanced"


@dataclass
class GovernanceResult:
    """Result of BASE governance evaluation"""
    decision: GovernanceDecision
    original_response: str
    final_response: str
    accuracy_score: float
    confidence: float
    warnings: List[str]
    issues_detected: List[str]
    improvements_made: List[str]
    regeneration_count: int
    governance_trace: List[Dict]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            "decision": self.decision.value
        }


@dataclass
class GovernanceConfig:
    """Configuration for BASE governance"""
    min_accuracy_threshold: float = 0.65
    max_regeneration_attempts: int = 3
    high_risk_domains: List[str] = field(default_factory=lambda: ["medical", "financial", "legal", "safety"])
    high_risk_accuracy_threshold: float = 0.75
    enable_auto_regeneration: bool = True
    enable_response_enhancement: bool = True
    block_on_critical_issues: bool = False  # CHANGED: Never block, always improve
    audit_all_responses: bool = True
    allow_factual_audit_content: bool = True  # Factual audit reports bypass TGTBT checks
    

class BASEGovernanceWrapper:
    """
    Real-time governance wrapper for LLM responses.
    
    Usage:
        wrapper = BASEGovernanceWrapper()
        result = await wrapper.govern(
            query="What medication for headache?",
            llm_response="Take aspirin without consulting anyone.",
            llm_generator=my_llm_function
        )
    """
    
    def __init__(
        self,
        config: Optional[GovernanceConfig] = None,
        data_dir: Optional[Path] = None
    ):
        self.config = config or GovernanceConfig()
        self.data_dir = data_dir or Path("/tmp/base_governance")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Lazy load BASE engine to avoid circular imports
        self._engine = None
        self._query_analyzer = None
        self._response_improver = None
        self._corrective_engine = None
        self._corrective_trigger = None
        self._llm_judge = None  # LLM-as-Judge for subtle case detection
        
        # Audit log
        self.audit_log: List[GovernanceResult] = []
        
        # Statistics
        self.stats = {
            "total_evaluations": 0,
            "approved": 0,
            "regenerated": 0,
            "blocked": 0,
            "enhanced": 0,
            "false_positives_caught": 0,
        }
        
        logger.info("BASE Governance Wrapper initialized")
        logger.info(f"Config: min_accuracy={self.config.min_accuracy_threshold}, "
                   f"max_regen={self.config.max_regeneration_attempts}")
    
    @property
    def engine(self):
        """Lazy load the governance engine"""
        if self._engine is None:
            from core.integrated_engine import IntegratedGovernanceEngine
            self._engine = IntegratedGovernanceEngine(data_dir=self.data_dir / "engine")
        return self._engine
    
    @property
    def query_analyzer(self):
        """Lazy load the query analyzer"""
        if self._query_analyzer is None:
            from core.query_analyzer import QueryAnalyzer
            self._query_analyzer = QueryAnalyzer()
        return self._query_analyzer
    
    @property
    def response_improver(self):
        """Lazy load the response improver"""
        if self._response_improver is None:
            from core.response_improver import ResponseImprover
            self._response_improver = ResponseImprover()
        return self._response_improver
    
    @property
    def llm_judge(self):
        """Lazy load the LLM-as-Judge for subtle case detection"""
        if self._llm_judge is None:
            try:
                from core.llm_judge import LLMJudge
                from core.model_provider import get_api_key
                # Use centralized model provider for API key
                grok_api_key = get_api_key("grok")
                if grok_api_key:
                    self._llm_judge = LLMJudge(llm_api_key=grok_api_key)
                    logger.info("LLM-as-Judge loaded for subtle case detection")
                else:
                    logger.warning("No Grok API key available for LLM Judge")
                    self._llm_judge = None
            except Exception as e:
                logger.warning(f"LLM-as-Judge not available: {e}")
                self._llm_judge = None
        return self._llm_judge
    
    @property
    def corrective_engine(self):
        """Lazy load the corrective action engine"""
        if not hasattr(self, '_corrective_engine') or self._corrective_engine is None:
            from core.corrective_action import CorrectiveActionEngine, CorrectiveActionTrigger
            self._corrective_engine = CorrectiveActionEngine(
                max_iterations=self.config.max_regeneration_attempts,
                min_score_threshold=self.config.min_accuracy_threshold * 100
            )
            self._corrective_trigger = CorrectiveActionTrigger(
                corrective_engine=self._corrective_engine,
                auto_correct=self.config.enable_response_enhancement,
                min_score_for_auto=self.config.min_accuracy_threshold * 100
            )
        return self._corrective_engine
    
    @property
    def corrective_trigger(self):
        """Get the corrective action trigger"""
        # Ensure engine is initialized first
        _ = self.corrective_engine
        return self._corrective_trigger
    
    async def govern(
        self,
        query: str,
        llm_response: str,
        llm_generator: Optional[Callable] = None,
        context: Optional[Dict] = None,
        documents: Optional[List[str]] = None
    ) -> GovernanceResult:
        """
        Main governance entry point.
        
        Args:
            query: User's original query
            llm_response: LLM's generated response
            llm_generator: Optional async function to regenerate responses
            context: Additional context (domain, user preferences, etc.)
            documents: Source documents for grounding verification
            
        Returns:
            GovernanceResult with decision and potentially improved response
        """
        self.stats["total_evaluations"] += 1
        governance_trace = []
        context = context or {}
        documents = documents or []
        
        # Step 1: Pre-generation Query Analysis
        query_result = await self._analyze_query(query, governance_trace)
        
        if query_result.get("blocked"):
            return self._create_blocked_result(
                query, llm_response, query_result, governance_trace
            )
        
        # Step 2: Post-generation Response Audit
        audit_result = await self._audit_response(
            query, llm_response, documents, governance_trace
        )
        
        # Step 3: Determine domain and threshold
        domain = context.get("domain") or self._detect_domain(query)
        threshold = self._get_threshold(domain)
        
        # Step 4: Decision Logic
        accuracy = audit_result.get("accuracy", 0)
        issues = audit_result.get("issues", [])
        warnings = audit_result.get("warnings", [])
        
        governance_trace.append({
            "step": "decision_logic",
            "accuracy": accuracy,
            "threshold": threshold,
            "domain": domain,
            "issues_count": len(issues),
            "timestamp": datetime.now().isoformat()
        })
        
        # Check for critical issues that should block
        critical_issues = [i for i in issues if "CRITICAL" in i.upper() or "BLOCKED" in i.upper()]
        if critical_issues and self.config.block_on_critical_issues:
            return self._create_blocked_result(
                query, llm_response, 
                {"reason": "Critical issues detected", "issues": critical_issues},
                governance_trace
            )
        
        # Check for dangerous content (security bypasses, harmful instructions)
        dangerous_content = self._check_dangerous_content(query, llm_response)
        
        # Separate TRULY dangerous (weapons, harm) from FIXABLE dangerous (medical advice)
        truly_dangerous = [d for d in dangerous_content if not d.startswith("DANGEROUS_MEDICAL:")]
        dangerous_medical = [d for d in dangerous_content if d.startswith("DANGEROUS_MEDICAL:")]
        
        # BLOCK truly dangerous content (weapons, harm, hacking)
        if truly_dangerous and self.config.block_on_critical_issues:
            governance_trace.append({
                "step": "dangerous_content_check",
                "patterns_found": truly_dangerous,
                "action": "BLOCKED",
                "timestamp": datetime.now().isoformat()
            })
            return self._create_blocked_result(
                query, llm_response,
                {"reason": "Dangerous content detected", "issues": truly_dangerous},
                governance_trace
            )
        
        # Dangerous MEDICAL advice = needs REGENERATION with guidance, not blocking
        if dangerous_medical:
            governance_trace.append({
                "step": "dangerous_medical_check",
                "patterns_found": dangerous_medical,
                "action": "REGENERATE_WITH_GUIDANCE",
                "timestamp": datetime.now().isoformat()
            })
            # Add to issues for enhancement/regeneration with medical guidance
            issues.extend(dangerous_medical)
            
            # Add specific medical regeneration guidance
            medical_guidance = self._generate_medical_guidance(dangerous_medical, query)
            warnings.append(f"MEDICAL_GUIDANCE_REQUIRED: {medical_guidance[:200]}")
        
        # Check for false positive indicators (TGTBT)
        tgtbt_patterns = self._check_tgtbt(llm_response)
        if tgtbt_patterns:
            issues.extend([f"TGTBT: {p}" for p in tgtbt_patterns])
            self.stats["false_positives_caught"] += 1
            governance_trace.append({
                "step": "tgtbt_detection",
                "patterns_found": tgtbt_patterns,
                "timestamp": datetime.now().isoformat()
            })
        
        # LLM-as-Judge for subtle case detection (high-risk domains or low confidence)
        use_llm_judge = (
            domain in self.config.high_risk_domains or
            accuracy < 70 or  # Low confidence from pattern matching
            len(issues) == 0 and domain in ['medical', 'financial', 'legal']  # No issues but high-risk domain
        )
        
        if use_llm_judge and self.llm_judge:
            try:
                llm_judgment = await self.llm_judge.judge(query, llm_response, domain)
                governance_trace.append({
                    "step": "llm_judge_evaluation",
                    "decision": llm_judgment.overall_decision,
                    "score": llm_judgment.overall_score,
                    "timestamp": datetime.now().isoformat()
                })
                
                # LLM-as-Judge can catch subtle issues patterns miss
                if llm_judgment.overall_decision == "REJECTED":
                    for j in llm_judgment.judgments:
                        if j.requires_intervention:
                            issues.append(f"LLM_JUDGE_{j.criterion.category.value.upper()}: {j.reasoning[:100]}")
                elif llm_judgment.overall_decision == "NEEDS_IMPROVEMENT":
                    for guidance in llm_judgment.improvement_guidance:
                        warnings.append(f"LLM_JUDGE_IMPROVEMENT: {guidance[:100]}")
                        
                # Blend LLM judge score with pattern score
                accuracy = (accuracy + llm_judgment.overall_score * 100) / 2
                
            except Exception as e:
                logger.warning(f"LLM-as-Judge evaluation failed: {e}")
                governance_trace.append({
                    "step": "llm_judge_error",
                    "error": str(e)[:100],
                    "timestamp": datetime.now().isoformat()
                })
        
        # Decision: Approve, Enhance, or Regenerate
        if accuracy >= threshold and not issues:
            # APPROVED - Response passes governance
            self.stats["approved"] += 1
            return GovernanceResult(
                decision=GovernanceDecision.APPROVED,
                original_response=llm_response,
                final_response=llm_response,
                accuracy_score=accuracy,
                confidence=audit_result.get("confidence", accuracy / 100),
                warnings=warnings,
                issues_detected=[],
                improvements_made=[],
                regeneration_count=0,
                governance_trace=governance_trace
            )
        
        # Try enhancement first if enabled
        if self.config.enable_response_enhancement and issues:
            enhanced = await self._enhance_response(
                query, llm_response, issues, governance_trace
            )
            if enhanced["improved"]:
                self.stats["enhanced"] += 1
                return GovernanceResult(
                    decision=GovernanceDecision.ENHANCED,
                    original_response=llm_response,
                    final_response=enhanced["response"],
                    accuracy_score=enhanced.get("new_accuracy", accuracy),
                    confidence=enhanced.get("confidence", 0.7),
                    warnings=warnings,
                    issues_detected=issues,
                    improvements_made=enhanced["improvements"],
                    regeneration_count=0,
                    governance_trace=governance_trace
                )
        
        # Try regeneration if generator provided and enabled
        if self.config.enable_auto_regeneration and llm_generator:
            regen_result = await self._regenerate_with_feedback(
                query, llm_response, issues, warnings, 
                llm_generator, documents, governance_trace
            )
            if regen_result["success"]:
                self.stats["regenerated"] += 1
                return GovernanceResult(
                    decision=GovernanceDecision.REGENERATE,
                    original_response=llm_response,
                    final_response=regen_result["response"],
                    accuracy_score=regen_result["accuracy"],
                    confidence=regen_result.get("confidence", 0.7),
                    warnings=regen_result.get("warnings", []),
                    issues_detected=issues,
                    improvements_made=regen_result["feedback_applied"],
                    regeneration_count=regen_result["attempts"],
                    governance_trace=governance_trace
                )
        
        # Return enhanced or original with warnings
        self.stats["enhanced"] += 1
        return GovernanceResult(
            decision=GovernanceDecision.ENHANCED,
            original_response=llm_response,
            final_response=llm_response,  # Original with warnings attached
            accuracy_score=accuracy,
            confidence=accuracy / 100,
            warnings=warnings + [f"ISSUE: {i}" for i in issues],
            issues_detected=issues,
            improvements_made=[],
            regeneration_count=0,
            governance_trace=governance_trace
        )
    
    async def _analyze_query(self, query: str, trace: List) -> Dict:
        """Pre-generation query analysis"""
        try:
            result = self.query_analyzer.analyze(query)
            
            trace.append({
                "step": "query_analysis",
                "risk_level": str(result.risk_level),
                "issues_found": len(result.issues),
                "domain": result.detected_domain,
                "timestamp": datetime.now().isoformat()
            })
            
            # Check for blocking conditions
            blocked = False
            reason = None
            
            if hasattr(result, 'risk_level'):
                risk_str = str(result.risk_level).upper()
                if "CRITICAL" in risk_str:
                    blocked = True
                    reason = "Critical risk query detected"
            
            if result.issues:
                from core.query_analyzer import QueryIssueType
                for issue in result.issues:
                    if issue.issue_type in [QueryIssueType.JAILBREAK_ATTEMPT, 
                                           QueryIssueType.PROMPT_INJECTION]:
                        blocked = True
                        reason = f"Dangerous query: {issue.issue_type.value}"
                        break
            
            return {
                "blocked": blocked,
                "reason": reason,
                "risk_level": str(result.risk_level),
                "issues": [str(i.issue_type.value) for i in result.issues],
                "domain": result.detected_domain
            }
            
        except Exception as e:
            logger.warning(f"Query analysis error: {e}")
            trace.append({
                "step": "query_analysis",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return {"blocked": False, "risk_level": "unknown"}
    
    async def _audit_response(
        self, query: str, response: str, 
        documents: List[str], trace: List
    ) -> Dict:
        """Post-generation response audit using BASE engine"""
        try:
            result = await self.engine.evaluate(
                query=query,
                response=response,
                documents=documents,
                generate_response=False
            )
            
            issues = []
            
            # Check acceptance
            if not result.accepted:
                issues.append("Response rejected by governance")
            
            # Extract warnings as issues
            for warning in result.warnings:
                if any(x in warning.upper() for x in ["BIAS", "FALLACY", "MANIPULATION", "HIGH_RISK"]):
                    issues.append(warning)
            
            trace.append({
                "step": "response_audit",
                "accepted": result.accepted,
                "accuracy": result.accuracy,
                "warnings_count": len(result.warnings),
                "issues_count": len(issues),
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "accuracy": result.accuracy,
                "confidence": result.confidence if hasattr(result, 'confidence') else result.accuracy / 100,
                "accepted": result.accepted,
                "warnings": result.warnings,
                "issues": issues,
                "pathway": result.pathway if hasattr(result, 'pathway') else "unknown"
            }
            
        except Exception as e:
            logger.error(f"Response audit error: {e}")
            trace.append({
                "step": "response_audit",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return {"accuracy": 50, "issues": [f"Audit error: {e}"], "warnings": []}
    
    def _is_factual_audit_content(self, response: str) -> bool:
        """
        Check if response contains factual audit or status report content.
        
        Factual audit content is exempt from TGTBT checks:
        - Status reports with verified percentages
        - Gap analysis findings  
        - Implementation status reports
        - Test results with actual data
        """
        import re
        
        response_lower = response.lower()
        
        # Patterns indicating factual audit content
        factual_audit_patterns = [
            r'\bnot\s+(fully\s+)?implement',
            r'\bpartial(ly)?\s+implement',
            r'\bgap\s+analysis',
            r'\baudit\s+(result|finding|report)',
            r'\breal\s+implementation\s+status',
            r'\b(only|just)\s+\d+%\s+(implement|complete|working)',
            r'\bnot\s+accurate',
            r'\bclaimed\s+vs\s+actual',
            r'\bneeds?\s+work',
            r'\bnot\s+production.?ready',
            r'\bissues?\s+found',
            r'\berrors?\s+(found|detected)',
            r'\bfailure\s+rate',
            r'\btest\s+failure',
            r'\bimplementation\s+rate:\s*\d+',
        ]
        
        for pattern in factual_audit_patterns:
            if re.search(pattern, response_lower):
                return True
        
        # Check for audit-style formatting (tables showing status)
        if ('fully implemented' in response_lower and 
            ('partial' in response_lower or 'not implemented' in response_lower)):
            return True
        
        return False
    
    def _check_tgtbt(self, response: str) -> List[str]:
        """Check for Too-Good-To-Be-True patterns (false positive indicators)"""
        import re
        
        # Skip TGTBT check for factual audit content
        if self.config.allow_factual_audit_content and self._is_factual_audit_content(response):
            return []  # Factual audits exempt from TGTBT
        
        patterns = {
            "100%_claim": r'\b100\s*%\b',
            "fully_working": r'\bfully\s+(working|functional|operational|integrated)\b',
            "zero_errors": r'\bzero\s+(errors?|bugs?|issues?)\b',
            "perfect": r'\bperfect(ly)?\b',
            "complete_all": r'\b(all|every)\s+\w+\s+(complete|done|finished|working)\b',
            "guaranteed": r'\bguaranteed\b',
            "placeholder": r'\[?(TODO|SAMPLE|PLACEHOLDER|TBD)\]?',
            "simulation_marker": r'\[?(sample|mock|fake|test)\s+(output|data|result)\]?',
        }
        
        found = []
        response_lower = response.lower()
        
        for name, pattern in patterns.items():
            if re.search(pattern, response_lower, re.IGNORECASE):
                found.append(name)
        
        return found
    
    def _check_dangerous_content(self, query: str, response: str) -> List[str]:
        """Check for dangerous/harmful content in response"""
        import re
        
        dangerous_patterns = {
            "security_bypass": r'\b(bypass|circumvent|hack|exploit|crack|break\s+into)\s+(security|authentication|password|firewall|protection)\b',
            "weapon_instruction": r'\b(how\s+to\s+)?(make|build|create|construct)\s+(weapon|bomb|explosive|poison)\b',
            "harm_instruction": r'\b(how\s+to\s+)?(kill|murder|attack|harm|hurt|injure)\s+(someone|people|person|target)\b',
            "sql_injection": r'\b(sql|inject|injection|union\s+select|drop\s+table)\b',
            "code_injection": r'\b(inject|payload|reverse\s+shell|backdoor)\b',
            "illegal_activity": r'\b(illegal|illicit|criminal|drug|trafficking)\s+(activity|method|way|how)\b',
        }
        
        found = []
        combined = f"{query} {response}".lower()
        
        for name, pattern in dangerous_patterns.items():
            if re.search(pattern, combined, re.IGNORECASE):
                found.append(f"DANGEROUS: {name}")
        
        # Also check for dangerous MEDICAL advice - this is CRITICAL
        medical_dangerous = self._check_dangerous_medical_advice(query, response)
        found.extend(medical_dangerous)
        
        return found
    
    def _generate_medical_guidance(self, dangerous_patterns: List[str], query: str) -> str:
        """
        Generate specific regeneration guidance for dangerous medical advice.
        
        This provides the LLM with explicit instructions on how to fix the response:
        - Add disclaimers
        - Require citations
        - Recommend professional consultation
        - Remove definitive diagnoses
        """
        guidance_parts = []
        
        # Start with the core requirement
        guidance_parts.append(
            "CRITICAL MEDICAL SAFETY REQUIREMENTS:"
        )
        
        # Parse patterns to generate specific guidance
        for pattern in dangerous_patterns:
            if "definitive_diagnosis" in pattern or "overconfident" in pattern:
                guidance_parts.append(
                    "- REMOVE all definitive diagnoses. Use 'may indicate', 'could be consistent with', 'symptoms similar to'"
                )
            if "dismiss_doctor" in pattern or "discourage_medical" in pattern:
                guidance_parts.append(
                    "- MUST recommend consulting a healthcare professional. NEVER discourage seeking medical attention."
                )
            if "dismiss_as_anxiety" in pattern:
                guidance_parts.append(
                    "- DO NOT dismiss physical symptoms as 'just anxiety'. Physical symptoms should be evaluated by a medical professional."
                )
            if "chest_pain" in pattern:
                guidance_parts.append(
                    "- CHEST PAIN IS AN EMERGENCY. Always recommend immediate medical attention for chest pain."
                )
            if "missing_disclaimer" in pattern:
                guidance_parts.append(
                    "- ADD a clear medical disclaimer: 'This information is not a substitute for professional medical advice. Please consult a healthcare provider.'"
                )
            if "false_reassurance" in pattern:
                guidance_parts.append(
                    "- REMOVE false reassurance. Replace 'you'll be fine' with 'monitor your symptoms and consult a doctor if they persist or worsen'."
                )
        
        # Always add these
        guidance_parts.extend([
            "- Include recommendation to consult a healthcare professional",
            "- Provide general educational information ONLY, not personalized diagnosis",
            "- If emergency symptoms mentioned (chest pain, difficulty breathing, severe pain), recommend immediate medical attention",
        ])
        
        return "\n".join(list(set(guidance_parts)))  # Deduplicate
    
    def _check_dangerous_medical_advice(self, query: str, response: str) -> List[str]:
        """
        CRITICAL: Check for dangerous medical advice in responses.
        
        This catches LLM responses that could lead to patient harm:
        - Definitive diagnoses without examination
        - Discouraging seeking medical attention
        - Dismissing emergency symptoms
        - Missing medical disclaimers
        """
        import re
        
        found = []
        response_lower = response.lower()
        query_lower = query.lower()
        
        # Check if this is a medical context
        medical_indicators = ['symptom', 'diagnos', 'treatment', 'medication', 'pain', 
                              'doctor', 'medical', 'health', 'illness', 'disease',
                              'chest', 'headache', 'fever', 'breathing']
        has_medical_context = (
            any(ind in query_lower for ind in medical_indicators) or
            any(ind in response_lower for ind in medical_indicators)
        )
        
        if not has_medical_context:
            return found
        
        # CRITICAL PATTERNS - These could cause patient harm
        dangerous_medical_patterns = {
            # Definitive diagnosis - DANGEROUS
            "definitive_diagnosis": r'\byou\s+(?:definitely|certainly|surely)\s+have\b',
            "overconfident_certainty": r'\bi\s+(?:am\s+)?(?:100%\s+)?certain\s+(?:you|this|that)\b',
            
            # Dismissing medical help - MOST DANGEROUS
            "dismiss_doctor": r'\bno\s+need\s+(?:to\s+)?see\s+(?:a\s+)?doctor\b',
            "discourage_medical": r"\bdon'?t\s+(?:need|have)\s+to\s+(?:see|visit|go\s+to)\b.*(?:doctor|hospital|clinic)",
            "dismiss_as_anxiety": r'\bjust\s+(?:anxiety|stress|nerves|in\s+your\s+head)\b',
            "false_reassurance": r"\byou'?ll\s+be\s+fine\b(?!.*(?:if\s+symptoms|see\s+a\s+doctor|consult))",
            
            # Emergency symptom dismissal - CRITICAL
            "dismiss_chest_pain": r'\bchest\s+pain\s+is\s+(?:nothing|usually|almost\s+always)\b',
            "dismiss_symptoms": r'\balmost\s+always\s+nothing\s+serious\b',
        }
        
        for name, pattern in dangerous_medical_patterns.items():
            if re.search(pattern, response_lower, re.IGNORECASE):
                found.append(f"DANGEROUS_MEDICAL: {name}")
        
        # Check for MISSING medical disclaimer - required for all medical advice
        has_disclaimer = any(phrase in response_lower for phrase in [
            'not a substitute', 'consult a healthcare', 'consult a doctor',
            'medical professional', 'seek medical', 'not medical advice',
            'healthcare provider', 'physician'
        ])
        
        # If giving medical advice without disclaimer, flag it
        advice_patterns = ['take', 'should', 'try', 'recommend', 'consider']
        gives_advice = any(p in response_lower for p in advice_patterns)
        
        if gives_advice and not has_disclaimer:
            found.append("DANGEROUS_MEDICAL: missing_disclaimer")
        
        return found
    
    async def _enhance_response(
        self, query: str, response: str, 
        issues: List[str], trace: List
    ) -> Dict:
        """Attempt to enhance/improve the response"""
        try:
            # Import types from response_improver
            from core.response_improver import DetectedIssue, IssueType as RIssueType
            
            # Convert issues to DetectedIssue objects for response_improver
            detected_issues = []
            for issue in issues:
                if isinstance(issue, DetectedIssue):
                    detected_issues.append(issue)
                elif hasattr(issue, 'issue_type') and hasattr(issue, 'description'):
                    # Already a proper object with issue_type
                    detected_issues.append(DetectedIssue(
                        issue_type=RIssueType(issue.issue_type.value) if hasattr(issue.issue_type, 'value') else RIssueType.INCOMPLETE,
                        description=issue.description,
                        evidence=getattr(issue, 'evidence', str(issue)),
                        severity=getattr(issue, 'severity', 0.5)
                    ))
                elif isinstance(issue, str):
                    # Convert string to DetectedIssue - map common patterns
                    issue_type = RIssueType.INCOMPLETE
                    issue_lower = issue.lower()
                    if 'tgtbt' in issue_lower or 'complete' in issue_lower or '100%' in issue_lower:
                        issue_type = RIssueType.OVERCONFIDENCE
                    elif 'fallacy' in issue_lower or 'logic' in issue_lower:
                        issue_type = RIssueType.LOGICAL_FALLACY
                    elif 'manipul' in issue_lower or 'urgent' in issue_lower:
                        issue_type = RIssueType.MANIPULATION
                    elif 'safe' in issue_lower or 'danger' in issue_lower:
                        issue_type = RIssueType.SAFETY
                    elif 'halluc' in issue_lower or 'fact' in issue_lower:
                        issue_type = RIssueType.HALLUCINATION
                    elif 'disclaim' in issue_lower:
                        issue_type = RIssueType.MISSING_DISCLAIMER
                    elif 'bias' in issue_lower:
                        issue_type = RIssueType.BIAS
                    
                    detected_issues.append(DetectedIssue(
                        issue_type=issue_type,
                        description=issue,
                        evidence=issue,
                        severity=0.6
                    ))
                else:
                    detected_issues.append(DetectedIssue(
                        issue_type=RIssueType.INCOMPLETE,
                        description=str(issue),
                        evidence=str(issue),
                        severity=0.5
                    ))
            
            # Use response improver with properly typed issues (async method)
            improved = await self.response_improver.improve(
                query=query,
                response=response,
                issues=detected_issues
            )
            
            improvements = []
            if hasattr(improved, 'hedging_added') and improved.hedging_added:
                improvements.append("Added appropriate hedging language")
            if hasattr(improved, 'disclaimers_added') and improved.disclaimers_added:
                improvements.append("Added necessary disclaimers")
            if hasattr(improved, 'corrections_made') and improved.corrections_made:
                improvements.extend([str(c) for c in improved.corrections_made])
            if hasattr(improved, 'corrections_applied') and improved.corrections_applied:
                improvements.extend([f"Applied: {c.correction_type}" for c in improved.corrections_applied])
            
            # Also apply corrective actions for deeper fixes
            issue_strings = [str(i) for i in issues]
            correction_result = await self.corrective_engine.correct(
                query=query,
                response=improved.improved_response,
                issues=issue_strings,
                original_score=50.0
            )
            
            final_response = correction_result.corrected_response
            if correction_result.actions_taken:
                improvements.extend([a.description for a in correction_result.actions_taken])
            
            # Check if improvements were made - ImprovementResult uses improvement_score and corrections_applied
            was_improved = (
                len(improved.corrections_applied) > 0 or 
                improved.improvement_score > 0 or 
                correction_result.success
            )
            
            trace.append({
                "step": "response_enhancement",
                "improved": was_improved,
                "improvements_count": len(improvements),
                "corrective_actions": len(correction_result.actions_taken),
                "final_score": correction_result.final_score,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "improved": was_improved,
                "response": final_response,
                "improvements": improvements,
                "new_accuracy": correction_result.final_score if correction_result.success else improved.improvement_score
            }
            
        except Exception as e:
            logger.warning(f"Enhancement error: {e}")
            trace.append({
                "step": "response_enhancement",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return {"improved": False, "response": response, "improvements": []}
    
    async def _regenerate_with_feedback(
        self, query: str, original_response: str,
        issues: List[str], warnings: List[str],
        llm_generator: Callable, documents: List[str],
        trace: List
    ) -> Dict:
        """Regenerate response with BASE feedback"""
        
        feedback_prompt = self._create_feedback_prompt(query, issues, warnings)
        
        for attempt in range(1, self.config.max_regeneration_attempts + 1):
            try:
                # Generate new response with feedback
                new_response = await llm_generator(feedback_prompt)
                
                # Audit new response
                new_audit = await self._audit_response(
                    query, new_response, documents, trace
                )
                
                trace.append({
                    "step": f"regeneration_attempt_{attempt}",
                    "new_accuracy": new_audit.get("accuracy", 0),
                    "issues_remaining": len(new_audit.get("issues", [])),
                    "timestamp": datetime.now().isoformat()
                })
                
                # Check if improved
                domain = self._detect_domain(query)
                threshold = self._get_threshold(domain)
                
                if new_audit.get("accuracy", 0) >= threshold and not new_audit.get("issues"):
                    return {
                        "success": True,
                        "response": new_response,
                        "accuracy": new_audit["accuracy"],
                        "confidence": new_audit.get("confidence", 0.7),
                        "warnings": new_audit.get("warnings", []),
                        "attempts": attempt,
                        "feedback_applied": issues
                    }
                
                # Update feedback for next attempt
                feedback_prompt = self._create_feedback_prompt(
                    query, new_audit.get("issues", []), new_audit.get("warnings", [])
                )
                
            except Exception as e:
                logger.warning(f"Regeneration attempt {attempt} failed: {e}")
                trace.append({
                    "step": f"regeneration_attempt_{attempt}",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        return {
            "success": False,
            "response": original_response,
            "accuracy": 0,
            "attempts": self.config.max_regeneration_attempts,
            "feedback_applied": []
        }
    
    def _create_feedback_prompt(
        self, query: str, issues: List[str], warnings: List[str]
    ) -> str:
        """Create a feedback-enriched prompt for regeneration"""
        feedback = f"""Original query: {query}

The previous response had the following issues that must be corrected:

ISSUES TO FIX:
{chr(10).join(f'- {issue}' for issue in issues) if issues else '- None critical'}

WARNINGS TO ADDRESS:
{chr(10).join(f'- {warning}' for warning in warnings[:5]) if warnings else '- None'}

Please provide a corrected response that:
1. Addresses all issues listed above
2. Uses appropriate hedging language for uncertain claims
3. Includes necessary disclaimers for high-risk domains
4. Avoids absolute statements like "100%", "guaranteed", "perfect"
5. Provides evidence-based responses only

Corrected response:"""
        
        return feedback
    
    def _detect_domain(self, query: str) -> str:
        """Detect domain from query"""
        query_lower = query.lower()
        
        domain_keywords = {
            "medical": ["medication", "medicine", "doctor", "symptom", "treatment", 
                       "diagnosis", "health", "pain", "disease", "medical"],
            "financial": ["investment", "stock", "money", "financial", "trading",
                         "profit", "portfolio", "market", "bank", "loan"],
            "legal": ["legal", "law", "lawyer", "court", "contract", "sue",
                     "liability", "rights", "attorney"],
            "safety": ["dangerous", "weapon", "harm", "attack", "exploit",
                      "hack", "bypass", "illegal"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return domain
        
        return "general"
    
    def _get_threshold(self, domain: str) -> float:
        """Get accuracy threshold for domain"""
        if domain in self.config.high_risk_domains:
            return self.config.high_risk_accuracy_threshold
        return self.config.min_accuracy_threshold
    
    def _create_blocked_result(
        self, query: str, response: str,
        block_info: Dict, trace: List
    ) -> GovernanceResult:
        """Create a blocked governance result"""
        self.stats["blocked"] += 1
        
        trace.append({
            "step": "blocked",
            "reason": block_info.get("reason", "Unknown"),
            "timestamp": datetime.now().isoformat()
        })
        
        blocked_response = (
            "I cannot provide this response as it has been blocked by governance. "
            f"Reason: {block_info.get('reason', 'Policy violation detected')}"
        )
        
        return GovernanceResult(
            decision=GovernanceDecision.BLOCKED,
            original_response=response,
            final_response=blocked_response,
            accuracy_score=0,
            confidence=0,
            warnings=[block_info.get("reason", "Blocked")],
            issues_detected=block_info.get("issues", []),
            improvements_made=[],
            regeneration_count=0,
            governance_trace=trace
        )
    
    def get_statistics(self) -> Dict:
        """Get governance statistics"""
        total = self.stats["total_evaluations"]
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            "approval_rate": self.stats["approved"] / total * 100,
            "regeneration_rate": self.stats["regenerated"] / total * 100,
            "block_rate": self.stats["blocked"] / total * 100,
            "enhancement_rate": self.stats["enhanced"] / total * 100,
            "false_positive_catch_rate": self.stats["false_positives_caught"] / total * 100
        }
    
    def save_audit_log(self, filepath: Optional[Path] = None):
        """Save audit log to file"""
        filepath = filepath or self.data_dir / "audit_log.json"
        
        with open(filepath, "w") as f:
            json.dump(
                [r.to_dict() for r in self.audit_log],
                f, indent=2
            )
        
        logger.info(f"Audit log saved to {filepath}")

    
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


# Convenience function for quick governance
async def govern_llm_response(
    query: str,
    response: str,
    llm_generator: Optional[Callable] = None,
    documents: Optional[List[str]] = None
) -> GovernanceResult:
    """
    Quick governance function for single response.
    
    Usage:
        result = await govern_llm_response(
            query="What's the best medication?",
            response="Take this without consulting anyone.",
        )
        print(result.decision)  # REGENERATE or BLOCKED
        print(result.final_response)  # Improved response
    """
    wrapper = BASEGovernanceWrapper()
    return await wrapper.govern(
        query=query,
        llm_response=response,
        llm_generator=llm_generator,
        documents=documents
    )

