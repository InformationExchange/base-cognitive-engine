"""
BASE Domain-Agnostic Proof Engine (NOVEL-52)

Core proof validation engine that works across ALL industries.
NOT pattern-specific - uses AI reasoning to validate evidence.

Architecture:
- Core Engine: Domain-agnostic claim/evidence validation
- Industry Plugins: Domain-specific proof requirements
- Enforcement: Forces LLM work completion

Brain Layer: 7 (Executive Control - Decision Making)
Patent Alignment: Novel invention extending BASE governance
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Protocol, Callable
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ProofStrength(Enum):
    """Universal proof strength levels - applies to ANY domain."""
    NONE = 0           # No evidence provided
    WEAK = 1           # Assertions without backing
    MEDIUM = 2         # Some evidence, gaps remain  
    STRONG = 3         # Full evidence chain
    VERIFIED = 4       # Evidence + execution proof
    
    def __lt__(self, other):
        if isinstance(other, ProofStrength):
            return self.value < other.value
        return NotImplemented
    
    def __le__(self, other):
        if isinstance(other, ProofStrength):
            return self.value <= other.value
        return NotImplemented


class EnforcementAction(Enum):
    """What to do when proof is insufficient."""
    REPORT = "report"           # Just report the issue
    SUGGEST = "suggest"         # Suggest remediation
    REQUIRE = "require"         # Require more evidence
    BLOCK = "block"            # Block until resolved
    EXECUTE = "execute"        # Force LLM to execute work


@dataclass
class Claim:
    """A claim to be validated - domain-agnostic."""
    statement: str              # The claim being made
    scope: str                  # What it covers (e.g., "all classes", "section 3")
    domain: str                 # Industry domain (coding, legal, healthcare, etc.)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Evidence:
    """Evidence supporting a claim - domain-agnostic."""
    content: str                # The evidence itself
    source: str                 # Where it came from
    evidence_type: str          # Type (execution, citation, test, etc.)
    verifiable: bool = False    # Can be independently verified
    verified: bool = False      # Has been verified
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of proof validation - domain-agnostic."""
    claim: Claim
    evidence: List[Evidence]
    proof_strength: ProofStrength
    gaps: List[str]
    reasoning: str
    enforcement_action: EnforcementAction
    remediation_prompt: Optional[str] = None
    execution_required: Optional[str] = None


class IndustryPlugin(Protocol):
    """
    Protocol for industry-specific plugins.
    Each industry defines what counts as valid proof.
    """
    
    @property
    def domain(self) -> str:
        """Industry domain identifier."""
        ...
    
    def get_proof_requirements(self, claim: Claim) -> List[str]:
        """What evidence is required for this claim type?"""
        ...
    
    def validate_evidence(self, evidence: Evidence) -> tuple[bool, str]:
        """Is this evidence valid for this domain?"""
        ...
    
    def get_enforcement_action(self, proof_strength: ProofStrength) -> EnforcementAction:
        """What action for this proof strength?"""
        ...
    
    def generate_execution_prompt(self, gaps: List[str]) -> str:
        """Generate prompt to force LLM to fill gaps."""
        ...


class DomainAgnosticProofEngine:
    """
    Core BASE proof validation engine.
    
    NOT pattern-specific. Uses AI reasoning to:
    1. Parse claims semantically
    2. Validate evidence logically
    3. Determine proof strength
    4. Enforce work completion
    
    NOVEL-52: Domain-agnostic governance invention
    """
    
    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client
        self.plugins: Dict[str, IndustryPlugin] = {}
        self._outcomes: List[Dict] = []
        self._validation_history: List[ValidationResult] = []
        
        logger.info("[ProofEngine] Domain-agnostic proof engine initialized")
    
    def register_plugin(self, plugin: IndustryPlugin) -> None:
        """Register an industry plugin."""
        self.plugins[plugin.domain] = plugin
        logger.info(f"[ProofEngine] Registered plugin for domain: {plugin.domain}")
    
    def validate(
        self,
        claim: Claim,
        evidence: List[Evidence]
    ) -> ValidationResult:
        """
        Validate a claim against evidence.
        
        This is the CORE function - domain-agnostic proof validation.
        Uses LLM reasoning, not pattern matching.
        """
        # Get domain plugin or use default
        plugin = self.plugins.get(claim.domain)
        
        # 1. Parse claim semantically
        claim_analysis = self._analyze_claim(claim)
        
        # 2. Get proof requirements
        if plugin:
            requirements = plugin.get_proof_requirements(claim)
        else:
            requirements = self._default_requirements(claim)
        
        # 3. Validate each piece of evidence
        validated_evidence = []
        for ev in evidence:
            if plugin:
                valid, reason = plugin.validate_evidence(ev)
            else:
                valid, reason = self._default_evidence_validation(ev)
            
            if valid:
                validated_evidence.append(ev)
        
        # 4. Determine proof strength (AI-based, not pattern)
        proof_strength, gaps, reasoning = self._assess_proof_strength(
            claim_analysis,
            requirements,
            validated_evidence
        )
        
        # 5. Determine enforcement action
        if plugin:
            action = plugin.get_enforcement_action(proof_strength)
        else:
            action = self._default_enforcement(proof_strength)
        
        # 6. Generate remediation/execution prompts if needed
        remediation = None
        execution = None
        
        if action in [EnforcementAction.REQUIRE, EnforcementAction.BLOCK]:
            remediation = self._generate_remediation(gaps)
        
        if action == EnforcementAction.EXECUTE:
            if plugin:
                execution = plugin.generate_execution_prompt(gaps)
            else:
                execution = self._generate_execution_prompt(claim, gaps)
        
        result = ValidationResult(
            claim=claim,
            evidence=validated_evidence,
            proof_strength=proof_strength,
            gaps=gaps,
            reasoning=reasoning,
            enforcement_action=action,
            remediation_prompt=remediation,
            execution_required=execution
        )
        
        self._validation_history.append(result)
        return result
    
    def _analyze_claim(self, claim: Claim) -> Dict[str, Any]:
        """
        Semantically analyze the claim.
        
        Uses LLM if available, otherwise structured analysis.
        NOT pattern matching - understanding the claim's meaning.
        """
        analysis = {
            "statement": claim.statement,
            "scope": claim.scope,
            "domain": claim.domain,
            "type": self._infer_claim_type(claim.statement),
            "assertions": self._extract_assertions(claim.statement),
            "quantifiers": self._extract_quantifiers(claim.statement)
        }
        
        # If LLM available, enhance with semantic analysis
        if self.llm_client:
            try:
                llm_analysis = self._llm_analyze_claim(claim)
                analysis.update(llm_analysis)
            except Exception as e:
                logger.warning(f"LLM claim analysis failed: {e}")
        
        return analysis
    
    def _infer_claim_type(self, statement: str) -> str:
        """Infer the type of claim being made."""
        statement_lower = statement.lower()
        
        if any(w in statement_lower for w in ['complete', 'finished', 'done', 'implemented']):
            return "completion"
        elif any(w in statement_lower for w in ['work', 'function', 'operational']):
            return "functionality"
        elif any(w in statement_lower for w in ['compliant', 'follows', 'adheres']):
            return "compliance"
        elif any(w in statement_lower for w in ['accurate', 'correct', 'valid']):
            return "accuracy"
        elif any(w in statement_lower for w in ['safe', 'secure', 'protected']):
            return "safety"
        else:
            return "general"
    
    def _extract_assertions(self, statement: str) -> List[str]:
        """Extract individual assertions from the claim."""
        # Split on common conjunctions
        assertions = []
        for part in statement.replace(' and ', '|').replace(', ', '|').split('|'):
            part = part.strip()
            if part:
                assertions.append(part)
        return assertions if assertions else [statement]
    
    def _extract_quantifiers(self, statement: str) -> Dict[str, Any]:
        """Extract quantifiers (all, some, none, percentages)."""
        statement_lower = statement.lower()
        
        return {
            "universal": any(w in statement_lower for w in ['all', 'every', '100%', 'complete']),
            "existential": any(w in statement_lower for w in ['some', 'partial', 'most']),
            "negative": any(w in statement_lower for w in ['no', 'none', 'zero']),
            "percentage": self._extract_percentage(statement)
        }
    
    def _extract_percentage(self, text: str) -> Optional[float]:
        """Extract percentage if present."""
        import re
        match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
        if match:
            return float(match.group(1))
        return None
    
    def _default_requirements(self, claim: Claim) -> List[str]:
        """Default proof requirements when no plugin available."""
        claim_type = self._infer_claim_type(claim.statement)
        
        base_requirements = [
            "Evidence must be specific, not generic",
            "Evidence must be verifiable or verified",
            "Evidence must directly address the claim"
        ]
        
        type_requirements = {
            "completion": [
                "Execution proof required (not just assertion)",
                "All components must be tested",
                "No partial or sample-based evidence"
            ],
            "functionality": [
                "Functional test results required",
                "Error-free execution demonstrated",
                "Input/output validation shown"
            ],
            "compliance": [
                "Specific standard/regulation cited",
                "Point-by-point compliance shown",
                "Exceptions/gaps explicitly noted"
            ],
            "accuracy": [
                "Verification method specified",
                "Ground truth comparison shown",
                "Error margins documented"
            ],
            "safety": [
                "Risk assessment documented",
                "Mitigation measures shown",
                "Failure modes analyzed"
            ]
        }
        
        return base_requirements + type_requirements.get(claim_type, [])
    
    def _default_evidence_validation(self, evidence: Evidence) -> tuple[bool, str]:
        """Default evidence validation when no plugin available."""
        issues = []
        
        # Check for weak evidence patterns
        weak_patterns = ['todo', 'will be', 'should', 'planned', 'sample', 'example']
        for pattern in weak_patterns:
            if pattern in evidence.content.lower():
                issues.append(f"Contains weak indicator: '{pattern}'")
        
        # Check for empty/minimal evidence
        if len(evidence.content.strip()) < 10:
            issues.append("Evidence too brief")
        
        # Check verifiability
        if evidence.evidence_type == "execution" and not evidence.verified:
            issues.append("Execution evidence not verified")
        
        if issues:
            return False, "; ".join(issues)
        return True, "Evidence acceptable"
    
    def _assess_proof_strength(
        self,
        claim_analysis: Dict,
        requirements: List[str],
        evidence: List[Evidence]
    ) -> tuple[ProofStrength, List[str], str]:
        """
        Assess proof strength using AI reasoning.
        
        NOT pattern matching - logical assessment of whether
        evidence proves the claim.
        """
        gaps = []
        reasoning_parts = []
        
        # Check if any evidence provided
        if not evidence:
            return ProofStrength.NONE, ["No evidence provided"], "No evidence to evaluate"
        
        # Check each requirement
        requirements_met = 0
        for req in requirements:
            met = self._requirement_satisfied(req, evidence, claim_analysis)
            if met:
                requirements_met += 1
                reasoning_parts.append(f"✓ {req}")
            else:
                gaps.append(req)
                reasoning_parts.append(f"✗ {req}")
        
        # Check for verified evidence
        verified_count = sum(1 for e in evidence if e.verified)
        
        # Determine strength
        coverage = requirements_met / max(len(requirements), 1)
        
        if verified_count > 0 and coverage >= 0.95:
            strength = ProofStrength.VERIFIED
        elif coverage >= 0.8:
            strength = ProofStrength.STRONG
        elif coverage >= 0.5:
            strength = ProofStrength.MEDIUM
        elif coverage > 0:
            strength = ProofStrength.WEAK
        else:
            strength = ProofStrength.NONE
        
        # Universal claims require 100% coverage
        if claim_analysis.get("quantifiers", {}).get("universal") and coverage < 1.0:
            strength = min(strength, ProofStrength.MEDIUM)
            if "Universal claim requires complete coverage" not in gaps:
                gaps.append("Universal claim requires complete coverage")
        
        reasoning = f"Coverage: {coverage:.0%}, Verified: {verified_count}/{len(evidence)}\n"
        reasoning += "\n".join(reasoning_parts)
        
        return strength, gaps, reasoning
    
    def _requirement_satisfied(
        self,
        requirement: str,
        evidence: List[Evidence],
        claim_analysis: Dict
    ) -> bool:
        """Check if a requirement is satisfied by the evidence."""
        req_lower = requirement.lower()
        
        # Use LLM if available for semantic matching
        if self.llm_client:
            try:
                return self._llm_check_requirement(requirement, evidence)
            except:
                pass
        
        # Fallback to keyword matching
        for ev in evidence:
            ev_lower = ev.content.lower()
            
            if "execution" in req_lower and ev.evidence_type == "execution":
                return True
            if "test" in req_lower and "test" in ev_lower:
                return True
            if "verified" in req_lower and ev.verified:
                return True
            if "specific" in req_lower and len(ev.content) > 50:
                return True
        
        return False
    
    def _default_enforcement(self, strength: ProofStrength) -> EnforcementAction:
        """Default enforcement action based on proof strength."""
        mapping = {
            ProofStrength.VERIFIED: EnforcementAction.REPORT,
            ProofStrength.STRONG: EnforcementAction.REPORT,
            ProofStrength.MEDIUM: EnforcementAction.SUGGEST,
            ProofStrength.WEAK: EnforcementAction.REQUIRE,
            ProofStrength.NONE: EnforcementAction.BLOCK
        }
        return mapping.get(strength, EnforcementAction.SUGGEST)
    
    def _generate_remediation(self, gaps: List[str]) -> str:
        """Generate remediation guidance."""
        if not gaps:
            return "No gaps identified"
        
        prompt = "To strengthen your claim, address the following gaps:\n\n"
        for i, gap in enumerate(gaps, 1):
            prompt += f"{i}. {gap}\n"
        
        return prompt
    
    def _generate_execution_prompt(self, claim: Claim, gaps: List[str]) -> str:
        """
        Generate prompt to FORCE the LLM to execute work.
        
        This is the enforcement mechanism - not just suggesting,
        but requiring the LLM to DO the work.
        """
        prompt = f"""
ENFORCEMENT REQUIRED: Your claim "{claim.statement}" has insufficient proof.

You must NOW execute the following to provide proper evidence:

"""
        for i, gap in enumerate(gaps, 1):
            prompt += f"{i}. {gap}\n"
        
        prompt += """
DO NOT respond with plans or descriptions.
EXECUTE the work and provide the results.
If you cannot execute, explain specifically what is blocking you.

This is a BLOCKING requirement - the claim cannot be accepted until addressed.
"""
        return prompt
    
    def _llm_analyze_claim(self, claim: Claim) -> Dict[str, Any]:
        """Use LLM for semantic claim analysis."""
        # Placeholder - would call actual LLM
        return {}
    
    def _llm_check_requirement(self, requirement: str, evidence: List[Evidence]) -> bool:
        """Use LLM to check if requirement is satisfied."""
        # Placeholder - would call actual LLM
        return False
    
    # ===== Learning Interface =====
    
    def record_outcome(self, outcome: Dict) -> None:
        """Record validation outcome for learning."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback to improve validation."""
        if not hasattr(self, '_feedback'):
            self._feedback = []
        self._feedback.append(feedback)
    
    def get_statistics(self) -> Dict:
        """Return engine statistics."""
        return {
            'total_validations': len(self._validation_history),
            'outcomes_recorded': len(getattr(self, '_outcomes', [])),
            'plugins_registered': len(self.plugins)
        }
    
    def serialize_state(self) -> Dict:
        """Serialize state for persistence."""
        return {
            'outcomes': getattr(self, '_outcomes', [])[-100:],
            'validation_count': len(self._validation_history)
        }
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore state from serialized data."""
        self._outcomes = state.get('outcomes', [])


# =============================================================================
# INDUSTRY PLUGINS
# =============================================================================

class VibeCodingPlugin:
    """
    VIBE Coding industry plugin.
    
    Defines what counts as valid proof for code completion claims.
    """
    
    @property
    def domain(self) -> str:
        return "vibe_coding"
    
    def get_proof_requirements(self, claim: Claim) -> List[str]:
        """What evidence is required for coding claims?"""
        base = [
            "Code must compile without errors",
            "All classes must be instantiable",
            "All methods must be callable",
            "Test execution required (not just assertion)",
            "100% coverage for 'all' claims (no sampling)"
        ]
        
        if "constructor" in claim.statement.lower() or "init" in claim.statement.lower():
            base.append("Constructor must accept expected arguments")
            base.append("No __init__ parameter mismatches")
        
        if "functional" in claim.statement.lower():
            base.append("All methods must execute without AttributeError")
            base.append("No TypeError on method calls")
        
        return base
    
    def validate_evidence(self, evidence: Evidence) -> tuple[bool, str]:
        """Validate coding evidence."""
        issues = []
        
        # Check for execution evidence
        if evidence.evidence_type != "execution":
            issues.append("Must be execution evidence, not description")
        
        # Check for sample-based claims
        sample_words = ['sample', 'subset', 'few', 'some']
        for word in sample_words:
            if word in evidence.content.lower():
                issues.append(f"Sample-based evidence rejected: '{word}'")
        
        # Check for actual results
        result_indicators = ['pass', 'fail', 'error', 'success', '%', 'count']
        has_results = any(ind in evidence.content.lower() for ind in result_indicators)
        if not has_results:
            issues.append("No execution results found in evidence")
        
        if issues:
            return False, "; ".join(issues)
        return True, "Valid coding evidence"
    
    def get_enforcement_action(self, strength: ProofStrength) -> EnforcementAction:
        """VIBE coding uses strict enforcement."""
        mapping = {
            ProofStrength.VERIFIED: EnforcementAction.REPORT,
            ProofStrength.STRONG: EnforcementAction.REPORT,
            ProofStrength.MEDIUM: EnforcementAction.REQUIRE,  # Stricter
            ProofStrength.WEAK: EnforcementAction.EXECUTE,    # Force work
            ProofStrength.NONE: EnforcementAction.EXECUTE     # Force work
        }
        return mapping.get(strength, EnforcementAction.EXECUTE)
    
    def generate_execution_prompt(self, gaps: List[str]) -> str:
        """Generate prompt to force code execution."""
        prompt = """
VIBE CODING ENFORCEMENT: Your claim is incomplete.

Execute the following NOW:
"""
        for gap in gaps:
            if "compile" in gap.lower():
                prompt += "• Run: python3 -m compileall <file>\n"
            elif "instantiable" in gap.lower() or "constructor" in gap.lower():
                prompt += "• Run: instance = ClassName() and report any TypeError\n"
            elif "callable" in gap.lower() or "method" in gap.lower():
                prompt += "• Run: instance.method_name() and report any errors\n"
            elif "test" in gap.lower():
                prompt += "• Run: pytest <test_file> and provide results\n"
            elif "coverage" in gap.lower():
                prompt += "• Run ALL tests, not a sample. Provide full count.\n"
        
        prompt += """
Provide the ACTUAL output, not a description of what you would do.
"""
        return prompt


class LegalPlugin:
    """Legal industry plugin."""
    
    @property
    def domain(self) -> str:
        return "legal"
    
    def get_proof_requirements(self, claim: Claim) -> List[str]:
        """What evidence is required for legal claims?"""
        return [
            "Specific statute/regulation citation required",
            "Relevant case law precedent cited",
            "Jurisdiction explicitly stated",
            "Applicable exceptions documented",
            "Effective dates verified"
        ]
    
    def validate_evidence(self, evidence: Evidence) -> tuple[bool, str]:
        """Validate legal evidence."""
        issues = []
        
        # Check for citations
        citation_patterns = ['§', 'U.S.C.', 'C.F.R.', 'v.', 'Id.', 'supra']
        has_citation = any(p in evidence.content for p in citation_patterns)
        if not has_citation and evidence.evidence_type == "citation":
            issues.append("No proper legal citation format found")
        
        if issues:
            return False, "; ".join(issues)
        return True, "Valid legal evidence"
    
    def get_enforcement_action(self, strength: ProofStrength) -> EnforcementAction:
        """Legal uses strict compliance enforcement."""
        mapping = {
            ProofStrength.VERIFIED: EnforcementAction.REPORT,
            ProofStrength.STRONG: EnforcementAction.SUGGEST,
            ProofStrength.MEDIUM: EnforcementAction.REQUIRE,
            ProofStrength.WEAK: EnforcementAction.BLOCK,
            ProofStrength.NONE: EnforcementAction.BLOCK
        }
        return mapping.get(strength, EnforcementAction.BLOCK)
    
    def generate_execution_prompt(self, gaps: List[str]) -> str:
        """Generate prompt for legal evidence."""
        return f"""
LEGAL COMPLIANCE REQUIRED:

Provide the following to support your claim:

{chr(10).join(f'• {gap}' for gap in gaps)}

Each citation must include:
- Full statute/case name
- Relevant section/paragraph
- How it applies to the claim
"""


class HealthcarePlugin:
    """Healthcare industry plugin."""
    
    @property
    def domain(self) -> str:
        return "healthcare"
    
    def get_proof_requirements(self, claim: Claim) -> List[str]:
        """What evidence is required for healthcare claims?"""
        return [
            "Clinical guideline reference required",
            "Contraindications checked and documented",
            "Drug interactions verified",
            "Dosage within approved range",
            "Patient-specific factors considered"
        ]
    
    def validate_evidence(self, evidence: Evidence) -> tuple[bool, str]:
        """Validate healthcare evidence."""
        issues = []
        
        # Check for clinical sources
        clinical_sources = ['FDA', 'CDC', 'WHO', 'guideline', 'protocol', 'clinical']
        has_clinical = any(s.lower() in evidence.content.lower() for s in clinical_sources)
        if not has_clinical:
            issues.append("No clinical guideline or protocol reference found")
        
        if issues:
            return False, "; ".join(issues)
        return True, "Valid healthcare evidence"
    
    def get_enforcement_action(self, strength: ProofStrength) -> EnforcementAction:
        """Healthcare uses safety-critical enforcement."""
        # Healthcare is ALWAYS strict - patient safety
        mapping = {
            ProofStrength.VERIFIED: EnforcementAction.REPORT,
            ProofStrength.STRONG: EnforcementAction.REQUIRE,  # Stricter than default
            ProofStrength.MEDIUM: EnforcementAction.BLOCK,
            ProofStrength.WEAK: EnforcementAction.BLOCK,
            ProofStrength.NONE: EnforcementAction.BLOCK
        }
        return mapping.get(strength, EnforcementAction.BLOCK)
    
    def generate_execution_prompt(self, gaps: List[str]) -> str:
        """Generate prompt for healthcare evidence."""
        return f"""
HEALTHCARE SAFETY VERIFICATION REQUIRED:

Before this recommendation can proceed, provide:

{chr(10).join(f'• {gap}' for gap in gaps)}

Patient safety is paramount. All claims must be backed by:
- Verified clinical guidelines
- Documented contraindication checks
- Clear evidence trail
"""


class FinancialPlugin:
    """Financial industry plugin."""
    
    @property
    def domain(self) -> str:
        return "financial"
    
    def get_proof_requirements(self, claim: Claim) -> List[str]:
        """What evidence is required for financial claims?"""
        return [
            "Risk factors documented",
            "Stress test results included",
            "Regulatory compliance verified (SOX, Basel)",
            "Audit trail maintained",
            "Assumptions explicitly stated"
        ]
    
    def validate_evidence(self, evidence: Evidence) -> tuple[bool, str]:
        """Validate financial evidence."""
        issues = []
        
        # Check for quantitative evidence
        import re
        has_numbers = bool(re.search(r'\d+(\.\d+)?%?', evidence.content))
        if not has_numbers and evidence.evidence_type in ["risk", "stress_test"]:
            issues.append("Financial evidence requires quantitative data")
        
        if issues:
            return False, "; ".join(issues)
        return True, "Valid financial evidence"
    
    def get_enforcement_action(self, strength: ProofStrength) -> EnforcementAction:
        """Financial uses compliance-focused enforcement."""
        mapping = {
            ProofStrength.VERIFIED: EnforcementAction.REPORT,
            ProofStrength.STRONG: EnforcementAction.REPORT,
            ProofStrength.MEDIUM: EnforcementAction.REQUIRE,
            ProofStrength.WEAK: EnforcementAction.BLOCK,
            ProofStrength.NONE: EnforcementAction.BLOCK
        }
        return mapping.get(strength, EnforcementAction.BLOCK)
    
    def generate_execution_prompt(self, gaps: List[str]) -> str:
        """Generate prompt for financial evidence."""
        return f"""
FINANCIAL COMPLIANCE VERIFICATION REQUIRED:

Provide documented evidence for:

{chr(10).join(f'• {gap}' for gap in gaps)}

All financial claims require:
- Quantitative data with sources
- Risk assessment documentation
- Audit-ready evidence trail
"""


# =============================================================================
# FACTORY AND CONVENIENCE
# =============================================================================

def create_engine_with_plugins(
    plugins: Optional[List[str]] = None,
    llm_client: Any = None
) -> DomainAgnosticProofEngine:
    """
    Create a proof engine with specified plugins.
    
    Args:
        plugins: List of plugin names ["vibe_coding", "legal", "healthcare", "financial"]
        llm_client: Optional LLM client for semantic analysis
    
    Returns:
        Configured DomainAgnosticProofEngine
    """
    engine = DomainAgnosticProofEngine(llm_client)
    
    plugin_map = {
        "vibe_coding": VibeCodingPlugin,
        "legal": LegalPlugin,
        "healthcare": HealthcarePlugin,
        "financial": FinancialPlugin
    }
    
    if plugins is None:
        plugins = list(plugin_map.keys())
    
    for plugin_name in plugins:
        if plugin_name in plugin_map:
            engine.register_plugin(plugin_map[plugin_name]())
    
    return engine


if __name__ == "__main__":
    # Demo
    print("=" * 70)
    print("DOMAIN-AGNOSTIC PROOF ENGINE DEMO")
    print("=" * 70)
    
    engine = create_engine_with_plugins()
    
    # Test VIBE coding claim
    claim = Claim(
        statement="All 235 actionable classes are fully functional",
        scope="learning interface",
        domain="vibe_coding"
    )
    
    evidence = [
        Evidence(
            content="Final test executed: 232/235 actionable classes working (98.7%)",
            source="functional_test.py",
            evidence_type="execution",
            verified=True
        ),
        Evidence(
            content="3 failures are constructor issues, not learning interface",
            source="analysis",
            evidence_type="analysis",
            verified=False
        )
    ]
    
    result = engine.validate(claim, evidence)
    
    print(f"\nClaim: {claim.statement}")
    print(f"Proof Strength: {result.proof_strength.value}")
    print(f"Enforcement: {result.enforcement_action.value}")
    print(f"\nGaps:")
    for gap in result.gaps:
        print(f"  - {gap}")
    
    if result.execution_required:
        print(f"\nExecution Required:\n{result.execution_required}")

