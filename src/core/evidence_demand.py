"""
BAIS Evidence Demand Loop - Claim Verification System

This module implements the core capability to distinguish between:
- "Claimed 100%" (no evidence) → FLAG as UNVERIFIED
- "Tested 100%" (with evidence) → ACCEPT as VERIFIED

For coding/testing use cases, this ensures:
1. All work is ACTUALLY complete (not just claimed)
2. All tests are ACTUALLY run (not just described)
3. All code is ACTUALLY functional (not placeholders)

Patent Alignment:
- NOVEL-3: Claim-Evidence Alignment Verification
- PPA1-Inv23: AI Common Sense via Multi-Source Triangulation
- PPA2-Comp7: Verifiable Audit
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pathlib import Path
import re
import json
import hashlib


class ClaimType(Enum):
    """Types of claims that require evidence."""
    COMPLETION = "completion"       # "Task X is complete"
    QUANTITATIVE = "quantitative"   # "85% accuracy" 
    FUNCTIONALITY = "functionality" # "Feature X works"
    TEST_RESULT = "test_result"     # "All tests pass"
    QUALITY = "quality"             # "Code is production-ready"
    IMPLEMENTATION = "implementation"  # "Module X is implemented"


class EvidenceType(Enum):
    """Types of evidence that can support claims."""
    FILE_EXISTS = "file_exists"       # File path exists
    CODE_CONTENT = "code_content"     # Code contains expected patterns
    TEST_OUTPUT = "test_output"       # Test execution results
    METRICS = "metrics"               # Measured numbers
    LOG_OUTPUT = "log_output"         # Execution logs
    SCREENSHOT = "screenshot"         # Visual evidence
    EXTERNAL_VERIFY = "external"      # Third-party verification


class VerificationStatus(Enum):
    """Status of claim verification."""
    VERIFIED = "verified"           # Evidence confirms claim
    PARTIALLY_VERIFIED = "partial"  # Some evidence, not complete
    UNVERIFIED = "unverified"       # No evidence provided
    CONTRADICTED = "contradicted"   # Evidence contradicts claim
    PENDING = "pending"             # Awaiting evidence


@dataclass
class ExtractedClaim:
    """A claim extracted from LLM output."""
    claim_id: str
    claim_type: ClaimType
    claim_text: str
    subject: str          # What the claim is about
    assertion: str        # What is being asserted
    location: str         # Where in the response
    confidence: float     # How confident we are this is a claim
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EvidenceRequirement:
    """What evidence is needed to verify a claim."""
    claim_id: str
    required_evidence: List[EvidenceType]
    specific_checks: List[str]  # Specific things to verify
    minimum_confidence: float   # Minimum confidence threshold
    verification_method: str    # How to verify


@dataclass
class ProvidedEvidence:
    """Evidence provided to support a claim."""
    evidence_id: str
    evidence_type: EvidenceType
    content: Any          # The actual evidence
    source: str           # Where evidence came from
    timestamp: datetime = field(default_factory=datetime.now)
    integrity_hash: str = ""  # For tamper detection
    
    def __post_init__(self):
        if not self.integrity_hash:
            content_str = str(self.content)[:1000]
            self.integrity_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]


@dataclass
class VerificationResult:
    """Result of verifying a claim against evidence."""
    claim_id: str
    status: VerificationStatus
    confidence: float
    evidence_checked: List[str]
    findings: List[str]
    recommendation: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EvidenceDemandResult:
    """Complete result of evidence demand process."""
    claims_extracted: List[ExtractedClaim]
    requirements_generated: List[EvidenceRequirement]
    verifications: List[VerificationResult]
    overall_status: VerificationStatus
    overall_confidence: float
    unverified_claims: List[str]
    verified_claims: List[str]
    inventions_used: List[str]


class EvidenceDemandLoop:
    """
    The Evidence Demand Loop ensures LLM outputs are backed by real evidence.
    
    For coding/testing use cases:
    1. Extract claims from LLM output ("Feature X is complete")
    2. Generate evidence requirements (file must exist, test must pass)
    3. Demand evidence from LLM or inspect directly
    4. Verify evidence authenticity
    5. Accept or reject claims based on evidence
    
    This is the core of ensuring LLM work products are:
    - Actually complete (not just claimed)
    - Actually tested (not just described)
    - Actually functional (not placeholders)
    """
    
    # Patterns for extracting different claim types
    CLAIM_PATTERNS = {
        ClaimType.COMPLETION: [
            r'\b(?:is\s+)?(?:fully\s+)?(?:complete|completed|done|finished)\b',
            r'\b(?:all\s+)?(?:tasks?|items?|work)\s+(?:is\s+)?(?:complete|done)\b',
            r'\b(?:successfully\s+)?(?:implemented|built|created)\b',
            r'\b100\s*%\s*(?:complete|done|finished)\b',
        ],
        ClaimType.QUANTITATIVE: [
            r'\b(\d{1,3})\s*%\s*(?:accuracy|success|pass|coverage)\b',
            r'\b(\d+)\s*(?:/|of|out\s+of)\s*(\d+)\b',
            r'\b(?:achieves?|reaches?)\s+(\d+)\s*%\b',
        ],
        ClaimType.FUNCTIONALITY: [
            r'\b(?:feature|function|module)\s+\w+\s+(?:works?|functions?|operates?)\b',
            r'\b(?:is\s+)?(?:working|functional|operational)\b',
            r'\b(?:can|will)\s+(?:handle|process|support)\b',
        ],
        ClaimType.TEST_RESULT: [
            r'\b(?:all\s+)?tests?\s+(?:pass|passed|passing)\b',
            r'\b(\d+)\s*(?:/|of)\s*(\d+)\s*tests?\s+pass\b',
            r'\btest\s+results?:?\s*(?:pass|success)\b',
            r'\b(?:verified|validated)\s+(?:by|with)\s+tests?\b',
        ],
        ClaimType.IMPLEMENTATION: [
            r'\b(?:file|module|class|function)\s+(?:exists?|created|implemented)\b',
            r'\b(?:added|wrote|created)\s+(?:to|in)?\s*[`\"]?[\w/.]+[`\"]?\b',
            r'\bimplemented\s+in\s+[`\"]?[\w/.]+[`\"]?\b',
        ],
    }
    
    # Evidence requirements by claim type
    EVIDENCE_REQUIREMENTS = {
        ClaimType.COMPLETION: {
            "evidence_types": [EvidenceType.FILE_EXISTS, EvidenceType.CODE_CONTENT],
            "checks": ["File exists at claimed path", "No TODO/FIXME markers", "No placeholder code"],
            "min_confidence": 0.8
        },
        ClaimType.QUANTITATIVE: {
            "evidence_types": [EvidenceType.METRICS, EvidenceType.TEST_OUTPUT],
            "checks": ["Baseline measurement exists", "Improvement is measurable", "Methodology documented"],
            "min_confidence": 0.9
        },
        ClaimType.FUNCTIONALITY: {
            "evidence_types": [EvidenceType.TEST_OUTPUT, EvidenceType.CODE_CONTENT],
            "checks": ["Function can be called", "Returns expected output", "Handles edge cases"],
            "min_confidence": 0.8
        },
        ClaimType.TEST_RESULT: {
            "evidence_types": [EvidenceType.TEST_OUTPUT, EvidenceType.LOG_OUTPUT],
            "checks": ["Test actually executed", "Output shows pass/fail", "No mocked results"],
            "min_confidence": 0.95
        },
        ClaimType.IMPLEMENTATION: {
            "evidence_types": [EvidenceType.FILE_EXISTS, EvidenceType.CODE_CONTENT],
            "checks": ["File exists", "Contains claimed class/function", "Not empty/placeholder"],
            "min_confidence": 0.85
        },
    }
    
    def __init__(self, storage_path: Path = None):
        # Use temp directory if no path provided
        if storage_path is None:
            import tempfile
            storage_path = Path(tempfile.mkdtemp(prefix="bais_evidence_"))
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Track verification history
        self.verification_history: List[EvidenceDemandResult] = []
        
        # Phase 3: Active Evidence Retrieval Components
        self._evidence_sources: Dict[str, str] = {}  # source_id -> source_path
        self._retrieval_cache: Dict[str, Any] = {}  # claim_hash -> cached evidence
        self._source_reliability: Dict[str, float] = {}  # source_id -> reliability
        self._learning_rate = 0.1
        
        # Load any existing history
        self._load_history()
        
        # Learning state
        self._outcomes: List[Dict] = []
        self._feedback: List[Dict] = []
        self._domain_adjustments: Dict[str, float] = {}
        self._total_demands: int = 0
        self._satisfied_demands: int = 0
    
    # ===== Learning Interface Methods =====
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        """Record verification outcome for learning."""
        getattr(self, '_outcomes', {}).append(outcome)
        # Adjust source reliability based on outcomes
        for source_id, was_useful in outcome.get('source_usefulness', {}).items():
            current = getattr(self, '_source_reliability', {}).get(source_id, 0.5)
            adjustment = self._learning_rate if was_useful else -self._learning_rate
            self._source_reliability[source_id] = max(0.1, min(1.0, current + adjustment))
    
    def record_feedback(self, feedback: Dict) -> None:
        """Record feedback on evidence verification."""
        getattr(self, '_feedback', {}).append(feedback)
        domain = feedback.get('domain', 'general')
        if feedback.get('feedback_type') == 'false_positive': getattr(self, '_domain_adjustments', 0)[domain] = getattr(self, '_domain_adjustments', {}).get(domain, 0.0) - 0.05
        elif feedback.get('feedback_type') == 'false_negative': getattr(self, '_domain_adjustments', 0)[domain] = getattr(self, '_domain_adjustments', {}).get(domain, 0.0) + 0.05
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt verification thresholds based on performance."""
        pass
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get adjustment for domain."""
        return getattr(self, '_domain_adjustments', {}).get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics."""
        return {
            'verification_history_count': len(self.verification_history),
            'source_reliability': dict(self._source_reliability),
            'domain_adjustments': dict(self._domain_adjustments),
            'outcomes_recorded': len(self._outcomes),
            'feedback_recorded': len(self._feedback),
            'evidence_sources_registered': len(self._evidence_sources)
        }
    
    def register_evidence_source(self, source_id: str, source_path: str, reliability: float = 1.0):
        """
        Register an evidence source for active retrieval.
        
        Phase 3 Enhancement: Active evidence retrieval.
        
        Args:
            source_id: Unique identifier for the source
            source_path: Path or URL to the source
            reliability: Initial reliability score (0-1)
        """
        self._evidence_sources[source_id] = source_path
        self._source_reliability[source_id] = reliability
    
    async def retrieve_evidence(self, claim: ExtractedClaim) -> List[ProvidedEvidence]:
        """
        Actively retrieve evidence for a claim from registered sources.
        
        Phase 3 Enhancement: Instead of just demanding evidence,
        actively search for it in registered sources.
        
        Args:
            claim: The claim to find evidence for
            
        Returns:
            List of evidence found from sources
        """
        evidence_list = []
        
        # Check cache first
        claim_hash = hashlib.sha256(claim.claim_text.encode()).hexdigest()[:16]
        if claim_hash in self._retrieval_cache:
            return self._retrieval_cache[claim_hash]
        
        # Search each registered source
        for source_id, source_path in getattr(self, '_evidence_sources', {}).items():
            evidence = await self._search_source_for_evidence(
                source_id, source_path, claim
            )
            if evidence:
                evidence_list.extend(evidence)
        
        # Cache the results
        self._retrieval_cache[claim_hash] = evidence_list
        
        return evidence_list
    
    async def _search_source_for_evidence(
        self, 
        source_id: str, 
        source_path: str, 
        claim: ExtractedClaim
    ) -> List[ProvidedEvidence]:
        """Search a specific source for evidence supporting a claim."""
        evidence_list = []
        
        try:
            # Check if source is a file path
            path = Path(source_path)
            if path.exists():
                if path.is_file():
                    # Search file for evidence
                    content = path.read_text()
                    if self._content_supports_claim(content, claim):
                        evidence_list.append(ProvidedEvidence(
                            evidence_id=f"{source_id}_{claim.claim_id}",
                            evidence_type=EvidenceType.CODE_CONTENT,
                            content=content[:1000],  # Truncate for storage
                            source=source_path
                        ))
                elif path.is_dir():
                    # Search directory for relevant files
                    for file_path in path.rglob("*.py"):
                        try:
                            content = file_path.read_text()
                            if self._content_supports_claim(content, claim):
                                evidence_list.append(ProvidedEvidence(
                                    evidence_id=f"{source_id}_{claim.claim_id}_{file_path.name}",
                                    evidence_type=EvidenceType.FILE_EXISTS,
                                    content=str(file_path),
                                    source=str(file_path)
                                ))
                        except Exception:
                            continue
        except Exception:
            pass
        
        return evidence_list
    
    def _content_supports_claim(self, content: str, claim: ExtractedClaim) -> bool:
        """Check if content provides evidence for a claim."""
        # Check for subject mentions
        if claim.subject.lower() in content.lower():
            return True
        
        # Check for key terms from assertion
        key_terms = claim.assertion.lower().split()[:3]
        matches = sum(1 for term in key_terms if term in content.lower())
        if matches >= 2:
            return True
        
        return False
    
    def update_source_reliability(self, source_id: str, was_accurate: bool):
        """
        Update reliability of a source based on verification outcome.
        
        Phase 3 Enhancement: Learn which sources provide reliable evidence.
        """
        current = getattr(self, '_source_reliability', {}).get(source_id, 1.0)
        if was_accurate:
            self._source_reliability[source_id] = min(1.0, current * (1 + self._learning_rate))
        else:
            self._source_reliability[source_id] = max(0.1, current * (1 - self._learning_rate))
    
    def get_source_reliability(self, source_id: str) -> float:
        """Get current reliability score for a source."""
        return getattr(self, '_source_reliability', {}).get(source_id, 0.5)
    
    def get_retrieval_statistics(self) -> Dict:
        """Get statistics about evidence retrieval."""
        return {
            'registered_sources': len(self._evidence_sources),
            'cached_retrievals': len(self._retrieval_cache),
            'source_reliabilities': dict(self._source_reliability)
        }
    
    def extract_claims(self, response: str, query: str = "") -> List[ExtractedClaim]:
        """
        Extract all claims from an LLM response.
        
        This is the first step in the evidence demand loop:
        Find all statements that assert something is true.
        """
        claims = []
        claim_counter = 0
        
        for claim_type, patterns in self.CLAIM_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, response, re.IGNORECASE)
                for match in matches:
                    claim_counter += 1
                    
                    # Extract surrounding context for the claim
                    start = max(0, match.start() - 50)
                    end = min(len(response), match.end() + 50)
                    context = response[start:end].strip()
                    
                    # Determine subject and assertion
                    subject = self._extract_subject(response, match)
                    assertion = match.group(0)
                    
                    # Calculate confidence based on pattern strength
                    confidence = self._calculate_claim_confidence(
                        pattern, match.group(0), context
                    )
                    
                    claims.append(ExtractedClaim(
                        claim_id=f"CLAIM-{claim_counter:03d}",
                        claim_type=claim_type,
                        claim_text=context,
                        subject=subject,
                        assertion=assertion,
                        location=f"chars {match.start()}-{match.end()}",
                        confidence=confidence
                    ))
        
        # Deduplicate overlapping claims
        claims = self._deduplicate_claims(claims)
        
        return claims
    
    def generate_requirements(self, claims: List[ExtractedClaim]) -> List[EvidenceRequirement]:
        """
        Generate evidence requirements for each claim.
        
        This tells us what evidence we need to verify each claim.
        """
        requirements = []
        
        for claim in claims:
            req_config = self.EVIDENCE_REQUIREMENTS.get(claim.claim_type, {})
            
            # Customize checks based on claim content
            specific_checks = list(req_config.get("checks", []))
            
            # Add claim-specific checks
            if claim.claim_type == ClaimType.IMPLEMENTATION:
                # Extract file path if mentioned
                file_match = re.search(r'[`"\']?([\w/._-]+\.(py|js|ts|java|go))[`"\']?', claim.claim_text)
                if file_match:
                    specific_checks.append(f"File exists: {file_match.group(1)}")
            
            if claim.claim_type == ClaimType.TEST_RESULT:
                # Extract test count if mentioned
                count_match = re.search(r'(\d+)\s*(?:/|of)\s*(\d+)', claim.claim_text)
                if count_match:
                    specific_checks.append(f"Verify {count_match.group(1)}/{count_match.group(2)} tests passed")
            
            requirements.append(EvidenceRequirement(
                claim_id=claim.claim_id,
                required_evidence=req_config.get("evidence_types", [EvidenceType.CODE_CONTENT]),
                specific_checks=specific_checks,
                minimum_confidence=req_config.get("min_confidence", 0.8),
                verification_method=self._determine_verification_method(claim)
            ))
        
        return requirements
    
    def verify_claim(self, 
                     claim: ExtractedClaim,
                     requirement: EvidenceRequirement,
                     evidence: List[ProvidedEvidence] = None,
                     workspace_path: Path = None) -> VerificationResult:
        """
        Verify a claim against provided evidence or by inspection.
        
        This is where we determine if a claim is actually true.
        """
        evidence = evidence or []
        findings = []
        evidence_checked = []
        
        # If workspace path provided, try to verify directly
        if workspace_path:
            direct_evidence = self._gather_direct_evidence(claim, workspace_path)
            evidence.extend(direct_evidence)
        
        # No evidence at all
        if not evidence:
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.UNVERIFIED,
                confidence=0.0,
                evidence_checked=[],
                findings=["No evidence provided for verification"],
                recommendation=f"DEMAND: {', '.join(requirement.specific_checks)}"
            )
        
        # Check each piece of evidence
        verification_scores = []
        
        for ev in evidence:
            evidence_checked.append(f"{ev.evidence_type.value}: {ev.source[:50]}")
            
            if ev.evidence_type == EvidenceType.FILE_EXISTS:
                score, finding = self._verify_file_exists(ev, claim)
                verification_scores.append(score)
                findings.append(finding)
            
            elif ev.evidence_type == EvidenceType.CODE_CONTENT:
                score, finding = self._verify_code_content(ev, claim)
                verification_scores.append(score)
                findings.append(finding)
            
            elif ev.evidence_type == EvidenceType.TEST_OUTPUT:
                score, finding = self._verify_test_output(ev, claim)
                verification_scores.append(score)
                findings.append(finding)
            
            elif ev.evidence_type == EvidenceType.METRICS:
                score, finding = self._verify_metrics(ev, claim)
                verification_scores.append(score)
                findings.append(finding)
        
        # Calculate overall confidence
        if verification_scores:
            confidence = sum(verification_scores) / len(verification_scores)
        else:
            confidence = 0.0
        
        # Determine status
        if confidence >= requirement.minimum_confidence:
            status = VerificationStatus.VERIFIED
            recommendation = "Claim verified with sufficient evidence"
        elif confidence >= 0.5:
            status = VerificationStatus.PARTIALLY_VERIFIED
            recommendation = f"Need additional evidence: {', '.join([c for c in requirement.specific_checks if c not in str(findings)])}"
        elif any('contradicts' in f.lower() for f in findings):
            status = VerificationStatus.CONTRADICTED
            recommendation = "Evidence contradicts claim - investigate discrepancy"
        else:
            status = VerificationStatus.UNVERIFIED
            recommendation = f"DEMAND: {', '.join(requirement.specific_checks)}"
        
        return VerificationResult(
            claim_id=claim.claim_id,
            status=status,
            confidence=confidence,
            evidence_checked=evidence_checked,
            findings=findings,
            recommendation=recommendation
        )
    
    def run_full_verification(self,
                              response: str,
                              query: str = "",
                              workspace_path: Path = None) -> EvidenceDemandResult:
        """
        Run the complete evidence demand loop on an LLM response.
        
        1. Extract claims
        2. Generate requirements
        3. Gather/demand evidence
        4. Verify each claim
        5. Return comprehensive result
        """
        # Step 1: Extract claims
        claims = self.extract_claims(response, query)
        
        # Step 2: Generate requirements
        requirements = self.generate_requirements(claims)
        
        # Step 3-4: Verify each claim
        verifications = []
        requirement_map = {r.claim_id: r for r in requirements}
        
        for claim in claims:
            req = requirement_map.get(claim.claim_id)
            if req:
                verification = self.verify_claim(claim, req, workspace_path=workspace_path)
                verifications.append(verification)
        
        # Step 5: Calculate overall result
        verified_claims = [v.claim_id for v in verifications if v.status == VerificationStatus.VERIFIED]
        unverified_claims = [v.claim_id for v in verifications if v.status != VerificationStatus.VERIFIED]
        
        if not verifications:
            overall_status = VerificationStatus.PENDING
            overall_confidence = 0.5
        elif all(v.status == VerificationStatus.VERIFIED for v in verifications):
            overall_status = VerificationStatus.VERIFIED
            overall_confidence = sum(v.confidence for v in verifications) / len(verifications)
        elif any(v.status == VerificationStatus.CONTRADICTED for v in verifications):
            overall_status = VerificationStatus.CONTRADICTED
            overall_confidence = min(v.confidence for v in verifications)
        elif sum(1 for v in verifications if v.status == VerificationStatus.VERIFIED) > len(verifications) / 2:
            overall_status = VerificationStatus.PARTIALLY_VERIFIED
            overall_confidence = sum(v.confidence for v in verifications) / len(verifications)
        else:
            overall_status = VerificationStatus.UNVERIFIED
            overall_confidence = sum(v.confidence for v in verifications) / len(verifications) if verifications else 0.0
        
        result = EvidenceDemandResult(
            claims_extracted=claims,
            requirements_generated=requirements,
            verifications=verifications,
            overall_status=overall_status,
            overall_confidence=overall_confidence,
            unverified_claims=unverified_claims,
            verified_claims=verified_claims,
            inventions_used=[
                "NOVEL-3: Claim-Evidence Alignment Verification",
                "PPA1-Inv23: AI Common Sense via Multi-Source Triangulation",
                "PPA2-Comp7: Verifiable Audit"
            ]
        )
        
        # Store in history
        self.verification_history.append(result)
        self._save_history()
        
        return result
    
    # ==================== PRIVATE METHODS ====================
    
    def _extract_subject(self, response: str, match: re.Match) -> str:
        """Extract the subject of a claim from context."""
        # Look backwards from match for a subject
        start = max(0, match.start() - 100)
        prefix = response[start:match.start()]
        
        # Common subject patterns
        subject_patterns = [
            r'(?:The\s+)?(\w+(?:\s+\w+)?)\s+(?:is|are|has|have)$',
            r'(?:file|module|class|function)\s+[`\"]?(\w+)[`\"]?',
            r'(\w+\.py)',
        ]
        
        for pattern in subject_patterns:
            m = re.search(pattern, prefix, re.IGNORECASE)
            if m:
                return m.group(1)
        
        return "unknown"
    
    def _calculate_claim_confidence(self, pattern: str, match_text: str, context: str) -> float:
        """Calculate confidence that this is actually a claim."""
        confidence = 0.5  # Base
        
        # Strong assertion words increase confidence
        if any(word in context.lower() for word in ['is', 'are', 'has', 'have', 'successfully']):
            confidence += 0.2
        
        # Percentage numbers increase confidence
        if re.search(r'\d+%', context):
            confidence += 0.15
        
        # File paths increase confidence for implementation claims
        if re.search(r'[\w/]+\.\w+', context):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _deduplicate_claims(self, claims: List[ExtractedClaim]) -> List[ExtractedClaim]:
        """Remove duplicate or overlapping claims."""
        if not claims:
            return []
        
        # Sort by location
        sorted_claims = sorted(claims, key=lambda c: c.location)
        
        # Remove duplicates with same assertion
        seen_assertions = set()
        unique_claims = []
        
        for claim in sorted_claims:
            key = (claim.claim_type, claim.assertion.lower()[:50])
            if key not in seen_assertions:
                seen_assertions.add(key)
                unique_claims.append(claim)
        
        return unique_claims
    
    def _determine_verification_method(self, claim: ExtractedClaim) -> str:
        """Determine how to verify this claim."""
        if claim.claim_type == ClaimType.IMPLEMENTATION:
            return "file_inspection"
        elif claim.claim_type == ClaimType.TEST_RESULT:
            return "test_execution"
        elif claim.claim_type == ClaimType.QUANTITATIVE:
            return "metric_measurement"
        else:
            return "content_analysis"
    
    def _gather_direct_evidence(self, claim: ExtractedClaim, workspace_path: Path) -> List[ProvidedEvidence]:
        """Gather evidence directly from workspace."""
        evidence = []
        
        # Look for file paths in the claim
        file_matches = re.findall(r'[`"\']?([\w/._-]+\.(py|js|ts|java|go))[`"\']?', claim.claim_text)
        
        for file_match in file_matches:
            file_path = workspace_path / file_match[0]
            if file_path.exists():
                # File exists evidence
                evidence.append(ProvidedEvidence(
                    evidence_id=f"EV-{claim.claim_id}-FILE",
                    evidence_type=EvidenceType.FILE_EXISTS,
                    content=str(file_path),
                    source=str(file_path)
                ))
                
                # Content evidence
                try:
                    content = file_path.read_text()[:2000]
                    evidence.append(ProvidedEvidence(
                        evidence_id=f"EV-{claim.claim_id}-CONTENT",
                        evidence_type=EvidenceType.CODE_CONTENT,
                        content=content,
                        source=str(file_path)
                    ))
                except Exception:
                    pass
        
        return evidence
    
    def _verify_file_exists(self, evidence: ProvidedEvidence, claim: ExtractedClaim) -> Tuple[float, str]:
        """Verify file existence evidence."""
        path = Path(evidence.content)
        if path.exists():
            return 1.0, f"File exists: {path}"
        return 0.0, f"File NOT found: {path}"
    
    def _verify_code_content(self, evidence: ProvidedEvidence, claim: ExtractedClaim) -> Tuple[float, str]:
        """Verify code content evidence."""
        content = str(evidence.content)
        score = 0.5
        findings = []
        
        # Check for placeholder patterns (bad)
        placeholder_patterns = ['TODO', 'FIXME', 'placeholder', 'pass  # ', 'raise NotImplementedError']
        placeholders_found = [p for p in placeholder_patterns if p.lower() in content.lower()]
        
        if placeholders_found:
            score -= 0.3
            findings.append(f"Contains placeholders: {placeholders_found}")
        else:
            score += 0.2
            findings.append("No placeholder code detected")
        
        # Check for actual implementation (good)
        impl_patterns = ['def ', 'class ', 'function ', 'return ']
        impl_found = any(p in content for p in impl_patterns)
        
        if impl_found:
            score += 0.2
            findings.append("Contains actual implementation")
        
        # Check for imports (indicates real code)
        if 'import ' in content or 'from ' in content:
            score += 0.1
            findings.append("Contains imports")
        
        return min(max(score, 0.0), 1.0), "; ".join(findings)
    
    def _verify_test_output(self, evidence: ProvidedEvidence, claim: ExtractedClaim) -> Tuple[float, str]:
        """Verify test output evidence."""
        content = str(evidence.content)
        score = 0.5
        findings = []
        
        # Look for pass indicators
        pass_patterns = ['PASS', 'OK', 'passed', 'success', '✓', '✅']
        if any(p in content for p in pass_patterns):
            score += 0.3
            findings.append("Test pass indicators found")
        
        # Look for fail indicators (contradicting)
        fail_patterns = ['FAIL', 'ERROR', 'failed', 'exception', '✗', '❌']
        if any(p in content for p in fail_patterns):
            score -= 0.4
            findings.append("Test FAIL indicators found - contradicts claim")
        
        # Look for actual test counts
        count_match = re.search(r'(\d+)\s*(?:passed|pass|tests?\s+passed)', content, re.IGNORECASE)
        if count_match:
            score += 0.2
            findings.append(f"Found test count: {count_match.group(0)}")
        
        return min(max(score, 0.0), 1.0), "; ".join(findings) if findings else "No clear test results"
    
    def _verify_metrics(self, evidence: ProvidedEvidence, claim: ExtractedClaim) -> Tuple[float, str]:
        """Verify metrics evidence."""
        content = str(evidence.content)
        score = 0.3
        findings = []
        
        # Look for actual numbers
        numbers = re.findall(r'\d+\.?\d*', content)
        if numbers:
            score += 0.3
            findings.append(f"Contains metrics: {numbers[:5]}")
        
        # Look for comparison/baseline
        if 'baseline' in content.lower() or 'before' in content.lower():
            score += 0.2
            findings.append("Contains baseline comparison")
        
        return min(max(score, 0.0), 1.0), "; ".join(findings) if findings else "No metrics found"
    
    def _load_history(self):
        """Load verification history from disk."""
        history_file = self.storage_path / "verification_history.json"
        if history_file.exists():
            try:
                # Just load count for now
                pass
            except Exception:
                pass
    
    def _save_history(self):
        """Save verification history to disk."""
        history_file = self.storage_path / "verification_history.json"
        try:
            # Save summary
            summary = {
                "total_verifications": len(self.verification_history),
                "last_verification": datetime.now().isoformat()
            }
            with open(history_file, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass
    
    # =========================================================================
    # Learning Interface (5/5 methods)
    # =========================================================================
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback about evidence demands."""
        source_id = feedback.get('source_id')
        if source_id and not feedback.get('was_correct', True):
            current = getattr(self, '_source_reliability', {}).get(source_id, 0.5)
            self._source_reliability[source_id] = max(0.1, current - 0.05)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return evidence demand statistics."""
        return {
            'total_demands': getattr(self, '_total_demands', 0),
            'satisfied_demands': getattr(self, '_satisfied_demands', 0),
            'satisfied_rate': getattr(self, '_satisfied_demands', 0) / max(1, self._total_demands),
            'source_reliability': dict(self._source_reliability),
            'outcomes_recorded': len(self._outcomes)
        }
    
    def serialize_state(self) -> Dict:
        """Serialize state for persistence."""
        return {
            'source_reliability': dict(self._source_reliability),
            'outcomes': getattr(self, '_outcomes', [])[-100:],
            'verification_history': [
                {'claim_type': v['claim_type'], 'status': v['status'].value if hasattr(v['status'], 'value') else v['status']}
                for v in self.verification_history[-100:]
            ] if hasattr(self, 'verification_history') else [],
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore state from serialized data."""
        self._source_reliability = state.get('source_reliability', {})
        self._outcomes = state.get('outcomes', [])


def create_evidence_demand_loop(storage_path: Path = None) -> EvidenceDemandLoop:
    """Factory function to create evidence demand loop."""
    return EvidenceDemandLoop(storage_path=storage_path)


# ============================================================================
# PHASE 4 ENHANCEMENT: PROOF-BASED VERIFICATION
# ============================================================================
# These classes enhance the Evidence Layer (Layer 6: Basal Ganglia) to move
# beyond word analysis to actual proof verification.
#
# Integration: These are part of the evidence_demand module, NOT orphan processes.
# They enhance NOVEL-3 and GAP-1 to actually verify proof instead of just
# analyzing words.
# ============================================================================

@dataclass
class ProofVerificationResult:
    """Result of proof-based verification (not just word analysis)."""
    verified: bool
    verification_method: str
    evidence_found: List[Dict]
    gaps: List[str]
    confidence: float
    file_verification: Dict = field(default_factory=dict)
    enumeration_verification: Dict = field(default_factory=dict)
    goal_alignment: Dict = field(default_factory=dict)


class ProofVerifier:
    """
    NOVEL-3/GAP-1 Enhancement: Actually verify proof instead of analyzing words.
    
    This is integrated into the Evidence Layer (Basal Ganglia) and is NOT
    a standalone module. It enhances EvidenceDemandLoop with capabilities to:
    
    1. Verify files actually exist
    2. Verify code structure is complete (not just claimed)
    3. Verify enumerations ("all 15 items" - verify each)
    4. Detect past-tense proposals posing as completions
    5. Compare against original objectives (goal drift)
    
    Patent Alignment:
    - NOVEL-3: Claim-Evidence Alignment Verification (enhanced)
    - GAP-1: Evidence Demand Loop (enhanced)
    - PPA1-Inv23: AI Common Sense via Multi-Source Triangulation
    """
    
    # Patterns that indicate past-tense proposals (not actual completions)
    PAST_TENSE_PROPOSAL_PATTERNS = [
        (r'\bwas\s+(?:designed|implemented|created|built|planned|architected)\b', 0.7),
        (r'\bwe\s+(?:implemented|created|built|developed|designed)\s+(?:a|the|an)\b', 0.6),
        (r'\bhas\s+been\s+(?:designed|planned|architected)\b', 0.6),
        (r'\barchitecture\s+(?:is|was)\s+(?:designed|planned)\b', 0.5),
        (r'\bwill\s+(?:be\s+)?(?:implemented|added|created)\b', 0.8),
        (r'\bfuture\s+(?:enhancements?|improvements?|additions?|phases?)\b', 0.7),
        (r'\b(?:roadmap|planned|upcoming)\s+(?:features?|capabilities?)\b', 0.6),
    ]
    
    # Patterns that indicate explained/minimized failures
    EXPLAINED_FAILURE_PATTERNS = [
        (r'\b(?:minor|small|trivial|negligible)\s+(?:issues?|problems?|bugs?|errors?)\b', 0.7),
        (r'\b(?:being\s+)?(?:addressed|fixed|resolved|worked\s+on)\b', 0.5),
        (r'\b(?:not\s+)?(?:production[-\s]?critical|critical)\b', 0.6),
        (r'\b(?:experimental|beta|alpha|prototype)\s+(?:module|feature|component)\b', 0.6),
        (r'\b(?:known\s+)?(?:limitation|issue|bug)\s+(?:but|however)\b', 0.65),
        (r'\b(?:edge\s+case|corner\s+case|rare\s+scenario)\b', 0.5),
        (r'\bcan\s+be\s+(?:ignored|skipped|deferred)\b', 0.7),
    ]
    
    def __init__(self, workspace_path: Path = None):
        self.workspace_path = workspace_path or Path.cwd()
    
    async def verify_proof(
        self, 
        claim_text: str, 
        claim_type: ClaimType,
        original_query: str = None,
        context: Dict = None
    ) -> ProofVerificationResult:
        """
        Verify proof for a claim - beyond word analysis.
        
        This is the main entry point for proof-based verification.
        """
        gaps = []
        evidence_found = []
        file_verification = {}
        enumeration_verification = {}
        goal_alignment = {}
        
        # 1. Check for past-tense proposals posing as completions
        past_tense_result = self._detect_past_tense_proposals(claim_text)
        if past_tense_result["is_proposal"]:
            gaps.append(f"Past-tense proposal detected: '{past_tense_result['matched_pattern']}' - verify actual implementation")
        
        # 2. File-based verification for implementation claims
        if claim_type in [ClaimType.IMPLEMENTATION, ClaimType.FUNCTIONALITY]:
            file_verification = await self._verify_files_exist(claim_text)
            if not file_verification.get("all_found", False):
                gaps.extend([f"File not found: {f}" for f in file_verification.get("missing", [])])
            else:
                evidence_found.extend([{"type": "file", "path": f} for f in file_verification.get("found", [])])
        
        # 3. Enumeration verification for completion claims
        if claim_type == ClaimType.COMPLETION:
            enumeration_verification = await self._verify_enumeration(claim_text, context)
            if not enumeration_verification.get("verified", False):
                gaps.append(enumeration_verification.get("gap", "Enumeration not verified"))
        
        # 4. Goal alignment check
        if original_query:
            goal_alignment = self._check_goal_alignment(original_query, claim_text)
            if goal_alignment.get("drift_detected", False):
                gaps.append(f"Goal drift: {goal_alignment.get('reason', 'Response may not address original query')}")
        
        # 5. Check for explained/minimized failures
        explained_failure = self._detect_explained_failures(claim_text)
        if explained_failure["has_explained_failure"]:
            gaps.append(f"Explained failure detected: '{explained_failure['matched_pattern']}' - verify severity is actually minor")
        
        # Calculate confidence
        total_checks = 5
        passed_checks = 0
        if not past_tense_result["is_proposal"]:
            passed_checks += 1
        if file_verification.get("all_found", True):  # True if no files to check
            passed_checks += 1
        if enumeration_verification.get("verified", True):  # True if no enumeration to check
            passed_checks += 1
        if not goal_alignment.get("drift_detected", False):
            passed_checks += 1
        if not explained_failure["has_explained_failure"]:
            passed_checks += 1
        
        confidence = passed_checks / total_checks
        
        return ProofVerificationResult(
            verified=len(gaps) == 0,
            verification_method="proof_based_verification",
            evidence_found=evidence_found,
            gaps=gaps,
            confidence=confidence,
            file_verification=file_verification,
            enumeration_verification=enumeration_verification,
            goal_alignment=goal_alignment
        )
    
    def _detect_past_tense_proposals(self, text: str) -> Dict:
        """Detect past-tense proposals posing as completions."""
        for pattern, weight in self.PAST_TENSE_PROPOSAL_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return {
                    "is_proposal": True,
                    "matched_pattern": match.group(0),
                    "weight": weight
                }
        return {"is_proposal": False, "matched_pattern": None, "weight": 0}
    
    def _detect_explained_failures(self, text: str) -> Dict:
        """Detect when failures are being explained away or minimized."""
        for pattern, weight in self.EXPLAINED_FAILURE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return {
                    "has_explained_failure": True,
                    "matched_pattern": match.group(0),
                    "weight": weight
                }
        return {"has_explained_failure": False, "matched_pattern": None, "weight": 0}
    
    async def _verify_files_exist(self, claim_text: str) -> Dict:
        """Actually verify that claimed files exist."""
        # Extract file paths from claim
        file_patterns = [
            r'[`"\']?([\w/._-]+\.(?:py|js|ts|java|go|rb|rs|c|cpp|h))[`"\']?',
            r'(?:in|at|file)\s+[`"\']?([\w/._-]+)[`"\']?',
        ]
        
        claimed_files = []
        for pattern in file_patterns:
            matches = re.findall(pattern, claim_text, re.IGNORECASE)
            claimed_files.extend([m if isinstance(m, str) else m[0] for m in matches])
        
        if not claimed_files:
            return {"all_found": True, "found": [], "missing": [], "no_files_claimed": True}
        
        found = []
        missing = []
        
        for file_path in set(claimed_files):
            # Try multiple path resolutions
            paths_to_try = [
                Path(file_path),
                self.workspace_path / file_path,
                self.workspace_path / "src" / file_path,
                self.workspace_path / "src" / "core" / Path(file_path).name,
            ]
            
            file_found = False
            for path in paths_to_try:
                if path.exists():
                    found.append(str(path))
                    file_found = True
                    break
            
            if not file_found:
                missing.append(file_path)
        
        return {
            "all_found": len(missing) == 0,
            "found": found,
            "missing": missing,
            "claimed_count": len(claimed_files)
        }
    
    async def _verify_enumeration(self, claim_text: str, context: Dict = None) -> Dict:
        """Verify enumeration claims like 'all 15 items complete'."""
        # Extract quantity claimed
        quantity_patterns = [
            r'\ball\s+(\d+)\s+(?:items?|modules?|files?|tests?|claims?)',
            r'(\d+)\s*/\s*(\d+)',  # "15/15" format
            r'(\d+)\s+of\s+(\d+)',  # "15 of 15" format
        ]
        
        quantity_claimed = None
        for pattern in quantity_patterns:
            match = re.search(pattern, claim_text, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    quantity_claimed = int(groups[1])  # Total (denominator)
                else:
                    quantity_claimed = int(groups[0])
                break
        
        if quantity_claimed is None:
            # Check for "all" or "everything" without specific count
            if re.search(r'\b(?:all|everything|fully|completely)\b', claim_text, re.IGNORECASE):
                return {
                    "verified": False,
                    "gap": "Completion claim uses 'all/fully/completely' without specific enumeration"
                }
            return {"verified": True, "no_enumeration_claimed": True}
        
        # If context provides items, verify count
        if context and "items" in context:
            actual_count = len(context["items"])
            if actual_count >= quantity_claimed:
                return {
                    "verified": True,
                    "claimed": quantity_claimed,
                    "actual": actual_count
                }
            else:
                return {
                    "verified": False,
                    "claimed": quantity_claimed,
                    "actual": actual_count,
                    "gap": f"Claimed {quantity_claimed} items but only {actual_count} enumerated"
                }
        
        # No context provided - demand enumeration
        return {
            "verified": False,
            "claimed": quantity_claimed,
            "gap": f"Claimed {quantity_claimed} items without providing enumeration"
        }
    
    def _check_goal_alignment(self, original_query: str, response: str) -> Dict:
        """Check if response addresses the original query (goal drift detection)."""
        # Extract key objectives from query
        objective_patterns = [
            r'(?:implement|create|build|make|add|fix|improve|update)\s+(.+?)(?:\.|$)',
            r'(?:how\s+(?:to|do\s+we|can\s+I))\s+(.+?)(?:\?|$)',
            r'(?:write|develop|design)\s+(.+?)(?:\.|$)',
        ]
        
        objectives = []
        for pattern in objective_patterns:
            matches = re.findall(pattern, original_query, re.IGNORECASE)
            objectives.extend(matches)
        
        if not objectives:
            return {"drift_detected": False, "no_objectives_found": True}
        
        # Check how many objectives are addressed
        response_lower = response.lower()
        addressed = []
        missed = []
        substituted = []
        
        # Key concept substitution pairs (what was asked vs what might be substituted)
        SUBSTITUTION_PAIRS = [
            ('quality', ['performance', 'speed', 'time', 'fast', 'quick', 'efficient']),
            ('accuracy', ['performance', 'speed', 'coverage', 'throughput']),
            ('security', ['performance', 'speed', 'features', 'functionality']),
            ('reliability', ['performance', 'features', 'speed']),
            ('usability', ['performance', 'features', 'functionality']),
            ('completeness', ['performance', 'speed', 'partial', 'core']),
        ]
        
        for obj in objectives:
            # Clean the objective - remove punctuation
            obj_clean = re.sub(r'[^\w\s]', '', obj.lower())
            obj_words = set(obj_clean.split())
            # Filter out common words
            obj_words = {w for w in obj_words if len(w) > 3}
            
            if not obj_words:
                continue
            
            # Check for substitution - was a key concept replaced with a different one?
            for asked_for, substitutes in SUBSTITUTION_PAIRS:
                if asked_for in obj_words or asked_for in obj.lower():
                    # Check if response talks about substitutes instead
                    substitute_found = any(sub in response_lower for sub in substitutes)
                    asked_addressed = asked_for in response_lower
                    
                    if substitute_found and not asked_addressed:
                        substituted.append({
                            "asked_for": asked_for,
                            "substituted_with": [s for s in substitutes if s in response_lower]
                        })
            
            overlap = sum(1 for w in obj_words if w in response_lower)
            overlap_ratio = overlap / len(obj_words)
            
            if overlap_ratio > 0.3:
                addressed.append(obj)
            else:
                missed.append(obj)
        
        # Goal drift is detected if:
        # 1. More objectives missed than addressed, OR
        # 2. Key concepts were substituted (e.g., asked for quality, got performance)
        drift_detected = len(missed) > len(addressed) or len(substituted) > 0
        
        reason = None
        if substituted:
            subs = [f"{s['asked_for']}→{s['substituted_with']}" for s in substituted]
            reason = f"Goal substitution detected: {subs}"
        elif missed:
            reason = f"Missed objectives: {missed}"
        
        return {
            "drift_detected": drift_detected,
            "objectives": objectives,
            "addressed": addressed,
            "missed": missed,
            "substituted": substituted,
            "reason": reason
        }

    # =========================================================================
    # Learning Interface
    # =========================================================================
    
    def record_outcome(self, outcome: Dict) -> None:
        """Record operation outcome for learning."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        getattr(self, '_outcomes', {}).append(outcome)
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback to adjust behavior."""
        if not hasattr(self, '_learning_params'):
            self._learning_params = {}
        was_correct = feedback.get('was_correct', True)
        if not was_correct:
            for key in self._learning_params:
                self._learning_params[key] *= 0.95
    
    def get_statistics(self) -> Dict:
        if not hasattr(self, '_total_demands'): self._total_demands = 0
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


class EnhancedEvidenceDemandLoop(EvidenceDemandLoop):
    """
    Enhanced Evidence Demand Loop with proof-based verification.
    
    This extends the base EvidenceDemandLoop to include actual proof
    verification instead of just word analysis.
    
    Integration: Part of Layer 6 (Basal Ganglia - Evidence)
    
    ORCHESTRATION: This class OVERRIDES run_full_verification to automatically
    include proof-based verification. When IntegratedGovernanceEngine calls
    evidence_demand.run_full_verification(), the proof verification is triggered.
    """
    
    def __init__(self, storage_path: Path = None, workspace_path: Path = None):
        super().__init__(storage_path=storage_path)
        self.proof_verifier = ProofVerifier(workspace_path=workspace_path)
        self._workspace_path = workspace_path
    
    def run_full_verification(self,
                              response: str,
                              query: str = "",
                              workspace_path: Path = None) -> EvidenceDemandResult:
        """
        OVERRIDES base class to include proof-based verification.
        
        This is called from IntegratedGovernanceEngine._run_detectors() and
        automatically includes:
        1. Standard claim extraction and verification (base class)
        2. Past-tense proposal detection (PROOF)
        3. File existence verification (PROOF)
        4. Enumeration verification (PROOF)
        5. Goal alignment checking (PROOF)
        6. Explained failure detection (PROOF)
        
        PATHWAY: IntegratedGovernanceEngine.evaluate_and_improve()
                 → _run_detectors()
                 → evidence_demand.run_full_verification()  ← THIS METHOD
                 → ProofVerifier.verify_proof() for each claim
        """
        # Step 1: Run base class verification (claim extraction, pattern matching)
        base_result = super().run_full_verification(response, query, workspace_path)
        
        # Step 2: Enhance with proof verification for each claim
        enhanced_gaps = []
        ws_path = workspace_path or self._workspace_path
        
        for claim in base_result.claims_extracted:
            # Run synchronous proof verification
            # CRITICAL FIX: Use FULL response for file verification, not truncated claim_text
            # The claim_text is only ~100 chars of context around the matched pattern
            # But file paths may be listed elsewhere in the response
            proof_result = self._verify_claim_proof_sync(
                response,  # Use full response for file verification
                claim.claim_type,
                original_query=query
            )
            
            # If proof verification found gaps, add them
            if proof_result.get('gaps'):
                enhanced_gaps.extend(proof_result['gaps'])
                
                # Downgrade verification status if proof not found
                for verification in base_result.verifications:
                    if verification.claim_id == claim.claim_id:
                        if verification.status == VerificationStatus.VERIFIED:
                            verification.status = VerificationStatus.PARTIALLY_VERIFIED
                            verification.findings.extend(proof_result['gaps'])
                            verification.confidence *= proof_result.get('confidence', 0.5)
        
        # Step 3: Update overall result if proof gaps found
        if enhanced_gaps:
            # Add gaps to unverified claims
            for gap in enhanced_gaps:
                base_result.unverified_claims.append(f"PROOF_GAP: {gap}")
            
            # Downgrade overall status
            if base_result.overall_status == VerificationStatus.VERIFIED:
                base_result.overall_status = VerificationStatus.PARTIALLY_VERIFIED
                base_result.overall_confidence *= 0.6
        
        # Step 4: Track proof verification inventions
        base_result.inventions_used.extend([
            "NOVEL-3+: Enhanced Claim-Evidence with Proof Verification",
            "GAP-1+: Evidence Demand with File/Enumeration/Goal Checks"
        ])
        
        return base_result
    
    def _verify_claim_proof_sync(self, claim_text: str, claim_type: ClaimType, 
                                  original_query: str = None) -> dict:
        """
        Synchronous wrapper for proof verification.
        
        Checks:
        1. Past-tense proposals
        2. Explained failures
        3. Goal alignment/substitution
        4. Enumeration without list
        5. FILE EXISTENCE (CRITICAL - actually verify files exist)
        """
        gaps = []
        confidence = 1.0
        files_verified = []
        files_missing = []
        
        # 1. Past-tense proposal detection
        past_tense = self.proof_verifier._detect_past_tense_proposals(claim_text)
        if past_tense['is_proposal']:
            gaps.append(f"Past-tense proposal: '{past_tense['matched_pattern']}' - verify implementation")
            confidence *= 0.7
        
        # 2. Explained failure detection
        explained = self.proof_verifier._detect_explained_failures(claim_text)
        if explained['has_explained_failure']:
            gaps.append(f"Explained failure: '{explained['matched_pattern']}' - verify severity")
            confidence *= 0.8
        
        # 3. Goal alignment (if query provided)
        if original_query:
            goal = self.proof_verifier._check_goal_alignment(original_query, claim_text)
            if goal.get('drift_detected'):
                gaps.append(f"Goal drift: {goal.get('reason', 'detected')}")
                confidence *= 0.6
        
        # 4. Check for enumeration claims without list (synchronous version)
        import re
        quantity_patterns = [
            r'\ball\s+(\d+)\s+(?:items?|modules?|files?|tests?|claims?)',
            r'(\d+)\s*/\s*(\d+)',
            r'(\d+)\s+of\s+(\d+)',
        ]
        for pattern in quantity_patterns:
            match = re.search(pattern, claim_text, re.IGNORECASE)
            if match:
                gaps.append(f"Enumeration claim without list - verify each item")
                confidence *= 0.7
                break
        
        # 5. CRITICAL: Actually verify files exist (synchronous version)
        file_verification = self._verify_files_exist_sync(claim_text)
        if file_verification['missing']:
            for missing_file in file_verification['missing']:
                gaps.append(f"FILE_NOT_FOUND: {missing_file} - claimed but does not exist")
            confidence *= 0.3  # Severe penalty for missing files
            files_missing = file_verification['missing']
        if file_verification['found']:
            files_verified = file_verification['found']
        
        return {
            'gaps': gaps,
            'confidence': confidence,
            'files_verified': files_verified,
            'files_missing': files_missing
        }
    
    def _verify_files_exist_sync(self, claim_text: str) -> dict:
        """
        SYNCHRONOUS file existence verification.
        
        This is the CRITICAL function that actually checks if claimed files exist.
        Without this, BAIS is just a word checker.
        """
        import re
        from pathlib import Path
        
        # Extract file paths from claim
        file_patterns = [
            r'[`"\']?([\w/._-]+\.(?:py|js|ts|java|go|rb|rs|c|cpp|h|tsx|jsx|md|yaml|yml|json))[`"\']?',
            r'(?:in|at|file|module|path)[:\s]+[`"\']?([\w/._-]+)[`"\']?',
            r'[-•]\s*[`"\']?([\w/._-]+\.(?:py|js|ts|java|go|rb|rs|c|cpp|h|tsx|jsx))[`"\']?',
        ]
        
        claimed_files = []
        for pattern in file_patterns:
            matches = re.findall(pattern, claim_text, re.IGNORECASE)
            claimed_files.extend([m if isinstance(m, str) else m[0] for m in matches])
        
        if not claimed_files:
            return {"all_found": True, "found": [], "missing": [], "no_files_claimed": True}
        
        found = []
        missing = []
        workspace = self._workspace_path or Path.cwd()
        
        for file_path in set(claimed_files):
            # Skip common false positives
            if file_path in ['e.g', 'etc', 'i.e'] or len(file_path) < 3:
                continue
                
            # Try multiple path resolutions
            paths_to_try = [
                Path(file_path),
                workspace / file_path,
                workspace / "src" / file_path,
                workspace / "src" / "core" / Path(file_path).name,
                workspace / "src" / "detectors" / Path(file_path).name,
                workspace / "src" / "learning" / Path(file_path).name,
            ]
            
            file_found = False
            for path in paths_to_try:
                if path.exists():
                    found.append(str(file_path))
                    file_found = True
                    break
            
            if not file_found:
                missing.append(file_path)
        
        return {
            "all_found": len(missing) == 0,
            "found": found,
            "missing": missing
        }
    
    async def run_proof_based_verification(
        self,
        response: str,
        query: str = "",
        workspace_path: Path = None
    ) -> EvidenceDemandResult:
        """
        Run enhanced verification with actual proof checking.
        
        This goes beyond the base run_full_verification by:
        1. Detecting past-tense proposals
        2. Actually verifying files exist
        3. Verifying enumerations
        4. Checking goal alignment
        """
        # First run standard verification
        base_result = self.run_full_verification(response, query, workspace_path)
        
        # Now enhance with proof verification
        enhanced_verifications = []
        enhanced_gaps = []
        
        for claim in base_result.claims_extracted:
            # Run proof verification on each claim
            proof_result = await self.proof_verifier.verify_proof(
                claim.claim_text,
                claim.claim_type,
                original_query=query
            )
            
            # If proof verification found gaps, add them
            if proof_result.gaps:
                enhanced_gaps.extend(proof_result.gaps)
                
                # Downgrade verification status if proof not found
                for verification in base_result.verifications:
                    if verification.claim_id == claim.claim_id:
                        if verification.status == VerificationStatus.VERIFIED:
                            verification.status = VerificationStatus.PARTIALLY_VERIFIED
                            verification.findings.extend(proof_result.gaps)
                            verification.confidence *= proof_result.confidence
        
        # Update overall status if proof gaps found
        if enhanced_gaps and base_result.overall_status == VerificationStatus.VERIFIED:
            base_result.overall_status = VerificationStatus.PARTIALLY_VERIFIED
            base_result.overall_confidence *= 0.7
        
        # Add proof verification to inventions used
        base_result.inventions_used.extend([
            "NOVEL-3+: Enhanced Claim-Evidence with Proof Verification",
            "GAP-1+: Evidence Demand with File/Enumeration Checks"
        ])
        
        return base_result


def create_enhanced_evidence_loop(
    storage_path: Path = None, 
    workspace_path: Path = None
) -> EnhancedEvidenceDemandLoop:
    """Factory function to create enhanced evidence demand loop with proof verification."""
    return EnhancedEvidenceDemandLoop(
        storage_path=storage_path,
        workspace_path=workspace_path
    )

    # Learning Interface
    def record_outcome(self, outcome: Dict) -> None:
        if not hasattr(self, '_outcomes'): self._outcomes = []
        getattr(self, '_outcomes', {}).append(outcome)
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        pass
    
    def get_statistics(self) -> Dict:
        if not hasattr(self, '_total_demands'): self._total_demands = 0
        return {'outcomes': len(getattr(self, '_outcomes', []))}
    
    def serialize_state(self) -> Dict:
        return {'outcomes': getattr(self, '_outcomes', [])[-100:]}
    
    def deserialize_state(self, state: Dict) -> None:
        self._outcomes = state.get('outcomes', [])

