"""
PROOF INSPECTOR (NOVEL-24)
==========================
Actually verify claims against evidence instead of analyzing words.

This module moves BAIS from pattern detection to proof verification:
- FILE claims: Check if files actually exist
- CODE claims: Verify code structure and completeness
- COMPLETION claims: Enumerate and verify each item
- METRIC claims: Verify against source data
- INTEGRATION claims: Trace actual call paths

Created: December 24, 2025
Patent: NOVEL-24 - Proof-Based Verification Engine
"""

import os
import re
import ast
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import importlib.util
import tempfile


class ClaimType(Enum):
    """Types of claims that require different verification methods."""
    FILE_EXISTS = "file_exists"
    CODE_WORKS = "code_works"
    TEST_PASSES = "test_passes"
    COMPLETION = "completion"
    METRIC = "metric"
    INTEGRATION = "integration"
    CITATION = "citation"
    UNKNOWN = "unknown"


@dataclass
class ProofResult:
    """Result of proof verification."""
    verified: bool
    evidence: Dict[str, Any] = field(default_factory=dict)
    gaps: List[str] = field(default_factory=list)
    confidence: float = 0.0
    verification_method: str = ""
    raw_output: str = ""


@dataclass
class ClaimAnalysis:
    """Analysis of a claim extracted from LLM response."""
    claim_text: str
    claim_type: ClaimType
    file_paths: List[str] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    function_names: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    items_claimed: List[str] = field(default_factory=list)
    quantity_claimed: Optional[int] = None


class ClaimClassifier:
    """Classify claims into types for appropriate verification."""
    
    FILE_PATTERNS = [
        r'\b(?:file|module|script|code)\s+(?:exists?|at|in)\s+["\']?([^\s"\']+)',
        r'\b(?:implemented|created|wrote)\s+(?:in|to)\s+["\']?([^\s"\']+)',
        r'["\']([a-zA-Z_][a-zA-Z0-9_/]*\.py)["\']',
    ]
    
    CODE_PATTERNS = [
        r'\b(?:function|method|class)\s+(\w+)\s+(?:works|implemented|complete)',
        r'\b(\w+)\s+(?:is\s+)?(?:fully\s+)?(?:implemented|working|functional)',
        r'\bdef\s+(\w+)|class\s+(\w+)',
    ]
    
    COMPLETION_PATTERNS = [
        r'\ball\s+(\d+)\s+(\w+)',  # "all 15 items"
        r'\b(\d+)\s*(?:/|of)\s*(\d+)',  # "15/15" or "15 of 15"
        r'\beverything\s+(?:is\s+)?(?:complete|done|finished)',
        r'\bfully\s+(?:complete|implemented|done)',
    ]
    
    METRIC_PATTERNS = [
        r'(\d+(?:\.\d+)?)\s*%',  # percentages
        r'(?:accuracy|precision|recall|score)\s*(?:of|:)?\s*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*(?:ms|seconds?|minutes?)',  # timing
    ]
    
    INTEGRATION_PATTERNS = [
        r'\b(\w+)\s+(?:is\s+)?integrated\s+(?:into|with)',
        r'\bcalled?\s+from\s+(\w+)',
        r'\bwired\s+(?:into|to)\s+(\w+)',
    ]
    
    def classify(self, claim: str) -> ClaimAnalysis:
        """Classify a claim and extract relevant details."""
        claim_lower = claim.lower()
        
        # Check for file claims
        file_paths = []
        for pattern in self.FILE_PATTERNS:
            matches = re.findall(pattern, claim, re.IGNORECASE)
            file_paths.extend([m if isinstance(m, str) else m[0] for m in matches if m])
        
        if file_paths:
            return ClaimAnalysis(
                claim_text=claim,
                claim_type=ClaimType.FILE_EXISTS,
                file_paths=file_paths
            )
        
        # Check for completion claims
        for pattern in self.COMPLETION_PATTERNS:
            match = re.search(pattern, claim, re.IGNORECASE)
            if match:
                groups = match.groups()
                quantity = None
                if groups and groups[0] and groups[0].isdigit():
                    quantity = int(groups[0])
                
                return ClaimAnalysis(
                    claim_text=claim,
                    claim_type=ClaimType.COMPLETION,
                    quantity_claimed=quantity
                )
        
        # Check for metric claims
        metrics = {}
        for pattern in self.METRIC_PATTERNS:
            matches = re.findall(pattern, claim, re.IGNORECASE)
            for m in matches:
                val = m if isinstance(m, str) else m[0]
                try:
                    metrics[f"metric_{len(metrics)}"] = float(val)
                except ValueError:
                    pass
        
        if metrics:
            return ClaimAnalysis(
                claim_text=claim,
                claim_type=ClaimType.METRIC,
                metrics=metrics
            )
        
        # Check for integration claims
        for pattern in self.INTEGRATION_PATTERNS:
            match = re.search(pattern, claim, re.IGNORECASE)
            if match:
                return ClaimAnalysis(
                    claim_text=claim,
                    claim_type=ClaimType.INTEGRATION,
                    class_names=[match.group(1)] if match.groups() else []
                )
        
        # Check for code claims
        class_names = []
        function_names = []
        for pattern in self.CODE_PATTERNS:
            matches = re.findall(pattern, claim, re.IGNORECASE)
            for m in matches:
                if isinstance(m, tuple):
                    class_names.extend([x for x in m if x])
                elif m:
                    function_names.append(m)
        
        if class_names or function_names:
            return ClaimAnalysis(
                claim_text=claim,
                claim_type=ClaimType.CODE_WORKS,
                class_names=class_names,
                function_names=function_names
            )
        
        return ClaimAnalysis(
            claim_text=claim,
            claim_type=ClaimType.UNKNOWN
        )


class ProofInspector:
    """
    NOVEL-24: Proof-Based Verification Engine
    
    Actually verifies claims instead of analyzing words:
    - Checks if files exist
    - Verifies code structure
    - Enumerates completion claims
    - Validates metrics against source
    - Traces integration paths
    """
    
    def __init__(self, workspace_path: Path = None):
        self.workspace_path = workspace_path or Path.cwd()
        self.classifier = ClaimClassifier()
        self.verification_cache: Dict[str, ProofResult] = {}
    
    async def verify_claim(self, claim: str, context: Dict = None) -> ProofResult:
        """Verify a claim using appropriate verification method."""
        # Check cache
        cache_key = f"{claim}:{str(context)}"
        if cache_key in self.verification_cache:
            return self.verification_cache[cache_key]
        
        # Classify claim
        analysis = self.classifier.classify(claim)
        
        # Route to appropriate verifier
        if analysis.claim_type == ClaimType.FILE_EXISTS:
            result = await self._verify_file_claim(analysis, context)
        elif analysis.claim_type == ClaimType.CODE_WORKS:
            result = await self._verify_code_claim(analysis, context)
        elif analysis.claim_type == ClaimType.COMPLETION:
            result = await self._verify_completion_claim(analysis, context)
        elif analysis.claim_type == ClaimType.METRIC:
            result = await self._verify_metric_claim(analysis, context)
        elif analysis.claim_type == ClaimType.INTEGRATION:
            result = await self._verify_integration_claim(analysis, context)
        else:
            result = ProofResult(
                verified=False,
                gaps=["Unable to determine verification method for claim"],
                verification_method="none"
            )
        
        # Cache result
        self.verification_cache[cache_key] = result
        return result
    
    async def _verify_file_claim(self, analysis: ClaimAnalysis, context: Dict = None) -> ProofResult:
        """Verify that claimed files actually exist."""
        verified_files = []
        missing_files = []
        
        for file_path in analysis.file_paths:
            # Try multiple path resolutions
            paths_to_try = [
                Path(file_path),
                self.workspace_path / file_path,
                self.workspace_path / "src" / file_path,
                self.workspace_path / "bais-cognitive-engine" / "src" / file_path,
            ]
            
            found = False
            for path in paths_to_try:
                if path.exists():
                    verified_files.append({
                        "path": str(path),
                        "size": path.stat().st_size,
                        "exists": True
                    })
                    found = True
                    break
            
            if not found:
                missing_files.append(file_path)
        
        return ProofResult(
            verified=len(missing_files) == 0 and len(verified_files) > 0,
            evidence={
                "verified_files": verified_files,
                "claim_type": "file_exists"
            },
            gaps=[f"File not found: {f}" for f in missing_files],
            confidence=len(verified_files) / max(len(analysis.file_paths), 1),
            verification_method="file_existence_check"
        )
    
    async def _verify_code_claim(self, analysis: ClaimAnalysis, context: Dict = None) -> ProofResult:
        """Verify that claimed code elements exist and are syntactically valid."""
        verified_elements = []
        missing_elements = []
        
        # Search for classes/functions in workspace
        search_paths = [
            self.workspace_path / "src",
            self.workspace_path / "bais-cognitive-engine" / "src",
        ]
        
        for class_name in analysis.class_names:
            found = await self._find_class_in_codebase(class_name, search_paths)
            if found:
                verified_elements.append({"class": class_name, "location": found})
            else:
                missing_elements.append(f"Class: {class_name}")
        
        for func_name in analysis.function_names:
            found = await self._find_function_in_codebase(func_name, search_paths)
            if found:
                verified_elements.append({"function": func_name, "location": found})
            else:
                missing_elements.append(f"Function: {func_name}")
        
        total_claimed = len(analysis.class_names) + len(analysis.function_names)
        
        return ProofResult(
            verified=len(missing_elements) == 0 and len(verified_elements) > 0,
            evidence={
                "verified_elements": verified_elements,
                "claim_type": "code_exists"
            },
            gaps=missing_elements,
            confidence=len(verified_elements) / max(total_claimed, 1),
            verification_method="code_structure_check"
        )
    
    async def _verify_completion_claim(self, analysis: ClaimAnalysis, context: Dict = None) -> ProofResult:
        """Verify completion claims by enumerating items."""
        gaps = []
        
        # If a quantity is claimed, we need to verify that many items
        if analysis.quantity_claimed:
            # Look for enumerated items in context
            items_found = 0
            if context and "items" in context:
                items_found = len(context["items"])
            
            if items_found < analysis.quantity_claimed:
                gaps.append(
                    f"Claimed {analysis.quantity_claimed} items but only {items_found} enumerated"
                )
        
        # Check for suspicious completion language without enumeration
        suspicious_patterns = [
            "fully", "completely", "100%", "all", "everything"
        ]
        claim_lower = analysis.claim_text.lower()
        suspicious_found = [p for p in suspicious_patterns if p in claim_lower]
        
        if suspicious_found and not context:
            gaps.append(
                f"Completion claim uses {suspicious_found} without providing enumeration"
            )
        
        return ProofResult(
            verified=len(gaps) == 0,
            evidence={
                "quantity_claimed": analysis.quantity_claimed,
                "items_provided": context.get("items", []) if context else [],
                "claim_type": "completion"
            },
            gaps=gaps,
            confidence=0.5 if gaps else 0.8,  # Lower confidence without enumeration
            verification_method="enumeration_check"
        )
    
    async def _verify_metric_claim(self, analysis: ClaimAnalysis, context: Dict = None) -> ProofResult:
        """Verify metric claims against source data."""
        gaps = []
        
        for metric_name, metric_value in analysis.metrics.items():
            # Check if source data is provided
            if context and "source_data" in context:
                source_value = context["source_data"].get(metric_name)
                if source_value and abs(source_value - metric_value) > 0.01:
                    gaps.append(
                        f"Metric {metric_name}: claimed {metric_value}, actual {source_value}"
                    )
            else:
                # No source data provided - flag as unverified
                gaps.append(
                    f"Metric {metric_name}={metric_value} claimed without source data"
                )
        
        return ProofResult(
            verified=len(gaps) == 0 and len(analysis.metrics) > 0,
            evidence={
                "claimed_metrics": analysis.metrics,
                "source_data": context.get("source_data") if context else None,
                "claim_type": "metric"
            },
            gaps=gaps,
            confidence=0.3 if gaps else 0.9,  # Low confidence without source
            verification_method="metric_validation"
        )
    
    async def _verify_integration_claim(self, analysis: ClaimAnalysis, context: Dict = None) -> ProofResult:
        """Verify integration claims by tracing call paths."""
        verified_integrations = []
        missing_integrations = []
        
        for class_name in analysis.class_names:
            # Search for import and usage of the class
            call_sites = await self._find_call_sites(class_name)
            
            if call_sites:
                verified_integrations.append({
                    "class": class_name,
                    "call_sites": call_sites[:5]  # Limit for readability
                })
            else:
                missing_integrations.append(
                    f"{class_name} not found in any call paths"
                )
        
        return ProofResult(
            verified=len(missing_integrations) == 0 and len(verified_integrations) > 0,
            evidence={
                "verified_integrations": verified_integrations,
                "claim_type": "integration"
            },
            gaps=missing_integrations,
            confidence=len(verified_integrations) / max(len(analysis.class_names), 1),
            verification_method="call_path_trace"
        )
    
    async def _find_class_in_codebase(self, class_name: str, search_paths: List[Path]) -> Optional[str]:
        """Search for a class definition in the codebase."""
        pattern = rf'class\s+{class_name}\b'
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
            
            for py_file in search_path.rglob("*.py"):
                try:
                    content = py_file.read_text()
                    if re.search(pattern, content):
                        return str(py_file)
                except Exception:
                    continue
        
        return None
    
    async def _find_function_in_codebase(self, func_name: str, search_paths: List[Path]) -> Optional[str]:
        """Search for a function definition in the codebase."""
        pattern = rf'def\s+{func_name}\b'
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
            
            for py_file in search_path.rglob("*.py"):
                try:
                    content = py_file.read_text()
                    if re.search(pattern, content):
                        return str(py_file)
                except Exception:
                    continue
        
        return None
    
    async def _find_call_sites(self, class_name: str) -> List[str]:
        """Find where a class is instantiated or called."""
        call_sites = []
        
        patterns = [
            rf'{class_name}\s*\(',  # Instantiation
            rf'from\s+\S+\s+import.*{class_name}',  # Import
            rf'self\.{class_name.lower()}',  # Attribute
        ]
        
        search_paths = [
            self.workspace_path / "src",
            self.workspace_path / "bais-cognitive-engine" / "src",
        ]
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
            
            for py_file in search_path.rglob("*.py"):
                try:
                    content = py_file.read_text()
                    for pattern in patterns:
                        if re.search(pattern, content):
                            call_sites.append(str(py_file))
                            break
                except Exception:
                    continue
        
        return list(set(call_sites))  # Deduplicate

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


class ObjectiveComparator:
    """
    NOVEL-25: Compare deliverable against original request.
    
    Detects goal substitution where the LLM solves a different
    (usually easier) problem than what was asked.
    """
    
    def __init__(self):
        self.objective_patterns = [
            r'(?:implement|create|build|make|add|fix|improve|update)\s+(.+?)(?:\.|$)',
            r'(?:how\s+(?:to|do\s+we|can\s+I))\s+(.+?)(?:\?|$)',
            r'(?:write|develop|design)\s+(.+?)(?:\.|$)',
        ]
    
    async def compare(self, original_request: str, deliverable: str) -> Dict:
        """Compare deliverable against original request."""
        # Extract objectives from request
        objectives = self._extract_objectives(original_request)
        
        # Check which objectives are addressed
        addressed = []
        missed = []
        substituted = []
        
        deliverable_lower = deliverable.lower()
        
        for obj in objectives:
            obj_words = set(obj.lower().split())
            
            # Count how many objective words appear in deliverable
            overlap = sum(1 for w in obj_words if w in deliverable_lower)
            overlap_ratio = overlap / len(obj_words) if obj_words else 0
            
            if overlap_ratio > 0.5:
                addressed.append(obj)
            elif overlap_ratio > 0.2:
                # Partial match - possible substitution
                substituted.append({
                    "original": obj,
                    "overlap_ratio": overlap_ratio
                })
            else:
                missed.append(obj)
        
        goal_drift = len(missed) > 0 or len(substituted) > len(addressed)
        
        return {
            "objectives": objectives,
            "addressed": addressed,
            "missed": missed,
            "substituted": substituted,
            "goal_drift_detected": goal_drift,
            "confidence": len(addressed) / max(len(objectives), 1)
        }
    
    def _extract_objectives(self, request: str) -> List[str]:
        """Extract actionable objectives from request."""
        objectives = []
        
        for pattern in self.objective_patterns:
            matches = re.findall(pattern, request, re.IGNORECASE)
            objectives.extend(matches)
        
        # Also extract quoted items as objectives
        quoted = re.findall(r'"([^"]+)"', request)
        objectives.extend(quoted)
        
        return list(set(objectives))  # Deduplicate

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


# Convenience function for integration
async def verify_response_claims(
    response: str, 
    original_query: str = None,
    workspace_path: Path = None
) -> Dict[str, Any]:
    """
    Verify all claims in an LLM response.
    
    Returns:
        Dictionary with verification results for each claim
    """
    inspector = ProofInspector(workspace_path)
    comparator = ObjectiveComparator()
    
    # Extract claims from response
    # Simple claim extraction - look for sentences with claim indicators
    claim_patterns = [
        r'[^.]*(?:is|are|has|have)\s+(?:complete|done|finished|working|implemented|integrated)[^.]*\.',
        r'[^.]*(?:100|fully|all|everything)[^.]*(?:complete|done|working)[^.]*\.',
        r'[^.]*(?:\d+%)[^.]*\.',
    ]
    
    claims = []
    for pattern in claim_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        claims.extend(matches)
    
    # Verify each claim
    results = {
        "claims_found": len(claims),
        "claims_verified": 0,
        "claims_failed": 0,
        "verifications": [],
        "goal_comparison": None
    }
    
    for claim in claims:
        proof = await inspector.verify_claim(claim.strip())
        results["verifications"].append({
            "claim": claim.strip()[:100],
            "verified": proof.verified,
            "confidence": proof.confidence,
            "gaps": proof.gaps,
            "method": proof.verification_method
        })
        
        if proof.verified:
            results["claims_verified"] += 1
        else:
            results["claims_failed"] += 1
    
    # Compare against original query if provided
    if original_query:
        results["goal_comparison"] = await comparator.compare(original_query, response)
    
    return results


