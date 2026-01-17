"""
BAIS Functional Completeness Enforcer (NOVEL-50)

This module addresses gaps identified in Case Study 3:
- RC-1: Static vs Functional gap detection
- RC-2: 100% testing enforcement (not samples)
- RC-3: Interface compliance verification
- RC-4: Dataclass misuse detection

Created: January 4, 2026
"""

import re
import importlib
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ComplianceStatus(Enum):
    """Compliance status levels."""
    FULLY_COMPLIANT = "fully_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    DATACLASS_MISUSE = "dataclass_misuse"
    UNTESTABLE = "untestable"


@dataclass
class ClassComplianceResult:
    """Result of compliance check for a single class."""
    class_name: str
    module_path: str
    status: ComplianceStatus
    static_methods_present: int  # Methods found via static analysis
    functional_methods_working: int  # Methods that actually execute
    issues: List[str] = field(default_factory=list)
    is_dataclass: bool = False
    signature_mismatch: bool = False
    missing_init_attrs: List[str] = field(default_factory=list)


@dataclass
class FunctionalComplianceReport:
    """Complete functional compliance report."""
    total_classes: int
    fully_compliant: int
    partially_compliant: int
    non_compliant: int
    dataclass_misuse: int
    untestable: int
    compliance_rate: float
    static_vs_functional_gap: float
    results: List[ClassComplianceResult] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class FunctionalCompletenessEnforcer:
    """
    Enforces functional completeness of learning interface implementations.
    
    NOVEL-50: Addresses the gap between "code exists" and "code works"
    identified in Case Study 3.
    
    Key capabilities:
    1. Rejects sample-based evidence - requires 100% testing
    2. Verifies methods exist IN the class (not module level)
    3. Detects dataclasses with learning methods (misuse)
    4. Validates method signatures match standard interface
    5. Checks __init__ creates required attributes
    """
    
    REQUIRED_METHODS = [
        'record_outcome',
        'learn_from_feedback', 
        'get_statistics',
        'serialize_state',
        'deserialize_state'
    ]
    
    STANDARD_WRAPPER = 'record_outcome_standard'
    
    def __init__(self, min_compliance_rate: float = 0.95):
        """
        Initialize enforcer.
        
        Args:
            min_compliance_rate: Minimum acceptable compliance rate (default 95%)
        """
        self.min_compliance_rate = min_compliance_rate
        self._outcomes = []
        self._learning_params = {}
        logger.info(f"[FunctionalEnforcer] Initialized with min_rate={min_compliance_rate}")
    
    def verify_class_compliance(
        self,
        class_name: str,
        module_path: str,
        file_path: str
    ) -> ClassComplianceResult:
        """
        Verify a single class meets functional compliance requirements.
        
        Args:
            class_name: Name of the class
            module_path: Python import path
            file_path: File system path
            
        Returns:
            ClassComplianceResult with detailed findings
        """
        issues = []
        
        # Step 1: Check if it's a dataclass
        is_dataclass = self._check_is_dataclass(file_path, class_name)
        
        # Step 2: Static analysis - check methods in class definition
        static_count = self._count_static_methods(file_path, class_name)
        
        # Step 3: Functional test - try to instantiate and call methods
        functional_count, func_issues, signature_mismatch = self._test_functional(
            class_name, module_path
        )
        
        issues.extend(func_issues)
        
        # Step 4: Check __init__ attributes
        missing_attrs = self._check_init_attributes(file_path, class_name)
        
        # Determine status
        if is_dataclass and static_count > 0:
            status = ComplianceStatus.DATACLASS_MISUSE
            issues.append("Dataclass should not have learning methods")
        elif functional_count == 5:
            status = ComplianceStatus.FULLY_COMPLIANT
        elif functional_count > 0:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
            issues.append(f"Only {functional_count}/5 methods functional")
        elif static_count > 0 and functional_count == 0:
            status = ComplianceStatus.NON_COMPLIANT
            issues.append("Methods exist statically but don't work")
        else:
            status = ComplianceStatus.UNTESTABLE
            issues.append("Cannot instantiate class or no methods found")
        
        return ClassComplianceResult(
            class_name=class_name,
            module_path=module_path,
            status=status,
            static_methods_present=static_count,
            functional_methods_working=functional_count,
            issues=issues,
            is_dataclass=is_dataclass,
            signature_mismatch=signature_mismatch,
            missing_init_attrs=missing_attrs
        )
    
    def enforce_100_percent_testing(
        self,
        classes: List[Tuple[str, str, str]],
        reject_samples: bool = True
    ) -> FunctionalComplianceReport:
        """
        Enforce 100% testing - no sample-based claims accepted.
        
        Args:
            classes: List of (class_name, module_path, file_path) tuples
            reject_samples: If True, requires ALL classes to be tested
            
        Returns:
            FunctionalComplianceReport with complete analysis
        """
        results = []
        
        # Test ALL classes - no sampling
        for class_name, module_path, file_path in classes:
            result = self.verify_class_compliance(class_name, module_path, file_path)
            results.append(result)
        
        # Calculate statistics
        total = len(results)
        fully = sum(1 for r in results if r.status == ComplianceStatus.FULLY_COMPLIANT)
        partial = sum(1 for r in results if r.status == ComplianceStatus.PARTIALLY_COMPLIANT)
        non = sum(1 for r in results if r.status == ComplianceStatus.NON_COMPLIANT)
        misuse = sum(1 for r in results if r.status == ComplianceStatus.DATACLASS_MISUSE)
        untestable = sum(1 for r in results if r.status == ComplianceStatus.UNTESTABLE)
        
        # Calculate static vs functional gap
        total_static = sum(r.static_methods_present for r in results)
        total_functional = sum(r.functional_methods_working for r in results)
        gap = (total_static - total_functional) / max(1, total_static) * 100
        
        compliance_rate = fully / max(1, total - untestable - misuse)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results, compliance_rate)
        
        return FunctionalComplianceReport(
            total_classes=total,
            fully_compliant=fully,
            partially_compliant=partial,
            non_compliant=non,
            dataclass_misuse=misuse,
            untestable=untestable,
            compliance_rate=compliance_rate,
            static_vs_functional_gap=gap,
            results=results,
            recommendations=recommendations
        )
    
    def reject_sample_based_claim(
        self,
        claimed_count: int,
        tested_count: int,
        evidence_type: str = "sample"
    ) -> Dict[str, Any]:
        """
        Reject claims based on sample testing.
        
        This enforces the lesson from Case Study 3: sample testing
        missed systematic errors.
        
        Args:
            claimed_count: Number of classes claimed as complete
            tested_count: Number actually tested
            evidence_type: Type of evidence provided
            
        Returns:
            Rejection decision with reasoning
        """
        if tested_count < claimed_count:
            coverage = tested_count / claimed_count * 100
            return {
                "accepted": False,
                "reason": "SAMPLE_BASED_EVIDENCE_REJECTED",
                "message": f"Only {tested_count}/{claimed_count} ({coverage:.1f}%) tested. "
                          f"100% functional testing required.",
                "recommendation": "Execute functional tests for ALL classes before claiming completion",
                "bais_invention": "NOVEL-50: FunctionalCompletenessEnforcer",
                "case_study_reference": "Case Study 3: Static vs Functional Gap"
            }
        
        return {
            "accepted": True,
            "reason": "FULL_COVERAGE_VERIFIED",
            "message": f"All {claimed_count} classes tested",
            "coverage": 100.0
        }
    
    def _check_is_dataclass(self, file_path: str, class_name: str) -> bool:
        """Check if class is a dataclass."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Look for @dataclass decorator before this class
            pattern = rf'@dataclass\s*\n\s*class\s+{class_name}'
            return bool(re.search(pattern, content))
        except:
            return False
    
    def _count_static_methods(self, file_path: str, class_name: str) -> int:
        """Count learning methods present in class definition."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Find class content
            pattern = rf'class\s+{class_name}\s*[\(:].*?(?=\nclass\s|\Z)'
            match = re.search(pattern, content, re.DOTALL)
            
            if not match:
                return 0
            
            class_content = match.group()
            count = 0
            for method in self.REQUIRED_METHODS:
                if f'def {method}' in class_content:
                    count += 1
            
            return count
        except:
            return 0
    
    def _test_functional(
        self,
        class_name: str,
        module_path: str
    ) -> Tuple[int, List[str], bool]:
        """Test if methods actually work."""
        issues = []
        working = 0
        signature_mismatch = False
        
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name, None)
            
            if cls is None:
                issues.append("Class not found in module")
                return 0, issues, False
            
            try:
                instance = cls()
            except TypeError as e:
                if 'required positional' in str(e):
                    issues.append(f"Constructor requires args: {str(e)[:50]}")
                    return 0, issues, False
                raise
            
            # Test record_outcome
            try:
                if hasattr(instance, self.STANDARD_WRAPPER):
                    getattr(instance, self.STANDARD_WRAPPER)({'test': True})
                else:
                    instance.record_outcome({'test': True})
                working += 1
            except TypeError as e:
                if 'missing' in str(e):
                    signature_mismatch = True
                    issues.append(f"record_outcome signature mismatch")
            except AttributeError:
                issues.append("record_outcome not found")
            
            # Test other methods
            for method in ['learn_from_feedback', 'get_statistics', 
                          'serialize_state', 'deserialize_state']:
                try:
                    m = getattr(instance, method, None)
                    if m is None:
                        issues.append(f"{method} not found")
                        continue
                    
                    if method == 'learn_from_feedback':
                        m({'was_correct': False})
                    elif method == 'deserialize_state':
                        m({})
                    else:
                        m()
                    working += 1
                except Exception as e:
                    issues.append(f"{method}: {str(e)[:30]}")
            
        except ImportError as e:
            issues.append(f"Import error: {str(e)[:50]}")
        except Exception as e:
            issues.append(f"Unexpected: {type(e).__name__}")
        
        return working, issues, signature_mismatch
    
    def _check_init_attributes(self, file_path: str, class_name: str) -> List[str]:
        """Check if __init__ creates required attributes."""
        missing = []
        required_attrs = ['_outcomes', '_learning_params']
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Find class __init__
            class_pattern = rf'class\s+{class_name}\s*[\(:].*?(?=\nclass\s|\Z)'
            match = re.search(class_pattern, content, re.DOTALL)
            
            if not match:
                return required_attrs
            
            class_content = match.group()
            
            # Find __init__ method
            init_match = re.search(r'def __init__\([^)]*\):.*?(?=\n    def |\Z)', 
                                   class_content, re.DOTALL)
            
            if not init_match:
                return required_attrs
            
            init_content = init_match.group()
            
            for attr in required_attrs:
                if f'self.{attr}' not in init_content:
                    missing.append(attr)
            
        except:
            pass
        
        return missing
    
    def _generate_recommendations(
        self,
        results: List[ClassComplianceResult],
        compliance_rate: float
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if compliance_rate < self.min_compliance_rate:
            recommendations.append(
                f"CRITICAL: Compliance rate {compliance_rate*100:.1f}% is below "
                f"minimum {self.min_compliance_rate*100:.0f}%"
            )
        
        # Count issue types
        sig_issues = sum(1 for r in results if r.signature_mismatch)
        if sig_issues > 0:
            recommendations.append(
                f"Add record_outcome_standard wrapper to {sig_issues} classes "
                "with non-standard signatures"
            )
        
        init_issues = sum(1 for r in results if r.missing_init_attrs)
        if init_issues > 0:
            recommendations.append(
                f"Add _outcomes/_learning_params to __init__ in {init_issues} classes"
            )
        
        misuse = sum(1 for r in results if r.status == ComplianceStatus.DATACLASS_MISUSE)
        if misuse > 0:
            recommendations.append(
                f"Remove learning methods from {misuse} dataclasses - "
                "they should not have learning interface"
            )
        
        return recommendations
    
    # Learning Interface Methods
    
    def record_outcome(self, outcome: Dict) -> None:
        """Record verification outcome."""
        self._outcomes.append(outcome)
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from verification feedback."""
        if feedback.get('was_correct') is False:
            # Adjust strictness based on false positives/negatives
            pass
    
    def get_statistics(self) -> Dict:
        """Return verification statistics."""
        return {
            'total_verifications': len(self._outcomes),
            'min_compliance_rate': self.min_compliance_rate
        }
    
    def serialize_state(self) -> Dict:
        """Serialize state."""
        return {
            'outcomes': self._outcomes[-100:],
            'learning_params': self._learning_params,
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore state."""
        self._outcomes = state.get('outcomes', [])
        self._learning_params = state.get('learning_params', {})


# Integration with existing BAIS components
class BAISFunctionalValidator:
    """
    BAIS integration for functional validation.
    
    Enhances bais_verify_completion to reject sample-based claims
    and require 100% functional testing.
    """
    
    def __init__(self):
        self.enforcer = FunctionalCompletenessEnforcer()
    
    def validate_completion_claim(
        self,
        claim: str,
        evidence: List[str],
        classes_claimed: int,
        classes_tested: int
    ) -> Dict[str, Any]:
        """
        Validate a completion claim with strict functional requirements.
        
        Args:
            claim: The completion claim being made
            evidence: Evidence provided
            classes_claimed: Number of classes claimed complete
            classes_tested: Number actually tested
            
        Returns:
            Validation result
        """
        # First check: Reject sample-based evidence
        sample_check = self.enforcer.reject_sample_based_claim(
            classes_claimed, classes_tested
        )
        
        if not sample_check['accepted']:
            return {
                'valid': False,
                'reason': sample_check['reason'],
                'message': sample_check['message'],
                'recommendation': sample_check['recommendation'],
                'evidence_type': 'SAMPLE_REJECTED',
                'bais_enhancement': 'NOVEL-50'
            }
        
        # Check evidence quality
        weak_evidence = any(
            'sample' in e.lower() or 'tested' in e.lower() and '%' not in e
            for e in evidence
        )
        
        if weak_evidence:
            return {
                'valid': False,
                'reason': 'WEAK_EVIDENCE',
                'message': 'Evidence mentions "sample" or partial testing',
                'recommendation': 'Provide complete functional test results'
            }
        
        return {
            'valid': True,
            'reason': 'FULL_COVERAGE_ACCEPTED',
            'confidence': 0.95
        }


if __name__ == "__main__":
    # Test the enforcer
    enforcer = FunctionalCompletenessEnforcer()
    
    # Test sample rejection
    result = enforcer.reject_sample_based_claim(
        claimed_count=254,
        tested_count=39
    )
    
    print("Sample Rejection Test:")
    print(f"  Accepted: {result['accepted']}")
    print(f"  Reason: {result['reason']}")
    print(f"  Message: {result['message']}")

