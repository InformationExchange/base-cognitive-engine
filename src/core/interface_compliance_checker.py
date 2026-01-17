"""
BAIS Interface Compliance Checker (NOVEL-51)

Ensures learning interface methods are:
1. Actually inside the class (not module level)
2. Have correct signatures
3. Have required attributes initialized in __init__
4. Are not added to dataclasses

Created: January 4, 2026
Based on Case Study 3 findings
"""

import re
import ast
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class InterfaceViolation:
    """A single interface compliance violation."""
    class_name: str
    file_path: str
    violation_type: str
    description: str
    line_number: Optional[int] = None
    fix_suggestion: str = ""


@dataclass
class InterfaceComplianceResult:
    """Result of interface compliance check."""
    file_path: str
    total_classes_checked: int
    compliant_classes: int
    violations: List[InterfaceViolation] = field(default_factory=list)
    
    @property
    def is_compliant(self) -> bool:
        return len(self.violations) == 0
    
    @property
    def compliance_rate(self) -> float:
        if self.total_classes_checked == 0:
            return 1.0
        return self.compliant_classes / self.total_classes_checked


class InterfaceComplianceChecker:
    """
    Checks that learning interface methods are properly implemented.
    
    NOVEL-51: Prevents the "methods exist but outside class" problem
    found in Case Study 3.
    
    Checks:
    1. Methods are indented inside class (not module level)
    2. Method signatures match standard interface
    3. __init__ creates required attributes
    4. Dataclasses don't have learning methods
    5. Wrapper exists for non-standard signatures
    """
    
    REQUIRED_METHODS = {
        'record_outcome': 'def record_outcome(self, outcome: Dict)',
        'learn_from_feedback': 'def learn_from_feedback(self, feedback: Dict)',
        'get_statistics': 'def get_statistics(self) -> Dict',
        'serialize_state': 'def serialize_state(self) -> Dict',
        'deserialize_state': 'def deserialize_state(self, state: Dict)'
    }
    
    REQUIRED_INIT_ATTRS = ['_outcomes', '_learning_params']
    
    STANDARD_WRAPPER = 'record_outcome_standard'
    
    def __init__(self):
        self._outcomes = []
        self._learning_params = {}
        logger.info("[InterfaceCompliance] Checker initialized")
    
    def check_file(self, file_path: str) -> InterfaceComplianceResult:
        """
        Check a Python file for interface compliance.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            InterfaceComplianceResult with all violations found
        """
        violations = []
        classes_checked = 0
        compliant = 0
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Find all class definitions
            class_pattern = r'^class\s+(\w+)\s*[\(:]'
            for match in re.finditer(class_pattern, content, re.MULTILINE):
                class_name = match.group(1)
                
                # Skip non-actionable classes
                if any(x in class_name for x in ['Config', 'Result', 'Error', 'Enum', 'Type']):
                    continue
                
                classes_checked += 1
                class_violations = self._check_class(content, class_name, file_path)
                
                if class_violations:
                    violations.extend(class_violations)
                else:
                    compliant += 1
        
        except Exception as e:
            violations.append(InterfaceViolation(
                class_name="<file>",
                file_path=file_path,
                violation_type="FILE_ERROR",
                description=f"Could not parse file: {str(e)[:50]}"
            ))
        
        return InterfaceComplianceResult(
            file_path=file_path,
            total_classes_checked=classes_checked,
            compliant_classes=compliant,
            violations=violations
        )
    
    def _check_class(
        self,
        content: str,
        class_name: str,
        file_path: str
    ) -> List[InterfaceViolation]:
        """Check a single class for violations."""
        violations = []
        
        # Find class boundaries
        class_pattern = rf'class\s+{class_name}\s*[\(:].*?(?=\nclass\s|\Z)'
        match = re.search(class_pattern, content, re.DOTALL)
        
        if not match:
            return violations
        
        class_content = match.group()
        class_start_line = content[:match.start()].count('\n') + 1
        
        # Check 1: Is it a dataclass with learning methods?
        if self._is_dataclass_with_methods(content, class_name):
            violations.append(InterfaceViolation(
                class_name=class_name,
                file_path=file_path,
                violation_type="DATACLASS_MISUSE",
                description="Dataclass should not have learning interface methods",
                line_number=class_start_line,
                fix_suggestion="Remove learning methods from dataclass or convert to regular class"
            ))
        
        # Check 2: Are methods at correct indentation (inside class)?
        for method_name in self.REQUIRED_METHODS:
            if f'def {method_name}' in class_content:
                # Check indentation
                method_match = re.search(
                    rf'^(\s*)def {method_name}', 
                    class_content, 
                    re.MULTILINE
                )
                if method_match:
                    indent = len(method_match.group(1))
                    if indent < 4:
                        violations.append(InterfaceViolation(
                            class_name=class_name,
                            file_path=file_path,
                            violation_type="METHOD_OUTSIDE_CLASS",
                            description=f"{method_name} appears to be at module level (indent={indent})",
                            fix_suggestion=f"Move {method_name} inside {class_name} class with proper indentation"
                        ))
        
        # Check 3: Does __init__ create required attributes?
        if 'def __init__' in class_content:
            init_match = re.search(
                r'def __init__\([^)]*\):.*?(?=\n    def |\Z)',
                class_content,
                re.DOTALL
            )
            if init_match:
                init_content = init_match.group()
                for attr in self.REQUIRED_INIT_ATTRS:
                    if f'self.{attr}' not in init_content:
                        violations.append(InterfaceViolation(
                            class_name=class_name,
                            file_path=file_path,
                            violation_type="MISSING_INIT_ATTR",
                            description=f"__init__ does not initialize {attr}",
                            fix_suggestion=f"Add 'self.{attr} = []' to __init__"
                        ))
        
        # Check 4: Non-standard signature without wrapper
        if 'def record_outcome' in class_content:
            has_wrapper = self.STANDARD_WRAPPER in class_content
            
            # Check if signature is non-standard
            sig_match = re.search(
                r'def record_outcome\(self,\s*(\w+)',
                class_content
            )
            if sig_match and sig_match.group(1) != 'outcome' and not has_wrapper:
                violations.append(InterfaceViolation(
                    class_name=class_name,
                    file_path=file_path,
                    violation_type="MISSING_WRAPPER",
                    description=f"record_outcome has non-standard signature but no wrapper",
                    fix_suggestion=f"Add {self.STANDARD_WRAPPER}(self, outcome: Dict) wrapper method"
                ))
        
        return violations
    
    def _is_dataclass_with_methods(self, content: str, class_name: str) -> bool:
        """Check if class is a dataclass with learning methods."""
        # Check for @dataclass decorator
        pattern = rf'@dataclass\s*\n\s*class\s+{class_name}'
        if not re.search(pattern, content):
            return False
        
        # Check if it has learning methods
        class_pattern = rf'class\s+{class_name}\s*[\(:].*?(?=\nclass\s|\Z)'
        match = re.search(class_pattern, content, re.DOTALL)
        
        if match:
            class_content = match.group()
            return any(f'def {m}' in class_content for m in self.REQUIRED_METHODS)
        
        return False
    
    def check_directory(self, directory: str) -> Dict[str, Any]:
        """
        Check all Python files in a directory.
        
        Args:
            directory: Path to directory
            
        Returns:
            Summary of all violations found
        """
        results = []
        total_violations = 0
        total_classes = 0
        compliant_classes = 0
        
        for path in Path(directory).rglob('*.py'):
            if '__pycache__' in str(path) or 'test' in path.name.lower():
                continue
            
            result = self.check_file(str(path))
            results.append(result)
            total_violations += len(result.violations)
            total_classes += result.total_classes_checked
            compliant_classes += result.compliant_classes
        
        return {
            'total_files': len(results),
            'total_classes': total_classes,
            'compliant_classes': compliant_classes,
            'compliance_rate': compliant_classes / max(1, total_classes),
            'total_violations': total_violations,
            'violation_types': self._summarize_violations(results),
            'files_with_violations': [
                r.file_path for r in results if not r.is_compliant
            ]
        }
    
    def _summarize_violations(
        self,
        results: List[InterfaceComplianceResult]
    ) -> Dict[str, int]:
        """Summarize violation counts by type."""
        summary = {}
        for result in results:
            for v in result.violations:
                summary[v.violation_type] = summary.get(v.violation_type, 0) + 1
        return summary
    
    # Learning Interface Methods
    
    def record_outcome(self, outcome: Dict) -> None:
        """Record compliance check outcome."""
        self._outcomes.append(outcome)
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback."""
        pass
    
    def get_statistics(self) -> Dict:
        """Return statistics."""
        return {
            'total_checks': len(self._outcomes),
            'required_methods': list(self.REQUIRED_METHODS.keys())
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


if __name__ == "__main__":
    # Test the checker
    checker = InterfaceComplianceChecker()
    
    # Check a single file
    result = checker.check_file("core/engine.py")
    
    print(f"File: {result.file_path}")
    print(f"Classes checked: {result.total_classes_checked}")
    print(f"Compliant: {result.compliant_classes}")
    print(f"Violations: {len(result.violations)}")
    
    for v in result.violations[:5]:
        print(f"  - {v.class_name}: {v.violation_type}")
        print(f"    {v.description}")

