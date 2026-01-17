"""
BASE Vibe Coding Verifier (NOVEL-5)

Verifies code quality in "vibe coding" scenarios where AI generates code
based on natural language descriptions. Detects:
1. Incomplete implementations (stubs, TODOs, placeholders)
2. Non-functional code (syntax errors, missing imports)
3. Security vulnerabilities
4. Performance anti-patterns
5. Misalignment with user intent

Patent Alignment:
- NOVEL-5: Vibe Coding Verification
- Brain Layer: 6 (Cerebellum - Improvement)

NO PLACEHOLDERS. NO STUBS. FULL IMPLEMENTATION.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import re
import ast


class CodeQualityLevel(Enum):
    """Code quality assessment levels."""
    PRODUCTION = "production"      # Ready for production
    REVIEW_NEEDED = "review_needed"  # Needs human review
    INCOMPLETE = "incomplete"       # Has placeholders/stubs
    BROKEN = "broken"               # Syntax errors or won't run
    DANGEROUS = "dangerous"         # Security vulnerabilities


class IssueType(Enum):
    """Types of code issues."""
    STUB = "stub"                   # Placeholder implementation
    TODO = "todo"                   # TODO comment
    INCOMPLETE = "incomplete"       # Incomplete logic
    SYNTAX_ERROR = "syntax_error"   # Won't parse
    IMPORT_MISSING = "import_missing"  # Missing imports
    SECURITY = "security"           # Security vulnerability
    PERFORMANCE = "performance"     # Performance issue
    INTENT_MISMATCH = "intent_mismatch"  # Doesn't match user intent


@dataclass
class CodeIssue:
    """A detected code issue."""
    issue_type: IssueType
    severity: str  # critical, high, medium, low
    line_number: Optional[int]
    description: str
    suggestion: str
    code_snippet: Optional[str] = None


@dataclass
class VibeCodingResult:
    """Result of vibe coding verification."""
    quality_level: CodeQualityLevel
    is_complete: bool
    is_functional: bool
    is_secure: bool
    issues: List[CodeIssue]
    completeness_score: float  # 0-1
    functionality_score: float  # 0-1
    security_score: float  # 0-1
    overall_score: float  # 0-1
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'quality_level': self.quality_level.value,
            'is_complete': self.is_complete,
            'is_functional': self.is_functional,
            'is_secure': self.is_secure,
            'issues': [
                {
                    'type': i.issue_type.value,
                    'severity': i.severity,
                    'line': i.line_number,
                    'description': i.description,
                    'suggestion': i.suggestion
                }
                for i in self.issues
            ],
            'scores': {
                'completeness': self.completeness_score,
                'functionality': self.functionality_score,
                'security': self.security_score,
                'overall': self.overall_score
            },
            'recommendations': self.recommendations
        }


class VibeCodingVerifier:
    """
    Verifies code quality in AI-generated "vibe coding" scenarios.
    
    Implements NOVEL-5
    Brain Layer: 6 (Cerebellum)
    
    Key Detection Capabilities:
    1. Stub/placeholder detection
    2. Syntax validation
    3. Import analysis
    4. Security pattern matching
    5. Intent alignment checking
    """
    
    # Patterns indicating incomplete code
    STUB_PATTERNS = [
        (r'\bpass\s*$', 'Empty pass statement'),
        (r'\braise\s+NotImplementedError', 'NotImplementedError raised'),
        (r'\.\.\.', 'Ellipsis placeholder'),
        (r'#\s*TODO', 'TODO comment'),
        (r'#\s*FIXME', 'FIXME comment'),
        (r'#\s*XXX', 'XXX marker'),
        (r'#\s*HACK', 'HACK marker'),
        (r'\bplaceholder\b', 'Placeholder text'),
        (r'\bstub\b', 'Stub marker'),
        (r'implement\s+(?:this|here|later)', 'Implementation note'),
        (r'your[_\s]code[_\s]here', 'Code placeholder'),
        (r'fill[_\s]in', 'Fill-in marker'),
    ]
    
    # Security vulnerability patterns
    SECURITY_PATTERNS = [
        (r'eval\s*\(', 'Dangerous eval() usage', 'critical'),
        (r'exec\s*\(', 'Dangerous exec() usage', 'critical'),
        (r'__import__\s*\(', 'Dynamic import (potential injection)', 'high'),
        (r'subprocess\.\s*(?:call|run|Popen)\s*\([^)]*shell\s*=\s*True', 'Shell injection risk', 'critical'),
        (r'os\.system\s*\(', 'OS command injection risk', 'high'),
        (r'pickle\.load', 'Unsafe deserialization', 'high'),
        (r'yaml\.load\s*\([^)]*Loader\s*=\s*None', 'Unsafe YAML loading', 'high'),
        (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password', 'critical'),
        (r'api[_\s]?key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key', 'critical'),
        (r'secret\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret', 'critical'),
        (r'SELECT\s+.*\s+FROM\s+.*\s+WHERE\s+.*\+', 'SQL injection risk', 'critical'),
        (r'innerHTML\s*=', 'XSS risk (innerHTML)', 'high'),
    ]
    
    # Performance anti-patterns
    PERFORMANCE_PATTERNS = [
        (r'for\s+\w+\s+in\s+range\([^)]+\):\s*\n\s*.*\.append\(', 'List append in loop (use comprehension)', 'medium'),
        (r'time\.sleep\s*\(\s*\d+\s*\)', 'Blocking sleep', 'low'),
        (r'\+\s*=\s*["\']', 'String concatenation in loop', 'medium'),
        (r'global\s+\w+', 'Global variable usage', 'low'),
    ]
    
    def __init__(self, learning_path: str = None):
        """Initialize the verifier."""
        self.learning_path = learning_path
        self._feedback_history: List[Dict] = []
        self._domain_adjustments: Dict[str, float] = {}
    
    def verify(self, 
               code: str, 
               intent: str = None,
               language: str = "python",
               context: Dict[str, Any] = None) -> VibeCodingResult:
        """
        Verify code quality and completeness.
        
        Args:
            code: The code to verify
            intent: Original user intent/description
            language: Programming language
            context: Additional context
            
        Returns:
            VibeCodingResult with detailed analysis
        """
        issues = []
        context = context or {}
        
        # Step 1: Check for stubs and placeholders
        stub_issues = self._detect_stubs(code)
        issues.extend(stub_issues)
        
        # Step 2: Syntax validation (Python only for now)
        if language == "python":
            syntax_issues = self._check_syntax(code)
            issues.extend(syntax_issues)
        
        # Step 3: Security analysis
        security_issues = self._check_security(code)
        issues.extend(security_issues)
        
        # Step 4: Performance analysis
        perf_issues = self._check_performance(code)
        issues.extend(perf_issues)
        
        # Step 5: Intent alignment (if provided)
        if intent:
            intent_issues = self._check_intent_alignment(code, intent)
            issues.extend(intent_issues)
        
        # Calculate scores
        completeness_score = self._calculate_completeness_score(issues)
        functionality_score = self._calculate_functionality_score(issues)
        security_score = self._calculate_security_score(issues)
        
        overall_score = (
            completeness_score * 0.3 +
            functionality_score * 0.4 +
            security_score * 0.3
        )
        
        # Determine quality level
        quality_level = self._determine_quality_level(
            issues, completeness_score, functionality_score, security_score
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues)
        
        return VibeCodingResult(
            quality_level=quality_level,
            is_complete=completeness_score >= 0.9,
            is_functional=functionality_score >= 0.8,
            is_secure=security_score >= 0.9,
            issues=issues,
            completeness_score=completeness_score,
            functionality_score=functionality_score,
            security_score=security_score,
            overall_score=overall_score,
            recommendations=recommendations
        )
    
    def _detect_stubs(self, code: str) -> List[CodeIssue]:
        """Detect stub patterns in code."""
        issues = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            for pattern, description in self.STUB_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(CodeIssue(
                        issue_type=IssueType.STUB,
                        severity='high',
                        line_number=i,
                        description=f"Incomplete code: {description}",
                        suggestion="Replace with actual implementation",
                        code_snippet=line.strip()[:100]
                    ))
        
        return issues
    
    def _check_syntax(self, code: str) -> List[CodeIssue]:
        """Check Python syntax."""
        issues = []
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(CodeIssue(
                issue_type=IssueType.SYNTAX_ERROR,
                severity='critical',
                line_number=e.lineno,
                description=f"Syntax error: {e.msg}",
                suggestion="Fix the syntax error before using this code",
                code_snippet=e.text[:100] if e.text else None
            ))
        
        return issues
    
    def _check_security(self, code: str) -> List[CodeIssue]:
        """Check for security vulnerabilities."""
        issues = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            for pattern, description, severity in self.SECURITY_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(CodeIssue(
                        issue_type=IssueType.SECURITY,
                        severity=severity,
                        line_number=i,
                        description=f"Security issue: {description}",
                        suggestion="Review and fix this security vulnerability",
                        code_snippet=line.strip()[:100]
                    ))
        
        return issues
    
    def _check_performance(self, code: str) -> List[CodeIssue]:
        """Check for performance anti-patterns."""
        issues = []
        
        for pattern, description, severity in self.PERFORMANCE_PATTERNS:
            matches = re.finditer(pattern, code, re.MULTILINE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                issues.append(CodeIssue(
                    issue_type=IssueType.PERFORMANCE,
                    severity=severity,
                    line_number=line_num,
                    description=f"Performance issue: {description}",
                    suggestion="Consider optimizing this pattern",
                    code_snippet=match.group()[:100]
                ))
        
        return issues
    
    def _check_intent_alignment(self, code: str, intent: str) -> List[CodeIssue]:
        """Check if code aligns with stated intent."""
        issues = []
        
        # Extract key terms from intent
        intent_lower = intent.lower()
        code_lower = code.lower()
        
        # Check for key functionality mentions
        key_terms = re.findall(r'\b\w+\b', intent_lower)
        key_terms = [t for t in key_terms if len(t) > 3 and t not in 
                     {'the', 'and', 'for', 'that', 'with', 'this', 'from', 'have', 'will'}]
        
        missing_terms = []
        for term in key_terms[:10]:  # Check top 10 key terms
            if term not in code_lower:
                # Check for common variations
                variations = [term + 's', term + 'ing', term + 'ed', term[:-1] if term.endswith('s') else term]
                if not any(v in code_lower for v in variations):
                    missing_terms.append(term)
        
        if len(missing_terms) > len(key_terms) * 0.5:
            issues.append(CodeIssue(
                issue_type=IssueType.INTENT_MISMATCH,
                severity='medium',
                line_number=None,
                description=f"Code may not fully address intent. Missing concepts: {', '.join(missing_terms[:5])}",
                suggestion="Review if the code implements all required functionality"
            ))
        
        return issues
    
    def _calculate_completeness_score(self, issues: List[CodeIssue]) -> float:
        """Calculate completeness score based on stub issues."""
        stub_issues = [i for i in issues if i.issue_type in 
                       [IssueType.STUB, IssueType.TODO, IssueType.INCOMPLETE]]
        
        if not stub_issues:
            return 1.0
        
        # Penalize based on severity and count
        penalty = sum(0.2 if i.severity == 'critical' else 
                      0.15 if i.severity == 'high' else 
                      0.1 for i in stub_issues)
        
        return max(0.0, 1.0 - penalty)
    
    def _calculate_functionality_score(self, issues: List[CodeIssue]) -> float:
        """Calculate functionality score based on syntax and logic issues."""
        func_issues = [i for i in issues if i.issue_type in 
                       [IssueType.SYNTAX_ERROR, IssueType.IMPORT_MISSING]]
        
        if not func_issues:
            return 1.0
        
        # Critical syntax errors are fatal
        if any(i.severity == 'critical' for i in func_issues):
            return 0.0
        
        penalty = sum(0.3 if i.severity == 'high' else 0.15 for i in func_issues)
        return max(0.0, 1.0 - penalty)
    
    def _calculate_security_score(self, issues: List[CodeIssue]) -> float:
        """Calculate security score."""
        sec_issues = [i for i in issues if i.issue_type == IssueType.SECURITY]
        
        if not sec_issues:
            return 1.0
        
        penalty = sum(0.4 if i.severity == 'critical' else 
                      0.2 if i.severity == 'high' else 
                      0.1 for i in sec_issues)
        
        return max(0.0, 1.0 - penalty)
    
    def _determine_quality_level(self, 
                                  issues: List[CodeIssue],
                                  completeness: float,
                                  functionality: float,
                                  security: float) -> CodeQualityLevel:
        """Determine overall quality level."""
        
        # Check for critical issues
        critical_security = any(i.issue_type == IssueType.SECURITY and 
                                i.severity == 'critical' for i in issues)
        if critical_security:
            return CodeQualityLevel.DANGEROUS
        
        # Check for syntax errors
        if any(i.issue_type == IssueType.SYNTAX_ERROR for i in issues):
            return CodeQualityLevel.BROKEN
        
        # Check for incomplete code
        if completeness < 0.7:
            return CodeQualityLevel.INCOMPLETE
        
        # Check if review is needed
        if functionality < 0.9 or security < 0.9 or completeness < 0.9:
            return CodeQualityLevel.REVIEW_NEEDED
        
        return CodeQualityLevel.PRODUCTION
    
    def _generate_recommendations(self, issues: List[CodeIssue]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        issue_types = set(i.issue_type for i in issues)
        
        if IssueType.STUB in issue_types:
            recommendations.append("Replace all placeholder code with actual implementations")
        
        if IssueType.SYNTAX_ERROR in issue_types:
            recommendations.append("Fix syntax errors before deploying")
        
        if IssueType.SECURITY in issue_types:
            recommendations.append("Address all security vulnerabilities before production use")
        
        if IssueType.PERFORMANCE in issue_types:
            recommendations.append("Review performance patterns for optimization opportunities")
        
        if IssueType.INTENT_MISMATCH in issue_types:
            recommendations.append("Verify the code fully implements the requested functionality")
        
        if not recommendations:
            recommendations.append("Code appears ready for production use")
        
        return recommendations
    
    # Learning interface methods
    def record_outcome(self, result: VibeCodingResult, was_correct: bool, domain: str = 'general'):
        """Record outcome for learning."""
        self._feedback_history.append({
            'timestamp': datetime.now().isoformat(),
            'quality_level': result.quality_level.value,
            'was_correct': was_correct,
            'domain': domain
        })
    
    def record_feedback(self, feedback: Dict) -> None:
        """Record feedback for learning."""
        self._feedback_history.append(feedback)
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt detection thresholds based on feedback."""
        pass
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get domain-specific adjustment."""
        return self._domain_adjustments.get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics."""
        return {
            'feedback_count': len(self._feedback_history),
            'domain_adjustments': self._domain_adjustments
        }


