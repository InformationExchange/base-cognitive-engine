"""
Implementation Completeness Analyzer (PPA3-NEW-1)

New BASE invention to detect partial implementations by checking
interface compliance at the code level, not just text analysis.

This addresses the gap where BASE accepted "class exists" as proof
of "fully implemented" when methods were actually missing.
"""

import logging
import inspect
from typing import Dict, List, Any, Optional, Type, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ComplianceLevel(Enum):
    """Implementation compliance levels."""
    FULL = "full"           # 100% of interface implemented
    SUBSTANTIAL = "substantial"  # 80-99% implemented
    PARTIAL = "partial"     # 50-79% implemented
    MINIMAL = "minimal"     # 20-49% implemented
    STUB = "stub"           # <20% implemented


@dataclass
class InterfaceDefinition:
    """Defines an expected interface contract."""
    name: str
    required_methods: List[str]
    optional_methods: List[str] = field(default_factory=list)
    required_attributes: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class ComplianceResult:
    """Result of compliance check."""
    class_name: str
    interface_name: str
    compliance_score: float
    compliance_level: ComplianceLevel
    missing_required: List[str]
    missing_optional: List[str]
    missing_attributes: List[str]
    present_methods: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'class_name': self.class_name,
            'interface_name': self.interface_name,
            'compliance_score': self.compliance_score,
            'compliance_level': self.compliance_level.value,
            'missing_required': self.missing_required,
            'missing_optional': self.missing_optional,
            'missing_attributes': self.missing_attributes,
            'present_methods': self.present_methods,
            'recommendations': self.recommendations,
        }


# Standard interface definitions for BASE modules
LEARNING_INTERFACE = InterfaceDefinition(
    name="LearningCapable",
    required_methods=[
        'record_outcome',
        'record_feedback', 
        'adapt_thresholds',
        'get_domain_adjustment',
        'get_learning_statistics'
    ],
    optional_methods=[
        'save_state',
        'load_state',
        'reset_learning'
    ],
    description="Standard interface for modules with learning capabilities"
)

PERSISTENCE_INTERFACE = InterfaceDefinition(
    name="Persistable",
    required_methods=[
        'save_state',
        'load_state'
    ],
    optional_methods=[
        'get_state_path',
        'clear_state'
    ],
    description="Standard interface for modules with state persistence"
)

DETECTOR_INTERFACE = InterfaceDefinition(
    name="Detector",
    required_methods=[
        'detect',
        'analyze'
    ],
    optional_methods=[
        'configure',
        'reset',
        'get_statistics'
    ],
    required_attributes=[
        'name',
        'version'
    ],
    description="Standard interface for detector modules"
)

AI_ENHANCED_INTERFACE = InterfaceDefinition(
    name="AIEnhanced",
    required_methods=[
        'analyze_with_ai',
        'learn_from_feedback'
    ],
    optional_methods=[
        'get_ai_statistics',
        'toggle_ai'
    ],
    required_attributes=[
        'ai_enabled'
    ],
    description="Standard interface for AI-enhanced modules"
)


class ImplementationCompletenessAnalyzer:
    """
    Analyzes code implementations against expected interfaces.
    
    This is a new BASE invention (PPA3-NEW-1) that enables detection
    of partial implementations at the code level.
    """
    
    def __init__(self):
        self.interfaces: Dict[str, InterfaceDefinition] = {
            'learning': LEARNING_INTERFACE,
            'persistence': PERSISTENCE_INTERFACE,
            'detector': DETECTOR_INTERFACE,
            'ai_enhanced': AI_ENHANCED_INTERFACE,
        }
        self.analysis_history: List[ComplianceResult] = []
        logger.info("[ImplementationCompleteness] Analyzer initialized")
    
    def register_interface(self, name: str, interface: InterfaceDefinition):
        """Register a custom interface definition."""
        self.interfaces[name] = interface
        logger.info(f"[ImplementationCompleteness] Registered interface: {name}")
    
    def analyze(
        self,
        class_obj: Type,
        interface_name: str = 'learning'
    ) -> ComplianceResult:
        """
        Analyze a class against an interface definition.
        
        Args:
            class_obj: The class to analyze
            interface_name: Name of the interface to check against
            
        Returns:
            ComplianceResult with detailed findings
        """
        interface = self.interfaces.get(interface_name)
        if not interface:
            raise ValueError(f"Unknown interface: {interface_name}")
        
        class_name = class_obj.__name__
        
        # Check required methods
        missing_required = []
        present_methods = []
        for method in interface.required_methods:
            if hasattr(class_obj, method):
                present_methods.append(method)
            else:
                missing_required.append(method)
        
        # Check optional methods
        missing_optional = []
        for method in interface.optional_methods:
            if hasattr(class_obj, method):
                present_methods.append(method)
            else:
                missing_optional.append(method)
        
        # Check required attributes
        missing_attributes = []
        for attr in interface.required_attributes:
            if not hasattr(class_obj, attr):
                missing_attributes.append(attr)
        
        # Calculate compliance score (required methods only)
        total_required = len(interface.required_methods)
        implemented = total_required - len(missing_required)
        compliance_score = implemented / total_required if total_required > 0 else 1.0
        
        # Determine compliance level
        if compliance_score >= 1.0:
            level = ComplianceLevel.FULL
        elif compliance_score >= 0.8:
            level = ComplianceLevel.SUBSTANTIAL
        elif compliance_score >= 0.5:
            level = ComplianceLevel.PARTIAL
        elif compliance_score >= 0.2:
            level = ComplianceLevel.MINIMAL
        else:
            level = ComplianceLevel.STUB
        
        # Generate recommendations
        recommendations = []
        if missing_required:
            recommendations.append(
                f"Add missing required methods: {', '.join(missing_required)}"
            )
        if missing_attributes:
            recommendations.append(
                f"Add missing attributes: {', '.join(missing_attributes)}"
            )
        if level != ComplianceLevel.FULL:
            recommendations.append(
                f"Current compliance: {compliance_score:.0%} - "
                f"need {len(missing_required)} more methods for full compliance"
            )
        
        result = ComplianceResult(
            class_name=class_name,
            interface_name=interface.name,
            compliance_score=compliance_score,
            compliance_level=level,
            missing_required=missing_required,
            missing_optional=missing_optional,
            missing_attributes=missing_attributes,
            present_methods=present_methods,
            recommendations=recommendations
        )
        
        self.analysis_history.append(result)
        return result
    
    def analyze_module(
        self,
        module_path: str,
        class_name: str,
        interface_name: str = 'learning'
    ) -> ComplianceResult:
        """
        Analyze a class by importing from module path.
        
        Args:
            module_path: Python module path (e.g., 'core.drift_detection')
            class_name: Name of the class to analyze
            interface_name: Interface to check against
            
        Returns:
            ComplianceResult
        """
        try:
            mod = __import__(module_path, fromlist=[class_name])
            cls = getattr(mod, class_name)
            return self.analyze(cls, interface_name)
        except ImportError as e:
            return ComplianceResult(
                class_name=class_name,
                interface_name=interface_name,
                compliance_score=0.0,
                compliance_level=ComplianceLevel.STUB,
                missing_required=['MODULE_NOT_FOUND'],
                missing_optional=[],
                missing_attributes=[],
                present_methods=[],
                recommendations=[f"Fix import error: {e}"]
            )
        except AttributeError:
            return ComplianceResult(
                class_name=class_name,
                interface_name=interface_name,
                compliance_score=0.0,
                compliance_level=ComplianceLevel.STUB,
                missing_required=['CLASS_NOT_FOUND'],
                missing_optional=[],
                missing_attributes=[],
                present_methods=[],
                recommendations=[f"Class {class_name} not found in {module_path}"]
            )
    
    def batch_analyze(
        self,
        modules: List[tuple],
        interface_name: str = 'learning'
    ) -> Dict[str, ComplianceResult]:
        """
        Analyze multiple modules at once.
        
        Args:
            modules: List of (class_name, module_path) tuples
            interface_name: Interface to check against
            
        Returns:
            Dict mapping class names to results
        """
        results = {}
        for class_name, module_path in modules:
            result = self.analyze_module(module_path, class_name, interface_name)
            results[class_name] = result
        return results
    
    def generate_compliance_report(
        self,
        results: Dict[str, ComplianceResult]
    ) -> str:
        """Generate a human-readable compliance report."""
        lines = [
            "=" * 70,
            "IMPLEMENTATION COMPLETENESS REPORT",
            "=" * 70,
            ""
        ]
        
        # Summary statistics
        full = sum(1 for r in results.values() if r.compliance_level == ComplianceLevel.FULL)
        partial = sum(1 for r in results.values() if r.compliance_level in 
                     [ComplianceLevel.SUBSTANTIAL, ComplianceLevel.PARTIAL])
        incomplete = sum(1 for r in results.values() if r.compliance_level in
                        [ComplianceLevel.MINIMAL, ComplianceLevel.STUB])
        
        lines.append(f"Total Modules: {len(results)}")
        lines.append(f"  Full Compliance:    {full}")
        lines.append(f"  Partial:            {partial}")
        lines.append(f"  Incomplete:         {incomplete}")
        lines.append("")
        
        # Detailed results
        lines.append("-" * 70)
        lines.append(f"{'Class':<35} {'Score':<8} {'Level':<12} {'Missing'}")
        lines.append("-" * 70)
        
        for name, result in sorted(results.items(), key=lambda x: x[1].compliance_score):
            missing = len(result.missing_required)
            lines.append(
                f"{name:<35} {result.compliance_score:>5.0%}   "
                f"{result.compliance_level.value:<12} {missing} methods"
            )
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def get_base_prompt(self, result: ComplianceResult) -> str:
        """
        Generate a BASE-style prompt to correct the implementation.
        
        This is what BASE would use to prompt the LLM to fix issues.
        """
        if result.compliance_level == ComplianceLevel.FULL:
            return f"Class {result.class_name} has full {result.interface_name} compliance."
        
        prompt = f"""BASE IMPLEMENTATION GAP DETECTED

Class: {result.class_name}
Interface: {result.interface_name}
Compliance: {result.compliance_score:.0%}
Status: {result.compliance_level.value.upper()}

Missing Required Methods:
{chr(10).join(f'  - {m}' for m in result.missing_required)}

ACTION REQUIRED:
Add the following methods to {result.class_name}:

"""
        for method in result.missing_required:
            prompt += f"""
    def {method}(self, ...):
        \"\"\"[Add implementation for {result.interface_name} compliance]\"\"\"
        pass
"""
        
        prompt += f"""
This class will remain flagged as PARTIAL until all required methods are implemented.
"""
        return prompt

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


# Module-level instance
_analyzer: Optional[ImplementationCompletenessAnalyzer] = None


def get_completeness_analyzer() -> ImplementationCompletenessAnalyzer:
    """Get or create the completeness analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = ImplementationCompletenessAnalyzer()
    return _analyzer


# Test
if __name__ == "__main__":
    print("=" * 70)
    print("Implementation Completeness Analyzer - Test")
    print("=" * 70)
    
    analyzer = ImplementationCompletenessAnalyzer()
    
    # Test modules
    test_modules = [
        ("EnhancedMonitoringEngine", "core.realtime_monitoring"),
        ("BiasEvolutionTracker", "core.bias_evolution_tracker"),
        ("GroundingDetector", "detectors.grounding"),
        ("DriftDetectionManager", "core.drift_detection"),
    ]
    
    results = analyzer.batch_analyze(test_modules, 'learning')
    
    print(analyzer.generate_compliance_report(results))
    
    # Show BASE prompt for one partial
    for name, result in results.items():
        if result.compliance_level != ComplianceLevel.FULL:
            print("\n" + "=" * 70)
            print("EXAMPLE BASE CORRECTION PROMPT:")
            print("=" * 70)
            print(analyzer.get_base_prompt(result))
            break

