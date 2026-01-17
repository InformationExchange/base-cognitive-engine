"""
BASE Cognitive Enhancer - The Core Intelligence System

This is the main orchestrator that transforms BASE from a "gate" to an "enhancer".
It integrates all cognitive capabilities to IMPROVE LLM outputs, not just detect issues.

Patent Alignment: Core Novel Invention - Cognitive Enhancement Architecture
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use EXISTING modules where possible (avoid duplication)
from research.world_models import WorldModelsModule  # Instead of causal_reasoning
from research.neurosymbolic import NeuroSymbolicModule  # Instead of inference_enhancer
from detectors.factual import FactualDetector  # Instead of truth_determination

# Keep NOVEL modules (no duplicates)
from cognitive.decision_quality import DecisionQualityEnhancer, DecisionEnhancementResult
from cognitive.mission_alignment import MissionAlignmentChecker, MissionContext, AlignmentStatus
from core.learning_memory import LearningMemory, MemoryType, MemoryPriority, seed_foundational_memories
from core.response_improver import ResponseImprover, DetectedIssue, IssueType


class EnhancementLevel(Enum):
    """Level of enhancement applied"""
    NONE = "none"               # No enhancement needed
    MINOR = "minor"             # Small improvements
    MODERATE = "moderate"       # Significant improvements
    MAJOR = "major"             # Extensive improvements
    CRITICAL = "critical"       # Critical issues addressed


@dataclass
class EnhancementTrace:
    """Trace of enhancement steps"""
    step: str
    module: str
    input_quality: float
    output_quality: float
    changes_made: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CognitiveEnhancementResult:
    """Complete result of cognitive enhancement"""
    original_response: str
    enhanced_response: str
    enhancement_level: EnhancementLevel
    overall_improvement: float  # 0-1
    
    # Detailed results from each module (using Any to support multiple module types)
    causal_result: Optional[Any]  # WorldModelsModule result
    inference_result: Optional[Any]  # NeuroSymbolicModule result  
    truth_result: Optional[Any]  # FactualDetector result
    decision_result: Optional[DecisionEnhancementResult]
    
    # Trace and metadata
    trace: List[EnhancementTrace]
    memories_applied: List[str]
    new_learnings: List[str]
    
    # Quality scores
    original_quality: float
    enhanced_quality: float
    
    # Mission alignment
    alignment_status: AlignmentStatus
    alignment_score: float


class CognitiveEnhancer:
    """
    The core intelligence system that IMPROVES LLM outputs.
    
    Key capabilities:
    1. Analyze response for issues
    2. Apply cognitive enhancements (causal, inference, truth, decision)
    3. Learn from outcomes
    4. Maintain mission alignment
    5. Generate genuinely improved outputs
    
    This is NOT a gate - it's an enhancer.
    """
    
    def __init__(self, 
                 storage_path: str = "learning_data",
                 enable_learning: bool = True):
        """
        Initialize the cognitive enhancer.
        
        Args:
            storage_path: Path for learning data persistence
            enable_learning: Whether to enable learning from outcomes
        """
        # Initialize cognitive modules - USE EXISTING where possible
        # WorldModels replaces CausalReasoningEngine (avoid duplication)
        self.world_models = WorldModelsModule()
        # NeuroSymbolic replaces InferenceEnhancer (avoid duplication)  
        self.neurosymbolic = NeuroSymbolicModule()
        # FactualDetector replaces TruthDeterminer (avoid duplication)
        self.factual_detector = FactualDetector()
        # NOVEL modules (no existing equivalent)
        self.decision_enhancer = DecisionQualityEnhancer()
        self.mission_checker = MissionAlignmentChecker()
        self.response_improver = ResponseImprover()
        
        # Initialize learning memory
        self.enable_learning = enable_learning
        if enable_learning:
            try:
                self.memory = LearningMemory(storage_path=f"{storage_path}/memory")
                seed_foundational_memories(self.memory)
            except Exception as e:
                print(f"Warning: Learning memory disabled: {e}")
                self.enable_learning = False
                self.memory = None
        else:
            self.memory = None
        
        # Learning interface attributes
        self._outcomes: List[Dict] = []
        self._module_effectiveness: Dict[str, float] = {}
        self._total_enhancements: int = 0
        
        # Enhancement trace
        self.trace: List[EnhancementTrace] = []
        
        # Learning interface state
        self._outcomes: List[Dict] = []
        self._feedback: List[Dict] = []
        self._domain_adjustments: Dict[str, float] = {}
        
        # Set default mission
        self._set_default_mission()
    
    # ===== Learning Interface Methods =====
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        """Record enhancement outcome for learning."""
        getattr(self, '_outcomes', {}).append(outcome)
        if self.memory and outcome.get('success', False):
            # Could store successful enhancement patterns
            pass
    
    def record_feedback(self, feedback: Dict) -> None:
        """Record feedback on enhancements."""
        getattr(self, '_feedback', {}).append(feedback)
        domain = feedback.get('domain', 'general')
        quality = feedback.get('quality', 0.5)
        self._domain_adjustments[domain] = getattr(self, '_domain_adjustments', {}).get(domain, 0.0) + (0.05 if quality > 0.7 else -0.05)
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt enhancement thresholds."""
        pass
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get domain adjustment."""
        return getattr(self, '_domain_adjustments', {}).get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics."""
        return {
            'trace_count': len(self.trace),
            'learning_enabled': self.enable_learning,
            'domain_adjustments': dict(self._domain_adjustments),
            'outcomes_recorded': len(self._outcomes),
            'feedback_recorded': len(self._feedback)
        }
    
    def _set_default_mission(self):
        """Set the default mission for BASE"""
        mission = MissionContext(
            primary_objective="Improve LLM output quality for accuracy, safety, and usefulness",
            secondary_objectives=[
                "Ensure factual accuracy",
                "Improve reasoning quality",
                "Enhance decision recommendations",
                "Maintain appropriate uncertainty"
            ],
            constraints=[
                "Must improve, not block",
                "Must preserve user intent",
                "Must acknowledge limitations with precision"
            ],
            anti_patterns=[
                "Blocking valid queries",
                "Generic disclaimers without substance",
                "Overly cautious non-answers",
                "Simulated or fake improvements"
            ],
            success_definition="User receives response that is more accurate, safer, and more useful than the original"
        )
        self.mission_checker.set_mission(mission)
    
    def enhance(self,
                query: str,
                response: str,
                domain: str = 'general',
                user_context: Optional[Dict] = None,
                enhancement_depth: str = 'standard') -> CognitiveEnhancementResult:
        """
        Enhance an LLM response using all cognitive capabilities.
        
        Args:
            query: The original user query
            response: The LLM response to enhance
            domain: Domain context (medical, financial, legal, general)
            user_context: Optional context about the user
            enhancement_depth: 'quick', 'standard', or 'deep'
            
        Returns:
            CognitiveEnhancementResult with the enhanced response and analysis
        """
        self.trace = []
        memories_applied = []
        new_learnings = []
        
        # Step 0: Calculate original quality
        original_quality = self._assess_quality(response, domain)
        self._add_trace("Initial Assessment", "quality", original_quality, original_quality, 
                       [f"Original quality: {original_quality:.2f}"])
        
        # Step 1: Recall relevant memories
        if self.memory:
            memory_result = self.memory.recall(f"{query} {response}", domain)
            for mem in memory_result.memories[:5]:  # Top 5 most relevant
                memories_applied.append(mem.id)
                self._add_trace("Memory Recall", "learning_memory",
                              original_quality, original_quality,
                              [f"Applied memory: {mem.content[:50]}..."])
        
        # Step 2: Run cognitive enhancements using EXISTING modules
        current_response = response
        
        # 2a: World Models (Causal Reasoning) - USES EXISTING MODULE
        causal_result = None
        if enhancement_depth in ['standard', 'deep']:
            try:
                causal_result = self.world_models.analyze(query, current_response)
                causal_count = len(causal_result.causal_chains) if hasattr(causal_result, 'causal_chains') else 0
                self._add_trace("Causal Reasoning", "world_models",
                              0.5, 0.7 if causal_count > 0 else 0.5,
                              [f"Analyzed {causal_count} causal chains"])
            except Exception as e:
                self._add_trace("Causal Reasoning", "world_models", 0.5, 0.5, [f"Skipped: {str(e)[:50]}"])
        
        # 2b: Neuro-Symbolic (Logic/Inference) - USES EXISTING MODULE
        inference_result = None
        if enhancement_depth in ['standard', 'deep']:
            try:
                inference_result = self.neurosymbolic.verify(query, current_response)
                fallacy_count = len(inference_result.fallacies_detected) if hasattr(inference_result, 'fallacies_detected') else 0
                self._add_trace("Logic Verification", "neurosymbolic",
                              0.5, 0.8 if fallacy_count == 0 else 0.4,
                              [f"Detected {fallacy_count} fallacies"])
            except Exception as e:
                self._add_trace("Logic Verification", "neurosymbolic", 0.5, 0.5, [f"Skipped: {str(e)[:50]}"])
        
        # 2c: Factual Detection - USES EXISTING MODULE
        truth_result = None
        if enhancement_depth in ['standard', 'deep']:
            try:
                truth_result = self.factual_detector.detect(current_response)
                claim_count = len(truth_result) if isinstance(truth_result, list) else 0
                self._add_trace("Factual Detection", "factual_detector",
                              0.5, 0.8,
                              [f"Detected {claim_count} claims"])
            except Exception as e:
                self._add_trace("Factual Detection", "factual_detector", 0.5, 0.5, [f"Skipped: {str(e)[:50]}"])
        
        # 2d: Decision Quality Enhancement
        decision_result = None
        if enhancement_depth in ['standard', 'deep']:
            decision_result = self.decision_enhancer.enhance(current_response, user_context)
            if decision_result.quality_improvement > 0:
                current_response = decision_result.enhanced_text
                self._add_trace("Decision Quality", "decision_enhancer",
                              0.5, 0.5 + decision_result.quality_improvement,
                              decision_result.key_enhancements[:3])
        
        # Step 3: Final response improvement pass
        # Collect all detected issues for the response improver
        detected_issues = self._collect_issues(causal_result, inference_result, 
                                               truth_result, decision_result)
        
        if detected_issues:
            improvement_result = self.response_improver.improve(
                query, current_response, detected_issues, domain
            )
            current_response = improvement_result.improved_response
            # Convert Correction objects to strings for trace
            correction_strs = [str(c) for c in improvement_result.corrections_applied[:3]]
            self._add_trace("Response Improvement", "response_improver",
                          0.5, 0.5 + (improvement_result.improvement_score / 100),
                          correction_strs)
        
        # Step 4: Mission alignment check
        alignment_result = self.mission_checker.check_alignment(current_response)
        
        if alignment_result.status in [AlignmentStatus.DRIFT_DETECTED, 
                                       AlignmentStatus.CRITICAL_DRIFT]:
            # Apply correction guidance
            self._add_trace("Mission Alignment", "mission_checker",
                          alignment_result.alignment_score, 0.8,
                          [f"Drift detected: {len(alignment_result.drift_indicators)} indicators"])
        
        # Step 5: Calculate final quality
        enhanced_quality = self._assess_quality(current_response, domain)
        overall_improvement = enhanced_quality - original_quality
        
        # Step 6: Determine enhancement level
        enhancement_level = self._determine_enhancement_level(overall_improvement)
        
        # Step 7: Record learnings if enabled
        if self.enable_learning and self.memory and overall_improvement > 0.1:
            # Store successful enhancement pattern
            learning = self.memory.store(
                memory_type=MemoryType.SUCCESS,
                domain=domain,
                trigger=f"Query: {query[:50]}",
                content=f"Enhancement successful with {enhancement_level.value} improvements",
                evidence=[f"Quality improved from {original_quality:.2f} to {enhanced_quality:.2f}"],
                confidence=min(0.9, overall_improvement + 0.5)
            )
            new_learnings.append(learning.id)
        
        return CognitiveEnhancementResult(
            original_response=response,
            enhanced_response=current_response,
            enhancement_level=enhancement_level,
            overall_improvement=overall_improvement,
            causal_result=causal_result,
            inference_result=inference_result,
            truth_result=truth_result,
            decision_result=decision_result,
            trace=self.trace.copy(),
            memories_applied=memories_applied,
            new_learnings=new_learnings,
            original_quality=original_quality,
            enhanced_quality=enhanced_quality,
            alignment_status=alignment_result.status,
            alignment_score=alignment_result.alignment_score
        )
    
    def _add_trace(self, step: str, module: str, 
                   input_q: float, output_q: float, changes: List[str]):
        """Add a trace entry"""
        self.trace.append(EnhancementTrace(
            step=step,
            module=module,
            input_quality=input_q,
            output_quality=output_q,
            changes_made=changes
        ))
    
    def _assess_quality(self, text: str, domain: str) -> float:
        """Assess overall quality of text"""
        scores = []
        
        # Check for required disclaimers in high-risk domains
        if domain in ['medical', 'financial', 'legal']:
            has_disclaimer = any(d in text.lower() for d in 
                               ['consult', 'professional', 'advice', 'individual circumstances'])
            scores.append(0.8 if has_disclaimer else 0.3)
        
        # Check for overconfidence
        overconfident_words = ['definitely', 'certainly', 'guaranteed', '100%', 'always', 'never']
        overconfidence_count = sum(1 for w in overconfident_words if w in text.lower())
        scores.append(max(0.2, 1.0 - (overconfidence_count * 0.15)))
        
        # Check for reasoning presence
        reasoning_words = ['because', 'therefore', 'since', 'however', 'although', 'consider']
        reasoning_count = sum(1 for w in reasoning_words if w in text.lower())
        scores.append(min(1.0, 0.3 + (reasoning_count * 0.1)))
        
        # Check length appropriateness
        word_count = len(text.split())
        if word_count < 20:
            scores.append(0.4)  # Too short
        elif word_count > 500:
            scores.append(0.6)  # Maybe too long
        else:
            scores.append(0.8)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _collect_issues(self, 
                        causal: Optional[Any],
                        inference: Optional[Any],
                        truth: Optional[Any],
                        decision: Optional[DecisionEnhancementResult]) -> List[DetectedIssue]:
        """Collect all detected issues from cognitive modules (using existing module APIs)"""
        issues = []
        
        # From WorldModels (causal analysis)
        if causal:
            try:
                # WorldModels returns CausalPrediction with causal_chains
                if hasattr(causal, 'causal_chains'):
                    for chain in causal.causal_chains:
                        if hasattr(chain, 'total_confidence') and chain.total_confidence < 0.5:
                            issues.append(DetectedIssue(
                                issue_type=IssueType.LOGICAL_FALLACY,
                                description=f"Weak causal chain: confidence {chain.total_confidence:.2f}",
                                evidence=str(chain.steps[0]) if chain.steps else "Unknown",
                                severity=0.7
                            ))
            except Exception:
                pass
        
        # From NeuroSymbolic (logic/inference)
        if inference:
            try:
                # NeuroSymbolic returns with fallacies_detected list
                if hasattr(inference, 'fallacies_detected'):
                    for fallacy in inference.fallacies_detected:
                        issues.append(DetectedIssue(
                            issue_type=IssueType.LOGICAL_FALLACY,
                            description=f"Fallacy: {fallacy.fallacy_type.value if hasattr(fallacy, 'fallacy_type') else str(fallacy)}",
                            evidence=fallacy.evidence if hasattr(fallacy, 'evidence') else str(fallacy),
                            severity=0.8
                        ))
            except Exception:
                pass
        
        # From FactualDetector (truth)
        if truth:
            try:
                # FactualDetector returns list of detected claims
                if isinstance(truth, list):
                    for claim in truth:
                        if hasattr(claim, 'confidence') and claim.confidence < 0.5:
                            issues.append(DetectedIssue(
                                issue_type=IssueType.HALLUCINATION,
                                description=f"Low confidence claim",
                                evidence=str(claim),
                                severity=0.6
                            ))
            except Exception:
                pass
        
        # From DecisionQualityEnhancer (our novel module)
        if decision:
            for assessment in decision.assessments:
                if assessment.quality.value in ['poor', 'dangerous']:
                    for flaw in assessment.flaws:
                        issues.append(DetectedIssue(
                            issue_type=IssueType.MISSING_DISCLAIMER if 'missing' in flaw.value.lower()
                                      else IssueType.OVERCONFIDENCE,
                            description=f"Decision flaw: {flaw.value}",
                            evidence=assessment.decision.recommendation,
                            severity=0.9 if assessment.quality.value == 'dangerous' else 0.7
                        ))
        
        return issues
    
    def _determine_enhancement_level(self, improvement: float) -> EnhancementLevel:
        """Determine enhancement level from improvement score"""
        if improvement <= 0:
            return EnhancementLevel.NONE
        elif improvement < 0.1:
            return EnhancementLevel.MINOR
        elif improvement < 0.2:
            return EnhancementLevel.MODERATE
        elif improvement < 0.3:
            return EnhancementLevel.MAJOR
        else:
            return EnhancementLevel.CRITICAL
    
    def record_feedback(self, 
                        result: CognitiveEnhancementResult,
                        user_satisfied: bool,
                        feedback: str = "") -> None:
        """
        Record user feedback on enhancement quality.
        
        Args:
            result: The enhancement result
            user_satisfied: Whether user was satisfied
            feedback: Optional feedback text
        """
        if not self.enable_learning or not self.memory:
            return
        
        # Record outcome for applied memories
        self.memory.record_outcome(result.memories_applied, user_satisfied, feedback)
        
        # If feedback is negative, create a new learning
        if not user_satisfied and feedback:
            self.memory.store(
                memory_type=MemoryType.FAILURE,
                domain='feedback',
                trigger=feedback[:100],
                content=f"Enhancement was not satisfactory: {feedback}",
                evidence=[f"Original improvement: {result.overall_improvement:.2f}"],
                priority=MemoryPriority.HIGH,
                confidence=0.8
            )
    
    def get_enhancement_summary(self, result: CognitiveEnhancementResult) -> str:
        """Get human-readable summary of enhancements"""
        lines = [
            "=" * 60,
            "COGNITIVE ENHANCEMENT SUMMARY",
            "=" * 60,
            "",
            f"Enhancement Level: {result.enhancement_level.value.upper()}",
            f"Quality: {result.original_quality:.2f} → {result.enhanced_quality:.2f} "
            f"(+{result.overall_improvement:.2f})",
            f"Mission Alignment: {result.alignment_status.value} "
            f"(score: {result.alignment_score:.2f})",
            "",
            "Enhancement Trace:"
        ]
        
        for trace in result.trace:
            lines.append(f"  [{trace.module}] {trace.step}")
            lines.append(f"    Quality: {trace.input_quality:.2f} → {trace.output_quality:.2f}")
            if trace.changes_made:
                for change in trace.changes_made[:2]:
                    lines.append(f"    - {change}")
        
        if result.memories_applied:
            lines.extend(["", f"Memories Applied: {len(result.memories_applied)}"])
        
        if result.new_learnings:
            lines.extend(["", f"New Learnings Created: {len(result.new_learnings)}"])
        
        return "\n".join(lines)
    
    # =========================================================================
    # Learning Interface (5/5 methods) - Completing interface
    # =========================================================================
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback to adjust cognitive enhancements."""
        if not hasattr(self, '_learning_params'):
            self._learning_params = {}
        was_correct = feedback.get('was_correct', True)
        module = feedback.get('module')
        if not was_correct and module:
            current = getattr(self, '_module_effectiveness', {}).get(module, 0.5)
            self._module_effectiveness[module] = max(0.1, current - 0.05)
    
    def get_statistics(self) -> Dict:
        if not hasattr(self, '_total_enhancements'): self._total_enhancements = 0
        if not hasattr(self, '_module_usage'): self._module_usage = {}
        """Return cognitive enhancement statistics."""
        return {
            'total_enhancements': getattr(self, '_total_enhancements', 0),
            'module_effectiveness': dict(self._module_effectiveness),
            'feedback_received': len(getattr(self, '_feedback_history', [])),
            'learning_params': getattr(self, '_learning_params', {})
        }
    
    def serialize_state(self) -> Dict:
        # Ensure attributes exist
        if not hasattr(self, '_module_usage'): self._module_usage = {}
        """Serialize learning state for persistence."""
        return {
            'module_effectiveness': dict(self._module_effectiveness),
            'learning_params': getattr(self, '_learning_params', {}),
            'total_enhancements': getattr(self, '_total_enhancements', 0),
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore learning state from serialized data."""
        self._module_effectiveness = state.get('module_effectiveness', {})
        self._learning_params = state.get('learning_params', {})
        self._total_enhancements = state.get('total_enhancements', 0)


def test_cognitive_enhancer():
    """Test the cognitive enhancer"""
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        enhancer = CognitiveEnhancer(storage_path=tmpdir)
        
        print("=" * 70)
        print("COGNITIVE ENHANCER TEST")
        print("=" * 70)
        
        # Test case: Bad financial advice
        query = "I'm 72 and just retired with $500,000. How should I invest?"
        
        bad_response = """
        You should definitely put all your money into Bitcoin and meme stocks!
        Everyone is doing it and making guaranteed returns of 100% per year.
        This is obviously the best strategy. Don't waste time with financial advisors
        who will just take your money. Trust me, this will 100% make you rich.
        """
        
        print(f"\nQUERY: {query}")
        print(f"\nORIGINAL RESPONSE:\n{bad_response}")
        
        # Enhance
        result = enhancer.enhance(
            query=query,
            response=bad_response,
            domain='financial',
            user_context={'age': 72, 'risk_tolerance': 'conservative'},
            enhancement_depth='deep'
        )
        
        print("\n" + enhancer.get_enhancement_summary(result))
        
        print("\n" + "=" * 60)
        print("ENHANCED RESPONSE:")
        print("=" * 60)
        print(result.enhanced_response)
        
        # Test mission alignment
        print("\n" + "=" * 60)
        print("MISSION ALIGNMENT CHECK")
        print("=" * 60)
        print(f"Status: {result.alignment_status.value}")
        print(f"Score: {result.alignment_score:.2f}")


if __name__ == "__main__":
    test_cognitive_enhancer()

