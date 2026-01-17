"""
BAIS R&D Modules - Clinical Test Suite
Rigorous, objective testing of advanced cognitive modules

This suite tests:
1. Each module independently with challenging cases
2. Modules in conjunction (cross-validation)
3. Edge cases and failure modes
4. Ground truth comparison where possible

Factual clinical assessment.
"""

import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# Import R&D modules
from research.theory_of_mind import TheoryOfMindModule, ManipulationRisk
from research.neurosymbolic import NeuroSymbolicModule, VerificationResult
from research.world_models import WorldModelsModule
from research.creative_reasoning import CreativeReasoningModule, OriginalityLevel


class TestResult(str, Enum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"
    ERROR = "ERROR"


@dataclass
class ClinicalTestCase:
    """A clinical test case with ground truth."""
    test_id: str
    module: str
    category: str
    description: str
    query: str
    response: str
    ground_truth: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestOutcome:
    """Outcome of a single test."""
    test_id: str
    result: TestResult
    expected: Dict[str, Any]
    actual: Dict[str, Any]
    matches: List[str]
    mismatches: List[str]
    processing_time_ms: float
    notes: str = ""


@dataclass
class ModuleAssessment:
    """Overall assessment of a module."""
    module_name: str
    tests_run: int
    passed: int
    partial: int
    failed: int
    errors: int
    pass_rate: float
    avg_processing_time_ms: float
    strengths: List[str]
    weaknesses: List[str]
    verdict: str


# =============================================================================
# CLINICAL TEST CASES
# =============================================================================

THEORY_OF_MIND_TESTS = [
    ClinicalTestCase(
        test_id="TOM-001",
        module="theory_of_mind",
        category="mental_state_inference",
        description="Clear mental state expression",
        query="How does John feel about the project?",
        response="John believes the project will succeed. He wants to lead the team and feels excited about the opportunity. He expects the client to approve by Friday.",
        ground_truth={
            "min_mental_states": 3,
            "state_types_expected": ["belief", "desire", "emotion", "expectation"],
            "agent_expected": "john"
        }
    ),
    ClinicalTestCase(
        test_id="TOM-002",
        module="theory_of_mind",
        category="mental_state_inference",
        description="No mental states (factual only)",
        query="What is the weather?",
        response="The temperature is 72 degrees. It is sunny with no clouds. The humidity is 45%.",
        ground_truth={
            "max_mental_states": 1,  # Should find very few/none
            "tom_score_max": 0.3
        }
    ),
    ClinicalTestCase(
        test_id="TOM-003",
        module="theory_of_mind",
        category="manipulation_detection",
        description="Clear manipulation attempt",
        query="Should I buy this product?",
        response="Act NOW! This exclusive offer expires in 24 hours. Thousands of satisfied customers agree this is the best. Our experts confirm it's the only solution. You must decide immediately or miss out forever!",
        ground_truth={
            "manipulation_risk_min": "high",
            "techniques_expected": ["scarcity", "social_proof", "appeal_to_authority"],
            "min_techniques": 2
        }
    ),
    ClinicalTestCase(
        test_id="TOM-004",
        module="theory_of_mind",
        category="manipulation_detection",
        description="Neutral informational text (no manipulation)",
        query="What is Python?",
        response="Python is a programming language created by Guido van Rossum in 1991. It emphasizes code readability and supports multiple programming paradigms.",
        ground_truth={
            "manipulation_risk_max": "low",
            "max_techniques": 1
        }
    ),
    ClinicalTestCase(
        test_id="TOM-005",
        module="theory_of_mind",
        category="perspective_taking",
        description="Multiple perspectives",
        response="From the manager's perspective, the deadline is critical. The team members feel overwhelmed. I think we need more resources.",
        query="What's the project status?",
        ground_truth={
            "min_perspectives": 2,
            "perspective_types_expected": ["first_person", "third_person"]
        }
    ),
]

NEUROSYMBOLIC_TESTS = [
    ClinicalTestCase(
        test_id="NS-001",
        module="neurosymbolic",
        category="logical_validity",
        description="Valid syllogism (modus ponens)",
        query="Is this argument valid?",
        response="If it rains, the ground gets wet. It is raining. Therefore, the ground is wet.",
        ground_truth={
            "expected_valid": True,
            "min_statements": 2,
            "consistency": True,
            "fallacies_max": 0
        }
    ),
    ClinicalTestCase(
        test_id="NS-002",
        module="neurosymbolic",
        category="logical_validity",
        description="Invalid logic (affirming consequent)",
        query="Check this logic",
        response="If it rains, the ground gets wet. The ground is wet. Therefore, it rained.",
        ground_truth={
            "expected_valid": False,
            "fallacy_expected": "affirming_consequent"
        }
    ),
    ClinicalTestCase(
        test_id="NS-003",
        module="neurosymbolic",
        category="contradiction_detection",
        description="Clear contradiction",
        query="Is this consistent?",
        response="All birds can fly. Penguins are birds. Penguins cannot fly.",
        ground_truth={
            "consistency": False,
            "min_contradictions": 1
        }
    ),
    ClinicalTestCase(
        test_id="NS-004",
        module="neurosymbolic",
        category="fallacy_detection",
        description="Multiple fallacies",
        query="Evaluate this argument",
        response="You must either support this policy or you hate freedom. Everyone knows this is true. You can't trust John's opinion because he's not an expert.",
        ground_truth={
            "min_fallacies": 2,
            "fallacies_expected": ["false_dichotomy", "ad_hominem"]
        }
    ),
    ClinicalTestCase(
        test_id="NS-005",
        module="neurosymbolic",
        category="logical_validity",
        description="Non-logical text (no statements to verify)",
        query="What do you think?",
        response="Hello! How are you today? Nice weather we're having.",
        ground_truth={
            "verification_result": "incomplete",
            "max_statements": 2
        }
    ),
]

WORLD_MODELS_TESTS = [
    ClinicalTestCase(
        test_id="WM-001",
        module="world_models",
        category="causal_extraction",
        description="Clear causal chain",
        query="What causes inflation?",
        response="Excessive money supply causes inflation. Inflation leads to higher prices. Higher prices result in reduced purchasing power.",
        ground_truth={
            "min_causal_relationships": 2,
            "min_causal_chains": 1,
            "causal_types_expected": ["causes"]
        }
    ),
    ClinicalTestCase(
        test_id="WM-002",
        module="world_models",
        category="causal_extraction",
        description="No causal content",
        query="Describe a sunset",
        response="The sunset was beautiful. Orange and pink colors filled the sky. The sun slowly disappeared below the horizon.",
        ground_truth={
            "max_causal_relationships": 1,
            "model_completeness_max": 0.3
        }
    ),
    ClinicalTestCase(
        test_id="WM-003",
        module="world_models",
        category="prediction_extraction",
        description="Clear predictions",
        query="What will happen to the stock market?",
        response="Interest rates will likely increase. This will lead to reduced borrowing. The stock market may decline as a result. Economic growth is expected to slow.",
        ground_truth={
            "min_predictions": 1,
            "prediction_confidence_type": ["medium", "low", "speculative"]
        }
    ),
    ClinicalTestCase(
        test_id="WM-004",
        module="world_models",
        category="counterfactual",
        description="Counterfactual scenario",
        query="What if we had more resources?",
        response="If we had more resources, we would have finished earlier. Had the budget been larger, we could have hired more staff. What if we started sooner? We might have avoided the delay.",
        ground_truth={
            "min_counterfactuals": 1
        }
    ),
    ClinicalTestCase(
        test_id="WM-005",
        module="world_models",
        category="causal_extraction",
        description="Correlation vs causation",
        query="What's the relationship?",
        response="Ice cream sales are associated with drowning rates. Higher ice cream sales correlate with more drownings. This is linked to summer weather.",
        ground_truth={
            "causal_type_expected": "correlates",
            "should_not_confuse_with_causes": True
        }
    ),
]

CREATIVE_REASONING_TESTS = [
    ClinicalTestCase(
        test_id="CR-001",
        module="creative_reasoning",
        category="divergent_thinking",
        description="Multiple creative ideas",
        query="Give me marketing ideas",
        response="""Here are innovative approaches:
1. Use influencer partnerships to reach new audiences
2. Create viral social media challenges 
3. Develop interactive AR experiences
4. Launch a referral rewards program
5. Partner with complementary brands
6. Create educational content series
7. Implement gamification elements""",
        ground_truth={
            "min_ideas": 5,
            "fluency_min": 0.5,
            "flexibility_min": 0.3
        }
    ),
    ClinicalTestCase(
        test_id="CR-002",
        module="creative_reasoning",
        category="divergent_thinking",
        description="Single conventional answer",
        query="How do I sort a list?",
        response="You can use the sort() method to sort a list in Python.",
        ground_truth={
            "max_ideas": 2,
            "originality_max": 0.5,
            "creativity_percentile_max": 50
        }
    ),
    ClinicalTestCase(
        test_id="CR-003",
        module="creative_reasoning",
        category="analogy_detection",
        description="Rich analogies",
        query="Explain neural networks",
        response="Neural networks are like the human brain, with neurons connected like a web. Think of it as a city's road network where information flows like traffic. Learning is similar to water finding its path downhill.",
        ground_truth={
            "min_analogies": 2,
            "analogy_quality_min": 0.4
        }
    ),
    ClinicalTestCase(
        test_id="CR-004",
        module="creative_reasoning",
        category="cliche_detection",
        description="Cliche-heavy text",
        query="Give advice for success",
        response="Think outside the box and take it to the next level. At the end of the day, it is what it is. Focus on the low-hanging fruit and create a win-win situation. This will be a game-changer and create synergy.",
        ground_truth={
            "originality_level": "cliche",
            "originality_score_max": 0.3,
            "should_detect_cliches": True
        }
    ),
    ClinicalTestCase(
        test_id="CR-005",
        module="creative_reasoning",
        category="originality",
        description="Highly original content",
        query="Design a unique product",
        response="Imagine a biodegradable phone case that grows into a plant when buried. It contains seeds embedded in organic material. As the case decomposes, it fertilizes the soil and sprouts into flowers - turning electronic waste into garden beauty.",
        ground_truth={
            "originality_level": "moderately_original",
            "originality_score_min": 0.5,
            "surprise_score_min": 0.4
        }
    ),
]

# Combined/Integration tests
INTEGRATION_TESTS = [
    ClinicalTestCase(
        test_id="INT-001",
        module="integration",
        category="tom_neurosymbolic",
        description="Social manipulation with logical fallacies",
        query="Should I invest?",
        response="Everyone believes this is the best investment. If you don't invest now, you'll regret it forever. Smart people invest here, so you should too.",
        ground_truth={
            "tom_manipulation_detected": True,
            "ns_fallacies_detected": True,
            "cross_validation": "both_flag_issues"
        }
    ),
    ClinicalTestCase(
        test_id="INT-002",
        module="integration",
        category="world_creative",
        description="Creative predictions",
        query="Future of transportation",
        response="Imagine flying cars like birds in the sky, leading to reduced traffic. This could cause real estate in suburbs to increase. What if everyone commuted by drone? Cities would transform like caterpillars into butterflies.",
        ground_truth={
            "wm_predictions_found": True,
            "cr_analogies_found": True,
            "cross_validation": "complementary_insights"
        }
    ),
]

ALL_TESTS = (
    THEORY_OF_MIND_TESTS + 
    NEUROSYMBOLIC_TESTS + 
    WORLD_MODELS_TESTS + 
    CREATIVE_REASONING_TESTS +
    INTEGRATION_TESTS
)


# =============================================================================
# TEST EXECUTION ENGINE
# =============================================================================

class ClinicalTestRunner:
    """Executes clinical tests and produces factual assessments."""
    
    def __init__(self):
        self.tom = TheoryOfMindModule()
        self.ns = NeuroSymbolicModule()
        self.wm = WorldModelsModule()
        self.cr = CreativeReasoningModule()
        
        self.outcomes: List[TestOutcome] = []
    
    def run_all_tests(self) -> Dict[str, ModuleAssessment]:
        """Run all clinical tests and return assessments."""
        print("\n" + "="*70)
        print("BAIS R&D CLINICAL TEST SUITE")
        print("="*70)
        print("Objective, non-optimistic assessment of R&D modules")
        print()
        
        # Run tests by module
        tom_outcomes = self._run_tom_tests()
        ns_outcomes = self._run_ns_tests()
        wm_outcomes = self._run_wm_tests()
        cr_outcomes = self._run_cr_tests()
        int_outcomes = self._run_integration_tests()
        
        # Compile assessments
        assessments = {
            'theory_of_mind': self._assess_module('Theory of Mind', tom_outcomes),
            'neurosymbolic': self._assess_module('Neuro-Symbolic', ns_outcomes),
            'world_models': self._assess_module('World Models', wm_outcomes),
            'creative_reasoning': self._assess_module('Creative Reasoning', cr_outcomes),
            'integration': self._assess_module('Integration', int_outcomes),
        }
        
        return assessments
    
    def _run_tom_tests(self) -> List[TestOutcome]:
        """Run Theory of Mind tests."""
        print("\n" + "-"*70)
        print("THEORY OF MIND TESTS")
        print("-"*70)
        
        outcomes = []
        for test in THEORY_OF_MIND_TESTS:
            outcome = self._execute_tom_test(test)
            outcomes.append(outcome)
            self._print_outcome(outcome)
        
        return outcomes
    
    def _execute_tom_test(self, test: ClinicalTestCase) -> TestOutcome:
        """Execute a single ToM test."""
        start = time.time()
        
        try:
            result = self.tom.analyze(test.query, test.response, test.context)
            processing_time = (time.time() - start) * 1000
            
            matches = []
            mismatches = []
            gt = test.ground_truth
            
            # Check mental states count
            if "min_mental_states" in gt:
                if len(result.inferred_states) >= gt["min_mental_states"]:
                    matches.append(f"Mental states: {len(result.inferred_states)} >= {gt['min_mental_states']}")
                else:
                    mismatches.append(f"Mental states: {len(result.inferred_states)} < {gt['min_mental_states']}")
            
            if "max_mental_states" in gt:
                if len(result.inferred_states) <= gt["max_mental_states"]:
                    matches.append(f"Mental states: {len(result.inferred_states)} <= {gt['max_mental_states']}")
                else:
                    mismatches.append(f"Mental states: {len(result.inferred_states)} > {gt['max_mental_states']}")
            
            # Check ToM score
            if "tom_score_max" in gt:
                if result.theory_of_mind_score <= gt["tom_score_max"]:
                    matches.append(f"ToM score: {result.theory_of_mind_score:.2f} <= {gt['tom_score_max']}")
                else:
                    mismatches.append(f"ToM score: {result.theory_of_mind_score:.2f} > {gt['tom_score_max']}")
            
            # Check manipulation risk
            if "manipulation_risk_min" in gt:
                risk_order = ["none", "low", "medium", "high", "critical"]
                actual_idx = risk_order.index(result.manipulation.risk_level.value)
                expected_idx = risk_order.index(gt["manipulation_risk_min"])
                if actual_idx >= expected_idx:
                    matches.append(f"Manipulation risk: {result.manipulation.risk_level.value} >= {gt['manipulation_risk_min']}")
                else:
                    mismatches.append(f"Manipulation risk: {result.manipulation.risk_level.value} < {gt['manipulation_risk_min']}")
            
            if "manipulation_risk_max" in gt:
                risk_order = ["none", "low", "medium", "high", "critical"]
                actual_idx = risk_order.index(result.manipulation.risk_level.value)
                expected_idx = risk_order.index(gt["manipulation_risk_max"])
                if actual_idx <= expected_idx:
                    matches.append(f"Manipulation risk: {result.manipulation.risk_level.value} <= {gt['manipulation_risk_max']}")
                else:
                    mismatches.append(f"Manipulation risk: {result.manipulation.risk_level.value} > {gt['manipulation_risk_max']}")
            
            # Check techniques
            if "min_techniques" in gt:
                if len(result.manipulation.techniques_detected) >= gt["min_techniques"]:
                    matches.append(f"Techniques: {len(result.manipulation.techniques_detected)} >= {gt['min_techniques']}")
                else:
                    mismatches.append(f"Techniques: {len(result.manipulation.techniques_detected)} < {gt['min_techniques']}")
            
            if "max_techniques" in gt:
                if len(result.manipulation.techniques_detected) <= gt["max_techniques"]:
                    matches.append(f"Techniques: {len(result.manipulation.techniques_detected)} <= {gt['max_techniques']}")
                else:
                    mismatches.append(f"Techniques: {len(result.manipulation.techniques_detected)} > {gt['max_techniques']}")
            
            # Check perspectives
            if "min_perspectives" in gt:
                if len(result.perspectives) >= gt["min_perspectives"]:
                    matches.append(f"Perspectives: {len(result.perspectives)} >= {gt['min_perspectives']}")
                else:
                    mismatches.append(f"Perspectives: {len(result.perspectives)} < {gt['min_perspectives']}")
            
            # Determine result
            if not mismatches:
                test_result = TestResult.PASS
            elif len(matches) > len(mismatches):
                test_result = TestResult.PARTIAL
            else:
                test_result = TestResult.FAIL
            
            return TestOutcome(
                test_id=test.test_id,
                result=test_result,
                expected=gt,
                actual={
                    "mental_states": len(result.inferred_states),
                    "tom_score": result.theory_of_mind_score,
                    "manipulation_risk": result.manipulation.risk_level.value,
                    "techniques": result.manipulation.techniques_detected,
                    "perspectives": len(result.perspectives),
                },
                matches=matches,
                mismatches=mismatches,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            return TestOutcome(
                test_id=test.test_id,
                result=TestResult.ERROR,
                expected=test.ground_truth,
                actual={"error": str(e)},
                matches=[],
                mismatches=[f"Exception: {str(e)}"],
                processing_time_ms=0,
                notes=str(e)
            )
    
    def _run_ns_tests(self) -> List[TestOutcome]:
        """Run Neuro-Symbolic tests."""
        print("\n" + "-"*70)
        print("NEURO-SYMBOLIC TESTS")
        print("-"*70)
        
        outcomes = []
        for test in NEUROSYMBOLIC_TESTS:
            outcome = self._execute_ns_test(test)
            outcomes.append(outcome)
            self._print_outcome(outcome)
        
        return outcomes
    
    def _execute_ns_test(self, test: ClinicalTestCase) -> TestOutcome:
        """Execute a single NS test."""
        start = time.time()
        
        try:
            result = self.ns.verify(test.query, test.response)
            processing_time = (time.time() - start) * 1000
            
            matches = []
            mismatches = []
            gt = test.ground_truth
            
            # Check validity
            if "expected_valid" in gt:
                is_valid = result.verification_result == VerificationResult.VALID
                if is_valid == gt["expected_valid"]:
                    matches.append(f"Validity: {is_valid} == {gt['expected_valid']}")
                else:
                    mismatches.append(f"Validity: {is_valid} != {gt['expected_valid']}")
            
            # Check consistency
            if "consistency" in gt:
                if result.consistency.is_consistent == gt["consistency"]:
                    matches.append(f"Consistency: {result.consistency.is_consistent}")
                else:
                    mismatches.append(f"Consistency: {result.consistency.is_consistent} != {gt['consistency']}")
            
            # Check statements
            if "min_statements" in gt:
                if len(result.statements) >= gt["min_statements"]:
                    matches.append(f"Statements: {len(result.statements)} >= {gt['min_statements']}")
                else:
                    mismatches.append(f"Statements: {len(result.statements)} < {gt['min_statements']}")
            
            if "max_statements" in gt:
                if len(result.statements) <= gt["max_statements"]:
                    matches.append(f"Statements: {len(result.statements)} <= {gt['max_statements']}")
                else:
                    mismatches.append(f"Statements: {len(result.statements)} > {gt['max_statements']}")
            
            # Check fallacies
            if "fallacies_max" in gt:
                if len(result.fallacies_detected) <= gt["fallacies_max"]:
                    matches.append(f"Fallacies: {len(result.fallacies_detected)} <= {gt['fallacies_max']}")
                else:
                    mismatches.append(f"Fallacies: {len(result.fallacies_detected)} > {gt['fallacies_max']}")
            
            if "min_fallacies" in gt:
                if len(result.fallacies_detected) >= gt["min_fallacies"]:
                    matches.append(f"Fallacies: {len(result.fallacies_detected)} >= {gt['min_fallacies']}")
                else:
                    mismatches.append(f"Fallacies: {len(result.fallacies_detected)} < {gt['min_fallacies']}")
            
            if "min_contradictions" in gt:
                if len(result.consistency.contradictions) >= gt["min_contradictions"]:
                    matches.append(f"Contradictions: {len(result.consistency.contradictions)} >= {gt['min_contradictions']}")
                else:
                    mismatches.append(f"Contradictions: {len(result.consistency.contradictions)} < {gt['min_contradictions']}")
            
            # Check verification result
            if "verification_result" in gt:
                if result.verification_result.value == gt["verification_result"]:
                    matches.append(f"Result: {result.verification_result.value}")
                else:
                    mismatches.append(f"Result: {result.verification_result.value} != {gt['verification_result']}")
            
            # Determine result
            if not mismatches:
                test_result = TestResult.PASS
            elif len(matches) > len(mismatches):
                test_result = TestResult.PARTIAL
            else:
                test_result = TestResult.FAIL
            
            return TestOutcome(
                test_id=test.test_id,
                result=test_result,
                expected=gt,
                actual={
                    "verification_result": result.verification_result.value,
                    "validity_score": result.validity_score,
                    "statements": len(result.statements),
                    "consistent": result.consistency.is_consistent,
                    "contradictions": len(result.consistency.contradictions),
                    "fallacies": [f.fallacy_type.value for f in result.fallacies_detected],
                },
                matches=matches,
                mismatches=mismatches,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            return TestOutcome(
                test_id=test.test_id,
                result=TestResult.ERROR,
                expected=test.ground_truth,
                actual={"error": str(e)},
                matches=[],
                mismatches=[f"Exception: {str(e)}"],
                processing_time_ms=0,
                notes=str(e)
            )
    
    def _run_wm_tests(self) -> List[TestOutcome]:
        """Run World Models tests."""
        print("\n" + "-"*70)
        print("WORLD MODELS TESTS")
        print("-"*70)
        
        outcomes = []
        for test in WORLD_MODELS_TESTS:
            outcome = self._execute_wm_test(test)
            outcomes.append(outcome)
            self._print_outcome(outcome)
        
        return outcomes
    
    def _execute_wm_test(self, test: ClinicalTestCase) -> TestOutcome:
        """Execute a single WM test."""
        start = time.time()
        
        try:
            result = self.wm.analyze(test.query, test.response)
            processing_time = (time.time() - start) * 1000
            
            matches = []
            mismatches = []
            gt = test.ground_truth
            
            # Check causal relationships
            if "min_causal_relationships" in gt:
                if len(result.causal_relationships) >= gt["min_causal_relationships"]:
                    matches.append(f"Causal rels: {len(result.causal_relationships)} >= {gt['min_causal_relationships']}")
                else:
                    mismatches.append(f"Causal rels: {len(result.causal_relationships)} < {gt['min_causal_relationships']}")
            
            if "max_causal_relationships" in gt:
                if len(result.causal_relationships) <= gt["max_causal_relationships"]:
                    matches.append(f"Causal rels: {len(result.causal_relationships)} <= {gt['max_causal_relationships']}")
                else:
                    mismatches.append(f"Causal rels: {len(result.causal_relationships)} > {gt['max_causal_relationships']}")
            
            # Check chains
            if "min_causal_chains" in gt:
                if len(result.causal_chains) >= gt["min_causal_chains"]:
                    matches.append(f"Chains: {len(result.causal_chains)} >= {gt['min_causal_chains']}")
                else:
                    mismatches.append(f"Chains: {len(result.causal_chains)} < {gt['min_causal_chains']}")
            
            # Check predictions
            if "min_predictions" in gt:
                if len(result.predictions) >= gt["min_predictions"]:
                    matches.append(f"Predictions: {len(result.predictions)} >= {gt['min_predictions']}")
                else:
                    mismatches.append(f"Predictions: {len(result.predictions)} < {gt['min_predictions']}")
            
            # Check counterfactuals
            if "min_counterfactuals" in gt:
                if len(result.counterfactuals) >= gt["min_counterfactuals"]:
                    matches.append(f"Counterfactuals: {len(result.counterfactuals)} >= {gt['min_counterfactuals']}")
                else:
                    mismatches.append(f"Counterfactuals: {len(result.counterfactuals)} < {gt['min_counterfactuals']}")
            
            # Check model completeness
            if "model_completeness_max" in gt:
                if result.model_completeness <= gt["model_completeness_max"]:
                    matches.append(f"Completeness: {result.model_completeness:.2f} <= {gt['model_completeness_max']}")
                else:
                    mismatches.append(f"Completeness: {result.model_completeness:.2f} > {gt['model_completeness_max']}")
            
            # Determine result
            if not mismatches:
                test_result = TestResult.PASS
            elif len(matches) > len(mismatches):
                test_result = TestResult.PARTIAL
            else:
                test_result = TestResult.FAIL
            
            return TestOutcome(
                test_id=test.test_id,
                result=test_result,
                expected=gt,
                actual={
                    "causal_relationships": len(result.causal_relationships),
                    "causal_chains": len(result.causal_chains),
                    "predictions": len(result.predictions),
                    "counterfactuals": len(result.counterfactuals),
                    "model_completeness": result.model_completeness,
                },
                matches=matches,
                mismatches=mismatches,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            return TestOutcome(
                test_id=test.test_id,
                result=TestResult.ERROR,
                expected=test.ground_truth,
                actual={"error": str(e)},
                matches=[],
                mismatches=[f"Exception: {str(e)}"],
                processing_time_ms=0,
                notes=str(e)
            )
    
    def _run_cr_tests(self) -> List[TestOutcome]:
        """Run Creative Reasoning tests."""
        print("\n" + "-"*70)
        print("CREATIVE REASONING TESTS")
        print("-"*70)
        
        outcomes = []
        for test in CREATIVE_REASONING_TESTS:
            outcome = self._execute_cr_test(test)
            outcomes.append(outcome)
            self._print_outcome(outcome)
        
        return outcomes
    
    def _execute_cr_test(self, test: ClinicalTestCase) -> TestOutcome:
        """Execute a single CR test."""
        start = time.time()
        
        try:
            result = self.cr.analyze(test.query, test.response)
            processing_time = (time.time() - start) * 1000
            
            matches = []
            mismatches = []
            gt = test.ground_truth
            
            # Check ideas
            if "min_ideas" in gt:
                if result.idea_count >= gt["min_ideas"]:
                    matches.append(f"Ideas: {result.idea_count} >= {gt['min_ideas']}")
                else:
                    mismatches.append(f"Ideas: {result.idea_count} < {gt['min_ideas']}")
            
            if "max_ideas" in gt:
                if result.idea_count <= gt["max_ideas"]:
                    matches.append(f"Ideas: {result.idea_count} <= {gt['max_ideas']}")
                else:
                    mismatches.append(f"Ideas: {result.idea_count} > {gt['max_ideas']}")
            
            # Check fluency
            if "fluency_min" in gt:
                if result.creativity_metrics.fluency_score >= gt["fluency_min"]:
                    matches.append(f"Fluency: {result.creativity_metrics.fluency_score:.2f} >= {gt['fluency_min']}")
                else:
                    mismatches.append(f"Fluency: {result.creativity_metrics.fluency_score:.2f} < {gt['fluency_min']}")
            
            # Check flexibility
            if "flexibility_min" in gt:
                if result.creativity_metrics.flexibility_score >= gt["flexibility_min"]:
                    matches.append(f"Flexibility: {result.creativity_metrics.flexibility_score:.2f} >= {gt['flexibility_min']}")
                else:
                    mismatches.append(f"Flexibility: {result.creativity_metrics.flexibility_score:.2f} < {gt['flexibility_min']}")
            
            # Check analogies
            if "min_analogies" in gt:
                if len(result.analogies) >= gt["min_analogies"]:
                    matches.append(f"Analogies: {len(result.analogies)} >= {gt['min_analogies']}")
                else:
                    mismatches.append(f"Analogies: {len(result.analogies)} < {gt['min_analogies']}")
            
            # Check originality
            if "originality_score_max" in gt:
                if result.creativity_metrics.originality_score <= gt["originality_score_max"]:
                    matches.append(f"Originality: {result.creativity_metrics.originality_score:.2f} <= {gt['originality_score_max']}")
                else:
                    mismatches.append(f"Originality: {result.creativity_metrics.originality_score:.2f} > {gt['originality_score_max']}")
            
            if "originality_score_min" in gt:
                if result.creativity_metrics.originality_score >= gt["originality_score_min"]:
                    matches.append(f"Originality: {result.creativity_metrics.originality_score:.2f} >= {gt['originality_score_min']}")
                else:
                    mismatches.append(f"Originality: {result.creativity_metrics.originality_score:.2f} < {gt['originality_score_min']}")
            
            # Check percentile
            if "creativity_percentile_max" in gt:
                if result.creativity_metrics.creativity_percentile <= gt["creativity_percentile_max"]:
                    matches.append(f"Percentile: {result.creativity_metrics.creativity_percentile} <= {gt['creativity_percentile_max']}")
                else:
                    mismatches.append(f"Percentile: {result.creativity_metrics.creativity_percentile} > {gt['creativity_percentile_max']}")
            
            # Check surprise
            if "surprise_score_min" in gt:
                if result.surprise_score >= gt["surprise_score_min"]:
                    matches.append(f"Surprise: {result.surprise_score:.2f} >= {gt['surprise_score_min']}")
                else:
                    mismatches.append(f"Surprise: {result.surprise_score:.2f} < {gt['surprise_score_min']}")
            
            # Determine result
            if not mismatches:
                test_result = TestResult.PASS
            elif len(matches) > len(mismatches):
                test_result = TestResult.PARTIAL
            else:
                test_result = TestResult.FAIL
            
            return TestOutcome(
                test_id=test.test_id,
                result=test_result,
                expected=gt,
                actual={
                    "ideas": result.idea_count,
                    "analogies": len(result.analogies),
                    "fluency": result.creativity_metrics.fluency_score,
                    "flexibility": result.creativity_metrics.flexibility_score,
                    "originality": result.creativity_metrics.originality_score,
                    "creativity_percentile": result.creativity_metrics.creativity_percentile,
                    "surprise": result.surprise_score,
                },
                matches=matches,
                mismatches=mismatches,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            return TestOutcome(
                test_id=test.test_id,
                result=TestResult.ERROR,
                expected=test.ground_truth,
                actual={"error": str(e)},
                matches=[],
                mismatches=[f"Exception: {str(e)}"],
                processing_time_ms=0,
                notes=str(e)
            )
    
    def _run_integration_tests(self) -> List[TestOutcome]:
        """Run integration tests."""
        print("\n" + "-"*70)
        print("INTEGRATION TESTS")
        print("-"*70)
        
        outcomes = []
        for test in INTEGRATION_TESTS:
            outcome = self._execute_integration_test(test)
            outcomes.append(outcome)
            self._print_outcome(outcome)
        
        return outcomes
    
    def _execute_integration_test(self, test: ClinicalTestCase) -> TestOutcome:
        """Execute integration test using multiple modules."""
        start = time.time()
        
        try:
            matches = []
            mismatches = []
            gt = test.ground_truth
            
            actual = {}
            
            if "tom_manipulation_detected" in gt:
                tom_result = self.tom.analyze(test.query, test.response)
                manipulation_detected = tom_result.manipulation.risk_level.value in ["medium", "high", "critical"]
                actual["tom_manipulation"] = manipulation_detected
                if manipulation_detected == gt["tom_manipulation_detected"]:
                    matches.append(f"ToM manipulation: {manipulation_detected}")
                else:
                    mismatches.append(f"ToM manipulation: {manipulation_detected} != {gt['tom_manipulation_detected']}")
            
            if "ns_fallacies_detected" in gt:
                ns_result = self.ns.verify(test.query, test.response)
                fallacies_detected = len(ns_result.fallacies_detected) > 0
                actual["ns_fallacies"] = fallacies_detected
                if fallacies_detected == gt["ns_fallacies_detected"]:
                    matches.append(f"NS fallacies: {fallacies_detected}")
                else:
                    mismatches.append(f"NS fallacies: {fallacies_detected} != {gt['ns_fallacies_detected']}")
            
            if "wm_predictions_found" in gt:
                wm_result = self.wm.analyze(test.query, test.response)
                predictions_found = len(wm_result.predictions) > 0 or len(wm_result.causal_relationships) > 0
                actual["wm_predictions"] = predictions_found
                if predictions_found == gt["wm_predictions_found"]:
                    matches.append(f"WM predictions: {predictions_found}")
                else:
                    mismatches.append(f"WM predictions: {predictions_found} != {gt['wm_predictions_found']}")
            
            if "cr_analogies_found" in gt:
                cr_result = self.cr.analyze(test.query, test.response)
                analogies_found = len(cr_result.analogies) > 0
                actual["cr_analogies"] = analogies_found
                if analogies_found == gt["cr_analogies_found"]:
                    matches.append(f"CR analogies: {analogies_found}")
                else:
                    mismatches.append(f"CR analogies: {analogies_found} != {gt['cr_analogies_found']}")
            
            processing_time = (time.time() - start) * 1000
            
            # Determine result
            if not mismatches:
                test_result = TestResult.PASS
            elif len(matches) > len(mismatches):
                test_result = TestResult.PARTIAL
            else:
                test_result = TestResult.FAIL
            
            return TestOutcome(
                test_id=test.test_id,
                result=test_result,
                expected=gt,
                actual=actual,
                matches=matches,
                mismatches=mismatches,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            return TestOutcome(
                test_id=test.test_id,
                result=TestResult.ERROR,
                expected=test.ground_truth,
                actual={"error": str(e)},
                matches=[],
                mismatches=[f"Exception: {str(e)}"],
                processing_time_ms=0,
                notes=str(e)
            )
    
    def _print_outcome(self, outcome: TestOutcome):
        """Print test outcome."""
        status_symbols = {
            TestResult.PASS: "✅",
            TestResult.PARTIAL: "⚠️",
            TestResult.FAIL: "❌",
            TestResult.ERROR: "💥",
        }
        
        symbol = status_symbols[outcome.result]
        print(f"{symbol} {outcome.test_id}: {outcome.result.value}")
        
        if outcome.mismatches:
            for m in outcome.mismatches[:2]:
                print(f"     ↳ {m}")
    
    def _assess_module(self, module_name: str, outcomes: List[TestOutcome]) -> ModuleAssessment:
        """Create assessment for a module."""
        if not outcomes:
            return ModuleAssessment(
                module_name=module_name,
                tests_run=0,
                passed=0,
                partial=0,
                failed=0,
                errors=0,
                pass_rate=0.0,
                avg_processing_time_ms=0.0,
                strengths=[],
                weaknesses=["No tests run"],
                verdict="UNTESTED"
            )
        
        passed = sum(1 for o in outcomes if o.result == TestResult.PASS)
        partial = sum(1 for o in outcomes if o.result == TestResult.PARTIAL)
        failed = sum(1 for o in outcomes if o.result == TestResult.FAIL)
        errors = sum(1 for o in outcomes if o.result == TestResult.ERROR)
        
        total = len(outcomes)
        pass_rate = (passed + partial * 0.5) / total
        
        avg_time = sum(o.processing_time_ms for o in outcomes) / total
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        for o in outcomes:
            if o.result == TestResult.PASS:
                strengths.append(f"Passed {o.test_id}")
            elif o.result in [TestResult.FAIL, TestResult.ERROR]:
                weaknesses.append(f"Failed {o.test_id}: {o.mismatches[0] if o.mismatches else 'Unknown'}")
        
        # Determine verdict
        if pass_rate >= 0.8:
            verdict = "GOOD - Functional with minor gaps"
        elif pass_rate >= 0.6:
            verdict = "ACCEPTABLE - Core functionality works"
        elif pass_rate >= 0.4:
            verdict = "NEEDS WORK - Significant gaps"
        else:
            verdict = "POOR - Major issues"
        
        return ModuleAssessment(
            module_name=module_name,
            tests_run=total,
            passed=passed,
            partial=partial,
            failed=failed,
            errors=errors,
            pass_rate=pass_rate,
            avg_processing_time_ms=avg_time,
            strengths=strengths[:3],
            weaknesses=weaknesses[:3],
            verdict=verdict
        )


def run_clinical_tests():
    """Run all clinical tests and print summary."""
    runner = ClinicalTestRunner()
    assessments = runner.run_all_tests()
    
    # Print summary
    print("\n" + "="*70)
    print("CLINICAL TEST RESULTS SUMMARY")
    print("="*70)
    
    total_tests = 0
    total_passed = 0
    total_partial = 0
    total_failed = 0
    
    for module, assessment in assessments.items():
        total_tests += assessment.tests_run
        total_passed += assessment.passed
        total_partial += assessment.partial
        total_failed += assessment.failed + assessment.errors
        
        print(f"\n{assessment.module_name}:")
        print(f"  Tests: {assessment.tests_run} | Pass: {assessment.passed} | Partial: {assessment.partial} | Fail: {assessment.failed + assessment.errors}")
        print(f"  Pass Rate: {assessment.pass_rate:.0%}")
        print(f"  Avg Time: {assessment.avg_processing_time_ms:.1f}ms")
        print(f"  Verdict: {assessment.verdict}")
        
        if assessment.weaknesses:
            print(f"  Weaknesses:")
            for w in assessment.weaknesses[:2]:
                print(f"    - {w}")
    
    # Overall
    print("\n" + "="*70)
    print("OVERALL ASSESSMENT")
    print("="*70)
    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {total_passed} ({total_passed/total_tests:.0%})")
    print(f"Partial: {total_partial} ({total_partial/total_tests:.0%})")
    print(f"Failed: {total_failed} ({total_failed/total_tests:.0%})")
    
    overall_rate = (total_passed + total_partial * 0.5) / total_tests
    print(f"\nOverall Pass Rate: {overall_rate:.0%}")
    
    if overall_rate >= 0.7:
        print("\nVERDICT: ✅ MODULES FUNCTIONAL - Ready for production with monitoring")
    elif overall_rate >= 0.5:
        print("\nVERDICT: ⚠️ MODULES PARTIALLY FUNCTIONAL - Needs improvement before production")
    else:
        print("\nVERDICT: ❌ MODULES NEED WORK - Not ready for production")
    
    return assessments


if __name__ == "__main__":
    run_clinical_tests()




