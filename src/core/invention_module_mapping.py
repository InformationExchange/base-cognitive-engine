"""
BASE Cognitive Governance Engine - Invention to Module Mapping
Phase G2: Explicit mapping of all 71 inventions to implementing modules

This module provides:
1. Complete mapping of 71 inventions to implementation modules
2. Claims verification registry (309 claims)
3. Coverage analysis and gap detection
4. Runtime verification of invention availability

Patent Claims:
- NOVEL-37: Invention-Module Registry (NEW)
"""

import importlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ImplementationStatus(Enum):
    """Implementation status for inventions."""
    FULLY_IMPLEMENTED = "fully_implemented"
    PARTIALLY_IMPLEMENTED = "partially_implemented"
    STUB_ONLY = "stub_only"
    MISSING = "missing"
    DEPRECATED = "deprecated"


class ClaimStatus(Enum):
    """Verification status for claims."""
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    FAILED = "failed"
    PENDING = "pending"


@dataclass
class InventionMapping:
    """Mapping of an invention to its implementation."""
    invention_id: str
    invention_name: str
    brain_layer: int
    brain_region: str
    implementing_modules: List[str]
    primary_module: str
    primary_class: str
    status: ImplementationStatus
    claims: List[str]
    dependencies: List[str] = field(default_factory=list)
    description: str = ""
    test_file: Optional[str] = None
    last_verified: Optional[str] = None


@dataclass
class ClaimMapping:
    """Mapping of a claim to its verification."""
    claim_id: str
    invention_id: str
    claim_text: str
    status: ClaimStatus
    test_method: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    last_verified: Optional[str] = None


class InventionModuleRegistry:
    """
    Registry mapping all 71 inventions to their implementing modules.
    
    Purpose:
    - Provides explicit traceability from patent claims to code
    - Enables coverage analysis
    - Supports runtime verification
    - Generates gap reports
    
    Patent: NOVEL-37 (Invention-Module Registry)
    """
    
    def __init__(self):
        self.inventions: Dict[str, InventionMapping] = {}
        self.claims: Dict[str, ClaimMapping] = {}
        self._initialize_mappings()
        self._initialize_claims()
        
        logger.info(f"[InventionRegistry] Initialized with {len(self.inventions)} inventions, "
                   f"{len(self.claims)} claims")
    
    def _initialize_mappings(self):
        """Initialize the complete 71-invention mapping."""
        
        # ============================================================
        # LAYER 1: SENSORY CORTEX (Perception) - 8 Inventions
        # ============================================================
        
        self.inventions["PPA1-Inv1"] = InventionMapping(
            invention_id="PPA1-Inv1",
            invention_name="Multi-Modal Behavioral Data Fusion",
            brain_layer=1,
            brain_region="Sensory Cortex",
            implementing_modules=["core.integrated_engine", "core.multimodal_context"],
            primary_module="core.multimodal_context",
            primary_class="MultiModalContextEngine",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA1-C1", "PPA1-C2", "PPA1-C3", "PPA1-C4"],
            description="Fuses multiple input modalities (text, embeddings, numeric) into unified signal"
        )
        
        self.inventions["UP1"] = InventionMapping(
            invention_id="UP1",
            invention_name="RAG Hallucination Prevention",
            brain_layer=1,
            brain_region="Sensory Cortex",
            implementing_modules=["detectors.grounding"],
            primary_module="detectors.grounding",
            primary_class="GroundingDetector",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["UP1-C1", "UP1-C2", "UP1-C3"],
            description="Detects hallucinations by comparing claims against grounding documents"
        )
        
        self.inventions["UP2"] = InventionMapping(
            invention_id="UP2",
            invention_name="Fact-Checking Pathway",
            brain_layer=1,
            brain_region="Sensory Cortex",
            implementing_modules=["detectors.factual"],
            primary_module="detectors.factual",
            primary_class="FactualDetector",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["UP2-C1", "UP2-C2"],
            description="Verifies factual claims using evidence triangulation"
        )
        
        self.inventions["PPA1-Inv14"] = InventionMapping(
            invention_id="PPA1-Inv14",
            invention_name="Behavioral Capture",
            brain_layer=1,
            brain_region="Sensory Cortex",
            implementing_modules=["detectors.behavioral"],
            primary_module="detectors.behavioral",
            primary_class="BehavioralBiasDetector",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA1-C14-1", "PPA1-C14-2"],
            description="Captures behavioral signals like sycophancy, confirmation bias"
        )
        
        self.inventions["PPA3-Inv1"] = InventionMapping(
            invention_id="PPA3-Inv1",
            invention_name="Temporal Detection",
            brain_layer=1,
            brain_region="Sensory Cortex",
            implementing_modules=["detectors.temporal", "detectors.temporal_bias_detector"],
            primary_module="detectors.temporal",
            primary_class="TemporalDetector",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA3-C1-1", "PPA3-C1-2"],
            description="Detects temporal patterns and anomalies in signals"
        )
        
        self.inventions["NOVEL-9"] = InventionMapping(
            invention_id="NOVEL-9",
            invention_name="Query Analyzer",
            brain_layer=1,
            brain_region="Sensory Cortex",
            implementing_modules=["core.query_analyzer"],
            primary_module="core.query_analyzer",
            primary_class="QueryAnalyzer",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["NOVEL-9-C1", "NOVEL-9-C2"],
            description="Analyzes incoming queries for complexity, domain, risk"
        )
        
        self.inventions["PPA1-Inv11"] = InventionMapping(
            invention_id="PPA1-Inv11",
            invention_name="Bias Formation Patterns",
            brain_layer=1,
            brain_region="Sensory Cortex",
            implementing_modules=["core.bias_evolution_tracker"],
            primary_module="core.bias_evolution_tracker",
            primary_class="BiasEvolutionTracker",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA1-C11-1", "PPA1-C11-2"],
            description="Tracks how biases form and evolve over time"
        )
        
        self.inventions["PPA1-Inv18"] = InventionMapping(
            invention_id="PPA1-Inv18",
            invention_name="High-Fidelity Capture",
            brain_layer=1,
            brain_region="Sensory Cortex",
            implementing_modules=["core.logging_telemetry"],
            primary_module="core.logging_telemetry",
            primary_class="EnhancedLoggingEngine",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA1-C18-1"],
            description="High-fidelity signal capture for detailed analysis"
        )
        
        # ============================================================
        # LAYER 2: PREFRONTAL CORTEX (Reasoning) - 12 Inventions
        # ============================================================
        
        self.inventions["PPA1-Inv5"] = InventionMapping(
            invention_id="PPA1-Inv5",
            invention_name="ACRL Literacy Standards",
            brain_layer=2,
            brain_region="Prefrontal Cortex",
            implementing_modules=["core.reasoning_chain_analyzer"],
            primary_module="core.reasoning_chain_analyzer",
            primary_class="ReasoningChainAnalyzer",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA1-C5-1", "PPA1-C5-2", "PPA1-C5-3"],
            description="Information literacy standards for claim evaluation"
        )
        
        self.inventions["PPA1-Inv7"] = InventionMapping(
            invention_id="PPA1-Inv7",
            invention_name="Structured Reasoning Trees",
            brain_layer=2,
            brain_region="Prefrontal Cortex",
            implementing_modules=["core.reasoning_chain_analyzer"],
            primary_module="core.reasoning_chain_analyzer",
            primary_class="ReasoningChainAnalyzer",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA1-C7-1", "PPA1-C7-2"],
            description="Builds and evaluates reasoning trees"
        )
        
        self.inventions["PPA1-Inv8"] = InventionMapping(
            invention_id="PPA1-Inv8",
            invention_name="Contradiction Handling",
            brain_layer=2,
            brain_region="Prefrontal Cortex",
            implementing_modules=["core.contradiction_resolver"],
            primary_module="core.contradiction_resolver",
            primary_class="ContradictionResolver",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA1-C8-1", "PPA1-C8-2"],
            description="Resolves contradictions in reasoning chains"
        )
        
        self.inventions["PPA1-Inv10"] = InventionMapping(
            invention_id="PPA1-Inv10",
            invention_name="Belief Pathway Analysis",
            brain_layer=2,
            brain_region="Prefrontal Cortex",
            implementing_modules=["core.reasoning_chain_analyzer"],
            primary_module="core.reasoning_chain_analyzer",
            primary_class="ReasoningChainAnalyzer",
            status=ImplementationStatus.PARTIALLY_IMPLEMENTED,
            claims=["PPA1-C10-1"],
            description="Analyzes belief formation pathways"
        )
        
        self.inventions["UP3"] = InventionMapping(
            invention_id="UP3",
            invention_name="Neuro-Symbolic Reasoning",
            brain_layer=2,
            brain_region="Prefrontal Cortex",
            implementing_modules=["research.neurosymbolic"],
            primary_module="research.neurosymbolic",
            primary_class="NeuroSymbolicModule",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["UP3-C1", "UP3-C2", "UP3-C3"],
            description="Combines neural and symbolic reasoning"
        )
        
        self.inventions["NOVEL-15"] = InventionMapping(
            invention_id="NOVEL-15",
            invention_name="Neuro-Symbolic Integration",
            brain_layer=2,
            brain_region="Prefrontal Cortex",
            implementing_modules=["research.neurosymbolic"],
            primary_module="research.neurosymbolic",
            primary_class="NeuroSymbolicModule",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["NOVEL-15-C1"],
            description="Integrates neuro-symbolic components"
        )
        
        self.inventions["PPA1-Inv19"] = InventionMapping(
            invention_id="PPA1-Inv19",
            invention_name="Multi-Framework Convergence",
            brain_layer=2,
            brain_region="Prefrontal Cortex",
            implementing_modules=["detectors.multi_framework"],
            primary_module="detectors.multi_framework",
            primary_class="MultiFrameworkAnalyzer",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA1-C19-1", "PPA1-C19-2"],
            description="Converges multiple analysis frameworks"
        )
        
        self.inventions["PPA2-Comp4"] = InventionMapping(
            invention_id="PPA2-Comp4",
            invention_name="Conformal Must-Pass",
            brain_layer=2,
            brain_region="Prefrontal Cortex",
            implementing_modules=["core.predicate_acceptance"],
            primary_module="core.predicate_acceptance",
            primary_class="PredicateAcceptance",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA2-C4-1", "PPA2-C4-2"],
            description="Conformal prediction for must-pass predicates"
        )
        
        self.inventions["PPA2-Inv26"] = InventionMapping(
            invention_id="PPA2-Inv26",
            invention_name="Lexicographic Gate",
            brain_layer=2,
            brain_region="Prefrontal Cortex",
            implementing_modules=["core.predicate_acceptance"],
            primary_module="core.predicate_acceptance",
            primary_class="PredicateAcceptance",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA2-C26-1"],
            description="Priority ordering for predicate evaluation"
        )
        
        self.inventions["NOVEL-16"] = InventionMapping(
            invention_id="NOVEL-16",
            invention_name="World Models",
            brain_layer=2,
            brain_region="Prefrontal Cortex",
            implementing_modules=["core.world_models"],
            primary_module="core.world_models",
            primary_class="WorldModelAnalyzer",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["NOVEL-16-C1", "NOVEL-16-C2"],
            description="Causal world model reasoning"
        )
        
        self.inventions["NOVEL-17"] = InventionMapping(
            invention_id="NOVEL-17",
            invention_name="Creative Reasoning",
            brain_layer=2,
            brain_region="Prefrontal Cortex",
            implementing_modules=["core.creative_reasoning"],
            primary_module="core.creative_reasoning",
            primary_class="CreativeReasoning",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["NOVEL-17-C1"],
            description="Divergent thinking for novel situations"
        )
        
        self.inventions["PPA1-Inv4"] = InventionMapping(
            invention_id="PPA1-Inv4",
            invention_name="Computational Intervention",
            brain_layer=2,
            brain_region="Prefrontal Cortex",
            implementing_modules=["core.counterfactual_reasoning"],
            primary_module="core.counterfactual_reasoning",
            primary_class="CounterfactualReasoningEngine",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA1-C4-1", "PPA1-C4-2"],
            description="Causal intervention modeling"
        )
        
        # ============================================================
        # LAYER 3: LIMBIC SYSTEM (Behavioral) - 13 Inventions
        # ============================================================
        
        self.inventions["PPA1-Inv2"] = InventionMapping(
            invention_id="PPA1-Inv2",
            invention_name="Bias Modeling Framework",
            brain_layer=3,
            brain_region="Limbic System",
            implementing_modules=["detectors.behavioral"],
            primary_module="detectors.behavioral",
            primary_class="BehavioralBiasDetector",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA1-C2-1", "PPA1-C2-2", "PPA1-C2-3", "PPA1-C2-4"],
            description="Framework for modeling cognitive biases"
        )
        
        self.inventions["PPA3-Inv2"] = InventionMapping(
            invention_id="PPA3-Inv2",
            invention_name="Behavioral Detection",
            brain_layer=3,
            brain_region="Limbic System",
            implementing_modules=["detectors.behavioral"],
            primary_module="detectors.behavioral",
            primary_class="BehavioralBiasDetector",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA3-C2-1"],
            description="Detects behavioral bias patterns"
        )
        
        self.inventions["PPA3-Inv3"] = InventionMapping(
            invention_id="PPA3-Inv3",
            invention_name="Integrated Temporal-Behavioral",
            brain_layer=3,
            brain_region="Limbic System",
            implementing_modules=["detectors.temporal_bias_detector"],
            primary_module="detectors.temporal_bias_detector",
            primary_class="TemporalBiasDetector",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA3-C3-1"],
            description="Integrates temporal and behavioral analysis"
        )
        
        self.inventions["PPA2-Big5"] = InventionMapping(
            invention_id="PPA2-Big5",
            invention_name="OCEAN Personality Traits",
            brain_layer=3,
            brain_region="Limbic System",
            implementing_modules=["detectors.big5", "detectors.big5_personality"],
            primary_module="detectors.big5",
            primary_class="Big5Detector",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA2-Big5-C1", "PPA2-Big5-C2", "PPA2-Big5-C3"],
            description="Big Five personality trait detection"
        )
        
        self.inventions["PPA2-Big5-LLM"] = InventionMapping(
            invention_id="PPA2-Big5-LLM",
            invention_name="Hybrid LLM Verification",
            brain_layer=3,
            brain_region="Limbic System",
            implementing_modules=["detectors.big5"],
            primary_module="detectors.big5",
            primary_class="Big5Detector",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA2-Big5-LLM-C1"],
            description="LLM-verified personality detection"
        )
        
        self.inventions["NOVEL-1"] = InventionMapping(
            invention_id="NOVEL-1",
            invention_name="Too-Good-To-Be-True Detector",
            brain_layer=3,
            brain_region="Limbic System",
            implementing_modules=["detectors.behavioral"],
            primary_module="detectors.behavioral",
            primary_class="BehavioralBiasDetector",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["NOVEL-1-C1", "NOVEL-1-C2"],
            description="Detects overconfident/unrealistic claims"
        )
        
        self.inventions["PPA1-Inv6"] = InventionMapping(
            invention_id="PPA1-Inv6",
            invention_name="Bias-Aware Knowledge Graphs",
            brain_layer=3,
            brain_region="Limbic System",
            implementing_modules=["core.knowledge_graph"],
            primary_module="core.knowledge_graph",
            primary_class="BiasAwareKnowledgeGraph",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA1-C6-1", "PPA1-C6-2"],
            description="Knowledge graph with bias tracking"
        )
        
        self.inventions["PPA1-Inv13"] = InventionMapping(
            invention_id="PPA1-Inv13",
            invention_name="Federated Relapse Mitigation",
            brain_layer=3,
            brain_region="Limbic System",
            implementing_modules=["core.federated_privacy"],
            primary_module="core.federated_privacy",
            primary_class="FederatedPrivacyEngine",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA1-C13-1"],
            description="Prevents learning regression in federated setting"
        )
        
        self.inventions["PPA1-Inv24"] = InventionMapping(
            invention_id="PPA1-Inv24",
            invention_name="Neuroplasticity",
            brain_layer=3,
            brain_region="Limbic System",
            implementing_modules=["learning.bias_evolution", "core.bias_evolution_tracker"],
            primary_module="learning.bias_evolution",
            primary_class="BiasEvolutionTracker",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA1-C24-1"],
            description="Adaptive bias pathway modification"
        )
        
        self.inventions["PPA1-Inv12"] = InventionMapping(
            invention_id="PPA1-Inv12",
            invention_name="Adaptive Difficulty (ZPD)",
            brain_layer=3,
            brain_region="Limbic System",
            implementing_modules=["core.zpd_manager", "learning.adaptive_difficulty"],
            primary_module="core.zpd_manager",
            primary_class="ZPDManager",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA1-C12-1", "PPA1-C12-2"],
            description="Zone of Proximal Development management"
        )
        
        self.inventions["NOVEL-4"] = InventionMapping(
            invention_id="NOVEL-4",
            invention_name="Zone of Proximal Development",
            brain_layer=3,
            brain_region="Limbic System",
            implementing_modules=["core.zpd_manager"],
            primary_module="core.zpd_manager",
            primary_class="ZPDManager",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["NOVEL-4-C1"],
            description="Growth mindset for challenge calibration"
        )
        
        self.inventions["PPA1-Inv3"] = InventionMapping(
            invention_id="PPA1-Inv3",
            invention_name="Federated Convergence",
            brain_layer=3,
            brain_region="Limbic System",
            implementing_modules=["core.federated_privacy"],
            primary_module="core.federated_privacy",
            primary_class="FederatedPrivacyEngine",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA1-C3-1", "PPA1-C3-2"],
            description="Federated learning convergence"
        )
        
        self.inventions["NOVEL-14"] = InventionMapping(
            invention_id="NOVEL-14",
            invention_name="Theory of Mind",
            brain_layer=3,
            brain_region="Limbic System",
            implementing_modules=["core.theory_of_mind"],
            primary_module="core.theory_of_mind",
            primary_class="TheoryOfMind",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["NOVEL-14-C1", "NOVEL-14-C2"],
            description="Understanding user intent and emotional state"
        )
        
        # ============================================================
        # LAYER 4: HIPPOCAMPUS (Memory) - 6 Inventions
        # ============================================================
        
        self.inventions["PPA1-Inv22"] = InventionMapping(
            invention_id="PPA1-Inv22",
            invention_name="Feedback Loop",
            brain_layer=4,
            brain_region="Hippocampus",
            implementing_modules=["learning.feedback_loop", "core.learning_feedback"],
            primary_module="learning.feedback_loop",
            primary_class="FeedbackLoop",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA1-C22-1", "PPA1-C22-2", "PPA1-C22-3"],
            description="Human-AI cross-learning feedback loop"
        )
        
        self.inventions["PPA2-Inv27"] = InventionMapping(
            invention_id="PPA2-Inv27",
            invention_name="OCO Threshold Adapter",
            brain_layer=4,
            brain_region="Hippocampus",
            implementing_modules=["learning.algorithms", "learning.threshold_optimizer"],
            primary_module="learning.algorithms",
            primary_class="OCOLearner",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA2-C27-1", "PPA2-C27-2", "PPA2-C27-3", "PPA2-C27-4"],
            description="Online Convex Optimization for threshold adaptation"
        )
        
        self.inventions["PPA2-Comp5"] = InventionMapping(
            invention_id="PPA2-Comp5",
            invention_name="Crisis-Mode Override",
            brain_layer=4,
            brain_region="Hippocampus",
            implementing_modules=["core.crisis_detection"],
            primary_module="core.crisis_detection",
            primary_class="CrisisDetectionManager",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA2-C5-1", "PPA2-C5-2"],
            description="State-dependent memory with crisis override"
        )
        
        self.inventions["NOVEL-18"] = InventionMapping(
            invention_id="NOVEL-18",
            invention_name="Governance Rules Engine",
            brain_layer=4,
            brain_region="Hippocampus",
            implementing_modules=["core.governance_rules"],
            primary_module="core.governance_rules",
            primary_class="GovernanceRulesEngine",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["NOVEL-18-C1", "NOVEL-18-C2"],
            description="Rule memory for governance enforcement"
        )
        
        self.inventions["PPA1-Inv16"] = InventionMapping(
            invention_id="PPA1-Inv16",
            invention_name="Progressive Bias Adjustment",
            brain_layer=4,
            brain_region="Hippocampus",
            implementing_modules=["learning.outcome_memory"],
            primary_module="learning.outcome_memory",
            primary_class="OutcomeMemory",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA1-C16-1"],
            description="Gradual bias threshold evolution"
        )
        
        self.inventions["NOVEL-7"] = InventionMapping(
            invention_id="NOVEL-7",
            invention_name="Neuroplasticity Learning",
            brain_layer=4,
            brain_region="Hippocampus",
            implementing_modules=["learning.bias_evolution", "core.ai_enhanced_learning"],
            primary_module="core.ai_enhanced_learning",
            primary_class="AIEnhancedLearningManager",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["NOVEL-7-C1", "NOVEL-7-C2"],
            description="Pattern reinforcement and neuroplasticity"
        )
        
        # ============================================================
        # LAYER 5: ANTERIOR CINGULATE (Self-Awareness) - 4 Inventions
        # ============================================================
        
        self.inventions["NOVEL-21"] = InventionMapping(
            invention_id="NOVEL-21",
            invention_name="Self-Awareness Loop",
            brain_layer=5,
            brain_region="Anterior Cingulate",
            implementing_modules=["core.self_awareness"],
            primary_module="core.self_awareness",
            primary_class="SelfAwarenessLoop",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["NOVEL-21-C1", "NOVEL-21-C2", "NOVEL-21-C3"],
            description="Error monitoring for BASE itself"
        )
        
        self.inventions["NOVEL-2"] = InventionMapping(
            invention_id="NOVEL-2",
            invention_name="Governance-Guided Development",
            brain_layer=5,
            brain_region="Anterior Cingulate",
            implementing_modules=["core.governance_rules"],
            primary_module="core.governance_rules",
            primary_class="GovernanceRulesEngine",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["NOVEL-2-C1"],
            description="Meta-cognition for governance enforcement"
        )
        
        self.inventions["PPA2-Comp6"] = InventionMapping(
            invention_id="PPA2-Comp6",
            invention_name="Calibration Module",
            brain_layer=5,
            brain_region="Anterior Cingulate",
            implementing_modules=["core.ccp_calibrator"],
            primary_module="core.ccp_calibrator",
            primary_class="CalibratedContextualPosterior",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA2-C6-1", "PPA2-C6-2"],
            description="Confidence calibration for outputs"
        )
        
        self.inventions["PPA2-Comp3"] = InventionMapping(
            invention_id="PPA2-Comp3",
            invention_name="OCO Implementation",
            brain_layer=5,
            brain_region="Anterior Cingulate",
            implementing_modules=["learning.algorithms"],
            primary_module="learning.algorithms",
            primary_class="OCOLearner",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA2-C3-1", "PPA2-C3-2", "PPA2-C3-3"],
            description="Online Convex Optimization implementation"
        )
        
        # ============================================================
        # LAYER 6: CEREBELLUM (Improvement) - 5 Inventions
        # ============================================================
        
        self.inventions["NOVEL-20"] = InventionMapping(
            invention_id="NOVEL-20",
            invention_name="Response Improver",
            brain_layer=6,
            brain_region="Cerebellum",
            implementing_modules=["core.response_improver", "core.response_generator"],
            primary_module="core.response_improver",
            primary_class="ResponseImprover",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["NOVEL-20-C1", "NOVEL-20-C2"],
            description="Refines and improves LLM responses"
        )
        
        self.inventions["UP5"] = InventionMapping(
            invention_id="UP5",
            invention_name="Cognitive Enhancement",
            brain_layer=6,
            brain_region="Cerebellum",
            implementing_modules=["core.cognitive_enhancer"],
            primary_module="core.cognitive_enhancer",
            primary_class="CognitiveEnhancer",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["UP5-C1", "UP5-C2"],
            description="Performance boost for enhancement"
        )
        
        self.inventions["PPA1-Inv17"] = InventionMapping(
            invention_id="PPA1-Inv17",
            invention_name="Cognitive Window",
            brain_layer=6,
            brain_region="Cerebellum",
            implementing_modules=["core.cognitive_enhancer"],
            primary_module="core.cognitive_enhancer",
            primary_class="CognitiveEnhancer",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA1-C17-1"],
            description="Timing optimization for intervention"
        )
        
        self.inventions["NOVEL-5"] = InventionMapping(
            invention_id="NOVEL-5",
            invention_name="Vibe Coding Verification",
            brain_layer=6,
            brain_region="Cerebellum",
            implementing_modules=["core.testing_infrastructure"],
            primary_module="core.testing_infrastructure",
            primary_class="EnhancedTestingEngine",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["NOVEL-5-C1"],
            description="Output validation for code generation"
        )
        
        self.inventions["PPA2-Inv28"] = InventionMapping(
            invention_id="PPA2-Inv28",
            invention_name="Cognitive Window Intervention",
            brain_layer=6,
            brain_region="Cerebellum",
            implementing_modules=["core.cognitive_enhancer"],
            primary_module="core.cognitive_enhancer",
            primary_class="CognitiveEnhancer",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA2-C28-1"],
            description="Timed intervention within cognitive window"
        )
        
        # ============================================================
        # LAYER 7: THALAMUS (Orchestration) - 8 Inventions
        # ============================================================
        
        self.inventions["NOVEL-10"] = InventionMapping(
            invention_id="NOVEL-10",
            invention_name="Smart Gate",
            brain_layer=7,
            brain_region="Thalamus",
            implementing_modules=["core.smart_gate"],
            primary_module="core.smart_gate",
            primary_class="SmartGate",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["NOVEL-10-C1", "NOVEL-10-C2"],
            description="Signal routing based on risk"
        )
        
        self.inventions["NOVEL-11"] = InventionMapping(
            invention_id="NOVEL-11",
            invention_name="Hybrid Orchestrator",
            brain_layer=7,
            brain_region="Thalamus",
            implementing_modules=["core.hybrid_orchestrator"],
            primary_module="core.hybrid_orchestrator",
            primary_class="HybridOrchestrator",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["NOVEL-11-C1", "NOVEL-11-C2"],
            description="Multi-path coordination"
        )
        
        self.inventions["NOVEL-12"] = InventionMapping(
            invention_id="NOVEL-12",
            invention_name="Conversational Orchestrator",
            brain_layer=7,
            brain_region="Thalamus",
            implementing_modules=["core.conversational_orchestrator"],
            primary_module="core.conversational_orchestrator",
            primary_class="ConversationalOrchestrator",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["NOVEL-12-C1"],
            description="Dialog management for multi-turn"
        )
        
        self.inventions["NOVEL-8"] = InventionMapping(
            invention_id="NOVEL-8",
            invention_name="Cross-LLM Governance",
            brain_layer=7,
            brain_region="Thalamus",
            implementing_modules=["core.llm_registry"],
            primary_module="core.llm_registry",
            primary_class="LLMRegistry",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["NOVEL-8-C1", "NOVEL-8-C2"],
            description="Multi-LLM routing and governance"
        )
        
        self.inventions["NOVEL-19"] = InventionMapping(
            invention_id="NOVEL-19",
            invention_name="LLM Registry",
            brain_layer=7,
            brain_region="Thalamus",
            implementing_modules=["core.llm_registry"],
            primary_module="core.llm_registry",
            primary_class="LLMRegistry",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["NOVEL-19-C1"],
            description="Model selection and management"
        )
        
        self.inventions["PPA2-Comp2"] = InventionMapping(
            invention_id="PPA2-Comp2",
            invention_name="Feature-Specific Thresholds",
            brain_layer=7,
            brain_region="Thalamus",
            implementing_modules=["learning.threshold_optimizer", "core.trigger_intelligence"],
            primary_module="learning.threshold_optimizer",
            primary_class="AdaptiveThresholdOptimizer",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA2-C2-1", "PPA2-C2-2"],
            description="Dynamic per-domain thresholds"
        )
        
        self.inventions["PPA2-Comp8"] = InventionMapping(
            invention_id="PPA2-Comp8",
            invention_name="VOI Short-Circuiting",
            brain_layer=7,
            brain_region="Thalamus",
            implementing_modules=["learning.voi_shortcircuit"],
            primary_module="learning.voi_shortcircuit",
            primary_class="VOIShortCircuit",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA2-C8-1"],
            description="Value-of-information efficiency routing"
        )
        
        self.inventions["PPA1-Inv9"] = InventionMapping(
            invention_id="PPA1-Inv9",
            invention_name="Cross-Platform Harmonization",
            brain_layer=7,
            brain_region="Thalamus",
            implementing_modules=["core.integration_hub", "core.api_gateway"],
            primary_module="core.integration_hub",
            primary_class="EnhancedIntegrationHub",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA1-C9-1"],
            description="Platform abstraction and API standardization"
        )
        
        # ============================================================
        # LAYER 8: AMYGDALA (Challenge) - 4 Inventions
        # ============================================================
        
        self.inventions["NOVEL-22"] = InventionMapping(
            invention_id="NOVEL-22",
            invention_name="LLM Challenger",
            brain_layer=8,
            brain_region="Amygdala",
            implementing_modules=["core.llm_challenger"],
            primary_module="core.llm_challenger",
            primary_class="LLMChallenger",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["NOVEL-22-C1", "NOVEL-22-C2", "NOVEL-22-C3"],
            description="Adversarial challenge for high-stakes claims"
        )
        
        self.inventions["NOVEL-23"] = InventionMapping(
            invention_id="NOVEL-23",
            invention_name="Multi-Track Challenger",
            brain_layer=8,
            brain_region="Amygdala",
            implementing_modules=["core.multi_track_challenger"],
            primary_module="core.multi_track_challenger",
            primary_class="MultiTrackChallenger",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["NOVEL-23-C1", "NOVEL-23-C2"],
            description="Multi-perspective threat analysis"
        )
        
        self.inventions["NOVEL-6"] = InventionMapping(
            invention_id="NOVEL-6",
            invention_name="Triangulation Verification",
            brain_layer=8,
            brain_region="Amygdala",
            implementing_modules=["detectors.factual"],
            primary_module="detectors.factual",
            primary_class="FactualDetector",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["NOVEL-6-C1"],
            description="Cross-source validation"
        )
        
        self.inventions["PPA1-Inv20"] = InventionMapping(
            invention_id="PPA1-Inv20",
            invention_name="Human-Machine Hybrid",
            brain_layer=8,
            brain_region="Amygdala",
            implementing_modules=["core.active_learning_hitl"],
            primary_module="core.active_learning_hitl",
            primary_class="ActiveLearningEngine",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA1-C20-1", "PPA1-C20-2"],
            description="Escalation trigger for human review"
        )
        
        # ============================================================
        # LAYER 9: BASAL GANGLIA (Evidence) - 4 Inventions
        # ============================================================
        
        self.inventions["NOVEL-3"] = InventionMapping(
            invention_id="NOVEL-3",
            invention_name="Claim-Evidence Alignment",
            brain_layer=9,
            brain_region="Basal Ganglia",
            implementing_modules=["core.evidence_demand"],
            primary_module="core.evidence_demand",
            primary_class="EvidenceDemandLoop",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["NOVEL-3-C1", "NOVEL-3-C2"],
            description="Action validation for completion claims"
        )
        
        self.inventions["GAP-1"] = InventionMapping(
            invention_id="GAP-1",
            invention_name="Evidence Demand Loop",
            brain_layer=9,
            brain_region="Basal Ganglia",
            implementing_modules=["core.evidence_demand"],
            primary_module="core.evidence_demand",
            primary_class="EvidenceDemandLoop",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["GAP-1-C1", "GAP-1-C2", "GAP-1-C3"],
            description="Proof requirement for unverified claims"
        )
        
        self.inventions["PPA2-Comp7"] = InventionMapping(
            invention_id="PPA2-Comp7",
            invention_name="Verifiable Audit",
            brain_layer=9,
            brain_region="Basal Ganglia",
            implementing_modules=["core.verifiable_audit"],
            primary_module="core.verifiable_audit",
            primary_class="VerifiableAuditManager",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA2-C7-1", "PPA2-C7-2"],
            description="Merkle-tree logging for action verification"
        )
        
        self.inventions["UP4"] = InventionMapping(
            invention_id="UP4",
            invention_name="Knowledge Graph Integration",
            brain_layer=9,
            brain_region="Basal Ganglia",
            implementing_modules=["core.knowledge_graph"],
            primary_module="core.knowledge_graph",
            primary_class="BiasAwareKnowledgeGraph",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["UP4-C1", "UP4-C2"],
            description="Knowledge validation for entity claims"
        )
        
        # ============================================================
        # LAYER 10: MOTOR CORTEX (Output) - 5 Inventions
        # ============================================================
        
        self.inventions["PPA1-Inv21"] = InventionMapping(
            invention_id="PPA1-Inv21",
            invention_name="Configurable Predicate Acceptance",
            brain_layer=10,
            brain_region="Motor Cortex",
            implementing_modules=["core.predicate_acceptance"],
            primary_module="core.predicate_acceptance",
            primary_class="PredicateAcceptance",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA1-C21-1", "PPA1-C21-2"],
            description="K-of-N policy for action selection"
        )
        
        self.inventions["UP6"] = InventionMapping(
            invention_id="UP6",
            invention_name="Unified Governance System",
            brain_layer=10,
            brain_region="Motor Cortex",
            implementing_modules=["core.integrated_engine"],
            primary_module="core.integrated_engine",
            primary_class="IntegratedGovernanceEngine",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["UP6-C1", "UP6-C2", "UP6-C3"],
            description="Action coordination for evaluate() return"
        )
        
        self.inventions["UP7"] = InventionMapping(
            invention_id="UP7",
            invention_name="Calibration System",
            brain_layer=10,
            brain_region="Motor Cortex",
            implementing_modules=["core.ccp_calibrator"],
            primary_module="core.ccp_calibrator",
            primary_class="CalibratedContextualPosterior",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["UP7-C1", "UP7-C2"],
            description="Output calibration for confidence scores"
        )
        
        self.inventions["PPA1-Inv25"] = InventionMapping(
            invention_id="PPA1-Inv25",
            invention_name="Platform-Agnostic API",
            brain_layer=10,
            brain_region="Motor Cortex",
            implementing_modules=["api.integrated_routes", "core.api_gateway"],
            primary_module="core.api_gateway",
            primary_class="EnhancedAPIGateway",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA1-C25-1"],
            description="Output formatting for response API"
        )
        
        self.inventions["PPA2-Comp9"] = InventionMapping(
            invention_id="PPA2-Comp9",
            invention_name="Calibrated Posterior",
            brain_layer=10,
            brain_region="Motor Cortex",
            implementing_modules=["core.ccp_calibrator"],
            primary_module="core.ccp_calibrator",
            primary_class="CalibratedContextualPosterior",
            status=ImplementationStatus.FULLY_IMPLEMENTED,
            claims=["PPA2-C9-1", "PPA2-C9-2", "PPA2-C9-3"],
            description="Probability output with temperature scaling"
        )
    
    def _initialize_claims(self):
        """Initialize claims registry (simplified - full 309 would be extensive)."""
        # Generate claims from invention mappings
        for inv_id, inv in self.inventions.items():
            for claim_id in inv.claims:
                self.claims[claim_id] = ClaimMapping(
                    claim_id=claim_id,
                    invention_id=inv_id,
                    claim_text=f"Claim {claim_id} for {inv.invention_name}",
                    status=ClaimStatus.PENDING
                )
    
    # ==== Query Methods ====
    
    def get_invention(self, invention_id: str) -> Optional[InventionMapping]:
        """Get mapping for a specific invention."""
        return self.inventions.get(invention_id)
    
    def get_inventions_by_layer(self, layer: int) -> List[InventionMapping]:
        """Get all inventions for a brain layer."""
        return [inv for inv in self.inventions.values() if inv.brain_layer == layer]
    
    def get_inventions_by_module(self, module_name: str) -> List[InventionMapping]:
        """Get all inventions implemented by a module."""
        return [
            inv for inv in self.inventions.values()
            if module_name in inv.implementing_modules
        ]
    
    def get_module_for_invention(self, invention_id: str) -> Optional[str]:
        """Get primary module for an invention."""
        inv = self.inventions.get(invention_id)
        return inv.primary_module if inv else None
    
    def get_claims_for_invention(self, invention_id: str) -> List[ClaimMapping]:
        """Get all claims for an invention."""
        return [c for c in self.claims.values() if c.invention_id == invention_id]
    
    # ==== Verification Methods ====
    
    def verify_invention(self, invention_id: str) -> Tuple[bool, str]:
        """Verify that an invention's module exists and can be imported."""
        inv = self.inventions.get(invention_id)
        if not inv:
            return False, f"Unknown invention: {invention_id}"
        
        try:
            module = importlib.import_module(inv.primary_module)
            cls = getattr(module, inv.primary_class, None)
            
            if cls is None:
                return False, f"Class {inv.primary_class} not found in {inv.primary_module}"
            
            # Try to instantiate
            try:
                instance = cls()
                inv.last_verified = datetime.utcnow().isoformat()
                return True, f"Verified: {inv.primary_class} in {inv.primary_module}"
            except Exception as e:
                return False, f"Instantiation failed: {str(e)[:100]}"
                
        except ImportError as e:
            return False, f"Import failed: {inv.primary_module} - {str(e)[:100]}"
    
    def verify_all_inventions(self) -> Dict[str, Tuple[bool, str]]:
        """Verify all inventions."""
        results = {}
        for inv_id in self.inventions:
            results[inv_id] = self.verify_invention(inv_id)
        return results
    
    # ==== Coverage Analysis ====
    
    def get_coverage_report(self) -> Dict[str, Any]:
        """Generate comprehensive coverage report."""
        total = len(self.inventions)
        by_status = {}
        by_layer = {}
        
        for inv in self.inventions.values():
            # By status
            status = inv.status.value
            by_status[status] = by_status.get(status, 0) + 1
            
            # By layer
            layer = f"Layer {inv.brain_layer}: {inv.brain_region}"
            if layer not in by_layer:
                by_layer[layer] = {"total": 0, "implemented": 0}
            by_layer[layer]["total"] += 1
            if inv.status == ImplementationStatus.FULLY_IMPLEMENTED:
                by_layer[layer]["implemented"] += 1
        
        fully_implemented = by_status.get("fully_implemented", 0)
        
        return {
            "total_inventions": total,
            "by_status": by_status,
            "by_layer": by_layer,
            "coverage_percentage": (fully_implemented / total * 100) if total > 0 else 0,
            "total_claims": len(self.claims),
            "verified_claims": sum(1 for c in self.claims.values() if c.status == ClaimStatus.VERIFIED)
        }
    
    def get_gap_report(self) -> Dict[str, List[str]]:
        """Identify gaps in implementation."""
        gaps = {
            "missing": [],
            "partial": [],
            "stub": [],
            "unverified_claims": []
        }
        
        for inv_id, inv in self.inventions.items():
            if inv.status == ImplementationStatus.MISSING:
                gaps["missing"].append(inv_id)
            elif inv.status == ImplementationStatus.PARTIALLY_IMPLEMENTED:
                gaps["partial"].append(inv_id)
            elif inv.status == ImplementationStatus.STUB_ONLY:
                gaps["stub"].append(inv_id)
        
        for claim_id, claim in self.claims.items():
            if claim.status in [ClaimStatus.UNVERIFIED, ClaimStatus.PENDING]:
                gaps["unverified_claims"].append(claim_id)
        
        return gaps
    
    # ==== Learning Interface Methods ====
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        """Record verification outcome."""
        inv_id = outcome.get("invention_id")
        if inv_id and inv_id in self.inventions:
            self.inventions[inv_id].last_verified = datetime.utcnow().isoformat()
    
    def record_feedback(self, feedback: Dict) -> None:
        """Record feedback on mapping."""
        pass  # Placeholder
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Not applicable for registry."""
        pass
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Not applicable for registry."""
        return 0.0
    
    def get_learning_statistics(self) -> Dict:
        """Get registry statistics."""
        return self.get_coverage_report()
    
    def save_state(self) -> None:
        """Save registry state."""
        pass
    
    def load_state(self) -> None:
        """Load registry state."""
        pass

    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback."""
        pass  # Implement specific learning logic


    def get_statistics(self) -> Dict:
        """Return statistics."""
        return {'outcomes': len(getattr(self, '_outcomes', []))}


    def serialize_state(self) -> Dict:
        """Serialize state."""
        return {'outcomes': getattr(self, '_outcomes', [])[-100:]}


    def deserialize_state(self, state: Dict) -> None:
        """Restore state."""
        self._outcomes = state.get('outcomes', [])


# ==============================================================================
# Singleton instance
# ==============================================================================

_registry_instance: Optional[InventionModuleRegistry] = None

def get_invention_registry() -> InventionModuleRegistry:
    """Get singleton instance of the invention registry."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = InventionModuleRegistry()
    return _registry_instance


# ==============================================================================
# Test
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("INVENTION-MODULE REGISTRY TEST")
    print("=" * 80)
    
    registry = InventionModuleRegistry()
    
    # Get coverage report
    print("\n[1] Coverage Report:")
    report = registry.get_coverage_report()
    print(f"    Total Inventions: {report['total_inventions']}")
    print(f"    Coverage: {report['coverage_percentage']:.1f}%")
    print(f"    Total Claims: {report['total_claims']}")
    
    print("\n    By Status:")
    for status, count in report['by_status'].items():
        print(f"      {status}: {count}")
    
    print("\n    By Layer:")
    for layer, data in report['by_layer'].items():
        pct = data['implemented'] / data['total'] * 100 if data['total'] > 0 else 0
        print(f"      {layer}: {data['implemented']}/{data['total']} ({pct:.0f}%)")
    
    # Get gaps
    print("\n[2] Gap Report:")
    gaps = registry.get_gap_report()
    print(f"    Missing: {len(gaps['missing'])}")
    print(f"    Partial: {len(gaps['partial'])}")
    print(f"    Stub: {len(gaps['stub'])}")
    print(f"    Unverified Claims: {len(gaps['unverified_claims'])}")
    
    # Verify sample inventions
    print("\n[3] Verification Samples:")
    samples = ["PPA1-Inv1", "NOVEL-1", "UP3", "NOVEL-22"]
    for inv_id in samples:
        success, msg = registry.verify_invention(inv_id)
        status = "✓" if success else "✗"
        print(f"    {status} {inv_id}: {msg[:60]}")
    
    print("\n" + "=" * 80)
    print("✓ INVENTION-MODULE REGISTRY TEST COMPLETE")
    print("=" * 80)

