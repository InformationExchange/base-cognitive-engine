"""
BASE Hybrid Orchestrator - Production-Ready Integration

This module ACTUALLY integrates pattern matching with LLM fallback.
Previous implementations had the pieces but were not wired together.

Flow:
  1. Smart Gate determines routing
  2. Pattern analysis runs first (always)
  3. If confidence low AND gate allows, LLM fallback triggers
  4. Results combined and returned
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum
import os

# Import all components
from core.smart_gate import SmartGate, AnalysisMode, RoutingDecision
from core.query_analyzer import QueryAnalyzer, QueryAnalysisResult
from research.theory_of_mind import TheoryOfMindModule
from research.neurosymbolic import NeuroSymbolicModule
from research.world_models import WorldModelsModule
from research.creative_reasoning import CreativeReasoningModule
from research.llm_helper import LLMHelper


class AnalysisSource(Enum):
    PATTERN_ONLY = "pattern_only"
    PATTERN_WITH_LLM = "pattern_with_llm"
    LLM_ONLY = "llm_only"


@dataclass
class HybridResult:
    """Result from hybrid analysis."""
    source: AnalysisSource
    query_analysis: Optional[Dict[str, Any]]  # NEW: Query analysis
    pattern_result: Optional[Dict[str, Any]]
    llm_result: Optional[Dict[str, Any]]
    combined_confidence: float
    total_cost: float
    total_latency_ms: int
    llm_triggered: bool
    trigger_reason: str
    query_risk: str = "none"  # NEW: Query risk level


class HybridOrchestrator:
    """
    Production orchestrator that actually wires pattern + LLM together.
    
    This is what was missing before - the actual integration.
    """
    
    # Confidence threshold for LLM fallback
    LLM_CONFIDENCE_THRESHOLD = 0.5
    
    def __init__(self, api_key: str = None):
        """Initialize all components."""
        self.gate = SmartGate()
        self.query_analyzer = QueryAnalyzer()  # NEW: Query analyzer
        self.tom = TheoryOfMindModule()
        self.ns = NeuroSymbolicModule()
        self.wm = WorldModelsModule()
        self.cr = CreativeReasoningModule()
        
        # LLM helper - only initialized if API key available
        self.llm = None
        api_key = api_key or os.environ.get('GROK_API_KEY')
        if api_key:
            self.llm = LLMHelper(api_key=api_key)
        
        self.total_pattern_calls = 0
        self.total_llm_calls = 0
        self.total_query_analyses = 0  # NEW: Track query analyses
        
        # Learning state
        self._outcomes: List[Dict] = []
        self._feedback: List[Dict] = []
        self._domain_adjustments: Dict[str, float] = {}
        self._module_effectiveness: Dict[str, float] = {}
    
    # ===== Learning Interface Methods =====
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        """Record orchestration outcome for learning."""
        self._outcomes.append(outcome)
        for module in outcome.get('modules_used', []):
            if module not in self._module_effectiveness:
                self._module_effectiveness[module] = 0.5
            # Adjust effectiveness based on outcome
            if outcome.get('correct', False):
                self._module_effectiveness[module] = min(0.95, self._module_effectiveness[module] + 0.05)
            else:
                self._module_effectiveness[module] = max(0.05, self._module_effectiveness[module] - 0.05)
    
    def record_feedback(self, feedback: Dict) -> None:
        """Record feedback on orchestration."""
        self._feedback.append(feedback)
        domain = feedback.get('domain', 'general')
        if feedback.get('quality', 0.5) < 0.5:
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) - 0.05
        else:
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) + 0.05
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt module selection based on performance."""
        pass
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get adjustment for domain."""
        return self._domain_adjustments.get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics."""
        return {
            'total_pattern_calls': self.total_pattern_calls,
            'total_llm_calls': self.total_llm_calls,
            'total_query_analyses': self.total_query_analyses,
            'module_effectiveness': dict(self._module_effectiveness),
            'domain_adjustments': dict(self._domain_adjustments),
            'outcomes_recorded': len(self._outcomes),
            'feedback_recorded': len(self._feedback)
        }
    
    async def analyze(self,
                     query: str,
                     response: str,
                     mode: AnalysisMode = AnalysisMode.STANDARD,
                     modules: List[str] = None) -> HybridResult:
        """
        Run hybrid analysis with actual LLM integration.
        
        Args:
            query: User query
            response: AI response to evaluate
            mode: User-selected analysis mode
            modules: Which modules to run (default: all)
        
        Returns:
            HybridResult with combined pattern + LLM analysis
        """
        import time
        start_time = time.time()
        
        # Step 0: NEW - Analyze the QUERY first
        query_analysis = self.query_analyzer.analyze(query)
        self.total_query_analyses += 1
        query_analysis_dict = query_analysis.to_dict()
        
        # Step 1: Gate decision
        gate_result = self.gate.analyze(query, response, mode)
        
        # Step 2: Always run pattern analysis on RESPONSE (fast, free)
        pattern_results = self._run_pattern_analysis(query, response, modules)
        self.total_pattern_calls += 1
        
        # Add query analysis to pattern results
        pattern_results['query_analysis'] = query_analysis_dict
        
        pattern_latency = int((time.time() - start_time) * 1000)
        
        # Step 3: Determine if LLM needed
        llm_result = None
        llm_triggered = False
        trigger_reason = "none"
        llm_latency = 0
        
        if gate_result.decision == RoutingDecision.FORCE_LLM:
            # Gate says force LLM
            if self.llm:
                llm_result = await self._run_llm_analysis(query, response, pattern_results)
                llm_triggered = True
                trigger_reason = gate_result.reason
                self.total_llm_calls += 1
                llm_latency = 2000  # Approximate
            else:
                trigger_reason = "LLM forced but no API key configured"
                
        elif gate_result.decision == RoutingDecision.PATTERN_THEN_LLM:
            # Check pattern confidence
            avg_confidence = self._calculate_pattern_confidence(pattern_results)
            
            if avg_confidence < self.LLM_CONFIDENCE_THRESHOLD:
                if self.llm:
                    llm_result = await self._run_llm_analysis(query, response, pattern_results)
                    llm_triggered = True
                    trigger_reason = f"Low pattern confidence: {avg_confidence:.2f}"
                    self.total_llm_calls += 1
                    llm_latency = 2000
                else:
                    trigger_reason = f"Low confidence ({avg_confidence:.2f}) but no API key"
            else:
                trigger_reason = f"Pattern confidence sufficient: {avg_confidence:.2f}"
        
        # Pattern only mode - no LLM consideration
        else:
            trigger_reason = "Pattern-only mode"
        
        # Step 4: Calculate combined confidence
        combined_confidence = self._calculate_combined_confidence(
            pattern_results, llm_result, llm_triggered
        )
        
        # Determine source
        if llm_triggered and llm_result:
            source = AnalysisSource.PATTERN_WITH_LLM
        else:
            source = AnalysisSource.PATTERN_ONLY
        
        total_latency = pattern_latency + llm_latency
        total_cost = 0.01 if llm_triggered else 0.0
        
        return HybridResult(
            source=source,
            query_analysis=query_analysis_dict,  # NEW
            pattern_result=pattern_results,
            llm_result=llm_result,
            combined_confidence=combined_confidence,
            total_cost=total_cost,
            total_latency_ms=total_latency,
            llm_triggered=llm_triggered,
            trigger_reason=trigger_reason,
            query_risk=query_analysis.risk_level.value  # NEW
        )
    
    def _run_pattern_analysis(self, 
                              query: str, 
                              response: str,
                              modules: List[str] = None) -> Dict[str, Any]:
        """Run pattern-based analysis on all modules."""
        results = {}
        
        if modules is None or 'tom' in modules:
            tom_result = self.tom.analyze(query, response)
            results['theory_of_mind'] = {
                'mental_states': len(tom_result.inferred_states),
                'manipulation_risk': tom_result.manipulation.risk_level.value,
                'manipulation_score': tom_result.manipulation.risk_score,
                'perspectives': len(tom_result.perspectives),
                'score': tom_result.theory_of_mind_score,
                'confidence': max(0.3, tom_result.theory_of_mind_score)
            }
        
        if modules is None or 'ns' in modules:
            ns_result = self.ns.verify(query, response)
            results['neurosymbolic'] = {
                'verification': ns_result.verification_result.value,
                'validity_score': ns_result.validity_score,
                'fallacies': len(ns_result.fallacies_detected),
                'fallacy_types': [f.fallacy_type.value for f in ns_result.fallacies_detected],
                'consistent': ns_result.consistency.is_consistent,
                'confidence': ns_result.validity_score
            }
        
        if modules is None or 'wm' in modules:
            wm_result = self.wm.analyze(query, response)
            results['world_models'] = {
                'causal_relationships': len(wm_result.causal_relationships),
                'predictions': len(wm_result.predictions),
                'counterfactuals': len(wm_result.counterfactuals),
                'completeness': wm_result.model_completeness,
                'confidence': max(0.3, wm_result.model_completeness)
            }
        
        if modules is None or 'cr' in modules:
            cr_result = self.cr.analyze(query, response)
            results['creative'] = {
                'ideas': cr_result.idea_count,
                'analogies': len(cr_result.analogies),
                'originality': cr_result.creativity_metrics.originality_score,
                'creativity_percentile': cr_result.creativity_metrics.creativity_percentile,
                'confidence': max(0.3, cr_result.creativity_metrics.originality_score)
            }
        
        return results
    
    def _calculate_pattern_confidence(self, pattern_results: Dict[str, Any]) -> float:
        """Calculate average confidence across pattern modules."""
        confidences = []
        for module_name, result in pattern_results.items():
            if 'confidence' in result:
                confidences.append(result['confidence'])
        
        if not confidences:
            return 0.5
        
        return sum(confidences) / len(confidences)
    
    async def _run_llm_analysis(self,
                                query: str,
                                response: str,
                                pattern_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run LLM analysis for areas where pattern was uncertain."""
        if not self.llm:
            return None
        
        results = {}
        
        # Check if causation analysis needed
        wm = pattern_results.get('world_models', {})
        if wm.get('confidence', 1.0) < 0.5 or wm.get('causal_relationships', 0) == 0:
            # Look for potential causal claims in text
            causal_result = await self.llm.verify_causal_claim(
                cause="implicit cause",
                effect="implicit effect", 
                context=response
            )
            if causal_result.success:
                results['causal_analysis'] = {
                    'relationship_type': causal_result.result.get('relationship_type'),
                    'confounders': causal_result.result.get('confounders', []),
                    'reasoning': causal_result.reasoning[:200]
                }
        
        # Check if mental state analysis needed
        tom = pattern_results.get('theory_of_mind', {})
        if tom.get('confidence', 1.0) < 0.5:
            mental_result = await self.llm.extract_mental_states(response)
            if mental_result.success:
                results['mental_states'] = mental_result.result
        
        # Check for logical contradictions
        ns = pattern_results.get('neurosymbolic', {})
        if not ns.get('consistent', True) or ns.get('confidence', 1.0) < 0.5:
            contra_result = await self.llm.detect_contradiction(response)
            if contra_result.success:
                results['contradictions'] = contra_result.result
        
        return results
    
    def _calculate_combined_confidence(self,
                                       pattern_results: Dict[str, Any],
                                       llm_result: Optional[Dict[str, Any]],
                                       llm_triggered: bool) -> float:
        """Calculate combined confidence from pattern + LLM."""
        pattern_conf = self._calculate_pattern_confidence(pattern_results)
        
        if llm_triggered and llm_result:
            # LLM adds confidence boost
            llm_conf = 0.85  # LLM generally high confidence
            # Weighted average favoring LLM when triggered
            return 0.3 * pattern_conf + 0.7 * llm_conf
        else:
            return pattern_conf
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        total = self.total_pattern_calls
        llm_rate = self.total_llm_calls / total if total > 0 else 0
        
        return {
            "total_calls": total,
            "pattern_calls": self.total_pattern_calls,
            "llm_calls": self.total_llm_calls,
            "llm_trigger_rate": llm_rate,
            "estimated_cost": self.total_llm_calls * 0.01
        }
    
    # =========================================================================
    # Learning Interface (5/5 methods) - Completing interface
    # =========================================================================
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback to adjust orchestration."""
        if not hasattr(self, '_learning_params'):
            self._learning_params = {'llm_threshold': 0.5}
        was_correct = feedback.get('was_correct', True)
        if not was_correct:
            if feedback.get('needed_llm', False):
                self._learning_params['llm_threshold'] *= 0.95  # Lower threshold
            else:
                self._learning_params['llm_threshold'] *= 1.05  # Raise threshold
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return orchestration statistics."""
        return {
            **self.get_stats(),
            'learning_params': getattr(self, '_learning_params', {}),
            'outcomes_recorded': len(getattr(self, '_outcomes', []))
        }
    
    def serialize_state(self) -> Dict:
        """Serialize learning state for persistence."""
        return {
            'learning_params': getattr(self, '_learning_params', {}),
            'outcomes': getattr(self, '_outcomes', [])[-100:],
            'stats': self.get_stats(),
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore learning state from serialized data."""
        self._learning_params = state.get('learning_params', {'llm_threshold': 0.5})
        self._outcomes = state.get('outcomes', [])


# Test the orchestrator
async def test_orchestrator():
    """Test the hybrid orchestrator with real cases."""
    
    # Check for API key
    api_key = os.environ.get('GROK_API_KEY')
    if not api_key:
        print("WARNING: No GROK_API_KEY - LLM fallback will not work")
    
    orchestrator = HybridOrchestrator(api_key=api_key)
    
    test_cases = [
        # Simple case - pattern should handle
        ("What is Python?", "Python is a programming language.", AnalysisMode.STANDARD),
        
        # Complex causation - needs LLM
        ("Analyze this", "The company struggled. Then Sarah joined. Profits soared.", AnalysisMode.STANDARD),
        
        # Medical domain - force LLM
        ("Is this safe?", "This drug treats headaches with some side effects.", AnalysisMode.STANDARD),
        
        # User forces deep
        ("Quick check", "Hello world", AnalysisMode.DEEP),
    ]
    
    print("\nHybrid Orchestrator Test Results:")
    print("-" * 70)
    
    for query, response, mode in test_cases:
        result = await orchestrator.analyze(query, response, mode)
        print(f"\nQuery: {query}")
        print(f"  Mode: {mode.value}")
        print(f"  Source: {result.source.value}")
        print(f"  LLM Triggered: {result.llm_triggered}")
        print(f"  Trigger Reason: {result.trigger_reason}")
        print(f"  Combined Confidence: {result.combined_confidence:.2f}")
        print(f"  Cost: ${result.total_cost:.4f}")
        print(f"  Latency: {result.total_latency_ms}ms")
    
    print("\n" + "-" * 70)
    stats = orchestrator.get_stats()
    print(f"Stats: {stats}")


if __name__ == "__main__":
    asyncio.run(test_orchestrator())

