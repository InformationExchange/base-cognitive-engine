"""
BASE Conversational Orchestrator

Implements the production-ready conversational loop:
1. Pattern analysis (fast, free)
2. Confidence check
3. If low: Generate clarifying question OR call LLM
4. If user responds: Use LLM to incorporate context
5. Return enriched analysis

This is how BASE works in production when people use the APIs.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
import time
import os

from core.smart_gate import SmartGate, AnalysisMode, RoutingDecision
from core.query_analyzer import QueryAnalyzer, QueryAnalysisResult, QueryRisk
from research.theory_of_mind import TheoryOfMindModule
from research.neurosymbolic import NeuroSymbolicModule
from research.world_models import WorldModelsModule
from research.creative_reasoning import CreativeReasoningModule
from research.llm_helper import LLMHelper


class ConversationState(Enum):
    """State of the conversational analysis."""
    INITIAL = "initial"
    PATTERN_COMPLETE = "pattern_complete"
    AWAITING_USER = "awaiting_user"
    LLM_PROCESSING = "llm_processing"
    COMPLETE = "complete"


@dataclass
class ClarifyingQuestion:
    """A question to ask the user for clarification."""
    question: str
    reason: str
    module: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationResult:
    """Result from conversational analysis."""
    state: ConversationState
    query_analysis: Optional[Dict[str, Any]] = None  # NEW: Query analysis
    query_risk: str = "none"  # NEW: Query risk level
    pattern_result: Optional[Dict[str, Any]] = None
    llm_result: Optional[Dict[str, Any]] = None
    clarifying_question: Optional[ClarifyingQuestion] = None
    final_result: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    total_time_ms: int = 0
    total_cost: float = 0.0
    conversation_history: List[Dict[str, str]] = field(default_factory=list)


class ConversationalOrchestrator:
    """
    Production-ready orchestrator with conversational capabilities.
    
    Usage:
        orchestrator = ConversationalOrchestrator()
        
        # Initial analysis
        result = await orchestrator.analyze("Some text to analyze")
        
        if result.state == ConversationState.AWAITING_USER:
            # Ask user the clarifying question
            print(result.clarifying_question.question)
            
            # User responds
            result = await orchestrator.continue_with_response("User's answer")
        
        # Final result
        print(result.final_result)
    """
    
    CONFIDENCE_THRESHOLD = 0.4  # Below this, ask for clarification
    
    def __init__(self, api_key: str = None):
        """Initialize orchestrator with all modules."""
        self.gate = SmartGate()
        self.query_analyzer = QueryAnalyzer()  # NEW: Query analyzer
        self.tom = TheoryOfMindModule()
        self.ns = NeuroSymbolicModule()
        self.wm = WorldModelsModule()
        self.cr = CreativeReasoningModule()
        
        api_key = api_key or os.environ.get('GROK_API_KEY')
        self.llm = LLMHelper(api_key=api_key) if api_key else None
        
        # Conversation state
        self._current_text = ""
        self._current_query = ""
        self._pattern_result = None
        self._query_analysis = None  # NEW
        self._conversation_history = []
        
        # Stats
        self.stats = {
            "pattern_only": 0,
            "llm_fallback": 0,
            "user_clarification": 0,
            "query_risks_detected": 0,  # NEW
            "total_cost": 0.0
        }
        
        # Learning state
        self._outcomes: List[Dict] = []
        self._feedback: List[Dict] = []
        self._domain_adjustments: Dict[str, float] = {}
    
    # ===== Learning Interface Methods =====
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        """Record conversation outcome for learning."""
        self._outcomes.append(outcome)
    
    def record_feedback(self, feedback: Dict) -> None:
        """Record feedback on conversation handling."""
        self._feedback.append(feedback)
        domain = feedback.get('domain', 'general')
        quality = feedback.get('quality', 0.5)
        adjustment = 0.05 if quality > 0.7 else -0.05 if quality < 0.3 else 0
        self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) + adjustment
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt confidence thresholds."""
        pass
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get domain adjustment."""
        return self._domain_adjustments.get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics."""
        return {
            'stats': dict(self.stats),
            'domain_adjustments': dict(self._domain_adjustments),
            'outcomes_recorded': len(self._outcomes),
            'feedback_recorded': len(self._feedback)
        }
    
    async def analyze(self, 
                     query: str,
                     text: str,
                     mode: AnalysisMode = AnalysisMode.STANDARD,
                     auto_llm: bool = True) -> ConversationResult:
        """
        Analyze text with conversational capabilities.
        
        Args:
            query: What to analyze
            text: Text to analyze
            mode: Analysis mode (QUICK, STANDARD, DEEP)
            auto_llm: If True, automatically call LLM on low confidence.
                      If False, return clarifying question for user.
        """
        start_time = time.time()
        self._current_text = text
        self._current_query = query
        self._conversation_history = []
        
        # Step 0: NEW - Analyze the QUERY first
        query_analysis = self.query_analyzer.analyze(query)
        self._query_analysis = query_analysis
        query_analysis_dict = query_analysis.to_dict()
        
        # Track risky queries
        if query_analysis.risk_level in [QueryRisk.HIGH, QueryRisk.CRITICAL]:
            self.stats["query_risks_detected"] += 1
        
        # Step 1: Smart gate routing
        gate_result = self.gate.analyze(query, text, mode)
        
        # Step 2: Pattern analysis on RESPONSE (always run - fast and free)
        pattern_result = self._run_pattern_analysis(text)
        pattern_result["query_analysis"] = query_analysis_dict  # Include query analysis
        self._pattern_result = pattern_result
        
        pattern_time = int((time.time() - start_time) * 1000)
        
        # Step 3: Calculate confidence
        confidence = self._calculate_confidence(pattern_result)
        
        # Step 4: Determine next step based on confidence
        if mode == AnalysisMode.QUICK:
            # Quick mode: always return pattern result
            self.stats["pattern_only"] += 1
            return ConversationResult(
                state=ConversationState.COMPLETE,
                query_analysis=query_analysis_dict,
                query_risk=query_analysis.risk_level.value,
                pattern_result=pattern_result,
                final_result=pattern_result,
                confidence=confidence,
                total_time_ms=pattern_time,
                total_cost=0.0
            )
        
        if confidence >= self.CONFIDENCE_THRESHOLD:
            # High confidence: return pattern result
            self.stats["pattern_only"] += 1
            return ConversationResult(
                state=ConversationState.COMPLETE,
                query_analysis=query_analysis_dict,
                query_risk=query_analysis.risk_level.value,
                pattern_result=pattern_result,
                final_result=pattern_result,
                confidence=confidence,
                total_time_ms=pattern_time,
                total_cost=0.0
            )
        
        # Low confidence: need clarification or LLM
        if auto_llm and self.llm:
            # Auto LLM mode: call LLM directly
            llm_result = await self._run_llm_analysis(text, pattern_result)
            llm_time = int((time.time() - start_time) * 1000) - pattern_time
            
            combined_confidence = self._combine_confidence(confidence, llm_result)
            final_result = self._merge_results(pattern_result, llm_result)
            
            self.stats["llm_fallback"] += 1
            self.stats["total_cost"] += 0.01
            
            return ConversationResult(
                state=ConversationState.COMPLETE,
                query_analysis=query_analysis_dict,
                query_risk=query_analysis.risk_level.value,
                pattern_result=pattern_result,
                llm_result=llm_result,
                final_result=final_result,
                confidence=combined_confidence,
                total_time_ms=int((time.time() - start_time) * 1000),
                total_cost=0.01
            )
        
        else:
            # Conversational mode: ask user for clarification
            question = await self._generate_clarifying_question(text, pattern_result)
            
            self._conversation_history.append({
                "role": "system",
                "content": f"Analyzing: {text}"
            })
            self._conversation_history.append({
                "role": "assistant",
                "content": question.question
            })
            
            return ConversationResult(
                state=ConversationState.AWAITING_USER,
                query_analysis=query_analysis_dict,
                query_risk=query_analysis.risk_level.value,
                pattern_result=pattern_result,
                clarifying_question=question,
                confidence=confidence,
                total_time_ms=pattern_time,
                conversation_history=self._conversation_history.copy()
            )
    
    async def continue_with_response(self, user_response: str) -> ConversationResult:
        """
        Continue analysis with user's response to clarifying question.
        
        Args:
            user_response: User's answer to the clarifying question
        """
        start_time = time.time()
        
        if not self._pattern_result:
            raise ValueError("No analysis in progress. Call analyze() first.")
        
        self._conversation_history.append({
            "role": "user",
            "content": user_response
        })
        
        # Use LLM to incorporate user response
        if self.llm:
            enriched_context = f"{self._current_text}\n\nUser clarification: {user_response}"
            llm_result = await self._run_llm_with_context(
                self._current_text, 
                user_response,
                self._pattern_result
            )
            
            combined_confidence = 0.85  # User-verified = high confidence
            final_result = self._merge_results(self._pattern_result, llm_result)
            final_result["user_verified"] = True
            final_result["user_context"] = user_response
            
            self._conversation_history.append({
                "role": "assistant",
                "content": f"Analysis updated with your input. Confidence: HIGH"
            })
            
            self.stats["user_clarification"] += 1
            self.stats["total_cost"] += 0.01
            
            return ConversationResult(
                state=ConversationState.COMPLETE,
                pattern_result=self._pattern_result,
                llm_result=llm_result,
                final_result=final_result,
                confidence=combined_confidence,
                total_time_ms=int((time.time() - start_time) * 1000),
                total_cost=0.01,
                conversation_history=self._conversation_history.copy()
            )
        else:
            # No LLM available - use pattern result with user context
            final_result = self._pattern_result.copy()
            final_result["user_context"] = user_response
            final_result["note"] = "No LLM available for verification"
            
            return ConversationResult(
                state=ConversationState.COMPLETE,
                pattern_result=self._pattern_result,
                final_result=final_result,
                confidence=0.6,
                total_time_ms=int((time.time() - start_time) * 1000),
                total_cost=0.0,
                conversation_history=self._conversation_history.copy()
            )
    
    def _run_pattern_analysis(self, text: str) -> Dict[str, Any]:
        """Run all pattern modules."""
        results = {}
        
        # Theory of Mind
        tom_result = self.tom.analyze("analyze", text)
        results["theory_of_mind"] = {
            "mental_states": len(tom_result.inferred_states),
            "manipulation_risk": tom_result.manipulation.risk_level.value,
            "perspectives": len(tom_result.perspectives),
            "confidence": tom_result.theory_of_mind_score
        }
        
        # Neuro-Symbolic
        ns_result = self.ns.verify("verify", text)
        results["logic"] = {
            "validity": ns_result.validity_score,
            "fallacies": [f.fallacy_type.value for f in ns_result.fallacies_detected],
            "consistent": ns_result.consistency.is_consistent,
            "confidence": ns_result.validity_score
        }
        
        # World Models
        wm_result = self.wm.analyze("analyze", text)
        results["causation"] = {
            "causal_relationships": len(wm_result.causal_relationships),
            "predictions": len(wm_result.predictions),
            "counterfactuals": len(wm_result.counterfactuals),
            "confidence": wm_result.model_completeness
        }
        
        # Creative Reasoning
        cr_result = self.cr.analyze("analyze", text)
        results["creativity"] = {
            "ideas": cr_result.idea_count,
            "analogies": len(cr_result.analogies),
            "originality": cr_result.creativity_metrics.originality_score,
            "confidence": cr_result.creativity_metrics.originality_score
        }
        
        return results
    
    def _calculate_confidence(self, pattern_result: Dict[str, Any]) -> float:
        """Calculate overall confidence from pattern results."""
        confidences = []
        for module_name, result in pattern_result.items():
            if "confidence" in result:
                confidences.append(result["confidence"])
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    async def _run_llm_analysis(self, 
                                text: str, 
                                pattern_result: Dict[str, Any]) -> Dict[str, Any]:
        """Run LLM analysis for areas where pattern was uncertain."""
        results = {}
        
        # Check if causation needs LLM
        if pattern_result.get("causation", {}).get("confidence", 1.0) < 0.3:
            causal = await self.llm.verify_causal_claim("implicit", "implicit", text)
            if causal.success:
                results["causal_analysis"] = causal.result
        
        # Check if mental states need LLM
        if pattern_result.get("theory_of_mind", {}).get("confidence", 1.0) < 0.3:
            mental = await self.llm.extract_mental_states(text)
            if mental.success:
                results["mental_states"] = mental.result
        
        return results
    
    async def _run_llm_with_context(self,
                                    original_text: str,
                                    user_response: str,
                                    pattern_result: Dict[str, Any]) -> Dict[str, Any]:
        """Run LLM analysis with user-provided context."""
        enriched = f"{original_text}\n\nAdditional context from user: {user_response}"
        return await self._run_llm_analysis(enriched, pattern_result)
    
    async def _generate_clarifying_question(self,
                                            text: str,
                                            pattern_result: Dict[str, Any]) -> ClarifyingQuestion:
        """Generate a clarifying question based on uncertain areas."""
        
        # Find the lowest confidence module
        lowest_conf = 1.0
        lowest_module = "general"
        
        for module, result in pattern_result.items():
            conf = result.get("confidence", 1.0)
            if conf < lowest_conf:
                lowest_conf = conf
                lowest_module = module
        
        # Generate question based on module
        if lowest_module == "causation":
            if self.llm:
                q_result = await self.llm._call_llm(
                    "You are a causation analyst. Generate ONE specific clarifying question.",
                    f"Text: {text}\n\nI cannot determine causal relationships. Ask ONE question to clarify."
                )
                if q_result.success:
                    question = q_result.result
                else:
                    question = "Can you explain what caused this outcome?"
            else:
                question = "Can you explain what caused this outcome?"
            
            return ClarifyingQuestion(
                question=question,
                reason="Cannot determine causal relationships from text",
                module="world_models",
                context={"original_text": text}
            )
        
        elif lowest_module == "theory_of_mind":
            return ClarifyingQuestion(
                question="What emotions or intentions are involved in this situation?",
                reason="Cannot clearly identify mental states",
                module="theory_of_mind",
                context={"original_text": text}
            )
        
        else:
            return ClarifyingQuestion(
                question="Can you provide more context about this situation?",
                reason="General low confidence in analysis",
                module=lowest_module,
                context={"original_text": text}
            )
    
    def _combine_confidence(self, 
                           pattern_conf: float, 
                           llm_result: Dict[str, Any]) -> float:
        """Combine pattern and LLM confidence."""
        if llm_result:
            return 0.3 * pattern_conf + 0.7 * 0.85
        return pattern_conf
    
    def _merge_results(self,
                      pattern_result: Dict[str, Any],
                      llm_result: Dict[str, Any]) -> Dict[str, Any]:
        """Merge pattern and LLM results."""
        merged = pattern_result.copy()
        if llm_result:
            merged["llm_enrichment"] = llm_result
        return merged
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        total = self.stats["pattern_only"] + self.stats["llm_fallback"] + self.stats["user_clarification"]
        return {
            **self.stats,
            "total_calls": total,
            "llm_rate": (self.stats["llm_fallback"] + self.stats["user_clarification"]) / total if total > 0 else 0
        }

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


# Quick test
async def test_conversational():
    """Test the conversational orchestrator."""
    orchestrator = ConversationalOrchestrator()
    
    print("=" * 60)
    print("CONVERSATIONAL ORCHESTRATOR TEST")
    print("=" * 60)
    
    # Test 1: High confidence (pattern sufficient)
    print("\n1. High confidence case:")
    result = await orchestrator.analyze(
        "analyze",
        "Smoking causes lung cancer. This leads to higher healthcare costs.",
        auto_llm=False
    )
    print(f"   State: {result.state.value}")
    print(f"   Confidence: {result.confidence:.2f}")
    print(f"   Causal found: {result.pattern_result['causation']['causal_relationships']}")
    
    # Test 2: Low confidence with auto LLM
    print("\n2. Low confidence + auto LLM:")
    result = await orchestrator.analyze(
        "analyze",
        "The company struggled. Then Sarah joined. Profits improved.",
        auto_llm=True
    )
    print(f"   State: {result.state.value}")
    print(f"   Confidence: {result.confidence:.2f}")
    print(f"   Cost: ${result.total_cost:.2f}")
    
    # Test 3: Low confidence with user clarification
    print("\n3. Low confidence + user clarification:")
    result = await orchestrator.analyze(
        "analyze",
        "Things changed after the meeting.",
        auto_llm=False
    )
    print(f"   State: {result.state.value}")
    if result.clarifying_question:
        print(f"   Question: {result.clarifying_question.question}")
        
        # Simulate user response
        result = await orchestrator.continue_with_response(
            "The team agreed to new deadlines which improved productivity."
        )
        print(f"   After response - State: {result.state.value}")
        print(f"   Final confidence: {result.confidence:.2f}")
    
    print("\n" + "=" * 60)
    print("Stats:", orchestrator.get_stats())


if __name__ == "__main__":
    asyncio.run(test_conversational())

