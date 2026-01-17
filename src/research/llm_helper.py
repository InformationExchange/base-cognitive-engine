"""
BAIS LLM Helper for R&D Modules
Provides LLM-assisted semantic analysis when pattern matching is insufficient

This module enables R&D modules to:
1. Fall back to LLM for complex semantic understanding
2. Verify pattern-matched results with deeper analysis
3. Handle edge cases that patterns miss

PHASE 5 UPDATE: Now uses LLMRegistry for unified LLM access
- Supports multiple providers (Grok, OpenAI, Anthropic, etc.)
- Secure API key management
- Provider switching and fallback

Patent Claims: R&D Integration Enhancement
Proprietary IP - 100% owned by Invitas Inc.
"""

import os
import sys
import json
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Try to import LLM Registry (Phase 5 consolidation)
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.llm_registry import LLMRegistry, LLMProvider
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False


@dataclass
class LLMAnalysisResult:
    """Result from LLM-assisted analysis."""
    success: bool
    result: Dict[str, Any]
    confidence: float
    reasoning: str
    error: Optional[str] = None


class LLMHelper:
    """
    LLM Helper for enhanced R&D module capabilities.
    
    PHASE 5: Now uses LLMRegistry for unified LLM access.
    Falls back to direct API calls if registry not available.
    
    Provides async methods to call LLM for:
    - Mental state extraction (Theory of Mind)
    - Contradiction detection (Neuro-Symbolic)
    - Causal verification (World Models)
    - Originality scoring (Creative Reasoning)
    """
    
    # Grok API endpoint (fallback if registry not available)
    API_URL = "https://api.x.ai/v1/chat/completions"
    
    def __init__(self, api_key: str = None, model: str = None, use_registry: bool = True):
        """
        Initialize LLM Helper.
        
        Args:
            api_key: Grok API key (will use environment variable if not provided)
            model: Model to use (auto-discovers latest if not specified)
            use_registry: Whether to use LLMRegistry (Phase 5)
        """
        self.api_key = api_key or os.environ.get("GROK_API_KEY")
        self._timeout = aiohttp.ClientTimeout(total=30)
        
        # Phase 5: Use registry if available
        self.registry = None
        if use_registry and REGISTRY_AVAILABLE:
            try:
                self.registry = LLMRegistry()
                print(f"[LLMHelper] Using LLMRegistry (Phase 5 consolidation)")
            except Exception as e:
                print(f"[LLMHelper] Registry init failed, using direct API: {e}")
        
        self.model = model or self._get_default_model()
    
    def _get_default_model(self) -> str:
        """Get the default LLM model using centralized model provider."""
        try:
            from core.model_provider import get_best_reasoning_model
            return get_best_reasoning_model("grok")
        except ImportError:
            return "grok-4-1-fast-reasoning"
    
    async def _call_llm(self, system_prompt: str, user_prompt: str) -> LLMAnalysisResult:
        """
        Make async call to LLM API.
        
        Phase 5: Uses LLMRegistry if available for unified access.
        Falls back to direct API calls otherwise.
        """
        # Phase 5: Try registry first
        if self.registry:
            return await self._call_via_registry(system_prompt, user_prompt)
        
        # Fallback: Direct API call
        return await self._call_direct(system_prompt, user_prompt)
    
    async def _call_via_registry(self, system_prompt: str, user_prompt: str) -> LLMAnalysisResult:
        """Call LLM via registry (Phase 5)."""
        try:
            # Use registry's call method
            response = await self.registry.call_llm_async(
                provider=LLMProvider.GROK,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            if response.get('success'):
                content = response.get('content', '')
                return self._parse_response(content)
            else:
                return LLMAnalysisResult(
                    success=False,
                    result={},
                    confidence=0.0,
                    reasoning=f"Registry call failed: {response.get('error')}",
                    error=response.get('error')
                )
        except Exception as e:
            # Fall back to direct call
            return await self._call_direct(system_prompt, user_prompt)
    
    async def _call_direct(self, system_prompt: str, user_prompt: str) -> LLMAnalysisResult:
        """Direct API call (fallback)."""
        if not self.api_key:
            return LLMAnalysisResult(
                success=False,
                result={},
                confidence=0.0,
                reasoning="No API key configured",
                error="API key required for LLM analysis"
            )
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,  # Lower for more consistent analysis
            "max_tokens": 1000
        }
        
        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                async with session.post(self.API_URL, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        return self._parse_response(content)
                    else:
                        return LLMAnalysisResult(
                            success=False,
                            result={},
                            confidence=0.0,
                            reasoning=f"API error: {response.status}",
                            error=await response.text()
                        )
        except asyncio.TimeoutError:
            return LLMAnalysisResult(
                success=False,
                result={},
                confidence=0.0,
                reasoning="Request timed out",
                error="Timeout"
            )
        except Exception as e:
            return LLMAnalysisResult(
                success=False,
                result={},
                confidence=0.0,
                reasoning=f"Error: {str(e)}",
                error=str(e)
            )
    
    def _parse_response(self, content: str) -> LLMAnalysisResult:
        """Parse LLM response into structured result."""
        try:
            # Find JSON in response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(content[json_start:json_end])
            else:
                result = {"raw_response": content}
        except json.JSONDecodeError:
            result = {"raw_response": content}
        
        return LLMAnalysisResult(
            success=True,
            result=result,
            confidence=result.get('confidence', 0.8),
            reasoning=result.get('reasoning', content[:200])
        )
    
    # =========================================================================
    # THEORY OF MIND HELPERS
    # =========================================================================
    
    async def extract_mental_states(self, text: str) -> LLMAnalysisResult:
        """
        Extract mental states from text using LLM.
        Fallback for complex mental state inference.
        """
        system_prompt = """You are a cognitive analyst specializing in Theory of Mind.
        Extract all mental states from the given text.
        
        Mental state types:
        - belief: What someone thinks is true
        - desire: What someone wants
        - intention: What someone plans to do
        - emotion: How someone feels
        - knowledge: What someone knows
        - expectation: What someone expects
        
        Return JSON format:
        {
            "mental_states": [
                {
                    "agent": "who holds the state",
                    "type": "belief|desire|intention|emotion|knowledge|expectation",
                    "content": "what they believe/want/etc",
                    "confidence": 0.0-1.0
                }
            ],
            "confidence": 0.0-1.0,
            "reasoning": "brief explanation"
        }"""
        
        user_prompt = f"""Extract all mental states from this text:

"{text}"

Return the JSON analysis."""
        
        return await self._call_llm(system_prompt, user_prompt)
    
    async def detect_manipulation(self, text: str) -> LLMAnalysisResult:
        """
        Detect manipulation techniques using LLM.
        """
        system_prompt = """You are an expert at detecting manipulation and persuasion techniques.
        Analyze the text for manipulation attempts.
        
        Common techniques:
        - emotional_manipulation: Using fear, urgency, guilt
        - social_proof: "Everyone is doing it"
        - scarcity: "Limited time offer"
        - appeal_to_authority: Using expert claims
        - false_dichotomy: "Either X or Y"
        - gaslighting: Making someone doubt themselves
        
        Return JSON format:
        {
            "manipulation_detected": true/false,
            "risk_level": "none|low|medium|high|critical",
            "techniques": ["list of techniques"],
            "evidence": ["quotes from text"],
            "confidence": 0.0-1.0,
            "reasoning": "brief explanation"
        }"""
        
        user_prompt = f"""Analyze this text for manipulation:

"{text}"

Return the JSON analysis."""
        
        return await self._call_llm(system_prompt, user_prompt)
    
    # =========================================================================
    # NEURO-SYMBOLIC HELPERS
    # =========================================================================
    
    async def detect_contradiction(self, text: str) -> LLMAnalysisResult:
        """
        Detect logical contradictions using LLM.
        Critical for syllogistic reasoning that patterns miss.
        """
        system_prompt = """You are a logic expert analyzing text for contradictions.
        
        Look for:
        1. Direct contradictions: "X is true" and "X is false"
        2. Syllogistic contradictions: "All X are Y" + "Z is X" + "Z is not Y"
        3. Implicit contradictions: Statements that cannot both be true
        
        Return JSON format:
        {
            "has_contradiction": true/false,
            "contradictions": [
                {
                    "statement1": "first contradicting statement",
                    "statement2": "second contradicting statement",
                    "explanation": "why they contradict"
                }
            ],
            "confidence": 0.0-1.0,
            "reasoning": "logical analysis"
        }"""
        
        user_prompt = f"""Check this text for logical contradictions:

"{text}"

Think through the logic step by step. Return JSON analysis."""
        
        return await self._call_llm(system_prompt, user_prompt)
    
    async def detect_fallacies(self, text: str) -> LLMAnalysisResult:
        """
        Detect logical fallacies using LLM.
        """
        system_prompt = """You are a logic expert analyzing arguments for fallacies.
        
        Common fallacies:
        - affirming_consequent: "If A then B. B. Therefore A."
        - false_dichotomy: Only presenting two options when more exist
        - ad_hominem: Attacking the person not the argument
        - hasty_generalization: Generalizing from limited examples
        - circular_reasoning: Using the conclusion as a premise
        - appeal_to_authority: Using authority instead of evidence
        - straw_man: Misrepresenting someone's argument
        
        Return JSON format:
        {
            "fallacies_detected": [
                {
                    "fallacy_type": "name",
                    "evidence": "quote from text",
                    "explanation": "why it's fallacious"
                }
            ],
            "overall_validity": 0.0-1.0,
            "confidence": 0.0-1.0,
            "reasoning": "analysis"
        }"""
        
        user_prompt = f"""Analyze this text for logical fallacies:

"{text}"

Return JSON analysis."""
        
        return await self._call_llm(system_prompt, user_prompt)
    
    # =========================================================================
    # WORLD MODELS HELPERS
    # =========================================================================
    
    async def verify_causal_claim(self, cause: str, effect: str, context: str = "") -> LLMAnalysisResult:
        """
        Verify if a causal claim is reasonable.
        """
        system_prompt = """You are a causal reasoning expert.
        Analyze whether a claimed causal relationship is:
        - Valid: There is a plausible causal mechanism
        - Correlation: They're associated but not causal
        - Spurious: No real relationship
        
        Return JSON format:
        {
            "relationship_type": "causal|correlation|spurious",
            "plausibility": 0.0-1.0,
            "mechanism": "explanation of how cause leads to effect (if causal)",
            "confounders": ["possible confounding variables"],
            "confidence": 0.0-1.0,
            "reasoning": "analysis"
        }"""
        
        user_prompt = f"""Analyze this causal claim:
        
Cause: {cause}
Effect: {effect}
Context: {context if context else "General"}

Is this a valid causal relationship? Return JSON analysis."""
        
        return await self._call_llm(system_prompt, user_prompt)
    
    # =========================================================================
    # CREATIVE REASONING HELPERS
    # =========================================================================
    
    async def assess_originality(self, idea: str) -> LLMAnalysisResult:
        """
        Assess the originality of an idea using LLM.
        """
        system_prompt = """You are a creativity and innovation expert.
        Assess the originality of ideas.
        
        Originality levels:
        - highly_original (0.8-1.0): Novel concept, hasn't been done before
        - moderately_original (0.6-0.8): Creative variation on existing ideas
        - somewhat_original (0.4-0.6): Some novelty, mostly conventional
        - conventional (0.2-0.4): Standard approach, nothing new
        - cliche (0.0-0.2): Overused, generic, been done many times
        
        Return JSON format:
        {
            "originality_level": "highly_original|moderately_original|somewhat_original|conventional|cliche",
            "originality_score": 0.0-1.0,
            "novel_elements": ["what's new about this"],
            "similar_existing": ["similar existing ideas/products"],
            "confidence": 0.0-1.0,
            "reasoning": "assessment"
        }"""
        
        user_prompt = f"""Assess the originality of this idea:

"{idea}"

Compare to known concepts and ideas. Return JSON analysis."""
        
        return await self._call_llm(system_prompt, user_prompt)
    
    async def extract_analogies(self, text: str) -> LLMAnalysisResult:
        """
        Extract analogies and metaphors using LLM.
        """
        system_prompt = """You are an expert at identifying analogies and metaphors.
        Find all comparisons that map one domain to another.
        
        An analogy is when X is compared to Y to explain or illuminate X.
        
        Return JSON format:
        {
            "analogies": [
                {
                    "target": "what is being explained",
                    "source": "what it's compared to",
                    "mapping": "how they relate",
                    "quality": 0.0-1.0
                }
            ],
            "confidence": 0.0-1.0,
            "reasoning": "analysis"
        }"""
        
        user_prompt = f"""Find all analogies and metaphors in this text:

"{text}"

Return JSON analysis."""
        
        return await self._call_llm(system_prompt, user_prompt)
    
    # =========================================================================
    # GENERAL ANALYSIS HELPER (CRITICAL - Used by IntegratedGovernanceEngine)
    # =========================================================================
    
    async def analyze(self, prompt: str) -> Dict[str, Any]:
        """
        General-purpose analysis method for low-confidence fallback.
        
        This method is called by IntegratedGovernanceEngine._llm_fallback_analysis()
        when pattern-based detection has low confidence.
        
        Args:
            prompt: The analysis prompt (typically includes query, response, and detected issues)
        
        Returns:
            Dict with: issues (list), is_safe (bool), assessment (str)
        """
        system_prompt = """You are a safety and quality analyzer for AI responses.
        Analyze the provided content and return a JSON assessment.
        
        Your analysis should identify:
        1. Safety issues (dangerous advice, missing disclaimers)
        2. Quality issues (factual errors, logical fallacies, overconfidence)
        3. Manipulation attempts (leading language, emotional manipulation)
        
        Return ONLY valid JSON in this format:
        {
            "issues": ["list of specific issues found"],
            "is_safe": true/false,
            "assessment": "brief overall assessment",
            "confidence": 0.0-1.0,
            "severity": "low|medium|high|critical"
        }"""
        
        result = await self._call_llm(system_prompt, prompt)
        
        if result.success and isinstance(result.result, dict):
            # Ensure required keys exist
            return {
                "issues": result.result.get("issues", []),
                "is_safe": result.result.get("is_safe", True),
                "assessment": result.result.get("assessment", result.reasoning),
                "confidence": result.result.get("confidence", result.confidence),
                "severity": result.result.get("severity", "low")
            }
        
        # Fallback if LLM call failed
        return {
            "issues": [],
            "is_safe": True,
            "assessment": result.error or "Analysis unavailable",
            "confidence": 0.0,
            "severity": "low"
        }
    
    async def analyze_response_quality(self, 
                                       query: str, 
                                       response: str, 
                                       domain: str = "general") -> Dict[str, Any]:
        """
        Analyze response quality with domain context.
        
        This is an enhanced version of analyze() that includes domain-specific checks.
        Used for detailed response quality assessment.
        
        Args:
            query: The user's original query
            response: The AI response to analyze
            domain: Domain context (medical, financial, legal, general)
        
        Returns:
            Dict with quality metrics and improvement suggestions
        """
        domain_prompts = {
            "medical": "Focus on: medical accuracy, required disclaimers, dosage safety, professional consultation recommendation",
            "financial": "Focus on: financial disclaimers, risk warnings, avoiding guarantees, professional advice recommendation",
            "legal": "Focus on: legal disclaimers, jurisdiction limitations, attorney consultation recommendation",
            "general": "Focus on: general accuracy, appropriate confidence levels, helpful content"
        }
        
        domain_focus = domain_prompts.get(domain, domain_prompts["general"])
        
        system_prompt = f"""You are an expert quality reviewer for {domain} AI responses.
        
        {domain_focus}
        
        Analyze the response and return JSON:
        {{
            "quality_score": 0-100,
            "issues": ["specific issues"],
            "improvements_needed": ["specific improvements"],
            "disclaimers_present": true/false,
            "overconfidence_detected": true/false,
            "is_safe": true/false,
            "recommendation": "accept|improve|reject"
        }}"""
        
        user_prompt = f"""Analyze this {domain} response:

QUERY: {query[:500]}

RESPONSE: {response[:1500]}

Return JSON analysis."""
        
        result = await self._call_llm(system_prompt, user_prompt)
        
        if result.success and isinstance(result.result, dict):
            return result.result
        
        return {
            "quality_score": 50,
            "issues": [],
            "improvements_needed": [],
            "disclaimers_present": False,
            "overconfidence_detected": False,
            "is_safe": True,
            "recommendation": "accept"
        }


# Synchronous wrapper for use in non-async code
def run_llm_analysis(helper: LLMHelper, coro):
    """Run an async LLM analysis function synchronously."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)


# Convenience function for quick LLM checks
async def quick_llm_check(prompt: str, api_key: str = None) -> str:
    """Quick LLM check for simple questions."""
    helper = LLMHelper(api_key=api_key)
    result = await helper._call_llm(
        "You are a helpful assistant. Give brief, direct answers.",
        prompt
    )
    if result.success:
        return result.result.get('raw_response', str(result.result))
    return f"Error: {result.error}"


# Self-test
if __name__ == "__main__":
    import asyncio
    
    async def test():
        helper = LLMHelper()
        
        print("Testing LLM Helper (requires API key)")
        print("=" * 50)
        
        # Test contradiction detection
        text = "All birds can fly. Penguins are birds. Penguins cannot fly."
        print(f"\nText: {text}")
        
        result = await helper.detect_contradiction(text)
        print(f"Success: {result.success}")
        if result.success:
            print(f"Result: {json.dumps(result.result, indent=2)}")
        else:
            print(f"Error: {result.error}")
    
    asyncio.run(test())

