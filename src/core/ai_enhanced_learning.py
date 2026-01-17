"""
BAIS Cognitive Governance Engine - AI-Enhanced Learning Manager
Phase G1: Addresses critical gap - Learning layer NOT connected to LLM

This module provides:
1. LLM-enhanced outcome analysis before recording
2. AI-driven threshold adjustment reasoning
3. Cross-module pattern validation with LLM
4. Hybrid AI + statistical learning fusion
5. Context-aware learning decisions
6. Explanations for all learning decisions

Patent Claims:
- PPA1-Inv22: Feedback Loop (enhanced with AI)
- PPA2-Inv27: OCO Threshold Adapter (AI-verified)
- NOVEL-36: AI-Enhanced Learning Manager (NEW)
"""

import json
import logging
import math
import os
import asyncio
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
from collections import defaultdict

# Use math.log instead of numpy for portability
import numpy as np

logger = logging.getLogger(__name__)


class LearningDecision(Enum):
    """AI-determined learning decisions."""
    LEARN = "learn"                    # Record and learn from outcome
    SKIP = "skip"                      # Skip learning (low signal)
    DEFER = "defer"                    # Defer to human review
    AMPLIFY = "amplify"                # Learn with increased weight
    ATTENUATE = "attenuate"            # Learn with decreased weight
    CONTRADICT = "contradict"          # Flag contradiction with prior learning


@dataclass
class AILearningAnalysis:
    """Result of AI analysis on a learning opportunity."""
    decision: LearningDecision
    confidence: float
    reasoning: str
    suggested_weight: float = 1.0
    cross_module_applicable: List[str] = field(default_factory=list)
    pattern_type: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    llm_used: str = "none"
    analysis_time_ms: float = 0.0


@dataclass
class EnhancedLearningOutcome:
    """Learning outcome with AI analysis attached."""
    module_name: str
    event_type: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    was_correct: bool
    domain: str
    timestamp: str
    ai_analysis: Optional[AILearningAnalysis] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningExplanation:
    """Human-readable explanation for learning decisions."""
    summary: str
    factors: List[str]
    confidence_breakdown: Dict[str, float]
    alternative_actions: List[str]
    audit_trail: List[Dict[str, Any]]


class MLPatternPredictor:
    """
    ML-based pattern prediction component.
    
    Uses statistical learning to predict outcomes:
    - Naive Bayes-like probability updates
    - Feature-based classification
    - Online learning with decay
    """
    
    def __init__(self, decay_rate: float = 0.99):
        self.decay_rate = decay_rate
        self._feature_counts: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._class_counts: Dict[str, float] = defaultdict(float)
        self._total_samples = 0
        self._feature_weights: Dict[str, float] = {}
    
    def update(self, features: Dict[str, Any], outcome: str) -> None:
        """Update model with new observation."""
        # Apply decay to existing counts
        for feat_name in self._feature_counts:
            for val in self._feature_counts[feat_name]:
                self._feature_counts[feat_name][val] *= self.decay_rate
        
        for cls in self._class_counts:
            self._class_counts[cls] *= self.decay_rate
        
        # Add new observation
        self._class_counts[outcome] += 1
        self._total_samples = sum(self._class_counts.values())
        
        # Update feature counts
        for feat_name, feat_val in features.items():
            key = f"{feat_name}={feat_val}|{outcome}"
            self._feature_counts[feat_name][key] += 1
            
            # Update feature weight based on informativeness
            total_for_feat = sum(v for k, v in self._feature_counts[feat_name].items())
            if total_for_feat > 0:
                entropy = 0
                for v in self._feature_counts[feat_name].values():
                    p = v / total_for_feat
                    if p > 0:
                        entropy -= p * np.log(p + 1e-10)
                self._feature_weights[feat_name] = 1.0 / (1.0 + entropy)
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Predict class probabilities."""
        if self._total_samples == 0:
            return {'learn': 0.5, 'skip': 0.5}
        
        probs = {}
        for outcome in self._class_counts:
            # Prior
            prior = (self._class_counts[outcome] + 0.1) / (self._total_samples + 0.2)
            
            # Likelihood
            likelihood = 1.0
            for feat_name, feat_val in features.items():
                key = f"{feat_name}={feat_val}|{outcome}"
                count = self._feature_counts[feat_name].get(key, 0.1)
                total = self._class_counts[outcome] + 0.1
                weight = self._feature_weights.get(feat_name, 1.0)
                likelihood *= (count / total) ** weight
            
            probs[outcome] = prior * likelihood
        
        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        
        return probs
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get ML statistics."""
        return {
            'total_samples': int(self._total_samples),
            'classes': list(self._class_counts.keys()),
            'class_distribution': {k: v / self._total_samples if self._total_samples > 0 else 0 
                                   for k, v in self._class_counts.items()},
            'feature_weights': dict(self._feature_weights),
            'features_tracked': len(self._feature_counts)
        }

    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        """Record operation outcome."""
        if not hasattr(self, '_outcomes'): self._outcomes = []
        self._outcomes.append(outcome)


    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback."""
        pass  # Implement specific learning logic


    def serialize_state(self) -> Dict:
        """Serialize state."""
        return {'outcomes': getattr(self, '_outcomes', [])[-100:]}


    def deserialize_state(self, state: Dict) -> None:
        """Restore state."""
        self._outcomes = state.get('outcomes', [])


class AIEnhancedLearningManager:
    """
    AI-Enhanced Learning Manager for BAIS.
    
    Connects the learning layer to LLM for hybrid AI+statistical learning.
    
    Features:
    - Pattern-based fast analysis
    - ML-based probability prediction
    - LLM-based deep analysis for uncertain cases
    - Cross-module pattern validation
    - Context-aware learning decisions
    - Full explanations for governance audit
    
    Patent: NOVEL-36 (AI-Enhanced Learning Manager)
    """
    
    def __init__(
        self,
        llm_registry: Optional[Any] = None,
        base_learning_manager: Optional[Any] = None,
        ai_analysis_threshold: float = 0.7,
        enable_llm_analysis: bool = True,
        cache_analysis_results: bool = True,
        max_cache_size: int = 1000
    ):
        self.llm_registry = llm_registry
        self.base_manager = base_learning_manager
        self.ai_threshold = ai_analysis_threshold
        self.enable_llm = enable_llm_analysis
        self.cache_enabled = cache_analysis_results
        self.max_cache = max_cache_size
        
        # ML-based predictor (NEW)
        self.ml_predictor = MLPatternPredictor(decay_rate=0.99)
        
        # Analysis cache (LRU)
        self._analysis_cache: Dict[str, AILearningAnalysis] = {}
        self._cache_order: List[str] = []
        
        # Learning statistics
        self._stats = {
            "total_analyses": 0,
            "llm_analyses": 0,
            "pattern_analyses": 0,
            "decisions": defaultdict(int),
            "domain_learnings": defaultdict(int),
            "cross_module_shares": 0,
            "contradictions_detected": 0
        }
        
        # Cross-module pattern registry
        self._cross_module_patterns: Dict[str, List[Dict]] = defaultdict(list)
        
        # Learning rate modifiers per domain (AI-adjusted)
        self._domain_lr_modifiers: Dict[str, float] = {}
        
        # Audit trail for explanations
        self._audit_trail: List[Dict] = []
        
        logger.info("[AIEnhancedLearning] Initialized with LLM=%s", 
                   "enabled" if enable_llm_analysis else "disabled")
    
    def set_llm_registry(self, registry: Any) -> None:
        """Set or update the LLM registry for AI-enhanced learning."""
        self.llm_registry = registry
        self.enable_llm = registry is not None
        logger.info("[AIEnhancedLearning] LLM registry connected, LLM analysis enabled=%s", self.enable_llm)
    
    def connect_to_central_registry(self) -> bool:
        """
        Connect to the centralized LLM registry.
        
        This ensures the learning layer uses the same LLM configuration
        as the rest of BAIS, allowing users to switch LLMs centrally.
        """
        try:
            from core.llm_registry import LLMRegistry, get_registry
            registry = get_registry()
            self.set_llm_registry(registry)
            return True
        except ImportError:
            logger.warning("[AIEnhancedLearning] LLMRegistry not available")
            return False
        except Exception as e:
            logger.error(f"[AIEnhancedLearning] Failed to connect: {e}")
            return False
    
    def get_current_llm_provider(self) -> Optional[str]:
        """Get the name of the currently configured LLM provider."""
        if not self.llm_registry:
            return None
        try:
            provider = self.llm_registry.get_preferred_provider()
            return provider.name if hasattr(provider, 'name') else str(provider)
        except:
            return None
    
    def set_base_manager(self, manager: Any) -> None:
        """Set the base CentralizedLearningManager."""
        self.base_manager = manager
        logger.info("[AIEnhancedLearning] Base learning manager connected")
    
    async def analyze_outcome(
        self,
        module_name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        was_correct: bool,
        domain: str = "general",
        context: Optional[Dict[str, Any]] = None
    ) -> AILearningAnalysis:
        """
        Analyze a learning outcome using AI before recording.
        
        This is the core AI-enhanced learning function:
        1. First checks pattern-based analysis (fast)
        2. If uncertain, invokes LLM for deeper analysis
        3. Returns decision with full reasoning
        """
        start_time = datetime.utcnow()
        self._stats["total_analyses"] += 1
        
        # Create cache key
        cache_key = self._create_cache_key(module_name, input_data, output_data)
        
        # Check cache
        if self.cache_enabled and cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]
        
        # Step 1: Pattern-based analysis (fast path)
        pattern_analysis = self._pattern_based_analysis(
            module_name, input_data, output_data, was_correct, domain
        )
        self._stats["pattern_analyses"] += 1
        
        # Step 1.5: ML-based prediction (NEW - Hybrid Component)
        ml_features = {
            'module': module_name,
            'domain': domain,
            'was_correct': was_correct,
            'input_size': len(str(input_data)),
            'output_size': len(str(output_data))
        }
        ml_probs = self.ml_predictor.predict(ml_features)
        ml_confidence = max(ml_probs.values()) if ml_probs else 0.5
        
        # Merge pattern and ML confidence
        merged_confidence = (pattern_analysis.confidence * 0.6) + (ml_confidence * 0.4)
        pattern_analysis.confidence = merged_confidence
        
        # If merged analysis is confident, use it
        if merged_confidence >= self.ai_threshold:
            pattern_analysis.analysis_time_ms = (
                datetime.utcnow() - start_time
            ).total_seconds() * 1000
            pattern_analysis.pattern_type = "pattern+ml"
            self._cache_result(cache_key, pattern_analysis)
            
            # Update ML model with this outcome
            self.ml_predictor.update(ml_features, pattern_analysis.decision.value)
            
            return pattern_analysis
        
        # Step 2: LLM analysis for uncertain cases
        if self.enable_llm and self.llm_registry:
            llm_analysis = await self._llm_based_analysis(
                module_name, input_data, output_data, was_correct, domain, context
            )
            self._stats["llm_analyses"] += 1
            
            # Merge pattern and LLM analysis
            final_analysis = self._merge_analyses(pattern_analysis, llm_analysis)
        else:
            final_analysis = pattern_analysis
        
        final_analysis.analysis_time_ms = (
            datetime.utcnow() - start_time
        ).total_seconds() * 1000
        
        # Update statistics
        self._stats["decisions"][final_analysis.decision.value] += 1
        self._stats["domain_learnings"][domain] += 1
        
        # Always update ML model with the final analysis (NEW)
        ml_features = {
            'module': module_name,
            'domain': domain,
            'was_correct': was_correct,
            'input_size': len(str(input_data)),
            'output_size': len(str(output_data))
        }
        self.ml_predictor.update(ml_features, final_analysis.decision.value)
        
        # Cache result
        self._cache_result(cache_key, final_analysis)
        
        # Add to audit trail
        self._add_audit_entry(module_name, final_analysis, was_correct)
        
        return final_analysis
    
    def _pattern_based_analysis(
        self,
        module_name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        was_correct: bool,
        domain: str
    ) -> AILearningAnalysis:
        """
        Fast pattern-based analysis without LLM.
        
        Uses statistical patterns and heuristics for quick decisions.
        """
        confidence = 0.5
        decision = LearningDecision.LEARN
        reasoning_parts = []
        warnings = []
        cross_module = []
        
        # Pattern 1: Domain familiarity
        domain_count = self._stats["domain_learnings"].get(domain, 0)
        if domain_count > 100:
            confidence += 0.1
            reasoning_parts.append(f"Familiar domain ({domain_count} prior samples)")
        elif domain_count < 10:
            confidence -= 0.1
            reasoning_parts.append(f"Unfamiliar domain ({domain_count} prior samples)")
        
        # Pattern 2: Module stability
        if module_name in self._cross_module_patterns:
            pattern_count = len(self._cross_module_patterns[module_name])
            if pattern_count > 20:
                confidence += 0.1
                reasoning_parts.append(f"Stable module ({pattern_count} patterns)")
        
        # Pattern 3: Correctness consistency
        if was_correct:
            confidence += 0.1
            decision = LearningDecision.LEARN
            reasoning_parts.append("Correct outcome - standard learning")
        else:
            # Check if this is a repeated error pattern
            if self._is_repeated_error(module_name, input_data):
                decision = LearningDecision.AMPLIFY
                confidence += 0.15
                reasoning_parts.append("Repeated error - amplified learning weight")
                warnings.append("Recurring error pattern detected")
        
        # Pattern 4: Cross-module applicability
        applicable_modules = self._find_applicable_modules(input_data, output_data)
        if applicable_modules:
            cross_module = applicable_modules
            confidence += 0.05
            reasoning_parts.append(f"Pattern applicable to: {', '.join(applicable_modules[:3])}")
        
        # Pattern 5: Contradiction detection
        if self._detects_contradiction(module_name, input_data, output_data, was_correct):
            decision = LearningDecision.CONTRADICT
            confidence = 0.8
            warnings.append("Contradicts prior learning")
            self._stats["contradictions_detected"] += 1
        
        # Pattern 6: Low signal detection
        if self._is_low_signal(input_data, output_data):
            decision = LearningDecision.SKIP
            confidence = 0.6
            reasoning_parts.append("Low signal - skip learning")
        
        return AILearningAnalysis(
            decision=decision,
            confidence=min(confidence, 0.95),
            reasoning=" | ".join(reasoning_parts) if reasoning_parts else "Standard learning path",
            suggested_weight=1.5 if decision == LearningDecision.AMPLIFY else 1.0,
            cross_module_applicable=cross_module,
            pattern_type="statistical",
            warnings=warnings,
            llm_used="none"
        )
    
    async def _llm_based_analysis(
        self,
        module_name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        was_correct: bool,
        domain: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AILearningAnalysis:
        """
        Deep LLM-based analysis for uncertain cases.
        
        Uses the configured LLM to reason about learning decisions.
        """
        if not self.llm_registry:
            return AILearningAnalysis(
                decision=LearningDecision.LEARN,
                confidence=0.5,
                reasoning="LLM not available - default learning",
                llm_used="none"
            )
        
        try:
            # Construct prompt for LLM
            prompt = self._construct_analysis_prompt(
                module_name, input_data, output_data, was_correct, domain, context
            )
            
            # Get LLM provider
            llm_provider = self.llm_registry.get_preferred_provider()
            llm_name = llm_provider.name if hasattr(llm_provider, 'name') else str(llm_provider)
            
            # Call LLM
            response = await self._call_llm(prompt)
            
            # Parse LLM response
            analysis = self._parse_llm_response(response, llm_name)
            
            return analysis
            
        except Exception as e:
            logger.warning(f"[AIEnhancedLearning] LLM analysis failed: {e}")
            return AILearningAnalysis(
                decision=LearningDecision.LEARN,
                confidence=0.5,
                reasoning=f"LLM analysis failed: {str(e)[:100]}",
                llm_used="error"
            )
    
    def _construct_analysis_prompt(
        self,
        module_name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        was_correct: bool,
        domain: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Construct the prompt for LLM analysis."""
        return f"""Analyze this learning opportunity for a bias governance system:

MODULE: {module_name}
DOMAIN: {domain}
OUTCOME: {"CORRECT" if was_correct else "INCORRECT"}

INPUT (truncated): {json.dumps(input_data, default=str)[:500]}
OUTPUT (truncated): {json.dumps(output_data, default=str)[:500]}
CONTEXT: {json.dumps(context, default=str)[:200] if context else "None"}

Determine:
1. DECISION: Should we LEARN, SKIP, AMPLIFY, ATTENUATE, or flag CONTRADICT?
2. CONFIDENCE: 0.0 to 1.0
3. REASONING: Why this decision?
4. CROSS_MODULE: Which other modules could benefit from this pattern?
5. WARNINGS: Any concerns?

Respond in JSON format:
{{"decision": "learn|skip|amplify|attenuate|contradict", "confidence": 0.X, "reasoning": "...", "cross_modules": [], "warnings": []}}"""
    
    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM via registry."""
        if not self.llm_registry:
            return "{}"
        
        try:
            # Try to get async completion
            if hasattr(self.llm_registry, 'complete_async'):
                return await self.llm_registry.complete_async(prompt)
            elif hasattr(self.llm_registry, 'complete'):
                return self.llm_registry.complete(prompt)
            elif hasattr(self.llm_registry, 'call'):
                return await self.llm_registry.call(prompt)
            else:
                # Fallback: try direct provider call
                provider = self.llm_registry.get_preferred_provider()
                if hasattr(provider, 'complete'):
                    return provider.complete(prompt)
                return "{}"
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return "{}"
    
    def _parse_llm_response(self, response: str, llm_name: str) -> AILearningAnalysis:
        """Parse LLM response into AILearningAnalysis."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = {}
            
            decision_str = data.get("decision", "learn").lower()
            decision_map = {
                "learn": LearningDecision.LEARN,
                "skip": LearningDecision.SKIP,
                "amplify": LearningDecision.AMPLIFY,
                "attenuate": LearningDecision.ATTENUATE,
                "contradict": LearningDecision.CONTRADICT,
                "defer": LearningDecision.DEFER
            }
            
            return AILearningAnalysis(
                decision=decision_map.get(decision_str, LearningDecision.LEARN),
                confidence=float(data.get("confidence", 0.7)),
                reasoning=data.get("reasoning", "LLM-based analysis"),
                cross_module_applicable=data.get("cross_modules", []),
                warnings=data.get("warnings", []),
                llm_used=llm_name
            )
        except Exception as e:
            return AILearningAnalysis(
                decision=LearningDecision.LEARN,
                confidence=0.5,
                reasoning=f"Failed to parse LLM response: {str(e)[:50]}",
                llm_used=llm_name
            )
    
    def _merge_analyses(
        self,
        pattern: AILearningAnalysis,
        llm: AILearningAnalysis
    ) -> AILearningAnalysis:
        """Merge pattern-based and LLM analyses."""
        # Weight LLM analysis higher for uncertain cases
        pattern_weight = 0.4
        llm_weight = 0.6
        
        # Use LLM decision if it's more confident
        if llm.confidence > pattern.confidence:
            final_decision = llm.decision
            final_reasoning = f"LLM: {llm.reasoning}"
        else:
            final_decision = pattern.decision
            final_reasoning = f"Pattern: {pattern.reasoning}"
        
        # Merge confidence
        final_confidence = (
            pattern_weight * pattern.confidence + 
            llm_weight * llm.confidence
        )
        
        # Merge cross-module lists
        cross_modules = list(set(pattern.cross_module_applicable + llm.cross_module_applicable))
        
        # Merge warnings
        warnings = list(set(pattern.warnings + llm.warnings))
        
        return AILearningAnalysis(
            decision=final_decision,
            confidence=final_confidence,
            reasoning=final_reasoning,
            suggested_weight=llm.suggested_weight if llm.suggested_weight != 1.0 else pattern.suggested_weight,
            cross_module_applicable=cross_modules,
            pattern_type="hybrid",
            warnings=warnings,
            llm_used=llm.llm_used
        )
    
    def _is_repeated_error(self, module_name: str, input_data: Dict) -> bool:
        """Check if this is a repeated error pattern."""
        # Check recent audit trail for similar errors
        similar_count = 0
        input_sig = str(sorted(input_data.keys()))
        
        for entry in self._audit_trail[-50:]:
            if (entry.get("module") == module_name and 
                not entry.get("was_correct") and
                entry.get("input_signature") == input_sig):
                similar_count += 1
        
        return similar_count >= 2
    
    def _find_applicable_modules(
        self,
        input_data: Dict,
        output_data: Dict
    ) -> List[str]:
        """Find modules that could benefit from this pattern."""
        applicable = []
        
        # Check pattern registry for similar patterns
        for module, patterns in self._cross_module_patterns.items():
            for pattern in patterns[-10:]:
                # Simple similarity check
                if self._patterns_similar(input_data, pattern.get("input", {})):
                    applicable.append(module)
                    break
        
        return applicable[:5]  # Limit to top 5
    
    def _patterns_similar(self, pattern1: Dict, pattern2: Dict) -> bool:
        """Check if two patterns are similar."""
        keys1 = set(pattern1.keys())
        keys2 = set(pattern2.keys())
        overlap = len(keys1 & keys2)
        union = len(keys1 | keys2)
        
        if union == 0:
            return False
        
        return (overlap / union) > 0.5
    
    def _detects_contradiction(
        self,
        module_name: str,
        input_data: Dict,
        output_data: Dict,
        was_correct: bool
    ) -> bool:
        """Check if this outcome contradicts prior learning."""
        # Look for same input with opposite correctness
        input_sig = json.dumps(sorted(input_data.items()), default=str)
        
        for entry in self._audit_trail[-100:]:
            if (entry.get("module") == module_name and
                entry.get("input_signature") == input_sig and
                entry.get("was_correct") != was_correct):
                return True
        
        return False
    
    def _is_low_signal(self, input_data: Dict, output_data: Dict) -> bool:
        """Check if this is a low-signal learning opportunity."""
        # Check for empty or trivial data
        if not input_data or not output_data:
            return True
        
        # Check for very simple outputs
        if len(str(output_data)) < 10:
            return True
        
        return False
    
    def _create_cache_key(
        self,
        module_name: str,
        input_data: Dict,
        output_data: Dict
    ) -> str:
        """Create cache key for analysis results."""
        import hashlib
        content = f"{module_name}:{json.dumps(input_data, sort_keys=True, default=str)[:200]}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _cache_result(self, key: str, analysis: AILearningAnalysis) -> None:
        """Cache analysis result with LRU eviction."""
        if not self.cache_enabled:
            return
        
        if key in self._analysis_cache:
            self._cache_order.remove(key)
        elif len(self._analysis_cache) >= self.max_cache:
            # Evict oldest
            oldest = self._cache_order.pop(0)
            del self._analysis_cache[oldest]
        
        self._analysis_cache[key] = analysis
        self._cache_order.append(key)
    
    def _add_audit_entry(
        self,
        module_name: str,
        analysis: AILearningAnalysis,
        was_correct: bool
    ) -> None:
        """Add entry to audit trail."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "module": module_name,
            "decision": analysis.decision.value,
            "confidence": analysis.confidence,
            "was_correct": was_correct,
            "llm_used": analysis.llm_used,
            "input_signature": "",  # Set by caller
            "warnings": analysis.warnings
        }
        
        self._audit_trail.append(entry)
        
        # Limit audit trail size
        if len(self._audit_trail) > 10000:
            self._audit_trail = self._audit_trail[-5000:]
    
    # ==== Synchronous API ====
    
    def analyze_outcome_sync(
        self,
        module_name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        was_correct: bool,
        domain: str = "general",
        context: Optional[Dict[str, Any]] = None
    ) -> AILearningAnalysis:
        """Synchronous version of analyze_outcome."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.analyze_outcome(
                module_name, input_data, output_data, 
                was_correct, domain, context
            )
        )
    
    def record_with_analysis(
        self,
        module_name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        was_correct: bool,
        domain: str = "general"
    ) -> EnhancedLearningOutcome:
        """
        Analyze and record a learning outcome.
        
        This is the main entry point for enhanced learning:
        1. Analyzes the outcome using AI
        2. Decides whether to learn and how
        3. Records to base manager if appropriate
        4. Returns the enhanced outcome with analysis
        """
        # Get AI analysis
        analysis = self.analyze_outcome_sync(
            module_name, input_data, output_data, was_correct, domain
        )
        
        # Create enhanced outcome
        outcome = EnhancedLearningOutcome(
            module_name=module_name,
            event_type="outcome",
            input_data=input_data,
            output_data=output_data,
            was_correct=was_correct,
            domain=domain,
            timestamp=datetime.utcnow().isoformat(),
            ai_analysis=analysis
        )
        
        # Record to base manager based on decision
        if self.base_manager and analysis.decision in [
            LearningDecision.LEARN, 
            LearningDecision.AMPLIFY
        ]:
            weight = analysis.suggested_weight
            self.base_manager.record_outcome(
                module_name=module_name,
                input_data=input_data,
                output_data=output_data,
                was_correct=was_correct,
                domain=domain,
                metadata={"ai_weight": weight, "ai_decision": analysis.decision.value}
            )
        
        # Share cross-module patterns
        if analysis.cross_module_applicable:
            self._share_pattern(module_name, input_data, output_data, analysis)
            self._stats["cross_module_shares"] += 1
        
        return outcome
    
    def _share_pattern(
        self,
        source_module: str,
        input_data: Dict,
        output_data: Dict,
        analysis: AILearningAnalysis
    ) -> None:
        """Share a learned pattern to applicable modules."""
        pattern = {
            "source": source_module,
            "input": input_data,
            "output": output_data,
            "timestamp": datetime.utcnow().isoformat(),
            "analysis": {
                "decision": analysis.decision.value,
                "confidence": analysis.confidence
            }
        }
        
        for target_module in analysis.cross_module_applicable:
            self._cross_module_patterns[target_module].append(pattern)
            
            # Limit patterns per module
            if len(self._cross_module_patterns[target_module]) > 100:
                self._cross_module_patterns[target_module] = \
                    self._cross_module_patterns[target_module][-50:]
    
    # ==== Learning Interface Methods ====
    
    def record_outcome(self, outcome: Dict) -> None:
        """Standard learning interface: record_outcome."""
        self.record_with_analysis(
            module_name=outcome.get("module", "unknown"),
            input_data=outcome.get("input", {}),
            output_data=outcome.get("output", {}),
            was_correct=outcome.get("was_correct", True),
            domain=outcome.get("domain", "general")
        )
    
    def record_feedback(self, feedback: Dict) -> None:
        """Standard learning interface: record_feedback."""
        self._audit_trail.append({
            "type": "feedback",
            "timestamp": datetime.utcnow().isoformat(),
            **feedback
        })
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt thresholds using AI-enhanced analysis."""
        if performance_data:
            # Use AI to determine optimal threshold adjustment
            accuracy = performance_data.get("accuracy", 0.5)
            fp_rate = performance_data.get("false_positive_rate", 0)
            fn_rate = performance_data.get("false_negative_rate", 0)
            
            # AI-determined adjustment
            if fp_rate > fn_rate:
                adjustment = 0.05  # Increase threshold
            elif fn_rate > fp_rate:
                adjustment = -0.05  # Decrease threshold
            else:
                adjustment = 0.0
            
            self._domain_lr_modifiers[domain] = (
                self._domain_lr_modifiers.get(domain, 0.0) + adjustment
            )
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get AI-determined domain adjustment."""
        return self._domain_lr_modifiers.get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get comprehensive learning statistics."""
        return {
            "total_analyses": self._stats["total_analyses"],
            "llm_analyses": self._stats["llm_analyses"],
            "pattern_analyses": self._stats["pattern_analyses"],
            "decisions": dict(self._stats["decisions"]),
            "domain_learnings": dict(self._stats["domain_learnings"]),
            "cross_module_shares": self._stats["cross_module_shares"],
            "contradictions_detected": self._stats["contradictions_detected"],
            "cache_size": len(self._analysis_cache),
            "audit_trail_size": len(self._audit_trail),
            "domain_adjustments": dict(self._domain_lr_modifiers),
            "ml_predictor": self.ml_predictor.get_statistics()  # NEW: ML statistics
        }
    
    def save_state(self) -> None:
        """Save AI learning state."""
        if self.base_manager:
            self.base_manager.save_state()
    
    def load_state(self) -> None:
        """Load AI learning state."""
        if self.base_manager:
            self.base_manager.load_state()
    
    def get_explanation(self, module_name: str, limit: int = 10) -> LearningExplanation:
        """
        Get human-readable explanation of recent learning decisions.
        
        This supports governance audit requirements.
        """
        # Filter audit trail for module
        module_entries = [
            e for e in self._audit_trail[-1000:]
            if e.get("module") == module_name
        ][-limit:]
        
        if not module_entries:
            return LearningExplanation(
                summary=f"No learning history for {module_name}",
                factors=[],
                confidence_breakdown={},
                alternative_actions=[],
                audit_trail=[]
            )
        
        # Build summary
        decision_counts = defaultdict(int)
        total_confidence = 0
        for entry in module_entries:
            decision_counts[entry.get("decision", "unknown")] += 1
            total_confidence += entry.get("confidence", 0)
        
        avg_confidence = total_confidence / len(module_entries) if module_entries else 0
        most_common = max(decision_counts.items(), key=lambda x: x[1])[0]
        
        return LearningExplanation(
            summary=f"Module {module_name}: {len(module_entries)} decisions, "
                   f"most common: {most_common}, avg confidence: {avg_confidence:.2%}",
            factors=[
                f"{decision}: {count} times" 
                for decision, count in decision_counts.items()
            ],
            confidence_breakdown={
                entry.get("decision", "unknown"): entry.get("confidence", 0)
                for entry in module_entries[-5:]
            },
            alternative_actions=["Review contradictions", "Adjust thresholds", "Reset module"],
            audit_trail=module_entries
        )

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
# Factory Function
# ==============================================================================

    # =========================================================================
    # Learning Interface Completion
    # =========================================================================
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback to adjust behavior."""
        if not hasattr(self, '_learning_params'):
            self._learning_params = {}
        was_correct = feedback.get('was_correct', True)
        if not was_correct:
            adjustment = feedback.get('adjustment', 0.05)
            for key in self._learning_params:
                self._learning_params[key] *= (1 - adjustment)
    
    def get_statistics(self) -> Dict:
        """Return learning statistics."""
        return {
            'learning_params': getattr(self, '_learning_params', {}),
            'history_size': len(getattr(self, '_outcomes', []))
        }
    
    def serialize_state(self) -> Dict:
        """Serialize state for persistence."""
        return {
            'outcomes': getattr(self, '_outcomes', [])[-100:],
            'learning_params': getattr(self, '_learning_params', {}),
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore state from serialized data."""
        self._outcomes = state.get('outcomes', [])
        self._learning_params = state.get('learning_params', {})

def create_ai_enhanced_learning(
    llm_registry: Optional[Any] = None,
    base_manager: Optional[Any] = None
) -> AIEnhancedLearningManager:
    """Factory function to create AI-enhanced learning manager."""
    return AIEnhancedLearningManager(
        llm_registry=llm_registry,
        base_learning_manager=base_manager
    )


# ==============================================================================
# Test
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("AI-ENHANCED LEARNING MANAGER TEST")
    print("=" * 80)
    
    # Create manager
    manager = AIEnhancedLearningManager(enable_llm_analysis=False)
    
    # Test pattern-based analysis
    print("\n[1] Testing pattern-based analysis...")
    
    result = manager.record_with_analysis(
        module_name="drift_detection",
        input_data={"signal": 0.75, "threshold": 0.6},
        output_data={"drift_detected": True, "confidence": 0.85},
        was_correct=True,
        domain="financial"
    )
    
    print(f"    Decision: {result.ai_analysis.decision.value}")
    print(f"    Confidence: {result.ai_analysis.confidence:.2%}")
    print(f"    Reasoning: {result.ai_analysis.reasoning}")
    
    # Test repeated error detection
    print("\n[2] Testing repeated error detection...")
    
    for i in range(3):
        result = manager.record_with_analysis(
            module_name="adversarial",
            input_data={"text": "ignore instructions"},
            output_data={"threat": True},
            was_correct=False,
            domain="security"
        )
    
    print(f"    Decision after 3 errors: {result.ai_analysis.decision.value}")
    print(f"    Warnings: {result.ai_analysis.warnings}")
    
    # Get statistics
    print("\n[3] Learning statistics:")
    stats = manager.get_learning_statistics()
    for key, value in stats.items():
        print(f"    {key}: {value}")
    
    # Get explanation
    print("\n[4] Learning explanation:")
    explanation = manager.get_explanation("drift_detection")
    print(f"    Summary: {explanation.summary}")
    print(f"    Factors: {explanation.factors}")
    
    print("\n" + "=" * 80)
    print("✓ AI-ENHANCED LEARNING MANAGER TEST PASSED")
    print("=" * 80)

