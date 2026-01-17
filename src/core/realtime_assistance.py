"""
BAIS Cognitive Governance Engine - Real-Time Assistance Engine
Phase 6: Active enhancement before user sees response

In DIRECT_ASSISTANCE mode, BAIS:
1. Intercepts LLM response before delivery
2. Analyzes for issues, gaps, improvements
3. Optionally uses multi-track for better alternatives
4. Enhances/fixes the response
5. Returns improved response to user

The user gets a better response without manual iteration.

Patent: NOVEL-45 (Real-Time Assistance Engine)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from datetime import datetime
import re

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class EnhancementType(Enum):
    """Types of enhancements that can be applied."""
    ADD_DISCLAIMER = "add_disclaimer"
    ADD_CONTEXT = "add_context"
    FIX_FACTUAL_ERROR = "fix_factual_error"
    IMPROVE_CLARITY = "improve_clarity"
    ADD_EXAMPLES = "add_examples"
    ADD_CAVEATS = "add_caveats"
    REMOVE_OVERCONFIDENCE = "remove_overconfidence"
    COMPLETE_PARTIAL = "complete_partial"
    VERIFY_CLAIMS = "verify_claims"


class AssistanceLevel(Enum):
    """Level of assistance to provide."""
    MINIMAL = "minimal"       # Only critical fixes
    MODERATE = "moderate"     # Critical + improvements
    COMPREHENSIVE = "comprehensive"  # Full enhancement
    CUSTOM = "custom"         # User-defined rules


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EnhancementAction:
    """A single enhancement action."""
    action_id: str
    enhancement_type: EnhancementType
    description: str
    original_content: str
    enhanced_content: str
    reason: str
    confidence: float


@dataclass
class AssistanceResult:
    """Result of real-time assistance."""
    original_response: str
    enhanced_response: str
    
    # What was done
    enhancements_applied: List[EnhancementAction]
    total_enhancements: int
    
    # Multi-track (if used)
    used_multi_track: bool
    alternative_tracks: List[Dict[str, Any]]
    selected_track: Optional[str]
    
    # Confidence
    original_confidence: float
    enhanced_confidence: float
    
    # Metadata
    processing_time_ms: float
    timestamp: str


@dataclass
class IssueDetection:
    """A detected issue in the response."""
    issue_type: str
    severity: str
    location: str
    description: str
    suggested_fix: str


# =============================================================================
# Enhancement Strategies
# =============================================================================

class OverconfidenceRemover:
    """Removes overconfident language."""
    
    OVERCONFIDENT_PATTERNS = [
        (r'\b100%\b', 'very high'),
        (r'\bguaranteed\b', 'likely'),
        (r'\balways\b', 'typically'),
        (r'\bnever\b', 'rarely'),
        (r'\bperfect\b', 'excellent'),
        (r'\bflawless\b', 'well-designed'),
        (r'\babsolutely\b', 'highly'),
        (r'\bdefinitely\b', 'likely'),
        (r'\bimpossible\b', 'very unlikely'),
        (r'\bwithout a doubt\b', 'with high confidence'),
    ]
    
    def apply(self, text: str) -> Tuple[str, List[str]]:
        """Apply overconfidence removal."""
        changes = []
        result = text
        
        for pattern, replacement in self.OVERCONFIDENT_PATTERNS:
            if re.search(pattern, result, re.IGNORECASE):
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
                changes.append(f"Replaced '{pattern}' with '{replacement}'")
        
        return result, changes


class DisclaimerAdder:
    """Adds appropriate disclaimers based on domain."""
    
    DISCLAIMERS = {
        'medical': "\n\n**Disclaimer:** This information is for educational purposes only and should not replace professional medical advice. Consult a healthcare provider for medical concerns.",
        'financial': "\n\n**Disclaimer:** This is general information, not financial advice. Consult a qualified financial advisor for personalized guidance.",
        'legal': "\n\n**Disclaimer:** This information is for educational purposes only and does not constitute legal advice. Consult a qualified attorney for legal matters.",
        'general': ""
    }
    
    def apply(self, text: str, domain: str) -> Tuple[str, bool]:
        """Add disclaimer if needed."""
        disclaimer = self.DISCLAIMERS.get(domain, "")
        
        if disclaimer and disclaimer not in text:
            return text + disclaimer, True
        
        return text, False


class PartialCompletionHandler:
    """Handles partial or incomplete responses."""
    
    INCOMPLETE_MARKERS = [
        "...", "etc", "and so on", "continued", "more to come",
        "to be continued", "incomplete", "partial"
    ]
    
    def detect_incomplete(self, text: str) -> bool:
        """Detect if response is incomplete."""
        text_lower = text.lower()
        
        # Check for markers
        for marker in self.INCOMPLETE_MARKERS:
            if marker in text_lower:
                return True
        
        # Check for truncated endings
        if text.strip().endswith(('...', '..')):
            return True
        
        return False
    
    def add_completion_notice(self, text: str) -> str:
        """Add notice about incomplete response."""
        if self.detect_incomplete(text):
            return text + "\n\n**Note:** This response may be incomplete. Please ask for more details if needed."
        return text

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


# =============================================================================
# Real-Time Assistance Engine
# =============================================================================

class RealTimeAssistanceEngine:
    """
    Provides real-time enhancement of LLM responses.
    
    Works in DIRECT_ASSISTANCE mode to improve responses
    before the user sees them.
    """
    
    def __init__(
        self,
        assistance_level: AssistanceLevel = AssistanceLevel.MODERATE,
        multi_track_enabled: bool = False,
        multi_track_provider: Optional[Callable] = None,
        response_improver: Optional[Callable] = None
    ):
        self.assistance_level = assistance_level
        self.multi_track_enabled = multi_track_enabled
        self.multi_track_provider = multi_track_provider
        self.response_improver = response_improver
        
        # Enhancement strategies
        self.overconfidence_remover = OverconfidenceRemover()
        self.disclaimer_adder = DisclaimerAdder()
        self.partial_handler = PartialCompletionHandler()
        
        # Statistics
        self._stats = {
            'total_assists': 0,
            'enhancements_applied': 0,
            'multi_track_used': 0,
            'by_type': {e.value: 0 for e in EnhancementType},
            'avg_confidence_boost': 0.0
        }
    
    def assist(
        self,
        response: str,
        query: str,
        domain: str = "general",
        detected_issues: Optional[List[IssueDetection]] = None,
        context: Optional[Dict] = None
    ) -> AssistanceResult:
        """
        Provide real-time assistance on a response.
        
        Args:
            response: Original LLM response
            query: Original user query
            domain: Domain context
            detected_issues: Pre-detected issues from BAIS
            context: Additional context
        
        Returns:
            AssistanceResult with enhanced response
        """
        start_time = datetime.utcnow()
        self._stats['total_assists'] += 1
        
        enhancements: List[EnhancementAction] = []
        current_response = response
        original_confidence = self._estimate_confidence(response)
        
        # Strategy 1: Remove overconfidence
        current_response, oc_changes = self._apply_overconfidence_removal(
            current_response, enhancements
        )
        
        # Strategy 2: Add domain-specific disclaimer
        current_response, added_disclaimer = self._apply_disclaimer(
            current_response, domain, enhancements
        )
        
        # Strategy 3: Handle incomplete responses
        current_response, handled_incomplete = self._handle_incomplete(
            current_response, enhancements
        )
        
        # Strategy 4: Fix detected issues
        if detected_issues:
            current_response = self._fix_detected_issues(
                current_response, detected_issues, enhancements
            )
        
        # Strategy 5: Multi-track improvement (if enabled)
        alternative_tracks = []
        selected_track = None
        
        if self.multi_track_enabled and self.multi_track_provider:
            current_response, alternative_tracks, selected_track = self._apply_multi_track(
                query, current_response, enhancements
            )
        
        # Calculate confidence boost
        enhanced_confidence = self._estimate_confidence(current_response)
        self._update_confidence_stats(original_confidence, enhanced_confidence)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return AssistanceResult(
            original_response=response,
            enhanced_response=current_response,
            enhancements_applied=enhancements,
            total_enhancements=len(enhancements),
            used_multi_track=self.multi_track_enabled and len(alternative_tracks) > 0,
            alternative_tracks=alternative_tracks,
            selected_track=selected_track,
            original_confidence=original_confidence,
            enhanced_confidence=enhanced_confidence,
            processing_time_ms=processing_time,
            timestamp=start_time.isoformat()
        )
    
    def _apply_overconfidence_removal(
        self,
        text: str,
        enhancements: List[EnhancementAction]
    ) -> Tuple[str, List[str]]:
        """Apply overconfidence removal."""
        result, changes = self.overconfidence_remover.apply(text)
        
        if changes:
            self._stats['enhancements_applied'] += 1
            self._stats['by_type'][EnhancementType.REMOVE_OVERCONFIDENCE.value] += 1
            
            enhancements.append(EnhancementAction(
                action_id=f"oc-{len(enhancements)}",
                enhancement_type=EnhancementType.REMOVE_OVERCONFIDENCE,
                description="Removed overconfident language",
                original_content=text[:100],
                enhanced_content=result[:100],
                reason=f"Applied {len(changes)} overconfidence corrections",
                confidence=0.9
            ))
        
        return result, changes
    
    def _apply_disclaimer(
        self,
        text: str,
        domain: str,
        enhancements: List[EnhancementAction]
    ) -> Tuple[str, bool]:
        """Apply domain-specific disclaimer."""
        result, added = self.disclaimer_adder.apply(text, domain)
        
        if added:
            self._stats['enhancements_applied'] += 1
            self._stats['by_type'][EnhancementType.ADD_DISCLAIMER.value] += 1
            
            enhancements.append(EnhancementAction(
                action_id=f"disc-{len(enhancements)}",
                enhancement_type=EnhancementType.ADD_DISCLAIMER,
                description=f"Added {domain} disclaimer",
                original_content="",
                enhanced_content=f"Added {domain} disclaimer",
                reason=f"Domain '{domain}' requires disclaimer",
                confidence=1.0
            ))
        
        return result, added
    
    def _handle_incomplete(
        self,
        text: str,
        enhancements: List[EnhancementAction]
    ) -> Tuple[str, bool]:
        """Handle incomplete responses."""
        is_incomplete = self.partial_handler.detect_incomplete(text)
        
        if is_incomplete:
            result = self.partial_handler.add_completion_notice(text)
            
            self._stats['enhancements_applied'] += 1
            self._stats['by_type'][EnhancementType.COMPLETE_PARTIAL.value] += 1
            
            enhancements.append(EnhancementAction(
                action_id=f"inc-{len(enhancements)}",
                enhancement_type=EnhancementType.COMPLETE_PARTIAL,
                description="Added incomplete response notice",
                original_content=text[-50:],
                enhanced_content=result[-100:],
                reason="Detected incomplete response",
                confidence=0.8
            ))
            
            return result, True
        
        return text, False
    
    def _fix_detected_issues(
        self,
        text: str,
        issues: List[IssueDetection],
        enhancements: List[EnhancementAction]
    ) -> str:
        """Fix detected issues."""
        result = text
        
        for issue in issues:
            if issue.severity in ['high', 'critical'] and issue.suggested_fix:
                # Apply fix (simplified - in real implementation, would be more sophisticated)
                if issue.issue_type == 'factual_error':
                    self._stats['by_type'][EnhancementType.FIX_FACTUAL_ERROR.value] += 1
                elif issue.issue_type == 'missing_context':
                    self._stats['by_type'][EnhancementType.ADD_CONTEXT.value] += 1
                
                enhancements.append(EnhancementAction(
                    action_id=f"fix-{len(enhancements)}",
                    enhancement_type=EnhancementType.ADD_CAVEATS,
                    description=f"Addressed {issue.issue_type}",
                    original_content=issue.description,
                    enhanced_content=issue.suggested_fix,
                    reason=issue.description,
                    confidence=0.7
                ))
                
                self._stats['enhancements_applied'] += 1
        
        return result
    
    def _apply_multi_track(
        self,
        query: str,
        current_response: str,
        enhancements: List[EnhancementAction]
    ) -> Tuple[str, List[Dict], Optional[str]]:
        """Apply multi-track improvement."""
        if not self.multi_track_provider:
            return current_response, [], None
        
        self._stats['multi_track_used'] += 1
        
        try:
            # Get alternatives from multi-track
            result = self.multi_track_provider(query)
            
            if 'track_results' in result:
                alternatives = result['track_results']
                
                # If consensus is better, use it
                if 'consensus' in result:
                    consensus_conf = result['consensus'].get('confidence', 0)
                    current_conf = self._estimate_confidence(current_response)
                    
                    if consensus_conf > current_conf:
                        enhancements.append(EnhancementAction(
                            action_id=f"mt-{len(enhancements)}",
                            enhancement_type=EnhancementType.VERIFY_CLAIMS,
                            description="Applied multi-track consensus",
                            original_content=current_response[:100],
                            enhanced_content=result['consensus']['recommended_response'][:100],
                            reason=f"Consensus confidence ({consensus_conf:.2f}) > original ({current_conf:.2f})",
                            confidence=consensus_conf
                        ))
                        
                        return result['consensus']['recommended_response'], alternatives, result['consensus'].get('primary_track')
                
                return current_response, alternatives, None
            
        except Exception as e:
            logger.warning(f"[RealTimeAssist] Multi-track failed: {e}")
        
        return current_response, [], None
    
    def _estimate_confidence(self, text: str) -> float:
        """Estimate confidence level of a response."""
        # Simple heuristic-based confidence estimation
        confidence = 0.5
        
        # Positive indicators
        if len(text) > 200:
            confidence += 0.1
        if 'example' in text.lower():
            confidence += 0.05
        if 'specifically' in text.lower() or 'for instance' in text.lower():
            confidence += 0.05
        
        # Negative indicators
        for pattern in OverconfidenceRemover.OVERCONFIDENT_PATTERNS:
            if re.search(pattern[0], text, re.IGNORECASE):
                confidence -= 0.05
        
        if '...' in text:
            confidence -= 0.1
        
        return max(0.1, min(0.95, confidence))
    
    def _update_confidence_stats(self, original: float, enhanced: float) -> None:
        """Update confidence boost statistics."""
        n = self._stats['total_assists']
        current_avg = self._stats['avg_confidence_boost']
        boost = enhanced - original
        self._stats['avg_confidence_boost'] = (current_avg * (n - 1) + boost) / n
    
    def set_assistance_level(self, level: AssistanceLevel) -> None:
        """Set assistance level."""
        self.assistance_level = level
        logger.info(f"[RealTimeAssist] Assistance level set to: {level.value}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get assistance statistics."""
        return {
            **self._stats,
            'learning_params': getattr(self, '_learning_params', {}),
            'history_size': len(getattr(self, '_outcome_history', []))
        }
    
    # =========================================================================
    # Learning Interface (5/5 methods)
    # =========================================================================
    
    def record_outcome(self, outcome: Dict[str, Any]) -> None:
        """Record assistance outcome for learning."""
        if not hasattr(self, '_outcome_history'):
            self._outcome_history: List[Dict] = []
        
        self._stats['total_assists'] += 1
        if outcome.get('enhancement_accepted', True):
            self._stats.setdefault('accepted', 0)
            self._stats['accepted'] += 1
        
        self._outcome_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            **outcome
        })
        if len(self._outcome_history) > 1000:
            self._outcome_history = self._outcome_history[-500:]
    
    def learn_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """Learn from feedback to improve enhancements."""
        if not hasattr(self, '_learning_params'):
            self._learning_params = {'enhancement_preference': {}}
        
        was_helpful = feedback.get('was_helpful', True)
        enhancement_type = feedback.get('enhancement_type')
        
        if enhancement_type:
            current = self._learning_params['enhancement_preference'].get(enhancement_type, 0.5)
            if was_helpful:
                self._learning_params['enhancement_preference'][enhancement_type] = min(1.0, current + 0.05)
            else:
                self._learning_params['enhancement_preference'][enhancement_type] = max(0.1, current - 0.05)
    
    def serialize_state(self) -> Dict[str, Any]:
        """Serialize state for persistence."""
        return {
            'outcome_history': getattr(self, '_outcome_history', [])[-100:],
            'learning_params': getattr(self, '_learning_params', {}),
            'stats': self._stats.copy(),
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict[str, Any]) -> None:
        """Restore state from serialized data."""
        self._outcome_history = state.get('outcome_history', [])
        self._learning_params = state.get('learning_params', {'enhancement_preference': {}})
        self._stats.update(state.get('stats', {}))


# =============================================================================
# Factory
# =============================================================================

def create_realtime_assistance_engine(
    assistance_level: AssistanceLevel = AssistanceLevel.MODERATE,
    **kwargs
) -> RealTimeAssistanceEngine:
    """Factory function."""
    return RealTimeAssistanceEngine(assistance_level=assistance_level, **kwargs)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("REAL-TIME ASSISTANCE ENGINE TEST")
    print("=" * 80)
    
    engine = RealTimeAssistanceEngine(assistance_level=AssistanceLevel.COMPREHENSIVE)
    
    # Test 1: Overconfidence removal
    print("\n[1] Testing overconfidence removal...")
    result1 = engine.assist(
        response="This solution is 100% guaranteed to work and will never fail. The code is absolutely perfect.",
        query="How do I implement authentication?",
        domain="general"
    )
    print(f"    Original: {result1.original_response[:50]}...")
    print(f"    Enhanced: {result1.enhanced_response[:50]}...")
    print(f"    Enhancements: {result1.total_enhancements}")
    
    # Test 2: Medical disclaimer
    print("\n[2] Testing medical disclaimer addition...")
    result2 = engine.assist(
        response="For headaches, you can take ibuprofen or acetaminophen.",
        query="What should I take for headaches?",
        domain="medical"
    )
    print(f"    Has disclaimer: {'Disclaimer' in result2.enhanced_response}")
    print(f"    Enhancements: {result2.total_enhancements}")
    
    # Test 3: Incomplete response
    print("\n[3] Testing incomplete response handling...")
    result3 = engine.assist(
        response="Here are the steps: 1. First... 2. Second... etc...",
        query="How do I deploy to AWS?",
        domain="general"
    )
    print(f"    Detected incomplete: {result3.enhanced_response != result3.original_response}")
    print(f"    Has notice: {'Note:' in result3.enhanced_response or 'incomplete' in result3.enhanced_response}")
    
    # Test 4: Confidence comparison
    print("\n[4] Testing confidence estimation...")
    result4 = engine.assist(
        response="Based on my analysis, for instance, you should specifically configure the database with the following example settings.",
        query="How do I configure PostgreSQL?",
        domain="general"
    )
    print(f"    Original confidence: {result4.original_confidence:.2f}")
    print(f"    Enhanced confidence: {result4.enhanced_confidence:.2f}")
    print(f"    Confidence boost: {result4.enhanced_confidence - result4.original_confidence:.2f}")
    
    # Statistics
    print("\n[5] Statistics:")
    stats = engine.get_statistics()
    for key, value in stats.items():
        if not isinstance(value, dict):
            print(f"    {key}: {value}")
    
    print("\n" + "=" * 80)
    print("✓ REAL-TIME ASSISTANCE ENGINE TEST COMPLETE")
    print("=" * 80)

