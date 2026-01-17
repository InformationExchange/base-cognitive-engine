"""
BASE Cognitive Governance Engine - Semantic Mode Selector
Auto-detects appropriate mode with manual override capability

Features:
- Analyzes query to detect task type
- Auto-selects appropriate BASE mode
- Manual override via prompt keywords or API parameter
- Learning from user corrections

Patent: NOVEL-48 (Semantic Mode Selector)
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Task Types
# =============================================================================

class TaskType(Enum):
    """Detected task types."""
    CODE_IMPLEMENTATION = "code_implementation"
    CODE_FIX = "code_fix"
    CODE_REVIEW = "code_review"
    CONTENT_GENERATION = "content_generation"
    REPORT_WRITING = "report_writing"
    ANALYSIS = "analysis"
    CONVERSATION = "conversation"
    QUESTION_ANSWER = "question_answer"
    DATA_PROCESSING = "data_processing"
    UNKNOWN = "unknown"


class ContentType(Enum):
    """Detected content types."""
    CODE = "code"
    PROSE = "prose"
    DATA = "data"
    MIXED = "mixed"
    UNKNOWN = "unknown"


# =============================================================================
# Mode Mapping
# =============================================================================

# Import BASEMode - but define string version for independence
class RecommendedMode(Enum):
    AUDIT_ONLY = "audit_only"
    AUDIT_AND_REMEDIATE = "audit_and_remediate"
    DIRECT_ASSISTANCE = "direct_assistance"


# Default mappings
TASK_TO_MODE: Dict[TaskType, RecommendedMode] = {
    TaskType.CODE_IMPLEMENTATION: RecommendedMode.AUDIT_AND_REMEDIATE,
    TaskType.CODE_FIX: RecommendedMode.AUDIT_AND_REMEDIATE,
    TaskType.CODE_REVIEW: RecommendedMode.AUDIT_ONLY,
    TaskType.CONTENT_GENERATION: RecommendedMode.DIRECT_ASSISTANCE,
    TaskType.REPORT_WRITING: RecommendedMode.DIRECT_ASSISTANCE,
    TaskType.ANALYSIS: RecommendedMode.AUDIT_ONLY,
    TaskType.CONVERSATION: RecommendedMode.DIRECT_ASSISTANCE,
    TaskType.QUESTION_ANSWER: RecommendedMode.DIRECT_ASSISTANCE,
    TaskType.DATA_PROCESSING: RecommendedMode.AUDIT_AND_REMEDIATE,
    TaskType.UNKNOWN: RecommendedMode.AUDIT_ONLY,
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ModeSelection:
    """Result of mode selection."""
    mode: RecommendedMode
    source: str  # "auto", "prompt_override", "api_override", "default"
    task_type: TaskType
    content_type: ContentType
    confidence: float
    reasoning: str
    alternatives: List[RecommendedMode] = field(default_factory=list)


@dataclass 
class OverrideRequest:
    """Manual override request."""
    mode: Optional[RecommendedMode]
    source: str  # "prompt", "api"
    explicit: bool  # Was it explicitly stated?


# =============================================================================
# Semantic Mode Selector
# =============================================================================

class SemanticModeSelector:
    """
    Automatically selects BASE mode based on query/content analysis.
    
    Priority:
    1. Explicit API override (highest)
    2. Prompt keyword override
    3. Auto-detected from content
    4. Default fallback
    """
    
    def __init__(self):
        # Keywords for task detection
        self._code_keywords = {
            'implement', 'create', 'build', 'develop', 'code', 'program',
            'function', 'class', 'method', 'api', 'endpoint', 'database',
            'fix', 'debug', 'refactor', 'optimize', 'deploy', 'test'
        }
        
        self._content_keywords = {
            'write', 'draft', 'compose', 'generate', 'create content',
            'blog', 'article', 'post', 'email', 'message', 'letter',
            'report', 'summary', 'documentation', 'readme'
        }
        
        self._analysis_keywords = {
            'analyze', 'review', 'audit', 'check', 'evaluate', 'assess',
            'compare', 'inspect', 'examine', 'investigate'
        }
        
        self._question_keywords = {
            'what', 'how', 'why', 'when', 'where', 'explain', 'describe',
            'tell me', 'help me understand', 'can you'
        }
        
        # Prompt override patterns
        self._override_patterns = {
            RecommendedMode.AUDIT_ONLY: [
                r'\baudit[- ]?only\b',
                r'\bjust[- ]?audit\b',
                r'\breport[- ]?only\b',
                r'\bdon\'?t[- ]?block\b',
                r'\bno[- ]?blocking\b'
            ],
            RecommendedMode.AUDIT_AND_REMEDIATE: [
                r'\benforce\b',
                r'\bstrict[- ]?mode\b',
                r'\bblock[- ]?on[- ]?issues?\b',
                r'\bmust[- ]?complete\b',
                r'\bverify[- ]?completion\b',
                r'\bexecution[- ]?proof\b'
            ],
            RecommendedMode.DIRECT_ASSISTANCE: [
                r'\benhance\b',
                r'\bpolish\b',
                r'\bimprove[- ]?response\b',
                r'\bdirect[- ]?assist(ance)?\b',
                r'\breal[- ]?time\b'
            ]
        }
        
        # Statistics
        self._stats = {
            'total_selections': 0,
            'auto_detected': 0,
            'prompt_override': 0,
            'api_override': 0,
            'by_mode': {m.value: 0 for m in RecommendedMode}
        }
        
        # Learning from corrections
        self._correction_history: List[Dict] = []
        self._outcomes = []

    def select_mode(
        self,
        query: str,
        response: Optional[str] = None,
        api_mode_override: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> ModeSelection:
        """
        Select appropriate BASE mode.
        
        Args:
            query: User's query/request
            response: LLM response (if available)
            api_mode_override: Explicit mode from API parameter
            context: Additional context
        
        Returns:
            ModeSelection with recommended mode and reasoning
        """
        self._stats['total_selections'] += 1
        
        # Priority 1: API override
        if api_mode_override:
            mode = self._parse_mode_string(api_mode_override)
            if mode:
                self._stats['api_override'] += 1
                self._stats['by_mode'][mode.value] += 1
                return ModeSelection(
                    mode=mode,
                    source="api_override",
                    task_type=TaskType.UNKNOWN,
                    content_type=ContentType.UNKNOWN,
                    confidence=1.0,
                    reasoning=f"Explicit API override: {api_mode_override}"
                )
        
        # Priority 2: Prompt keyword override
        prompt_override = self._detect_prompt_override(query)
        if prompt_override and prompt_override.explicit:
            self._stats['prompt_override'] += 1
            self._stats['by_mode'][prompt_override.mode.value] += 1
            return ModeSelection(
                mode=prompt_override.mode,
                source="prompt_override",
                task_type=self._detect_task_type(query),
                content_type=self._detect_content_type(query, response),
                confidence=0.95,
                reasoning=f"Prompt keyword detected: requesting {prompt_override.mode.value}"
            )
        
        # Priority 3: Auto-detect from content
        task_type = self._detect_task_type(query)
        content_type = self._detect_content_type(query, response)
        
        # Get mode from task type
        mode = TASK_TO_MODE.get(task_type, RecommendedMode.AUDIT_ONLY)
        
        # Adjust based on content type
        if content_type == ContentType.CODE and mode == RecommendedMode.DIRECT_ASSISTANCE:
            # Code should use enforcement, not enhancement
            mode = RecommendedMode.AUDIT_AND_REMEDIATE
        
        # Adjust based on context
        if context:
            domain = context.get('domain', '')
            if domain in ['medical', 'legal', 'financial']:
                # Sensitive domains need stricter handling
                if mode == RecommendedMode.DIRECT_ASSISTANCE:
                    mode = RecommendedMode.AUDIT_AND_REMEDIATE
        
        # Calculate confidence
        confidence = self._calculate_confidence(query, task_type, content_type)
        
        # Get alternatives
        alternatives = [m for m in RecommendedMode if m != mode]
        
        self._stats['auto_detected'] += 1
        self._stats['by_mode'][mode.value] += 1
        
        return ModeSelection(
            mode=mode,
            source="auto",
            task_type=task_type,
            content_type=content_type,
            confidence=confidence,
            reasoning=self._generate_reasoning(task_type, content_type, mode),
            alternatives=alternatives
        )
    
    def _parse_mode_string(self, mode_str: str) -> Optional[RecommendedMode]:
        """Parse mode string to enum."""
        mode_str = mode_str.lower().strip()
        
        mappings = {
            'audit_only': RecommendedMode.AUDIT_ONLY,
            'audit-only': RecommendedMode.AUDIT_ONLY,
            'audit': RecommendedMode.AUDIT_ONLY,
            'audit_and_remediate': RecommendedMode.AUDIT_AND_REMEDIATE,
            'audit-and-remediate': RecommendedMode.AUDIT_AND_REMEDIATE,
            'enforce': RecommendedMode.AUDIT_AND_REMEDIATE,
            'enforcement': RecommendedMode.AUDIT_AND_REMEDIATE,
            'strict': RecommendedMode.AUDIT_AND_REMEDIATE,
            'direct_assistance': RecommendedMode.DIRECT_ASSISTANCE,
            'direct-assistance': RecommendedMode.DIRECT_ASSISTANCE,
            'enhance': RecommendedMode.DIRECT_ASSISTANCE,
            'enhancement': RecommendedMode.DIRECT_ASSISTANCE,
            'assist': RecommendedMode.DIRECT_ASSISTANCE,
        }
        
        return mappings.get(mode_str)
    
    def _detect_prompt_override(self, query: str) -> Optional[OverrideRequest]:
        """Detect mode override from prompt keywords."""
        query_lower = query.lower()
        
        for mode, patterns in self._override_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return OverrideRequest(
                        mode=mode,
                        source="prompt",
                        explicit=True
                    )
        
        return None
    
    def _detect_task_type(self, query: str) -> TaskType:
        """Detect task type from query."""
        query_lower = query.lower()
        
        # Check for code-related tasks
        code_score = sum(1 for kw in self._code_keywords if kw in query_lower)
        
        # Check for content generation
        content_score = sum(1 for kw in self._content_keywords if kw in query_lower)
        
        # Check for analysis
        analysis_score = sum(1 for kw in self._analysis_keywords if kw in query_lower)
        
        # Check for Q&A
        question_score = sum(1 for kw in self._question_keywords if kw in query_lower)
        
        # Determine type
        scores = {
            TaskType.CODE_IMPLEMENTATION: code_score,
            TaskType.CONTENT_GENERATION: content_score,
            TaskType.ANALYSIS: analysis_score,
            TaskType.QUESTION_ANSWER: question_score
        }
        
        if max(scores.values()) == 0:
            return TaskType.UNKNOWN
        
        # Special cases
        if 'fix' in query_lower or 'bug' in query_lower or 'error' in query_lower:
            return TaskType.CODE_FIX
        if 'review' in query_lower and code_score > 0:
            return TaskType.CODE_REVIEW
        if 'report' in query_lower:
            return TaskType.REPORT_WRITING
        
        return max(scores, key=scores.get)
    
    def _detect_content_type(self, query: str, response: Optional[str]) -> ContentType:
        """Detect content type."""
        text = f"{query} {response or ''}"
        
        # Check for code indicators
        code_indicators = ['```', 'def ', 'class ', 'function ', 'import ', 'const ', 'var ', 'let ']
        has_code = any(ind in text for ind in code_indicators)
        
        # Check for prose indicators
        prose_len = len(text.split())
        has_prose = prose_len > 50 and not has_code
        
        if has_code and has_prose:
            return ContentType.MIXED
        elif has_code:
            return ContentType.CODE
        elif has_prose:
            return ContentType.PROSE
        else:
            return ContentType.UNKNOWN
    
    def _calculate_confidence(
        self, 
        query: str, 
        task_type: TaskType, 
        content_type: ContentType
    ) -> float:
        """Calculate confidence in mode selection."""
        confidence = 0.5  # Base
        
        # Task type clarity
        if task_type != TaskType.UNKNOWN:
            confidence += 0.2
        
        # Content type clarity
        if content_type != ContentType.UNKNOWN:
            confidence += 0.15
        
        # Query length (longer = more context = more confidence)
        if len(query) > 100:
            confidence += 0.1
        
        # Keyword density
        query_lower = query.lower()
        total_keywords = (
            sum(1 for kw in self._code_keywords if kw in query_lower) +
            sum(1 for kw in self._content_keywords if kw in query_lower) +
            sum(1 for kw in self._analysis_keywords if kw in query_lower)
        )
        if total_keywords >= 3:
            confidence += 0.1
        
        return min(0.95, confidence)
    
    def _generate_reasoning(
        self, 
        task_type: TaskType, 
        content_type: ContentType, 
        mode: RecommendedMode
    ) -> str:
        """Generate human-readable reasoning."""
        reasons = []
        
        if task_type == TaskType.CODE_IMPLEMENTATION:
            reasons.append("Detected code implementation task")
        elif task_type == TaskType.CODE_FIX:
            reasons.append("Detected code fix/debug task")
        elif task_type == TaskType.CONTENT_GENERATION:
            reasons.append("Detected content generation task")
        elif task_type == TaskType.REPORT_WRITING:
            reasons.append("Detected report writing task")
        elif task_type == TaskType.ANALYSIS:
            reasons.append("Detected analysis task")
        
        if content_type == ContentType.CODE:
            reasons.append("Content is primarily code")
        elif content_type == ContentType.PROSE:
            reasons.append("Content is primarily text/prose")
        
        mode_reason = {
            RecommendedMode.AUDIT_AND_REMEDIATE: "Using enforcement to verify completion",
            RecommendedMode.DIRECT_ASSISTANCE: "Using enhancement for content polish",
            RecommendedMode.AUDIT_ONLY: "Using audit-only for reporting"
        }
        reasons.append(mode_reason.get(mode, ""))
        
        return "; ".join(r for r in reasons if r)
    
    def record_correction(
        self, 
        query: str, 
        auto_mode: RecommendedMode, 
        corrected_mode: RecommendedMode
    ) -> None:
        """Record when user corrects auto-selection (for learning)."""
        self._correction_history.append({
            'query_preview': query[:100],
            'auto_mode': auto_mode.value,
            'corrected_mode': corrected_mode.value
        })
        logger.info(f"[SemanticSelector] Correction recorded: {auto_mode.value} → {corrected_mode.value}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get selector statistics."""
        return {
            **self._stats,
            'corrections': len(self._correction_history),
            'learning_params': getattr(self, '_learning_params', {}),
            'history_size': len(getattr(self, '_outcome_history', []))
        }
    
    # =========================================================================
    # Learning Interface (5/5 methods)
    # =========================================================================
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict[str, Any]) -> None:
        """Record mode selection outcome for learning."""
        if not hasattr(self, '_outcome_history'):
            self._outcome_history: List[Dict] = []
        
        self._stats['total_selections'] += 1
        
        self._outcome_history.append({
            'timestamp': datetime.now().isoformat(),
            **outcome
        })
        if len(self._outcome_history) > 1000:
            self._outcome_history = self._outcome_history[-500:]
    
    def learn_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """Learn from feedback to improve mode selection."""
        if not hasattr(self, '_learning_params'):
            self._learning_params = {'mode_adjustments': {}}
        
        was_correct = feedback.get('was_correct', True)
        selected_mode = feedback.get('mode')
        correct_mode = feedback.get('correct_mode')
        
        if not was_correct and selected_mode and correct_mode:
            # Record this as a correction
            self.record_correction(
                query=feedback.get('query', ''),
                auto_mode=RecommendedMode(selected_mode) if selected_mode else None,
                corrected_mode=RecommendedMode(correct_mode) if correct_mode else None
            )
    
    def serialize_state(self) -> Dict[str, Any]:
        """Serialize state for persistence."""
        return {
            'outcome_history': getattr(self, '_outcome_history', [])[-100:],
            'learning_params': getattr(self, '_learning_params', {}),
            'stats': self._stats.copy(),
            'correction_history': self._correction_history[-100:],
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict[str, Any]) -> None:
        """Restore state from serialized data."""
        self._outcome_history = state.get('outcome_history', [])
        self._learning_params = state.get('learning_params', {'mode_adjustments': {}})
        self._stats.update(state.get('stats', {}))
        self._correction_history = state.get('correction_history', [])


# =============================================================================
# Integration Helper
# =============================================================================

def integrate_with_orchestrator(orchestrator, selector: SemanticModeSelector):
    """
    Integrate SemanticModeSelector with BASEv2Orchestrator.
    
    Modifies orchestrator.govern() to auto-select mode.
    """
    original_govern = orchestrator.govern
    
    def enhanced_govern(
        query: str,
        response: str,
        evidence: Optional[List[str]] = None,
        domain: str = "general",
        context: Optional[Dict] = None,
        llm_provider = None,
        mode_override: Optional[str] = None  # NEW: API override
    ):
        # Auto-select mode
        selection = selector.select_mode(
            query=query,
            response=response,
            api_mode_override=mode_override,
            context={'domain': domain, **(context or {})}
        )
        
        # Set mode on orchestrator
        from core.governance_modes import BASEMode
        mode_map = {
            'audit_only': BASEMode.AUDIT_ONLY,
            'audit_and_remediate': BASEMode.AUDIT_AND_REMEDIATE,
            'direct_assistance': BASEMode.DIRECT_ASSISTANCE
        }
        orchestrator.set_mode(mode_map[selection.mode.value])
        
        # Log selection
        logger.info(f"[SemanticSelector] Mode: {selection.mode.value} ({selection.source})")
        
        # Call original
        result = original_govern(query, response, evidence, domain, context, llm_provider)
        
        # Add selection info to result
        result._mode_selection = selection
        
        return result
    
    orchestrator.govern = enhanced_govern
    return orchestrator


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 75)
    print("SEMANTIC MODE SELECTOR TEST")
    print("=" * 75)
    
    selector = SemanticModeSelector()
    
    # Test cases
    test_cases = [
        ("Implement a REST API for user authentication", None, None),
        ("Write a summary report of Q3 sales performance", None, None),
        ("Analyze the security vulnerabilities in this code", None, None),
        ("What is the best way to handle errors in Python?", None, None),
        ("Fix the bug in the login function", None, None),
        # With API override
        ("Implement a feature", None, "audit_only"),
        # With prompt override
        ("Implement a feature, use strict mode and verify completion", None, None),
        ("Write a report, just audit don't block", None, None),
    ]
    
    print("\nTest Results:")
    print("-" * 75)
    
    for query, response, api_override in test_cases:
        result = selector.select_mode(query, response, api_override)
        print(f"\nQuery: \"{query[:50]}...\"")
        if api_override:
            print(f"  API Override: {api_override}")
        print(f"  Mode: {result.mode.value}")
        print(f"  Source: {result.source}")
        print(f"  Task Type: {result.task_type.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Reasoning: {result.reasoning}")
    
    # Statistics
    print("\n" + "-" * 75)
    print("Statistics:")
    stats = selector.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 75)
    print("✓ SEMANTIC MODE SELECTOR TEST COMPLETE")
    print("=" * 75)

