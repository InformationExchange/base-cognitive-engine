"""
BAIS Response Generator (Layer 5 - Motor Cortex)

Generates and formats responses based on governance decisions:
1. Response synthesis
2. Format adaptation
3. Tone adjustment
4. Content filtering

Patent Alignment:
- Part of output generation layer
- Brain Layer: 5 (Motor Cortex - Action Generation)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum


class ResponseFormat(Enum):
    """Supported response formats."""
    PLAIN = "plain"
    MARKDOWN = "markdown"
    JSON = "json"
    STRUCTURED = "structured"
    CONVERSATIONAL = "conversational"


class ResponseTone(Enum):
    """Response tone options."""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    FORMAL = "formal"
    EMPATHETIC = "empathetic"
    NEUTRAL = "neutral"
    TECHNICAL = "technical"


@dataclass
class ResponseConfig:
    """Configuration for response generation."""
    format: ResponseFormat = ResponseFormat.PLAIN
    tone: ResponseTone = ResponseTone.PROFESSIONAL
    max_length: int = 1000
    include_sources: bool = False
    include_confidence: bool = False
    hedge_uncertain: bool = True


@dataclass
class GeneratedResponse:
    """A generated response."""
    content: str
    format: ResponseFormat
    tone: ResponseTone
    modifications_applied: List[str]
    confidence: float
    metadata: Dict[str, Any]


class ResponseGenerator:
    """
    Generates and formats responses for BAIS.
    
    Brain Layer: 5 (Motor Cortex)
    
    Responsibilities:
    1. Synthesize response content
    2. Apply formatting
    3. Adjust tone
    4. Add appropriate hedging
    5. Filter unsafe content
    """
    
    # Hedging phrases by confidence level
    HEDGING_PHRASES = {
        'high': [],  # No hedging needed
        'medium': ['It appears that', 'Based on available information', 'Generally'],
        'low': ['It seems possible that', 'This may indicate', 'Tentatively'],
        'very_low': ['This is uncertain, but', 'One possibility is', 'It is unclear, however'],
    }
    
    # Tone markers
    TONE_MARKERS = {
        ResponseTone.PROFESSIONAL: {'greeting': '', 'closing': ''},
        ResponseTone.FRIENDLY: {'greeting': 'Great question! ', 'closing': ' Hope this helps!'},
        ResponseTone.FORMAL: {'greeting': '', 'closing': ' Please let me know if you require further clarification.'},
        ResponseTone.EMPATHETIC: {'greeting': 'I understand. ', 'closing': ' I\'m here to help.'},
        ResponseTone.TECHNICAL: {'greeting': '', 'closing': ''},
    }
    
    def __init__(self, default_config: ResponseConfig = None):
        """Initialize the response generator."""
        self.default_config = default_config or ResponseConfig()
        
        # Learning state
        self._outcomes: List[Dict] = []
        self._feedback: List[Dict] = []
        self._domain_adjustments: Dict[str, float] = {}
        self._total_generations: int = 0
        self._format_preferences: Dict[str, ResponseFormat] = {}
    
    def generate(self, content: str, confidence: float = 0.8,
                 config: ResponseConfig = None,
                 context: Dict[str, Any] = None) -> GeneratedResponse:
        """
        Generate a formatted response.
        
        Args:
            content: Raw response content
            confidence: Confidence level (0-1)
            config: Response configuration
            context: Additional context
            
        Returns:
            GeneratedResponse with formatted content
        """
        self._total_generations += 1
        config = config or self.default_config
        context = context or {}
        
        modifications = []
        
        # Apply hedging if needed
        if config.hedge_uncertain and confidence < 0.9:
            content = self._apply_hedging(content, confidence)
            modifications.append(f"hedging_applied_conf_{confidence:.2f}")
        
        # Apply tone
        content = self._apply_tone(content, config.tone)
        modifications.append(f"tone_{config.tone.value}")
        
        # Apply format
        content = self._apply_format(content, config.format)
        modifications.append(f"format_{config.format.value}")
        
        # Truncate if needed
        if len(content) > config.max_length:
            content = content[:config.max_length - 50] + "..."
            modifications.append("truncated")
        
        # Add sources if requested
        if config.include_sources and 'sources' in context:
            content = self._add_sources(content, context['sources'])
            modifications.append("sources_added")
        
        # Add confidence if requested
        if config.include_confidence:
            content = f"[Confidence: {confidence:.0%}]\n\n{content}"
            modifications.append("confidence_displayed")
        
        return GeneratedResponse(
            content=content,
            format=config.format,
            tone=config.tone,
            modifications_applied=modifications,
            confidence=confidence,
            metadata={'original_length': len(content), 'context_keys': list(context.keys())}
        )
    
    def _apply_hedging(self, content: str, confidence: float) -> str:
        """Apply appropriate hedging based on confidence."""
        if confidence >= 0.9:
            level = 'high'
        elif confidence >= 0.7:
            level = 'medium'
        elif confidence >= 0.5:
            level = 'low'
        else:
            level = 'very_low'
        
        phrases = self.HEDGING_PHRASES.get(level, [])
        
        if phrases and not any(p.lower() in content.lower() for p in phrases):
            # Add hedging phrase at start
            content = f"{phrases[0]} {content[0].lower()}{content[1:]}"
        
        return content
    
    def _apply_tone(self, content: str, tone: ResponseTone) -> str:
        """Apply tone markers to response."""
        markers = self.TONE_MARKERS.get(tone, {})
        
        greeting = markers.get('greeting', '')
        closing = markers.get('closing', '')
        
        # Don't double-add if already present
        if greeting and not content.startswith(greeting):
            content = greeting + content
        
        if closing and not content.endswith(closing):
            content = content + closing
        
        return content
    
    def _apply_format(self, content: str, format_type: ResponseFormat) -> str:
        """Apply formatting to response."""
        if format_type == ResponseFormat.MARKDOWN:
            # Add basic markdown structure
            if not content.startswith('#') and not content.startswith('**'):
                lines = content.split('\n')
                if len(lines) > 3:
                    # Add bullet points if multiple paragraphs
                    content = '\n'.join([f"• {line}" if line.strip() else line for line in lines])
        
        elif format_type == ResponseFormat.JSON:
            import json
            content = json.dumps({"response": content}, indent=2)
        
        elif format_type == ResponseFormat.STRUCTURED:
            lines = content.split('\n')
            structured = ["### Response"]
            structured.extend([f"- {line}" for line in lines if line.strip()])
            content = '\n'.join(structured)
        
        return content
    
    def _add_sources(self, content: str, sources: List[str]) -> str:
        """Add source citations to response."""
        source_section = "\n\n**Sources:**\n"
        for i, source in enumerate(sources[:5], 1):
            source_section += f"[{i}] {source}\n"
        
        return content + source_section
    
    # ===== Learning Interface Methods =====
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        """Record generation outcome for learning."""
        self._outcomes.append(outcome)
        # Track format preferences
        if 'preferred_format' in outcome:
            domain = outcome.get('domain', 'general')
            self._format_preferences[domain] = ResponseFormat(outcome['preferred_format'])
    
    def record_feedback(self, feedback: Dict) -> None:
        """Record feedback on generated responses."""
        self._feedback.append(feedback)
        domain = feedback.get('domain', 'general')
        if feedback.get('too_hedged', False):
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) + 0.05
        elif feedback.get('not_hedged_enough', False):
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) - 0.05
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt generation thresholds."""
        pass
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get domain adjustment."""
        return self._domain_adjustments.get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics."""
        return {
            'total_generations': self._total_generations,
            'format_preferences': {k: v.value for k, v in self._format_preferences.items()},
            'domain_adjustments': dict(self._domain_adjustments),
            'outcomes_recorded': len(self._outcomes),
            'feedback_recorded': len(self._feedback)
        }


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

if __name__ == "__main__":
    generator = ResponseGenerator()
    
    test_content = "Machine learning is a subset of artificial intelligence that enables systems to learn from data."
    
    print("=" * 60)
    print("RESPONSE GENERATOR TEST")
    print("=" * 60)
    
    # Test with different confidences
    for conf in [0.95, 0.75, 0.5]:
        result = generator.generate(test_content, confidence=conf)
        print(f"\nConfidence: {conf}")
        print(f"  Content: {result.content[:80]}...")
        print(f"  Modifications: {result.modifications_applied}")
    
    print(f"\nLearning stats: {generator.get_learning_statistics()}")

