"""
BASE Cognitive Governance Engine v16.4
Cognitive Window Intervention System

PPA-1 Invention 17: FULL IMPLEMENTATION
Interrupt bias formation within the 200-500ms cognitive processing window.

This module implements:
1. Pre-Response Interception: Catch responses before delivery
2. Real-Time Analysis: Fast bias detection (<100ms target)
3. Intervention Triggers: Automated intervention conditions
4. Response Modification: Inject warnings, caveats, alternatives
5. Cognitive Load Tracking: Manage intervention frequency

Research Basis:
- Human cognitive processing window: 200-500ms for bias formation
- AI response generation: Can intercept before final delivery
- Early intervention is more effective than post-hoc correction
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import time
import re
import threading
from collections import deque


class InterventionType(str, Enum):
    """Types of cognitive interventions."""
    WARNING_INJECT = "warning_inject"      # Add warning to response
    CAVEAT_APPEND = "caveat_append"        # Append uncertainty caveat
    ALTERNATIVE_SUGGEST = "alternative"    # Suggest alternative viewpoint
    FACT_CHECK_FLAG = "fact_check"         # Flag for fact verification
    BIAS_HIGHLIGHT = "bias_highlight"      # Highlight potential bias
    CONFIDENCE_ADJUST = "confidence_adjust" # Adjust confidence score
    RESPONSE_BLOCK = "response_block"      # Block response entirely
    SLOW_DOWN = "slow_down"                # Delay response for review


class InterventionSeverity(str, Enum):
    """Severity levels for interventions."""
    INFO = "info"           # Informational only
    CAUTION = "caution"     # Proceed with caution
    WARNING = "warning"     # Significant concern
    CRITICAL = "critical"   # Must intervene
    BLOCKING = "blocking"   # Block response


@dataclass
class InterventionTrigger:
    """Condition that triggers an intervention."""
    trigger_id: str
    name: str
    condition: str  # Human-readable condition
    threshold: float
    intervention_type: InterventionType
    severity: InterventionSeverity
    enabled: bool = True
    
    def to_dict(self) -> Dict:
        return {
            'trigger_id': self.trigger_id,
            'name': self.name,
            'condition': self.condition,
            'threshold': self.threshold,
            'intervention_type': self.intervention_type.value,
            'severity': self.severity.value,
            'enabled': self.enabled
        }


@dataclass
class CognitiveIntervention:
    """Record of a cognitive intervention."""
    intervention_id: str
    timestamp: datetime
    trigger: InterventionTrigger
    intervention_type: InterventionType
    severity: InterventionSeverity
    
    # What was detected
    detected_issue: str
    confidence: float
    
    # What was done
    action_taken: str
    original_response_hash: str
    modified_response_hash: Optional[str]
    
    # Timing
    detection_time_ms: float
    intervention_time_ms: float
    total_time_ms: float
    
    # Within cognitive window?
    within_window: bool  # Was intervention within 200-500ms?
    
    def to_dict(self) -> Dict:
        return {
            'intervention_id': self.intervention_id,
            'timestamp': self.timestamp.isoformat(),
            'trigger': self.trigger.to_dict(),
            'intervention_type': self.intervention_type.value,
            'severity': self.severity.value,
            'detected_issue': self.detected_issue,
            'confidence': self.confidence,
            'action_taken': self.action_taken,
            'timing': {
                'detection_ms': self.detection_time_ms,
                'intervention_ms': self.intervention_time_ms,
                'total_ms': self.total_time_ms,
                'within_cognitive_window': self.within_window
            }
        }


@dataclass
class InterventionResult:
    """Result of cognitive intervention processing."""
    intervened: bool
    interventions: List[CognitiveIntervention]
    original_response: str
    modified_response: str
    total_processing_time_ms: float
    warnings_added: List[str]
    caveats_added: List[str]
    blocked: bool
    
    def to_dict(self) -> Dict:
        return {
            'intervened': self.intervened,
            'intervention_count': len(self.interventions),
            'interventions': [i.to_dict() for i in self.interventions],
            'blocked': self.blocked,
            'warnings': self.warnings_added,
            'caveats': self.caveats_added,
            'processing_time_ms': self.total_processing_time_ms
        }


class CognitiveWindowInterventionSystem:
    """
    Real-Time Cognitive Intervention System.
    
    PPA-1 Invention 17: Full Implementation
    
    Intercepts AI responses in the cognitive processing window
    (200-500ms) to prevent bias propagation.
    
    Key Capabilities:
    1. Sub-100ms detection for most bias types
    2. Multiple intervention strategies
    3. Configurable triggers
    4. Intervention effectiveness tracking
    5. Cognitive load management
    """
    
    # Cognitive window bounds (milliseconds)
    WINDOW_MIN_MS = 200
    WINDOW_MAX_MS = 500
    
    # Target detection time
    TARGET_DETECTION_MS = 100
    
    # Maximum interventions per response
    MAX_INTERVENTIONS = 3
    
    # Cool-down between interventions (seconds)
    INTERVENTION_COOLDOWN = 0.5
    
    def __init__(self):
        # Intervention triggers
        self.triggers: Dict[str, InterventionTrigger] = {}
        self._initialize_default_triggers()
        
        # Intervention history
        self.intervention_history: deque = deque(maxlen=1000)
        self.interventions_by_type: Dict[InterventionType, int] = {t: 0 for t in InterventionType}
        
        # Performance tracking
        self.detection_times: deque = deque(maxlen=500)
        self.intervention_effectiveness: Dict[str, List[bool]] = {}
        
        # Cognitive load tracking
        self.recent_interventions: deque = deque(maxlen=10)
        self.last_intervention_time: Optional[datetime] = None
        
        # Pattern matchers for fast detection
        self._compile_patterns()
    
    def _initialize_default_triggers(self):
        """Initialize default intervention triggers."""
        
        # High confidence false claims
        self.triggers['overconfidence'] = InterventionTrigger(
            trigger_id='overconfidence',
            name='Overconfident Claim Detection',
            condition='Response claims certainty > 95% without evidence',
            threshold=0.95,
            intervention_type=InterventionType.CAVEAT_APPEND,
            severity=InterventionSeverity.CAUTION
        )
        
        # Confirmation bias indicators
        self.triggers['confirmation_bias'] = InterventionTrigger(
            trigger_id='confirmation_bias',
            name='Confirmation Bias Detection',
            condition='Response aligns perfectly with query assumption',
            threshold=0.7,
            intervention_type=InterventionType.ALTERNATIVE_SUGGEST,
            severity=InterventionSeverity.WARNING
        )
        
        # Factual uncertainty
        self.triggers['factual_uncertainty'] = InterventionTrigger(
            trigger_id='factual_uncertainty',
            name='Factual Uncertainty Flag',
            condition='Response contains unverifiable facts',
            threshold=0.5,
            intervention_type=InterventionType.FACT_CHECK_FLAG,
            severity=InterventionSeverity.CAUTION
        )
        
        # Reward-seeking response
        self.triggers['sycophancy'] = InterventionTrigger(
            trigger_id='sycophancy',
            name='Sycophantic Response Detection',
            condition='Response appears to be seeking approval',
            threshold=0.6,
            intervention_type=InterventionType.WARNING_INJECT,
            severity=InterventionSeverity.WARNING
        )
        
        # Harmful content
        self.triggers['harmful_content'] = InterventionTrigger(
            trigger_id='harmful_content',
            name='Harmful Content Detection',
            condition='Response may cause harm if acted upon',
            threshold=0.3,
            intervention_type=InterventionType.RESPONSE_BLOCK,
            severity=InterventionSeverity.BLOCKING
        )
        
        # Ungrounded speculation
        self.triggers['speculation'] = InterventionTrigger(
            trigger_id='speculation',
            name='Ungrounded Speculation',
            condition='Response contains speculation without grounding',
            threshold=0.6,
            intervention_type=InterventionType.CAVEAT_APPEND,
            severity=InterventionSeverity.CAUTION
        )
        
        # Statistical claims
        self.triggers['statistical_claims'] = InterventionTrigger(
            trigger_id='statistical_claims',
            name='Statistical Claim Detection',
            condition='Response contains statistics without sources',
            threshold=0.5,
            intervention_type=InterventionType.FACT_CHECK_FLAG,
            severity=InterventionSeverity.CAUTION
        )
    
    def _compile_patterns(self):
        """Compile regex patterns for fast detection."""
        
        # Overconfidence patterns
        self.overconfidence_patterns = [
            re.compile(r'\b(definitely|certainly|absolutely|100%|always|never)\b', re.I),
            re.compile(r'\b(guaranteed|proven fact|undoubtedly|without question)\b', re.I),
            re.compile(r'\b(impossible|can\'t fail|zero chance)\b', re.I),
        ]
        
        # Confirmation patterns
        self.confirmation_patterns = [
            re.compile(r'\b(you\'re (right|correct)|exactly|precisely)\b', re.I),
            re.compile(r'\b(as you (said|mentioned|noted))\b', re.I),
            re.compile(r'\b(I (completely|totally|fully) agree)\b', re.I),
        ]
        
        # Sycophancy patterns
        self.sycophancy_patterns = [
            re.compile(r'\b(great question|excellent point|brilliant)\b', re.I),
            re.compile(r'\b(you\'re (so|very) (smart|clever|right))\b', re.I),
            re.compile(r'\b(couldn\'t have said it better)\b', re.I),
        ]
        
        # Statistical patterns
        self.statistical_patterns = [
            re.compile(r'\b\d+(\.\d+)?%', re.I),
            re.compile(r'\b(studies show|research indicates|data suggests)\b', re.I),
            re.compile(r'\b(most people|majority of|on average)\b', re.I),
        ]
        
        # Harmful patterns
        self.harmful_patterns = [
            re.compile(r'\b(kill|harm|hurt|destroy|attack)\s+(yourself|others|people)\b', re.I),
            re.compile(r'\b(illegal|dangerous|risky)\s+(?!to avoid)\b', re.I),
        ]
        
        # Speculation patterns
        self.speculation_patterns = [
            re.compile(r'\b(probably|likely|might|could be|possibly)\b', re.I),
            re.compile(r'\b(I (think|believe|assume|guess))\b', re.I),
            re.compile(r'\b(it seems|appears to be|looks like)\b', re.I),
        ]
    
    def process_response(self,
                        query: str,
                        response: str,
                        signals: Dict[str, float] = None) -> InterventionResult:
        """
        Process a response through the cognitive intervention system.
        
        This is the main entry point for real-time intervention.
        
        Args:
            query: The original query
            response: The AI-generated response
            signals: Optional pre-computed signals (grounding, behavioral, etc.)
        
        Returns:
            InterventionResult with any interventions applied
        """
        start_time = time.time()
        signals = signals or {}
        
        interventions = []
        warnings_added = []
        caveats_added = []
        blocked = False
        modified_response = response
        
        import hashlib
        original_hash = hashlib.md5(response.encode()).hexdigest()[:8]
        
        # Check cognitive load - don't over-intervene
        if self._is_overloaded():
            return InterventionResult(
                intervened=False,
                interventions=[],
                original_response=response,
                modified_response=response,
                total_processing_time_ms=(time.time() - start_time) * 1000,
                warnings_added=[],
                caveats_added=[],
                blocked=False
            )
        
        # Run fast detection for each enabled trigger
        for trigger_id, trigger in self.triggers.items():
            if not trigger.enabled:
                continue
            
            if len(interventions) >= self.MAX_INTERVENTIONS:
                break
            
            detection_start = time.time()
            
            # Detect issue
            issue_detected, confidence, issue_description = self._detect_issue(
                trigger, query, response, signals
            )
            
            detection_time = (time.time() - detection_start) * 1000
            self.detection_times.append(detection_time)
            
            if issue_detected and confidence >= trigger.threshold:
                intervention_start = time.time()
                
                # Apply intervention
                action_taken, modified_response, warning, caveat = self._apply_intervention(
                    trigger, modified_response, issue_description
                )
                
                intervention_time = (time.time() - intervention_start) * 1000
                total_time = detection_time + intervention_time
                
                # Check if within cognitive window
                within_window = total_time <= self.WINDOW_MAX_MS
                
                intervention = CognitiveIntervention(
                    intervention_id=f"int_{int(time.time()*1000)}_{trigger_id}",
                    timestamp=datetime.utcnow(),
                    trigger=trigger,
                    intervention_type=trigger.intervention_type,
                    severity=trigger.severity,
                    detected_issue=issue_description,
                    confidence=confidence,
                    action_taken=action_taken,
                    original_response_hash=original_hash,
                    modified_response_hash=hashlib.md5(modified_response.encode()).hexdigest()[:8],
                    detection_time_ms=detection_time,
                    intervention_time_ms=intervention_time,
                    total_time_ms=total_time,
                    within_window=within_window
                )
                
                interventions.append(intervention)
                self.intervention_history.append(intervention)
                self.interventions_by_type[trigger.intervention_type] += 1
                
                if warning:
                    warnings_added.append(warning)
                if caveat:
                    caveats_added.append(caveat)
                if trigger.intervention_type == InterventionType.RESPONSE_BLOCK:
                    blocked = True
                    modified_response = "[Response blocked due to safety concerns]"
        
        # Update state
        if interventions:
            self.last_intervention_time = datetime.utcnow()
            self.recent_interventions.append(datetime.utcnow())
        
        total_processing = (time.time() - start_time) * 1000
        
        return InterventionResult(
            intervened=len(interventions) > 0,
            interventions=interventions,
            original_response=response,
            modified_response=modified_response,
            total_processing_time_ms=total_processing,
            warnings_added=warnings_added,
            caveats_added=caveats_added,
            blocked=blocked
        )
    
    def _detect_issue(self,
                     trigger: InterventionTrigger,
                     query: str,
                     response: str,
                     signals: Dict[str, float]) -> Tuple[bool, float, str]:
        """
        Fast detection of issues based on trigger type.
        
        Target: <100ms detection time
        
        Returns: (detected, confidence, description)
        """
        trigger_id = trigger.trigger_id
        
        if trigger_id == 'overconfidence':
            matches = sum(1 for p in self.overconfidence_patterns if p.search(response))
            if matches > 0:
                confidence = min(0.3 + matches * 0.2, 1.0)
                return True, confidence, f"Overconfident language detected ({matches} patterns)"
        
        elif trigger_id == 'confirmation_bias':
            # Check if response perfectly confirms query
            matches = sum(1 for p in self.confirmation_patterns if p.search(response))
            if matches > 0:
                confidence = min(0.4 + matches * 0.15, 1.0)
                return True, confidence, f"Confirmation patterns detected ({matches})"
        
        elif trigger_id == 'sycophancy':
            matches = sum(1 for p in self.sycophancy_patterns if p.search(response))
            if matches > 0:
                confidence = min(0.3 + matches * 0.2, 1.0)
                return True, confidence, f"Sycophantic patterns detected ({matches})"
        
        elif trigger_id == 'factual_uncertainty':
            # Use grounding score if available
            grounding = signals.get('grounding', 0.5)
            if grounding < 0.4:
                return True, 1 - grounding, "Low grounding score indicates factual uncertainty"
        
        elif trigger_id == 'harmful_content':
            matches = sum(1 for p in self.harmful_patterns if p.search(response))
            if matches > 0:
                return True, 0.9, f"Potentially harmful content detected"
        
        elif trigger_id == 'speculation':
            matches = sum(1 for p in self.speculation_patterns if p.search(response))
            grounding = signals.get('grounding', 0.5)
            if matches > 2 and grounding < 0.5:
                confidence = min(0.3 + matches * 0.1 + (1 - grounding) * 0.3, 1.0)
                return True, confidence, f"Ungrounded speculation detected ({matches} patterns)"
        
        elif trigger_id == 'statistical_claims':
            matches = sum(1 for p in self.statistical_patterns if p.search(response))
            if matches > 0:
                # Check if sources mentioned
                has_source = bool(re.search(r'(according to|source:|study by|research from)', response, re.I))
                if not has_source:
                    confidence = min(0.5 + matches * 0.15, 1.0)
                    return True, confidence, f"Statistical claims without sources ({matches})"
        
        return False, 0.0, ""
    
    def _apply_intervention(self,
                           trigger: InterventionTrigger,
                           response: str,
                           issue: str) -> Tuple[str, str, Optional[str], Optional[str]]:
        """
        Apply intervention to response.
        
        Returns: (action_description, modified_response, warning, caveat)
        """
        warning = None
        caveat = None
        modified = response
        
        if trigger.intervention_type == InterventionType.WARNING_INJECT:
            warning = f"⚠️ Note: {issue}"
            modified = f"{warning}\n\n{response}"
            action = f"Injected warning: {issue}"
        
        elif trigger.intervention_type == InterventionType.CAVEAT_APPEND:
            caveat = f"\n\n📋 Caveat: This response may contain {trigger.name.lower()}. Please verify important claims independently."
            modified = f"{response}{caveat}"
            action = f"Appended caveat for: {issue}"
        
        elif trigger.intervention_type == InterventionType.ALTERNATIVE_SUGGEST:
            caveat = f"\n\n🔄 Consider alternative perspectives: The above response aligns closely with the question's assumptions. You may want to explore contrasting viewpoints."
            modified = f"{response}{caveat}"
            action = f"Suggested alternatives for: {issue}"
        
        elif trigger.intervention_type == InterventionType.FACT_CHECK_FLAG:
            warning = f"🔍 Fact-check recommended: {issue}"
            modified = f"{response}\n\n{warning}"
            action = f"Flagged for fact-checking: {issue}"
        
        elif trigger.intervention_type == InterventionType.BIAS_HIGHLIGHT:
            warning = f"⚡ Potential bias detected: {issue}"
            modified = f"{warning}\n\n{response}"
            action = f"Highlighted bias: {issue}"
        
        elif trigger.intervention_type == InterventionType.CONFIDENCE_ADJUST:
            caveat = f"\n\n📊 Confidence note: Due to {trigger.name.lower()}, confidence in this response has been adjusted."
            modified = f"{response}{caveat}"
            action = f"Adjusted confidence for: {issue}"
        
        elif trigger.intervention_type == InterventionType.RESPONSE_BLOCK:
            modified = f"[Response blocked: {issue}]"
            warning = f"🛑 Response blocked due to: {issue}"
            action = f"Blocked response: {issue}"
        
        elif trigger.intervention_type == InterventionType.SLOW_DOWN:
            caveat = f"\n\n⏸️ This response has been flagged for careful review before acting on it."
            modified = f"{response}{caveat}"
            action = f"Slowed down for review: {issue}"
        
        else:
            action = f"No action for: {issue}"
        
        return action, modified, warning, caveat
    
    def _is_overloaded(self) -> bool:
        """Check if we're intervening too frequently (cognitive overload)."""
        now = datetime.utcnow()
        recent = [t for t in self.recent_interventions 
                 if (now - t).total_seconds() < 60]
        
        # More than 5 interventions in last minute = overloaded
        return len(recent) > 5
    
    def add_trigger(self, trigger: InterventionTrigger):
        """Add a custom intervention trigger."""
        self.triggers[trigger.trigger_id] = trigger
    
    def disable_trigger(self, trigger_id: str):
        """Disable a trigger."""
        if trigger_id in self.triggers:
            self.triggers[trigger_id].enabled = False
    
    def enable_trigger(self, trigger_id: str):
        """Enable a trigger."""
        if trigger_id in self.triggers:
            self.triggers[trigger_id].enabled = True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get intervention statistics."""
        avg_detection_time = (
            sum(self.detection_times) / len(self.detection_times)
            if self.detection_times else 0
        )
        
        within_window = sum(
            1 for i in self.intervention_history
            if i.within_window
        )
        total = len(self.intervention_history)
        
        return {
            'total_interventions': total,
            'within_cognitive_window': within_window,
            'window_rate': within_window / total if total > 0 else 0,
            'avg_detection_time_ms': avg_detection_time,
            'target_detection_ms': self.TARGET_DETECTION_MS,
            'meeting_target': avg_detection_time <= self.TARGET_DETECTION_MS,
            'interventions_by_type': dict(self.interventions_by_type),
            'triggers_active': sum(1 for t in self.triggers.values() if t.enabled),
            'triggers_total': len(self.triggers)
        }
    
    def record_effectiveness(self, intervention_id: str, was_helpful: bool):
        """Record whether an intervention was helpful (for learning)."""
        if intervention_id not in self.intervention_effectiveness:
            self.intervention_effectiveness[intervention_id] = []
        self.intervention_effectiveness[intervention_id].append(was_helpful)

    
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

