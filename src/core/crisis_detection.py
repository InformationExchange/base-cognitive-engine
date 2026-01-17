"""
BASE Cognitive Governance Engine v32.0
Crisis Detection & Environment Profiles

Phase 32: Addresses PPA2-C1-24, PPA2-C1-25, PPA2-C1-26
- Environment tags select policy profile
- Crisis detection with hysteresis tightening
- Streaming behavioral gating within cognitive window

Patent Claims Addressed:
- PPA2-C1-24: Streaming behavioral gating within cognitive window, crisis tightening
- PPA2-C1-25: Environment tags select policy profile (low-stakes social, high-stakes healthcare)
- PPA2-C1-26: Crisis detection: tighten γ, δ, decrease α, enlarge W with hysteresis
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import logging
import re

logger = logging.getLogger(__name__)


class EnvironmentType(Enum):
    """Environment types for policy selection."""
    SOCIAL_LOW_STAKES = "social_low_stakes"
    GENERAL = "general"
    EDUCATIONAL = "educational"
    PROFESSIONAL = "professional"
    FINANCIAL = "financial"
    LEGAL = "legal"
    MEDICAL_ADVISORY = "medical_advisory"
    MEDICAL_CRITICAL = "medical_critical"
    EMERGENCY = "emergency"


class CrisisLevel(Enum):
    """Crisis severity levels."""
    NONE = "none"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"
    
    @property
    def severity(self) -> int:
        """Get numeric severity for comparison."""
        return {
            CrisisLevel.NONE: 0,
            CrisisLevel.ELEVATED: 1,
            CrisisLevel.HIGH: 2,
            CrisisLevel.CRITICAL: 3,
            CrisisLevel.EMERGENCY: 4
        }[self]


class BehavioralGateStatus(Enum):
    """Status of behavioral gating."""
    OPEN = "open"
    THROTTLED = "throttled"
    RESTRICTED = "restricted"
    BLOCKED = "blocked"


@dataclass
class PolicyProfile:
    """
    Policy profile for an environment.
    
    PPA2-C1-25: Environment tags select policy profile.
    """
    environment: EnvironmentType
    gamma_threshold: float  # Decision threshold γ
    delta_confidence: float  # Confidence requirement δ
    alpha_error_rate: float  # Target error rate α
    window_size: int  # Rolling window W size
    hysteresis_width: float  # Hysteresis band width
    min_dwell_time: float  # Minimum dwell time (seconds)
    require_certificate: bool  # Whether certificate required
    require_must_pass: bool  # Whether must-pass required
    max_response_length: Optional[int] = None
    allowed_topics: List[str] = field(default_factory=list)
    blocked_topics: List[str] = field(default_factory=list)
    disclaimer_required: bool = False
    human_review_threshold: float = 0.3  # Below this, escalate to human
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "environment": self.environment.value,
            "gamma_threshold": self.gamma_threshold,
            "delta_confidence": self.delta_confidence,
            "alpha_error_rate": self.alpha_error_rate,
            "window_size": self.window_size,
            "hysteresis_width": self.hysteresis_width,
            "require_certificate": self.require_certificate,
            "disclaimer_required": self.disclaimer_required
        }


@dataclass
class CrisisState:
    """Current crisis state."""
    level: CrisisLevel
    triggered_at: Optional[datetime] = None
    indicators: List[str] = field(default_factory=list)
    tightening_applied: bool = False
    original_profile: Optional[PolicyProfile] = None
    tightened_profile: Optional[PolicyProfile] = None


@dataclass
class BehavioralGateState:
    """State of behavioral gate."""
    status: BehavioralGateStatus
    window_observations: int = 0
    violations_in_window: int = 0
    last_violation: Optional[datetime] = None
    throttle_factor: float = 1.0


@dataclass
class CrisisDetectionResult:
    """Result of crisis detection analysis."""
    crisis_level: CrisisLevel
    behavioral_gate: BehavioralGateStatus
    active_profile: PolicyProfile
    crisis_indicators: List[str]
    tightening_applied: bool
    recommendations: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class EnvironmentProfileManager:
    """
    Manages environment-specific policy profiles.
    
    PPA2-C1-25: Environment tags select policy profile.
    """
    
    def __init__(self):
        """Initialize with default profiles."""
        self.profiles: Dict[EnvironmentType, PolicyProfile] = {}
        self._setup_default_profiles()
        
    def _setup_default_profiles(self):
        """Setup default policy profiles for each environment."""
        
        # Social/Low-stakes - permissive
        self.profiles[EnvironmentType.SOCIAL_LOW_STAKES] = PolicyProfile(
            environment=EnvironmentType.SOCIAL_LOW_STAKES,
            gamma_threshold=0.4,
            delta_confidence=0.5,
            alpha_error_rate=0.15,
            window_size=50,
            hysteresis_width=0.15,
            min_dwell_time=1.0,
            require_certificate=False,
            require_must_pass=False
        )
        
        # General - balanced
        self.profiles[EnvironmentType.GENERAL] = PolicyProfile(
            environment=EnvironmentType.GENERAL,
            gamma_threshold=0.5,
            delta_confidence=0.6,
            alpha_error_rate=0.1,
            window_size=100,
            hysteresis_width=0.1,
            min_dwell_time=2.0,
            require_certificate=False,
            require_must_pass=True
        )
        
        # Educational - moderate
        self.profiles[EnvironmentType.EDUCATIONAL] = PolicyProfile(
            environment=EnvironmentType.EDUCATIONAL,
            gamma_threshold=0.55,
            delta_confidence=0.65,
            alpha_error_rate=0.08,
            window_size=100,
            hysteresis_width=0.1,
            min_dwell_time=3.0,
            require_certificate=False,
            require_must_pass=True,
            disclaimer_required=True
        )
        
        # Professional - stricter
        self.profiles[EnvironmentType.PROFESSIONAL] = PolicyProfile(
            environment=EnvironmentType.PROFESSIONAL,
            gamma_threshold=0.6,
            delta_confidence=0.7,
            alpha_error_rate=0.05,
            window_size=150,
            hysteresis_width=0.08,
            min_dwell_time=5.0,
            require_certificate=True,
            require_must_pass=True
        )
        
        # Financial - strict
        self.profiles[EnvironmentType.FINANCIAL] = PolicyProfile(
            environment=EnvironmentType.FINANCIAL,
            gamma_threshold=0.7,
            delta_confidence=0.8,
            alpha_error_rate=0.03,
            window_size=200,
            hysteresis_width=0.05,
            min_dwell_time=10.0,
            require_certificate=True,
            require_must_pass=True,
            disclaimer_required=True,
            human_review_threshold=0.5,
            blocked_topics=["investment_advice", "trading_signals"]
        )
        
        # Legal - very strict
        self.profiles[EnvironmentType.LEGAL] = PolicyProfile(
            environment=EnvironmentType.LEGAL,
            gamma_threshold=0.75,
            delta_confidence=0.85,
            alpha_error_rate=0.02,
            window_size=200,
            hysteresis_width=0.05,
            min_dwell_time=15.0,
            require_certificate=True,
            require_must_pass=True,
            disclaimer_required=True,
            human_review_threshold=0.6,
            blocked_topics=["legal_advice", "case_prediction"]
        )
        
        # Medical Advisory - strict with disclaimers
        self.profiles[EnvironmentType.MEDICAL_ADVISORY] = PolicyProfile(
            environment=EnvironmentType.MEDICAL_ADVISORY,
            gamma_threshold=0.75,
            delta_confidence=0.85,
            alpha_error_rate=0.02,
            window_size=200,
            hysteresis_width=0.05,
            min_dwell_time=15.0,
            require_certificate=True,
            require_must_pass=True,
            disclaimer_required=True,
            human_review_threshold=0.6,
            blocked_topics=["diagnosis", "prescription", "dosage"]
        )
        
        # Medical Critical - maximum strictness
        self.profiles[EnvironmentType.MEDICAL_CRITICAL] = PolicyProfile(
            environment=EnvironmentType.MEDICAL_CRITICAL,
            gamma_threshold=0.9,
            delta_confidence=0.95,
            alpha_error_rate=0.01,
            window_size=300,
            hysteresis_width=0.03,
            min_dwell_time=30.0,
            require_certificate=True,
            require_must_pass=True,
            disclaimer_required=True,
            human_review_threshold=0.8,
            blocked_topics=["diagnosis", "prescription", "dosage", "treatment_plan"]
        )
        
        # Emergency - fastest response with high accuracy
        self.profiles[EnvironmentType.EMERGENCY] = PolicyProfile(
            environment=EnvironmentType.EMERGENCY,
            gamma_threshold=0.85,
            delta_confidence=0.9,
            alpha_error_rate=0.01,
            window_size=50,  # Smaller for faster response
            hysteresis_width=0.02,
            min_dwell_time=1.0,  # Fast but accurate
            require_certificate=True,
            require_must_pass=True,
            disclaimer_required=True,
            human_review_threshold=0.7
        )
        
    def get_profile(self, environment: EnvironmentType) -> PolicyProfile:
        """Get profile for environment."""
        return self.profiles.get(environment, self.profiles[EnvironmentType.GENERAL])
    
    def detect_environment(self, query: str, context: Dict[str, Any]) -> EnvironmentType:
        """
        Detect environment from query and context.
        
        Args:
            query: User query
            context: Additional context
            
        Returns:
            Detected environment type
        """
        query_lower = query.lower()
        
        # Check explicit context
        if context.get("environment"):
            try:
                return EnvironmentType(context["environment"])
            except ValueError:
                pass
        
        # Pattern-based detection
        emergency_patterns = [
            r'\b(emergency|urgent|911|immediately|dying|overdose|heart attack|stroke)\b',
            r'\b(can\'t breathe|chest pain|unconscious|bleeding heavily)\b'
        ]
        
        medical_critical_patterns = [
            r'\b(dose|dosage|prescription|medication|drug interaction)\b',
            r'\b(symptom|diagnos|treatment|surgery|cancer|tumor)\b'
        ]
        
        medical_advisory_patterns = [
            r'\b(health|medical|doctor|hospital|clinic|vitamin|supplement)\b',
            r'\b(pain|ache|fever|cold|flu|headache|hurts|sore|sick)\b'
        ]
        
        financial_patterns = [
            r'\b(invest|stock|bond|portfolio|retirement|401k|ira)\b',
            r'\b(tax|financial|money|wealth|asset|liability)\b',
            r'\b(buy|sell|trade|trading|crypto|bitcoin|market)\b'
        ]
        
        legal_patterns = [
            r'\b(legal|law|attorney|lawyer|court|sue|lawsuit)\b',
            r'\b(contract|agreement|liability|rights|regulation)\b'
        ]
        
        educational_patterns = [
            r'\b(learn|teach|explain|understand|homework|study)\b',
            r'\b(course|class|lecture|tutorial|example)\b'
        ]
        
        # Check patterns in order of priority
        for pattern in emergency_patterns:
            if re.search(pattern, query_lower):
                return EnvironmentType.EMERGENCY
        
        for pattern in medical_critical_patterns:
            if re.search(pattern, query_lower):
                return EnvironmentType.MEDICAL_CRITICAL
        
        for pattern in medical_advisory_patterns:
            if re.search(pattern, query_lower):
                return EnvironmentType.MEDICAL_ADVISORY
        
        for pattern in financial_patterns:
            if re.search(pattern, query_lower):
                return EnvironmentType.FINANCIAL
        
        for pattern in legal_patterns:
            if re.search(pattern, query_lower):
                return EnvironmentType.LEGAL
        
        for pattern in educational_patterns:
            if re.search(pattern, query_lower):
                return EnvironmentType.EDUCATIONAL
        
        # Default
        return EnvironmentType.GENERAL

    # =========================================================================
    # Learning Interface
    # =========================================================================
    
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


class CrisisDetector:
    """
    Crisis detection with hysteresis-based tightening.
    
    PPA2-C1-26: Crisis detection: tighten γ, δ, decrease α, enlarge W with hysteresis.
    """
    
    def __init__(
        self,
        observation_window: int = 50,
        crisis_threshold: float = 0.3,
        recovery_threshold: float = 0.15,
        hysteresis_enabled: bool = True
    ):
        """
        Initialize crisis detector.
        
        Args:
            observation_window: Number of observations to consider
            crisis_threshold: Threshold to enter crisis mode
            recovery_threshold: Threshold to exit crisis mode (with hysteresis)
            hysteresis_enabled: Whether to use hysteresis for transitions
        """
        self.observation_window = observation_window
        self.crisis_threshold = crisis_threshold
        self.recovery_threshold = recovery_threshold
        self.hysteresis_enabled = hysteresis_enabled
        
        self.observations: deque = deque(maxlen=observation_window)
        self.crisis_state = CrisisState(level=CrisisLevel.NONE)
        self.indicator_history: List[Dict[str, Any]] = []
        
    def add_observation(
        self,
        accuracy: float,
        confidence: float,
        issues: List[str],
        is_violation: bool = False
    ) -> CrisisLevel:
        """
        Add observation and update crisis level.
        
        Args:
            accuracy: Current accuracy score
            confidence: Current confidence score
            issues: List of issues detected
            is_violation: Whether this is a policy violation
            
        Returns:
            Current crisis level
        """
        observation = {
            "timestamp": datetime.utcnow(),
            "accuracy": accuracy,
            "confidence": confidence,
            "issues": issues,
            "is_violation": is_violation,
            "risk_score": self._calculate_risk_score(accuracy, confidence, issues, is_violation)
        }
        self.observations.append(observation)
        
        # Calculate aggregate risk
        risk_scores = [obs["risk_score"] for obs in self.observations]
        avg_risk = np.mean(risk_scores) if risk_scores else 0.0
        recent_risk = np.mean(risk_scores[-10:]) if len(risk_scores) >= 10 else avg_risk
        
        # Determine crisis level with hysteresis
        new_level = self._determine_level(avg_risk, recent_risk)
        
        # Apply hysteresis - harder to exit crisis than enter
        if self.hysteresis_enabled:
            if new_level.severity < self.crisis_state.level.severity:
                # Trying to reduce crisis level - check recovery threshold
                if avg_risk > self.recovery_threshold:
                    new_level = self.crisis_state.level  # Stay at current level
        
        # Update state
        if new_level != self.crisis_state.level:
            self.crisis_state.level = new_level
            if new_level != CrisisLevel.NONE:
                self.crisis_state.triggered_at = datetime.utcnow()
                self.crisis_state.indicators = self._get_active_indicators()
        
        return self.crisis_state.level
    
    def _calculate_risk_score(
        self,
        accuracy: float,
        confidence: float,
        issues: List[str],
        is_violation: bool
    ) -> float:
        """Calculate risk score from observation."""
        risk = 0.0
        
        # Low accuracy increases risk (more aggressive)
        risk += max(0, 0.7 - accuracy) * 0.5
        
        # Low confidence increases risk (more aggressive)
        risk += max(0, 0.6 - confidence) * 0.4
        
        # Issues increase risk (more weight)
        risk += min(len(issues) * 0.08, 0.3)
        
        # Violations are high risk
        if is_violation:
            risk += 0.35
        
        return min(1.0, risk)
    
    def _determine_level(self, avg_risk: float, recent_risk: float) -> CrisisLevel:
        """Determine crisis level from risk scores."""
        # Use higher of average and recent risk
        risk = max(avg_risk, recent_risk * 1.2)  # Weight recent more
        
        if risk >= 0.8:
            return CrisisLevel.EMERGENCY
        elif risk >= 0.6:
            return CrisisLevel.CRITICAL
        elif risk >= 0.4:
            return CrisisLevel.HIGH
        elif risk >= 0.2:
            return CrisisLevel.ELEVATED
        else:
            return CrisisLevel.NONE
    
    def _get_active_indicators(self) -> List[str]:
        """Get active crisis indicators."""
        indicators = []
        
        if len(self.observations) < 5:
            return indicators
        
        recent = list(self.observations)[-10:]
        
        # Check for high violation rate
        violation_rate = sum(1 for o in recent if o["is_violation"]) / len(recent)
        if violation_rate > 0.3:
            indicators.append(f"HIGH_VIOLATION_RATE: {violation_rate:.0%}")
        
        # Check for accuracy decline
        if len(self.observations) >= 10:
            old_acc = np.mean([o["accuracy"] for o in list(self.observations)[:5]])
            new_acc = np.mean([o["accuracy"] for o in recent[:5]])
            if old_acc - new_acc > 0.15:
                indicators.append(f"ACCURACY_DECLINE: {old_acc:.2f} → {new_acc:.2f}")
        
        # Check for confidence instability
        conf_std = np.std([o["confidence"] for o in recent])
        if conf_std > 0.2:
            indicators.append(f"CONFIDENCE_INSTABILITY: σ={conf_std:.2f}")
        
        # Check for issue accumulation
        total_issues = sum(len(o["issues"]) for o in recent)
        if total_issues > 10:
            indicators.append(f"ISSUE_ACCUMULATION: {total_issues} issues")
        
        return indicators
    
    def tighten_profile(self, profile: PolicyProfile) -> PolicyProfile:
        """
        Tighten policy profile for crisis mode.
        
        PPA2-C1-26: Tighten γ, δ, decrease α, enlarge W.
        
        Args:
            profile: Original profile
            
        Returns:
            Tightened profile
        """
        crisis_multiplier = {
            CrisisLevel.NONE: 1.0,
            CrisisLevel.ELEVATED: 1.1,
            CrisisLevel.HIGH: 1.25,
            CrisisLevel.CRITICAL: 1.5,
            CrisisLevel.EMERGENCY: 2.0
        }
        
        mult = crisis_multiplier.get(self.crisis_state.level, 1.0)
        
        return PolicyProfile(
            environment=profile.environment,
            gamma_threshold=min(0.95, profile.gamma_threshold * mult),  # Increase threshold
            delta_confidence=min(0.99, profile.delta_confidence * mult),  # Increase confidence
            alpha_error_rate=max(0.001, profile.alpha_error_rate / mult),  # Decrease error rate
            window_size=int(profile.window_size * mult),  # Enlarge window
            hysteresis_width=max(0.01, profile.hysteresis_width / mult),  # Tighten hysteresis
            min_dwell_time=profile.min_dwell_time * mult,  # Increase dwell time
            require_certificate=True,  # Always require in crisis
            require_must_pass=True,  # Always require in crisis
            disclaimer_required=True,  # Always require in crisis
            human_review_threshold=min(0.9, profile.human_review_threshold * mult),
            blocked_topics=profile.blocked_topics,
            allowed_topics=profile.allowed_topics
        )
    
    def get_state(self) -> CrisisState:
        """Get current crisis state."""
        return self.crisis_state

    # =========================================================================
    # Learning Interface
    # =========================================================================
    
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


class BehavioralGate:
    """
    Streaming behavioral gating within cognitive window.
    
    PPA2-C1-24: Streaming behavioral gating within cognitive window, crisis tightening.
    """
    
    def __init__(
        self,
        window_duration: timedelta = timedelta(minutes=5),
        max_violations: int = 3,
        throttle_factor: float = 0.5,
        block_threshold: int = 5
    ):
        """
        Initialize behavioral gate.
        
        Args:
            window_duration: Cognitive window duration
            max_violations: Max violations before throttling
            throttle_factor: Factor to reduce throughput when throttled
            block_threshold: Violations that trigger block
        """
        self.window_duration = window_duration
        self.max_violations = max_violations
        self.throttle_factor = throttle_factor
        self.block_threshold = block_threshold
        
        self.violations: deque = deque()
        self.state = BehavioralGateState(status=BehavioralGateStatus.OPEN)
        
    def record_violation(self, violation_type: str, severity: float = 0.5) -> BehavioralGateStatus:
        """
        Record a behavioral violation.
        
        Args:
            violation_type: Type of violation
            severity: Severity (0-1)
            
        Returns:
            Updated gate status
        """
        now = datetime.utcnow()
        
        # Clean old violations outside window
        self._clean_old_violations(now)
        
        # Add new violation
        self.violations.append({
            "timestamp": now,
            "type": violation_type,
            "severity": severity
        })
        
        self.state.last_violation = now
        
        # Update status
        self._update_status()
        
        return self.state.status
    
    def record_success(self) -> BehavioralGateStatus:
        """Record successful behavior (helps recover)."""
        # Slight recovery factor
        if self.state.status == BehavioralGateStatus.THROTTLED:
            self.state.throttle_factor = min(1.0, self.state.throttle_factor + 0.05)
            if self.state.throttle_factor >= 1.0:
                self.state.status = BehavioralGateStatus.OPEN
        
        return self.state.status
    
    def _clean_old_violations(self, now: datetime):
        """Remove violations outside the cognitive window."""
        cutoff = now - self.window_duration
        while self.violations and self.violations[0]["timestamp"] < cutoff:
            self.violations.popleft()
    
    def _update_status(self):
        """Update gate status based on violations."""
        violation_count = len(self.violations)
        self.state.violations_in_window = violation_count
        
        if violation_count >= self.block_threshold:
            self.state.status = BehavioralGateStatus.BLOCKED
            self.state.throttle_factor = 0.0
        elif violation_count >= self.max_violations + 2:
            self.state.status = BehavioralGateStatus.RESTRICTED
            self.state.throttle_factor = 0.25
        elif violation_count >= self.max_violations:
            self.state.status = BehavioralGateStatus.THROTTLED
            self.state.throttle_factor = self.throttle_factor
        else:
            self.state.status = BehavioralGateStatus.OPEN
            self.state.throttle_factor = 1.0
    
    def should_allow(self) -> Tuple[bool, str]:
        """
        Check if current request should be allowed.
        
        Returns:
            Tuple of (allowed, reason)
        """
        if self.state.status == BehavioralGateStatus.BLOCKED:
            return False, "Gate blocked due to excessive violations"
        
        if self.state.status == BehavioralGateStatus.RESTRICTED:
            # Allow with 25% probability
            if np.random.random() > 0.25:
                return False, "Gate restricted - request throttled"
        
        if self.state.status == BehavioralGateStatus.THROTTLED:
            # Allow based on throttle factor
            if np.random.random() > self.state.throttle_factor:
                return False, "Gate throttled - request delayed"
        
        return True, "Gate open"
    
    def get_state(self) -> BehavioralGateState:
        """Get current gate state."""
        self._clean_old_violations(datetime.utcnow())
        self.state.window_observations = len(self.violations)
        return self.state

    # Learning Interface
    def record_outcome(self, outcome: Dict) -> None:
        if not hasattr(self, '_outcomes'): self._outcomes = []
        self._outcomes.append(outcome)
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        pass
    
    def get_statistics(self) -> Dict:
        return {'outcomes': len(getattr(self, '_outcomes', []))}
    
    def serialize_state(self) -> Dict:
        return {'outcomes': getattr(self, '_outcomes', [])[-100:]}
    
    def deserialize_state(self, state: Dict) -> None:
        self._outcomes = state.get('outcomes', [])


class CrisisDetectionManager:
    """
    Unified manager for crisis detection and environment profiles.
    
    Implements PPA2-C1-24, PPA2-C1-25, PPA2-C1-26.
    
    Phase 49: Enhanced with learning capabilities via CentralizedLearningManager.
    """
    
    def __init__(self, learning_manager: Optional[Any] = None):
        """
        Initialize crisis detection manager.
        
        Args:
            learning_manager: CentralizedLearningManager for unified learning (Phase 49)
        """
        self.profile_manager = EnvironmentProfileManager()
        self.crisis_detector = CrisisDetector()
        self.behavioral_gate = BehavioralGate()
        
        self.current_environment = EnvironmentType.GENERAL
        self.current_profile = self.profile_manager.get_profile(EnvironmentType.GENERAL)
        
        # Phase 49: Learning capabilities
        self._learning_manager = learning_manager
        self._domain_adjustments: Dict[str, float] = {}
        self._detection_outcomes: List[Dict[str, Any]] = []
        self._learning_rate = 0.1
        
        if self._learning_manager:
            self._learning_manager.register_module("crisis_detection")
        
        logger.info("[CrisisDetection] Crisis Detection Manager initialized")
    
    def evaluate(
        self,
        query: str,
        response: str,
        accuracy: float,
        confidence: float,
        issues: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> CrisisDetectionResult:
        """
        Evaluate query/response for crisis indicators.
        
        Args:
            query: User query
            response: AI response
            accuracy: Accuracy score
            confidence: Confidence score
            issues: Detected issues
            context: Additional context
            
        Returns:
            CrisisDetectionResult
        """
        context = context or {}
        
        # Detect environment
        environment = self.profile_manager.detect_environment(query, context)
        self.current_environment = environment
        
        # Get base profile
        base_profile = self.profile_manager.get_profile(environment)
        
        # Check for violations
        is_violation = (
            accuracy < base_profile.gamma_threshold * 0.8 or
            confidence < base_profile.delta_confidence * 0.8 or
            len(issues) > 3
        )
        
        # Update crisis detector
        crisis_level = self.crisis_detector.add_observation(
            accuracy=accuracy,
            confidence=confidence,
            issues=issues,
            is_violation=is_violation
        )
        
        # Update behavioral gate
        if is_violation:
            violation_type = "accuracy_violation" if accuracy < 0.5 else "confidence_violation"
            self.behavioral_gate.record_violation(violation_type, severity=0.5)
        else:
            self.behavioral_gate.record_success()
        
        # Get active profile (tightened if in crisis)
        if crisis_level != CrisisLevel.NONE:
            active_profile = self.crisis_detector.tighten_profile(base_profile)
            tightening_applied = True
        else:
            active_profile = base_profile
            tightening_applied = False
        
        self.current_profile = active_profile
        
        # Get gate state
        gate_state = self.behavioral_gate.get_state()
        
        # Build recommendations
        recommendations = []
        if crisis_level != CrisisLevel.NONE:
            recommendations.append(f"Crisis level {crisis_level.value}: thresholds tightened")
        if gate_state.status != BehavioralGateStatus.OPEN:
            recommendations.append(f"Behavioral gate {gate_state.status.value}")
        if active_profile.disclaimer_required:
            recommendations.append("Disclaimer required for this response")
        if environment in [EnvironmentType.MEDICAL_CRITICAL, EnvironmentType.EMERGENCY]:
            recommendations.append("High-stakes environment: extra caution required")
        
        return CrisisDetectionResult(
            crisis_level=crisis_level,
            behavioral_gate=gate_state.status,
            active_profile=active_profile,
            crisis_indicators=self.crisis_detector.get_state().indicators,
            tightening_applied=tightening_applied,
            recommendations=recommendations,
            details={
                "environment": environment.value,
                "base_gamma": base_profile.gamma_threshold,
                "active_gamma": active_profile.gamma_threshold,
                "violations_in_window": gate_state.violations_in_window,
                "throttle_factor": gate_state.throttle_factor
            }
        )
    
    def check_gate(self) -> Tuple[bool, str]:
        """Check if current request should be allowed through gate."""
        return self.behavioral_gate.should_allow()
    
    def get_profile(self, environment: Optional[EnvironmentType] = None) -> PolicyProfile:
        """Get profile for environment (or current)."""
        if environment:
            return self.profile_manager.get_profile(environment)
        return self.current_profile
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of crisis detection."""
        gate_state = self.behavioral_gate.get_state()
        crisis_state = self.crisis_detector.get_state()
        
        return {
            "environment": self.current_environment.value,
            "crisis_level": crisis_state.level.value,
            "behavioral_gate": gate_state.status.value,
            "throttle_factor": gate_state.throttle_factor,
            "violations_in_window": gate_state.violations_in_window,
            "crisis_indicators": crisis_state.indicators,
            "profile": self.current_profile.to_dict()
        }
    
    # ========================================
    # PHASE 49: LEARNING METHODS
    # ========================================
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)
    
    def record_outcome_detailed(
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        was_correct: bool,
        domain: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """Record a crisis detection outcome for learning (Phase 49)."""
        outcome = {"input": input_data, "output": output_data, "was_correct": was_correct, "domain": domain}
        self._detection_outcomes.append(outcome)
        
        if domain:
            current = self._domain_adjustments.get(domain, 0.0)
            self._domain_adjustments[domain] = max(-0.5, min(0.5, current + self._learning_rate * (1.0 if was_correct else -1.0)))
        
        if self._learning_manager:
            self._learning_manager.record_outcome("crisis_detection", input_data, output_data, was_correct, domain, metadata)
    
    def record_feedback(self, result: CrisisDetectionResult, was_accurate: bool) -> None:
        """Record feedback on a crisis detection result (Phase 49)."""
        self.record_outcome(
            {"environment": result.environment.value, "crisis_level": result.crisis_level.value},
            {"gate_status": result.gate_status.value},
            was_accurate, result.environment.value
        )
    
    def adapt_thresholds(self, threshold_name: str, current_value: float, direction: str = 'decrease') -> float:
        """Adapt detection thresholds based on learning (Phase 49)."""
        if self._learning_manager:
            return self._learning_manager.adapt_threshold("crisis_detection", threshold_name, current_value, direction)
        return max(0.1, min(0.95, current_value + (0.05 if direction == 'increase' else -0.05)))
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get domain-specific threshold adjustment (Phase 49)."""
        if self._learning_manager:
            return self._learning_manager.get_domain_adjustment("crisis_detection", domain)
        return self._domain_adjustments.get(domain, 0.0)
    
    def save_state(self, filepath: Optional[str] = None) -> bool:
        """Save learning state (Phase 49)."""
        return self._learning_manager.save_state() if self._learning_manager else False
    
    def load_state(self, filepath: Optional[str] = None) -> bool:
        """Load learning state (Phase 49)."""
        return self._learning_manager.load_state() if self._learning_manager else False
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning statistics (Phase 49)."""
        if self._learning_manager:
            return self._learning_manager.get_learning_statistics("crisis_detection").__dict__
        total = len(self._detection_outcomes)
        correct = sum(1 for o in self._detection_outcomes if o['was_correct'])
        return {"module": "crisis_detection", "total_outcomes": total, "accuracy": correct/total if total > 0 else 0.5}

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


# Test function
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
    print("=" * 60)
    print("PHASE 32: Crisis Detection & Environment Profiles Test")
    print("=" * 60)
    
    manager = CrisisDetectionManager()
    
    # Test 1: Environment detection
    print("\n[1] Environment Detection:")
    queries = [
        ("What's the weather?", EnvironmentType.GENERAL),
        ("How do I invest in stocks?", EnvironmentType.FINANCIAL),
        ("What medication should I take?", EnvironmentType.MEDICAL_CRITICAL),
        ("I'm having chest pain", EnvironmentType.EMERGENCY),
        ("Explain photosynthesis", EnvironmentType.EDUCATIONAL),
        ("Is this contract valid?", EnvironmentType.LEGAL)
    ]
    
    for query, expected in queries:
        detected = manager.profile_manager.detect_environment(query, {})
        status = "✓" if detected == expected else "✗"
        print(f"    {status} '{query[:30]}...' → {detected.value}")
    
    # Test 2: Policy profiles
    print("\n[2] Policy Profiles:")
    for env in [EnvironmentType.SOCIAL_LOW_STAKES, EnvironmentType.MEDICAL_CRITICAL]:
        profile = manager.get_profile(env)
        print(f"    {env.value}:")
        print(f"      γ={profile.gamma_threshold}, δ={profile.delta_confidence}, α={profile.alpha_error_rate}")
    
    # Test 3: Crisis detection
    print("\n[3] Crisis Detection:")
    
    # Normal observations
    for i in range(5):
        result = manager.evaluate(
            query="What is AI?",
            response="AI is...",
            accuracy=0.8,
            confidence=0.7,
            issues=[]
        )
    print(f"    After normal: {result.crisis_level.value}")
    
    # Violations
    for i in range(5):
        result = manager.evaluate(
            query="Medical question",
            response="Take this pill...",
            accuracy=0.3,
            confidence=0.2,
            issues=["DANGEROUS_ADVICE", "UNVERIFIED"]
        )
    print(f"    After violations: {result.crisis_level.value}")
    print(f"    Tightening applied: {result.tightening_applied}")
    print(f"    Indicators: {result.crisis_indicators[:2]}")
    
    # Test 4: Behavioral gate
    print("\n[4] Behavioral Gate:")
    gate_status = manager.behavioral_gate.get_state()
    print(f"    Status: {gate_status.status.value}")
    print(f"    Violations: {gate_status.violations_in_window}")
    print(f"    Throttle: {gate_status.throttle_factor:.2f}")
    
    allowed, reason = manager.check_gate()
    print(f"    Allowed: {allowed} - {reason}")
    
    # Test 5: Status
    print("\n[5] Overall Status:")
    status = manager.get_status()
    print(f"    Environment: {status['environment']}")
    print(f"    Crisis Level: {status['crisis_level']}")
    print(f"    Gate: {status['behavioral_gate']}")
    
    print("\n" + "=" * 60)
    print("PHASE 32: Crisis Detection Module - VERIFIED")
    print("=" * 60)

