"""
BAIS Self-Awareness Loop - Cognitive Self-Correction System

This module creates the "cognitive capability in the AI brain to realize it's
BSing, distorting, or off-track and correct to produce smarter outcomes."

Key Capabilities:
1. DETECT: Realize when output is off-track (fabricated, incomplete, distorted)
2. CORRECT: Generate genuinely improved output  
3. REMEMBER: Store successful decisions and thresholds
4. LEARN: Adapt from failures by circumstance

Patent Alignment:
- PPA1-Inv22: Feedback Loop with Continuous Learning
- PPA1-Inv24: Dynamic Bias Evolution (Neuroplasticity)
- PPA2-Inv27: OCO Threshold Adapter
- NOVEL-20: Response Improver

This is the "brain that knows it's wrong and fixes itself."
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import hashlib
import json
from pathlib import Path
import re


class OffTrackType(str, Enum):
    """Types of off-track behavior detected in self."""
    FABRICATION = "fabrication"  # Made up numbers, claims
    DISTORTION = "distortion"    # Misrepresented facts
    INCOMPLETE = "incomplete"    # Didn't finish task
    OVERCONFIDENT = "overconfident"  # Claimed certainty without evidence
    SYCOPHANTIC = "sycophantic"  # Told user what they want to hear
    SCOPE_DRIFT = "scope_drift"  # Went off topic
    HEDGING_NOT_FIXING = "hedging_not_fixing"  # Changed words not substance


class SeverityLevel(str, Enum):
    """Severity of off-track behavior."""
    LOW = "low"           # Minor issue, can proceed
    MEDIUM = "medium"     # Needs correction before proceeding
    HIGH = "high"         # Major issue, must fix
    CRITICAL = "critical" # Dangerous, must not proceed


@dataclass
class OffTrackDetection:
    """Single detection of off-track behavior."""
    detection_id: str
    off_track_type: OffTrackType
    severity: SeverityLevel
    evidence: str  # What text triggered this
    location: str  # Where in the response
    confidence: float
    circumstance: Dict[str, Any]  # Domain, task type, etc.


@dataclass
class CorrectionAttempt:
    """Record of a correction attempt."""
    attempt_id: str
    original_text: str
    corrected_text: str
    correction_type: str  # What type of correction
    success: bool
    improvement_score: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SuccessfulDecision:
    """Memory of a successful decision."""
    decision_id: str
    circumstance: Dict[str, Any]  # domain, task_type, risk_level
    threshold_used: float
    detection_applied: List[str]
    outcome_quality: float
    timestamp: datetime = field(default_factory=datetime.now)
    times_reused: int = 0


@dataclass
class FailureLesson:
    """Lesson learned from a failure."""
    lesson_id: str
    circumstance: Dict[str, Any]
    what_failed: str
    why_failed: str
    what_works_instead: str
    confidence: float
    times_validated: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SelfAwarenessResult:
    """Complete result of self-awareness check."""
    is_off_track: bool
    detections: List[OffTrackDetection]
    corrections_needed: List[str]
    suggested_action: str
    confidence: float
    
    # If correction was attempted
    corrected_response: Optional[str] = None
    correction_success: bool = False
    improvement_score: float = 0.0
    
    # Learnings applied
    memories_applied: List[str] = field(default_factory=list)
    lessons_applied: List[str] = field(default_factory=list)


class SelfAwarenessLoop:
    """
    The cognitive self-correction system.
    
    This is NOT a filter - it's a brain that:
    1. Knows when it's wrong
    2. Fixes itself
    3. Remembers what worked
    4. Learns from failures
    """
    
    # Patterns that indicate fabrication
    FABRICATION_PATTERNS = [
        (r'\b\d{2,3}[-–]\d{2,3}%\b', "Range percentage without baseline", 0.9),
        (r'\b(?:achieves?|reaches?|attains?)\s+\d{2,3}[-–]?\d*%', "Achievement claim without evidence", 0.85),
        (r'\b(?:studies show|research indicates|experts agree)\b', "Vague authority claim", 0.7),
        (r'\b\d+(?:\.\d+)?x\s+(?:faster|better|improvement)', "Multiplier claim without baseline", 0.8),
        (r'\baccuracy\s+improvement\b', "Unverified accuracy claim", 0.75),
        # Phase 7: Inflated improvement claims
        (r'\+\d+%', "Inflated improvement percentage", 0.85),
        (r'\bimprovement\s+score\s*[:\s]*[+]?\d+', "Unverified improvement score", 0.8),
        (r'\b(?:is|now)\s+(?:safe|accurate|correct)\b', "Unverified safety/accuracy claim", 0.7),
    ]
    
    # Patterns that indicate overconfidence  
    OVERCONFIDENCE_PATTERNS = [
        (r'\b100\s*%\b', "Absolute certainty", 0.95),
        (r'\b(?:definitely|certainly|absolutely|guaranteed)\b', "Definite language", 0.85),
        (r'\b(?:always|never|every|all)\b(?!.*not)', "Universal quantifier", 0.75),
        (r'\b(?:fully|completely|entirely)\s+(?:working|implemented|done)', "Complete claims", 0.9),
    ]
    
    # Patterns that indicate sycophancy
    SYCOPHANCY_PATTERNS = [
        (r'(?:great|excellent|perfect)\s+(?:question|idea|suggestion)', "Excessive praise", 0.7),
        (r'you(?:\'re| are)\s+(?:right|correct|absolutely)', "Unconditional agreement", 0.75),
        (r'🎉|SUCCESS!|WORKING!', "Premature celebration", 0.85),
    ]
    
    # Patterns that indicate hedging without fixing
    HEDGING_PATTERNS = [
        (r'\b(?:may|might|could|possibly)\s+(?:achieve|work|improve)', "Hedged achievement", 0.5),
    ]
    
    def __init__(self, 
                 storage_path: Path = None,
                 enable_learning: bool = True):
        """
        Initialize the self-awareness loop.
        
        Args:
            storage_path: Where to store memories and lessons
            enable_learning: Whether to learn from outcomes
        """
        self.storage_path = storage_path or Path("learning_data/self_awareness")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.enable_learning = enable_learning
        
        # Memory stores
        self.successful_decisions: Dict[str, SuccessfulDecision] = {}
        self.failure_lessons: Dict[str, FailureLesson] = {}
        self.circumstance_index: Dict[str, List[str]] = {}  # circumstance_hash -> decision_ids
        
        # Load existing memories
        self._load_memories()
        
        # Statistics
        self.total_checks = 0
        self.off_track_detected = 0
        self.corrections_attempted = 0
        self.corrections_successful = 0
        
        # Learning state for interface compliance
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
        """
        Record the outcome of a self-awareness check for learning.
        
        Args:
            outcome: Dict with 'response', 'detections', 'corrections_applied', 'success'
        """
        self._outcomes.append(outcome)
        self.total_checks += 1
        if outcome.get('off_track', False):
            self.off_track_detected += 1
        if outcome.get('correction_success', False):
            self.corrections_successful += 1
    
    def record_feedback(self, feedback: Dict) -> None:
        """
        Record human feedback on self-awareness detection.
        
        Args:
            feedback: Dict with 'response', 'feedback_type', 'correction', 'domain'
        """
        self._feedback.append(feedback)
        domain = feedback.get('domain', 'general')
        if feedback.get('feedback_type') == 'false_positive':
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) - 0.05
        elif feedback.get('feedback_type') == 'false_negative':
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) + 0.05
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """
        Adapt detection thresholds based on performance.
        """
        if performance_data:
            if performance_data.get('false_positive_rate', 0) > 0.2:
                self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) - 0.1
            if performance_data.get('false_negative_rate', 0) > 0.2:
                self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) + 0.1
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get threshold adjustment for a domain."""
        return self._domain_adjustments.get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get statistics about the learning process."""
        detection_rate = self.off_track_detected / max(self.total_checks, 1)
        correction_rate = self.corrections_successful / max(self.corrections_attempted, 1)
        return {
            'total_checks': self.total_checks,
            'off_track_detected': self.off_track_detected,
            'detection_rate': detection_rate,
            'corrections_attempted': self.corrections_attempted,
            'corrections_successful': self.corrections_successful,
            'correction_success_rate': correction_rate,
            'successful_decisions_stored': len(self.successful_decisions),
            'failure_lessons_stored': len(self.failure_lessons),
            'domain_adjustments': dict(self._domain_adjustments),
            'outcomes_recorded': len(self._outcomes),
            'feedback_recorded': len(self._feedback)
        }
    
    def check_self(self,
                   response: str,
                   query: str,
                   circumstance: Dict[str, Any] = None) -> SelfAwarenessResult:
        """
        Check if the response is off-track and needs correction.
        
        This is the main entry point - "Is my output BS?"
        
        Args:
            response: The response to check
            query: The original query
            circumstance: Context (domain, task_type, risk_level)
            
        Returns:
            SelfAwarenessResult with detections and corrections
        """
        self.total_checks += 1
        circumstance = circumstance or {}
        
        detections = []
        
        # Check for fabrication
        detections.extend(self._check_fabrication(response, circumstance))
        
        # Check for overconfidence
        detections.extend(self._check_overconfidence(response, circumstance))
        
        # Check for sycophancy
        detections.extend(self._check_sycophancy(response, circumstance))
        
        # Check for incomplete work
        detections.extend(self._check_incomplete(response, query, circumstance))
        
        # Check for hedging without fixing (substance issue)
        detections.extend(self._check_hedging_not_fixing(response, circumstance))
        
        # Determine if off-track
        # Off-track if ANY significant issues OR multiple medium issues
        high_or_critical = any(d.severity.value in ['high', 'critical'] for d in detections)
        multiple_medium = len([d for d in detections if d.severity.value == 'medium']) >= 2
        is_off_track = len(detections) > 0 and (high_or_critical or multiple_medium)
        
        if is_off_track:
            self.off_track_detected += 1
        
        # Get relevant memories for this circumstance
        memories_applied, lessons_applied = self._get_relevant_learnings(circumstance)
        
        # Generate corrections needed
        corrections_needed = self._generate_corrections(detections)
        
        # Determine suggested action
        suggested_action = self._determine_action(detections, circumstance)
        
        # Calculate confidence
        confidence = self._calculate_confidence(detections)
        
        return SelfAwarenessResult(
            is_off_track=is_off_track,
            detections=detections,
            corrections_needed=corrections_needed,
            suggested_action=suggested_action,
            confidence=confidence,
            memories_applied=memories_applied,
            lessons_applied=lessons_applied
        )
    
    def correct_self(self,
                     response: str,
                     query: str,
                     detections: List[OffTrackDetection],
                     circumstance: Dict[str, Any] = None) -> Tuple[str, bool, float]:
        """
        Attempt to correct off-track response.
        
        This is NOT just word replacement - it's substantive correction.
        
        Args:
            response: The off-track response
            query: The original query
            detections: What was detected as off-track
            circumstance: Context
            
        Returns:
            (corrected_response, success, improvement_score)
        """
        self.corrections_attempted += 1
        circumstance = circumstance or {}
        
        corrected = response
        corrections_made = []
        
        for detection in detections:
            if detection.off_track_type == OffTrackType.FABRICATION:
                corrected, made = self._fix_fabrication(corrected, detection)
                if made:
                    corrections_made.append(f"Removed fabrication: {detection.evidence[:50]}")
                    
            elif detection.off_track_type == OffTrackType.OVERCONFIDENT:
                corrected, made = self._fix_overconfidence(corrected, detection)
                if made:
                    corrections_made.append(f"Added uncertainty: {detection.evidence[:50]}")
                    
            elif detection.off_track_type == OffTrackType.SYCOPHANTIC:
                corrected, made = self._fix_sycophancy(corrected, detection)
                if made:
                    corrections_made.append(f"Removed sycophancy: {detection.evidence[:50]}")
                    
            elif detection.off_track_type == OffTrackType.HEDGING_NOT_FIXING:
                corrected, made = self._fix_hedging(corrected, detection)
                if made:
                    corrections_made.append(f"Substantive fix: {detection.evidence[:50]}")
        
        # Calculate improvement score
        improvement_score = self._score_improvement(response, corrected, detections)
        
        success = improvement_score > 0.1 and corrected != response
        
        if success:
            self.corrections_successful += 1
            
            # Learn from this success
            if self.enable_learning:
                self._record_success(circumstance, corrected, improvement_score)
        
        return corrected, success, improvement_score
    
    def remember_success(self,
                         circumstance: Dict[str, Any],
                         threshold: float,
                         detections: List[str],
                         outcome_quality: float):
        """
        Remember a successful decision for future reference.
        
        Args:
            circumstance: Context of the decision
            threshold: Threshold that worked
            detections: Detections that were applied
            outcome_quality: Quality score of outcome
        """
        if not self.enable_learning:
            return
        
        decision_id = self._generate_id(circumstance)
        
        # Check if we already have this decision
        if decision_id in self.successful_decisions:
            # Update reuse count
            self.successful_decisions[decision_id].times_reused += 1
            return
        
        decision = SuccessfulDecision(
            decision_id=decision_id,
            circumstance=circumstance,
            threshold_used=threshold,
            detection_applied=detections,
            outcome_quality=outcome_quality
        )
        
        self.successful_decisions[decision_id] = decision
        
        # Index by circumstance
        circ_hash = self._hash_circumstance(circumstance)
        if circ_hash not in self.circumstance_index:
            self.circumstance_index[circ_hash] = []
        self.circumstance_index[circ_hash].append(decision_id)
        
        self._save_memories()
    
    def learn_from_failure(self,
                           circumstance: Dict[str, Any],
                           what_failed: str,
                           why_failed: str,
                           what_works: str):
        """
        Learn from a failure to avoid repeating it.
        
        Args:
            circumstance: Context where failure occurred
            what_failed: What specifically failed
            why_failed: Why it failed
            what_works: What should be done instead
        """
        if not self.enable_learning:
            return
        
        lesson_id = self._generate_id(circumstance, what_failed)
        
        # Check if we already have this lesson
        if lesson_id in self.failure_lessons:
            self.failure_lessons[lesson_id].times_validated += 1
            return
        
        lesson = FailureLesson(
            lesson_id=lesson_id,
            circumstance=circumstance,
            what_failed=what_failed,
            why_failed=why_failed,
            what_works_instead=what_works,
            confidence=0.5  # Start at 50%, increases with validation
        )
        
        self.failure_lessons[lesson_id] = lesson
        self._save_memories()
    
    def get_recommendation(self, circumstance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get recommendation for this circumstance based on memories.
        
        Args:
            circumstance: Current context
            
        Returns:
            Dict with threshold, detections to apply, and lessons to heed
        """
        circ_hash = self._hash_circumstance(circumstance)
        
        recommendation = {
            "threshold": 50.0,  # Default
            "detections_to_apply": [],
            "lessons_to_heed": [],
            "confidence": 0.0
        }
        
        # Check for matching successful decisions
        if circ_hash in self.circumstance_index:
            decision_ids = self.circumstance_index[circ_hash]
            if decision_ids:
                decisions = [self.successful_decisions[d] for d in decision_ids if d in self.successful_decisions]
                if decisions:
                    # Use the most successful threshold
                    best = max(decisions, key=lambda d: d.outcome_quality)
                    recommendation["threshold"] = best.threshold_used
                    recommendation["detections_to_apply"] = best.detection_applied
                    recommendation["confidence"] = best.outcome_quality
        
        # Check for matching lessons
        matching_lessons = []
        for lesson in self.failure_lessons.values():
            if self._circumstances_match(circumstance, lesson.circumstance):
                matching_lessons.append(lesson)
        
        if matching_lessons:
            recommendation["lessons_to_heed"] = [
                f"AVOID: {l.what_failed} because {l.why_failed}. INSTEAD: {l.what_works_instead}"
                for l in matching_lessons
            ]
        
        return recommendation
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics on self-awareness performance."""
        return {
            "total_checks": self.total_checks,
            "off_track_detected": self.off_track_detected,
            "detection_rate": self.off_track_detected / max(1, self.total_checks),
            "corrections_attempted": self.corrections_attempted,
            "corrections_successful": self.corrections_successful,
            "correction_success_rate": self.corrections_successful / max(1, self.corrections_attempted),
            "successful_decisions_stored": len(self.successful_decisions),
            "failure_lessons_stored": len(self.failure_lessons)
        }
    
    # ==================== PRIVATE METHODS ====================
    
    def _check_fabrication(self, response: str, circumstance: Dict) -> List[OffTrackDetection]:
        """Check for fabricated claims."""
        detections = []
        
        for pattern, description, confidence in self.FABRICATION_PATTERNS:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                detections.append(OffTrackDetection(
                    detection_id=self._generate_id(match.group(), "fabrication"),
                    off_track_type=OffTrackType.FABRICATION,
                    severity=SeverityLevel.HIGH,
                    evidence=match.group(),
                    location=f"chars {match.start()}-{match.end()}",
                    confidence=confidence,
                    circumstance=circumstance
                ))
        
        return detections
    
    def _check_overconfidence(self, response: str, circumstance: Dict) -> List[OffTrackDetection]:
        """Check for overconfident claims."""
        detections = []
        
        for pattern, description, confidence in self.OVERCONFIDENCE_PATTERNS:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                detections.append(OffTrackDetection(
                    detection_id=self._generate_id(match.group(), "overconfidence"),
                    off_track_type=OffTrackType.OVERCONFIDENT,
                    severity=SeverityLevel.MEDIUM,
                    evidence=match.group(),
                    location=f"chars {match.start()}-{match.end()}",
                    confidence=confidence,
                    circumstance=circumstance
                ))
        
        return detections
    
    def _check_sycophancy(self, response: str, circumstance: Dict) -> List[OffTrackDetection]:
        """Check for sycophantic patterns."""
        detections = []
        
        for pattern, description, confidence in self.SYCOPHANCY_PATTERNS:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                detections.append(OffTrackDetection(
                    detection_id=self._generate_id(match.group(), "sycophancy"),
                    off_track_type=OffTrackType.SYCOPHANTIC,
                    severity=SeverityLevel.MEDIUM,
                    evidence=match.group(),
                    location=f"chars {match.start()}-{match.end()}",
                    confidence=confidence,
                    circumstance=circumstance
                ))
        
        return detections
    
    def _check_incomplete(self, response: str, query: str, circumstance: Dict) -> List[OffTrackDetection]:
        """Check for incomplete work."""
        detections = []
        
        # Check for TODO/placeholder indicators
        incomplete_patterns = [
            (r'\bTODO\b', "TODO marker"),
            (r'\bFIXME\b', "FIXME marker"),
            (r'\bPLACEHOLDER\b', "Placeholder marker"),
            (r'\b(?:will|should|need to)\s+(?:implement|add|create)\b', "Future work"),
            (r'\.{3,}', "Ellipsis indicating incomplete"),
        ]
        
        for pattern, description in incomplete_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                detections.append(OffTrackDetection(
                    detection_id=self._generate_id(match.group(), "incomplete"),
                    off_track_type=OffTrackType.INCOMPLETE,
                    severity=SeverityLevel.HIGH,
                    evidence=match.group(),
                    location=f"chars {match.start()}-{match.end()}",
                    confidence=0.85,
                    circumstance=circumstance
                ))
        
        return detections
    
    def _check_hedging_not_fixing(self, response: str, circumstance: Dict) -> List[OffTrackDetection]:
        """Check if response just hedges without fixing substance."""
        detections = []
        
        # Look for hedged claims that still don't have evidence
        hedged_claim_pattern = r'\b(?:may|might|could|possibly)\s+(?:achieve|work|improve|provide)\s+\d{2,3}%?'
        
        matches = re.finditer(hedged_claim_pattern, response, re.IGNORECASE)
        for match in matches:
            # This is hedging but the claim is still unverified
            detections.append(OffTrackDetection(
                detection_id=self._generate_id(match.group(), "hedging"),
                off_track_type=OffTrackType.HEDGING_NOT_FIXING,
                severity=SeverityLevel.MEDIUM,
                evidence=match.group(),
                location=f"chars {match.start()}-{match.end()}",
                confidence=0.75,
                circumstance=circumstance
            ))
        
        return detections
    
    def _fix_fabrication(self, response: str, detection: OffTrackDetection) -> Tuple[str, bool]:
        """Fix a fabricated claim by removing it or marking as unverified."""
        original = detection.evidence
        
        # Instead of just hedging, mark as UNVERIFIED
        if re.search(r'\d{2,3}[-–]\d{2,3}%', original):
            # Range percentage - remove entirely and note
            replacement = "[CLAIM REMOVED: No benchmark data available]"
            corrected = response.replace(original, replacement)
            return corrected, corrected != response
        
        if re.search(r'achieves?\s+\d{2,3}%', original, re.IGNORECASE):
            # Achievement claim - note as unverified
            replacement = "[UNVERIFIED: No test data supports this claim]"
            corrected = response.replace(original, replacement)
            return corrected, corrected != response
        
        # Phase 7: Handle inflated improvement percentages (+45%, etc.)
        if re.search(r'\+\d+%', original):
            replacement = "[UNVERIFIED IMPROVEMENT CLAIM]"
            corrected = response.replace(original, replacement)
            return corrected, corrected != response
        
        # Phase 7: Handle "improvement score" claims
        if re.search(r'improvement\s+score', original, re.IGNORECASE):
            # Find the full line and replace
            corrected = re.sub(
                r'(?i)improvement\s+score\s*[:\s]*[+]?\d+%?', 
                "[IMPROVEMENT SCORE: UNVERIFIED - needs baseline comparison]", 
                response
            )
            return corrected, corrected != response
        
        # Phase 7: Handle "is now safe/accurate" claims
        if re.search(r'(?:is|now)\s+(?:safe|accurate|correct)', original, re.IGNORECASE):
            corrected = re.sub(
                r'(?i)(?:is|now)\s+(?:safe|accurate|correct)', 
                "may be improved", 
                response
            )
            return corrected, corrected != response
        
        return response, False
    
    def _fix_overconfidence(self, response: str, detection: OffTrackDetection) -> Tuple[str, bool]:
        """Fix overconfident language."""
        original = detection.evidence
        
        # Apply ALL relevant replacements to the full response, not just exact match
        replacements = {
            r'\b100\s*%\b': "[UNVERIFIED - requires test data]",
            r'\bdefinitely\b': "based on available evidence,",
            r'\bcertainly\b': "it appears",
            r'\babsolutely\b': "likely",
            r'\bguaranteed\b': "[UNVERIFIED]",
            r'\bfully\s+working\b': "[PARTIALLY IMPLEMENTED - see tests]",
            r'\bfully\s+complete\b': "[REQUIRES VERIFICATION]",
            r'\bcompletely\s+done\b': "[REQUIRES VERIFICATION]",
            r'\balways\b': "typically",
            r'\bnever\b': "rarely",
            r'\bevery\b': "many",
            r'\ball\b': "most",
            r'\bperfect\s+results\b': "variable results",
        }
        
        corrected = response
        for pattern, replacement in replacements.items():
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
        
        return corrected, corrected != response
    
    def _fix_sycophancy(self, response: str, detection: OffTrackDetection) -> Tuple[str, bool]:
        """Fix sycophantic patterns."""
        original = detection.evidence
        
        # Remove emojis and celebration
        corrected = response
        if '🎉' in corrected:
            corrected = corrected.replace('🎉', '')
        if 'SUCCESS!' in corrected:
            corrected = corrected.replace('SUCCESS!', 'Status:')
        if 'WORKING!' in corrected:
            corrected = corrected.replace('WORKING!', '[requires verification]')
        
        # Remove excessive praise
        praise_patterns = [
            (r'(?:great|excellent|perfect)\s+(?:question|idea)', 'Your query'),
            (r"you(?:'re| are)\s+(?:right|correct|absolutely)", 'Regarding your point,'),
        ]
        
        for pattern, replacement in praise_patterns:
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
        
        return corrected, corrected != response
    
    def _fix_hedging(self, response: str, detection: OffTrackDetection) -> Tuple[str, bool]:
        """Fix hedging without substance - requires actual evidence or removal."""
        original = detection.evidence
        
        # Hedged claims without evidence should be removed entirely
        replacement = "[CLAIM REMOVED: Cannot verify without test data]"
        corrected = response.replace(original, replacement)
        
        return corrected, corrected != response
    
    def _score_improvement(self, original: str, corrected: str, detections: List[OffTrackDetection]) -> float:
        """Score how much the correction improved the response."""
        if original == corrected:
            return 0.0
        
        # Count how many issues were addressed
        issues_fixed = 0
        for detection in detections:
            if detection.evidence not in corrected:
                issues_fixed += 1
        
        # Score based on issues fixed and their severity
        total_severity = sum(
            1.0 if d.severity == SeverityLevel.CRITICAL else
            0.75 if d.severity == SeverityLevel.HIGH else
            0.5 if d.severity == SeverityLevel.MEDIUM else 0.25
            for d in detections
        )
        
        if total_severity == 0:
            return 0.0
        
        return issues_fixed / len(detections) if detections else 0.0
    
    def _generate_corrections(self, detections: List[OffTrackDetection]) -> List[str]:
        """Generate list of corrections needed."""
        corrections = []
        
        for detection in detections:
            if detection.off_track_type == OffTrackType.FABRICATION:
                corrections.append(f"REMOVE fabricated claim '{detection.evidence[:50]}...' - no evidence")
            elif detection.off_track_type == OffTrackType.OVERCONFIDENT:
                corrections.append(f"ADD UNCERTAINTY to '{detection.evidence[:50]}...'")
            elif detection.off_track_type == OffTrackType.SYCOPHANTIC:
                corrections.append(f"REMOVE sycophancy '{detection.evidence[:50]}...'")
            elif detection.off_track_type == OffTrackType.INCOMPLETE:
                corrections.append(f"COMPLETE work indicated by '{detection.evidence[:50]}...'")
            elif detection.off_track_type == OffTrackType.HEDGING_NOT_FIXING:
                corrections.append(f"PROVIDE EVIDENCE or REMOVE claim '{detection.evidence[:50]}...'")
        
        return corrections
    
    def _determine_action(self, detections: List[OffTrackDetection], circumstance: Dict) -> str:
        """Determine what action to take based on detections."""
        if not detections:
            return "PROCEED - No issues detected"
        
        critical = [d for d in detections if d.severity == SeverityLevel.CRITICAL]
        high = [d for d in detections if d.severity == SeverityLevel.HIGH]
        
        if critical:
            return f"HALT - {len(critical)} critical issues must be resolved"
        elif high:
            return f"CORRECT - {len(high)} high-severity issues need fixing"
        else:
            return f"WARN - {len(detections)} issues found, proceed with caution"
    
    def _calculate_confidence(self, detections: List[OffTrackDetection]) -> float:
        """Calculate confidence in the self-awareness assessment."""
        if not detections:
            return 0.5  # Neutral when nothing found
        
        # Average confidence weighted by severity
        total = 0
        weight = 0
        
        for d in detections:
            severity_weight = {
                SeverityLevel.CRITICAL: 1.5,
                SeverityLevel.HIGH: 1.2,
                SeverityLevel.MEDIUM: 1.0,
                SeverityLevel.LOW: 0.8
            }.get(d.severity, 1.0)
            
            total += d.confidence * severity_weight
            weight += severity_weight
        
        return total / weight if weight > 0 else 0.5
    
    def _get_relevant_learnings(self, circumstance: Dict) -> Tuple[List[str], List[str]]:
        """Get relevant memories and lessons for this circumstance."""
        memories = []
        lessons = []
        
        circ_hash = self._hash_circumstance(circumstance)
        
        # Get matching successful decisions
        if circ_hash in self.circumstance_index:
            for dec_id in self.circumstance_index[circ_hash]:
                if dec_id in self.successful_decisions:
                    dec = self.successful_decisions[dec_id]
                    memories.append(f"Use threshold {dec.threshold_used} (quality: {dec.outcome_quality:.2f})")
        
        # Get matching lessons
        for lesson in self.failure_lessons.values():
            if self._circumstances_match(circumstance, lesson.circumstance):
                lessons.append(f"AVOID: {lesson.what_failed}")
        
        return memories, lessons
    
    def _record_success(self, circumstance: Dict, response: str, score: float):
        """Record a successful correction for learning."""
        decision_id = self._generate_id(circumstance, response[:100])
        
        self.successful_decisions[decision_id] = SuccessfulDecision(
            decision_id=decision_id,
            circumstance=circumstance,
            threshold_used=50.0,  # Current threshold
            detection_applied=["self_awareness"],
            outcome_quality=score
        )
        
        circ_hash = self._hash_circumstance(circumstance)
        if circ_hash not in self.circumstance_index:
            self.circumstance_index[circ_hash] = []
        self.circumstance_index[circ_hash].append(decision_id)
        
        self._save_memories()
    
    def _generate_id(self, *args) -> str:
        """Generate a unique ID from arguments."""
        content = json.dumps(args, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _hash_circumstance(self, circumstance: Dict) -> str:
        """Hash circumstance for indexing."""
        # Only use key fields for matching
        key_fields = {
            'domain': circumstance.get('domain', 'general'),
            'task_type': circumstance.get('task_type', 'general'),
            'risk_level': circumstance.get('risk_level', 'medium')
        }
        return self._generate_id(key_fields)
    
    def _circumstances_match(self, c1: Dict, c2: Dict) -> bool:
        """Check if two circumstances match enough to apply learnings."""
        # Match on domain at minimum
        if c1.get('domain') != c2.get('domain'):
            return False
        
        # Match on risk level if specified
        if c1.get('risk_level') and c2.get('risk_level'):
            if c1['risk_level'] != c2['risk_level']:
                return False
        
        return True
    
    def _load_memories(self):
        """Load memories from storage."""
        decisions_file = self.storage_path / "successful_decisions.json"
        lessons_file = self.storage_path / "failure_lessons.json"
        index_file = self.storage_path / "circumstance_index.json"
        
        try:
            if decisions_file.exists():
                with open(decisions_file, 'r') as f:
                    data = json.load(f)
                    for dec_id, dec_data in data.items():
                        self.successful_decisions[dec_id] = SuccessfulDecision(
                            decision_id=dec_data['decision_id'],
                            circumstance=dec_data['circumstance'],
                            threshold_used=dec_data['threshold_used'],
                            detection_applied=dec_data['detection_applied'],
                            outcome_quality=dec_data['outcome_quality'],
                            times_reused=dec_data.get('times_reused', 0)
                        )
        except Exception as e:
            print(f"[SelfAwareness] Could not load decisions: {e}")
        
        try:
            if lessons_file.exists():
                with open(lessons_file, 'r') as f:
                    data = json.load(f)
                    for les_id, les_data in data.items():
                        self.failure_lessons[les_id] = FailureLesson(
                            lesson_id=les_data['lesson_id'],
                            circumstance=les_data['circumstance'],
                            what_failed=les_data['what_failed'],
                            why_failed=les_data['why_failed'],
                            what_works_instead=les_data['what_works_instead'],
                            confidence=les_data.get('confidence', 0.5),
                            times_validated=les_data.get('times_validated', 0)
                        )
        except Exception as e:
            print(f"[SelfAwareness] Could not load lessons: {e}")
        
        try:
            if index_file.exists():
                with open(index_file, 'r') as f:
                    self.circumstance_index = json.load(f)
        except Exception as e:
            print(f"[SelfAwareness] Could not load index: {e}")
    
    def _save_memories(self):
        """Save memories to storage."""
        try:
            decisions_file = self.storage_path / "successful_decisions.json"
            with open(decisions_file, 'w') as f:
                data = {}
                for dec_id, dec in self.successful_decisions.items():
                    data[dec_id] = {
                        'decision_id': dec.decision_id,
                        'circumstance': dec.circumstance,
                        'threshold_used': dec.threshold_used,
                        'detection_applied': dec.detection_applied,
                        'outcome_quality': dec.outcome_quality,
                        'times_reused': dec.times_reused
                    }
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[SelfAwareness] Could not save decisions: {e}")
        
        try:
            lessons_file = self.storage_path / "failure_lessons.json"
            with open(lessons_file, 'w') as f:
                data = {}
                for les_id, les in self.failure_lessons.items():
                    data[les_id] = {
                        'lesson_id': les.lesson_id,
                        'circumstance': les.circumstance,
                        'what_failed': les.what_failed,
                        'why_failed': les.why_failed,
                        'what_works_instead': les.what_works_instead,
                        'confidence': les.confidence,
                        'times_validated': les.times_validated
                    }
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[SelfAwareness] Could not save lessons: {e}")
        
        try:
            index_file = self.storage_path / "circumstance_index.json"
            with open(index_file, 'w') as f:
                json.dump(self.circumstance_index, f, indent=2)
        except Exception as e:
            print(f"[SelfAwareness] Could not save index: {e}")

    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback."""
        pass  # Implement specific learning logic


    def serialize_state(self) -> Dict:
        """Serialize state."""
        return {'outcomes': getattr(self, '_outcomes', [])[-100:]}


    def deserialize_state(self, state: Dict) -> None:
        """Restore state."""
        self._outcomes = state.get('outcomes', [])


# Seed initial lessons from documented failures
def seed_initial_lessons(loop: SelfAwarenessLoop):
    """Seed the self-awareness loop with lessons from documented failures."""
    
    lessons = [
        {
            "circumstance": {"domain": "technical", "task_type": "implementation"},
            "what_failed": "Claiming 85-95% accuracy without benchmark",
            "why_failed": "No test data existed to support the claim",
            "what_works": "Only claim accuracy with actual test results showing baseline vs improved"
        },
        {
            "circumstance": {"domain": "general", "task_type": "status_report"},
            "what_failed": "Claiming FULLY WORKING after testing 1 of 20 capabilities",
            "why_failed": "Premature celebration before complete verification",
            "what_works": "Verify each capability individually before claiming complete"
        },
        {
            "circumstance": {"domain": "technical", "task_type": "architecture"},
            "what_failed": "Describing proposed architecture as if implemented",
            "why_failed": "Used present tense for future work",
            "what_works": "Use 'should be' for proposals, 'is' only for verified implementations"
        },
        {
            "circumstance": {"domain": "general", "task_type": "correction"},
            "what_failed": "Changing 'every' to 'many' and calling it fixed",
            "why_failed": "Hedging words doesn't fix unverified claims",
            "what_works": "Either provide evidence or remove the claim entirely"
        },
        {
            "circumstance": {"domain": "technical", "task_type": "documentation"},
            "what_failed": "Treating documentation as proof of implementation",
            "why_failed": "Documentation != code. Documented != implemented",
            "what_works": "Require file paths and test results as proof"
        }
    ]
    
    for lesson in lessons:
        loop.learn_from_failure(
            circumstance=lesson["circumstance"],
            what_failed=lesson["what_failed"],
            why_failed=lesson["why_failed"],
            what_works=lesson["what_works"]
        )
    
    return len(lessons)

