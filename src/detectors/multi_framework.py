"""
BASE Cognitive Governance Engine v16.5
Multi-Framework Convergence Engine

PPA-1 Invention 19: FULL IMPLEMENTATION
Fuse multiple psychological/cognitive frameworks for comprehensive bias analysis.

This module implements:
1. Multiple Framework Integration: Incorporate diverse bias theories
2. Framework Weighting: Domain-specific framework importance
3. Convergent Analysis: Find consensus across frameworks
4. Divergence Detection: Identify framework disagreements
5. Meta-Framework Learning: Improve framework weights from outcomes

═══════════════════════════════════════════════════════════════════════════════
7 PSYCHOLOGICAL FRAMEWORKS INTEGRATED
═══════════════════════════════════════════════════════════════════════════════

1. DUAL PROCESS THEORY (Kahneman, 2011):
   - System 1: Fast, intuitive, emotional (biases like availability, affect)
   - System 2: Slow, analytical, logical (deliberate reasoning)
   - Detection: Ratio of System1 vs System2 language indicators

2. CIALDINI'S PERSUASION PRINCIPLES (Cialdini, 1984):
   - Reciprocity, Commitment, Social Proof, Authority, Liking, Scarcity
   - Detection: Multiple principles present = potential manipulation
   - 6 sub-detectors with pattern matching

3. COGNITIVE DISSONANCE (Festinger, 1957):
   - Rationalization patterns when beliefs conflict with behavior
   - Selective exposure to confirming information
   - Detection: "but...still", "even though...anyway" patterns

4. ATTRIBUTION THEORY (Heider, 1958; Kelley, 1967):
   - Fundamental Attribution Error: Over-attribute to person vs situation
   - Self-Serving Bias: Credit success to self, blame failure on others
   - Detection: "they are lazy", "I succeeded because skill"

5. PROSPECT THEORY (Kahneman & Tversky, 1979):
   - Loss Aversion: Losses loom larger than gains
   - Framing Effects: Same info presented as gain vs loss
   - Detection: Loss-focused language, statistical framing

6. SOCIAL IDENTITY THEORY (Tajfel & Turner, 1979):
   - In-group favoritism: "We are better/right"
   - Out-group derogation: "They are wrong/bad"
   - Detection: We/us vs they/them language patterns

7. ANCHORING & ADJUSTMENT (Tversky & Kahneman, 1974):
   - Initial anchor biases subsequent judgments
   - Detection: "originally $X, now $Y" patterns

═══════════════════════════════════════════════════════════════════════════════
CONVERGENCE ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

Convergent Bias: Detected by 2+ frameworks → HIGH confidence
Divergent Bias: Detected by only 1 framework → LOW confidence

Scoring Formula:
weighted_score = Σ (severity_f × weight_f × learned_adjustment_f)
                 for each framework f

Domain-Specific Weights:
- Medical: Authority (0.25), Attribution (0.20), Anchoring (0.20)
- Financial: Prospect (0.30), Anchoring (0.25), Dual Process (0.15)
- Political: Social Identity (0.30), Dissonance (0.20), Persuasion (0.15)
═══════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
from enum import Enum
import re
import math


class BiasFramework(str, Enum):
    """Psychological frameworks for bias analysis."""
    DUAL_PROCESS = "dual_process"           # System 1/2 thinking
    PERSUASION = "persuasion"               # Cialdini's principles
    COGNITIVE_DISSONANCE = "dissonance"     # Belief-behavior conflict
    ATTRIBUTION = "attribution"             # Cause attribution biases
    PROSPECT = "prospect"                   # Loss aversion, framing
    SOCIAL_IDENTITY = "social_identity"     # In-group/out-group
    ANCHORING = "anchoring"                 # First information dominance


@dataclass
class FrameworkAnalysis:
    """Analysis from a single framework."""
    framework: BiasFramework
    bias_detected: bool
    confidence: float
    bias_types: List[str]
    evidence: List[str]
    severity: float  # 0-1
    recommendation: str
    
    def to_dict(self) -> Dict:
        return {
            'framework': self.framework.value,
            'bias_detected': self.bias_detected,
            'confidence': self.confidence,
            'bias_types': self.bias_types,
            'evidence': self.evidence[:5],
            'severity': self.severity,
            'recommendation': self.recommendation
        }


@dataclass
class ConvergenceResult:
    """Result of multi-framework convergence analysis."""
    frameworks_agree: int
    frameworks_disagree: int
    convergent_biases: List[str]  # Biases detected by multiple frameworks
    divergent_biases: List[str]   # Biases detected by only one framework
    overall_bias_score: float
    overall_confidence: float
    framework_results: Dict[BiasFramework, FrameworkAnalysis]
    recommendations: List[str]
    meta_analysis: str
    
    def to_dict(self) -> Dict:
        return {
            'frameworks_agree': self.frameworks_agree,
            'frameworks_disagree': self.frameworks_disagree,
            'convergent_biases': self.convergent_biases,
            'divergent_biases': self.divergent_biases,
            'overall_bias_score': self.overall_bias_score,
            'overall_confidence': self.overall_confidence,
            'framework_results': {k.value: v.to_dict() for k, v in self.framework_results.items()},
            'recommendations': self.recommendations,
            'meta_analysis': self.meta_analysis
        }


class MultiFrameworkConvergenceEngine:
    """
    Multi-Framework Convergence Engine.
    
    PPA-1 Invention 19: Full Implementation
    
    Analyzes bias through multiple psychological frameworks
    and finds convergent/divergent patterns.
    """
    
    # Framework weights by domain
    DOMAIN_WEIGHTS = {
        'medical': {
            BiasFramework.DUAL_PROCESS: 0.25,
            BiasFramework.ATTRIBUTION: 0.20,
            BiasFramework.ANCHORING: 0.20,
            BiasFramework.COGNITIVE_DISSONANCE: 0.15,
            BiasFramework.PROSPECT: 0.10,
            BiasFramework.SOCIAL_IDENTITY: 0.05,
            BiasFramework.PERSUASION: 0.05,
        },
        'financial': {
            BiasFramework.PROSPECT: 0.30,  # Loss aversion critical
            BiasFramework.ANCHORING: 0.25,
            BiasFramework.DUAL_PROCESS: 0.15,
            BiasFramework.PERSUASION: 0.10,
            BiasFramework.SOCIAL_IDENTITY: 0.10,
            BiasFramework.ATTRIBUTION: 0.05,
            BiasFramework.COGNITIVE_DISSONANCE: 0.05,
        },
        'political': {
            BiasFramework.SOCIAL_IDENTITY: 0.30,  # Group identity critical
            BiasFramework.COGNITIVE_DISSONANCE: 0.20,
            BiasFramework.PERSUASION: 0.15,
            BiasFramework.ATTRIBUTION: 0.15,
            BiasFramework.DUAL_PROCESS: 0.10,
            BiasFramework.ANCHORING: 0.05,
            BiasFramework.PROSPECT: 0.05,
        },
        'general': {
            BiasFramework.DUAL_PROCESS: 0.20,
            BiasFramework.PERSUASION: 0.15,
            BiasFramework.COGNITIVE_DISSONANCE: 0.15,
            BiasFramework.ATTRIBUTION: 0.15,
            BiasFramework.PROSPECT: 0.10,
            BiasFramework.SOCIAL_IDENTITY: 0.15,
            BiasFramework.ANCHORING: 0.10,
        }
    }
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def __init__(self):
        # Framework-specific pattern libraries
        self._initialize_patterns()
        
        # Learning: track which frameworks were correct
        self.framework_accuracy: Dict[BiasFramework, List[bool]] = {
            f: [] for f in BiasFramework
        }
        
        # Learned weight adjustments
        self.learned_adjustments: Dict[BiasFramework, float] = {
            f: 1.0 for f in BiasFramework
        }
    
    def _initialize_patterns(self):
        """Initialize framework-specific detection patterns."""
        
        # Dual Process Theory patterns
        self.dual_process_patterns = {
            'system1': [  # Fast, intuitive, emotional
                re.compile(r'\b(obviously|clearly|everyone knows)\b', re.I),
                re.compile(r'\b(feel|gut|instinct|intuition)\b', re.I),
                re.compile(r'\b(quick|fast|immediate|instant)\b', re.I),
            ],
            'system2': [  # Slow, analytical, logical
                re.compile(r'\b(analyze|consider|evaluate|assess)\b', re.I),
                re.compile(r'\b(evidence|data|research|study)\b', re.I),
                re.compile(r'\b(however|although|on the other hand)\b', re.I),
            ]
        }
        
        # Cialdini's Persuasion patterns
        self.persuasion_patterns = {
            'reciprocity': [
                re.compile(r'\b(give|offer|free|bonus|gift)\b', re.I),
            ],
            'commitment': [
                re.compile(r'\b(agree|commit|promise|pledge)\b', re.I),
            ],
            'social_proof': [
                re.compile(r'\b(everyone|most people|millions|popular)\b', re.I),
            ],
            'authority': [
                re.compile(r'\b(expert|doctor|professor|study shows)\b', re.I),
            ],
            'liking': [
                re.compile(r'\b(friend|like us|similar|share)\b', re.I),
            ],
            'scarcity': [
                re.compile(r'\b(limited|rare|exclusive|only|last chance)\b', re.I),
            ]
        }
        
        # Cognitive Dissonance patterns
        self.dissonance_patterns = {
            'rationalization': [
                re.compile(r'\b(but|however|even though).{0,30}(still|anyway)\b', re.I),
                re.compile(r'\b(justify|excuse|explain away)\b', re.I),
            ],
            'selective_exposure': [
                re.compile(r'\b(ignore|dismiss|overlook).{0,20}(evidence|facts)\b', re.I),
            ]
        }
        
        # Attribution patterns
        self.attribution_patterns = {
            'fundamental_error': [  # Overattribute to person, underattribute to situation
                re.compile(r'\b(they are|he is|she is).{0,20}(lazy|stupid|bad)\b', re.I),
                re.compile(r'\b(because (they|he|she)).{0,30}(character|personality)\b', re.I),
            ],
            'self_serving': [
                re.compile(r'\b(I succeeded because|my success).{0,30}(skill|ability|hard work)\b', re.I),
                re.compile(r'\b(failed because|failure).{0,30}(luck|circumstance|unfair)\b', re.I),
            ]
        }
        
        # Prospect Theory patterns
        self.prospect_patterns = {
            'loss_aversion': [
                re.compile(r'\b(lose|loss|risk|danger|threat)\b', re.I),
                re.compile(r'\b(protect|save|keep|maintain)\b', re.I),
            ],
            'framing': [
                re.compile(r'\b(\d+%\s+(success|survival|chance))\b', re.I),
                re.compile(r'\b(\d+%\s+(failure|death|risk))\b', re.I),
            ]
        }
        
        # Social Identity patterns
        self.social_identity_patterns = {
            'ingroup_favor': [
                re.compile(r'\b(we|us|our).{0,20}(better|right|correct)\b', re.I),
            ],
            'outgroup_derogation': [
                re.compile(r'\b(they|them|those people).{0,20}(wrong|bad|problem)\b', re.I),
            ]
        }
        
        # Anchoring patterns
        self.anchoring_patterns = {
            'initial_anchor': [
                re.compile(r'\b(originally|initially|first|started at)\s+[\$€£]?\d+', re.I),
                re.compile(r'\b(was|were)\s+[\$€£]?\d+.{0,30}now\s+[\$€£]?\d+', re.I),
            ]
        }
    
    def analyze(self, 
               query: str,
               response: str,
               domain: str = "general") -> ConvergenceResult:
        """
        Analyze text through all frameworks and find convergence.
        """
        framework_results = {}
        
        # Run each framework analysis
        framework_results[BiasFramework.DUAL_PROCESS] = self._analyze_dual_process(query, response)
        framework_results[BiasFramework.PERSUASION] = self._analyze_persuasion(response)
        framework_results[BiasFramework.COGNITIVE_DISSONANCE] = self._analyze_dissonance(query, response)
        framework_results[BiasFramework.ATTRIBUTION] = self._analyze_attribution(response)
        framework_results[BiasFramework.PROSPECT] = self._analyze_prospect(response)
        framework_results[BiasFramework.SOCIAL_IDENTITY] = self._analyze_social_identity(response)
        framework_results[BiasFramework.ANCHORING] = self._analyze_anchoring(response)
        
        # Find convergence
        convergence = self._find_convergence(framework_results, domain)
        
        return convergence
    
    def _analyze_dual_process(self, query: str, response: str) -> FrameworkAnalysis:
        """Analyze using Dual Process Theory (Kahneman)."""
        text = f"{query} {response}"
        
        system1_matches = sum(
            len(p.findall(text)) for p in self.dual_process_patterns['system1']
        )
        system2_matches = sum(
            len(p.findall(text)) for p in self.dual_process_patterns['system2']
        )
        
        bias_detected = system1_matches > system2_matches * 2
        bias_types = []
        evidence = []
        
        if bias_detected:
            bias_types.append("System 1 dominance")
            evidence.append(f"System 1 indicators: {system1_matches}, System 2: {system2_matches}")
        
        # Check for fast/intuitive language in response
        intuitive = any(p.search(response) for p in self.dual_process_patterns['system1'])
        if intuitive:
            evidence.append("Response uses intuitive/emotional language")
        
        severity = min(1.0, system1_matches / max(system2_matches + 1, 1) * 0.3)
        confidence = 0.6 if system1_matches + system2_matches > 3 else 0.3
        
        return FrameworkAnalysis(
            framework=BiasFramework.DUAL_PROCESS,
            bias_detected=bias_detected,
            confidence=confidence,
            bias_types=bias_types,
            evidence=evidence,
            severity=severity,
            recommendation="Add analytical reasoning and evidence to balance intuitive claims" if bias_detected else ""
        )
    
    def _analyze_persuasion(self, response: str) -> FrameworkAnalysis:
        """Analyze using Cialdini's Principles of Persuasion."""
        principles_found = []
        evidence = []
        
        for principle, patterns in self.persuasion_patterns.items():
            matches = sum(len(p.findall(response)) for p in patterns)
            if matches > 0:
                principles_found.append(principle)
                evidence.append(f"{principle}: {matches} instances")
        
        # Multiple persuasion principles = potential manipulation
        bias_detected = len(principles_found) >= 3
        severity = min(1.0, len(principles_found) * 0.2)
        
        return FrameworkAnalysis(
            framework=BiasFramework.PERSUASION,
            bias_detected=bias_detected,
            confidence=0.7 if principles_found else 0.3,
            bias_types=[f"persuasion_{p}" for p in principles_found],
            evidence=evidence,
            severity=severity,
            recommendation="Response may be using persuasive techniques rather than objective information" if bias_detected else ""
        )
    
    def _analyze_dissonance(self, query: str, response: str) -> FrameworkAnalysis:
        """Analyze for Cognitive Dissonance patterns."""
        evidence = []
        bias_types = []
        
        for pattern_type, patterns in self.dissonance_patterns.items():
            for p in patterns:
                if p.search(response):
                    bias_types.append(pattern_type)
                    evidence.append(f"Detected {pattern_type} pattern")
        
        bias_detected = len(bias_types) > 0
        severity = min(1.0, len(bias_types) * 0.3)
        
        return FrameworkAnalysis(
            framework=BiasFramework.COGNITIVE_DISSONANCE,
            bias_detected=bias_detected,
            confidence=0.5 if bias_types else 0.2,
            bias_types=bias_types,
            evidence=evidence,
            severity=severity,
            recommendation="Response may rationalize contradictory information" if bias_detected else ""
        )
    
    def _analyze_attribution(self, response: str) -> FrameworkAnalysis:
        """Analyze for Attribution Errors."""
        evidence = []
        bias_types = []
        
        for pattern_type, patterns in self.attribution_patterns.items():
            for p in patterns:
                if p.search(response):
                    bias_types.append(pattern_type)
                    evidence.append(f"Detected {pattern_type}")
        
        bias_detected = len(bias_types) > 0
        severity = min(1.0, len(bias_types) * 0.35)
        
        return FrameworkAnalysis(
            framework=BiasFramework.ATTRIBUTION,
            bias_detected=bias_detected,
            confidence=0.6 if bias_types else 0.2,
            bias_types=bias_types,
            evidence=evidence,
            severity=severity,
            recommendation="Consider situational factors alongside personal attributions" if bias_detected else ""
        )
    
    def _analyze_prospect(self, response: str) -> FrameworkAnalysis:
        """Analyze for Prospect Theory biases (loss aversion, framing)."""
        evidence = []
        bias_types = []
        
        loss_matches = sum(
            len(p.findall(response)) for p in self.prospect_patterns['loss_aversion']
        )
        framing_matches = sum(
            len(p.findall(response)) for p in self.prospect_patterns['framing']
        )
        
        if loss_matches > 3:
            bias_types.append("loss_aversion")
            evidence.append(f"Loss-focused language: {loss_matches} instances")
        
        if framing_matches > 0:
            bias_types.append("framing_effect")
            evidence.append("Statistical framing detected")
        
        bias_detected = len(bias_types) > 0
        severity = min(1.0, (loss_matches + framing_matches * 2) * 0.1)
        
        return FrameworkAnalysis(
            framework=BiasFramework.PROSPECT,
            bias_detected=bias_detected,
            confidence=0.6 if bias_types else 0.3,
            bias_types=bias_types,
            evidence=evidence,
            severity=severity,
            recommendation="Consider both gains and losses with neutral framing" if bias_detected else ""
        )
    
    def _analyze_social_identity(self, response: str) -> FrameworkAnalysis:
        """Analyze for Social Identity Theory biases."""
        evidence = []
        bias_types = []
        
        for pattern_type, patterns in self.social_identity_patterns.items():
            for p in patterns:
                if p.search(response):
                    bias_types.append(pattern_type)
                    evidence.append(f"Detected {pattern_type}")
        
        bias_detected = len(bias_types) > 0
        severity = min(1.0, len(bias_types) * 0.4)
        
        return FrameworkAnalysis(
            framework=BiasFramework.SOCIAL_IDENTITY,
            bias_detected=bias_detected,
            confidence=0.6 if bias_types else 0.2,
            bias_types=bias_types,
            evidence=evidence,
            severity=severity,
            recommendation="Avoid in-group/out-group language; treat all groups fairly" if bias_detected else ""
        )
    
    def _analyze_anchoring(self, response: str) -> FrameworkAnalysis:
        """Analyze for Anchoring bias."""
        evidence = []
        bias_types = []
        
        for patterns in self.anchoring_patterns.values():
            for p in patterns:
                matches = p.findall(response)
                if matches:
                    bias_types.append("anchoring")
                    evidence.append(f"Anchor reference found: {matches[0]}")
        
        bias_detected = len(bias_types) > 0
        
        return FrameworkAnalysis(
            framework=BiasFramework.ANCHORING,
            bias_detected=bias_detected,
            confidence=0.5 if bias_types else 0.2,
            bias_types=bias_types,
            evidence=evidence,
            severity=0.4 if bias_detected else 0.0,
            recommendation="Be aware of initial anchor values that may bias subsequent judgments" if bias_detected else ""
        )
    
    def _find_convergence(self,
                         results: Dict[BiasFramework, FrameworkAnalysis],
                         domain: str) -> ConvergenceResult:
        """Find convergent and divergent patterns across frameworks."""
        
        # Count agreements
        frameworks_detecting_bias = sum(1 for r in results.values() if r.bias_detected)
        frameworks_not_detecting = len(results) - frameworks_detecting_bias
        
        # Collect all bias types
        all_bias_types = {}
        for framework, analysis in results.items():
            for bias_type in analysis.bias_types:
                if bias_type not in all_bias_types:
                    all_bias_types[bias_type] = []
                all_bias_types[bias_type].append(framework)
        
        # Convergent = detected by 2+ frameworks
        convergent = [b for b, frameworks in all_bias_types.items() if len(frameworks) >= 2]
        divergent = [b for b, frameworks in all_bias_types.items() if len(frameworks) == 1]
        
        # Calculate weighted overall score
        weights = self.DOMAIN_WEIGHTS.get(domain, self.DOMAIN_WEIGHTS['general'])
        weighted_score = sum(
            results[f].severity * weights[f] * self.learned_adjustments[f]
            for f in BiasFramework
        )
        
        # Calculate confidence
        confidence_sum = sum(
            results[f].confidence * weights[f]
            for f in BiasFramework
        )
        
        # Collect recommendations
        recommendations = [
            r.recommendation for r in results.values() 
            if r.recommendation and r.bias_detected
        ]
        
        # Meta-analysis
        if frameworks_detecting_bias >= 5:
            meta = "STRONG CONVERGENCE: Multiple frameworks detect bias. High confidence in bias presence."
        elif frameworks_detecting_bias >= 3:
            meta = "MODERATE CONVERGENCE: Several frameworks detect bias. Recommend caution."
        elif frameworks_detecting_bias >= 1:
            meta = "WEAK CONVERGENCE: Few frameworks detect bias. May be false positive."
        else:
            meta = "NO CONVERGENCE: No frameworks detected significant bias."
        
        return ConvergenceResult(
            frameworks_agree=frameworks_detecting_bias,
            frameworks_disagree=frameworks_not_detecting,
            convergent_biases=convergent,
            divergent_biases=divergent,
            overall_bias_score=weighted_score,
            overall_confidence=confidence_sum,
            framework_results=results,
            recommendations=recommendations[:5],
            meta_analysis=meta
        )
    
    def record_outcome(self, 
                      framework: BiasFramework,
                      was_correct: bool):
        """Record whether a framework's detection was correct."""
        self.framework_accuracy[framework].append(was_correct)
        
        # Update learned adjustment
        recent = self.framework_accuracy[framework][-50:]
        if len(recent) >= 10:
            accuracy = sum(1 for x in recent if x) / len(recent)
            # Increase weight for accurate frameworks
            self.learned_adjustments[framework] = 0.5 + accuracy
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get framework statistics."""
        return {
            'framework_accuracy': {
                f.value: (
                    sum(1 for x in self.framework_accuracy[f] if x) / len(self.framework_accuracy[f])
                    if self.framework_accuracy[f] else 0.0
                )
                for f in BiasFramework
            },
            'learned_adjustments': {f.value: v for f, v in self.learned_adjustments.items()},
            'total_analyses': sum(len(v) for v in self.framework_accuracy.values())
        }

    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback."""
        pass  # Implement specific learning logic


    def serialize_state(self) -> Dict:
        """Serialize state."""
        return {'outcomes': getattr(self, '_outcomes', [])[-100:]}


    def deserialize_state(self, state: Dict) -> None:
        """Restore state."""
        self._outcomes = state.get('outcomes', [])

