"""
BAIS Cognitive Governance Engine v16.0
Main Engine - Orchestrates All Components

This is the central orchestrator that:
1. Processes incoming queries
2. Coordinates all detectors and analyzers
3. Makes acceptance decisions
4. Triggers learning updates
5. Provides comprehensive audit trails
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import time
import uuid
import json
import asyncio
import httpx

# Direct imports (PYTHONPATH set in Dockerfile)
from learning.threshold_optimizer import AdaptiveThresholdOptimizer, ThresholdDecision
from learning.state_machine import StateMachineWithHysteresis, OperationalState
from learning.outcome_memory import OutcomeMemory, DecisionRecord
from learning.algorithms import LearningOutcome

# Configuration (dual-mode support)
from core.config import get_config, BAISConfig

# Phase 2 detectors (ML or statistical based on config)
from detectors.grounding import GroundingDetector
from detectors.behavioral import BehavioralBiasDetector
from detectors.temporal import TemporalDetector, TemporalObservation
from detectors.factual import FactualDetector


@dataclass
class GovernanceSignals:
    """Signals from all detectors."""
    grounding_score: float = 0.0
    grounding_details: Dict = field(default_factory=dict)
    
    temporal_score: float = 0.0
    temporal_details: Dict = field(default_factory=dict)
    
    behavioral_score: float = 0.0
    behavioral_details: Dict = field(default_factory=dict)
    
    factual_score: float = 0.0
    factual_details: Dict = field(default_factory=dict)
    
    simulation_score: float = 0.0
    simulation_details: Dict = field(default_factory=dict)


@dataclass


class MustPassResult:
    """Result of must-pass predicate checks."""
    all_passed: bool = True
    evidence_sufficiency: Dict = field(default_factory=dict)
    answer_completeness: Dict = field(default_factory=dict)
    factual_grounding: Dict = field(default_factory=dict)
    safety_check: Dict = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)


@dataclass
class GovernanceDecision:
    """Complete governance decision."""
    session_id: str
    timestamp: datetime
    
    # Input
    query: str
    documents: List[Dict]
    
    # LLM Response
    response: str
    response_source: str  # 'llm', 'cached', 'error'
    
    # Signals
    signals: GovernanceSignals
    fused_score: float
    
    # Must-pass
    must_pass: MustPassResult
    
    # Threshold
    threshold_decision: ThresholdDecision
    
    # Decision
    accuracy: float
    accepted: bool
    rejection_reason: Optional[str]
    confidence: str  # HIGH, MEDIUM, LOW
    pathway: str  # VERIFIED, SKEPTICAL, ASSISTED, REJECTED
    
    # State
    operational_state: str
    state_multiplier: float
    
    # Audit
    processing_time_ms: float
    inventions_applied: List[str]
    warnings: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat(),
            'query': self.query[:500],
            'response': self.response[:1000] if self.response else None,
            'governance': {
                'accepted': self.accepted,
                'accuracy': self.accuracy,
                'confidence': self.confidence,
                'pathway': self.pathway,
                'rejection_reason': self.rejection_reason
            },
            'signals': {
                'grounding': self.signals.grounding_score,
                'temporal': self.signals.temporal_score,
                'behavioral': self.signals.behavioral_score,
                'factual': self.signals.factual_score,
                'simulation': self.signals.simulation_score,
                'fused': self.fused_score
            },
            'threshold': {
                'base': self.threshold_decision.base_threshold,
                'final': self.threshold_decision.final_threshold,
                'state_multiplier': self.threshold_decision.state_multiplier,
                'risk_multiplier': self.threshold_decision.risk_multiplier,
                'reasoning': self.threshold_decision.reasoning
            },
            'must_pass': {
                'all_passed': self.must_pass.all_passed,
                'failures': self.must_pass.failures
            },
            'state': {
                'operational': self.operational_state,
                'multiplier': self.state_multiplier
            },
            'audit': {
                'processing_time_ms': self.processing_time_ms,
                'inventions_applied': self.inventions_applied,
                'warnings': self.warnings,
                'recommendations': self.recommendations
            }
        }


class CognitiveGovernanceEngine:
    """
    Main Cognitive Governance Engine.
    
    This is the primary interface for governance operations.
    """
    
    VERSION = "16.2.0"  # Dual-mode: FULL (ML) + LITE (Statistical)
    
    # Fusion weights (will be learned)
    DEFAULT_FUSION_WEIGHTS = {
        'grounding': 0.30,
        'temporal': 0.15,
        'behavioral': 0.25,
        'factual': 0.20,
        'simulation': 0.10
    }
    
    # Domain-specific fusion weights
    DOMAIN_FUSION_WEIGHTS = {
        'medical': {'grounding': 0.35, 'temporal': 0.10, 'behavioral': 0.20, 'factual': 0.25, 'simulation': 0.10},
        'financial': {'grounding': 0.30, 'temporal': 0.20, 'behavioral': 0.20, 'factual': 0.20, 'simulation': 0.10},
        'legal': {'grounding': 0.30, 'temporal': 0.10, 'behavioral': 0.25, 'factual': 0.25, 'simulation': 0.10},
    }
    
    def __init__(self,
                 data_dir: Path = None,
                 llm_api_key: str = None,
                 llm_model: str = None,
                 algorithm: str = None,
                 config: BAISConfig = None):
        
        # Load configuration (determines FULL vs LITE mode)
        self.config = config or get_config()
        
        self.data_dir = data_dir or Path(self.config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.llm_api_key = llm_api_key or self.config.llm_api_key
        self.llm_model = llm_model or self.config.llm_model
        
        # Initialize components
        self.threshold_optimizer = AdaptiveThresholdOptimizer(
            data_dir=self.data_dir,
            algorithm=algorithm or self.config.learning_algorithm
        )
        
        self.state_machine = self.threshold_optimizer.state_machine
        self.outcome_memory = self.threshold_optimizer.outcome_memory
        
        # Learned fusion weights (loaded from persistence)
        self.fusion_weights = dict(self.DEFAULT_FUSION_WEIGHTS)
        self._load_fusion_weights()
        
        # Phase 2 Detectors - mode determined by config (FULL or LITE)
        # use_embeddings=None and use_nli=None means "auto from config"
        self.grounding_detector = GroundingDetector(
            use_embeddings=None,  # Auto from config
            learning_path=self.data_dir / "grounding_learning.json"
        )
        self.behavioral_detector = BehavioralBiasDetector(
            learning_path=self.data_dir
        )
        self.temporal_detector = TemporalDetector(
            storage_path=self.data_dir / "temporal.json"
        )
        self.factual_detector = FactualDetector(
            use_nli=None,  # Auto from config
            learning_path=self.data_dir / "factual_learning.json"
        )
        
        # Log mode
        print(f"[CognitiveGovernanceEngine] Mode: {self.config.get_mode_description()}")
        
        # Session cache
        self.recent_sessions: Dict[str, GovernanceDecision] = {}
    
    async def evaluate(self,
                       query: str,
                       documents: List[Dict],
                       response: str = None,
                       generate_response: bool = True,
                       context: Dict[str, Any] = None) -> GovernanceDecision:
        """
        Main evaluation method.
        
        Args:
            query: The user query
            documents: Source documents for grounding
            response: Optional pre-generated LLM response
            generate_response: Whether to generate response if not provided
            context: Additional context (domain, risk_level, etc.)
        
        Returns:
            Complete governance decision
        """
        start_time = time.time()
        session_id = str(uuid.uuid4())[:8]
        
        context = context or {}
        inventions_applied = []
        warnings = []
        recommendations = []
        
        # Detect domain if not provided
        domain = context.get('domain') or self._detect_domain(query, documents)
        context['domain'] = domain
        inventions_applied.append("PPA1-Inv1: Context Detection")
        
        # Get LLM response if needed
        if response is None and generate_response:
            response, source = await self._get_llm_response(query, documents)
        else:
            source = 'provided' if response else 'none'
        
        if not response:
            response = "ERROR: No response available"
            source = 'error'
        
        # Compute signals
        signals = await self._compute_signals(query, response, documents, context)
        inventions_applied.append("PPA1-Inv2: Multi-Signal Detection")
        
        # Fuse signals
        fused_score = self._fuse_signals(signals, domain)
        inventions_applied.append("PPA1-Inv3: Signal Fusion")
        
        # Check must-pass predicates
        must_pass = self._check_must_pass(query, response, documents)
        inventions_applied.append("PPA2-Inv1: Must-Pass Predicates")
        
        # Get threshold
        threshold_decision = self.threshold_optimizer.get_threshold(
            domain=domain,
            context=context
        )
        inventions_applied.append("PPA2-Inv3: Adaptive Threshold")
        
        # Compute accuracy
        accuracy = self._compute_accuracy(fused_score, signals)
        
        # Make decision
        accepted, rejection_reason, pathway = self._make_decision(
            accuracy=accuracy,
            threshold=threshold_decision.final_threshold,
            must_pass=must_pass,
            signals=signals
        )
        inventions_applied.append("PPA2-Inv2: Acceptance Control")
        
        # State machine is already integrated via threshold
        inventions_applied.append("PPA3-Inv1: State Machine with Hysteresis")
        
        # Check behavioral biases
        if signals.behavioral_score > 0.3:
            inventions_applied.append("PPA3-Inv2-5: Behavioral Bias Detection")
            warnings.append(f"Behavioral bias detected: {signals.behavioral_details}")
        
        # Determine confidence
        confidence = self._determine_confidence(accuracy, threshold_decision.confidence)
        
        # Generate warnings and recommendations
        if not accepted:
            recommendations.append("Consider regenerating response with more specific guidance")
        
        if signals.simulation_score > 0.4:
            warnings.append("Response may contain simulated/fabricated content")
        
        if confidence == "LOW":
            recommendations.append("Response has low confidence, consider manual review")
        
        # Build decision
        decision = GovernanceDecision(
            session_id=session_id,
            timestamp=datetime.utcnow(),
            query=query,
            documents=documents,
            response=response,
            response_source=source,
            signals=signals,
            fused_score=fused_score,
            must_pass=must_pass,
            threshold_decision=threshold_decision,
            accuracy=accuracy,
            accepted=accepted,
            rejection_reason=rejection_reason,
            confidence=confidence,
            pathway=pathway,
            operational_state=self.state_machine.state.value,
            state_multiplier=self.state_machine.get_multiplier(),
            processing_time_ms=(time.time() - start_time) * 1000,
            inventions_applied=inventions_applied,
            warnings=warnings,
            recommendations=recommendations
        )
        
        # Cache session
        self.recent_sessions[session_id] = decision
        if len(self.recent_sessions) > 100:
            # Remove oldest
            oldest = min(self.recent_sessions.keys())
            del self.recent_sessions[oldest]
        
        return decision
    
    def record_feedback(self,
                        session_id: str,
                        was_correct: bool,
                        feedback: str = None) -> Dict[str, Any]:
        """
        Record feedback on a decision to trigger learning.
        """
        decision = self.recent_sessions.get(session_id)
        
        if not decision:
            return {'error': 'Session not found', 'session_id': session_id}
        
        # Record outcome with threshold optimizer
        result = self.threshold_optimizer.record_outcome(
            domain=decision.threshold_decision.domain,
            accuracy=decision.accuracy,
            threshold_used=decision.threshold_decision.final_threshold,
            was_accepted=decision.accepted,
            was_correct=was_correct,
            context={'query': decision.query[:200]},
            query=decision.query,
            query_embedding=None  # Would need embedding model
        )
        
        # Also record with temporal detector for pattern learning
        temporal_obs = TemporalObservation(
            timestamp=decision.timestamp,
            accuracy=decision.accuracy,
            domain=decision.threshold_decision.domain,
            was_accepted=decision.accepted,
            was_correct=was_correct,
            features={
                'grounding': decision.signals.grounding_score,
                'behavioral': decision.signals.behavioral_score,
                'factual': decision.signals.factual_score
            }
        )
        temporal_result = self.temporal_detector.record(temporal_obs)
        
        return {
            'session_id': session_id,
            'was_correct': was_correct,
            'learning_triggered': True,
            'result': result,
            'temporal_signal': temporal_result.to_dict()
        }
    
    async def _get_llm_response(self, 
                                query: str, 
                                documents: List[Dict]) -> Tuple[str, str]:
        """Get response from LLM."""
        if not self.llm_api_key:
            return "ERROR: No API key configured", "error"
        
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                doc_context = "\n".join([d.get('content', '')[:500] for d in documents[:3]])
                
                response = await client.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.llm_api_key}"},
                    json={
                        "model": self.llm_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": f"Answer based on these documents:\n{doc_context}"
                            },
                            {"role": "user", "content": query}
                        ],
                        "max_tokens": 1000
                    }
                )
                
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"], "llm"
                else:
                    return f"LLM Error: {response.status_code}", "error"
        except Exception as e:
            return f"Error: {str(e)}", "error"
    
    async def _compute_signals(self,
                               query: str,
                               response: str,
                               documents: List[Dict],
                               context: Dict) -> GovernanceSignals:
        """Compute all governance signals using Phase 2 ML-based detectors."""
        signals = GovernanceSignals()
        
        # 1. Grounding score (Phase 2 detector)
        grounding_result = self.grounding_detector.analyze(response, documents, query)
        signals.grounding_score = grounding_result.score
        signals.grounding_details = grounding_result.to_dict()
        
        # 2. Temporal score (Phase 2 detector)
        temporal_signal = self.temporal_detector.get_signal()
        signals.temporal_score = temporal_signal.score
        signals.temporal_details = temporal_signal.to_dict()
        
        # 3. Behavioral bias score (Phase 2 detector - all 4 types)
        behavioral_result = self.behavioral_detector.detect_all(
            query=query, 
            response=response,
            history=[]  # Could be enhanced with session history
        )
        signals.behavioral_score = behavioral_result.total_bias_score
        signals.behavioral_details = behavioral_result.to_dict()
        
        # 4. Factual score (Phase 2 detector)
        factual_result = self.factual_detector.analyze(query, response, documents)
        signals.factual_score = factual_result.score
        signals.factual_details = factual_result.to_dict()
        
        # 5. Simulation detection (keep inline for simplicity)
        signals.simulation_score, signals.simulation_details = self._detect_simulation(
            response
        )
        
        return signals
    
    def _compute_grounding(self, 
                          response: str, 
                          documents: List[Dict]) -> Tuple[float, Dict]:
        """Compute grounding score."""
        doc_text = " ".join([d.get('content', '') for d in documents])
        
        if not doc_text.strip() or not response.strip():
            return 0.0, {'error': 'Missing content'}
        
        # Extract entities
        import re
        r_numbers = set(re.findall(r'\b\d+\.?\d*\b', response))
        d_numbers = set(re.findall(r'\b\d+\.?\d*\b', doc_text))
        
        r_words = set(w.lower() for w in re.findall(r'\b\w{4,}\b', response))
        d_words = set(w.lower() for w in re.findall(r'\b\w{4,}\b', doc_text))
        
        # Compute scores
        num_match = len(r_numbers & d_numbers) / max(len(r_numbers), 1) if r_numbers else 0.5
        word_match = len(r_words & d_words) / max(len(r_words), 1) if r_words else 0.0
        
        score = num_match * 0.6 + word_match * 0.4
        
        return score, {
            'number_match': num_match,
            'word_match': word_match,
            'response_numbers': len(r_numbers),
            'grounded_numbers': len(r_numbers & d_numbers)
        }
    
    def _compute_temporal(self) -> Tuple[float, Dict]:
        """Compute temporal score from recent history."""
        status = self.state_machine.get_status()
        
        recent_violations = status['recent_violations']
        total_ops = status['statistics']['total_violations'] + status['statistics']['total_successes']
        
        if total_ops == 0:
            return 0.5, {'status': 'no_history'}
        
        violation_rate = status['statistics']['violation_rate']
        
        # Lower violation rate = higher score
        score = 1.0 - min(violation_rate * 2, 1.0)
        
        return score, {
            'recent_violations': recent_violations,
            'violation_rate': violation_rate,
            'state': status['current_state']
        }
    
    def _compute_behavioral(self, query: str, response: str) -> Tuple[float, Dict]:
        """Compute behavioral bias score."""
        import re
        
        biases_detected = []
        total_score = 0.0
        
        # Confirmation bias patterns
        confirm_patterns = [
            (r'\bconfirm\b', 0.3),
            (r'\bvalidate\s+my\b', 0.4),
            (r'\bagree\s+that\b', 0.3),
            (r"\bI\s+(?:already|just)\s+know\b", 0.5),
        ]
        
        for pattern, weight in confirm_patterns:
            if re.search(pattern, query, re.I):
                biases_detected.append('confirmation')
                total_score += weight
                break
        
        # Reward seeking patterns (in response)
        reward_patterns = [
            (r'\babsolutely\b', 0.2),
            (r'\bdefinitely\b', 0.2),
            (r'\bof\s+course\b', 0.2),
        ]
        
        for pattern, weight in reward_patterns:
            if re.search(pattern, response, re.I):
                biases_detected.append('reward_seeking')
                total_score += weight
                break
        
        return min(total_score, 1.0), {
            'biases_detected': biases_detected,
            'raw_score': total_score
        }
    
    def _compute_factual(self, 
                        response: str, 
                        documents: List[Dict]) -> Tuple[float, Dict]:
        """Compute factual accuracy score."""
        doc_text = " ".join([d.get('content', '') for d in documents])
        
        # Check for contradictions
        import re
        r_nums = re.findall(r'(\d+\.?\d*)\s*(\w+)?', response)
        d_nums = re.findall(r'(\d+\.?\d*)\s*(\w+)?', doc_text)
        
        contradictions = []
        for r_num, r_unit in r_nums:
            for d_num, d_unit in d_nums:
                # Same unit but different number
                if r_unit and d_unit and r_unit.lower() == d_unit.lower():
                    try:
                        r_val = float(r_num)
                        d_val = float(d_num)
                        if r_val != 0 and d_val != 0:
                            ratio = r_val / d_val
                            if ratio < 0.5 or ratio > 2.0:
                                contradictions.append({
                                    'response': f"{r_num} {r_unit}",
                                    'document': f"{d_num} {d_unit}"
                                })
                    except:
                        pass
        
        # Score based on contradictions found
        if contradictions:
            score = max(0, 1.0 - len(contradictions) * 0.3)
        else:
            score = 0.8  # Baseline if no numbers to verify
        
        return score, {
            'contradictions_found': len(contradictions),
            'contradiction_details': contradictions[:3]
        }
    
    def _detect_simulation(self, response: str) -> Tuple[float, Dict]:
        """Detect simulated/fabricated content."""
        import re
        
        simulation_patterns = [
            (r'sample\s+output', 0.5),
            (r'expected\s+output', 0.4),
            (r'example\s+output', 0.4),
            (r'would\s+(?:output|print|return)', 0.3),
            (r'fabricated', 0.8),
            (r'simulated', 0.9),
        ]
        
        optimism_patterns = [
            (r'100%\s+accurate', 0.5),
            (r'zero\s+errors', 0.4),
            (r'works\s+perfectly', 0.4),
            (r'will\s+never\s+fail', 0.5),
        ]
        
        total_score = 0.0
        indicators = []
        
        for pattern, weight in simulation_patterns + optimism_patterns:
            if re.search(pattern, response, re.I):
                total_score += weight
                indicators.append(pattern)
        
        return min(total_score, 1.0), {
            'indicators': indicators,
            'raw_score': total_score
        }
    
    def _fuse_signals(self, signals: GovernanceSignals, domain: str) -> float:
        """Fuse signals using domain-specific weights."""
        weights = self.DOMAIN_FUSION_WEIGHTS.get(domain, self.DEFAULT_FUSION_WEIGHTS)
        
        fused = (
            weights['grounding'] * signals.grounding_score +
            weights['temporal'] * signals.temporal_score +
            weights['behavioral'] * (1.0 - signals.behavioral_score) +  # Invert: lower bias = higher score
            weights['factual'] * signals.factual_score +
            weights['simulation'] * (1.0 - signals.simulation_score)  # Invert: lower simulation = higher score
        )
        
        return fused
    
    def _check_must_pass(self,
                        query: str,
                        response: str,
                        documents: List[Dict]) -> MustPassResult:
        """Check must-pass predicates."""
        result = MustPassResult()
        doc_text = " ".join([d.get('content', '') for d in documents])
        
        # Evidence sufficiency
        import re
        q_terms = set(w.lower() for w in re.findall(r'\b\w{4,}\b', query))
        d_terms = set(w.lower() for w in re.findall(r'\b\w{4,}\b', doc_text))
        relevance = len(q_terms & d_terms) / max(len(q_terms), 1)
        
        result.evidence_sufficiency = {
            'passed': relevance >= 0.1 or not doc_text.strip(),
            'relevance': relevance
        }
        if not result.evidence_sufficiency['passed']:
            result.failures.append('evidence_sufficiency')
        
        # Answer completeness
        r_terms = set(w.lower() for w in re.findall(r'\b\w{4,}\b', response))
        coverage = len(q_terms & r_terms) / max(len(q_terms), 1)
        
        result.answer_completeness = {
            'passed': coverage >= 0.2,
            'coverage': coverage
        }
        if not result.answer_completeness['passed']:
            result.failures.append('answer_completeness')
        
        # Factual grounding (numbers)
        r_nums = set(re.findall(r'\b\d+\.?\d*\b', response))
        d_nums = set(re.findall(r'\b\d+\.?\d*\b', doc_text))
        grounded = len(r_nums & d_nums) / max(len(r_nums), 1) if r_nums else 1.0
        
        result.factual_grounding = {
            'passed': grounded >= 0.3 or not r_nums,
            'grounded_ratio': grounded
        }
        if not result.factual_grounding['passed']:
            result.failures.append('factual_grounding')
        
        # Safety check
        unsafe_patterns = [
            r'how\s+to\s+(?:make|build)\s+(?:bomb|weapon)',
            r'instructions\s+(?:for|to)\s+(?:harm|kill)',
        ]
        is_safe = not any(re.search(p, response, re.I) for p in unsafe_patterns)
        
        result.safety_check = {'passed': is_safe}
        if not is_safe:
            result.failures.append('safety_check')
        
        result.all_passed = len(result.failures) == 0
        return result
    
    def _compute_accuracy(self, fused_score: float, signals: GovernanceSignals) -> float:
        """Compute final accuracy score."""
        # Base accuracy from fused score
        base_accuracy = fused_score * 100
        
        # Apply penalties
        bias_penalty = signals.behavioral_score * 15
        simulation_penalty = signals.simulation_score * 20
        
        accuracy = base_accuracy - bias_penalty - simulation_penalty
        return max(0, min(100, accuracy))
    
    def _make_decision(self,
                      accuracy: float,
                      threshold: float,
                      must_pass: MustPassResult,
                      signals: GovernanceSignals) -> Tuple[bool, Optional[str], str]:
        """Make acceptance decision and determine pathway."""
        
        # Must-pass failure = immediate rejection
        if not must_pass.all_passed:
            reason = f"Must-pass failed: {must_pass.failures}"
            return False, reason, "REJECTED"
        
        # High simulation = skeptical pathway
        if signals.simulation_score > 0.5:
            reason = f"Simulation detected: {signals.simulation_score:.0%}"
            return False, reason, "SKEPTICAL"
        
        # High bias = assisted pathway (human review)
        if signals.behavioral_score > 0.5:
            reason = f"Behavioral bias detected: {signals.behavioral_score:.0%}"
            return False, reason, "ASSISTED"
        
        # Threshold check
        if accuracy >= threshold:
            return True, None, "VERIFIED"
        else:
            reason = f"Below threshold: {accuracy:.1f}% < {threshold:.1f}%"
            return False, reason, "SKEPTICAL"
    
    def _determine_confidence(self, accuracy: float, threshold_confidence: float) -> str:
        """Determine confidence level."""
        combined = (accuracy / 100 + threshold_confidence) / 2
        
        if combined >= 0.7:
            return "HIGH"
        elif combined >= 0.5:
            return "MEDIUM"
        return "LOW"
    
    def _detect_domain(self, query: str, documents: List[Dict]) -> str:
        """Detect domain from query and documents."""
        text = query.lower()
        
        domain_keywords = {
            'medical': ['medication', 'dose', 'symptom', 'treatment', 'diagnosis', 
                       'patient', 'doctor', 'hospital', 'health', 'disease'],
            'financial': ['invest', 'money', 'stock', 'market', 'portfolio', 
                         'return', 'risk', 'bank', 'loan', 'interest'],
            'legal': ['contract', 'law', 'court', 'legal', 'liability', 
                     'regulation', 'compliance', 'attorney', 'lawsuit'],
            'technical': ['code', 'software', 'api', 'database', 'server',
                         'algorithm', 'programming', 'bug', 'deploy']
        }
        
        scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for k in keywords if k in text)
            scores[domain] = score
        
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return 'general'
    
    def _load_fusion_weights(self):
        """Load learned fusion weights."""
        path = self.data_dir / "fusion_weights.json"
        if path.exists():
            try:
                with open(path) as f:
                    self.fusion_weights = json.load(f)
            except:
                pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status."""
        capabilities = self.config.get_capabilities_summary()
        
        return {
            'version': self.VERSION,
            'mode': capabilities['mode_description'],
            'capabilities': capabilities,
            'state_machine': self.state_machine.get_status(),
            'threshold_optimizer': self.threshold_optimizer.get_statistics(),
            'fusion_weights': self.fusion_weights,
            'recent_sessions': len(self.recent_sessions),
            'health': self.state_machine.get_health_assessment(),
            'detectors': {
                'grounding': f"GroundingDetector ({capabilities['detection_methods']['grounding']})",
                'behavioral': f"BehavioralBiasDetector ({capabilities['detection_methods']['behavioral']})",
                'temporal': f"TemporalDetector ({capabilities['detection_methods']['temporal']})",
                'factual': f"FactualDetector ({capabilities['detection_methods']['factual']})"
            },
            'temporal_signal': self.temporal_detector.get_signal().to_dict()
        }
    
    def get_learning_report(self) -> Dict[str, Any]:
        """Get learning report."""
        return self.threshold_optimizer.get_learning_report()

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

