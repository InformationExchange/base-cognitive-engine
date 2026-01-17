"""
BAIS Cognitive Governance Engine v16.0
Adaptive Threshold Optimizer

Orchestrates threshold learning using pluggable algorithms.
Provides context-aware, situation-aware threshold determination.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import numpy as np

from .algorithms import (
    LearningAlgorithm, LearningOutcome, LearningState,
    OCOLearner, BayesianLearner, ThompsonSamplingLearner, UCBLearner, EXP3Learner,
    create_algorithm, AlgorithmRegistry
)
from .state_machine import StateMachineWithHysteresis, OperationalState
from .outcome_memory import OutcomeMemory, DecisionRecord


@dataclass
class ThresholdDecision:
    """Result of threshold determination."""
    base_threshold: float
    state_multiplier: float
    risk_multiplier: float
    context_adjustment: float
    final_threshold: float
    domain: str
    confidence: float
    algorithm_used: str
    reasoning: List[str]


class AdaptiveThresholdOptimizer:
    """
    Orchestrates adaptive threshold learning and determination.
    
    Key Features:
    1. Pluggable learning algorithms (OCO, Bayesian, Thompson, UCB, EXP3)
    2. Context-aware threshold adjustment
    3. State machine integration
    4. Risk-based multipliers
    5. Full persistence and recovery
    6. Algorithm performance monitoring and switching
    """
    
    # Domain-specific base thresholds (starting points)
    DEFAULT_BASE_THRESHOLDS = {
        'medical': 70.0,
        'financial': 65.0,
        'legal': 65.0,
        'technical': 55.0,
        'general': 50.0
    }
    
    # Risk multipliers
    RISK_MULTIPLIERS = {
        'low': 1.0,
        'medium': 1.1,
        'high': 1.25,
        'critical': 1.5
    }
    
    def __init__(self,
                 data_dir: Path = None,
                 algorithm: str = 'oco',
                 state_machine: StateMachineWithHysteresis = None,
                 outcome_memory: OutcomeMemory = None):
        
        # Use temp directory if none provided (fixes read-only filesystem issues)
        if data_dir is None:
            import tempfile
            data_dir = Path(tempfile.mkdtemp(prefix="bais_threshold_"))
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize learning algorithm
        self.algorithm_name = algorithm
        self.algorithm = create_algorithm(algorithm)
        
        # Algorithm registry for monitoring and switching
        self.registry = AlgorithmRegistry(self.data_dir / "algorithm_registry.json")
        self.registry.set_algorithm(self.algorithm)
        
        # State machine integration
        self.state_machine = state_machine or StateMachineWithHysteresis(
            storage_path=self.data_dir / "state.json"
        )
        
        # Outcome memory integration
        self.outcome_memory = outcome_memory or OutcomeMemory(
            db_path=self.data_dir / "outcomes.db"
        )
        
        # Context-specific adjustments (learned)
        self.context_adjustments: Dict[str, float] = {}
        
        # Phase 5: Domain-specific base thresholds (updated by feedback)
        self.DEFAULT_BASE_THRESHOLD = 60.0
        self.domain_base_thresholds: Dict[str, float] = {
            'medical': 75.0,
            'financial': 70.0,
            'legal': 70.0,
            'safety': 85.0,
            'general': 60.0
        }
        
        # Performance tracking
        self.recent_accuracy: List[float] = []
        self.algorithm_performance: Dict[str, List[float]] = {}
        
        # Learning interface state
        self._outcomes_learning: List[Dict] = []
        self._feedback_learning: List[Dict] = []
        self._domain_adjustments_learning: Dict[str, float] = {}
        
        # Load persisted state
        self._load_state()
    
    # ===== Learning Interface Methods =====
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        """Record threshold outcome for learning."""
        self._outcomes_learning.append(outcome)
        # Also record in outcome memory for persistence
        if self.outcome_memory:
            self.outcome_memory.record({
                'domain': outcome.get('domain', 'general'),
                'threshold': outcome.get('threshold', 50.0),
                'correct': outcome.get('correct', False)
            })
    
    def record_feedback(self, feedback: Dict) -> None:
        """Record feedback on threshold decisions."""
        self._feedback_learning.append(feedback)
        domain = feedback.get('domain', 'general')
        if feedback.get('threshold_too_strict', False):
            self._domain_adjustments_learning[domain] = self._domain_adjustments_learning.get(domain, 0.0) - 2.0
        elif feedback.get('threshold_too_lenient', False):
            self._domain_adjustments_learning[domain] = self._domain_adjustments_learning.get(domain, 0.0) + 2.0
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt thresholds based on performance data."""
        if performance_data:
            fp_rate = performance_data.get('false_positive_rate', 0)
            fn_rate = performance_data.get('false_negative_rate', 0)
            current = self.domain_base_thresholds.get(domain, 60.0)
            if fp_rate > 0.2:
                self.domain_base_thresholds[domain] = min(90.0, current + 5.0)
            if fn_rate > 0.2:
                self.domain_base_thresholds[domain] = max(30.0, current - 5.0)
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get learned adjustment for domain."""
        return self._domain_adjustments_learning.get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics."""
        return {
            'algorithm': self.algorithm_name,
            'domain_base_thresholds': dict(self.domain_base_thresholds),
            'context_adjustments': dict(self.context_adjustments),
            'recent_accuracy': list(self.recent_accuracy[-20:]),
            'domain_adjustments': dict(self._domain_adjustments_learning),
            'outcomes_recorded': len(self._outcomes_learning),
            'feedback_recorded': len(self._feedback_learning)
        }
    
    def get_threshold(self,
                      domain: str,
                      context: Dict[str, Any] = None,
                      query_embedding: bytes = None) -> ThresholdDecision:
        """
        Determine the appropriate threshold for a decision.
        
        This is the main entry point for threshold determination.
        Considers:
        1. Learned base threshold for domain
        2. Current operational state
        3. Risk level from context
        4. Similar past cases
        5. Context-specific adjustments
        """
        reasoning = []
        
        # 1. Get learned base threshold
        base_threshold = self.algorithm.get_value(domain, context)
        reasoning.append(f"Base threshold from {self.algorithm_name}: {base_threshold:.1f}")
        
        # 2. Apply state multiplier
        state_multiplier = self.state_machine.get_multiplier()
        state = self.state_machine.state
        reasoning.append(f"State {state.value} multiplier: {state_multiplier:.2f}")
        
        # 3. Apply risk multiplier
        risk_level = self._determine_risk_level(context or {})
        risk_multiplier = self.RISK_MULTIPLIERS.get(risk_level, 1.0)
        reasoning.append(f"Risk level '{risk_level}' multiplier: {risk_multiplier:.2f}")
        
        # 4. Context-specific adjustment
        context_key = self._get_context_key(context)
        context_adjustment = self.context_adjustments.get(context_key, 0.0)
        if context_adjustment != 0:
            reasoning.append(f"Context adjustment: {context_adjustment:+.1f}")
        
        # 5. Similar case adjustment (optional)
        similar_adjustment = 0.0
        if query_embedding:
            similar_adjustment = self._get_similar_case_adjustment(
                query_embedding, domain
            )
            if similar_adjustment != 0:
                reasoning.append(f"Similar case adjustment: {similar_adjustment:+.1f}")
        
        # Compute final threshold
        final_threshold = (
            base_threshold * state_multiplier * risk_multiplier +
            context_adjustment +
            similar_adjustment
        )
        
        # Clamp to valid range
        final_threshold = max(30.0, min(90.0, final_threshold))
        reasoning.append(f"Final threshold: {final_threshold:.1f}")
        
        # Compute confidence
        confidence = self._compute_confidence(domain)
        
        return ThresholdDecision(
            base_threshold=base_threshold,
            state_multiplier=state_multiplier,
            risk_multiplier=risk_multiplier,
            context_adjustment=context_adjustment + similar_adjustment,
            final_threshold=final_threshold,
            domain=domain,
            confidence=confidence,
            algorithm_used=self.algorithm_name,
            reasoning=reasoning
        )
    
    def record_outcome(self,
                      domain: str,
                      accuracy: float,
                      threshold_used: float,
                      was_accepted: bool,
                      was_correct: bool,
                      context: Dict[str, Any] = None,
                      query: str = None,
                      query_embedding: bytes = None) -> Dict[str, Any]:
        """
        Record outcome and trigger learning.
        
        This is called after every decision to:
        1. Update learning algorithm
        2. Update state machine
        3. Persist to outcome memory
        4. Update context adjustments
        5. Track algorithm performance
        """
        # 1. Create learning outcome
        outcome = LearningOutcome(
            domain=domain,
            context_features=context or {},
            accuracy=accuracy,
            threshold_used=threshold_used,
            was_accepted=was_accepted,
            was_correct=was_correct
        )
        
        # 2. Update learning algorithm
        update_result = self.algorithm.update(outcome)
        
        # 3. Update state machine
        state_result = self.state_machine.record_outcome(
            domain=domain,
            accuracy=accuracy,
            threshold=threshold_used,
            was_accepted=was_accepted,
            was_correct=was_correct,
            details={'context': context}
        )
        
        # 4. Persist to outcome memory
        record = DecisionRecord(
            query=query or "",
            query_embedding=query_embedding,
            domain=domain,
            context_features=context or {},
            accuracy=accuracy,
            threshold_used=threshold_used,
            was_accepted=was_accepted,
            was_correct=was_correct,
            algorithm_used=self.algorithm_name
        )
        record_id = self.outcome_memory.record_decision(record)
        
        # 5. Record learning event
        self.outcome_memory.record_learning_event(
            event_type='threshold_update',
            domain=domain,
            old_value=update_result.get('old_value', threshold_used),
            new_value=update_result.get('new_value', threshold_used),
            trigger='outcome_feedback',
            metadata={
                'accuracy': accuracy,
                'was_accepted': was_accepted,
                'was_correct': was_correct,
                'gradient': update_result.get('gradient', 0),
                'learning_rate': update_result.get('learning_rate', 0)
            }
        )
        
        # 6. Update context adjustments
        self._update_context_adjustment(context, was_correct)
        
        # 7. Track algorithm performance
        self.recent_accuracy.append(accuracy)
        if len(self.recent_accuracy) > 100:
            self.recent_accuracy = self.recent_accuracy[-100:]
        
        if self.algorithm_name not in self.algorithm_performance:
            self.algorithm_performance[self.algorithm_name] = []
        self.algorithm_performance[self.algorithm_name].append(
            1.0 if was_correct else 0.0
        )
        
        # 8. Check if should switch algorithm
        switch_recommendation = self._check_algorithm_switch()
        
        # 9. Persist state
        self._save_state()
        
        return {
            'record_id': record_id,
            'learning_update': update_result,
            'state_update': state_result,
            'switch_recommendation': switch_recommendation
        }
    
    def update_from_feedback(self,
                            domain: str,
                            was_false_positive: bool,
                            was_false_negative: bool,
                            current_accuracy: float) -> Dict[str, Any]:
        """
        Update thresholds based on explicit user feedback.
        
        This is called when user indicates our decision was wrong.
        - False positive: We accepted something bad → raise threshold
        - False negative: We rejected something good → lower threshold
        
        Args:
            domain: Domain of the decision
            was_false_positive: If True, we accepted something we shouldn't have
            was_false_negative: If True, we rejected something we shouldn't have
            current_accuracy: The accuracy score of the decision
        
        Returns:
            Update result with new threshold information
        """
        result = {
            'domain': domain,
            'adjustment_made': False,
            'old_base': None,
            'new_base': None,
            'reason': ''
        }
        
        # Get current base threshold
        old_base = self.domain_base_thresholds.get(domain, self.DEFAULT_BASE_THRESHOLD)
        result['old_base'] = old_base
        
        if was_false_positive:
            # We accepted something bad - need to be MORE strict
            # Raise the base threshold for this domain
            adjustment = 5.0  # Raise by 5%
            new_base = min(90.0, old_base + adjustment)
            result['reason'] = 'Raising threshold due to false positive'
            result['adjustment_made'] = True
            
        elif was_false_negative:
            # We rejected something good - need to be LESS strict
            # Lower the base threshold for this domain
            adjustment = 3.0  # Lower by 3% (more conservative)
            new_base = max(30.0, old_base - adjustment)
            result['reason'] = 'Lowering threshold due to false negative'
            result['adjustment_made'] = True
            
        else:
            result['reason'] = 'No adjustment needed'
            return result
        
        # Apply the adjustment
        self.domain_base_thresholds[domain] = new_base
        result['new_base'] = new_base
        
        # Also update the learning algorithm
        outcome = LearningOutcome(
            domain=domain,
            context_features={},
            accuracy=current_accuracy,
            threshold_used=old_base,
            was_accepted=not was_false_positive,
            was_correct=False  # This is why we're getting feedback
        )
        self.algorithm.update(outcome)
        
        # Persist state
        self._save_state()
        
        return result
    
    def switch_algorithm(self, new_algorithm: str) -> Dict[str, Any]:
        """
        Switch to a different learning algorithm.
        
        Transfers learned state from current algorithm.
        """
        old_algorithm = self.algorithm_name
        old_stats = self.algorithm.get_statistics()
        
        # Create new algorithm
        new_algo = create_algorithm(new_algorithm)
        
        # Transfer learned values
        new_algo.initialize_from(self.algorithm)
        
        # Update registry
        self.registry.set_algorithm(new_algo)
        
        # Switch
        self.algorithm = new_algo
        self.algorithm_name = new_algorithm
        
        # Record event
        self.outcome_memory.record_learning_event(
            event_type='algorithm_switch',
            domain='all',
            old_value=0,
            new_value=0,
            trigger='manual_switch',
            metadata={
                'from_algorithm': old_algorithm,
                'to_algorithm': new_algorithm,
                'old_stats': old_stats
            }
        )
        
        self._save_state()
        
        return {
            'old_algorithm': old_algorithm,
            'new_algorithm': new_algorithm,
            'transferred_domains': list(old_stats.get('domain_values', {}).keys())
        }
    
    def _determine_risk_level(self, context: Dict[str, Any]) -> str:
        """Determine risk level from context."""
        # Check explicit risk level
        if 'risk_level' in context:
            return context['risk_level']
        
        # Compute from features
        risk_score = context.get('risk_score', 0.0)
        
        # Check for high-risk indicators
        high_risk_indicators = ['medication', 'investment', 'legal', 'surgery', 'diagnosis']
        query = context.get('query', '').lower()
        
        indicator_count = sum(1 for ind in high_risk_indicators if ind in query)
        risk_score += indicator_count * 0.15
        
        # Determine level
        if risk_score >= 0.7:
            return 'critical'
        elif risk_score >= 0.5:
            return 'high'
        elif risk_score >= 0.3:
            return 'medium'
        return 'low'
    
    def _get_context_key(self, context: Dict[str, Any]) -> str:
        """Generate key for context-specific adjustments."""
        if not context:
            return 'default'
        
        # Combine relevant context features
        parts = []
        if 'domain' in context:
            parts.append(f"d:{context['domain']}")
        if 'risk_level' in context:
            parts.append(f"r:{context['risk_level']}")
        if 'query_type' in context:
            parts.append(f"t:{context['query_type']}")
        
        return '_'.join(parts) if parts else 'default'
    
    def _update_context_adjustment(self, context: Dict[str, Any], was_correct: bool):
        """Update context-specific adjustment based on outcome."""
        key = self._get_context_key(context)
        
        current = self.context_adjustments.get(key, 0.0)
        
        # Small adjustment based on outcome
        if was_correct:
            # Slightly lower threshold worked
            new_value = current - 0.1
        else:
            # Threshold might need to be higher
            new_value = current + 0.2
        
        # Clamp adjustment
        new_value = max(-10.0, min(10.0, new_value))
        self.context_adjustments[key] = new_value
    
    def _get_similar_case_adjustment(self, 
                                     query_embedding: bytes,
                                     domain: str) -> float:
        """Get adjustment based on similar past cases."""
        similar = self.outcome_memory.find_similar_cases(
            query_embedding, domain, limit=5
        )
        
        if not similar:
            return 0.0
        
        # Weight by similarity and correctness
        adjustments = []
        for record, similarity in similar:
            if record.was_correct is not None and similarity > 0.7:
                if record.was_correct:
                    # Similar correct case: can be slightly more lenient
                    adjustments.append(-1.0 * similarity)
                else:
                    # Similar incorrect case: be more strict
                    adjustments.append(2.0 * similarity)
        
        if adjustments:
            return np.mean(adjustments)
        return 0.0
    
    def _compute_confidence(self, domain: str) -> float:
        """Compute confidence in threshold estimate."""
        stats = self.algorithm.get_statistics()
        
        # Based on sample count
        domain_samples = stats.get('update_counts', {}).get(domain, 0)
        sample_confidence = min(1.0, domain_samples / 100)
        
        # Based on recent accuracy
        if len(self.recent_accuracy) >= 10:
            recent_avg = np.mean(self.recent_accuracy[-20:])
            accuracy_confidence = recent_avg / 100
        else:
            accuracy_confidence = 0.5
        
        # Based on algorithm convergence
        convergence = stats.get('convergence_status', {}).get(domain, 'learning')
        convergence_confidence = {
            'converged': 1.0,
            'converging': 0.7,
            'learning': 0.5,
            'insufficient_data': 0.3,
            'no_data': 0.1
        }.get(convergence, 0.5)
        
        # Combined confidence
        return (sample_confidence * 0.3 + 
                accuracy_confidence * 0.4 + 
                convergence_confidence * 0.3)
    
    def _check_algorithm_switch(self) -> Optional[str]:
        """Check if should recommend switching algorithms."""
        if len(self.recent_accuracy) < 50:
            return None
        
        # Check if accuracy is declining
        recent_avg = np.mean(self.recent_accuracy[-20:])
        older_avg = np.mean(self.recent_accuracy[-50:-20])
        
        if recent_avg < older_avg - 5:
            # Accuracy declining, might need to try different algorithm
            return self.registry.get_best_algorithm()
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimizer statistics."""
        algo_stats = self.algorithm.get_statistics()
        state_status = self.state_machine.get_status()
        memory_stats = self.outcome_memory.get_statistics()
        
        return {
            'algorithm': {
                'name': self.algorithm_name,
                'statistics': algo_stats
            },
            'state_machine': state_status,
            'outcome_memory': memory_stats,
            'context_adjustments': dict(self.context_adjustments),
            'recent_accuracy': {
                'mean': np.mean(self.recent_accuracy) if self.recent_accuracy else 0,
                'std': np.std(self.recent_accuracy) if self.recent_accuracy else 0,
                'samples': len(self.recent_accuracy)
            },
            'algorithm_performance': {
                name: {
                    'accuracy': np.mean(perf) if perf else 0,
                    'samples': len(perf)
                }
                for name, perf in self.algorithm_performance.items()
            }
        }
    
    def get_learning_report(self) -> Dict[str, Any]:
        """Get detailed learning report."""
        algo_stats = self.algorithm.get_statistics()
        accuracy_by_domain = self.outcome_memory.get_accuracy_by_domain()
        learning_history = self.outcome_memory.get_learning_history(limit=50)
        
        return {
            'summary': {
                'total_decisions': self.outcome_memory.get_statistics()['total_decisions'],
                'algorithm': self.algorithm_name,
                'current_state': self.state_machine.state.value,
                'domains_learned': len(algo_stats.get('domain_values', {}))
            },
            'thresholds_by_domain': algo_stats.get('domain_values', {}),
            'accuracy_by_domain': accuracy_by_domain,
            'convergence_status': algo_stats.get('convergence_status', {}),
            'recent_learning_events': learning_history[:10],
            'recommendations': self._generate_recommendations(algo_stats, accuracy_by_domain)
        }
    
    def _generate_recommendations(self, 
                                  algo_stats: Dict,
                                  accuracy_by_domain: Dict) -> List[str]:
        """Generate recommendations based on current state."""
        recommendations = []
        
        # Check for domains with low accuracy
        for domain, stats in accuracy_by_domain.items():
            if stats.get('accuracy_rate', 1.0) < 0.8:
                recommendations.append(
                    f"Domain '{domain}' has low accuracy ({stats['accuracy_rate']:.0%}). "
                    f"Consider increasing threshold or reviewing detection logic."
                )
        
        # Check for unconverged domains
        convergence = algo_stats.get('convergence_status', {})
        for domain, status in convergence.items():
            if status == 'learning' and algo_stats.get('update_counts', {}).get(domain, 0) > 200:
                recommendations.append(
                    f"Domain '{domain}' still learning after many samples. "
                    f"Consider trying a different algorithm."
                )
        
        # Check state machine health
        state = self.state_machine.state
        if state in [OperationalState.CRISIS, OperationalState.DEGRADED]:
            recommendations.append(
                f"System in {state.value} state. Review recent violations and "
                f"consider manual threshold adjustment."
            )
        
        # Check algorithm performance
        for algo_name, perf in self.algorithm_performance.items():
            if len(perf) >= 50 and np.mean(perf[-50:]) < 0.7:
                recommendations.append(
                    f"Algorithm '{algo_name}' underperforming. Consider switching."
                )
        
        return recommendations if recommendations else ["System operating normally."]
    
    def _save_state(self):
        """Persist optimizer state."""
        state = {
            'algorithm_name': self.algorithm_name,
            'algorithm_state': self.algorithm.get_state().to_json(),
            'context_adjustments': self.context_adjustments,
            'recent_accuracy': self.recent_accuracy[-100:],
            'algorithm_performance': {
                k: v[-100:] for k, v in self.algorithm_performance.items()
            }
        }
        
        state_path = self.data_dir / "optimizer_state.json"
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Also save registry
        self.registry.save()
    
    def _load_state(self):
        """Load optimizer state."""
        state_path = self.data_dir / "optimizer_state.json"
        
        if not state_path.exists():
            return
        
        try:
            with open(state_path) as f:
                state = json.load(f)
            
            # Load algorithm
            if 'algorithm_name' in state:
                self.algorithm_name = state['algorithm_name']
                self.algorithm = create_algorithm(self.algorithm_name)
                
                if 'algorithm_state' in state:
                    algo_state = LearningState.from_json(state['algorithm_state'])
                    self.algorithm.load_state(algo_state)
            
            # Load other state
            self.context_adjustments = state.get('context_adjustments', {})
            self.recent_accuracy = state.get('recent_accuracy', [])
            self.algorithm_performance = state.get('algorithm_performance', {})
            
            # Update registry
            self.registry.set_algorithm(self.algorithm)
            
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")

    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback."""
        pass  # Implement specific learning logic


    def serialize_state(self) -> Dict:
        """Serialize state."""
        return {'outcomes': getattr(self, '_outcomes', [])[-100:]}


    def deserialize_state(self, state: Dict) -> None:
        """Restore state."""
        self._outcomes = state.get('outcomes', [])


# Alias for documentation compatibility
ThresholdOptimizer = AdaptiveThresholdOptimizer
