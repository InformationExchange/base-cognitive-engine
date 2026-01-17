"""
BASE Learning Integrator
Connects evaluation results to learning systems for continuous improvement

This module ensures that:
1. OCO thresholds adapt based on evaluation results
2. Conformal screener calibrates with real data
3. Audit system records all decisions
4. Learning persists across sessions
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import hashlib

# Data directory for persistence
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "learning_data")


@dataclass
class EvaluationFeedback:
    """Feedback from an evaluation to feed into learning."""
    claim_id: str
    query: str
    response: str
    domain: str
    effectiveness_score: float
    issues_found: List[str]
    was_blocked: bool
    ground_truth_correct: bool  # Did BASE make the right call?
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class LearningIntegrator:
    """
    Integrates evaluation results with BASE learning systems.
    
    This is the key component that enables BASE to learn from its mistakes
    and improve over time - a core patent claim for adaptive governance.
    """
    
    def __init__(self, persist: bool = True):
        self.persist = persist
        self.feedback_buffer: List[EvaluationFeedback] = []
        self.oco_learner = None
        self.conformal_screener = None
        self.audit_system = None
        
        # Create data directory
        if persist:
            os.makedirs(DATA_DIR, exist_ok=True)
        
        # Initialize learning components
        self._init_learners()
        
        # Load existing learning state
        if persist:
            self._load_state()
    
    def _init_learners(self):
        """Initialize learning components."""
        try:
            from learning.algorithms import OCOLearner
            self.oco_learner = OCOLearner()
            print("[Learning] OCO Learner initialized")
        except Exception as e:
            print(f"[Learning] OCO Learner failed: {e}")
        
        try:
            from learning.conformal import ConformalMustPassScreener
            self.conformal_screener = ConformalMustPassScreener()
            print("[Learning] Conformal Screener initialized")
        except Exception as e:
            print(f"[Learning] Conformal Screener failed: {e}")
        
        try:
            from learning.verifiable_audit import VerifiableAuditSystem
            # Don't instantiate if it requires file system
            self.audit_system = None  # Will use file-based audit
            print("[Learning] Using file-based audit system")
        except Exception as e:
            print(f"[Learning] Audit System failed: {e}")
    
    def record_evaluation(self, feedback: EvaluationFeedback):
        """
        Record an evaluation result and trigger learning updates.
        
        This is the main entry point for the learning loop.
        """
        self.feedback_buffer.append(feedback)
        
        # Update OCO thresholds
        if self.oco_learner:
            self._update_oco(feedback)
        
        # Add calibration point to conformal screener
        if self.conformal_screener:
            self._update_conformal(feedback)
        
        # Record to audit trail
        self._record_audit(feedback)
        
        # Persist periodically
        if len(self.feedback_buffer) % 10 == 0:
            self._save_state()
    
    def _update_oco(self, feedback: EvaluationFeedback):
        """Update OCO learner with evaluation feedback.
        
        CORRECTED LOGIC:
        - Score >= 70% (PASSED): Decision was good, small positive reinforcement
        - Score 50-69% (BASELINE): Detection didn't trigger, slightly lower threshold
        - Score < 50% (FAILED): Clearly missed, lower threshold more aggressively
        
        The key insight: 50% is BASELINE (no detection), not success.
        Only >= 70% is actually successful detection.
        """
        score = feedback.effectiveness_score
        
        if score >= 70:
            # Good detection - maintain or slightly raise threshold
            gradient = 0.01  # Small positive = slightly raise
            violation = False
        elif score >= 50:
            # Baseline - detection didn't trigger when it probably should have
            # Slightly lower threshold to be more sensitive
            gradient = -0.02  # Small negative = slightly lower
            violation = True
        else:
            # Clear failure - lower threshold more aggressively
            gradient = -0.05  # Larger negative = lower threshold
            violation = True
        
        # Update for the specific domain
        domain = feedback.domain or 'general'
        
        try:
            # OCO update: θ_{t+1} = θ_t + η * gradient
            if hasattr(self.oco_learner, 'thresholds'):
                current = self.oco_learner.thresholds.get(domain, 50.0)
                new_threshold = max(30, min(90, current + gradient * 10))
                self.oco_learner.thresholds[domain] = new_threshold
                
                if violation:
                    print(f"  [OCO] {domain} threshold: {current:.1f} → {new_threshold:.1f}")
        except Exception as e:
            print(f"  [OCO] Update error: {e}")
    
    def _update_conformal(self, feedback: EvaluationFeedback):
        """Add calibration point to conformal screener."""
        try:
            # Add score and label for calibration
            score = feedback.effectiveness_score / 100.0
            label = 1 if feedback.ground_truth_correct else 0
            
            if hasattr(self.conformal_screener, 'add_calibration_point'):
                self.conformal_screener.add_calibration_point(score, label)
            elif hasattr(self.conformal_screener, 'calibration_scores'):
                self.conformal_screener.calibration_scores.append(score)
                if hasattr(self.conformal_screener, 'calibration_labels'):
                    self.conformal_screener.calibration_labels.append(label)
        except Exception as e:
            print(f"  [Conformal] Update error: {e}")
    
    def _record_audit(self, feedback: EvaluationFeedback):
        """Record evaluation to audit trail."""
        if not self.persist:
            return
        
        audit_file = os.path.join(DATA_DIR, "audit_trail.jsonl")
        
        # Create audit record with hash for tamper evidence
        record = asdict(feedback)
        record['hash'] = hashlib.sha256(
            json.dumps(record, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        try:
            with open(audit_file, 'a') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            print(f"  [Audit] Record error: {e}")
    
    def _save_state(self):
        """Save learning state to disk."""
        if not self.persist:
            return
        
        state = {
            'timestamp': time.time(),
            'feedback_count': len(self.feedback_buffer),
            'oco_thresholds': dict(self.oco_learner.thresholds) if self.oco_learner else {},
            'conformal_calibration_count': len(self.conformal_screener.calibration_scores) if self.conformal_screener and hasattr(self.conformal_screener, 'calibration_scores') else 0,
        }
        
        state_file = os.path.join(DATA_DIR, "learning_state.json")
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            print(f"  [Learning] State saved: {len(self.feedback_buffer)} records")
        except Exception as e:
            print(f"  [Learning] Save error: {e}")
    
    def _load_state(self):
        """Load learning state from disk."""
        state_file = os.path.join(DATA_DIR, "learning_state.json")
        
        if not os.path.exists(state_file):
            return
        
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Restore OCO thresholds
            if self.oco_learner and 'oco_thresholds' in state:
                for domain, threshold in state['oco_thresholds'].items():
                    self.oco_learner.thresholds[domain] = threshold
                print(f"  [Learning] Restored OCO thresholds: {state['oco_thresholds']}")
            
            print(f"  [Learning] State loaded: {state.get('feedback_count', 0)} prior records")
        except Exception as e:
            print(f"  [Learning] Load error: {e}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get current learning statistics."""
        return {
            'total_feedback': len(self.feedback_buffer),
            'oco_thresholds': dict(self.oco_learner.thresholds) if self.oco_learner else {},
            'conformal_calibration_points': len(self.conformal_screener.calibration_scores) if self.conformal_screener and hasattr(self.conformal_screener, 'calibration_scores') else 0,
            'audit_records': self._count_audit_records(),
        }
    
    def _count_audit_records(self) -> int:
        """Count audit records."""
        audit_file = os.path.join(DATA_DIR, "audit_trail.jsonl")
        if not os.path.exists(audit_file):
            return 0
        try:
            with open(audit_file, 'r') as f:
                return sum(1 for _ in f)
        except:
            return 0
    
    def get_adapted_threshold(self, domain: str) -> float:
        """Get the adapted threshold for a domain."""
        if self.oco_learner and hasattr(self.oco_learner, 'thresholds'):
            return self.oco_learner.thresholds.get(domain, 50.0)
        return 50.0

    # Learning Interface
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

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


# Global instance for easy access
_integrator: Optional[LearningIntegrator] = None


def get_learning_integrator() -> LearningIntegrator:
    """Get the global learning integrator instance."""
    global _integrator
    if _integrator is None:
        _integrator = LearningIntegrator()
    return _integrator


def record_evaluation_feedback(
    claim_id: str,
    query: str,
    response: str,
    domain: str,
    effectiveness_score: float,
    issues_found: List[str],
    was_blocked: bool,
    ground_truth_correct: bool
):
    """Convenience function to record evaluation feedback."""
    feedback = EvaluationFeedback(
        claim_id=claim_id,
        query=query,
        response=response,
        domain=domain,
        effectiveness_score=effectiveness_score,
        issues_found=issues_found,
        was_blocked=was_blocked,
        ground_truth_correct=ground_truth_correct
    )
    get_learning_integrator().record_evaluation(feedback)


if __name__ == "__main__":
    # Test the learning integrator
    print("Testing Learning Integrator...")
    
    integrator = LearningIntegrator()
    
    # Simulate some evaluations
    test_feedbacks = [
        EvaluationFeedback(
            claim_id="PPA1-Inv1-Ind1",
            query="What is the best investment?",
            response="Buy this stock, guaranteed returns!",
            domain="financial",
            effectiveness_score=75,
            issues_found=["unrealistic_claims"],
            was_blocked=False,
            ground_truth_correct=True
        ),
        EvaluationFeedback(
            claim_id="PPA1-Inv2-Dep1",
            query="Medical advice needed",
            response="Take this medication immediately",
            domain="medical",
            effectiveness_score=40,
            issues_found=[],
            was_blocked=False,
            ground_truth_correct=False  # Should have flagged this
        ),
    ]
    
    for fb in test_feedbacks:
        integrator.record_evaluation(fb)
    
    # Check stats
    stats = integrator.get_learning_stats()
    print(f"\nLearning Stats: {json.dumps(stats, indent=2)}")
    
    # Check adapted thresholds
    print(f"\nAdapted Thresholds:")
    for domain in ['medical', 'financial', 'general']:
        threshold = integrator.get_adapted_threshold(domain)
        print(f"  {domain}: {threshold:.1f}")

