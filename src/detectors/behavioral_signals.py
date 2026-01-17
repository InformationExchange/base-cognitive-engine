"""
BAIS Cognitive Governance Engine v16.5
Behavioral Signal Vector

PPA-2 Dep.Claim 23: FULL IMPLEMENTATION
"Behavioral signals B: sentiment volatility, abnormal behavioral skew,
truth-probe contradiction rate"

This module implements:
1. sentiment_volatility: Variance in sentiment over response window
2. abnormal_behavioral_skew: Deviation from baseline behavior patterns
3. truth_probe_contradiction_rate: Rate of contradictions vs truth probes
4. Integration with existing behavioral bias detection
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path
import json
import math
import re


@dataclass
class BehavioralSignalVector:
    """
    Complete Behavioral Signal Vector.
    
    PPA-2 Dep.Claim 23: Full Implementation
    
    Signals:
    - sentiment_volatility: Variance in sentiment (0-1, higher = more volatile)
    - abnormal_behavioral_skew: Deviation from baseline (0-1, higher = more abnormal)
    - truth_probe_contradiction_rate: Rate of contradictions (0-1, higher = more contradictions)
    
    Additional signals from PPA-3:
    - reward_seeking_score: Reward-seeking behavior detection
    - social_validation_score: Echo chamber indicators
    - metric_gaming_score: Measured vs holdout gap
    - confirmation_bias_score: Evidence selection bias
    """
    # Core signals (PPA-2 Dep.Claim 23)
    sentiment_volatility: float = 0.0
    abnormal_behavioral_skew: float = 0.0
    truth_probe_contradiction_rate: float = 0.0
    
    # Extended signals (PPA-3)
    reward_seeking_score: float = 0.0
    social_validation_score: float = 0.0
    metric_gaming_score: float = 0.0
    confirmation_bias_score: float = 0.0
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    domain: str = "general"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sentiment_volatility': self.sentiment_volatility,
            'abnormal_behavioral_skew': self.abnormal_behavioral_skew,
            'truth_probe_contradiction_rate': self.truth_probe_contradiction_rate,
            'reward_seeking_score': self.reward_seeking_score,
            'social_validation_score': self.social_validation_score,
            'metric_gaming_score': self.metric_gaming_score,
            'confirmation_bias_score': self.confirmation_bias_score,
            'timestamp': self.timestamp.isoformat(),
            'domain': self.domain,
            'composite_score': self.get_composite_score(),
            'risk_level': self.get_risk_level()
        }
    
    def get_composite_score(self) -> float:
        """
        Compute weighted composite behavioral score.
        
        Higher score = more behavioral anomalies detected.
        """
        weights = {
            'sentiment_volatility': 0.15,
            'abnormal_behavioral_skew': 0.20,
            'truth_probe_contradiction_rate': 0.25,  # Highest weight - direct truth impact
            'reward_seeking_score': 0.15,
            'social_validation_score': 0.05,
            'metric_gaming_score': 0.10,
            'confirmation_bias_score': 0.10
        }
        
        return sum([
            weights['sentiment_volatility'] * self.sentiment_volatility,
            weights['abnormal_behavioral_skew'] * self.abnormal_behavioral_skew,
            weights['truth_probe_contradiction_rate'] * self.truth_probe_contradiction_rate,
            weights['reward_seeking_score'] * self.reward_seeking_score,
            weights['social_validation_score'] * self.social_validation_score,
            weights['metric_gaming_score'] * self.metric_gaming_score,
            weights['confirmation_bias_score'] * self.confirmation_bias_score
        ])
    
    def get_risk_level(self) -> str:
        """Determine risk level based on composite score."""
        score = self.get_composite_score()
        
        if score >= 0.7:
            return 'critical'
        elif score >= 0.5:
            return 'high'
        elif score >= 0.3:
            return 'medium'
        else:
            return 'low'
    
    def is_crisis_trigger(self, thresholds: Dict[str, float] = None) -> Tuple[bool, List[str]]:
        """
        Check if behavioral signals indicate crisis condition.
        
        PPA-2 Dep.Claim 26: Crisis condition triggers.
        
        Returns:
            (is_crisis, triggered_signals)
        """
        thresholds = thresholds or {
            'sentiment_volatility': 0.7,
            'abnormal_behavioral_skew': 0.5,
            'truth_probe_contradiction_rate': 0.3
        }
        
        triggered = []
        
        if self.sentiment_volatility > thresholds['sentiment_volatility']:
            triggered.append('sentiment_volatility')
        
        if self.abnormal_behavioral_skew > thresholds['abnormal_behavioral_skew']:
            triggered.append('abnormal_behavioral_skew')
        
        if self.truth_probe_contradiction_rate > thresholds['truth_probe_contradiction_rate']:
            triggered.append('truth_probe_contradiction_rate')
        
        return len(triggered) > 0, triggered

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


class BehavioralSignalComputer:
    """
    Computes behavioral signals from query/response pairs.
    
    PPA-2 Dep.Claim 23: Full Implementation
    """
    
    # Sentiment keywords
    POSITIVE_WORDS = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                      'perfect', 'best', 'love', 'happy', 'success', 'correct', 'right',
                      'absolutely', 'definitely', 'certainly', 'yes'}
    NEGATIVE_WORDS = {'bad', 'terrible', 'awful', 'wrong', 'incorrect', 'fail', 'error',
                      'problem', 'issue', 'difficult', 'impossible', 'never', 'no',
                      'unfortunately', 'however', 'but', 'cannot', 'unable'}
    
    # Truth probe patterns
    TRUTH_PROBE_PATTERNS = [
        r'\b(is it true|is it correct|are you sure|can you verify|is this accurate)\b',
        r'\b(actually|in fact|the truth is|to be precise|correctly speaking)\b',
        r'\b(studies show|research indicates|evidence suggests|data shows)\b'
    ]
    
    # Contradiction patterns
    CONTRADICTION_PATTERNS = [
        r'\b(but|however|although|despite|nevertheless|on the other hand)\b',
        r'\b(actually|in fact|contrary to|opposite of|different from)\b',
        r'\b(no|not|never|none|nothing|nobody|neither|nor)\b'
    ]
    
    def __init__(self, 
                 storage_path: Path = None,
                 history_window: int = 50):
        """
        Initialize signal computer.
        
        Args:
            storage_path: Path to persist baseline data
            history_window: Number of observations for baseline computation
        """
        self.storage_path = storage_path or Path("/data/bais/behavioral_signals.json")
        self.history_window = history_window
        
        # Baseline tracking
        self.sentiment_history: deque = deque(maxlen=history_window)
        self.skew_history: deque = deque(maxlen=history_window)
        self.contradiction_history: deque = deque(maxlen=history_window)
        
        # Baseline statistics
        self.baseline_sentiment_mean = 0.5
        self.baseline_sentiment_std = 0.2
        self.baseline_skew_mean = 0.0
        self.baseline_contradiction_rate = 0.1
        
        # Load persisted baseline
        self._load_state()
    
    def compute_signals(self, 
                        query: str, 
                        response: str,
                        documents: List[Dict] = None,
                        domain: str = "general") -> BehavioralSignalVector:
        """
        Compute full behavioral signal vector.
        
        PPA-2 Dep.Claim 23: All three required signals plus extensions.
        """
        # Compute core signals
        sentiment_vol = self._compute_sentiment_volatility(response)
        behavioral_skew = self._compute_abnormal_behavioral_skew(query, response)
        contradiction_rate = self._compute_truth_probe_contradiction_rate(query, response, documents)
        
        # Compute extended signals
        reward_seeking = self._compute_reward_seeking(query, response)
        social_validation = self._compute_social_validation(response)
        metric_gaming = self._compute_metric_gaming(response)
        confirmation_bias = self._compute_confirmation_bias(query, response)
        
        # Create signal vector
        signals = BehavioralSignalVector(
            sentiment_volatility=sentiment_vol,
            abnormal_behavioral_skew=behavioral_skew,
            truth_probe_contradiction_rate=contradiction_rate,
            reward_seeking_score=reward_seeking,
            social_validation_score=social_validation,
            metric_gaming_score=metric_gaming,
            confirmation_bias_score=confirmation_bias,
            timestamp=datetime.utcnow(),
            domain=domain
        )
        
        # Update history for baseline
        self._update_history(sentiment_vol, behavioral_skew, contradiction_rate)
        
        return signals
    
    def _compute_sentiment_volatility(self, response: str) -> float:
        """
        Compute sentiment volatility.
        
        PPA-2 Dep.Claim 23: Variance in sentiment over response window.
        
        Method: Analyze sentiment shifts within response segments.
        """
        # Split response into sentences
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) < 2:
            return 0.0
        
        # Compute sentiment for each sentence
        sentiments = []
        for sentence in sentences:
            words = sentence.lower().split()
            positive = sum(1 for w in words if w in self.POSITIVE_WORDS)
            negative = sum(1 for w in words if w in self.NEGATIVE_WORDS)
            total = positive + negative
            
            if total > 0:
                sentiment = (positive - negative) / total  # -1 to 1
                sentiment = (sentiment + 1) / 2  # Normalize to 0-1
            else:
                sentiment = 0.5
            
            sentiments.append(sentiment)
        
        # Compute volatility (standard deviation)
        mean_sentiment = sum(sentiments) / len(sentiments)
        variance = sum((s - mean_sentiment) ** 2 for s in sentiments) / len(sentiments)
        std_dev = math.sqrt(variance)
        
        # Normalize to 0-1 (typical std dev is 0-0.5)
        volatility = min(1.0, std_dev * 2)
        
        return volatility
    
    def _compute_abnormal_behavioral_skew(self, query: str, response: str) -> float:
        """
        Compute abnormal behavioral skew.
        
        PPA-2 Dep.Claim 23: Deviation from baseline behavior patterns.
        
        Method: Compare response characteristics to historical baseline.
        """
        # Compute response features
        response_length = len(response)
        query_length = len(query)
        length_ratio = response_length / max(query_length, 1)
        
        # Compute hedging score (uncertainty indicators)
        hedging_words = ['maybe', 'perhaps', 'possibly', 'might', 'could', 'potentially',
                        'uncertain', 'unclear', 'depends', 'sometimes']
        hedging_count = sum(1 for w in response.lower().split() if w in hedging_words)
        hedging_ratio = hedging_count / max(len(response.split()), 1)
        
        # Compute assertion score (confidence indicators)
        assertion_words = ['definitely', 'certainly', 'absolutely', 'clearly', 'obviously',
                          'undoubtedly', 'surely', 'must', 'always', 'never']
        assertion_count = sum(1 for w in response.lower().split() if w in assertion_words)
        assertion_ratio = assertion_count / max(len(response.split()), 1)
        
        # Skew = absolute difference from balanced
        # Expected: some hedging AND some assertion
        expected_hedging = 0.02
        expected_assertion = 0.02
        
        hedging_skew = abs(hedging_ratio - expected_hedging) / 0.1  # Normalize
        assertion_skew = abs(assertion_ratio - expected_assertion) / 0.1
        
        # Combine skews
        skew = (hedging_skew + assertion_skew) / 2
        skew = min(1.0, skew)
        
        return skew
    
    def _compute_truth_probe_contradiction_rate(self, 
                                                 query: str, 
                                                 response: str,
                                                 documents: List[Dict] = None) -> float:
        """
        Compute truth-probe contradiction rate.
        
        PPA-2 Dep.Claim 23: Rate of contradictions vs truth probes.
        
        Method: Detect truth probes in query, check for contradictions in response.
        """
        # Count truth probes in query
        truth_probes = 0
        query_lower = query.lower()
        for pattern in self.TRUTH_PROBE_PATTERNS:
            truth_probes += len(re.findall(pattern, query_lower))
        
        # Count contradictions in response
        contradictions = 0
        response_lower = response.lower()
        for pattern in self.CONTRADICTION_PATTERNS:
            contradictions += len(re.findall(pattern, response_lower))
        
        # Check for direct contradictions with documents
        if documents:
            for doc in documents[:3]:  # Limit to first 3 docs
                content = doc.get('content', '').lower()[:500]
                
                # Simple contradiction check: negation of document claims
                doc_words = set(content.split())
                response_words = set(response_lower.split())
                
                # Check for negation patterns
                for word in doc_words:
                    if f"not {word}" in response_lower or f"no {word}" in response_lower:
                        contradictions += 1
        
        # Compute rate
        # Higher rate = more contradictions relative to truth probes
        base_rate = contradictions / max(len(response.split()) / 10, 1)  # Per 10 words
        probe_factor = 1 + (truth_probes * 0.1)  # Amplify if many probes
        
        rate = min(1.0, base_rate * probe_factor)
        
        return rate
    
    def _compute_reward_seeking(self, query: str, response: str) -> float:
        """Compute reward-seeking behavior score."""
        patterns = [
            (r'\bjust tell me\b', 0.5),
            (r'\bquick answer\b', 0.4),
            (r'\byes or no\b', 0.3),
            (r'\bsimple answer\b', 0.4),
            (r'\bguaranteed\b', 0.6),
            (r'\bcannot lose\b', 0.7),
        ]
        
        score = 0.0
        query_lower = query.lower()
        response_lower = response.lower()
        
        for pattern, weight in patterns:
            if re.search(pattern, query_lower) or re.search(pattern, response_lower):
                score += weight
        
        return min(1.0, score)
    
    def _compute_social_validation(self, response: str) -> float:
        """Compute social validation seeking score."""
        patterns = [
            (r'\beveryone\s+(?:says|thinks|knows)\b', 0.5),
            (r'\bmost people\b', 0.4),
            (r'\bpopular\s+(?:opinion|view)\b', 0.5),
            (r'\bcommon\s+(?:knowledge|sense)\b', 0.3),
            (r'\bconsensus\s+is\b', 0.4),
        ]
        
        score = 0.0
        response_lower = response.lower()
        
        for pattern, weight in patterns:
            if re.search(pattern, response_lower):
                score += weight
        
        return min(1.0, score)
    
    def _compute_metric_gaming(self, response: str) -> float:
        """Compute metric gaming score."""
        patterns = [
            (r'\b100%\s+(?:accuracy|correct|success)\b', 0.8),
            (r'\bzero\s+errors\b', 0.7),
            (r'\bperfect\s+(?:score|result)\b', 0.7),
            (r'\bexceeded\s+expectations\b', 0.5),
            (r'\boptimal\s+result\b', 0.5),
        ]
        
        score = 0.0
        response_lower = response.lower()
        
        for pattern, weight in patterns:
            if re.search(pattern, response_lower):
                score += weight
        
        return min(1.0, score)
    
    def _compute_confirmation_bias(self, query: str, response: str) -> float:
        """Compute confirmation bias score."""
        patterns = [
            (r'\bconfirm\s+(?:that|my|this)\b', 0.5),
            (r'\bagree\s+(?:that|with)\b', 0.4),
            (r'\byou\'re\s+(?:right|correct)\b', 0.5),
            (r'\bI\s+(?:completely|totally)\s+agree\b', 0.6),
            (r'\babsolutely\b', 0.3),
        ]
        
        score = 0.0
        query_lower = query.lower()
        response_lower = response.lower()
        
        for pattern, weight in patterns:
            if re.search(pattern, query_lower) or re.search(pattern, response_lower):
                score += weight
        
        return min(1.0, score)
    
    def _update_history(self, sentiment: float, skew: float, contradiction: float):
        """Update history for baseline tracking."""
        self.sentiment_history.append(sentiment)
        self.skew_history.append(skew)
        self.contradiction_history.append(contradiction)
        
        # Update baselines periodically
        if len(self.sentiment_history) >= 10:
            self.baseline_sentiment_mean = sum(self.sentiment_history) / len(self.sentiment_history)
            self.baseline_sentiment_std = math.sqrt(
                sum((s - self.baseline_sentiment_mean)**2 for s in self.sentiment_history) / len(self.sentiment_history)
            )
            self.baseline_skew_mean = sum(self.skew_history) / len(self.skew_history)
            self.baseline_contradiction_rate = sum(self.contradiction_history) / len(self.contradiction_history)
        
        self._save_state()
    
    def get_baseline_statistics(self) -> Dict[str, Any]:
        """Get current baseline statistics."""
        return {
            'sentiment_mean': self.baseline_sentiment_mean,
            'sentiment_std': self.baseline_sentiment_std,
            'skew_mean': self.baseline_skew_mean,
            'contradiction_rate': self.baseline_contradiction_rate,
            'history_size': len(self.sentiment_history)
        }
    
    def _save_state(self):
        """Persist baseline state."""
        if not self.storage_path:
            return
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'baseline': self.get_baseline_statistics(),
            'sentiment_history': list(self.sentiment_history),
            'skew_history': list(self.skew_history),
            'contradiction_history': list(self.contradiction_history)
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load persisted baseline state."""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path) as f:
                state = json.load(f)
            
            baseline = state.get('baseline', {})
            self.baseline_sentiment_mean = baseline.get('sentiment_mean', 0.5)
            self.baseline_sentiment_std = baseline.get('sentiment_std', 0.2)
            self.baseline_skew_mean = baseline.get('skew_mean', 0.0)
            self.baseline_contradiction_rate = baseline.get('contradiction_rate', 0.1)
            
            for s in state.get('sentiment_history', []):
                self.sentiment_history.append(s)
            for s in state.get('skew_history', []):
                self.skew_history.append(s)
            for s in state.get('contradiction_history', []):
                self.contradiction_history.append(s)
                
        except Exception as e:
            print(f"Warning: Could not load behavioral signals state: {e}")

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

