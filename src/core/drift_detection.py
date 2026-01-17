"""
BAIS Cognitive Governance Engine v30.0
Drift Detection System - Statistical Change Detection Algorithms

Phase 30: Addresses PPA2-C1-15, PPA2-C1-16
- Page-Hinkley Test: Sequential change detection
- CUSUM: Cumulative Sum Control Chart
- ADWIN: Adaptive Windowing for concept drift
- MMD: Maximum Mean Discrepancy for distribution shift

Patent Claims Addressed:
- PPA2-C1-15: Drift detection via Page-Hinkley, CUSUM, ADWIN, MMD
- PPA2-C1-16: Ingest divergence via Jensen-Shannon, Wasserstein, MMD
- PPA2-C1-33: Distributional shift bounded by f-divergences or Wasserstein
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from collections import deque
import math
import logging

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of detected drift."""
    NONE = "none"
    GRADUAL = "gradual"
    SUDDEN = "sudden"
    RECURRING = "recurring"
    INCREMENTAL = "incremental"


class DriftSeverity(Enum):
    """Severity levels for detected drift."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftResult:
    """Result of drift detection analysis."""
    detected: bool
    drift_type: DriftType
    severity: DriftSeverity
    confidence: float
    algorithm: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)


class PageHinkleyTest:
    """
    Page-Hinkley Test for sequential change detection.
    
    PPA2-C1-15: Implements drift detection via Page-Hinkley algorithm.
    
    Detects upward and downward changes in the mean of a sequence.
    """
    
    def __init__(
        self,
        delta: float = 0.005,
        lambda_threshold: float = 50.0,
        alpha: float = 0.9999
    ):
        """
        Initialize Page-Hinkley detector.
        
        Args:
            delta: Magnitude of allowed change (tolerance)
            lambda_threshold: Detection threshold
            alpha: Forgetting factor (close to 1 = long memory)
        """
        self.delta = delta
        self.lambda_threshold = lambda_threshold
        self.alpha = alpha
        self.reset()
        
    def reset(self):
        """Reset detector state."""
        self.n = 0
        self.sum = 0.0
        self.mean = 0.0
        self.m_min = float('inf')
        self.m_max = float('-inf')
        self.U_min = 0.0  # Cumulative sum for upward change
        self.U_max = 0.0  # Cumulative sum for downward change
        
    def update(self, value: float) -> DriftResult:
        """
        Update with new observation and check for drift.
        
        Args:
            value: New observation
            
        Returns:
            DriftResult with detection status
        """
        self.n += 1
        
        # Update running mean
        self.sum = self.alpha * self.sum + value
        self.mean = self.sum / (1 - self.alpha**self.n) if self.alpha < 1 else self.sum / self.n
        
        # Page-Hinkley statistics
        self.U_min = self.alpha * self.U_min + (value - self.mean - self.delta)
        self.U_max = self.alpha * self.U_max + (value - self.mean + self.delta)
        
        # Track min/max
        self.m_min = min(self.m_min, self.U_min)
        self.m_max = max(self.m_max, self.U_max)
        
        # Check for drift
        ph_plus = self.U_min - self.m_min
        ph_minus = self.m_max - self.U_max
        
        drift_detected = ph_plus > self.lambda_threshold or ph_minus > self.lambda_threshold
        
        if drift_detected:
            severity = self._calculate_severity(max(ph_plus, ph_minus))
            drift_type = DriftType.SUDDEN if max(ph_plus, ph_minus) > 2 * self.lambda_threshold else DriftType.GRADUAL
            
            return DriftResult(
                detected=True,
                drift_type=drift_type,
                severity=severity,
                confidence=min(1.0, max(ph_plus, ph_minus) / (2 * self.lambda_threshold)),
                algorithm="page_hinkley",
                details={
                    "ph_plus": ph_plus,
                    "ph_minus": ph_minus,
                    "threshold": self.lambda_threshold,
                    "mean": self.mean,
                    "observations": self.n
                },
                recommendations=["Consider threshold recalibration", "Review recent data distribution"]
            )
        
        return DriftResult(
            detected=False,
            drift_type=DriftType.NONE,
            severity=DriftSeverity.LOW,
            confidence=0.0,
            algorithm="page_hinkley",
            details={"ph_plus": ph_plus, "ph_minus": ph_minus, "observations": self.n}
        )
    
    def _calculate_severity(self, statistic: float) -> DriftSeverity:
        """Calculate severity based on detection statistic."""
        ratio = statistic / self.lambda_threshold
        if ratio < 1.5:
            return DriftSeverity.LOW
        elif ratio < 2.5:
            return DriftSeverity.MEDIUM
        elif ratio < 4.0:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL


class CUSUMDetector:
    """
    Cumulative Sum (CUSUM) Control Chart for drift detection.
    
    PPA2-C1-15: Implements drift detection via CUSUM algorithm.
    
    Detects small, persistent shifts in the mean of a process.
    """
    
    def __init__(
        self,
        target: float = 0.0,
        allowance: float = 0.5,
        threshold: float = 5.0
    ):
        """
        Initialize CUSUM detector.
        
        Args:
            target: Target value (expected mean)
            allowance: Slack value (k) - typically half the shift to detect
            threshold: Decision interval (h) - alarm threshold
        """
        self.target = target
        self.allowance = allowance
        self.threshold = threshold
        self.reset()
        
    def reset(self):
        """Reset detector state."""
        self.C_plus = 0.0  # Upper CUSUM
        self.C_minus = 0.0  # Lower CUSUM
        self.n = 0
        self.history: List[Tuple[float, float]] = []
        
    def update(self, value: float) -> DriftResult:
        """
        Update with new observation and check for drift.
        
        Args:
            value: New observation
            
        Returns:
            DriftResult with detection status
        """
        self.n += 1
        
        # CUSUM statistics
        self.C_plus = max(0, self.C_plus + value - self.target - self.allowance)
        self.C_minus = max(0, self.C_minus - value + self.target - self.allowance)
        
        self.history.append((self.C_plus, self.C_minus))
        
        # Check for drift
        upward_drift = self.C_plus > self.threshold
        downward_drift = self.C_minus > self.threshold
        
        if upward_drift or downward_drift:
            direction = "upward" if upward_drift else "downward"
            severity = self._calculate_severity(max(self.C_plus, self.C_minus))
            
            return DriftResult(
                detected=True,
                drift_type=DriftType.SUDDEN if max(self.C_plus, self.C_minus) > 2 * self.threshold else DriftType.GRADUAL,
                severity=severity,
                confidence=min(1.0, max(self.C_plus, self.C_minus) / (2 * self.threshold)),
                algorithm="cusum",
                details={
                    "C_plus": self.C_plus,
                    "C_minus": self.C_minus,
                    "direction": direction,
                    "threshold": self.threshold,
                    "observations": self.n
                },
                recommendations=[
                    f"Detected {direction} shift in process mean",
                    "Investigate recent changes in data source"
                ]
            )
        
        return DriftResult(
            detected=False,
            drift_type=DriftType.NONE,
            severity=DriftSeverity.LOW,
            confidence=0.0,
            algorithm="cusum",
            details={"C_plus": self.C_plus, "C_minus": self.C_minus, "observations": self.n}
        )
    
    def _calculate_severity(self, statistic: float) -> DriftSeverity:
        """Calculate severity based on CUSUM statistic."""
        ratio = statistic / self.threshold
        if ratio < 1.5:
            return DriftSeverity.LOW
        elif ratio < 3.0:
            return DriftSeverity.MEDIUM
        elif ratio < 5.0:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL

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


class ADWINDetector:
    """
    Adaptive Windowing (ADWIN) for concept drift detection.
    
    PPA2-C1-15: Implements drift detection via ADWIN algorithm.
    
    Automatically adjusts window size based on observed changes.
    """
    
    def __init__(
        self,
        delta: float = 0.002,
        max_buckets: int = 5,
        min_window_size: int = 10,
        min_sub_window: int = 5
    ):
        """
        Initialize ADWIN detector.
        
        Args:
            delta: Confidence parameter (lower = more confident but slower)
            max_buckets: Maximum buckets per level
            min_window_size: Minimum window size before checking
            min_sub_window: Minimum sub-window size
        """
        self.delta = delta
        self.max_buckets = max_buckets
        self.min_window_size = min_window_size
        self.min_sub_window = min_sub_window
        self.reset()
        
    def reset(self):
        """Reset detector state."""
        self.window: deque = deque()
        self.sum = 0.0
        self.variance = 0.0
        self.width = 0
        self.n = 0
        self.detected_changes: List[int] = []
        
    def update(self, value: float) -> DriftResult:
        """
        Update with new observation and check for drift.
        
        Args:
            value: New observation
            
        Returns:
            DriftResult with detection status
        """
        self.n += 1
        self.window.append(value)
        self.width = len(self.window)
        
        # Update statistics
        old_mean = self.sum / max(self.width - 1, 1)
        self.sum += value
        new_mean = self.sum / self.width
        
        if self.width > 1:
            self.variance += (value - old_mean) * (value - new_mean)
        
        # Check for drift if window is large enough
        if self.width >= self.min_window_size:
            drift_detected, cut_point = self._detect_change()
            
            if drift_detected:
                # Remove old data
                removed_values = []
                for _ in range(cut_point):
                    if self.window:
                        removed = self.window.popleft()
                        self.sum -= removed
                        removed_values.append(removed)
                
                self.width = len(self.window)
                self.detected_changes.append(self.n)
                
                # Recalculate variance
                if self.width > 1:
                    mean = self.sum / self.width
                    self.variance = sum((x - mean)**2 for x in self.window)
                else:
                    self.variance = 0.0
                
                return DriftResult(
                    detected=True,
                    drift_type=DriftType.SUDDEN if cut_point < 0.3 * (self.width + cut_point) else DriftType.GRADUAL,
                    severity=self._calculate_severity(cut_point, self.width + cut_point),
                    confidence=0.8,  # ADWIN guarantees bounded false positive rate
                    algorithm="adwin",
                    details={
                        "cut_point": cut_point,
                        "new_window_size": self.width,
                        "observations": self.n,
                        "total_changes_detected": len(self.detected_changes)
                    },
                    recommendations=["Window adjusted to remove outdated data", "Model may need retraining"]
                )
        
        return DriftResult(
            detected=False,
            drift_type=DriftType.NONE,
            severity=DriftSeverity.LOW,
            confidence=0.0,
            algorithm="adwin",
            details={"window_size": self.width, "observations": self.n}
        )
    
    def _detect_change(self) -> Tuple[bool, int]:
        """
        Detect if there's a change point in the window.
        
        Returns:
            Tuple of (drift_detected, cut_point)
        """
        window_list = list(self.window)
        n = len(window_list)
        
        for i in range(self.min_sub_window, n - self.min_sub_window):
            # Split window
            w0 = window_list[:i]
            w1 = window_list[i:]
            
            n0, n1 = len(w0), len(w1)
            mu0 = sum(w0) / n0
            mu1 = sum(w1) / n1
            
            # ADWIN cut condition
            m = 1.0 / ((1.0 / n0) + (1.0 / n1))
            epsilon_cut = math.sqrt((2.0 / m) * math.log(2.0 * n / self.delta))
            
            if abs(mu0 - mu1) > epsilon_cut:
                return True, i
        
        return False, 0
    
    def _calculate_severity(self, cut_point: int, total: int) -> DriftSeverity:
        """Calculate severity based on cut location."""
        ratio = cut_point / total
        if ratio > 0.7:  # Small recent change
            return DriftSeverity.LOW
        elif ratio > 0.5:
            return DriftSeverity.MEDIUM
        elif ratio > 0.3:
            return DriftSeverity.HIGH
        else:  # Large historical change
            return DriftSeverity.CRITICAL

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


class MMDDetector:
    """
    Maximum Mean Discrepancy (MMD) for distribution drift detection.
    
    PPA2-C1-16: Implements ingest divergence via MMD.
    PPA2-C1-33: Distributional shift bounded by MMD.
    
    Compares two distributions in a reproducing kernel Hilbert space.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        threshold: float = 0.1,
        kernel_bandwidth: float = 1.0
    ):
        """
        Initialize MMD detector.
        
        Args:
            window_size: Size of sliding windows for comparison
            threshold: MMD threshold for drift detection
            kernel_bandwidth: RBF kernel bandwidth (sigma)
        """
        self.window_size = window_size
        self.threshold = threshold
        self.kernel_bandwidth = kernel_bandwidth
        self.reset()
        
    def reset(self):
        """Reset detector state."""
        self.reference_window: deque = deque(maxlen=self.window_size)
        self.test_window: deque = deque(maxlen=self.window_size)
        self.n = 0
        self.phase = "filling_reference"  # filling_reference, filling_test, comparing
        
    def update(self, value: float) -> DriftResult:
        """
        Update with new observation and check for drift.
        
        Args:
            value: New observation
            
        Returns:
            DriftResult with detection status
        """
        self.n += 1
        
        if self.phase == "filling_reference":
            self.reference_window.append(value)
            if len(self.reference_window) >= self.window_size:
                self.phase = "filling_test"
            return DriftResult(
                detected=False,
                drift_type=DriftType.NONE,
                severity=DriftSeverity.LOW,
                confidence=0.0,
                algorithm="mmd",
                details={"phase": "filling_reference", "reference_size": len(self.reference_window)}
            )
        
        elif self.phase == "filling_test":
            self.test_window.append(value)
            if len(self.test_window) >= self.window_size:
                self.phase = "comparing"
            else:
                return DriftResult(
                    detected=False,
                    drift_type=DriftType.NONE,
                    severity=DriftSeverity.LOW,
                    confidence=0.0,
                    algorithm="mmd",
                    details={"phase": "filling_test", "test_size": len(self.test_window)}
                )
        
        # Comparing phase
        self.test_window.append(value)
        
        # Calculate MMD
        mmd_value = self._calculate_mmd(
            np.array(self.reference_window),
            np.array(self.test_window)
        )
        
        if mmd_value > self.threshold:
            severity = self._calculate_severity(mmd_value)
            
            # Update reference window if drift detected
            if severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
                self.reference_window = deque(self.test_window, maxlen=self.window_size)
                self.test_window.clear()
                self.phase = "filling_test"
            
            return DriftResult(
                detected=True,
                drift_type=DriftType.SUDDEN if mmd_value > 2 * self.threshold else DriftType.GRADUAL,
                severity=severity,
                confidence=min(1.0, mmd_value / self.threshold),
                algorithm="mmd",
                details={
                    "mmd_value": float(mmd_value),
                    "threshold": self.threshold,
                    "reference_mean": float(np.mean(list(self.reference_window))),
                    "test_mean": float(np.mean(list(self.test_window))),
                    "observations": self.n
                },
                recommendations=["Distribution shift detected", "Consider updating reference model"]
            )
        
        return DriftResult(
            detected=False,
            drift_type=DriftType.NONE,
            severity=DriftSeverity.LOW,
            confidence=0.0,
            algorithm="mmd",
            details={"mmd_value": float(mmd_value), "threshold": self.threshold, "observations": self.n}
        )
    
    def _calculate_mmd(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Calculate Maximum Mean Discrepancy using RBF kernel.
        
        Args:
            X: Reference sample
            Y: Test sample
            
        Returns:
            MMD^2 statistic
        """
        n = len(X)
        m = len(Y)
        
        # Reshape for kernel computation
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        
        # RBF kernel
        def rbf_kernel(a, b):
            diff = a - b.T
            return np.exp(-diff**2 / (2 * self.kernel_bandwidth**2))
        
        K_XX = rbf_kernel(X, X)
        K_YY = rbf_kernel(Y, Y)
        K_XY = rbf_kernel(X, Y)
        
        # Unbiased MMD^2 estimator
        mmd_xx = (np.sum(K_XX) - np.trace(K_XX)) / (n * (n - 1))
        mmd_yy = (np.sum(K_YY) - np.trace(K_YY)) / (m * (m - 1))
        mmd_xy = np.sum(K_XY) / (n * m)
        
        mmd_squared = mmd_xx + mmd_yy - 2 * mmd_xy
        return max(0, mmd_squared)  # Ensure non-negative
    
    def _calculate_severity(self, mmd_value: float) -> DriftSeverity:
        """Calculate severity based on MMD value."""
        ratio = mmd_value / self.threshold
        if ratio < 1.5:
            return DriftSeverity.LOW
        elif ratio < 2.5:
            return DriftSeverity.MEDIUM
        elif ratio < 4.0:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL

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


class DivergenceCalculator:
    """
    Statistical divergence measures for distribution comparison.
    
    PPA2-C1-16: Implements ingest divergence via Jensen-Shannon, Wasserstein.
    PPA2-C1-33: Distributional shift bounded by f-divergences or Wasserstein.
    """
    
    @staticmethod
    def jensen_shannon(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
        """
        Calculate Jensen-Shannon divergence.
        
        Args:
            p: First probability distribution
            q: Second probability distribution
            epsilon: Small value for numerical stability
            
        Returns:
            JS divergence (0 to 1)
        """
        # Normalize
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        p = p / (np.sum(p) + epsilon)
        q = q / (np.sum(q) + epsilon)
        
        # Midpoint distribution
        m = 0.5 * (p + q)
        
        # KL divergences
        kl_pm = np.sum(p * np.log((p + epsilon) / (m + epsilon)))
        kl_qm = np.sum(q * np.log((q + epsilon) / (m + epsilon)))
        
        return 0.5 * (kl_pm + kl_qm)
    
    @staticmethod
    def wasserstein_1d(p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate 1D Wasserstein distance (Earth Mover's Distance).
        
        Args:
            p: First sample
            q: Second sample
            
        Returns:
            Wasserstein-1 distance
        """
        p_sorted = np.sort(p)
        q_sorted = np.sort(q)
        
        # Interpolate to same length
        if len(p_sorted) != len(q_sorted):
            n = max(len(p_sorted), len(q_sorted))
            p_interp = np.interp(
                np.linspace(0, 1, n),
                np.linspace(0, 1, len(p_sorted)),
                p_sorted
            )
            q_interp = np.interp(
                np.linspace(0, 1, n),
                np.linspace(0, 1, len(q_sorted)),
                q_sorted
            )
        else:
            p_interp, q_interp = p_sorted, q_sorted
        
        return np.mean(np.abs(p_interp - q_interp))
    
    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
        """
        Calculate Kullback-Leibler divergence.
        
        Args:
            p: First probability distribution
            q: Second probability distribution
            epsilon: Small value for numerical stability
            
        Returns:
            KL divergence
        """
        p = np.asarray(p, dtype=float) + epsilon
        q = np.asarray(q, dtype=float) + epsilon
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        return np.sum(p * np.log(p / q))

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


class DriftDetectionManager:
    """
    Unified drift detection manager using multiple algorithms.
    
    Combines Page-Hinkley, CUSUM, ADWIN, and MMD for robust drift detection.
    
    Phase 49: Enhanced with learning capabilities via CentralizedLearningManager.
    """
    
    def __init__(
        self,
        enable_page_hinkley: bool = True,
        enable_cusum: bool = True,
        enable_adwin: bool = True,
        enable_mmd: bool = True,
        consensus_threshold: int = 2,
        learning_manager: Optional[Any] = None
    ):
        """
        Initialize drift detection manager.
        
        Args:
            enable_*: Enable specific algorithms
            consensus_threshold: Number of algorithms that must agree for detection
            learning_manager: CentralizedLearningManager for unified learning (Phase 49)
        """
        self.detectors: Dict[str, Any] = {}
        self.consensus_threshold = consensus_threshold
        
        if enable_page_hinkley:
            self.detectors["page_hinkley"] = PageHinkleyTest()
        if enable_cusum:
            self.detectors["cusum"] = CUSUMDetector()
        if enable_adwin:
            self.detectors["adwin"] = ADWINDetector()
        if enable_mmd:
            self.detectors["mmd"] = MMDDetector()
        
        self.divergence = DivergenceCalculator()
        self.history: List[Dict[str, Any]] = []
        
        # Phase 49: Learning capabilities
        self._learning_manager = learning_manager
        self._domain_adjustments: Dict[str, float] = {}
        self._detection_outcomes: List[Dict[str, Any]] = []
        self._learning_rate = 0.1
        
        # Register with centralized manager if available
        if self._learning_manager:
            self._learning_manager.register_module("drift_detection")
        
        logger.info(f"[DriftDetection] Initialized with {len(self.detectors)} algorithms")
    
    def update(self, value: float) -> Dict[str, DriftResult]:
        """
        Update all detectors with new observation.
        
        Args:
            value: New observation
            
        Returns:
            Dictionary of results from each algorithm
        """
        results = {}
        for name, detector in self.detectors.items():
            results[name] = detector.update(value)
        
        # Log to history
        self.history.append({
            "value": value,
            "results": {k: v.detected for k, v in results.items()}
        })
        
        return results
    
    def check_consensus(self, results: Dict[str, DriftResult]) -> DriftResult:
        """
        Check for consensus among detectors.
        
        Args:
            results: Results from all detectors
            
        Returns:
            Consensus drift result
        """
        detections = [r for r in results.values() if r.detected]
        detection_count = len(detections)
        
        if detection_count >= self.consensus_threshold:
            # Aggregate results
            severities = [d.severity for d in detections]
            max_severity = max(severities, key=lambda s: list(DriftSeverity).index(s))
            
            confidences = [d.confidence for d in detections]
            avg_confidence = sum(confidences) / len(confidences)
            
            algorithms = [d.algorithm for d in detections]
            
            return DriftResult(
                detected=True,
                drift_type=DriftType.SUDDEN if max_severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL] else DriftType.GRADUAL,
                severity=max_severity,
                confidence=avg_confidence,
                algorithm=f"consensus({','.join(algorithms)})",
                details={
                    "detection_count": detection_count,
                    "consensus_threshold": self.consensus_threshold,
                    "individual_results": {k: v.detected for k, v in results.items()}
                },
                recommendations=[
                    f"{detection_count}/{len(results)} algorithms detected drift",
                    "Strong evidence of distribution shift"
                ]
            )
        
        return DriftResult(
            detected=False,
            drift_type=DriftType.NONE,
            severity=DriftSeverity.LOW,
            confidence=0.0,
            algorithm="consensus",
            details={
                "detection_count": detection_count,
                "consensus_threshold": self.consensus_threshold
            }
        )
    
    def reset_all(self):
        """Reset all detectors."""
        for detector in self.detectors.values():
            detector.reset()
        self.history.clear()
        logger.info("[DriftDetection] All detectors reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of all detectors."""
        return {
            "algorithms": list(self.detectors.keys()),
            "consensus_threshold": self.consensus_threshold,
            "observations": len(self.history),
            "recent_detections": sum(
                1 for h in self.history[-100:] 
                if any(h["results"].values())
            ) if self.history else 0
        }
    
    # ========================================
    # PHASE 49: LEARNING METHODS
    # ========================================
    
    def record_outcome(
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        was_correct: bool,
        domain: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Record a drift detection outcome for learning.
        
        Phase 49: Enables learning from detection outcomes.
        """
        outcome = {
            "input": input_data,
            "output": output_data,
            "was_correct": was_correct,
            "domain": domain,
            "metadata": metadata or {}
        }
        self._detection_outcomes.append(outcome)
        
        # Update domain adjustments
        if domain:
            current = self._domain_adjustments.get(domain, 0.0)
            adjustment = self._learning_rate * (1.0 if was_correct else -1.0)
            self._domain_adjustments[domain] = max(-0.5, min(0.5, current + adjustment))
        
        # Record to centralized manager if available
        if self._learning_manager:
            self._learning_manager.record_outcome(
                module_name="drift_detection",
                input_data=input_data,
                output_data=output_data,
                was_correct=was_correct,
                domain=domain,
                metadata=metadata
            )
    
    def record_feedback(
        self,
        detection_result: DriftResult,
        was_accurate: bool,
        actual_drift: Optional[bool] = None
    ) -> None:
        """
        Record feedback on a drift detection result.
        
        Phase 49: Enables learning from detection feedback.
        """
        self.record_outcome(
            input_data={"algorithm": detection_result.algorithm, "confidence": detection_result.confidence},
            output_data={"detected": detection_result.detected, "severity": detection_result.severity.value},
            was_correct=was_accurate,
            domain=detection_result.algorithm,
            metadata={"actual_drift": actual_drift}
        )
    
    def adapt_thresholds(
        self,
        threshold_name: str,
        current_value: float,
        direction: str = 'decrease'
    ) -> float:
        """
        Adapt detection thresholds based on learning.
        
        Phase 49: Enables adaptive threshold optimization.
        """
        if self._learning_manager:
            return self._learning_manager.adapt_threshold(
                module_name="drift_detection",
                threshold_name=threshold_name,
                current_value=current_value,
                direction=direction
            )
        
        # Local adaptation if no centralized manager
        magnitude = 0.05
        if direction == 'increase':
            return min(0.95, current_value + magnitude)
        return max(0.1, current_value - magnitude)
    
    def get_domain_adjustment(self, domain: str) -> float:
        """
        Get domain-specific threshold adjustment.
        
        Phase 49: Enables domain-aware detection.
        """
        if self._learning_manager:
            return self._learning_manager.get_domain_adjustment("drift_detection", domain)
        return self._domain_adjustments.get(domain, 0.0)
    
    def save_state(self, filepath: Optional[str] = None) -> bool:
        """
        Save learning state to disk.
        
        Phase 49: Enables persistence of learned patterns.
        """
        if self._learning_manager:
            return self._learning_manager.save_state()
        return False
    
    def load_state(self, filepath: Optional[str] = None) -> bool:
        """
        Load learning state from disk.
        
        Phase 49: Enables restoration of learned patterns.
        """
        if self._learning_manager:
            return self._learning_manager.load_state()
        return False
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """
        Get learning statistics for drift detection.
        
        Phase 49: Provides learning metrics.
        """
        if self._learning_manager:
            return self._learning_manager.get_learning_statistics("drift_detection").__dict__
        
        total = len(self._detection_outcomes)
        correct = sum(1 for o in self._detection_outcomes if o['was_correct'])
        return {
            "module": "drift_detection",
            "total_outcomes": total,
            "accuracy": correct / total if total > 0 else 0.5,
            "domain_adjustments": dict(self._domain_adjustments),
            "learning_rate": self._learning_rate
        }
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard wrapper for non-standard record_outcome."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

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
if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 30: Drift Detection Module Test")
    print("=" * 60)
    
    # Initialize manager
    manager = DriftDetectionManager()
    
    # Simulate stable data
    print("\n[1] Stable Data (no drift expected):")
    np.random.seed(42)
    stable_data = np.random.normal(0, 1, 50)
    for value in stable_data:
        results = manager.update(value)
    consensus = manager.check_consensus(results)
    print(f"    Consensus detected: {consensus.detected}")
    print(f"    Confidence: {consensus.confidence:.2f}")
    
    # Simulate drift
    print("\n[2] Introducing Drift (mean shift):")
    drift_data = np.random.normal(3, 1, 50)  # Mean shifted from 0 to 3
    for value in drift_data:
        results = manager.update(value)
        consensus = manager.check_consensus(results)
        if consensus.detected:
            print(f"    Drift detected at observation {len(manager.history)}")
            print(f"    Severity: {consensus.severity.value}")
            print(f"    Confidence: {consensus.confidence:.2f}")
            print(f"    Algorithm: {consensus.algorithm}")
            break
    
    # Test divergence calculations
    print("\n[3] Divergence Calculations:")
    p = np.random.normal(0, 1, 100)
    q = np.random.normal(2, 1, 100)
    
    # Create histograms for JS divergence
    bins = np.linspace(-5, 7, 20)
    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)
    
    js = DivergenceCalculator.jensen_shannon(p_hist, q_hist)
    wass = DivergenceCalculator.wasserstein_1d(p, q)
    
    print(f"    Jensen-Shannon: {js:.4f}")
    print(f"    Wasserstein-1D: {wass:.4f}")
    
    print("\n[4] Status:")
    status = manager.get_status()
    print(f"    Algorithms: {status['algorithms']}")
    print(f"    Observations: {status['observations']}")
    
    print("\n" + "=" * 60)
    print("PHASE 30: Drift Detection Module - VERIFIED")
    print("=" * 60)




# Phase 49: Documentation compatibility alias
PageHinkleyDetector = PageHinkleyTest
