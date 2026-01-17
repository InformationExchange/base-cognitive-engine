"""
Dimensional Learning Module

Patent: NOVEL-30 - Dimensional Learning Loop

This module manages the learning feedback loop for dimensional analysis.
It tracks which dimensions were useful for which circumstances and adapts
the dimension selection strategy over time.

Purpose:
- Record outcomes from dimensional analysis
- Learn which dimensions matter for which task types
- Adapt dimension selection based on accumulated evidence
- Provide recommendations for future similar queries

Integration:
- Receives outcomes from evaluation process
- Updates dimension effectiveness scores
- Informs DimensionalExpander dimension selection
- Persists learning state across sessions
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json
import os
import tempfile
import statistics

# Import from dimensional_expander
from .dimensional_expander import (
    DimensionCategory,
    TaskType,
    ComplexityLevel,
    TASK_DIMENSION_MAP
)


class OutcomeType(Enum):
    """Types of outcomes for learning."""
    HELPFUL = "helpful"             # Analysis was useful
    NOT_HELPFUL = "not_helpful"     # Analysis was not useful
    NEUTRAL = "neutral"             # Unclear impact
    OVER_ANALYZED = "over_analyzed" # Too many dimensions used
    UNDER_ANALYZED = "under_analyzed" # Too few dimensions used


@dataclass
class AnalysisOutcome:
    """Record of an analysis outcome."""
    query_hash: str                         # Hash of query for deduplication
    task_type: TaskType
    complexity: ComplexityLevel
    dimensions_used: List[DimensionCategory]
    outcome: OutcomeType
    confidence: float
    feedback_source: str                    # e.g., "user", "automated", "llm"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DimensionEffectiveness:
    """Effectiveness score for a dimension in a task type."""
    dimension: DimensionCategory
    task_type: TaskType
    effectiveness_score: float              # 0-1 how effective
    sample_count: int                       # Number of observations
    trend: str                              # "improving", "declining", "stable"
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class LearningRecommendation:
    """Recommendation for dimension selection."""
    task_type: TaskType
    recommended_dimensions: List[DimensionCategory]
    not_recommended: List[DimensionCategory]
    confidence: float
    reasoning: str
    based_on_samples: int


@dataclass
class LearningStatistics:
    """Statistics about the learning process."""
    total_outcomes_recorded: int
    outcomes_by_type: Dict[str, int]
    task_types_analyzed: int
    dimensions_tracked: int
    average_effectiveness: float
    improvement_rate: float                 # How much has learning improved accuracy
    last_learning_update: datetime


class DimensionalLearning:
    """
    Dimensional learning feedback loop manager.
    
    Tracks which dimensions are effective for which task types and
    adapts recommendations based on accumulated evidence.
    
    Key Functions:
    - Record analysis outcomes
    - Calculate dimension effectiveness per task type
    - Generate recommendations for future queries
    - Track learning progress and improvement
    
    Patent: NOVEL-30 - Dimensional Learning Loop
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        learning_rate: float = 0.1,
        min_samples_for_recommendation: int = 5
    ):
        """
        Initialize the dimensional learning manager.
        
        Args:
            storage_path: Path for persistence
            learning_rate: How quickly to adapt (0-1)
            min_samples_for_recommendation: Minimum samples before making recommendations
        """
        if storage_path:
            self.storage_path = storage_path
        else:
            self.storage_path = os.path.join(
                tempfile.gettempdir(),
                "bais_dimensional_learning.json"
            )
        
        self.learning_rate = learning_rate
        self.min_samples = min_samples_for_recommendation
        
        # Learning state
        self._outcomes: List[AnalysisOutcome] = []
        self._effectiveness: Dict[str, Dict[str, DimensionEffectiveness]] = {}
        self._task_dimension_overrides: Dict[TaskType, List[DimensionCategory]] = {}
        
        # Tracking
        self._initial_accuracy: Optional[float] = None
        self._current_accuracy: Optional[float] = None
        
        self._load_state()
    
    def record_outcome(
        self,
        query: str,
        task_type: TaskType,
        complexity: ComplexityLevel,
        dimensions_used: List[DimensionCategory],
        outcome: OutcomeType,
        confidence: float = 0.7,
        feedback_source: str = "automated",
        metadata: Optional[Dict] = None
    ):
        """
        Record an analysis outcome for learning.
        
        Args:
            query: The original query (for hashing)
            task_type: Task type of the query
            complexity: Complexity level
            dimensions_used: Dimensions that were analyzed
            outcome: Whether the analysis was helpful
            confidence: Confidence in the outcome assessment
            feedback_source: Source of feedback
            metadata: Additional metadata
        """
        # Create query hash for deduplication
        query_hash = str(hash(query))[:16]
        
        outcome_record = AnalysisOutcome(
            query_hash=query_hash,
            task_type=task_type,
            complexity=complexity,
            dimensions_used=dimensions_used,
            outcome=outcome,
            confidence=confidence,
            feedback_source=feedback_source,
            metadata=metadata or {}
        )
        
        self._outcomes.append(outcome_record)
        
        # Update effectiveness scores
        self._update_effectiveness(outcome_record)
        
        # Update accuracy tracking
        self._update_accuracy_tracking()
        
        # Persist
        self._save_state()
    
    def _update_effectiveness(self, outcome: AnalysisOutcome):
        """
        Update effectiveness scores based on new outcome.
        """
        task_key = outcome.task_type.value
        
        if task_key not in self._effectiveness:
            self._effectiveness[task_key] = {}
        
        # Determine score adjustment
        if outcome.outcome == OutcomeType.HELPFUL:
            adjustment = self.learning_rate * outcome.confidence
        elif outcome.outcome == OutcomeType.NOT_HELPFUL:
            adjustment = -self.learning_rate * outcome.confidence
        elif outcome.outcome == OutcomeType.OVER_ANALYZED:
            adjustment = -self.learning_rate * 0.5  # Slight negative for over-analysis
        elif outcome.outcome == OutcomeType.UNDER_ANALYZED:
            adjustment = self.learning_rate * 0.5   # Encourage more analysis
        else:
            adjustment = 0.0
        
        # Update each dimension used
        for dim in outcome.dimensions_used:
            dim_key = dim.value
            
            if dim_key not in self._effectiveness[task_key]:
                self._effectiveness[task_key][dim_key] = DimensionEffectiveness(
                    dimension=dim,
                    task_type=outcome.task_type,
                    effectiveness_score=0.5,  # Start neutral
                    sample_count=0,
                    trend="stable"
                )
            
            eff = self._effectiveness[task_key][dim_key]
            
            # Update score
            old_score = eff.effectiveness_score
            new_score = max(0.0, min(1.0, old_score + adjustment))
            eff.effectiveness_score = new_score
            eff.sample_count += 1
            eff.last_updated = datetime.now()
            
            # Update trend
            if new_score > old_score + 0.05:
                eff.trend = "improving"
            elif new_score < old_score - 0.05:
                eff.trend = "declining"
            else:
                eff.trend = "stable"
    
    def _update_accuracy_tracking(self):
        """Update accuracy tracking for improvement measurement."""
        if len(self._outcomes) < 10:
            return
        
        # Calculate accuracy from recent outcomes
        recent = self._outcomes[-50:]
        helpful_count = sum(1 for o in recent if o.outcome == OutcomeType.HELPFUL)
        self._current_accuracy = helpful_count / len(recent)
        
        # Set initial accuracy from first 10 outcomes
        if self._initial_accuracy is None and len(self._outcomes) >= 10:
            first_10 = self._outcomes[:10]
            helpful_count = sum(1 for o in first_10 if o.outcome == OutcomeType.HELPFUL)
            self._initial_accuracy = helpful_count / 10
    
    def get_recommended_dimensions(
        self,
        task_type: TaskType,
        complexity: ComplexityLevel
    ) -> LearningRecommendation:
        """
        Get recommended dimensions for a task type based on learning.
        
        Args:
            task_type: The task type
            complexity: Query complexity
        
        Returns:
            LearningRecommendation with suggested dimensions
        """
        task_key = task_type.value
        
        # Get base dimensions from mapping
        base_dimensions = TASK_DIMENSION_MAP.get(task_type, []).copy()
        
        # Check if we have enough data to make recommendations
        if task_key not in self._effectiveness:
            return LearningRecommendation(
                task_type=task_type,
                recommended_dimensions=base_dimensions,
                not_recommended=[],
                confidence=0.3,
                reasoning="Insufficient data - using default mapping",
                based_on_samples=0
            )
        
        task_effectiveness = self._effectiveness[task_key]
        total_samples = sum(e.sample_count for e in task_effectiveness.values())
        
        if total_samples < self.min_samples:
            return LearningRecommendation(
                task_type=task_type,
                recommended_dimensions=base_dimensions,
                not_recommended=[],
                confidence=0.4,
                reasoning=f"Limited data ({total_samples} samples) - using default with adjustments",
                based_on_samples=total_samples
            )
        
        # Sort dimensions by effectiveness
        scored_dimensions = []
        for dim in DimensionCategory:
            dim_key = dim.value
            if dim_key in task_effectiveness:
                eff = task_effectiveness[dim_key]
                scored_dimensions.append((dim, eff.effectiveness_score, eff.sample_count))
            else:
                # Default score for dimensions we haven't seen
                scored_dimensions.append((dim, 0.5, 0))
        
        scored_dimensions.sort(key=lambda x: x[1], reverse=True)
        
        # Determine how many to recommend based on complexity
        if complexity == ComplexityLevel.SIMPLE:
            max_dims = 2
        elif complexity == ComplexityLevel.MODERATE:
            max_dims = 4
        else:
            max_dims = 6
        
        # Recommend high-scoring dimensions
        recommended = []
        not_recommended = []
        
        for dim, score, samples in scored_dimensions:
            if len(recommended) < max_dims and score >= 0.4:
                recommended.append(dim)
            elif score < 0.3 and samples >= 3:
                not_recommended.append(dim)
        
        # Ensure we have at least some dimensions
        if not recommended:
            recommended = base_dimensions[:max_dims]
        
        # Calculate confidence based on sample size
        confidence = min(0.9, 0.5 + (total_samples / 100))
        
        return LearningRecommendation(
            task_type=task_type,
            recommended_dimensions=recommended,
            not_recommended=not_recommended,
            confidence=confidence,
            reasoning=f"Based on {total_samples} samples with learned effectiveness scores",
            based_on_samples=total_samples
        )
    
    def get_effectiveness_report(
        self,
        task_type: Optional[TaskType] = None
    ) -> Dict[str, Any]:
        """
        Get effectiveness report for dimensions.
        
        Args:
            task_type: Optional filter by task type
        
        Returns:
            Dictionary with effectiveness data
        """
        if task_type:
            task_key = task_type.value
            if task_key in self._effectiveness:
                return {
                    "task_type": task_key,
                    "dimensions": {
                        k: {
                            "score": v.effectiveness_score,
                            "samples": v.sample_count,
                            "trend": v.trend
                        }
                        for k, v in self._effectiveness[task_key].items()
                    }
                }
            return {"task_type": task_key, "dimensions": {}}
        
        # All task types
        return {
            task_key: {
                k: {
                    "score": v.effectiveness_score,
                    "samples": v.sample_count,
                    "trend": v.trend
                }
                for k, v in dims.items()
            }
            for task_key, dims in self._effectiveness.items()
        }
    
    def get_learning_statistics(self) -> LearningStatistics:
        """Get overall learning statistics."""
        outcome_counts = defaultdict(int)
        for o in self._outcomes:
            outcome_counts[o.outcome.value] += 1
        
        # Calculate average effectiveness
        all_scores = []
        for task_dims in self._effectiveness.values():
            for eff in task_dims.values():
                if eff.sample_count > 0:
                    all_scores.append(eff.effectiveness_score)
        
        avg_effectiveness = statistics.mean(all_scores) if all_scores else 0.5
        
        # Calculate improvement rate
        improvement_rate = 0.0
        if self._initial_accuracy and self._current_accuracy:
            improvement_rate = self._current_accuracy - self._initial_accuracy
        
        last_update = datetime.now()
        if self._outcomes:
            last_update = self._outcomes[-1].timestamp
        
        return LearningStatistics(
            total_outcomes_recorded=len(self._outcomes),
            outcomes_by_type=dict(outcome_counts),
            task_types_analyzed=len(self._effectiveness),
            dimensions_tracked=sum(len(dims) for dims in self._effectiveness.values()),
            average_effectiveness=avg_effectiveness,
            improvement_rate=improvement_rate,
            last_learning_update=last_update
        )
    
    def adapt_task_dimension_mapping(
        self,
        task_type: TaskType
    ) -> List[DimensionCategory]:
        """
        Adapt the dimension mapping for a task type based on learning.
        
        Returns the recommended dimension ordering for the task type.
        """
        task_key = task_type.value
        base_dims = TASK_DIMENSION_MAP.get(task_type, []).copy()
        
        if task_key not in self._effectiveness:
            return base_dims
        
        task_effectiveness = self._effectiveness[task_key]
        
        # Sort base dimensions by effectiveness
        def get_score(dim):
            if dim.value in task_effectiveness:
                return task_effectiveness[dim.value].effectiveness_score
            return 0.5
        
        sorted_dims = sorted(base_dims, key=get_score, reverse=True)
        
        # Store override
        self._task_dimension_overrides[task_type] = sorted_dims
        
        return sorted_dims
    
    def get_dimension_trend(
        self,
        dimension: DimensionCategory,
        task_type: TaskType,
        window_days: int = 7
    ) -> Dict[str, Any]:
        """
        Get trend data for a dimension over time.
        """
        task_key = task_type.value
        dim_key = dimension.value
        
        if task_key not in self._effectiveness:
            return {"dimension": dim_key, "trend": "unknown", "data_points": 0}
        
        if dim_key not in self._effectiveness[task_key]:
            return {"dimension": dim_key, "trend": "unknown", "data_points": 0}
        
        eff = self._effectiveness[task_key][dim_key]
        
        # Get recent outcomes for this dimension
        cutoff = datetime.now() - timedelta(days=window_days)
        recent_outcomes = [
            o for o in self._outcomes
            if o.task_type == task_type
            and dimension in o.dimensions_used
            and o.timestamp >= cutoff
        ]
        
        return {
            "dimension": dim_key,
            "current_score": eff.effectiveness_score,
            "trend": eff.trend,
            "samples": eff.sample_count,
            "recent_outcomes": len(recent_outcomes),
            "last_updated": eff.last_updated.isoformat()
        }
    
    def reset_learning(self, task_type: Optional[TaskType] = None):
        """
        Reset learning state.
        
        Args:
            task_type: If provided, only reset for this task type
        """
        if task_type:
            task_key = task_type.value
            if task_key in self._effectiveness:
                del self._effectiveness[task_key]
            if task_type in self._task_dimension_overrides:
                del self._task_dimension_overrides[task_type]
            # Remove outcomes for this task type
            self._outcomes = [o for o in self._outcomes if o.task_type != task_type]
        else:
            self._outcomes = []
            self._effectiveness = {}
            self._task_dimension_overrides = {}
            self._initial_accuracy = None
            self._current_accuracy = None
        
        self._save_state()
    
    def _load_state(self):
        """Load learning state from disk."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    
                    # Load effectiveness
                    for task_key, dims in data.get("effectiveness", {}).items():
                        self._effectiveness[task_key] = {}
                        for dim_key, eff_data in dims.items():
                            try:
                                self._effectiveness[task_key][dim_key] = DimensionEffectiveness(
                                    dimension=DimensionCategory(dim_key),
                                    task_type=TaskType(task_key),
                                    effectiveness_score=eff_data["score"],
                                    sample_count=eff_data["samples"],
                                    trend=eff_data.get("trend", "stable")
                                )
                            except ValueError:
                                continue
                    
                    # Load outcomes (last 500)
                    for o_data in data.get("outcomes", [])[-500:]:
                        try:
                            self._outcomes.append(AnalysisOutcome(
                                query_hash=o_data["query_hash"],
                                task_type=TaskType(o_data["task_type"]),
                                complexity=ComplexityLevel(o_data["complexity"]),
                                dimensions_used=[
                                    DimensionCategory(d) for d in o_data["dimensions_used"]
                                ],
                                outcome=OutcomeType(o_data["outcome"]),
                                confidence=o_data.get("confidence", 0.7),
                                feedback_source=o_data.get("feedback_source", "automated"),
                                timestamp=datetime.fromisoformat(o_data["timestamp"])
                            ))
                        except (ValueError, KeyError):
                            continue
                    
                    self._initial_accuracy = data.get("initial_accuracy")
                    self._current_accuracy = data.get("current_accuracy")
                    
        except (json.JSONDecodeError, IOError):
            pass
    
    def _save_state(self):
        """Save learning state to disk."""
        try:
            # Prepare effectiveness data
            effectiveness_data = {}
            for task_key, dims in self._effectiveness.items():
                effectiveness_data[task_key] = {
                    dim_key: {
                        "score": eff.effectiveness_score,
                        "samples": eff.sample_count,
                        "trend": eff.trend
                    }
                    for dim_key, eff in dims.items()
                }
            
            # Prepare outcomes data (last 500)
            outcomes_data = [
                {
                    "query_hash": o.query_hash,
                    "task_type": o.task_type.value,
                    "complexity": o.complexity.value,
                    "dimensions_used": [d.value for d in o.dimensions_used],
                    "outcome": o.outcome.value,
                    "confidence": o.confidence,
                    "feedback_source": o.feedback_source,
                    "timestamp": o.timestamp.isoformat()
                }
                for o in self._outcomes[-500:]
            ]
            
            data = {
                "effectiveness": effectiveness_data,
                "outcomes": outcomes_data,
                "initial_accuracy": self._initial_accuracy,
                "current_accuracy": self._current_accuracy,
                "updated_at": datetime.now().isoformat()
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except IOError:
            pass

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

