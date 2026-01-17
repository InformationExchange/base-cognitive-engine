"""
Dimension Correlator Module

Patent: NOVEL-29 - Cross-Dimensional Pattern Correlation

This module finds patterns and contradictions across dimensions identified
by the Dimensional Expander (NOVEL-28). It provides multi-dimensional
awareness by correlating signals across different analytical dimensions.

Purpose:
- Identify correlations between dimensions (e.g., economic decline + social unrest)
- Detect contradictions (e.g., positive economic data vs negative sentiment)
- Weight dimensions by their relevance to the specific query
- Provide LLM with correlated context for better analysis

Integration:
- Receives DimensionalAnalysis from DimensionalExpander
- Produces CorrelationResult for governance decisions
- Works with existing BASE detectors for comprehensive analysis
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import os
import tempfile

# Import from dimensional_expander
from .dimensional_expander import (
    DimensionCategory,
    DimensionalAnalysis,
    DimensionScore,
    TaskType,
    ComplexityLevel
)


class CorrelationType(Enum):
    """Types of correlations between dimensions."""
    POSITIVE = "positive"           # Dimensions reinforce each other
    NEGATIVE = "negative"           # Dimensions contradict each other
    NEUTRAL = "neutral"             # No significant correlation
    CAUSAL = "causal"               # One dimension may cause the other
    TEMPORAL = "temporal"           # Time-lagged correlation


class ContradictionSeverity(Enum):
    """Severity levels for detected contradictions."""
    LOW = "low"                     # Minor inconsistency
    MEDIUM = "medium"               # Significant contradiction
    HIGH = "high"                   # Major contradiction requiring attention
    CRITICAL = "critical"           # Fundamental contradiction


@dataclass
class Correlation:
    """Represents a correlation between two dimensions."""
    dimension_a: DimensionCategory
    dimension_b: DimensionCategory
    correlation_type: CorrelationType
    strength: float                         # 0-1 strength of correlation
    direction: str                          # e.g., "positive", "inverse"
    explanation: str                        # Human-readable explanation
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class Contradiction:
    """Represents a detected contradiction between dimensions."""
    dimension_a: DimensionCategory
    dimension_b: DimensionCategory
    severity: ContradictionSeverity
    description: str
    pattern_a: str                          # Pattern from dimension A
    pattern_b: str                          # Conflicting pattern from dimension B
    resolution_hint: Optional[str] = None
    confidence: float = 0.5


@dataclass
class DimensionWeight:
    """Weight assigned to a dimension for the query."""
    dimension: DimensionCategory
    weight: float                           # 0-1 weight
    reason: str                             # Why this weight


@dataclass
class CorrelationResult:
    """Complete correlation analysis result."""
    query: str
    dimensional_analysis: DimensionalAnalysis
    correlations: List[Correlation]
    contradictions: List[Contradiction]
    dimension_weights: List[DimensionWeight]
    overall_coherence: float                # 0-1 how coherent the dimensions are
    cross_dimension_insights: List[str]
    llm_reasoning_used: bool = False
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    
    def has_major_contradictions(self) -> bool:
        """Check if there are any high/critical contradictions."""
        return any(
            c.severity in [ContradictionSeverity.HIGH, ContradictionSeverity.CRITICAL]
            for c in self.contradictions
        )
    
    def get_dominant_correlation(self) -> Optional[Correlation]:
        """Get the strongest correlation."""
        if not self.correlations:
            return None
        return max(self.correlations, key=lambda c: c.strength)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the correlation result."""
        return {
            "task_type": self.dimensional_analysis.task_type.value,
            "dimensions_analyzed": self.dimensional_analysis.total_dimensions_analyzed,
            "correlations_found": len(self.correlations),
            "contradictions_found": len(self.contradictions),
            "major_contradictions": self.has_major_contradictions(),
            "overall_coherence": self.overall_coherence,
            "insights_count": len(self.cross_dimension_insights)
        }


# Known dimension correlation patterns
DIMENSION_CORRELATION_PATTERNS: Dict[Tuple[DimensionCategory, DimensionCategory], Dict] = {
    # Economic-Social correlations
    (DimensionCategory.ECONOMIC, DimensionCategory.SOCIAL): {
        "type": CorrelationType.CAUSAL,
        "description": "Economic conditions often influence social sentiment",
        "direction": "economic → social"
    },
    # Political-Economic correlations
    (DimensionCategory.POLITICAL, DimensionCategory.ECONOMIC): {
        "type": CorrelationType.POSITIVE,
        "description": "Political stability correlates with economic performance",
        "direction": "bidirectional"
    },
    # Demographic-Social correlations
    (DimensionCategory.DEMOGRAPHIC, DimensionCategory.SOCIAL): {
        "type": CorrelationType.POSITIVE,
        "description": "Demographic shifts influence social dynamics",
        "direction": "demographic → social"
    },
    # Technical-Logical correlations
    (DimensionCategory.TECHNICAL, DimensionCategory.LOGICAL): {
        "type": CorrelationType.POSITIVE,
        "description": "Technical implementation follows logical structure",
        "direction": "bidirectional"
    },
    # Economic-Temporal correlations
    (DimensionCategory.ECONOMIC, DimensionCategory.TEMPORAL): {
        "type": CorrelationType.TEMPORAL,
        "description": "Economic patterns exhibit cyclical behavior",
        "direction": "temporal-dependent"
    }
}


class DimensionCorrelator:
    """
    Cross-dimensional pattern correlation analyzer.
    
    This module correlates patterns across dimensions identified by
    the Dimensional Expander. It finds relationships, contradictions,
    and weights dimensions by relevance.
    
    Key Functions:
    - Identify dimension correlations (positive, negative, causal)
    - Detect contradictions between dimensions
    - Weight dimensions by query relevance
    - Generate cross-dimensional insights
    
    Patent: NOVEL-29 - Cross-Dimensional Pattern Correlation
    """
    
    def __init__(
        self,
        agent_config: Optional[Any] = None,
        storage_path: Optional[str] = None
    ):
        """
        Initialize the dimension correlator.
        
        Args:
            agent_config: AgentConfigManager for LLM configuration
            storage_path: Path for learning state persistence
        """
        self.agent_config = agent_config
        self._correlation_patterns = DIMENSION_CORRELATION_PATTERNS.copy()
        
        if storage_path:
            self.storage_path = storage_path
        else:
            self.storage_path = os.path.join(
                tempfile.gettempdir(),
                "base_correlation_learning.json"
            )
        
        self._learned_correlations: Dict[str, float] = {}
        self._load_learning_state()
    
    async def correlate(
        self,
        dimensional_analysis: DimensionalAnalysis,
        context: Optional[Dict] = None
    ) -> CorrelationResult:
        """
        Perform cross-dimensional correlation analysis.
        
        Args:
            dimensional_analysis: Result from DimensionalExpander
            context: Optional context for analysis
        
        Returns:
            CorrelationResult with correlations, contradictions, and insights
        """
        # Find correlations between all dimension pairs
        correlations = await self._find_correlations(dimensional_analysis)
        
        # Detect contradictions
        contradictions = await self._find_contradictions(dimensional_analysis)
        
        # Calculate dimension weights
        weights = self._calculate_weights(dimensional_analysis, context)
        
        # Generate cross-dimensional insights
        insights = self._generate_insights(
            dimensional_analysis, correlations, contradictions
        )
        
        # Calculate overall coherence
        coherence = self._calculate_coherence(correlations, contradictions)
        
        return CorrelationResult(
            query=dimensional_analysis.query,
            dimensional_analysis=dimensional_analysis,
            correlations=correlations,
            contradictions=contradictions,
            dimension_weights=weights,
            overall_coherence=coherence,
            cross_dimension_insights=insights,
            llm_reasoning_used=False  # Pattern-based by default
        )
    
    async def _find_correlations(
        self,
        analysis: DimensionalAnalysis
    ) -> List[Correlation]:
        """
        Find correlations between dimensions in the analysis.
        """
        correlations = []
        dimensions = [d.category for d in analysis.dimensions]
        
        # Check all dimension pairs
        for i, dim_a in enumerate(dimensions):
            for dim_b in dimensions[i + 1:]:
                correlation = self._get_correlation(dim_a, dim_b, analysis)
                if correlation:
                    correlations.append(correlation)
        
        return correlations
    
    def _get_correlation(
        self,
        dim_a: DimensionCategory,
        dim_b: DimensionCategory,
        analysis: DimensionalAnalysis
    ) -> Optional[Correlation]:
        """
        Get correlation between two dimensions.
        """
        # Check known patterns
        pattern_key = (dim_a, dim_b)
        reverse_key = (dim_b, dim_a)
        
        pattern = self._correlation_patterns.get(pattern_key) or \
                  self._correlation_patterns.get(reverse_key)
        
        if pattern:
            # Calculate strength based on dimension scores
            score_a = next(
                (d.relevance for d in analysis.dimensions if d.category == dim_a),
                0.5
            )
            score_b = next(
                (d.relevance for d in analysis.dimensions if d.category == dim_b),
                0.5
            )
            strength = (score_a + score_b) / 2
            
            return Correlation(
                dimension_a=dim_a,
                dimension_b=dim_b,
                correlation_type=pattern["type"],
                strength=strength,
                direction=pattern["direction"],
                explanation=pattern["description"],
                confidence=strength * 0.9
            )
        
        # Check for learned correlations
        learned_key = f"{dim_a.value}_{dim_b.value}"
        if learned_key in self._learned_correlations:
            strength = self._learned_correlations[learned_key]
            return Correlation(
                dimension_a=dim_a,
                dimension_b=dim_b,
                correlation_type=CorrelationType.POSITIVE if strength > 0 else CorrelationType.NEGATIVE,
                strength=abs(strength),
                direction="learned",
                explanation="Correlation learned from previous analyses",
                confidence=0.6
            )
        
        return None
    
    async def _find_contradictions(
        self,
        analysis: DimensionalAnalysis
    ) -> List[Contradiction]:
        """
        Find contradictions between dimensions.
        
        Contradictions occur when:
        - Short-term pattern conflicts with long-term pattern
        - One dimension shows positive signal, another shows negative
        - Evidence from one dimension conflicts with another
        """
        contradictions = []
        
        for i, dim_score_a in enumerate(analysis.dimensions):
            for dim_score_b in analysis.dimensions[i + 1:]:
                contradiction = self._check_contradiction(dim_score_a, dim_score_b)
                if contradiction:
                    contradictions.append(contradiction)
        
        return contradictions
    
    def _check_contradiction(
        self,
        dim_a: DimensionScore,
        dim_b: DimensionScore
    ) -> Optional[Contradiction]:
        """
        Check if two dimension scores contradict each other.
        """
        # Check for pattern conflicts
        # This is a simplified check - real implementation would be more sophisticated
        
        # High relevance in both with opposite patterns could indicate contradiction
        if dim_a.relevance > 0.7 and dim_b.relevance > 0.7:
            # Check for opposing short-term patterns
            if dim_a.short_term_patterns and dim_b.short_term_patterns:
                # Look for contradictory keywords
                positive_keywords = ['positive', 'growth', 'increase', 'improve']
                negative_keywords = ['negative', 'decline', 'decrease', 'worsen']
                
                a_patterns = ' '.join(dim_a.short_term_patterns).lower()
                b_patterns = ' '.join(dim_b.short_term_patterns).lower()
                
                a_positive = any(kw in a_patterns for kw in positive_keywords)
                a_negative = any(kw in a_patterns for kw in negative_keywords)
                b_positive = any(kw in b_patterns for kw in positive_keywords)
                b_negative = any(kw in b_patterns for kw in negative_keywords)
                
                if (a_positive and b_negative) or (a_negative and b_positive):
                    return Contradiction(
                        dimension_a=dim_a.category,
                        dimension_b=dim_b.category,
                        severity=ContradictionSeverity.MEDIUM,
                        description=f"Conflicting signals between {dim_a.category.value} and {dim_b.category.value}",
                        pattern_a=a_patterns[:100],
                        pattern_b=b_patterns[:100],
                        resolution_hint="Consider temporal or contextual factors",
                        confidence=0.6
                    )
        
        return None
    
    def _calculate_weights(
        self,
        analysis: DimensionalAnalysis,
        context: Optional[Dict] = None
    ) -> List[DimensionWeight]:
        """
        Calculate importance weights for each dimension.
        
        Weights are based on:
        - Relevance score from dimensional analysis
        - Task type (some dimensions matter more for certain tasks)
        - Query complexity
        """
        weights = []
        total_relevance = sum(d.relevance for d in analysis.dimensions)
        
        for dim_score in analysis.dimensions:
            # Base weight from relevance
            if total_relevance > 0:
                base_weight = dim_score.relevance / total_relevance
            else:
                base_weight = 1.0 / len(analysis.dimensions)
            
            # Task-type adjustment
            task_type = analysis.task_type
            task_adjustment = self._get_task_adjustment(dim_score.category, task_type)
            
            final_weight = min(1.0, base_weight * task_adjustment)
            
            weights.append(DimensionWeight(
                dimension=dim_score.category,
                weight=final_weight,
                reason=f"Relevance: {dim_score.relevance:.2f}, Task adjustment: {task_adjustment:.2f}"
            ))
        
        # Normalize weights to sum to 1
        total_weight = sum(w.weight for w in weights)
        if total_weight > 0:
            for w in weights:
                w.weight = w.weight / total_weight
        
        return weights
    
    def _get_task_adjustment(
        self,
        dimension: DimensionCategory,
        task_type: TaskType
    ) -> float:
        """
        Get weight adjustment factor based on task type.
        
        Some dimensions are more important for certain task types.
        """
        adjustments = {
            TaskType.CODING_TESTING: {
                DimensionCategory.TECHNICAL: 1.5,
                DimensionCategory.LOGICAL: 1.5,
                DimensionCategory.METHODOLOGICAL: 1.2,
            },
            TaskType.POLITICAL_SOCIAL: {
                DimensionCategory.ECONOMIC: 1.3,
                DimensionCategory.SOCIAL: 1.3,
                DimensionCategory.DEMOGRAPHIC: 1.2,
                DimensionCategory.POLITICAL: 1.4,
            },
            TaskType.FINANCIAL_TRADING: {
                DimensionCategory.ECONOMIC: 1.5,
                DimensionCategory.TEMPORAL: 1.3,
                DimensionCategory.RISK: 1.4,
            }
        }
        
        task_adjustments = adjustments.get(task_type, {})
        return task_adjustments.get(dimension, 1.0)
    
    def _generate_insights(
        self,
        analysis: DimensionalAnalysis,
        correlations: List[Correlation],
        contradictions: List[Contradiction]
    ) -> List[str]:
        """
        Generate cross-dimensional insights from the analysis.
        """
        insights = []
        
        # Insight from dominant dimensions
        if analysis.dominant_dimensions:
            dominant_str = ", ".join(d.value for d in analysis.dominant_dimensions[:2])
            insights.append(
                f"Primary analytical focus: {dominant_str}"
            )
        
        # Insight from correlations
        strong_correlations = [c for c in correlations if c.strength > 0.7]
        if strong_correlations:
            for corr in strong_correlations[:2]:
                insights.append(
                    f"Strong {corr.correlation_type.value} correlation: "
                    f"{corr.dimension_a.value} ↔ {corr.dimension_b.value}"
                )
        
        # Insight from contradictions
        if contradictions:
            for contr in contradictions[:2]:
                insights.append(
                    f"Potential contradiction ({contr.severity.value}): "
                    f"{contr.dimension_a.value} vs {contr.dimension_b.value}"
                )
        
        # Complexity insight
        if analysis.complexity == ComplexityLevel.COMPLEX:
            insights.append(
                f"Complex multi-dimensional query requiring {analysis.total_dimensions_analyzed} "
                f"dimensions for comprehensive analysis"
            )
        
        return insights
    
    def _calculate_coherence(
        self,
        correlations: List[Correlation],
        contradictions: List[Contradiction]
    ) -> float:
        """
        Calculate overall coherence score.
        
        High coherence = strong positive correlations, few contradictions
        Low coherence = many contradictions, weak correlations
        """
        if not correlations and not contradictions:
            return 0.5  # Neutral
        
        # Positive contribution from correlations
        correlation_score = 0.0
        if correlations:
            positive_corr = [c for c in correlations if c.correlation_type == CorrelationType.POSITIVE]
            correlation_score = sum(c.strength for c in positive_corr) / len(correlations)
        
        # Negative contribution from contradictions
        contradiction_penalty = 0.0
        if contradictions:
            severity_weights = {
                ContradictionSeverity.LOW: 0.1,
                ContradictionSeverity.MEDIUM: 0.2,
                ContradictionSeverity.HIGH: 0.3,
                ContradictionSeverity.CRITICAL: 0.5
            }
            for c in contradictions:
                contradiction_penalty += severity_weights.get(c.severity, 0.1)
        
        coherence = max(0.0, min(1.0, 0.5 + correlation_score - contradiction_penalty))
        return coherence
    
    def learn_from_outcome(
        self,
        correlation: Correlation,
        was_accurate: bool
    ):
        """
        Learn from correlation accuracy.
        
        Updates internal correlation strengths based on outcomes.
        """
        key = f"{correlation.dimension_a.value}_{correlation.dimension_b.value}"
        
        current = self._learned_correlations.get(key, 0.0)
        adjustment = 0.1 if was_accurate else -0.1
        
        self._learned_correlations[key] = max(-1.0, min(1.0, current + adjustment))
        self._save_learning_state()
    
    def _load_learning_state(self):
        """Load learning state from disk."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self._learned_correlations = data.get("learned_correlations", {})
        except (json.JSONDecodeError, IOError):
            pass
    
    def _save_learning_state(self):
        """Save learning state to disk."""
        try:
            data = {
                "learned_correlations": self._learned_correlations,
                "updated_at": datetime.now().isoformat()
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError:
            pass

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

