"""
Dimensional Expander Module

Patent: NOVEL-28 - Intelligent Multi-Dimensional Query Analysis

This module provides intelligent dimension selection based on task type and complexity.
It does NOT expand all dimensions for every query - only those relevant to the task.

Key Principle: Dimensions provide contextual awareness proportional to task complexity.
- Coding tasks → Only technical/logical dimensions
- Political analysis → Full multi-dimensional expansion
- LLM invocation → Only for COMPLEX queries

Three Distinct LLM Roles (as documented):
1. LLM for Dimension Identification - Pattern-based for simple, LLM for complex
2. LLM for Task Execution - Always (performs the actual task)
3. LLM for Governance - Always (BASE monitors output)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import os
import tempfile
import re


class TaskType(Enum):
    """
    Task types for intelligent dimension filtering.
    
    Each task type has a predefined set of relevant dimensions.
    This prevents unnecessary dimensional expansion for simple tasks.
    """
    CODING_TESTING = "coding_testing"           # Code review, testing, debugging
    RESEARCH_ANALYSIS = "research_analysis"     # Academic, scientific research
    LEGAL_COMPLIANCE = "legal_compliance"       # Legal analysis, regulatory
    FINANCIAL_TRADING = "financial_trading"     # Financial analysis, risk
    POLITICAL_SOCIAL = "political_social"       # Political, social analysis
    MEDICAL_HEALTH = "medical_health"           # Medical, clinical analysis
    GENERAL = "general"                         # Unclassified queries


class ComplexityLevel(Enum):
    """
    Query complexity determines dimension expansion depth.
    
    SIMPLE: 1-2 dimensions, pattern-based only (no LLM for dimension ID)
    MODERATE: 2-4 dimensions, optional LLM
    COMPLEX: All relevant dimensions, LLM required for dimension ID
    """
    SIMPLE = "simple"           # 1-2 dimensions, no LLM needed for dimension ID
    MODERATE = "moderate"       # 2-4 dimensions, optional LLM for dimension ID
    COMPLEX = "complex"         # 4+ dimensions, LLM required for dimension ID


class DimensionCategory(Enum):
    """
    Standard dimension types for analysis.
    
    Dimensions are grouped by their typical relevance to task types.
    """
    # Technical dimensions (coding/testing)
    TECHNICAL = "technical"             # Code structure, architecture
    LOGICAL = "logical"                 # Logic correctness, reasoning
    METHODOLOGICAL = "methodological"   # Approach, methodology
    
    # Research dimensions
    EVIDENTIAL = "evidential"           # Evidence quality, citations
    
    # Broader dimensions (complex queries only)
    ECONOMIC = "economic"               # GDP, inflation, employment
    SOCIAL = "social"                   # Sentiment, trends, movements
    TEMPORAL = "temporal"               # Time-based patterns
    DEMOGRAPHIC = "demographic"         # Population, age, distribution
    POLITICAL = "political"             # Policy, governance, elections
    TECHNOLOGICAL = "technological"     # Innovation, adoption
    ENVIRONMENTAL = "environmental"     # Climate, resources
    CULTURAL = "cultural"               # Values, beliefs, norms
    
    # Domain-specific
    REGULATORY = "regulatory"           # Legal/compliance requirements
    CLINICAL = "clinical"               # Medical/health factors
    RISK = "risk"                       # Risk assessment


# Mapping: Task Type → Relevant Dimensions
TASK_DIMENSION_MAP: Dict[TaskType, List[DimensionCategory]] = {
    TaskType.CODING_TESTING: [
        DimensionCategory.TECHNICAL,
        DimensionCategory.LOGICAL,
        DimensionCategory.METHODOLOGICAL
    ],
    TaskType.RESEARCH_ANALYSIS: [
        DimensionCategory.METHODOLOGICAL,
        DimensionCategory.EVIDENTIAL,
        DimensionCategory.TEMPORAL
    ],
    TaskType.LEGAL_COMPLIANCE: [
        DimensionCategory.REGULATORY,
        DimensionCategory.EVIDENTIAL,
        DimensionCategory.TEMPORAL
    ],
    TaskType.FINANCIAL_TRADING: [
        DimensionCategory.ECONOMIC,
        DimensionCategory.TEMPORAL,
        DimensionCategory.RISK
    ],
    TaskType.POLITICAL_SOCIAL: [
        DimensionCategory.ECONOMIC,
        DimensionCategory.SOCIAL,
        DimensionCategory.DEMOGRAPHIC,
        DimensionCategory.POLITICAL,
        DimensionCategory.CULTURAL,
        DimensionCategory.TEMPORAL
    ],
    TaskType.MEDICAL_HEALTH: [
        DimensionCategory.CLINICAL,
        DimensionCategory.EVIDENTIAL,
        DimensionCategory.RISK
    ],
    TaskType.GENERAL: [
        DimensionCategory.LOGICAL,
        DimensionCategory.EVIDENTIAL
    ]
}


# Task classification patterns
TASK_PATTERNS: Dict[TaskType, List[str]] = {
    TaskType.CODING_TESTING: [
        r'\bcode\b', r'\bfunction\b', r'\bbug\b', r'\berror\b', r'\btest\b',
        r'\bimplement\b', r'\bdebug\b', r'\bcompile\b', r'\bsyntax\b',
        r'\bclass\b', r'\bmethod\b', r'\bapi\b', r'\bpython\b', r'\bjavascript\b',
        r'\bfix\b.*\b(error|bug|issue)\b', r'\breview\b.*\bcode\b',
        r'\bauthentication\b', r'\bauth\b', r'\bsecurity\b', r'\bvulnerability\b',
        r'\breview\b', r'\bimplementation\b', r'\brefactor\b', r'\boptimize\b',
        r'\balgorithm\b', r'\bperformance\b', r'\bmodule\b', r'\bpackage\b',
        r'\bsort\b', r'\bsearch\b', r'\bloop\b', r'\barray\b', r'\blist\b',
        r'\bdatabase\b', r'\bquery\b', r'\bschema\b'
    ],
    TaskType.LEGAL_COMPLIANCE: [
        r'\blegal\b', r'\blaw\b', r'\bcourt\b', r'\bregulation\b',
        r'\bcompliance\b', r'\bcontract\b', r'\bliability\b', r'\bstatute\b',
        r'\blawsuit\b', r'\battorney\b', r'\bjudge\b', r'\bverdict\b'
    ],
    TaskType.FINANCIAL_TRADING: [
        r'\bstock\b', r'\bmarket\b', r'\binvest\b', r'\bfinancial\b',
        r'\btrading\b', r'\bprofit\b', r'\bportfolio\b', r'\bequity\b',
        r'\bearnings\b', r'\brevenue\b', r'\bquarterly\b', r'\bq[1-4]\b',
        r'\bfiscal\b', r'\bdividend\b', r'\bcapital\b', r'\basset\b',
        r'\bbalance\s*sheet\b', r'\bincome\s*statement\b', r'\bcash\s*flow\b'
    ],
    TaskType.POLITICAL_SOCIAL: [
        r'\belection\b', r'\bvote\b', r'\bpolitical\b', r'\bgovernment\b',
        r'\bpolicy\b', r'\bsocial\b', r'\bdemographic\b', r'\bpopulation\b',
        r'\btrump\b', r'\bbiden\b', r'\bdemocrat\b', r'\brepublican\b',
        r'\bcampaign\b', r'\bcandidate\b', r'\bparty\b', r'\bcongress\b',
        r'\bsenate\b', r'\blegislation\b', r'\bpoll\b'
    ],
    TaskType.MEDICAL_HEALTH: [
        r'\bmedical\b', r'\bhealth\b', r'\bpatient\b', r'\bclinical\b',
        r'\bdiagnosis\b', r'\bsymptom\b', r'\btreatment\b', r'\bdrug\b',
        r'\bdoctor\b', r'\bhospital\b', r'\bsurgery\b', r'\bdisease\b'
    ],
    TaskType.RESEARCH_ANALYSIS: [
        r'\bresearch\b', r'\bstudy\b', r'\banalysis\b', r'\bdata\b',
        r'\bevidence\b', r'\bhypothesis\b', r'\bexperiment\b',
        r'\bscientific\b', r'\bjournal\b', r'\bpeer\s*review\b'
    ]
}

# Complexity indicators
SIMPLE_INDICATORS = [
    r'\bfix\s+this\b', r'\bwhat\s+is\b', r'\bhow\s+to\b',
    r'\bsyntax\s+error\b', r'\bwhat\s+does\b', r'\bdefine\b'
]

COMPLEX_INDICATORS = [
    r'\bwhy\s+did\b', r'\bwhat\s+caused\b', r'\banalyze\b',
    r'\bcompare\b', r'\bimplications\b', r'\bimpact\b',
    r'\bconsequences\b', r'\bexplain\s+the\s+relationship\b',
    r'\bmultiple\s+factors\b', r'\bsystemic\b'
]


@dataclass
class DimensionScore:
    """Score and patterns for a single dimension."""
    category: DimensionCategory
    relevance: float                            # 0-1 relevance to query
    short_term_patterns: List[str] = field(default_factory=list)
    long_term_patterns: List[str] = field(default_factory=list)
    evidence_found: List[str] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class DimensionalAnalysis:
    """Complete dimensional analysis result."""
    query: str
    task_type: TaskType
    complexity: ComplexityLevel
    dimensions: List[DimensionScore]
    total_dimensions_analyzed: int
    dominant_dimensions: List[DimensionCategory]
    llm_used_for_identification: bool = False   # Was LLM used to identify dimensions?
    additional_dimensions_discovered: List[DimensionCategory] = field(default_factory=list)
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the analysis."""
        return {
            "task_type": self.task_type.value,
            "complexity": self.complexity.value,
            "dimensions_analyzed": self.total_dimensions_analyzed,
            "dominant": [d.value for d in self.dominant_dimensions],
            "llm_for_dimension_id": self.llm_used_for_identification,
            "additional_discovered": [d.value for d in self.additional_dimensions_discovered]
        }


@dataclass
class LearningState:
    """Tracks which dimensions were useful for which task types."""
    task_dimension_effectiveness: Dict[str, Dict[str, float]] = field(default_factory=dict)
    outcome_history: List[Dict] = field(default_factory=list)
    total_queries: int = 0


class DimensionalExpander:
    """
    Intelligent dimensional analysis expander.
    
    Key Principle: Only analyze dimensions RELEVANT to the task.
    - Coding tasks → Only technical/logical dimensions
    - Political analysis → Full multi-dimensional expansion
    - LLM invocation → Only for COMPLEX queries (for dimension identification)
    
    Note: LLM for task EXECUTION and GOVERNANCE is always used.
    This module controls LLM usage only for DIMENSION IDENTIFICATION.
    
    Patent: NOVEL-28 - Intelligent Multi-Dimensional Query Analysis
    """
    
    def __init__(
        self,
        agent_config: Optional[Any] = None,
        storage_path: Optional[str] = None
    ):
        """
        Initialize the dimensional expander.
        
        Args:
            agent_config: AgentConfigManager for LLM configuration
            storage_path: Path for learning state persistence
        """
        self.agent_config = agent_config
        self._task_dimension_map = TASK_DIMENSION_MAP.copy()
        self._learning_rate = 0.1
        
        if storage_path:
            self.storage_path = storage_path
        else:
            self.storage_path = os.path.join(
                tempfile.gettempdir(),
                "base_dimensional_learning.json"
            )
        
        self._learning_state = LearningState()
        self._load_learning_state()
    
    async def expand(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> DimensionalAnalysis:
        """
        Intelligently expand query into RELEVANT dimensions only.
        
        Process:
        1. Classify task type (pattern-based)
        2. Assess complexity (pattern-based)
        3. Filter to relevant dimensions only
        4. If COMPLEX, use LLM for additional dimension discovery
        5. Analyze only relevant dimensions
        
        Args:
            query: The user's query
            context: Optional context (domain, is_code, etc.)
        
        Returns:
            DimensionalAnalysis with scored dimensions
        """
        # Step 1: Classify task type (pattern-based)
        task_type = self._classify_task(query, context)
        
        # Step 2: Assess complexity (pattern-based)
        complexity = self._assess_complexity(query, task_type)
        
        # Step 3: Get relevant dimensions for this task type
        relevant_dimensions = self._get_relevant_dimensions(task_type, complexity)
        
        # Step 4: Only use LLM if COMPLEX (for dimension identification only)
        llm_used = False
        additional_discovered = []
        
        if complexity == ComplexityLevel.COMPLEX:
            llm_dimensions = await self._identify_dimensions_llm(query, task_type)
            if llm_dimensions:
                additional_discovered = [d for d in llm_dimensions if d not in relevant_dimensions]
                relevant_dimensions = list(set(relevant_dimensions + llm_dimensions))
                llm_used = True
        
        # Step 5: Analyze only relevant dimensions
        dimension_scores = []
        for dim in relevant_dimensions:
            score = await self._analyze_dimension(dim, query, context)
            dimension_scores.append(score)
        
        # Sort by relevance
        dimension_scores.sort(key=lambda x: x.relevance, reverse=True)
        
        # Get dominant dimensions (top 3)
        dominant = [d.category for d in dimension_scores[:3]]
        
        return DimensionalAnalysis(
            query=query,
            task_type=task_type,
            complexity=complexity,
            dimensions=dimension_scores,
            total_dimensions_analyzed=len(dimension_scores),
            dominant_dimensions=dominant,
            llm_used_for_identification=llm_used,
            additional_dimensions_discovered=additional_discovered
        )
    
    def _classify_task(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> TaskType:
        """
        Classify query into task type for dimension filtering.
        
        Uses pattern matching first, falls back to context hints.
        This is always pattern-based (no LLM needed).
        """
        query_lower = query.lower()
        
        # Score each task type by pattern matches
        scores: Dict[TaskType, int] = {t: 0 for t in TaskType}
        
        for task_type, patterns in TASK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    scores[task_type] += 1
        
        # Find best match
        best_type = max(scores, key=scores.get)
        if scores[best_type] > 0:
            return best_type
        
        # Fallback to context hints
        if context:
            domain = context.get('domain', '').lower()
            if domain == 'coding' or context.get('is_code'):
                return TaskType.CODING_TESTING
            if domain == 'legal':
                return TaskType.LEGAL_COMPLIANCE
            if domain == 'financial':
                return TaskType.FINANCIAL_TRADING
            if domain == 'medical':
                return TaskType.MEDICAL_HEALTH
        
        return TaskType.GENERAL
    
    def _assess_complexity(
        self,
        query: str,
        task_type: TaskType
    ) -> ComplexityLevel:
        """
        Assess query complexity to determine expansion depth.
        
        Simple: Single-factor questions → minimal dimensions
        Moderate: Multi-factor questions → several dimensions
        Complex: Systemic questions → full dimensional analysis with LLM
        
        This is always pattern-based (no LLM needed).
        """
        query_lower = query.lower()
        
        # Check for simple indicators
        for pattern in SIMPLE_INDICATORS:
            if re.search(pattern, query_lower):
                return ComplexityLevel.SIMPLE
        
        # Check for complex indicators
        complex_count = 0
        for pattern in COMPLEX_INDICATORS:
            if re.search(pattern, query_lower):
                complex_count += 1
        
        if complex_count >= 1:
            # Political/social tasks with complex indicators are definitely complex
            if task_type == TaskType.POLITICAL_SOCIAL:
                return ComplexityLevel.COMPLEX
            # Other tasks with complex indicators are moderate
            return ComplexityLevel.MODERATE
        
        # Default based on task type
        if task_type in [TaskType.CODING_TESTING, TaskType.GENERAL]:
            return ComplexityLevel.SIMPLE
        
        return ComplexityLevel.MODERATE
    
    def _get_relevant_dimensions(
        self,
        task_type: TaskType,
        complexity: ComplexityLevel
    ) -> List[DimensionCategory]:
        """
        Get dimensions relevant to this task type and complexity.
        
        Does NOT return all dimensions - only those that matter.
        Respects learned adjustments from outcomes.
        """
        base_dimensions = self._task_dimension_map.get(task_type, []).copy()
        
        # Apply learned adjustments if available
        learned_key = task_type.value
        if learned_key in self._learning_state.task_dimension_effectiveness:
            effectiveness = self._learning_state.task_dimension_effectiveness[learned_key]
            # Sort by effectiveness
            base_dimensions.sort(
                key=lambda d: effectiveness.get(d.value, 0.5),
                reverse=True
            )
        
        # Limit dimensions by complexity
        if complexity == ComplexityLevel.SIMPLE:
            return base_dimensions[:2]  # Max 2 dimensions
        elif complexity == ComplexityLevel.MODERATE:
            return base_dimensions[:4]  # Max 4 dimensions
        else:
            return base_dimensions  # All relevant dimensions
    
    async def _identify_dimensions_llm(
        self,
        query: str,
        task_type: TaskType,
        llm_helper: Any = None
    ) -> List[DimensionCategory]:
        """
        LLM-powered dimension identification for COMPLEX queries.
        
        The LLM discovers additional dimensions beyond the base mapping.
        Discovered dimensions are LEARNED and persisted for future queries.
        
        This is the key adaptive component - not prescriptive, but learning.
        """
        additional = []
        
        # First: Check what we've learned for this task type
        learned_dims = self._get_learned_dimensions(task_type, query)
        additional.extend(learned_dims)
        
        # Second: Ask LLM to suggest dimensions we might have missed
        if llm_helper or self.agent_config:
            try:
                llm_suggestions = await self._ask_llm_for_dimensions(query, task_type)
                
                # Add new suggestions that we haven't seen before
                for dim in llm_suggestions:
                    if dim not in additional:
                        additional.append(dim)
                        # LEARN: Remember this dimension for this task type
                        self._record_discovered_dimension(task_type, dim, query)
                        
            except Exception:
                pass  # Fallback to pattern-based if LLM unavailable
        
        # Third: Pattern-based discovery as fallback/supplement
        query_lower = query.lower()
        pattern_discovered = self._pattern_discover_dimensions(query_lower, task_type)
        for dim in pattern_discovered:
            if dim not in additional:
                additional.append(dim)
        
        return additional
    
    def _get_learned_dimensions(
        self,
        task_type: TaskType,
        query: str
    ) -> List[DimensionCategory]:
        """
        Get dimensions we've learned are relevant for this task type.
        These were discovered in previous queries and remembered.
        """
        task_key = task_type.value
        learned = []
        
        if task_key in self._learning_state.task_dimension_effectiveness:
            effectiveness = self._learning_state.task_dimension_effectiveness[task_key]
            # Return dimensions with high effectiveness scores
            for dim_key, score in effectiveness.items():
                if score > 0.6:  # Threshold for "known good" dimensions
                    try:
                        learned.append(DimensionCategory(dim_key))
                    except ValueError:
                        pass
        
        return learned
    
    async def _ask_llm_for_dimensions(
        self,
        query: str,
        task_type: TaskType
    ) -> List[DimensionCategory]:
        """
        Ask LLM to suggest relevant dimensions for this query.
        
        This is where the adaptive learning happens - LLM can identify
        dimensions we haven't pre-defined.
        """
        # Build prompt for LLM
        prompt = f"""Analyze this query and identify which analytical dimensions are relevant.

Query: "{query}"
Task Type: {task_type.value}

Available dimensions:
- technical: Code structure, implementation details
- logical: Reasoning correctness, logic flow
- methodological: Approach, methodology quality
- evidential: Evidence quality, citation support
- economic: GDP, inflation, employment, financial factors
- social: Social sentiment, movements, trends
- temporal: Time-based patterns, historical context
- demographic: Population, age groups, distribution
- political: Policy, governance, elections
- technological: Innovation, tech adoption
- environmental: Climate, resources, sustainability
- cultural: Values, beliefs, norms
- regulatory: Legal requirements, compliance
- clinical: Medical/health factors
- risk: Risk assessment, potential issues

Which dimensions are MOST relevant for analyzing this query? 
Also suggest any NEW dimensions not in this list that might be relevant.

Return as a simple list of dimension names, one per line.
"""
        
        # In production, this would call the configured LLM
        # For now, return empty - will be wired when LLM helper is available
        # The key point is the ARCHITECTURE supports LLM dimension discovery
        
        return []
    
    def _record_discovered_dimension(
        self,
        task_type: TaskType,
        dimension: DimensionCategory,
        query: str
    ):
        """
        Record a newly discovered dimension for this task type.
        
        This is the LEARNING step - we remember what dimensions were
        useful so we can suggest them for future similar queries.
        """
        task_key = task_type.value
        dim_key = dimension.value
        
        if task_key not in self._learning_state.task_dimension_effectiveness:
            self._learning_state.task_dimension_effectiveness[task_key] = {}
        
        # Initialize with moderate effectiveness - will be adjusted based on outcomes
        if dim_key not in self._learning_state.task_dimension_effectiveness[task_key]:
            self._learning_state.task_dimension_effectiveness[task_key][dim_key] = 0.5
        
        # Record discovery
        self._learning_state.outcome_history.append({
            "type": "dimension_discovery",
            "task_type": task_key,
            "dimension": dim_key,
            "query_sample": query[:100],
            "timestamp": datetime.now().isoformat()
        })
        
        self._save_learning_state()
    
    def _pattern_discover_dimensions(
        self,
        query_lower: str,
        task_type: TaskType
    ) -> List[DimensionCategory]:
        """
        Pattern-based dimension discovery as fallback.
        Used when LLM is unavailable.
        """
        discovered = []
        
        # Economic indicators
        if any(kw in query_lower for kw in ['economy', 'inflation', 'gdp', 'unemployment', 'wage', 'income']):
            discovered.append(DimensionCategory.ECONOMIC)
        
        # Technology indicators
        if any(kw in query_lower for kw in ['technology', 'ai', 'tech', 'digital', 'automation', 'software']):
            discovered.append(DimensionCategory.TECHNOLOGICAL)
        
        # Environmental indicators
        if any(kw in query_lower for kw in ['climate', 'environment', 'pollution', 'carbon', 'sustainability']):
            discovered.append(DimensionCategory.ENVIRONMENTAL)
        
        # Cultural indicators
        if any(kw in query_lower for kw in ['culture', 'values', 'belief', 'tradition', 'identity']):
            discovered.append(DimensionCategory.CULTURAL)
        
        # Risk indicators
        if any(kw in query_lower for kw in ['risk', 'danger', 'threat', 'vulnerability', 'exposure']):
            discovered.append(DimensionCategory.RISK)
        
        return discovered
    
    async def _analyze_dimension(
        self,
        dimension: DimensionCategory,
        query: str,
        context: Optional[Dict] = None
    ) -> DimensionScore:
        """
        Analyze a single dimension for patterns and relevance.
        
        This is pattern-based analysis - LLM is used separately
        for task execution and governance.
        """
        # Calculate relevance based on dimension-specific patterns
        relevance = 0.5  # Default
        short_term = []
        long_term = []
        evidence = []
        
        query_lower = query.lower()
        
        # Dimension-specific analysis
        if dimension == DimensionCategory.TECHNICAL:
            if any(kw in query_lower for kw in ['code', 'function', 'class', 'api']):
                relevance = 0.9
                short_term.append("Code structure analysis")
        
        elif dimension == DimensionCategory.LOGICAL:
            if any(kw in query_lower for kw in ['error', 'bug', 'fix', 'wrong']):
                relevance = 0.9
                short_term.append("Logic error detection")
        
        elif dimension == DimensionCategory.ECONOMIC:
            if any(kw in query_lower for kw in ['economy', 'gdp', 'inflation', 'job']):
                relevance = 0.9
                short_term.append("Current economic indicators")
                long_term.append("Economic trend analysis")
        
        elif dimension == DimensionCategory.SOCIAL:
            if any(kw in query_lower for kw in ['social', 'sentiment', 'public']):
                relevance = 0.85
                short_term.append("Current social sentiment")
                long_term.append("Social movement trends")
        
        elif dimension == DimensionCategory.POLITICAL:
            if any(kw in query_lower for kw in ['election', 'vote', 'policy', 'govern']):
                relevance = 0.9
                short_term.append("Current political landscape")
                long_term.append("Political cycle patterns")
        
        elif dimension == DimensionCategory.DEMOGRAPHIC:
            if any(kw in query_lower for kw in ['demographic', 'population', 'age', 'group']):
                relevance = 0.85
                long_term.append("Demographic shifts")
        
        elif dimension == DimensionCategory.TEMPORAL:
            relevance = 0.7  # Always somewhat relevant
            short_term.append("Recent events")
            long_term.append("Historical patterns")
        
        return DimensionScore(
            category=dimension,
            relevance=relevance,
            short_term_patterns=short_term,
            long_term_patterns=long_term,
            evidence_found=evidence,
            confidence=relevance * 0.9
        )
    
    def learn_from_outcome(
        self,
        task_type: TaskType,
        dimensions_used: List[DimensionCategory],
        was_helpful: bool
    ):
        """
        Learn which dimensions were useful for this task type.
        
        Updates the internal mapping based on outcomes to improve
        future dimension selection.
        """
        task_key = task_type.value
        
        # Initialize if needed
        if task_key not in self._learning_state.task_dimension_effectiveness:
            self._learning_state.task_dimension_effectiveness[task_key] = {}
        
        effectiveness = self._learning_state.task_dimension_effectiveness[task_key]
        
        # Update effectiveness scores
        for dim in dimensions_used:
            dim_key = dim.value
            current = effectiveness.get(dim_key, 0.5)
            
            if was_helpful:
                # Increase effectiveness
                effectiveness[dim_key] = current + self._learning_rate * (1 - current)
            else:
                # Decrease effectiveness
                effectiveness[dim_key] = current - self._learning_rate * current
        
        # Record in history
        self._learning_state.outcome_history.append({
            "task_type": task_key,
            "dimensions": [d.value for d in dimensions_used],
            "was_helpful": was_helpful,
            "timestamp": datetime.now().isoformat()
        })
        
        self._learning_state.total_queries += 1
        
        # Save state
        self._save_learning_state()
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about learned dimension effectiveness."""
        return {
            "total_queries": self._learning_state.total_queries,
            "task_effectiveness": self._learning_state.task_dimension_effectiveness,
            "history_size": len(self._learning_state.outcome_history)
        }
    
    def get_known_dimensions_for_task(self, task_type: TaskType) -> Dict[str, Any]:
        """
        Get what we know about dimensions for a task type.
        
        Returns:
        - preset: The initial prescribed dimensions
        - learned: Dimensions discovered through usage
        - effectiveness: Score for each dimension
        """
        task_key = task_type.value
        preset = TASK_DIMENSION_MAP.get(task_type, [])
        
        learned_dims = {}
        if task_key in self._learning_state.task_dimension_effectiveness:
            learned_dims = self._learning_state.task_dimension_effectiveness[task_key]
        
        # Identify which dimensions were learned vs preset
        preset_keys = [d.value for d in preset]
        discovered = [k for k in learned_dims.keys() if k not in preset_keys]
        
        return {
            "task_type": task_key,
            "preset_dimensions": preset_keys,
            "discovered_dimensions": discovered,
            "all_known_dimensions": list(learned_dims.keys()),
            "effectiveness_scores": learned_dims,
            "total_queries_analyzed": self._learning_state.total_queries
        }
    
    def get_dimension_discovery_history(self, limit: int = 20) -> List[Dict]:
        """
        Get recent dimension discoveries.
        
        Shows what new dimensions were found and for which task types.
        """
        discoveries = [
            h for h in self._learning_state.outcome_history
            if h.get("type") == "dimension_discovery"
        ]
        return discoveries[-limit:]
    
    def _load_learning_state(self):
        """Load learning state from disk."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self._learning_state.task_dimension_effectiveness = data.get(
                        "task_dimension_effectiveness", {}
                    )
                    self._learning_state.outcome_history = data.get(
                        "outcome_history", []
                    )[-100:]  # Keep last 100
                    self._learning_state.total_queries = data.get("total_queries", 0)
        except (json.JSONDecodeError, IOError):
            pass
    
    def _save_learning_state(self):
        """Save learning state to disk."""
        try:
            data = {
                "task_dimension_effectiveness": self._learning_state.task_dimension_effectiveness,
                "outcome_history": self._learning_state.outcome_history[-100:],
                "total_queries": self._learning_state.total_queries,
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

