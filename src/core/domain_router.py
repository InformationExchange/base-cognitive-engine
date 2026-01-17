"""
BASE Domain Router (Layer 4 - Thalamus)

Routes queries to appropriate processing pipelines based on domain:
1. Domain classification
2. Pipeline selection
3. Module activation
4. Resource allocation

Patent Alignment:
- Part of orchestration layer
- Brain Layer: 4 (Thalamus - Signal Routing)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from enum import Enum


class Domain(Enum):
    """Supported domain types."""
    GENERAL = "general"
    MEDICAL = "medical"
    FINANCIAL = "financial"
    LEGAL = "legal"
    TECHNICAL = "technical"
    SCIENTIFIC = "scientific"
    EDUCATIONAL = "educational"
    CREATIVE = "creative"


@dataclass
class DomainClassification:
    """Result of domain classification."""
    primary_domain: Domain
    secondary_domains: List[Domain]
    confidence: float
    evidence: List[str]
    risk_level: str  # low, medium, high, critical


@dataclass
class RoutingDecision:
    """Decision about how to route a query."""
    domain: Domain
    pipeline: str
    modules_to_activate: List[str]
    priority: int
    resource_allocation: Dict[str, float]
    confidence: float


class DomainRouter:
    """
    Routes queries to appropriate processing pipelines.
    
    Brain Layer: 4 (Thalamus)
    
    Responsibilities:
    1. Classify query domain
    2. Select processing pipeline
    3. Activate relevant modules
    4. Allocate resources
    """
    
    # Domain indicators
    DOMAIN_INDICATORS = {
        Domain.MEDICAL: ['health', 'medical', 'doctor', 'patient', 'symptom', 'treatment',
                         'diagnosis', 'disease', 'medicine', 'hospital', 'clinical', 'drug'],
        Domain.FINANCIAL: ['money', 'finance', 'investment', 'stock', 'market', 'bank',
                          'credit', 'loan', 'tax', 'profit', 'revenue', 'budget'],
        Domain.LEGAL: ['law', 'legal', 'court', 'attorney', 'lawsuit', 'contract',
                       'regulation', 'compliance', 'rights', 'liability'],
        Domain.TECHNICAL: ['code', 'software', 'programming', 'api', 'database', 'server',
                          'algorithm', 'system', 'debug', 'deploy'],
        Domain.SCIENTIFIC: ['research', 'experiment', 'hypothesis', 'data', 'analysis',
                           'study', 'theory', 'evidence', 'scientific'],
        Domain.EDUCATIONAL: ['learn', 'teach', 'student', 'course', 'education', 'school',
                            'curriculum', 'lesson', 'training'],
        Domain.CREATIVE: ['write', 'story', 'creative', 'art', 'design', 'imagine',
                         'fiction', 'poetry', 'narrative'],
    }
    
    # Domain risk levels
    DOMAIN_RISK_LEVELS = {
        Domain.MEDICAL: 'critical',
        Domain.FINANCIAL: 'high',
        Domain.LEGAL: 'high',
        Domain.TECHNICAL: 'medium',
        Domain.SCIENTIFIC: 'medium',
        Domain.EDUCATIONAL: 'low',
        Domain.CREATIVE: 'low',
        Domain.GENERAL: 'low',
    }
    
    # Module activation by domain
    DOMAIN_MODULES = {
        Domain.MEDICAL: ['factual_detector', 'grounding_detector', 'conservative_certificates'],
        Domain.FINANCIAL: ['factual_detector', 'grounding_detector', 'compliance_reporting'],
        Domain.LEGAL: ['neurosymbolic', 'factual_detector', 'compliance_reporting'],
        Domain.TECHNICAL: ['neurosymbolic', 'grounding_detector'],
        Domain.SCIENTIFIC: ['factual_detector', 'neurosymbolic', 'grounding_detector'],
        Domain.EDUCATIONAL: ['zpd_manager', 'theory_of_mind'],
        Domain.CREATIVE: ['theory_of_mind', 'creative_reasoning'],
        Domain.GENERAL: ['grounding_detector'],
    }
    
    def __init__(self):
        """Initialize the domain router."""
        # Learning state
        self._outcomes: List[Dict] = []
        self._feedback: List[Dict] = []
        self._domain_adjustments: Dict[str, float] = {}
        self._total_routings: int = 0
        self._routing_accuracy: List[bool] = []
    
    def classify(self, query: str, context: Dict[str, Any] = None) -> DomainClassification:
        """
        Classify the domain of a query.
        
        Args:
            query: User query
            context: Optional context
            
        Returns:
            DomainClassification result
        """
        self._total_routings += 1
        query_lower = query.lower()
        context = context or {}
        
        # Score each domain
        domain_scores: Dict[Domain, float] = {}
        domain_evidence: Dict[Domain, List[str]] = {}
        
        for domain, indicators in self.DOMAIN_INDICATORS.items():
            matches = [ind for ind in indicators if ind in query_lower]
            score = min(1.0, len(matches) * 0.25)
            domain_scores[domain] = score
            domain_evidence[domain] = matches
        
        # Sort by score
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_domains[0][1] > 0:
            primary = sorted_domains[0][0]
            confidence = sorted_domains[0][1]
            evidence = domain_evidence[primary]
        else:
            primary = Domain.GENERAL
            confidence = 0.5
            evidence = []
        
        # Get secondary domains
        secondary = [d for d, s in sorted_domains[1:4] if s > 0.25]
        
        # Determine risk level
        risk_level = self.DOMAIN_RISK_LEVELS.get(primary, 'low')
        
        return DomainClassification(
            primary_domain=primary,
            secondary_domains=secondary,
            confidence=confidence,
            evidence=evidence,
            risk_level=risk_level
        )
    
    def route(self, query: str, context: Dict[str, Any] = None) -> RoutingDecision:
        """
        Route a query to the appropriate pipeline.
        
        Args:
            query: User query
            context: Optional context
            
        Returns:
            RoutingDecision with routing information
        """
        classification = self.classify(query, context)
        
        # Select pipeline based on domain
        pipeline = f"{classification.primary_domain.value}_pipeline"
        
        # Get modules to activate
        modules = self.DOMAIN_MODULES.get(classification.primary_domain, ['grounding_detector'])
        
        # Add secondary domain modules
        for secondary in classification.secondary_domains:
            secondary_modules = self.DOMAIN_MODULES.get(secondary, [])
            for mod in secondary_modules:
                if mod not in modules:
                    modules.append(mod)
        
        # Priority based on risk
        priority_map = {'critical': 10, 'high': 7, 'medium': 5, 'low': 3}
        priority = priority_map.get(classification.risk_level, 3)
        
        # Resource allocation (simplified)
        resource_allocation = {
            'cpu': 0.5 + (priority / 20),
            'memory': 0.3 + (priority / 30),
            'timeout': 10 + priority * 2
        }
        
        return RoutingDecision(
            domain=classification.primary_domain,
            pipeline=pipeline,
            modules_to_activate=modules,
            priority=priority,
            resource_allocation=resource_allocation,
            confidence=classification.confidence
        )
    
    # ===== Learning Interface Methods =====
    
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        """Record routing outcome for learning."""
        self._outcomes.append(outcome)
        if 'routing_correct' in outcome:
            self._routing_accuracy.append(outcome['routing_correct'])
    
    def record_feedback(self, feedback: Dict) -> None:
        """Record feedback on routing decisions."""
        self._feedback.append(feedback)
        domain = feedback.get('domain', 'general')
        if feedback.get('wrong_domain', False):
            self._domain_adjustments[domain] = self._domain_adjustments.get(domain, 0.0) - 0.05
    
    def adapt_thresholds(self, domain: str = 'general', performance_data: Dict = None) -> None:
        """Adapt routing thresholds."""
        pass
    
    def get_domain_adjustment(self, domain: str) -> float:
        """Get domain adjustment."""
        return self._domain_adjustments.get(domain, 0.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics."""
        routing_acc = sum(self._routing_accuracy) / max(len(self._routing_accuracy), 1)
        
        return {
            'total_routings': self._total_routings,
            'routing_accuracy': routing_acc,
            'domain_adjustments': dict(self._domain_adjustments),
            'outcomes_recorded': len(self._outcomes),
            'feedback_recorded': len(self._feedback)
        }


    # =========================================================================
    # Learning Interface Completion
    # =========================================================================
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback to adjust behavior."""
        if not hasattr(self, '_learning_params'):
            self._learning_params = {}
        was_correct = feedback.get('was_correct', True)
        if not was_correct:
            adjustment = feedback.get('adjustment', 0.05)
            for key in self._learning_params:
                self._learning_params[key] *= (1 - adjustment)
    
    def get_statistics(self) -> Dict:
        """Return learning statistics."""
        return {
            'learning_params': getattr(self, '_learning_params', {}),
            'history_size': len(getattr(self, '_outcomes', []))
        }
    
    def serialize_state(self) -> Dict:
        """Serialize state for persistence."""
        return {
            'outcomes': getattr(self, '_outcomes', [])[-100:],
            'learning_params': getattr(self, '_learning_params', {}),
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore state from serialized data."""
        self._outcomes = state.get('outcomes', [])
        self._learning_params = state.get('learning_params', {})

if __name__ == "__main__":
    router = DomainRouter()
    
    test_queries = [
        "What are the symptoms of diabetes and treatment options?",
        "How do I invest in stocks and manage my portfolio?",
        "Write a creative story about a robot learning to love.",
        "Debug this Python function that's throwing an error.",
    ]
    
    print("=" * 60)
    print("DOMAIN ROUTER TEST")
    print("=" * 60)
    
    for query in test_queries:
        result = router.route(query)
        print(f"\nQuery: {query[:50]}...")
        print(f"  Domain: {result.domain.value}")
        print(f"  Priority: {result.priority}")
        print(f"  Modules: {result.modules_to_activate[:3]}")
    
    print(f"\nLearning stats: {router.get_learning_statistics()}")

