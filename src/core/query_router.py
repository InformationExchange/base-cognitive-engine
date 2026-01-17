"""
BASE Query Router - Upfront Assessment & Module Selection
Determines which governance modules to activate for each query.

This is the FIRST component that runs, analyzing the query before
any expensive processing happens.

Flow:
1. Receive query + documents + context
2. Quick analysis (< 10ms)
3. Return routing plan (which modules to run)
4. Engine executes only selected modules
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from enum import Enum


class QueryType(str, Enum):
    """Classification of query types."""
    FACTUAL = "factual"           # "What is X?", "When did Y?"
    PROCEDURAL = "procedural"     # "How do I X?", "Steps to Y"
    ANALYTICAL = "analytical"     # "Why does X?", "Compare X and Y"
    CREATIVE = "creative"         # "Write a story", "Generate ideas"
    CONVERSATIONAL = "conversational"  # "Hello", "Thanks"
    CODE = "code"                 # Code generation/review
    RESEARCH = "research"         # Deep research queries
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    """Query risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ModulePriority(str, Enum):
    """Module execution priority."""
    REQUIRED = "required"         # Must run
    RECOMMENDED = "recommended"   # Should run if time allows
    OPTIONAL = "optional"         # Nice to have
    SKIP = "skip"                 # Don't run


@dataclass
class QueryCharacteristics:
    """Characteristics extracted from query analysis."""
    query_type: QueryType
    domain: str
    risk_level: RiskLevel
    
    # Content indicators
    has_entities: bool = False
    has_factual_claims: bool = False
    has_documents: bool = False
    has_code: bool = False
    has_numbers: bool = False
    has_dates: bool = False
    has_comparisons: bool = False
    
    # Complexity indicators
    query_length: int = 0
    estimated_complexity: str = "low"  # low, medium, high
    
    # Special flags
    requires_human_review: bool = False
    is_sensitive_domain: bool = False
    
    # Extracted elements (for later use)
    detected_entities: List[str] = field(default_factory=list)
    detected_claims: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


@dataclass
class ModuleSelection:
    """Selection status for a single module."""
    enabled: bool
    priority: ModulePriority
    reason: str
    estimated_cost_ms: float = 0.0


@dataclass
class RoutingPlan:
    """Complete routing plan for query processing."""
    query_id: str
    characteristics: QueryCharacteristics
    
    # Module selections
    proactive_prevention: ModuleSelection
    rag_quality: ModuleSelection
    knowledge_graph: ModuleSelection
    fact_checking: ModuleSelection
    consistency_checker: ModuleSelection
    
    # Core detectors (always run, but can be tuned)
    grounding_detector: ModuleSelection
    behavioral_detector: ModuleSelection
    temporal_detector: ModuleSelection
    factual_detector: ModuleSelection
    
    # R&D Modules (optional advanced cognition)
    theory_of_mind: ModuleSelection = None
    neurosymbolic: ModuleSelection = None
    world_models: ModuleSelection = None
    creative_reasoning: ModuleSelection = None
    
    # Execution guidance
    parallel_execution: bool = True
    early_exit_enabled: bool = True
    max_processing_time_ms: float = 500.0
    
    # Cost estimates
    estimated_total_cost_ms: float = 0.0
    estimated_modules_count: int = 0
    
    def get_enabled_modules(self) -> List[str]:
        """Get list of enabled module names."""
        modules = []
        all_modules = ['proactive_prevention', 'rag_quality', 'knowledge_graph',
                       'fact_checking', 'consistency_checker', 'grounding_detector',
                       'behavioral_detector', 'temporal_detector', 'factual_detector',
                       'theory_of_mind', 'neurosymbolic', 'world_models', 'creative_reasoning']
        for name in all_modules:
            selection = getattr(self, name, None)
            if selection and selection.enabled:
                modules.append(name)
        return modules
    
    def get_enabled_rd_modules(self) -> List[str]:
        """Get list of enabled R&D modules."""
        rd_modules = ['theory_of_mind', 'neurosymbolic', 'world_models', 'creative_reasoning']
        return [name for name in rd_modules 
                if getattr(self, name, None) and getattr(self, name).enabled]

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


class QueryRouter:
    """
    Upfront query analysis and module routing.
    
    This component runs FIRST and determines the execution plan
    for all downstream modules.
    
    KEY: Uses ModuleRegistry for REAL-TIME awareness of available modules.
    The router never "forgets" - it always queries the registry.
    """
    
    # Import registry lazily to avoid circular imports
    _registry = None
    
    @property
    def registry(self):
        """Get the module registry (lazy load)."""
        if self._registry is None:
            from core.module_registry import get_registry
            self._registry = get_registry()
        return self._registry
    
    # Domain keywords for classification
    DOMAIN_KEYWORDS = {
        'medical': ['health', 'disease', 'treatment', 'symptom', 'doctor', 'patient', 
                   'diagnosis', 'medication', 'surgery', 'clinical', 'medical'],
        'financial': ['money', 'investment', 'stock', 'bank', 'finance', 'loan',
                     'credit', 'interest', 'portfolio', 'trading', 'crypto'],
        'legal': ['law', 'legal', 'court', 'attorney', 'lawsuit', 'contract',
                 'liability', 'regulation', 'compliance', 'rights'],
        'technical': ['code', 'programming', 'software', 'api', 'database', 'server',
                     'algorithm', 'function', 'debug', 'deploy'],
    }
    
    # High-risk indicators
    HIGH_RISK_PATTERNS = [
        r'\b(suicide|self-harm|kill|murder|weapon)\b',
        r'\b(illegal|fraud|scam|hack|exploit)\b',
        r'\b(medication|dosage|prescription)\b.*\b(take|use|inject)\b',
    ]
    
    # Query type patterns
    QUERY_TYPE_PATTERNS = {
        QueryType.FACTUAL: [
            r'^(what|who|when|where|which)\s+(is|are|was|were)\b',
            r'\b(define|explain|describe)\b',
        ],
        QueryType.PROCEDURAL: [
            r'^how\s+(do|can|to|should)\b',
            r'\b(steps|guide|tutorial|instructions)\b',
        ],
        QueryType.ANALYTICAL: [
            r'^why\s+(is|are|do|does|did)\b',
            r'\b(compare|contrast|analyze|evaluate)\b',
            r'\b(difference|similarity|pros|cons)\b',
        ],
        QueryType.CREATIVE: [
            r'\b(write|create|generate|compose|design)\b.*\b(story|poem|idea|content)\b',
            r'^(imagine|pretend|suppose)\b',
        ],
        QueryType.CONVERSATIONAL: [
            r'^(hi|hello|hey|thanks|thank you|bye|goodbye)\b',
            r'^(how are you|what\'s up)\b',
        ],
        QueryType.CODE: [
            r'\b(code|function|class|method|implement|debug|fix)\b',
            r'\b(python|javascript|java|typescript|rust|go)\b',
            r'```',
        ],
        QueryType.RESEARCH: [
            r'\b(research|study|investigate|explore|deep dive)\b',
            r'\b(comprehensive|thorough|detailed|in-depth)\b',
        ],
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Cost estimates (in ms)
        self.module_costs = {
            'proactive_prevention': 20,
            'rag_quality': 30,
            'knowledge_graph': 50,
            'fact_checking': 100,
            'consistency_checker': 5,
            'grounding_detector': 40,
            'behavioral_detector': 20,
            'temporal_detector': 10,
            'factual_detector': 50,
            # R&D Modules
            'theory_of_mind': 5,
            'neurosymbolic': 5,
            'world_models': 10,
            'creative_reasoning': 5,
        }
        
        # R&D Module selection patterns
        self.RD_MODULE_PATTERNS = {
            'theory_of_mind': {
                'patterns': [
                    r'\b(why did|why would|what was .* thinking|perspective|intent|motive)\b',
                    r'\b(feel|believe|want|expect|assume)\b.*\b(they|he|she)\b',
                    r'\b(persuade|manipulate|deceive|mislead)\b',
                    r'\b(social|emotional|relationship|conflict)\b',
                ],
                'domains': ['social', 'psychological', 'communication', 'hr'],
                'query_types': [QueryType.ANALYTICAL],
            },
            'neurosymbolic': {
                'patterns': [
                    r'\b(prove|verify|valid|logical|if .* then|implies|therefore)\b',
                    r'\b(constraint|satisfy|requirement|condition|rule)\b',
                    r'\b(always|never|all|none|every|must)\b',
                    r'\b(contradiction|inconsistent)\b',
                ],
                'domains': ['legal', 'technical', 'mathematical', 'compliance'],
                'query_types': [QueryType.ANALYTICAL],
            },
            'world_models': {
                'patterns': [
                    r'\b(what if|what would happen|predict|forecast|future|scenario)\b',
                    r'\b(cause|effect|lead to|result in|impact|consequence)\b',
                    r'\b(simulate|model|plan|strategy)\b',
                ],
                'domains': ['financial', 'strategic', 'scientific', 'policy'],
                'query_types': [QueryType.ANALYTICAL, QueryType.RESEARCH],
            },
            'creative_reasoning': {
                'patterns': [
                    r'\b(creative|innovative|novel|unique|original|new idea)\b',
                    r'\b(brainstorm|generate|imagine|invent|design)\b',
                    r'\b(like|similar to|analogy|metaphor)\b',
                    r'\b(alternative|different way|outside the box)\b',
                ],
                'domains': ['creative', 'marketing', 'design', 'innovation'],
                'query_types': [QueryType.CREATIVE],
            },
        }
    
    def analyze(self, 
                query: str, 
                response: str = None,
                documents: List[Dict] = None,
                context: Dict[str, Any] = None) -> RoutingPlan:
        """
        Analyze query and create routing plan.
        
        This is the main entry point - runs in < 10ms.
        
        Args:
            query: User query
            response: Pre-generated response (if available)
            documents: Source documents (if available)
            context: Additional context (domain override, etc.)
        
        Returns:
            RoutingPlan with module selections
        """
        documents = documents or []
        context = context or {}
        
        # Generate query ID
        query_id = hashlib.md5(f"{query}{len(documents)}".encode()).hexdigest()[:8]
        
        # Step 1: Extract query characteristics
        characteristics = self._analyze_characteristics(query, response, documents, context)
        
        # Step 2: Determine module selections
        proactive = self._select_proactive_prevention(characteristics)
        rag = self._select_rag_quality(characteristics, documents)
        kg = self._select_knowledge_graph(characteristics, response)
        fc = self._select_fact_checking(characteristics, response)
        consistency = self._select_consistency_checker(characteristics)
        
        # Core detectors (always enabled, but priority varies)
        grounding = self._select_grounding_detector(characteristics, documents)
        behavioral = self._select_behavioral_detector(characteristics)
        temporal = self._select_temporal_detector(characteristics)
        factual = self._select_factual_detector(characteristics, documents)
        
        # Step 3: Select R&D modules (advanced cognition)
        tom = self._select_theory_of_mind(characteristics, query, response)
        neuro = self._select_neurosymbolic(characteristics, query, response)
        world = self._select_world_models(characteristics, query, response)
        creative = self._select_creative_reasoning(characteristics, query, response)
        
        # Build routing plan
        plan = RoutingPlan(
            query_id=query_id,
            characteristics=characteristics,
            proactive_prevention=proactive,
            rag_quality=rag,
            knowledge_graph=kg,
            fact_checking=fc,
            consistency_checker=consistency,
            grounding_detector=grounding,
            behavioral_detector=behavioral,
            temporal_detector=temporal,
            factual_detector=factual,
            theory_of_mind=tom,
            neurosymbolic=neuro,
            world_models=world,
            creative_reasoning=creative,
            parallel_execution=True,
            early_exit_enabled=characteristics.risk_level == RiskLevel.CRITICAL,
        )
        
        # Calculate cost estimates
        plan.estimated_total_cost_ms = self._estimate_total_cost(plan)
        plan.estimated_modules_count = len(plan.get_enabled_modules())
        
        return plan
    
    def _analyze_characteristics(self, 
                                  query: str, 
                                  response: str,
                                  documents: List[Dict],
                                  context: Dict) -> QueryCharacteristics:
        """Extract all characteristics from query."""
        
        query_lower = query.lower()
        response_lower = (response or "").lower()
        combined = f"{query} {response or ''}"
        
        # Determine query type
        query_type = self._detect_query_type(query_lower)
        
        # Determine domain
        domain = context.get('domain') or self._detect_domain(query_lower)
        
        # Determine risk level
        risk_level = self._assess_risk(query, domain)
        
        # Content indicators
        has_entities = self._detect_entities(combined)
        has_factual_claims = self._detect_factual_claims(combined)
        has_code = self._detect_code(combined)
        has_numbers = bool(re.search(r'\b\d+(?:\.\d+)?(?:%|percent|million|billion)?\b', combined))
        has_dates = bool(re.search(r'\b(?:19|20)\d{2}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', combined.lower()))
        has_comparisons = any(word in combined.lower() for word in ['compare', 'versus', 'vs', 'better', 'worse', 'difference'])
        
        # Complexity
        query_length = len(query.split())
        if query_length < 10:
            complexity = "low"
        elif query_length < 30:
            complexity = "medium"
        else:
            complexity = "high"
        
        # Special flags
        is_sensitive = domain in ['medical', 'financial', 'legal']
        requires_human = risk_level == RiskLevel.CRITICAL
        
        # Extract entities and claims for later use
        detected_entities = self._extract_entity_candidates(combined) if has_entities else []
        detected_claims = self._extract_claim_candidates(combined) if has_factual_claims else []
        keywords = self._extract_keywords(query)
        
        return QueryCharacteristics(
            query_type=query_type,
            domain=domain,
            risk_level=risk_level,
            has_entities=has_entities,
            has_factual_claims=has_factual_claims,
            has_documents=len(documents) > 0,
            has_code=has_code,
            has_numbers=has_numbers,
            has_dates=has_dates,
            has_comparisons=has_comparisons,
            query_length=query_length,
            estimated_complexity=complexity,
            requires_human_review=requires_human,
            is_sensitive_domain=is_sensitive,
            detected_entities=detected_entities,
            detected_claims=detected_claims,
            keywords=keywords,
        )
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Classify query type."""
        for qtype, patterns in self.QUERY_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return qtype
        return QueryType.UNKNOWN
    
    def _detect_domain(self, query: str) -> str:
        """Detect domain from query content."""
        scores = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query)
            if score > 0:
                scores[domain] = score
        
        if scores:
            return max(scores, key=scores.get)
        return "general"
    
    def _assess_risk(self, query: str, domain: str) -> RiskLevel:
        """Assess query risk level."""
        # Check high-risk patterns
        for pattern in self.HIGH_RISK_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return RiskLevel.CRITICAL
        
        # Domain-based risk
        if domain in ['medical', 'legal']:
            return RiskLevel.HIGH
        elif domain == 'financial':
            return RiskLevel.MEDIUM
        
        return RiskLevel.LOW
    
    def _detect_entities(self, text: str) -> bool:
        """Quick check for entity presence."""
        indicators = [
            # Capitalized words (potential names)
            len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)) >= 2,
            # Organization patterns
            any(word in text for word in ['Inc.', 'Corp.', 'LLC', 'Ltd.', 'Company']),
            # Quoted terms
            text.count('"') >= 2,
        ]
        return sum(indicators) >= 1
    
    def _detect_factual_claims(self, text: str) -> bool:
        """Quick check for factual claims."""
        indicators = [
            # Numbers
            bool(re.search(r'\b\d+(?:\.\d+)?(?:%|percent|million|billion)?\b', text)),
            # Dates
            bool(re.search(r'\b(?:19|20)\d{2}\b', text)),
            # Definitive statements
            any(phrase in text.lower() for phrase in [
                'is the', 'was the', 'according to', 'research shows', 'officially'
            ]),
        ]
        return sum(indicators) >= 1
    
    def _detect_code(self, text: str) -> bool:
        """Check for code presence."""
        indicators = [
            '```' in text,
            'def ' in text,
            'function ' in text,
            'class ' in text,
            bool(re.search(r'\b(import|from|require|include)\b', text)),
        ]
        return any(indicators)
    
    def _extract_entity_candidates(self, text: str) -> List[str]:
        """Extract potential entity names."""
        # Simple extraction - capitalized phrases
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        # Deduplicate and limit
        return list(set(entities))[:10]
    
    def _extract_claim_candidates(self, text: str) -> List[str]:
        """Extract potential factual claims."""
        # Simple extraction - sentences with numbers or definitive language
        sentences = re.split(r'[.!?]', text)
        claims = []
        for sent in sentences:
            if re.search(r'\b\d+\b', sent) or any(w in sent.lower() for w in ['is the', 'was the']):
                claims.append(sent.strip())
        return claims[:5]
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract key terms from query."""
        # Remove common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where', 'which', 'who', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'be', 'this', 'that', 'it', 'and', 'or', 'but', 'if', 'do', 'does', 'did', 'can', 'could', 'would', 'should', 'will', 'i', 'you', 'me', 'my'}
        words = re.findall(r'\b\w+\b', query.lower())
        return [w for w in words if w not in stopwords and len(w) > 2][:10]
    
    # =============================================
    # MODULE SELECTION METHODS
    # =============================================
    
    def _select_proactive_prevention(self, chars: QueryCharacteristics) -> ModuleSelection:
        """Always run proactive prevention."""
        return ModuleSelection(
            enabled=True,
            priority=ModulePriority.REQUIRED,
            reason="Always runs first to assess risk",
            estimated_cost_ms=self.module_costs['proactive_prevention']
        )
    
    def _select_rag_quality(self, chars: QueryCharacteristics, documents: List) -> ModuleSelection:
        """Run RAG quality if documents provided."""
        if len(documents) > 0:
            return ModuleSelection(
                enabled=True,
                priority=ModulePriority.REQUIRED if chars.is_sensitive_domain else ModulePriority.RECOMMENDED,
                reason=f"Documents provided ({len(documents)} docs)",
                estimated_cost_ms=self.module_costs['rag_quality']
            )
        return ModuleSelection(
            enabled=False,
            priority=ModulePriority.SKIP,
            reason="No documents provided",
            estimated_cost_ms=0
        )
    
    def _select_knowledge_graph(self, chars: QueryCharacteristics, response: str) -> ModuleSelection:
        """Run KG if entities detected."""
        if chars.has_entities:
            priority = ModulePriority.REQUIRED if chars.is_sensitive_domain else ModulePriority.RECOMMENDED
            return ModuleSelection(
                enabled=True,
                priority=priority,
                reason=f"Entities detected: {chars.detected_entities[:3]}",
                estimated_cost_ms=self.module_costs['knowledge_graph']
            )
        return ModuleSelection(
            enabled=False,
            priority=ModulePriority.SKIP,
            reason="No entities detected",
            estimated_cost_ms=0
        )
    
    def _select_fact_checking(self, chars: QueryCharacteristics, response: str) -> ModuleSelection:
        """Run fact-checking if claims detected."""
        # Always run for sensitive domains
        if chars.is_sensitive_domain:
            return ModuleSelection(
                enabled=True,
                priority=ModulePriority.REQUIRED,
                reason=f"Sensitive domain ({chars.domain}) - always verify facts",
                estimated_cost_ms=self.module_costs['fact_checking']
            )
        
        if chars.has_factual_claims:
            return ModuleSelection(
                enabled=True,
                priority=ModulePriority.RECOMMENDED,
                reason=f"Factual claims detected ({len(chars.detected_claims)} claims)",
                estimated_cost_ms=self.module_costs['fact_checking']
            )
        
        if chars.query_type == QueryType.FACTUAL:
            return ModuleSelection(
                enabled=True,
                priority=ModulePriority.RECOMMENDED,
                reason="Query type is factual",
                estimated_cost_ms=self.module_costs['fact_checking']
            )
        
        return ModuleSelection(
            enabled=False,
            priority=ModulePriority.SKIP,
            reason="No factual claims detected",
            estimated_cost_ms=0
        )
    
    def _select_consistency_checker(self, chars: QueryCharacteristics) -> ModuleSelection:
        """Always run consistency checker."""
        return ModuleSelection(
            enabled=True,
            priority=ModulePriority.REQUIRED,
            reason="Always validates signal consistency",
            estimated_cost_ms=self.module_costs['consistency_checker']
        )
    
    def _select_grounding_detector(self, chars: QueryCharacteristics, documents: List) -> ModuleSelection:
        """Configure grounding detector."""
        if len(documents) > 0:
            return ModuleSelection(
                enabled=True,
                priority=ModulePriority.REQUIRED,
                reason="Documents available for grounding",
                estimated_cost_ms=self.module_costs['grounding_detector']
            )
        return ModuleSelection(
            enabled=True,
            priority=ModulePriority.OPTIONAL,
            reason="No documents - limited grounding",
            estimated_cost_ms=self.module_costs['grounding_detector'] * 0.5
        )
    
    def _select_behavioral_detector(self, chars: QueryCharacteristics) -> ModuleSelection:
        """Configure behavioral detector."""
        priority = ModulePriority.REQUIRED if chars.is_sensitive_domain else ModulePriority.RECOMMENDED
        return ModuleSelection(
            enabled=True,
            priority=priority,
            reason="Checks for bias patterns",
            estimated_cost_ms=self.module_costs['behavioral_detector']
        )
    
    def _select_temporal_detector(self, chars: QueryCharacteristics) -> ModuleSelection:
        """Configure temporal detector."""
        return ModuleSelection(
            enabled=True,
            priority=ModulePriority.RECOMMENDED,
            reason="Tracks response patterns over time",
            estimated_cost_ms=self.module_costs['temporal_detector']
        )
    
    def _select_factual_detector(self, chars: QueryCharacteristics, documents: List) -> ModuleSelection:
        """Configure factual detector."""
        if chars.query_type == QueryType.FACTUAL or chars.has_factual_claims:
            return ModuleSelection(
                enabled=True,
                priority=ModulePriority.REQUIRED,
                reason="Factual query - high priority",
                estimated_cost_ms=self.module_costs['factual_detector']
            )
        return ModuleSelection(
            enabled=True,
            priority=ModulePriority.RECOMMENDED,
            reason="Standard factual checks",
            estimated_cost_ms=self.module_costs['factual_detector']
        )
    
    # =============================================
    # R&D MODULE SELECTION METHODS
    # =============================================
    
    def _select_theory_of_mind(self, chars: QueryCharacteristics, query: str, response: str) -> ModuleSelection:
        """Select Theory of Mind for social/psychological queries."""
        config = self.RD_MODULE_PATTERNS['theory_of_mind']
        query_lower = query.lower()
        
        # Check patterns
        for pattern in config['patterns']:
            if re.search(pattern, query_lower):
                return ModuleSelection(
                    enabled=True,
                    priority=ModulePriority.RECOMMENDED,
                    reason=f"Query involves mental states/perspectives",
                    estimated_cost_ms=self.module_costs['theory_of_mind']
                )
        
        # Check domain
        if chars.domain in config['domains']:
            return ModuleSelection(
                enabled=True,
                priority=ModulePriority.OPTIONAL,
                reason=f"Domain ({chars.domain}) may benefit from ToM analysis",
                estimated_cost_ms=self.module_costs['theory_of_mind']
            )
        
        return ModuleSelection(
            enabled=False,
            priority=ModulePriority.SKIP,
            reason="No ToM triggers detected",
            estimated_cost_ms=0
        )
    
    def _select_neurosymbolic(self, chars: QueryCharacteristics, query: str, response: str) -> ModuleSelection:
        """Select Neuro-Symbolic for logical/verification queries."""
        config = self.RD_MODULE_PATTERNS['neurosymbolic']
        combined = f"{query} {response or ''}".lower()
        
        # Check patterns
        for pattern in config['patterns']:
            if re.search(pattern, combined):
                return ModuleSelection(
                    enabled=True,
                    priority=ModulePriority.RECOMMENDED,
                    reason="Query involves logical reasoning/verification",
                    estimated_cost_ms=self.module_costs['neurosymbolic']
                )
        
        # Check domain (always for legal/compliance)
        if chars.domain in config['domains']:
            priority = ModulePriority.REQUIRED if chars.domain == 'legal' else ModulePriority.RECOMMENDED
            return ModuleSelection(
                enabled=True,
                priority=priority,
                reason=f"Domain ({chars.domain}) requires logical verification",
                estimated_cost_ms=self.module_costs['neurosymbolic']
            )
        
        return ModuleSelection(
            enabled=False,
            priority=ModulePriority.SKIP,
            reason="No logical verification triggers detected",
            estimated_cost_ms=0
        )
    
    def _select_world_models(self, chars: QueryCharacteristics, query: str, response: str) -> ModuleSelection:
        """Select World Models for prediction/causal queries."""
        config = self.RD_MODULE_PATTERNS['world_models']
        query_lower = query.lower()
        
        # Check patterns
        for pattern in config['patterns']:
            if re.search(pattern, query_lower):
                return ModuleSelection(
                    enabled=True,
                    priority=ModulePriority.RECOMMENDED,
                    reason="Query involves prediction/causal reasoning",
                    estimated_cost_ms=self.module_costs['world_models']
                )
        
        # Check domain
        if chars.domain in config['domains']:
            return ModuleSelection(
                enabled=True,
                priority=ModulePriority.RECOMMENDED,
                reason=f"Domain ({chars.domain}) benefits from causal modeling",
                estimated_cost_ms=self.module_costs['world_models']
            )
        
        return ModuleSelection(
            enabled=False,
            priority=ModulePriority.SKIP,
            reason="No causal/prediction triggers detected",
            estimated_cost_ms=0
        )
    
    def _select_creative_reasoning(self, chars: QueryCharacteristics, query: str, response: str) -> ModuleSelection:
        """Select Creative Reasoning for generative/innovative queries."""
        config = self.RD_MODULE_PATTERNS['creative_reasoning']
        combined = f"{query} {response or ''}".lower()
        
        # Check patterns
        for pattern in config['patterns']:
            if re.search(pattern, combined):
                return ModuleSelection(
                    enabled=True,
                    priority=ModulePriority.RECOMMENDED,
                    reason="Query involves creative/innovative thinking",
                    estimated_cost_ms=self.module_costs['creative_reasoning']
                )
        
        # Check query type
        if chars.query_type == QueryType.CREATIVE:
            return ModuleSelection(
                enabled=True,
                priority=ModulePriority.REQUIRED,
                reason="Creative query type - assess divergent thinking",
                estimated_cost_ms=self.module_costs['creative_reasoning']
            )
        
        # Check domain
        if chars.domain in config['domains']:
            return ModuleSelection(
                enabled=True,
                priority=ModulePriority.RECOMMENDED,
                reason=f"Domain ({chars.domain}) benefits from creativity analysis",
                estimated_cost_ms=self.module_costs['creative_reasoning']
            )
        
        return ModuleSelection(
            enabled=False,
            priority=ModulePriority.SKIP,
            reason="No creative reasoning triggers detected",
            estimated_cost_ms=0
        )
    
    def _estimate_total_cost(self, plan: RoutingPlan) -> float:
        """Estimate total processing time."""
        total = 0.0
        all_modules = ['proactive_prevention', 'rag_quality', 'knowledge_graph',
                       'fact_checking', 'consistency_checker', 'grounding_detector',
                       'behavioral_detector', 'temporal_detector', 'factual_detector',
                       'theory_of_mind', 'neurosymbolic', 'world_models', 'creative_reasoning']
        
        for name in all_modules:
            selection = getattr(plan, name, None)
            if selection and selection.enabled:
                total += selection.estimated_cost_ms
        
        # Reduce by ~40% if parallel execution
        if plan.parallel_execution:
            total *= 0.6
        
        return total
    
    def get_available_modules_from_registry(self) -> Dict[str, Any]:
        """
        Query the registry for REAL-TIME module availability.
        
        This ensures the router NEVER has stale knowledge.
        Returns current status of all modules.
        """
        try:
            available = {}
            for module_id in self.registry.get_available_modules():
                cap = self.registry.get_capability(module_id)
                if cap:
                    available[module_id] = {
                        "name": cap.module_name,
                        "status": cap.status.value,
                        "health": cap.health_score,
                        "cost_ms": cap.estimated_cost_ms,
                        "always_run": cap.always_run,
                        "provides": cap.provides_signals,
                        "triggers": [t.name for t in cap.trigger_conditions],
                    }
            return available
        except Exception as e:
            # Fallback to hardcoded if registry unavailable
            return self.module_costs
    
    def analyze_with_registry(self, 
                               query: str, 
                               response: str = None,
                               documents: List[Dict] = None,
                               context: Dict[str, Any] = None) -> RoutingPlan:
        """
        Enhanced analysis using the Module Registry.
        
        This version queries the registry for real-time module status
        before making routing decisions.
        """
        # Get current module status from registry
        current_modules = self.get_available_modules_from_registry()
        
        # Update our cost estimates from registry (real-time)
        for module_id, info in current_modules.items():
            if module_id in self.module_costs:
                self.module_costs[module_id] = info.get('cost_ms', self.module_costs[module_id])
        
        # Get triggered modules from registry
        triggered_by_registry = self.registry.get_triggered_modules(
            query, response or "", documents or [], context or {}
        )
        
        # Now do standard analysis
        plan = self.analyze(query, response, documents, context)
        
        # Validate against registry - disable modules that aren't available
        for module_name in ['knowledge_graph', 'fact_checking', 'rag_quality', 
                           'proactive_prevention', 'consistency_checker']:
            selection = getattr(plan, module_name)
            cap = self.registry.get_capability(module_name)
            
            if cap:
                # Check if module is actually available
                from core.module_registry import ModuleStatus
                if cap.status not in [ModuleStatus.AVAILABLE, ModuleStatus.DEGRADED]:
                    selection.enabled = False
                    selection.reason = f"Module not available (status: {cap.status.value})"
                
                # Update cost from registry
                selection.estimated_cost_ms = cap.estimated_cost_ms
        
        # Add registry-based metadata
        plan.registry_triggered = triggered_by_registry
        plan.registry_available = list(current_modules.keys())
        
        return plan

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


# =============================================
# ROUTING PLAN PRINTER (for debugging)
# =============================================

def print_routing_plan(plan: RoutingPlan):
    """Pretty print routing plan for debugging."""
    print("\n" + "="*60)
    print(f"ROUTING PLAN: {plan.query_id}")
    print("="*60)
    
    chars = plan.characteristics
    print(f"\nQuery Analysis:")
    print(f"  Type: {chars.query_type.value}")
    print(f"  Domain: {chars.domain}")
    print(f"  Risk Level: {chars.risk_level.value}")
    print(f"  Complexity: {chars.estimated_complexity}")
    
    print(f"\nContent Flags:")
    print(f"  Has Documents: {chars.has_documents}")
    print(f"  Has Entities: {chars.has_entities} {chars.detected_entities[:3] if chars.has_entities else ''}")
    print(f"  Has Claims: {chars.has_factual_claims}")
    print(f"  Has Code: {chars.has_code}")
    print(f"  Sensitive Domain: {chars.is_sensitive_domain}")
    
    print(f"\nModule Selections:")
    for name in ['proactive_prevention', 'rag_quality', 'knowledge_graph',
                 'fact_checking', 'consistency_checker']:
        sel = getattr(plan, name)
        status = "✅" if sel.enabled else "⬚"
        print(f"  {status} {name}: {sel.priority.value} - {sel.reason}")
    
    print(f"\nCore Detectors:")
    for name in ['grounding_detector', 'behavioral_detector', 
                 'temporal_detector', 'factual_detector']:
        sel = getattr(plan, name)
        status = "✅" if sel.enabled else "⬚"
        print(f"  {status} {name}: {sel.priority.value}")
    
    print(f"\nEstimates:")
    print(f"  Modules: {plan.estimated_modules_count}")
    print(f"  Time: ~{plan.estimated_total_cost_ms:.0f}ms")
    print(f"  Parallel: {plan.parallel_execution}")
    print("="*60 + "\n")


# =============================================
# EXAMPLE USAGE
# =============================================

if __name__ == "__main__":
    router = QueryRouter()
    
    # Test cases
    test_cases = [
        {
            "query": "Hello, how are you?",
            "documents": [],
            "response": None,
        },
        {
            "query": "What is the capital of France?",
            "documents": [],
            "response": "The capital of France is Paris, with a population of 2.1 million.",
        },
        {
            "query": "What are the side effects of ibuprofen?",
            "documents": [{"content": "Ibuprofen may cause stomach pain..."}],
            "response": "Common side effects include stomach pain, nausea, and headache.",
        },
        {
            "query": "Compare Tesla and Ford stock performance in 2024",
            "documents": [
                {"content": "Tesla stock rose 15% in Q1..."},
                {"content": "Ford stock remained flat..."},
            ],
            "response": "Tesla outperformed Ford in 2024, with a 15% gain versus Ford's flat performance.",
        },
        {
            "query": "Write a Python function to sort a list",
            "documents": [],
            "response": "```python\ndef sort_list(items):\n    return sorted(items)\n```",
        },
    ]
    
    for i, tc in enumerate(test_cases, 1):
        print(f"\n{'#'*60}")
        print(f"TEST CASE {i}: {tc['query'][:50]}...")
        plan = router.analyze(
            query=tc['query'],
            response=tc.get('response'),
            documents=tc.get('documents', []),
        )
        print_routing_plan(plan)

