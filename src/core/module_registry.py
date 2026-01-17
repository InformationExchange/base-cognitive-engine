"""
BAIS Module Registry - Self-Aware Capability Management
Central catalog of all governance modules and their capabilities.

This ensures the Query Router ALWAYS knows what modules are available,
their current status, capabilities, and costs - in REAL TIME.

Pattern: Service Registry / Self-Describing Components
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Set
from enum import Enum
from abc import ABC, abstractmethod
import json
from pathlib import Path


class ModuleCategory(str, Enum):
    """Categories of governance modules."""
    DETECTOR = "detector"           # Signal detection (grounding, behavioral, etc.)
    PATHWAY = "pathway"             # Verification pathways (KG, fact-check, RAG)
    ENHANCEMENT = "enhancement"     # Enhancement modules (proactive, consistency)
    LEARNING = "learning"           # Learning components (OCO, feedback)
    AUDIT = "audit"                 # Audit components (merkle, hash chain)
    RESEARCH = "research"           # R&D modules (experimental)


class ModuleStatus(str, Enum):
    """Runtime status of a module."""
    AVAILABLE = "available"         # Ready to use
    INITIALIZING = "initializing"   # Starting up
    DEGRADED = "degraded"           # Working but limited
    UNAVAILABLE = "unavailable"     # Not working
    DISABLED = "disabled"           # Explicitly disabled


@dataclass
class TriggerCondition:
    """Describes when a module should be triggered."""
    name: str
    description: str
    check_function: str              # Name of function to call for check
    required_inputs: List[str]       # What inputs are needed (query, response, documents)
    priority_domains: List[str]      # Domains where this is high priority


@dataclass
class ModuleCapability:
    """
    Complete capability description for a module.
    Modules register themselves with this information.
    """
    # Identity
    module_id: str                   # Unique ID (e.g., "knowledge_graph")
    module_name: str                 # Human-readable name
    version: str                     # Module version
    category: ModuleCategory
    
    # Status
    status: ModuleStatus = ModuleStatus.AVAILABLE
    health_score: float = 1.0        # 0.0 to 1.0
    last_health_check: float = 0.0
    
    # Capabilities
    description: str = ""
    provides_signals: List[str] = field(default_factory=list)  # What signals it outputs
    requires_signals: List[str] = field(default_factory=list)  # What signals it needs
    
    # Triggers
    trigger_conditions: List[TriggerCondition] = field(default_factory=list)
    always_run: bool = False         # Should always execute
    
    # Cost & Performance
    estimated_cost_ms: float = 0.0   # Average execution time
    memory_usage_mb: float = 0.0     # Memory footprint
    requires_gpu: bool = False       # Needs GPU
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)  # Other modules needed
    conflicts_with: List[str] = field(default_factory=list)  # Incompatible modules
    
    # Configuration
    config_schema: Dict[str, Any] = field(default_factory=dict)
    current_config: Dict[str, Any] = field(default_factory=dict)
    
    # Patent/IP
    patent_claims: List[str] = field(default_factory=list)  # Related patent claims
    
    # Metadata
    registered_at: float = 0.0
    last_used: float = 0.0
    usage_count: int = 0
    success_rate: float = 1.0


class GovernanceModule(ABC):
    """
    Base class for all governance modules.
    Modules that extend this class automatically register themselves.
    """
    
    @property
    @abstractmethod
    def capability(self) -> ModuleCapability:
        """Return this module's capability descriptor."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the module's main function."""
        pass
    
    def check_trigger(self, query: str, response: str, 
                      documents: List[Dict], context: Dict) -> bool:
        """Check if this module should be triggered. Override in subclass."""
        return True
    
    def health_check(self) -> float:
        """Return health score 0.0-1.0. Override in subclass."""
        return 1.0

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


class ModuleRegistry:
    """
    Central registry of all governance modules.
    
    Features:
    - Modules self-register with their capabilities
    - Real-time status tracking
    - Dependency resolution
    - Query Router queries this to know what's available
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern - one registry for the entire system."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._modules: Dict[str, ModuleCapability] = {}
        self._module_instances: Dict[str, GovernanceModule] = {}
        self._category_index: Dict[ModuleCategory, Set[str]] = {
            cat: set() for cat in ModuleCategory
        }
        self._signal_providers: Dict[str, Set[str]] = {}  # signal -> modules that provide it
        self._signal_consumers: Dict[str, Set[str]] = {}  # signal -> modules that need it
        
        self._health_check_interval = 60  # seconds
        self._last_health_check = 0
        
        self._initialized = True
        
        # Register built-in modules on init
        self._register_builtin_modules()
    
    def _register_builtin_modules(self):
        """Register all built-in BAIS modules."""
        
        # Core Detectors
        self.register(ModuleCapability(
            module_id="grounding_detector",
            module_name="Grounding Detector",
            version="16.6.0",
            category=ModuleCategory.DETECTOR,
            description="Detects how well response is grounded in source documents",
            provides_signals=["grounding_score", "grounding_details"],
            trigger_conditions=[
                TriggerCondition(
                    name="has_documents",
                    description="Documents are provided for grounding",
                    check_function="_has_documents",
                    required_inputs=["documents"],
                    priority_domains=["medical", "legal", "financial"]
                )
            ],
            always_run=True,  # Always runs (score=0 if no docs)
            estimated_cost_ms=40,
            patent_claims=["PPA1-Inv1", "PPA1-Inv2"]
        ))
        
        self.register(ModuleCapability(
            module_id="behavioral_detector",
            module_name="Behavioral Bias Detector",
            version="16.6.0",
            category=ModuleCategory.DETECTOR,
            description="Detects behavioral biases (confirmation, reward-seeking, etc.)",
            provides_signals=["behavioral_score", "biases_detected"],
            always_run=True,
            estimated_cost_ms=20,
            patent_claims=["PPA3-Inv2", "PPA3-Inv3", "PPA3-Inv4", "PPA3-Inv5"]
        ))
        
        self.register(ModuleCapability(
            module_id="temporal_detector",
            module_name="Temporal Detector",
            version="16.6.0",
            category=ModuleCategory.DETECTOR,
            description="Tracks temporal patterns and stability",
            provides_signals=["temporal_score", "temporal_state"],
            always_run=True,
            estimated_cost_ms=10,
            patent_claims=["PPA1-Inv3"]
        ))
        
        self.register(ModuleCapability(
            module_id="factual_detector",
            module_name="Factual Detector",
            version="16.6.0",
            category=ModuleCategory.DETECTOR,
            description="Verifies factual accuracy using NLI",
            provides_signals=["factual_score", "factual_details"],
            trigger_conditions=[
                TriggerCondition(
                    name="has_factual_content",
                    description="Response contains factual claims",
                    check_function="_has_factual_claims",
                    required_inputs=["response"],
                    priority_domains=["medical", "financial", "legal", "technical"]
                )
            ],
            always_run=True,
            estimated_cost_ms=50,
            patent_claims=["PPA1-Inv4"]
        ))
        
        # New Pathways (to be migrated)
        self.register(ModuleCapability(
            module_id="knowledge_graph",
            module_name="Knowledge Graph Pathway",
            version="1.0.0",
            category=ModuleCategory.PATHWAY,
            status=ModuleStatus.INITIALIZING,  # Not yet migrated
            description="Entity/relationship extraction and verification",
            provides_signals=["kg_alignment_score", "entities", "relationships", "contradictions"],
            trigger_conditions=[
                TriggerCondition(
                    name="has_entities",
                    description="Response contains named entities",
                    check_function="_has_entities",
                    required_inputs=["response"],
                    priority_domains=["medical", "financial", "legal"]
                )
            ],
            estimated_cost_ms=50,
            patent_claims=["PPA1-Inv6", "Enhancement-4"]
        ))
        
        self.register(ModuleCapability(
            module_id="fact_checking",
            module_name="Fact-Checking Pathway",
            version="1.0.0",
            category=ModuleCategory.PATHWAY,
            status=ModuleStatus.INITIALIZING,
            description="Claim extraction and verification",
            provides_signals=["fact_check_coverage", "verified_claims", "contradicted_claims"],
            requires_signals=["grounding_score"],  # Uses grounding
            trigger_conditions=[
                TriggerCondition(
                    name="has_claims",
                    description="Response contains verifiable claims",
                    check_function="_has_factual_claims",
                    required_inputs=["response"],
                    priority_domains=["medical", "financial", "legal"]
                )
            ],
            estimated_cost_ms=100,
            patent_claims=["Enhancement-2"]
        ))
        
        self.register(ModuleCapability(
            module_id="rag_quality",
            module_name="RAG Quality Scoring",
            version="1.0.0",
            category=ModuleCategory.PATHWAY,
            status=ModuleStatus.INITIALIZING,
            description="Document quality scoring (relevance, coverage, diversity)",
            provides_signals=["rag_quality_score", "relevance", "coverage", "diversity"],
            trigger_conditions=[
                TriggerCondition(
                    name="has_documents",
                    description="Documents provided for RAG",
                    check_function="_has_documents",
                    required_inputs=["documents"],
                    priority_domains=["all"]
                )
            ],
            estimated_cost_ms=30,
            patent_claims=["Enhancement-1"]
        ))
        
        # Enhancements
        self.register(ModuleCapability(
            module_id="proactive_prevention",
            module_name="Proactive Hallucination Prevention",
            version="1.0.0",
            category=ModuleCategory.ENHANCEMENT,
            status=ModuleStatus.INITIALIZING,
            description="Pre-emptive risk assessment before generation",
            provides_signals=["risk_level", "risk_factors", "constraints"],
            always_run=True,
            estimated_cost_ms=20,
            patent_claims=["Enhancement-5"]
        ))
        
        self.register(ModuleCapability(
            module_id="consistency_checker",
            module_name="Cross-Pathway Consistency",
            version="1.0.0",
            category=ModuleCategory.ENHANCEMENT,
            status=ModuleStatus.INITIALIZING,
            description="Validates consistency across all signals",
            requires_signals=["grounding_score", "factual_score", "behavioral_score"],
            provides_signals=["consistency_score", "conflicts"],
            always_run=True,
            estimated_cost_ms=5,
            dependencies=["grounding_detector", "factual_detector", "behavioral_detector"],
            patent_claims=["Enhancement-6"]
        ))
        
        # R&D Modules
        self.register(ModuleCapability(
            module_id="theory_of_mind",
            module_name="Theory of Mind",
            version="0.1.0",
            category=ModuleCategory.RESEARCH,
            status=ModuleStatus.DISABLED,  # R&D - not enabled by default
            description="Mental state inference and perspective taking",
            provides_signals=["mental_states", "perspectives"],
            estimated_cost_ms=200,
            requires_gpu=True,
            patent_claims=["R&D-1"]
        ))
        
        self.register(ModuleCapability(
            module_id="neurosymbolic",
            module_name="Neuro-Symbolic Reasoning",
            version="0.1.0",
            category=ModuleCategory.RESEARCH,
            status=ModuleStatus.DISABLED,
            description="Combined neural and symbolic reasoning",
            provides_signals=["symbolic_proof", "reasoning_trace"],
            estimated_cost_ms=150,
            dependencies=["z3_solver"],  # External dependency
            patent_claims=["R&D-2"]
        ))
        
        self.register(ModuleCapability(
            module_id="world_models",
            module_name="World Models",
            version="0.1.0",
            category=ModuleCategory.RESEARCH,
            status=ModuleStatus.DISABLED,
            description="Forward prediction and simulation",
            provides_signals=["predictions", "simulations"],
            estimated_cost_ms=300,
            requires_gpu=True,
            patent_claims=["R&D-3"]
        ))
        
        self.register(ModuleCapability(
            module_id="creative_reasoning",
            module_name="Creative/Abstract Reasoning",
            version="0.1.0",
            category=ModuleCategory.RESEARCH,
            status=ModuleStatus.DISABLED,
            description="Divergent thinking and analogy",
            provides_signals=["creative_score", "analogies"],
            estimated_cost_ms=100,
            patent_claims=["R&D-4"]
        ))
        
        # Learning Components
        self.register(ModuleCapability(
            module_id="oco_learner",
            module_name="OCO Learning Algorithm",
            version="16.6.0",
            category=ModuleCategory.LEARNING,
            description="Online Convex Optimization for threshold learning",
            provides_signals=["threshold", "learning_update"],
            always_run=True,
            estimated_cost_ms=5,
            patent_claims=["PPA2-Inv27"]
        ))
        
        self.register(ModuleCapability(
            module_id="verifiable_audit",
            module_name="Verifiable Audit System",
            version="16.6.0",
            category=ModuleCategory.AUDIT,
            description="Merkle tree and hash chain audit logging",
            always_run=True,
            estimated_cost_ms=2,
            patent_claims=["PPA2-Inv26"]
        ))
    
    # =============================================
    # REGISTRATION METHODS
    # =============================================
    
    def register(self, capability: ModuleCapability, 
                 instance: GovernanceModule = None) -> bool:
        """
        Register a module with its capabilities.
        
        Args:
            capability: Module's capability descriptor
            instance: Optional module instance
        
        Returns:
            True if registered successfully
        """
        module_id = capability.module_id
        
        # Update registration time
        capability.registered_at = time.time()
        
        # Store capability
        self._modules[module_id] = capability
        
        # Store instance if provided
        if instance:
            self._module_instances[module_id] = instance
        
        # Update indexes
        self._category_index[capability.category].add(module_id)
        
        for signal in capability.provides_signals:
            if signal not in self._signal_providers:
                self._signal_providers[signal] = set()
            self._signal_providers[signal].add(module_id)
        
        for signal in capability.requires_signals:
            if signal not in self._signal_consumers:
                self._signal_consumers[signal] = set()
            self._signal_consumers[signal].add(module_id)
        
        return True
    
    def unregister(self, module_id: str) -> bool:
        """Remove a module from the registry."""
        if module_id not in self._modules:
            return False
        
        cap = self._modules[module_id]
        
        # Remove from indexes
        self._category_index[cap.category].discard(module_id)
        
        for signal in cap.provides_signals:
            if signal in self._signal_providers:
                self._signal_providers[signal].discard(module_id)
        
        for signal in cap.requires_signals:
            if signal in self._signal_consumers:
                self._signal_consumers[signal].discard(module_id)
        
        # Remove from storage
        del self._modules[module_id]
        if module_id in self._module_instances:
            del self._module_instances[module_id]
        
        return True
    
    # =============================================
    # QUERY METHODS
    # =============================================
    
    def get_capability(self, module_id: str) -> Optional[ModuleCapability]:
        """Get capability for a specific module."""
        return self._modules.get(module_id)
    
    def get_all_capabilities(self) -> Dict[str, ModuleCapability]:
        """Get all registered capabilities."""
        return dict(self._modules)
    
    def get_available_modules(self) -> List[str]:
        """Get list of available (not disabled/unavailable) modules."""
        return [
            mid for mid, cap in self._modules.items()
            if cap.status in [ModuleStatus.AVAILABLE, ModuleStatus.DEGRADED]
        ]
    
    def get_modules_by_category(self, category: ModuleCategory) -> List[str]:
        """Get modules in a specific category."""
        return list(self._category_index.get(category, set()))
    
    def get_modules_for_signal(self, signal: str) -> List[str]:
        """Get modules that provide a specific signal."""
        return list(self._signal_providers.get(signal, set()))
    
    def get_always_run_modules(self) -> List[str]:
        """Get modules that should always run."""
        return [
            mid for mid, cap in self._modules.items()
            if cap.always_run and cap.status == ModuleStatus.AVAILABLE
        ]
    
    def get_triggered_modules(self, 
                               query: str, 
                               response: str,
                               documents: List[Dict],
                               context: Dict) -> List[str]:
        """
        Get modules that should be triggered for this input.
        
        This is the KEY method that the Query Router calls.
        """
        triggered = []
        
        for module_id, cap in self._modules.items():
            # Skip unavailable/disabled
            if cap.status not in [ModuleStatus.AVAILABLE, ModuleStatus.DEGRADED]:
                continue
            
            # Always run modules
            if cap.always_run:
                triggered.append(module_id)
                continue
            
            # Check trigger conditions
            for trigger in cap.trigger_conditions:
                if self._check_trigger(trigger, query, response, documents, context):
                    triggered.append(module_id)
                    break
        
        return triggered
    
    def _check_trigger(self, trigger: TriggerCondition,
                       query: str, response: str,
                       documents: List[Dict], context: Dict) -> bool:
        """Check if a trigger condition is met."""
        
        check_fn = trigger.check_function
        
        if check_fn == "_has_documents":
            return len(documents) > 0
        
        elif check_fn == "_has_entities":
            import re
            combined = f"{query} {response or ''}"
            # Quick entity detection
            caps = len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', combined))
            return caps >= 2
        
        elif check_fn == "_has_factual_claims":
            import re
            combined = f"{query} {response or ''}"
            # Quick claim detection
            has_numbers = bool(re.search(r'\b\d+\b', combined))
            has_definitive = any(w in combined.lower() for w in ['is the', 'was the', 'are the'])
            return has_numbers or has_definitive
        
        return False
    
    # =============================================
    # STATUS & HEALTH
    # =============================================
    
    def update_status(self, module_id: str, status: ModuleStatus) -> bool:
        """Update module status."""
        if module_id not in self._modules:
            return False
        self._modules[module_id].status = status
        return True
    
    def update_health(self, module_id: str, health_score: float) -> bool:
        """Update module health score."""
        if module_id not in self._modules:
            return False
        cap = self._modules[module_id]
        cap.health_score = health_score
        cap.last_health_check = time.time()
        
        # Auto-update status based on health
        if health_score < 0.3:
            cap.status = ModuleStatus.UNAVAILABLE
        elif health_score < 0.7:
            cap.status = ModuleStatus.DEGRADED
        else:
            cap.status = ModuleStatus.AVAILABLE
        
        return True
    
    def run_health_checks(self):
        """Run health checks on all modules with instances."""
        for module_id, instance in self._module_instances.items():
            try:
                health = instance.health_check()
                self.update_health(module_id, health)
            except Exception as e:
                self.update_health(module_id, 0.0)
    
    def record_usage(self, module_id: str, success: bool):
        """Record module usage for statistics."""
        if module_id not in self._modules:
            return
        cap = self._modules[module_id]
        cap.usage_count += 1
        cap.last_used = time.time()
        
        # Update success rate (exponential moving average)
        alpha = 0.1
        cap.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * cap.success_rate
    
    # =============================================
    # DEPENDENCY RESOLUTION
    # =============================================
    
    def resolve_dependencies(self, module_ids: List[str]) -> List[str]:
        """
        Resolve dependencies and return execution order.
        
        Returns modules in order they should execute.
        """
        # Build dependency graph
        deps = {}
        for mid in module_ids:
            cap = self._modules.get(mid)
            if cap:
                deps[mid] = set(cap.dependencies) & set(module_ids)
        
        # Topological sort
        result = []
        visited = set()
        
        def visit(mid):
            if mid in visited:
                return
            visited.add(mid)
            for dep in deps.get(mid, []):
                visit(dep)
            result.append(mid)
        
        for mid in module_ids:
            visit(mid)
        
        return result
    
    # =============================================
    # SERIALIZATION
    # =============================================
    
    def to_dict(self) -> Dict:
        """Export registry state as dictionary."""
        return {
            "modules": {
                mid: {
                    "module_id": cap.module_id,
                    "module_name": cap.module_name,
                    "version": cap.version,
                    "category": cap.category.value,
                    "status": cap.status.value,
                    "health_score": cap.health_score,
                    "always_run": cap.always_run,
                    "estimated_cost_ms": cap.estimated_cost_ms,
                    "provides_signals": cap.provides_signals,
                    "requires_signals": cap.requires_signals,
                    "patent_claims": cap.patent_claims,
                    "usage_count": cap.usage_count,
                    "success_rate": cap.success_rate,
                }
                for mid, cap in self._modules.items()
            },
            "statistics": {
                "total_modules": len(self._modules),
                "available": len(self.get_available_modules()),
                "by_category": {
                    cat.value: len(mids)
                    for cat, mids in self._category_index.items()
                }
            }
        }
    
    def save(self, path: Path):
        """Save registry state to file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def print_status(self):
        """Print current registry status."""
        print("\n" + "="*70)
        print("BAIS MODULE REGISTRY STATUS")
        print("="*70)
        
        by_status = {}
        for mid, cap in self._modules.items():
            status = cap.status.value
            if status not in by_status:
                by_status[status] = []
            by_status[status].append((mid, cap))
        
        for status, modules in sorted(by_status.items()):
            status_icon = {
                'available': '✅',
                'initializing': '🔄',
                'degraded': '⚠️',
                'unavailable': '❌',
                'disabled': '⬚'
            }.get(status, '?')
            
            print(f"\n{status_icon} {status.upper()} ({len(modules)}):")
            for mid, cap in modules:
                print(f"   {mid}: {cap.module_name} v{cap.version}")
                print(f"      Signals: {cap.provides_signals}")
                print(f"      Cost: ~{cap.estimated_cost_ms}ms | Always: {cap.always_run}")
        
        print("\n" + "="*70)

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


# =============================================
# SINGLETON ACCESS
# =============================================

def get_registry() -> ModuleRegistry:
    """Get the global module registry instance."""
    return ModuleRegistry()


# =============================================
# EXAMPLE USAGE
# =============================================

if __name__ == "__main__":
    registry = get_registry()
    
    # Print current status
    registry.print_status()
    
    # Test trigger detection
    print("\n" + "="*70)
    print("TRIGGER TEST")
    print("="*70)
    
    test_cases = [
        {
            "query": "Hello",
            "response": "Hi there!",
            "documents": [],
        },
        {
            "query": "What is Tesla's stock price?",
            "response": "Tesla stock is trading at $248.50, up 5% from yesterday.",
            "documents": [{"content": "Tesla stock data..."}],
        },
    ]
    
    for tc in test_cases:
        print(f"\nQuery: {tc['query']}")
        triggered = registry.get_triggered_modules(
            tc['query'], tc['response'], tc['documents'], {}
        )
        print(f"Triggered modules: {triggered}")
    
    # Export to JSON
    print("\n" + "="*70)
    print("REGISTRY EXPORT")
    print("="*70)
    print(json.dumps(registry.to_dict()["statistics"], indent=2))






