"""
BASE Dynamic Plugin System with Shared Learning (NOVEL-54)

Enables:
1. Dynamic plugin creation from semantic understanding
2. Shared knowledge base across all plugins
3. Cross-domain intelligence
4. Industry-specific tuning with common foundation

Brain Layer: 6 (Knowledge Integration)
Patent Alignment: Novel invention for adaptive domain handling
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Protocol, Set, Tuple
from enum import Enum
from datetime import datetime
import logging
import json
import hashlib

logger = logging.getLogger(__name__)


# =============================================================================
# SHARED KNOWLEDGE BASE
# =============================================================================

@dataclass
class Learning:
    """A single learning/pattern."""
    pattern_id: str
    pattern_type: str              # "evidence_rule", "enforcement", "validation", etc.
    description: str
    conditions: Dict[str, Any]     # When this applies
    action: Dict[str, Any]         # What to do
    source_domains: Set[str]       # Which domains contributed
    confidence: float              # How confident (0-1)
    usage_count: int = 0
    success_rate: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def is_applicable(self, context: Dict) -> Tuple[bool, float]:
        """Check if this learning applies to given context."""
        score = 0.0
        total_conditions = len(self.conditions)
        
        if total_conditions == 0:
            return True, 1.0  # Universal pattern
        
        for key, expected in self.conditions.items():
            actual = context.get(key)
            if actual == expected:
                score += 1.0
            elif isinstance(expected, list) and actual in expected:
                score += 0.8
            elif str(expected).lower() in str(actual).lower():
                score += 0.5
        
        match_ratio = score / total_conditions
        return match_ratio > 0.5, match_ratio


class SharedKnowledgeBase:
    """
    Central repository of learnings shared across all plugins.
    
    Architecture:
    - Common patterns (apply to all)
    - Domain-specific patterns (tagged by source)
    - Cross-domain patterns (combinations)
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self._learnings: Dict[str, Learning] = {}
        self._domain_index: Dict[str, Set[str]] = {}  # domain -> pattern_ids
        self._type_index: Dict[str, Set[str]] = {}    # type -> pattern_ids
        self._usage_stats: Dict[str, Dict] = {}
        self.storage_path = storage_path
        
        # Seed with universal patterns
        self._seed_common_patterns()
    
    def _seed_common_patterns(self):
        """Seed base patterns that apply universally."""
        common_patterns = [
            Learning(
                pattern_id="common_citation",
                pattern_type="evidence_rule",
                description="Citations must have verifiable sources",
                conditions={},  # Applies universally
                action={"require": "source_verification", "min_confidence": 0.7},
                source_domains={"common"},
                confidence=0.95
            ),
            Learning(
                pattern_id="common_completion_proof",
                pattern_type="enforcement",
                description="Completion claims require execution proof",
                conditions={},
                action={"require": "execution_evidence", "reject_assertions": True},
                source_domains={"common"},
                confidence=0.99
            ),
            Learning(
                pattern_id="common_sample_insufficient",
                pattern_type="validation",
                description="Sample-based evidence is insufficient for 100% claims",
                conditions={"claim_type": "complete"},
                action={"reject": "sample_based", "require": "exhaustive"},
                source_domains={"common"},
                confidence=0.95
            ),
            Learning(
                pattern_id="common_multitrack_critical",
                pattern_type="verification",
                description="Critical claims require multi-track verification",
                conditions={"importance": ["high", "critical"]},
                action={"min_tracks": 3, "require_consensus": True},
                source_domains={"common"},
                confidence=0.9
            ),
        ]
        
        for learning in common_patterns:
            self.add_learning(learning)
    
    def add_learning(self, learning: Learning) -> str:
        """Add a learning to the knowledge base."""
        self._learnings[learning.pattern_id] = learning
        
        # Index by domains
        for domain in learning.source_domains:
            if domain not in self._domain_index:
                self._domain_index[domain] = set()
            self._domain_index[domain].add(learning.pattern_id)
        
        # Index by type
        if learning.pattern_type not in self._type_index:
            self._type_index[learning.pattern_type] = set()
        self._type_index[learning.pattern_type].add(learning.pattern_id)
        
        logger.info(f"[SharedKnowledge] Added: {learning.pattern_id} from {learning.source_domains}")
        return learning.pattern_id
    
    def get_applicable_learnings(
        self,
        context: Dict,
        domains: List[str],
        pattern_types: Optional[List[str]] = None
    ) -> List[Tuple[Learning, float]]:
        """
        Get all learnings applicable to a context.
        Returns (learning, relevance_score) tuples.
        """
        relevant = []
        seen_ids = set()
        
        # Always include common patterns
        domains_to_check = set(domains) | {"common"}
        
        for domain in domains_to_check:
            pattern_ids = self._domain_index.get(domain, set())
            
            for pid in pattern_ids:
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                
                learning = self._learnings.get(pid)
                if not learning:
                    continue
                
                # Filter by type if specified
                if pattern_types and learning.pattern_type not in pattern_types:
                    continue
                
                # Check applicability
                is_applicable, relevance = learning.is_applicable(context)
                if is_applicable:
                    # Boost score for learnings from requested domains
                    domain_overlap = len(learning.source_domains & set(domains))
                    boosted_relevance = relevance * (1 + domain_overlap * 0.1)
                    relevant.append((learning, min(boosted_relevance, 1.0)))
        
        # Sort by relevance
        relevant.sort(key=lambda x: x[1], reverse=True)
        return relevant
    
    def record_usage(self, pattern_id: str, was_successful: bool):
        """Record pattern usage for learning."""
        if pattern_id in self._learnings:
            learning = self._learnings[pattern_id]
            learning.usage_count += 1
            
            # Update success rate (exponential moving average)
            alpha = 0.1
            learning.success_rate = (
                alpha * (1.0 if was_successful else 0.0) +
                (1 - alpha) * learning.success_rate
            )
    
    def extract_cross_domain_patterns(self, min_domains: int = 2) -> List[Learning]:
        """Find patterns that appear across multiple domains."""
        cross_domain = []
        for learning in self._learnings.values():
            if len(learning.source_domains) >= min_domains:
                cross_domain.append(learning)
        return cross_domain
    
    def serialize(self) -> Dict:
        """Serialize knowledge base."""
        return {
            "learnings": {
                pid: {
                    "pattern_id": l.pattern_id,
                    "pattern_type": l.pattern_type,
                    "description": l.description,
                    "conditions": l.conditions,
                    "action": l.action,
                    "source_domains": list(l.source_domains),
                    "confidence": l.confidence,
                    "usage_count": l.usage_count,
                    "success_rate": l.success_rate
                }
                for pid, l in self._learnings.items()
            }
        }
    
    def deserialize(self, data: Dict):
        """Restore from serialized data."""
        for pid, ldata in data.get("learnings", {}).items():
            learning = Learning(
                pattern_id=ldata["pattern_id"],
                pattern_type=ldata["pattern_type"],
                description=ldata["description"],
                conditions=ldata["conditions"],
                action=ldata["action"],
                source_domains=set(ldata["source_domains"]),
                confidence=ldata["confidence"],
                usage_count=ldata.get("usage_count", 0),
                success_rate=ldata.get("success_rate", 1.0)
            )
            self.add_learning(learning)


# =============================================================================
# DYNAMIC PLUGIN
# =============================================================================

@dataclass
class DynamicPluginConfig:
    """Configuration for a dynamically created plugin."""
    domain: str
    description: str
    parent_domains: List[str]      # Inherit from these
    evidence_requirements: Dict[str, Any]
    enforcement_rules: Dict[str, Any]
    validation_patterns: List[Dict]
    llm_generated_rules: List[Dict]
    confidence: float = 0.8        # New plugins start lower
    created_at: datetime = field(default_factory=datetime.now)
    is_dynamic: bool = True


class DynamicPlugin:
    """
    A plugin that can be created dynamically and learns over time.
    
    Inherits from:
    - Common knowledge base
    - Parent domain plugins
    - LLM-generated rules
    """
    
    def __init__(
        self,
        config: DynamicPluginConfig,
        knowledge_base: SharedKnowledgeBase
    ):
        self.config = config
        self.knowledge_base = knowledge_base
        
        # Local learnings (before promoting to shared)
        self._local_learnings: List[Learning] = []
        self._outcomes: List[Dict] = []
        self._performance: Dict[str, float] = {
            "accuracy": 0.8,
            "coverage": 0.5
        }
    
    @property
    def domain(self) -> str:
        return self.config.domain
    
    def get_proof_requirements(self, claim_type: str) -> Dict[str, Any]:
        """Get proof requirements, combining inherited and local."""
        requirements = {}
        
        # Get from parent domains
        for parent in self.config.parent_domains:
            parent_learnings = self.knowledge_base.get_applicable_learnings(
                context={"claim_type": claim_type},
                domains=[parent],
                pattern_types=["evidence_rule"]
            )
            for learning, _ in parent_learnings:
                requirements.update(learning.action)
        
        # Add local requirements
        requirements.update(self.config.evidence_requirements.get(claim_type, {}))
        
        # Add LLM-generated rules
        for rule in self.config.llm_generated_rules:
            if rule.get("applies_to") == claim_type or rule.get("applies_to") == "all":
                requirements.update(rule.get("requirements", {}))
        
        return requirements
    
    def validate_evidence(self, evidence: List[str], context: Dict) -> Tuple[bool, List[str]]:
        """Validate evidence using inherited + local patterns."""
        issues = []
        
        # Get applicable validation patterns
        learnings = self.knowledge_base.get_applicable_learnings(
            context={**context, "domain": self.domain},
            domains=[self.domain] + self.config.parent_domains,
            pattern_types=["validation"]
        )
        
        for learning, relevance in learnings:
            action = learning.action
            
            # Check rejection rules
            if action.get("reject"):
                reject_pattern = action["reject"]
                for e in evidence:
                    if reject_pattern.lower() in e.lower():
                        issues.append(f"Rejected: {reject_pattern} evidence ({learning.description})")
            
            # Check requirement rules
            if action.get("require"):
                require_pattern = action["require"]
                found = any(require_pattern.lower() in e.lower() for e in evidence)
                if not found:
                    issues.append(f"Missing: {require_pattern} ({learning.description})")
        
        return len(issues) == 0, issues
    
    def get_enforcement_level(self, context: Dict) -> str:
        """Determine enforcement level for context."""
        # Check importance
        importance = context.get("importance", "normal")
        if importance in ["critical", "high"]:
            return "STRICT"
        
        # Check if multi-domain (needs more care)
        domains = context.get("domains", [self.domain])
        if len(domains) > 1:
            return "STRICT"
        
        return self.config.enforcement_rules.get("default_level", "MODERATE")
    
    def contribute_learning(self, learning: Learning) -> bool:
        """
        Contribute a learning back to shared knowledge base.
        Only promotes if confidence is high enough.
        """
        if learning.confidence < 0.7:
            # Keep local for now
            self._local_learnings.append(learning)
            return False
        
        # Add this domain as source
        learning.source_domains.add(self.domain)
        
        # Promote to shared
        self.knowledge_base.add_learning(learning)
        return True
    
    def record_outcome(self, outcome: Dict) -> None:
        """Record outcome for learning."""
        self._outcomes.append(outcome)
        
        # Update performance metrics
        if "was_correct" in outcome:
            alpha = 0.1
            self._performance["accuracy"] = (
                alpha * (1.0 if outcome["was_correct"] else 0.0) +
                (1 - alpha) * self._performance["accuracy"]
            )
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback to improve plugin."""
        # Extract potential new patterns from feedback
        if feedback.get("new_pattern"):
            pattern = feedback["new_pattern"]
            learning = Learning(
                pattern_id=f"{self.domain}_{hashlib.md5(str(pattern).encode()).hexdigest()[:8]}",
                pattern_type=pattern.get("type", "validation"),
                description=pattern.get("description", "Learned pattern"),
                conditions=pattern.get("conditions", {}),
                action=pattern.get("action", {}),
                source_domains={self.domain},
                confidence=0.6  # Starts lower
            )
            self._local_learnings.append(learning)
            
            # Promote if successful enough
            if len([o for o in self._outcomes[-10:] if o.get("was_correct")]) >= 8:
                self.contribute_learning(learning)
    
    def serialize(self) -> Dict:
        """Serialize plugin state."""
        return {
            "config": {
                "domain": self.config.domain,
                "description": self.config.description,
                "parent_domains": self.config.parent_domains,
                "evidence_requirements": self.config.evidence_requirements,
                "enforcement_rules": self.config.enforcement_rules,
                "validation_patterns": self.config.validation_patterns,
                "llm_generated_rules": self.config.llm_generated_rules,
                "confidence": self.config.confidence,
                "is_dynamic": self.config.is_dynamic
            },
            "local_learnings": [
                {
                    "pattern_id": l.pattern_id,
                    "description": l.description,
                    "confidence": l.confidence
                }
                for l in self._local_learnings
            ],
            "performance": self._performance
        }


# =============================================================================
# DYNAMIC PLUGIN FACTORY
# =============================================================================

class DynamicPluginFactory:
    """
    Creates plugins dynamically based on semantic understanding.
    
    Process:
    1. Analyze context to identify domains
    2. Check if plugin exists (memory + disk)
    3. If not, create from:
       - Parent domain learnings
       - LLM-generated rules
       - Common patterns
    4. Auto-persist to disk for permanence
    """
    
    DEFAULT_STORAGE_PATH = "data/plugins"
    
    def __init__(
        self,
        knowledge_base: SharedKnowledgeBase,
        llm_registry: Any = None,
        multi_track: Any = None,
        storage_path: Optional[str] = None,
        auto_persist: bool = True
    ):
        self.knowledge_base = knowledge_base
        self.llm_registry = llm_registry
        self.multi_track = multi_track
        self.storage_path = storage_path or self.DEFAULT_STORAGE_PATH
        self.auto_persist = auto_persist
        
        # Cache of created plugins
        self._plugins: Dict[str, DynamicPlugin] = {}
        
        # Load persisted plugins on startup
        self._load_persisted_plugins()
        
        # Domain hierarchy (what inherits from what)
        self._domain_hierarchy = {
            "aerospace_coding": ["vibe_coding", "legal"],
            "medical_device": ["healthcare", "vibe_coding"],
            "fintech": ["financial", "vibe_coding"],
            "legal_tech": ["legal", "vibe_coding"],
            "pharma": ["healthcare", "legal"],
            "insurance": ["financial", "legal"],
            "defense": ["aerospace_coding", "legal"],
        }
        
        # Pre-register static plugins
        self._register_static_plugins()
    
    def _load_persisted_plugins(self):
        """Load previously saved dynamic plugins from disk."""
        import os
        
        if not os.path.exists(self.storage_path):
            return
        
        try:
            # Load knowledge base
            kb_path = os.path.join(self.storage_path, "knowledge_base.json")
            if os.path.exists(kb_path):
                with open(kb_path, 'r') as f:
                    kb_data = json.load(f)
                    self.knowledge_base.deserialize(kb_data)
                    logger.info(f"[PluginFactory] Loaded {len(kb_data.get('learnings', {}))} learnings from disk")
            
            # Load dynamic plugins
            plugins_path = os.path.join(self.storage_path, "plugins")
            if os.path.exists(plugins_path):
                for filename in os.listdir(plugins_path):
                    if filename.endswith('.json'):
                        filepath = os.path.join(plugins_path, filename)
                        with open(filepath, 'r') as f:
                            plugin_data = json.load(f)
                            self._restore_plugin(plugin_data)
                            
            logger.info(f"[PluginFactory] Loaded {len(self._plugins)} plugins from disk")
        except Exception as e:
            logger.warning(f"[PluginFactory] Failed to load persisted plugins: {e}")
    
    def _restore_plugin(self, data: Dict):
        """Restore a plugin from serialized data."""
        config_data = data.get('config', {})
        if not config_data.get('is_dynamic', True):
            return  # Skip static plugins
        
        config = DynamicPluginConfig(
            domain=config_data['domain'],
            description=config_data['description'],
            parent_domains=config_data['parent_domains'],
            evidence_requirements=config_data['evidence_requirements'],
            enforcement_rules=config_data['enforcement_rules'],
            validation_patterns=config_data['validation_patterns'],
            llm_generated_rules=config_data['llm_generated_rules'],
            confidence=config_data.get('confidence', 0.7),
            is_dynamic=True
        )
        
        plugin = DynamicPlugin(config, self.knowledge_base)
        plugin._performance = data.get('performance', {"accuracy": 0.8, "coverage": 0.5})
        self._plugins[config.domain] = plugin
    
    def _persist_plugin(self, plugin: DynamicPlugin):
        """Save a dynamic plugin to disk."""
        import os
        
        if not self.auto_persist:
            return
        
        if not plugin.config.is_dynamic:
            return  # Don't persist static plugins
        
        try:
            plugins_path = os.path.join(self.storage_path, "plugins")
            os.makedirs(plugins_path, exist_ok=True)
            
            filepath = os.path.join(plugins_path, f"{plugin.domain}.json")
            with open(filepath, 'w') as f:
                json.dump(plugin.serialize(), f, indent=2, default=str)
            
            logger.info(f"[PluginFactory] Persisted plugin: {plugin.domain}")
        except Exception as e:
            logger.warning(f"[PluginFactory] Failed to persist plugin {plugin.domain}: {e}")
    
    def _persist_knowledge_base(self):
        """Save knowledge base to disk."""
        import os
        
        if not self.auto_persist:
            return
        
        try:
            os.makedirs(self.storage_path, exist_ok=True)
            
            kb_path = os.path.join(self.storage_path, "knowledge_base.json")
            with open(kb_path, 'w') as f:
                json.dump(self.knowledge_base.serialize(), f, indent=2, default=str)
            
            logger.info("[PluginFactory] Persisted knowledge base")
        except Exception as e:
            logger.warning(f"[PluginFactory] Failed to persist knowledge base: {e}")
    
    def _register_static_plugins(self):
        """Register the original static plugins."""
        static_domains = ["vibe_coding", "legal", "healthcare", "financial"]
        for domain in static_domains:
            config = DynamicPluginConfig(
                domain=domain,
                description=f"Static {domain} plugin",
                parent_domains=["common"],
                evidence_requirements={},  # Loaded from original plugins
                enforcement_rules={"default_level": "MODERATE"},
                validation_patterns=[],
                llm_generated_rules=[],
                confidence=0.95,
                is_dynamic=False
            )
            self._plugins[domain] = DynamicPlugin(config, self.knowledge_base)
    
    def detect_domains(self, content: str, context: Dict = None) -> List[str]:
        """
        Semantically detect which domains apply to content.
        Returns list of domains in order of relevance.
        """
        context = context or {}
        domains = []
        scores = {}
        
        # Keywords for domain detection
        domain_keywords = {
            "vibe_coding": ["code", "function", "class", "compile", "test", "api", "bug", "implement"],
            "legal": ["law", "regulation", "compliance", "contract", "statute", "court", "legal", "cfr", "usc"],
            "healthcare": ["medical", "patient", "diagnosis", "treatment", "clinical", "health", "fda", "hipaa"],
            "financial": ["bank", "transaction", "audit", "sox", "financial", "accounting", "sec", "gaap"],
            "aerospace_coding": ["faa", "aircraft", "drone", "flight", "avionics", "do-178", "safety-critical"],
            "pharma": ["drug", "pharmaceutical", "fda approval", "clinical trial", "compound"],
            "insurance": ["claim", "policy", "underwriting", "actuary", "premium"],
        }
        
        content_lower = content.lower()
        explicit_domain = context.get("domain")
        
        for domain, keywords in domain_keywords.items():
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > 0:
                scores[domain] = score
            if explicit_domain and domain == explicit_domain:
                scores[domain] = scores.get(domain, 0) + 10  # Boost explicit
        
        # Sort by score
        domains = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
        
        # If no domains detected, default to general
        if not domains:
            domains = ["general"]
        
        return domains
    
    def get_or_create_plugin(self, domain: str, context: Dict = None) -> DynamicPlugin:
        """
        Get existing plugin or create dynamically.
        """
        if domain in self._plugins:
            return self._plugins[domain]
        
        # Create dynamically
        logger.info(f"[PluginFactory] Creating dynamic plugin for: {domain}")
        
        # Determine parent domains
        parent_domains = self._domain_hierarchy.get(domain, ["common"])
        
        # Gather evidence requirements from parents
        evidence_requirements = {}
        for parent in parent_domains:
            if parent in self._plugins:
                parent_plugin = self._plugins[parent]
                # Inherit parent requirements
                evidence_requirements.update(
                    parent_plugin.config.evidence_requirements
                )
        
        # Generate rules using LLM
        llm_rules = self._generate_llm_rules(domain, context or {})
        
        # Create config
        config = DynamicPluginConfig(
            domain=domain,
            description=f"Dynamically created plugin for {domain}",
            parent_domains=parent_domains,
            evidence_requirements=evidence_requirements,
            enforcement_rules={
                "default_level": "MODERATE",
                "critical_domains": ["healthcare", "legal", "aerospace"]
            },
            validation_patterns=[],
            llm_generated_rules=llm_rules,
            confidence=0.7,  # Start lower for dynamic
            is_dynamic=True
        )
        
        plugin = DynamicPlugin(config, self.knowledge_base)
        self._plugins[domain] = plugin
        
        # Auto-persist to disk
        self._persist_plugin(plugin)
        
        return plugin
    
    def _generate_llm_rules(self, domain: str, context: Dict) -> List[Dict]:
        """
        Generate domain-specific rules using HYBRID approach:
        1. ML patterns - learned from similar domains
        2. AI generation - LLM creates rules from full context
        3. Validation - verify rules make sense
        """
        rules = []
        
        # STEP 1: ML PATTERNS - Get learned patterns from similar domains
        ml_rules = self._get_ml_patterns(domain, context)
        rules.extend(ml_rules)
        
        # STEP 2: AI GENERATION - Query LLM with full context
        if self.llm_registry or self.multi_track:
            ai_rules = self._query_llm_for_rules(domain, context, ml_rules)
            rules.extend(ai_rules)
        else:
            # Fallback to template rules if no LLM available
            fallback_rules = self._get_fallback_rules(domain)
            rules.extend(fallback_rules)
        
        # STEP 3: DEDUPE and VALIDATE
        rules = self._validate_and_dedupe_rules(rules)
        
        logger.info(f"[PluginFactory] Generated {len(rules)} rules for {domain} (ML: {len(ml_rules)}, AI: {len(rules) - len(ml_rules)})")
        
        return rules
    
    def _get_ml_patterns(self, domain: str, context: Dict) -> List[Dict]:
        """
        Extract ML patterns from SharedKnowledgeBase based on domain similarity.
        Uses patterns that have proven successful in similar domains.
        """
        ml_rules = []
        
        # Get applicable learnings from knowledge base
        parent_domains = self._domain_hierarchy.get(domain, ["common"])
        learnings = self.knowledge_base.get_applicable_learnings(
            context={**context, "domain": domain},
            domains=parent_domains,
            pattern_types=["evidence_rule", "enforcement", "validation"]
        )
        
        # Convert learnings to rules
        for learning, relevance in learnings:
            if relevance >= 0.5:  # Only high-relevance patterns
                ml_rules.append({
                    "applies_to": learning.conditions.get("claim_type", "all"),
                    "requirements": learning.action,
                    "description": learning.description,
                    "source": "ml_pattern",
                    "confidence": learning.confidence * relevance,
                    "from_domains": list(learning.source_domains)
                })
        
        return ml_rules
    
    def _query_llm_for_rules(self, domain: str, context: Dict, existing_rules: List[Dict]) -> List[Dict]:
        """
        Query LLM with FULL CONTEXT to generate accurate domain-specific rules.
        Uses multi-track for consensus if available.
        """
        import asyncio
        
        # Build comprehensive prompt with full context
        prompt = self._build_rule_generation_prompt(domain, context, existing_rules)
        
        try:
            # Use multi-track for higher accuracy if available
            if self.multi_track:
                return asyncio.get_event_loop().run_until_complete(
                    self._multi_track_rule_generation(prompt, domain)
                )
            elif self.llm_registry:
                return self._single_llm_rule_generation(prompt, domain)
        except Exception as e:
            logger.warning(f"[PluginFactory] LLM rule generation failed: {e}")
        
        return []
    
    def _build_rule_generation_prompt(self, domain: str, context: Dict, existing_rules: List[Dict]) -> str:
        """
        Build comprehensive prompt with full context for LLM.
        """
        # Get parent domain information
        parent_domains = self._domain_hierarchy.get(domain, ["common"])
        
        # Get sample queries/content from context
        sample_content = context.get("content", context.get("query", ""))[:500]
        importance = context.get("importance", "normal")
        
        # Format existing rules for context
        existing_rules_str = ""
        if existing_rules:
            existing_rules_str = "EXISTING RULES FROM ML PATTERNS:\n"
            for r in existing_rules[:5]:
                existing_rules_str += f"- {r.get('description', 'No description')}\n"
        
        prompt = f"""
TASK: Generate governance rules for a new domain plugin.

DOMAIN: {domain}
PARENT DOMAINS: {', '.join(parent_domains)}
IMPORTANCE LEVEL: {importance}

SAMPLE CONTENT FROM USER:
{sample_content}

{existing_rules_str}

CONTEXT ABOUT THIS DOMAIN:
- What industry regulations apply?
- What evidence/proof is typically required?
- What are common compliance requirements?
- What safety/risk considerations exist?

Please generate 3-5 specific governance rules for this domain.

FORMAT YOUR RESPONSE AS JSON:
[
  {{
    "applies_to": "all" or specific claim type,
    "requirements": {{
      "key": true/false or "value"
    }},
    "description": "Clear description of the rule",
    "enforcement_level": "MODERATE" or "STRICT",
    "rationale": "Why this rule is important for this domain"
  }}
]

IMPORTANT:
- Be specific to the {domain} domain
- Consider compliance, safety, and accuracy requirements
- Include verification/evidence requirements
- Consider industry-standard frameworks or certifications
"""
        return prompt
    
    async def _multi_track_rule_generation(self, prompt: str, domain: str) -> List[Dict]:
        """
        Use multiple LLMs for consensus on rule generation.
        """
        try:
            # Query multiple tracks
            results = await self.multi_track.challenge_all(prompt)
            
            # Extract rules from each response
            all_rules = []
            for result in results:
                rules = self._parse_llm_rules_response(result.get("response", ""))
                all_rules.extend(rules)
            
            # Find consensus rules (appear in 2+ responses)
            return self._find_consensus_rules(all_rules)
        except Exception as e:
            logger.warning(f"[PluginFactory] Multi-track generation failed: {e}")
            return []
    
    def _single_llm_rule_generation(self, prompt: str, domain: str) -> List[Dict]:
        """
        Query single LLM for rule generation.
        """
        try:
            # Get best available model
            model = self.llm_registry.get_best_model()
            response = self.llm_registry.query(model, prompt)
            return self._parse_llm_rules_response(response)
        except Exception as e:
            logger.warning(f"[PluginFactory] Single LLM generation failed: {e}")
            return []
    
    def _parse_llm_rules_response(self, response: str) -> List[Dict]:
        """
        Parse LLM response to extract structured rules.
        """
        import re
        
        rules = []
        
        # Try to find JSON array in response
        json_match = re.search(r'\[[\s\S]*?\]', response)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "description" in item:
                            item["source"] = "ai_generated"
                            rules.append(item)
            except json.JSONDecodeError:
                pass
        
        # If no JSON, try to extract rules from text
        if not rules:
            # Look for numbered items or bullet points
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    rules.append({
                        "applies_to": "all",
                        "requirements": {"compliance": True},
                        "description": line.lstrip('0123456789.-•) '),
                        "source": "ai_generated_text"
                    })
        
        return rules[:5]  # Limit to 5 rules
    
    def _find_consensus_rules(self, all_rules: List[Dict]) -> List[Dict]:
        """
        Find rules that appear in multiple LLM responses (consensus).
        """
        # Group by description similarity
        rule_counts = {}
        for rule in all_rules:
            desc = rule.get("description", "").lower()[:50]
            if desc:
                if desc not in rule_counts:
                    rule_counts[desc] = {"count": 0, "rule": rule}
                rule_counts[desc]["count"] += 1
        
        # Return rules with count >= 2 (consensus)
        consensus_rules = []
        for item in rule_counts.values():
            if item["count"] >= 2:
                item["rule"]["consensus_count"] = item["count"]
                consensus_rules.append(item["rule"])
        
        # If no consensus, return top rules
        if not consensus_rules and all_rules:
            return all_rules[:3]
        
        return consensus_rules
    
    def _validate_and_dedupe_rules(self, rules: List[Dict]) -> List[Dict]:
        """
        Validate rules are well-formed and remove duplicates.
        """
        seen_descriptions = set()
        valid_rules = []
        
        for rule in rules:
            # Must have description
            desc = rule.get("description", "")
            if not desc:
                continue
            
            # Dedupe by description
            desc_key = desc.lower()[:50]
            if desc_key in seen_descriptions:
                continue
            seen_descriptions.add(desc_key)
            
            # Must have requirements or applies_to
            if "requirements" not in rule and "applies_to" not in rule:
                rule["requirements"] = {"validation": True}
            
            valid_rules.append(rule)
        
        return valid_rules
    
    def _get_fallback_rules(self, domain: str) -> List[Dict]:
        """
        Fallback template rules when no LLM available.
        Based on domain keywords.
        """
        rules = []
        
        # Check if domain has known patterns
        if "aerospace" in domain or "defense" in domain:
            rules.append({
                "applies_to": "all",
                "requirements": {
                    "certification_reference": True,
                    "safety_analysis": True,
                    "traceability": True
                },
                "description": "Aerospace/defense requires certification traceability (DO-178C, AS9100)",
                "source": "fallback_template"
            })
        
        if "medical" in domain or "pharma" in domain or "health" in domain:
            rules.append({
                "applies_to": "all",
                "requirements": {
                    "clinical_evidence": True,
                    "regulatory_approval": True,
                    "disclaimer": "not_medical_advice"
                },
                "description": "Medical domains require clinical evidence and regulatory approval (FDA, HIPAA)",
                "source": "fallback_template"
            })
        
        if "legal" in domain or "compliance" in domain:
            rules.append({
                "applies_to": "all",
                "requirements": {
                    "jurisdiction_specified": True,
                    "statute_citation": True
                },
                "description": "Legal domains require jurisdiction specification and statute citations",
                "source": "fallback_template"
            })
        
        if "financial" in domain or "banking" in domain or "fintech" in domain:
            rules.append({
                "applies_to": "all",
                "requirements": {
                    "audit_trail": True,
                    "regulatory_compliance": True
                },
                "description": "Financial domains require audit trails and regulatory compliance (SOX, SEC, GAAP)",
                "source": "fallback_template"
            })
        
        if "insurance" in domain:
            rules.append({
                "applies_to": "all",
                "requirements": {
                    "actuarial_basis": True,
                    "policy_compliance": True
                },
                "description": "Insurance domains require actuarial basis and policy compliance",
                "source": "fallback_template"
            })
        
        return rules
    
    def get_hybrid_plugin(self, domains: List[str], context: Dict = None) -> DynamicPlugin:
        """
        Create a hybrid plugin that combines multiple domains.
        
        Example: Legal issue in aerospace coding
        - Combines: legal, vibe_coding, aerospace
        """
        if len(domains) == 1:
            return self.get_or_create_plugin(domains[0], context)
        
        # Create hybrid domain name
        hybrid_domain = "_".join(sorted(domains))
        
        if hybrid_domain in self._plugins:
            return self._plugins[hybrid_domain]
        
        logger.info(f"[PluginFactory] Creating hybrid plugin: {hybrid_domain}")
        
        # Gather all parent requirements
        evidence_requirements = {}
        all_rules = []
        
        for domain in domains:
            plugin = self.get_or_create_plugin(domain, context)
            evidence_requirements.update(plugin.config.evidence_requirements)
            all_rules.extend(plugin.config.llm_generated_rules)
        
        # Create hybrid config
        config = DynamicPluginConfig(
            domain=hybrid_domain,
            description=f"Hybrid plugin combining: {', '.join(domains)}",
            parent_domains=domains,
            evidence_requirements=evidence_requirements,
            enforcement_rules={
                "default_level": "STRICT",  # Hybrids are stricter
                "reason": "Cross-domain complexity"
            },
            validation_patterns=[],
            llm_generated_rules=all_rules,
            confidence=0.65,  # Hybrids start even lower
            is_dynamic=True
        )
        
        plugin = DynamicPlugin(config, self.knowledge_base)
        self._plugins[hybrid_domain] = plugin
        
        # Auto-persist to disk
        self._persist_plugin(plugin)
        
        return plugin
    
    def share_learning_across_domains(self, learning: Learning, min_success_rate: float = 0.8):
        """
        Promote a successful learning to be shared across domains.
        """
        if learning.success_rate >= min_success_rate:
            # Add to common patterns
            learning.source_domains.add("common")
            self.knowledge_base.add_learning(learning)
            logger.info(f"[PluginFactory] Promoted learning {learning.pattern_id} to common")
            
            # Persist updated knowledge base
            self._persist_knowledge_base()
    
    def get_statistics(self) -> Dict:
        """Return factory statistics."""
        return {
            "total_plugins": len(self._plugins),
            "static_plugins": sum(1 for p in self._plugins.values() if not p.config.is_dynamic),
            "dynamic_plugins": sum(1 for p in self._plugins.values() if p.config.is_dynamic),
            "total_learnings": len(self.knowledge_base._learnings),
            "cross_domain_learnings": len(self.knowledge_base.extract_cross_domain_patterns())
        }
    
    # ===== Learning Interface =====
    
    def record_outcome(self, outcome: Dict) -> None:
        """Record outcome for learning."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback."""
        domain = feedback.get("domain")
        if domain and domain in self._plugins:
            self._plugins[domain].learn_from_feedback(feedback)
    
    def serialize_state(self) -> Dict:
        """Serialize state."""
        return {
            "knowledge_base": self.knowledge_base.serialize(),
            "plugins": {
                domain: plugin.serialize()
                for domain, plugin in self._plugins.items()
                if plugin.config.is_dynamic
            }
        }
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore state."""
        if state.get("knowledge_base"):
            self.knowledge_base.deserialize(state["knowledge_base"])


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

class DynamicPluginOrchestrator:
    """
    Main entry point for dynamic plugin system.
    
    Workflow:
    1. Receives query/context
    2. Detects relevant domains
    3. Gets/creates appropriate plugin(s)
    4. Applies learnings from knowledge base
    5. Returns configured governance
    """
    
    def __init__(
        self,
        llm_registry: Any = None,
        multi_track: Any = None
    ):
        self.knowledge_base = SharedKnowledgeBase()
        self.factory = DynamicPluginFactory(
            self.knowledge_base,
            llm_registry,
            multi_track
        )
        self._outcomes: List[Dict] = []
    
    def process(self, content: str, context: Dict = None) -> Dict:
        """
        Process content and return governance configuration.
        """
        context = context or {}
        
        # Step 1: Detect domains
        domains = self.factory.detect_domains(content, context)
        
        # Step 2: Get or create plugin
        if len(domains) > 1 and context.get("combine_domains", True):
            plugin = self.factory.get_hybrid_plugin(domains, context)
        else:
            plugin = self.factory.get_or_create_plugin(domains[0], context)
        
        # Step 3: Get applicable learnings
        learnings = self.knowledge_base.get_applicable_learnings(
            context={**context, "content": content[:500]},
            domains=domains
        )
        
        # Step 4: Build governance config
        governance = {
            "domain": plugin.domain,
            "detected_domains": domains,
            "plugin_type": "dynamic" if plugin.config.is_dynamic else "static",
            "parent_domains": plugin.config.parent_domains,
            "enforcement_level": plugin.get_enforcement_level(context),
            "proof_requirements": plugin.get_proof_requirements(
                context.get("claim_type", "general")
            ),
            "applicable_learnings": [
                {
                    "id": l.pattern_id,
                    "description": l.description,
                    "relevance": r
                }
                for l, r in learnings[:10]
            ],
            "confidence": plugin.config.confidence
        }
        
        return governance
    
    def validate_with_learnings(
        self,
        evidence: List[str],
        domains: List[str],
        context: Dict = None
    ) -> Tuple[bool, List[str]]:
        """Validate evidence using all applicable learnings."""
        context = context or {}
        all_issues = []
        
        # Get plugin
        if len(domains) > 1:
            plugin = self.factory.get_hybrid_plugin(domains, context)
        else:
            plugin = self.factory.get_or_create_plugin(domains[0], context)
        
        # Validate
        is_valid, issues = plugin.validate_evidence(evidence, context)
        all_issues.extend(issues)
        
        return len(all_issues) == 0, all_issues
    
    def contribute_learning(
        self,
        domain: str,
        pattern_type: str,
        description: str,
        conditions: Dict,
        action: Dict,
        confidence: float = 0.7
    ) -> str:
        """Allow external contribution of learnings."""
        learning = Learning(
            pattern_id=f"{domain}_{hashlib.md5(description.encode()).hexdigest()[:8]}",
            pattern_type=pattern_type,
            description=description,
            conditions=conditions,
            action=action,
            source_domains={domain},
            confidence=confidence
        )
        return self.knowledge_base.add_learning(learning)
    
    # ===== Learning Interface =====
    
    def record_outcome(self, outcome: Dict) -> None:
        """Record outcome."""
        self._outcomes.append(outcome)
        self.factory.record_outcome(outcome)
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback."""
        self.factory.learn_from_feedback(feedback)
    
    def get_statistics(self) -> Dict:
        """Return statistics."""
        return {
            "factory": self.factory.get_statistics(),
            "total_outcomes": len(self._outcomes),
            "knowledge_base_size": len(self.knowledge_base._learnings)
        }
    
    def serialize_state(self) -> Dict:
        """Serialize state."""
        return self.factory.serialize_state()
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore state."""
        self.factory.deserialize_state(state)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DYNAMIC PLUGIN SYSTEM (NOVEL-54) - DEMO")
    print("=" * 70)
    
    orchestrator = DynamicPluginOrchestrator()
    
    # Test 1: Simple coding query
    print("\n[TEST 1] Simple Coding Query")
    print("-" * 60)
    result = orchestrator.process(
        content="Is this function implementation correct?",
        context={"claim_type": "completion"}
    )
    print(f"  Detected: {result['detected_domains']}")
    print(f"  Plugin: {result['domain']} ({result['plugin_type']})")
    print(f"  Enforcement: {result['enforcement_level']}")
    
    # Test 2: Cross-domain query (aerospace + legal + coding)
    print("\n[TEST 2] Cross-Domain: Aerospace + Legal + Coding")
    print("-" * 60)
    result = orchestrator.process(
        content="Is this drone control code compliant with FAA CFR regulations?",
        context={"claim_type": "compliance"}
    )
    print(f"  Detected: {result['detected_domains']}")
    print(f"  Plugin: {result['domain']} ({result['plugin_type']})")
    print(f"  Parents: {result['parent_domains']}")
    print(f"  Enforcement: {result['enforcement_level']}")
    print(f"  Learnings applied: {len(result['applicable_learnings'])}")
    for l in result['applicable_learnings'][:3]:
        print(f"    - {l['description'][:50]}... (relevance: {l['relevance']:.2f})")
    
    # Test 3: Medical + Coding
    print("\n[TEST 3] Cross-Domain: Medical Device Code")
    print("-" * 60)
    result = orchestrator.process(
        content="Does this patient monitoring code meet FDA safety requirements?",
        context={"importance": "critical"}
    )
    print(f"  Detected: {result['detected_domains']}")
    print(f"  Plugin: {result['domain']} ({result['plugin_type']})")
    print(f"  Enforcement: {result['enforcement_level']}")
    
    # Test 4: Show knowledge base growth
    print("\n[TEST 4] Knowledge Base Status")
    print("-" * 60)
    stats = orchestrator.get_statistics()
    print(f"  Total plugins: {stats['factory']['total_plugins']}")
    print(f"  Static: {stats['factory']['static_plugins']}")
    print(f"  Dynamic: {stats['factory']['dynamic_plugins']}")
    print(f"  Shared learnings: {stats['knowledge_base_size']}")
    print(f"  Cross-domain learnings: {stats['factory']['cross_domain_learnings']}")
    
    print("\n" + "=" * 70)
    print("SYSTEM READY - Plugins dynamically created and sharing learnings")
    print("=" * 70)

