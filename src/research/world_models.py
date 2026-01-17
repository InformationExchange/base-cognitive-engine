"""
BAIS World Models Module
Causal reasoning and forward prediction

This module enables BAIS to:
1. Extract causal relationships from text
2. Model cause-effect chains
3. Generate predictions based on causal models
4. Analyze counterfactual scenarios ("what if")

Patent Claims: R&D Invention 3
Proprietary IP - 100% owned by Invitas Inc.
"""

import re
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


class CausalRelationType(str, Enum):
    """Types of causal relationships."""
    CAUSES = "causes"               # A causes B
    ENABLES = "enables"             # A enables B (necessary but not sufficient)
    PREVENTS = "prevents"           # A prevents B
    CORRELATES = "correlates"       # A correlates with B (not causal)
    CONTRIBUTES = "contributes"     # A contributes to B (partial cause)


class TemporalOrder(str, Enum):
    """Temporal ordering of events."""
    BEFORE = "before"
    DURING = "during"
    AFTER = "after"
    SIMULTANEOUS = "simultaneous"
    UNKNOWN = "unknown"


class PredictionConfidence(str, Enum):
    """Confidence levels for predictions."""
    HIGH = "high"           # > 0.8
    MEDIUM = "medium"       # 0.5 - 0.8
    LOW = "low"             # 0.3 - 0.5
    SPECULATIVE = "speculative"  # < 0.3


@dataclass
class CausalRelationship:
    """Represents a causal relationship."""
    relationship_id: str
    cause: str
    effect: str
    relation_type: CausalRelationType
    strength: float             # 0-1 strength of relationship
    temporal_order: TemporalOrder
    evidence: str
    confidence: float


@dataclass
class CausalChain:
    """A chain of causal relationships."""
    chain_id: str
    steps: List[CausalRelationship]
    total_confidence: float     # Product of confidences
    start_event: str
    end_event: str
    length: int


@dataclass
class Prediction:
    """A predicted outcome."""
    prediction_id: str
    scenario: str               # The scenario being predicted
    prediction: str             # The predicted outcome
    confidence: PredictionConfidence
    confidence_score: float
    supporting_causes: List[str]
    assumptions: List[str]
    risks: List[str]            # What could make prediction wrong


@dataclass
class CounterfactualAnalysis:
    """Analysis of a counterfactual ("what if") scenario."""
    counterfactual_id: str
    original_event: str
    modified_event: str         # The "what if"
    original_outcome: str
    predicted_outcome: str
    difference_magnitude: str   # "minor", "moderate", "major"
    key_factors_affected: List[str]
    confidence: float


@dataclass
class CausalPrediction:
    """Complete world model analysis result."""
    # Causal extraction
    causal_relationships: List[CausalRelationship]
    causal_chains: List[CausalChain]
    
    # Predictions
    predictions: List[Prediction]
    primary_prediction: Optional[Prediction]
    
    # Counterfactual analysis
    counterfactuals: List[CounterfactualAnalysis]
    
    # Model quality
    model_completeness: float   # How complete the causal model is
    prediction_reliability: float
    causal_coverage: float      # What % of claims have causal backing
    
    # Identified gaps
    missing_causes: List[str]
    ungrounded_effects: List[str]
    
    # Meta
    processing_time_ms: float
    timestamp: float
    warnings: List[str] = field(default_factory=list)


class WorldModelsModule:
    """
    World Models cognitive module for BAIS.
    
    Implements:
    - Causal relationship extraction
    - Causal chain building
    - Forward prediction
    - Counterfactual analysis
    
    Uses pattern-based causal reasoning (no simulation engine required).
    
    Integration:
    - Triggered by Query Router for prediction/planning queries
    - Particularly important for financial, strategic, policy domains
    - Reports to Fact-Checking for prediction verification
    """
    
    # Causal language patterns - ENHANCED for better extraction
    CAUSAL_PATTERNS = {
        CausalRelationType.CAUSES: [
            r'(.+?)\s+(?:causes?|leads?\s+to|results?\s+in|produces?|creates?)\s+(.+)',
            r'(.+?)\s+(?:is\s+the\s+cause\s+of|is\s+responsible\s+for)\s+(.+)',
            r'because\s+of\s+(.+?),\s+(.+)',
            r'(.+?)\s+(?:therefore|thus|hence|consequently|so)\s+(.+)',
            # ADDED patterns
            r'(.+?)\s+(?:brings?\s+about|triggers?|sparks?|generates?)\s+(.+)',
            r'(.+?)\s+(?:gives?\s+rise\s+to|brings?\s+on)\s+(.+)',
            r'(.+?)\s+(?:induces?|prompts?|provokes?)\s+(.+)',
            r'due\s+to\s+(.+?),\s+(.+)',
            r'as\s+a\s+result\s+of\s+(.+?),\s+(.+)',
            r'(.+?)\s+makes?\s+(.+)\s+(?:happen|occur|increase|decrease)',
            # MORE direct causal verbs
            r'(.+?)\s+(?:destroys?|damages?|harms?|ruins?)\s+(.+)',
            r'(.+?)\s+(?:increases?|decreases?|raises?|lowers?)\s+(.+)',
            r'(.+?)\s+(?:improves?|worsens?|affects?|impacts?)\s+(.+)',
        ],
        CausalRelationType.ENABLES: [
            r'(.+?)\s+(?:enables?|allows?|permits?|makes?\s+possible)\s+(.+)',
            r'(.+?)\s+(?:is\s+necessary\s+for|is\s+required\s+for)\s+(.+)',
            # ADDED patterns
            r'(.+?)\s+(?:facilitates?|supports?|helps?)\s+(.+)',
            r'(.+?)\s+(?:empowers?|equips?)\s+(.+)',
        ],
        CausalRelationType.PREVENTS: [
            r'(.+?)\s+(?:prevents?|stops?|blocks?|inhibits?|avoids?)\s+(.+)',
            r'without\s+(.+?),\s+(.+)\s+would\s+not',
            # ADDED patterns
            r'(.+?)\s+(?:hinders?|impedes?|restricts?|limits?)\s+(.+)',
            r'(.+?)\s+(?:eliminates?|removes?|reduces?)\s+(.+)',
        ],
        CausalRelationType.CONTRIBUTES: [
            r'(.+?)\s+(?:contributes?\s+to|helps?\s+with|influences?)\s+(.+)',
            r'(.+?)\s+(?:is\s+a\s+factor\s+in|plays?\s+a\s+role\s+in)\s+(.+)',
            # ADDED patterns
            r'(.+?)\s+(?:impacts?|affects?)\s+(.+)',
            r'(.+?)\s+(?:has\s+(?:an?)\s+effect\s+on)\s+(.+)',
        ],
        CausalRelationType.CORRELATES: [
            r'(.+?)\s+(?:is\s+associated\s+with|correlates?\s+with|is\s+linked\s+to)\s+(.+)',
            r'(.+?)\s+and\s+(.+)\s+tend\s+to\s+occur\s+together',
            # ADDED patterns - important for distinguishing from causation
            r'(.+?)\s+(?:is\s+related\s+to|is\s+connected\s+to)\s+(.+)',
            r'(.+?)\s+(?:goes\s+along\s+with|accompanies?)\s+(.+)',
            r'(?:there\s+is\s+a\s+)?(?:correlation|relationship)\s+between\s+(.+?)\s+and\s+(.+)',
        ],
    }
    
    # Temporal markers
    TEMPORAL_MARKERS = {
        TemporalOrder.BEFORE: [r'\b(before|prior\s+to|preceding|earlier)\b'],
        TemporalOrder.DURING: [r'\b(during|while|as|when)\b'],
        TemporalOrder.AFTER: [r'\b(after|following|subsequent|later)\b'],
        TemporalOrder.SIMULTANEOUS: [r'\b(simultaneously|at\s+the\s+same\s+time|concurrently)\b'],
    }
    
    # Prediction language - ENHANCED with uncertain predictions
    PREDICTION_PATTERNS = [
        # BAIS Enhancement: Expanded prediction markers
        r'\bultimately\b',
        r'\binevitably\b',
        r'\beventually\b',
        r'\bin\s+the\s+(?:long|short)\s+(?:run|term)\b',
        r'\bgoing\s+forward\b',
        r'\bas\s+(?:time|things)\s+progress(?:es)?\b',
        r'\bover\s+time\b',
        r'\bsooner\s+or\s+later\b',
        r'\bdown\s+the\s+(?:road|line)\b',
        r'\bonce\s+(?:this|that|we|they)\b',
        r'\bafter\s+(?:this|that|we|they)\b',
        r'\bif\s+(?:this|that|we|they)\s+continue(?:s)?\b',
        r'\bat\s+this\s+(?:rate|pace)\b',
        r'\btrend(?:s)?\s+(?:show|suggest|indicate)\b',
        r'\bprojection(?:s)?\s+(?:show|suggest|indicate)\b',

        r'(?:will|would|is\s+likely\s+to|is\s+expected\s+to)\s+(.+)',
        r'(?:predict|forecast|expect|anticipate)\s+(?:that\s+)?(.+)',
        r'in\s+the\s+future,?\s+(.+)',
        r'(?:may|might|could)\s+(?:lead\s+to|result\s+in|cause)\s+(.+)',
        # ADDED patterns
        r'(?:this\s+)?will\s+(?:likely\s+)?(?:lead\s+to|result\s+in|cause)\s+(.+)',
        r'(?:going\s+forward|looking\s+ahead|in\s+time),?\s+(.+)',
        r'(?:projected|estimated|forecasted)\s+(?:to\s+)?(.+)',
        r'(?:is\s+set\s+to|is\s+poised\s+to|is\s+on\s+track\s+to)\s+(.+)',
        r'(?:should|ought\s+to)\s+(.+)',
        r'(?:eventually|ultimately|in\s+the\s+long\s+run)\s+(.+)',
        # PHASE 1 FIX: Uncertain predictions (WM-A6)
        r'(?:might|may|could)\s+(.+)',  # "might improve", "could crash"
        r'(?:possibly|perhaps|maybe)\s+(.+)',
        r'(?:there\s+is\s+a\s+chance|it\s+is\s+possible)\s+(?:that\s+)?(.+)',
        r'(?:it\s+)?(?:remains\s+uncertain|is\s+unclear)\s+(?:whether|if)\s+(.+)',
        r'(?:economy|market|situation)\s+(?:might|may|could)\s+(.+)',
    ]
    
    # Counterfactual patterns
    COUNTERFACTUAL_PATTERNS = [
        r'if\s+(.+?)\s+(?:had|were|was)\s+(.+?),?\s+(?:then\s+)?(.+)\s+would',
        r'what\s+if\s+(.+)',
        r'suppose\s+(.+)',
        r'imagine\s+(.+)',
        r'had\s+(.+?)\s+been\s+(.+?),\s+(.+)',
    ]
    
    def __init__(self,
                 min_causal_confidence: float = 0.50,
                 prediction_threshold: float = 0.40):
        """
        Initialize World Models module.
        
        Args:
            min_causal_confidence: Minimum confidence for causal extraction
            prediction_threshold: Threshold for making predictions
        """
        self.min_causal_confidence = min_causal_confidence
        self.prediction_threshold = prediction_threshold
    
    def analyze(self,
                query: str,
                response: str,
                context: Dict[str, Any] = None) -> CausalPrediction:
        """
        Perform complete world model analysis.
        
        Args:
            query: User query
            response: AI-generated response
            context: Additional context (domain, timeframe, etc.)
        
        Returns:
            CausalPrediction with complete analysis
        """
        start_time = time.time()
        context = context or {}
        warnings = []
        
        combined_text = f"{query} {response}"
        
        # 1. Extract causal relationships
        relationships = self._extract_causal_relationships(response)
        
        # 2. Build causal chains
        chains = self._build_causal_chains(relationships)
        
        # 3. Extract predictions
        predictions = self._extract_predictions(response, chains)
        primary = predictions[0] if predictions else None
        
        # 4. Analyze counterfactuals
        counterfactuals = self._analyze_counterfactuals(combined_text, relationships)
        
        # 5. Compute model quality
        completeness = self._compute_model_completeness(relationships, chains)
        reliability = self._compute_prediction_reliability(predictions, relationships)
        coverage = self._compute_causal_coverage(response, relationships)
        
        # 6. Identify gaps
        missing_causes = self._identify_missing_causes(response, relationships)
        ungrounded = self._identify_ungrounded_effects(response, relationships)
        
        if missing_causes:
            warnings.append(f"Missing causal explanations for {len(missing_causes)} effects")
        
        processing_time = (time.time() - start_time) * 1000
        
        return CausalPrediction(
            causal_relationships=relationships,
            causal_chains=chains,
            predictions=predictions,
            primary_prediction=primary,
            counterfactuals=counterfactuals,
            model_completeness=completeness,
            prediction_reliability=reliability,
            causal_coverage=coverage,
            missing_causes=missing_causes,
            ungrounded_effects=ungrounded,
            processing_time_ms=processing_time,
            timestamp=time.time(),
            warnings=warnings
        )
    
    def _extract_causal_relationships(self, text: str) -> List[CausalRelationship]:
        """Extract causal relationships from text - ENHANCED to process by sentence."""
        relationships = []
        rel_id = 0
        
        # Split into sentences for better extraction
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            sentence_lower = sentence.lower()
            
            for rel_type, patterns in self.CAUSAL_PATTERNS.items():
                for pattern in patterns:
                    for match in re.finditer(pattern, sentence_lower):
                        groups = match.groups()
                        if len(groups) >= 2:
                            cause = groups[0].strip()
                            effect = groups[-1].strip()
                            
                            # BAIS-GUIDED: Reduced minimum length from 3 to 1 for variable-style names
                            if len(cause) < 1 or len(effect) < 1:
                                continue
                            
                            # Determine temporal order
                            temporal = self._determine_temporal_order(sentence)
                            
                            # Calculate strength and confidence
                            strength = self._calculate_relationship_strength(
                                cause, effect, rel_type
                            )
                            confidence = self._calculate_causal_confidence(
                                match.group(), rel_type
                            )
                            
                            if confidence >= self.min_causal_confidence:
                                rel = CausalRelationship(
                                    relationship_id=f"R{rel_id}",
                                    cause=cause[:100],
                                    effect=effect[:100],
                                    relation_type=rel_type,
                                    strength=strength,
                                    temporal_order=temporal,
                                    evidence=match.group()[:150],
                                    confidence=confidence
                                )
                                relationships.append(rel)
                                rel_id += 1
        
        return self._deduplicate_relationships(relationships)[:15]
    
    def _determine_temporal_order(self, context: str) -> TemporalOrder:
        """Determine temporal ordering from context."""
        context_lower = context.lower()
        
        for temporal, patterns in self.TEMPORAL_MARKERS.items():
            for pattern in patterns:
                if re.search(pattern, context_lower):
                    return temporal
        
        return TemporalOrder.UNKNOWN
    
    def _calculate_relationship_strength(self,
                                          cause: str,
                                          effect: str,
                                          rel_type: CausalRelationType) -> float:
        """Calculate strength of causal relationship."""
        strength = 0.5  # Base strength
        
        # Direct causation is stronger
        if rel_type == CausalRelationType.CAUSES:
            strength += 0.3
        elif rel_type == CausalRelationType.ENABLES:
            strength += 0.2
        elif rel_type == CausalRelationType.CONTRIBUTES:
            strength += 0.1
        elif rel_type == CausalRelationType.CORRELATES:
            strength -= 0.2  # Correlation is weaker
        
        # Strong language increases strength
        strong_markers = ['directly', 'always', 'definitely', 'certainly', 'must']
        if any(m in f"{cause} {effect}".lower() for m in strong_markers):
            strength += 0.1
        
        return min(max(strength, 0.0), 1.0)
    
    def _calculate_causal_confidence(self,
                                      evidence: str,
                                      rel_type: CausalRelationType) -> float:
        """Calculate confidence in causal extraction."""
        confidence = 0.6  # Base
        
        # Strong causal language increases confidence
        strong_causal = ['causes', 'leads to', 'results in', 'therefore']
        if any(c in evidence.lower() for c in strong_causal):
            confidence += 0.2
        
        # Hedging decreases confidence
        hedges = ['may', 'might', 'could', 'possibly', 'perhaps']
        if any(h in evidence.lower() for h in hedges):
            confidence -= 0.15
        
        # Correlation vs causation awareness
        if rel_type == CausalRelationType.CORRELATES:
            confidence -= 0.1  # Correlation claims are less certain
        
        return min(max(confidence, 0.0), 1.0)
    
    def _deduplicate_relationships(self,
                                    relationships: List[CausalRelationship]) -> List[CausalRelationship]:
        """Remove duplicate relationships."""
        unique = []
        seen = set()
        
        for rel in sorted(relationships, key=lambda r: r.confidence, reverse=True):
            key = f"{rel.cause[:30]}:{rel.effect[:30]}:{rel.relation_type.value}"
            if key not in seen:
                seen.add(key)
                unique.append(rel)
        
        return unique
    
    def _build_causal_chains(self,
                              relationships: List[CausalRelationship]) -> List[CausalChain]:
        """Build causal chains from relationships."""
        chains = []
        
        if len(relationships) < 2:
            return chains
        
        # Build adjacency map
        effect_to_rel = {}
        for rel in relationships:
            effect_key = rel.effect[:30].lower()
            if effect_key not in effect_to_rel:
                effect_to_rel[effect_key] = []
            effect_to_rel[effect_key].append(rel)
        
        # Find chains starting from each relationship
        chain_id = 0
        for start_rel in relationships:
            chain_steps = [start_rel]
            current_effect = start_rel.effect[:30].lower()
            
            # Follow the chain
            visited = {start_rel.cause[:30].lower()}
            while len(chain_steps) < 5:  # Max chain length 5
                # Find next relationship where effect becomes cause
                found_next = False
                for rel in relationships:
                    if rel.cause[:30].lower() in current_effect and rel.cause[:30].lower() not in visited:
                        chain_steps.append(rel)
                        visited.add(rel.cause[:30].lower())
                        current_effect = rel.effect[:30].lower()
                        found_next = True
                        break
                
                if not found_next:
                    break
            
            # Only keep chains with 2+ steps
            if len(chain_steps) >= 2:
                total_conf = 1.0
                for step in chain_steps:
                    total_conf *= step.confidence
                
                chain = CausalChain(
                    chain_id=f"CH{chain_id}",
                    steps=chain_steps,
                    total_confidence=total_conf,
                    start_event=chain_steps[0].cause[:50],
                    end_event=chain_steps[-1].effect[:50],
                    length=len(chain_steps)
                )
                chains.append(chain)
                chain_id += 1
        
        return chains[:5]  # Limit to 5 chains
    
    def _extract_predictions(self,
                              text: str,
                              chains: List[CausalChain]) -> List[Prediction]:
        """Extract predictions from text."""
        predictions = []
        pred_id = 0
        
        text_lower = text.lower()
        
        for pattern in self.PREDICTION_PATTERNS:
            for match in re.finditer(pattern, text_lower):
                prediction_text = match.group(1) if match.groups() else match.group()
                
                # BAIS-GUIDED: Reduced minimum length from 10 to 5
                if len(prediction_text) < 5:
                    continue
                
                # Find supporting causes from causal chains
                supporting = self._find_supporting_causes(prediction_text, chains)
                
                # Calculate confidence
                confidence_score = self._calculate_prediction_confidence(
                    prediction_text, supporting
                )
                
                # Determine confidence level
                if confidence_score >= 0.7:
                    conf_level = PredictionConfidence.HIGH
                elif confidence_score >= 0.5:
                    conf_level = PredictionConfidence.MEDIUM
                elif confidence_score >= 0.3:
                    conf_level = PredictionConfidence.LOW
                else:
                    conf_level = PredictionConfidence.SPECULATIVE
                
                # Identify assumptions and risks
                assumptions = self._identify_assumptions(text, prediction_text)
                risks = self._identify_prediction_risks(prediction_text)
                
                pred = Prediction(
                    prediction_id=f"P{pred_id}",
                    scenario=f"Given current conditions",
                    prediction=prediction_text[:150],
                    confidence=conf_level,
                    confidence_score=confidence_score,
                    supporting_causes=supporting[:3],
                    assumptions=assumptions[:3],
                    risks=risks[:3]
                )
                predictions.append(pred)
                pred_id += 1
        
        return predictions[:5]
    
    def _find_supporting_causes(self,
                                 prediction: str,
                                 chains: List[CausalChain]) -> List[str]:
        """Find causal support for prediction."""
        supporting = []
        prediction_lower = prediction.lower()
        
        for chain in chains:
            # Check if prediction relates to chain end
            if any(word in chain.end_event.lower() for word in prediction_lower.split()[:3]):
                supporting.append(chain.start_event)
        
        return supporting
    
    def _calculate_prediction_confidence(self,
                                          prediction: str,
                                          supporting: List[str]) -> float:
        """Calculate prediction confidence."""
        confidence = 0.4  # Base
        
        # More causal support increases confidence
        confidence += min(len(supporting) * 0.15, 0.3)
        
        # Hedging language decreases confidence
        hedges = ['may', 'might', 'could', 'possibly', 'perhaps', 'uncertain']
        if any(h in prediction.lower() for h in hedges):
            confidence -= 0.15
        
        # Strong language increases confidence
        strong = ['will', 'definitely', 'certainly', 'must']
        if any(s in prediction.lower() for s in strong):
            confidence += 0.1
        
        return min(max(confidence, 0.0), 1.0)
    
    def _identify_assumptions(self, text: str, prediction: str) -> List[str]:
        """Identify assumptions underlying prediction."""
        assumptions = []
        
        # Look for assumption markers
        assumption_patterns = [
            r'assuming\s+(.+?)(?:\.|,)',
            r'given\s+that\s+(.+?)(?:\.|,)',
            r'if\s+(.+?)\s+remains',
            r'provided\s+that\s+(.+?)(?:\.|,)',
        ]
        
        text_lower = text.lower()
        for pattern in assumption_patterns:
            for match in re.finditer(pattern, text_lower):
                assumptions.append(match.group(1)[:80])
        
        # Default assumptions
        if not assumptions:
            assumptions = ["Current conditions continue", "No major disruptions occur"]
        
        return assumptions
    
    def _identify_prediction_risks(self, prediction: str) -> List[str]:
        """Identify what could invalidate prediction."""
        risks = []
        
        # Generic risk categories
        risk_keywords = {
            'market': "Market conditions could change",
            'economic': "Economic factors may shift",
            'regulatory': "Regulatory changes possible",
            'technology': "Technology disruption risk",
            'competitive': "Competitive landscape may change",
        }
        
        prediction_lower = prediction.lower()
        for keyword, risk in risk_keywords.items():
            if keyword in prediction_lower:
                risks.append(risk)
        
        if not risks:
            risks = ["Unforeseen external factors", "Data limitations"]
        
        return risks[:3]
    
    def _analyze_counterfactuals(self,
                                  text: str,
                                  relationships: List[CausalRelationship]) -> List[CounterfactualAnalysis]:
        """Analyze counterfactual scenarios."""
        counterfactuals = []
        cf_id = 0
        
        text_lower = text.lower()
        
        for pattern in self.COUNTERFACTUAL_PATTERNS:
            for match in re.finditer(pattern, text_lower):
                groups = match.groups()
                
                if len(groups) >= 1:
                    modified = groups[0][:100]
                    predicted = groups[-1][:100] if len(groups) > 1 else "Different outcome"
                    
                    # Estimate magnitude of change
                    magnitude = self._estimate_counterfactual_magnitude(
                        modified, relationships
                    )
                    
                    # Find affected factors
                    affected = self._find_affected_factors(modified, relationships)
                    
                    cf = CounterfactualAnalysis(
                        counterfactual_id=f"CF{cf_id}",
                        original_event="Actual situation",
                        modified_event=modified,
                        original_outcome="Current outcome",
                        predicted_outcome=predicted,
                        difference_magnitude=magnitude,
                        key_factors_affected=affected[:3],
                        confidence=0.5  # Counterfactuals are inherently uncertain
                    )
                    counterfactuals.append(cf)
                    cf_id += 1
        
        return counterfactuals[:3]
    
    def _estimate_counterfactual_magnitude(self,
                                            modified: str,
                                            relationships: List[CausalRelationship]) -> str:
        """Estimate how big a change the counterfactual would cause."""
        # Count how many causal chains would be affected
        affected_count = 0
        modified_lower = modified.lower()
        
        for rel in relationships:
            if any(word in rel.cause.lower() for word in modified_lower.split()[:3]):
                affected_count += 1
        
        if affected_count >= 3:
            return "major"
        elif affected_count >= 1:
            return "moderate"
        else:
            return "minor"
    
    def _find_affected_factors(self,
                                modified: str,
                                relationships: List[CausalRelationship]) -> List[str]:
        """Find factors affected by counterfactual."""
        affected = []
        modified_lower = modified.lower()
        
        for rel in relationships:
            if any(word in rel.cause.lower() for word in modified_lower.split()[:3]):
                affected.append(rel.effect[:50])
        
        return affected
    
    def _compute_model_completeness(self,
                                     relationships: List[CausalRelationship],
                                     chains: List[CausalChain]) -> float:
        """Compute how complete the causal model is."""
        if not relationships:
            return 0.0
        
        # Score based on relationship diversity and chain presence
        rel_types = set(r.relation_type for r in relationships)
        type_coverage = len(rel_types) / 5  # 5 types
        
        chain_bonus = min(len(chains) / 3, 0.3)  # Bonus for chains
        
        return min(type_coverage * 0.7 + chain_bonus, 1.0)
    
    def _compute_prediction_reliability(self,
                                         predictions: List[Prediction],
                                         relationships: List[CausalRelationship]) -> float:
        """Compute reliability of predictions."""
        if not predictions:
            return 0.0
        
        # Average confidence of predictions
        avg_conf = sum(p.confidence_score for p in predictions) / len(predictions)
        
        # Bonus for causal backing
        backed = sum(1 for p in predictions if p.supporting_causes) / len(predictions)
        
        return avg_conf * 0.6 + backed * 0.4
    
    def _compute_causal_coverage(self,
                                  text: str,
                                  relationships: List[CausalRelationship]) -> float:
        """Compute what % of claims have causal explanation."""
        # Count sentences with claims
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 10]
        if not sentences:
            return 0.0
        
        # Count sentences covered by relationships
        covered = 0
        for sent in sentences:
            sent_lower = sent.lower()
            for rel in relationships:
                if rel.cause.lower() in sent_lower or rel.effect.lower() in sent_lower:
                    covered += 1
                    break
        
        return covered / len(sentences)
    
    def _identify_missing_causes(self,
                                  text: str,
                                  relationships: List[CausalRelationship]) -> List[str]:
        """Identify effects without causal explanation."""
        missing = []
        
        # Find claims that should have causes
        effect_patterns = [
            r'(.+?)\s+(?:increased|decreased|changed|happened)',
            r'(.+?)\s+(?:was|is|became)\s+(?:affected|impacted)',
        ]
        
        text_lower = text.lower()
        all_causes = set(r.cause.lower()[:30] for r in relationships)
        
        for pattern in effect_patterns:
            for match in re.finditer(pattern, text_lower):
                effect = match.group(1)[:50]
                # Check if this effect has a documented cause
                has_cause = any(
                    effect.lower() in r.effect.lower() 
                    for r in relationships
                )
                if not has_cause and effect not in missing:
                    missing.append(effect)
        
        return missing[:5]
    
    def _identify_ungrounded_effects(self,
                                      text: str,
                                      relationships: List[CausalRelationship]) -> List[str]:
        """Identify claimed effects without grounding."""
        ungrounded = []
        
        for rel in relationships:
            # Check if effect has any evidence in original text
            if rel.effect.lower()[:20] not in text.lower():
                ungrounded.append(rel.effect[:50])
        
        return ungrounded[:5]


# Self-test
if __name__ == "__main__":
    module = WorldModelsModule()
    
    print("=" * 60)
    print("WORLD MODELS MODULE TEST")
    print("=" * 60)
    
    query = "What will happen if interest rates increase?"
    response = """
    If interest rates increase, borrowing costs will rise, which leads to reduced consumer spending.
    Reduced spending causes economic slowdown, which may result in lower corporate profits.
    Higher rates also cause housing prices to decrease as mortgages become more expensive.
    The stock market will likely decline in response to these factors.
    """
    
    result = module.analyze(query, response)
    
    print(f"\nCausal Relationships: {len(result.causal_relationships)}")
    for rel in result.causal_relationships[:3]:
        print(f"  - {rel.cause[:40]} --{rel.relation_type.value}--> {rel.effect[:40]}")
        print(f"    Confidence: {rel.confidence:.2f}, Strength: {rel.strength:.2f}")
    
    print(f"\nCausal Chains: {len(result.causal_chains)}")
    for chain in result.causal_chains[:2]:
        print(f"  - {chain.start_event[:30]} ... {chain.end_event[:30]} ({chain.length} steps)")
    
    print(f"\nPredictions: {len(result.predictions)}")
    if result.primary_prediction:
        pred = result.primary_prediction
        print(f"  Primary: {pred.prediction[:60]}...")
        print(f"  Confidence: {pred.confidence.value} ({pred.confidence_score:.2f})")
        print(f"  Supporting: {pred.supporting_causes}")
        print(f"  Risks: {pred.risks}")
    
    print(f"\nCounterfactuals: {len(result.counterfactuals)}")
    for cf in result.counterfactuals[:2]:
        print(f"  - If {cf.modified_event[:40]}...")
        print(f"    Magnitude: {cf.difference_magnitude}")
    
    print(f"\nModel Quality:")
    print(f"  Completeness: {result.model_completeness:.2f}")
    print(f"  Prediction Reliability: {result.prediction_reliability:.2f}")
    print(f"  Causal Coverage: {result.causal_coverage:.2%}")
    print(f"  Processing Time: {result.processing_time_ms:.1f}ms")

