"""
BAIS Decision Quality Enhancer

Improves the quality of decision-making in LLM outputs.
Ensures recommendations are well-reasoned and appropriate.

Patent Alignment: PPA3 - Superior Decisioning and Judgment
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set
import re


class DecisionQuality(Enum):
    """Quality level of decision/recommendation"""
    EXCELLENT = "excellent"     # Well-reasoned, comprehensive
    GOOD = "good"               # Solid reasoning, some gaps
    ADEQUATE = "adequate"       # Basic reasoning present
    POOR = "poor"               # Weak or missing reasoning
    DANGEROUS = "dangerous"     # Potentially harmful recommendation


class DecisionFlaw(Enum):
    """Types of decision flaws"""
    MISSING_ALTERNATIVES = "missing_alternatives"
    MISSING_RISKS = "missing_risks"
    OVERCONFIDENCE = "overconfidence"
    INAPPROPRIATE_FOR_CONTEXT = "inappropriate_for_context"
    MISSING_PERSONALIZATION = "missing_personalization"
    CONFLICT_OF_INTEREST = "conflict_of_interest"
    TIME_INSENSITIVITY = "time_insensitivity"
    COST_BLINDNESS = "cost_blindness"


@dataclass
class Decision:
    """A decision or recommendation extracted from text"""
    recommendation: str
    target_audience: str
    context: str
    confidence_level: float
    alternatives_mentioned: List[str]
    risks_mentioned: List[str]
    caveats_mentioned: List[str]


@dataclass
class DecisionAssessment:
    """Assessment of a decision's quality"""
    decision: Decision
    quality: DecisionQuality
    flaws: List[DecisionFlaw]
    missing_elements: List[str]
    improvement_suggestions: List[str]
    enhanced_recommendation: str


@dataclass
class DecisionEnhancementResult:
    """Complete result of decision enhancement"""
    decisions_found: List[Decision]
    assessments: List[DecisionAssessment]
    overall_quality: DecisionQuality
    enhanced_text: str
    quality_improvement: float
    key_enhancements: List[str]


class DecisionQualityEnhancer:
    """
    Enhances decision quality in LLM outputs.
    
    Key capabilities:
    1. Extract decisions and recommendations
    2. Assess decision quality
    3. Identify missing elements
    4. Generate improved decisions
    """
    
    # Patterns for extracting decisions
    DECISION_MARKERS = [
        r'(?:I|we)\s+recommend',
        r'you\s+should',
        r'(?:the\s+)?best\s+(?:option|choice|approach)',
        r'(?:I|we)\s+suggest',
        r'(?:I|we)\s+advise',
        r'(?:my|our)\s+recommendation',
        r'consider\s+(?:doing|using)',
        # Imperative commands (common in manipulative advice)
        r'put\s+(?:all\s+)?(?:your|the)\s+money',
        r'invest\s+(?:in|your)',
        r'buy\s+(?:this|these|now)',
        r'don\'?t\s+(?:listen|trust|waste)',
        r'trust\s+me',
        r'this\s+is\s+(?:your|the)\s+(?:best|last|only)',
    ]
    
    # Risk domains that require extra caution
    HIGH_RISK_DOMAINS = {
        'medical': ['treatment', 'medication', 'diagnosis', 'symptom', 'doctor', 'health'],
        'financial': ['invest', 'money', 'stock', 'retirement', 'savings', 'loan'],
        'legal': ['legal', 'law', 'sue', 'court', 'contract', 'lawyer'],
        'safety': ['dangerous', 'risk', 'harm', 'emergency', 'crisis']
    }
    
    # Required elements by domain
    DOMAIN_REQUIREMENTS = {
        'medical': {
            'caveats': ['consult a healthcare provider', 'professional advice', 
                       'individual circumstances', 'not medical advice'],
            'risks': True,
            'alternatives': True
        },
        'financial': {
            'caveats': ['financial advisor', 'risk tolerance', 'individual circumstances',
                       'not financial advice'],
            'risks': True,
            'alternatives': True
        },
        'legal': {
            'caveats': ['consult an attorney', 'legal counsel', 'not legal advice',
                       'jurisdiction varies'],
            'risks': True,
            'alternatives': True
        },
        'general': {
            'caveats': ['individual circumstances may vary'],
            'risks': False,
            'alternatives': False
        }
    }
    
    # Quality improvement templates
    ENHANCEMENT_TEMPLATES = {
        'add_alternatives': "\n\nAlternatively, you might consider: {alternatives}",
        'add_risks': "\n\nPotential risks to be aware of: {risks}",
        'add_caveat': "\n\nNote: {caveat}",
        'soften_confidence': "Based on the information provided, {recommendation}",
        'add_personalization': "Given your specific situation ({context}), {recommendation}"
    }
    
    def __init__(self):
        self.enhancement_log: List[str] = []
    
    def enhance(self, text: str, user_context: Optional[Dict] = None) -> DecisionEnhancementResult:
        """
        Enhance decision quality in text.
        
        Args:
            text: Text containing decisions/recommendations
            user_context: Optional context about the user (age, risk tolerance, etc.)
            
        Returns:
            DecisionEnhancementResult with enhanced recommendations
        """
        self.enhancement_log = []
        self._log("Starting decision quality enhancement")
        
        # Detect domain
        domain = self._detect_domain(text)
        self._log(f"Detected domain: {domain}")
        
        # Step 1: Extract decisions
        decisions = self._extract_decisions(text, user_context)
        self._log(f"Found {len(decisions)} decisions/recommendations")
        
        # Step 2: Assess each decision
        assessments = []
        for decision in decisions:
            assessment = self._assess_decision(decision, domain)
            assessments.append(assessment)
        
        # Step 3: Calculate overall quality
        overall_quality = self._calculate_overall_quality(assessments)
        
        # Step 4: Generate enhanced text
        enhanced_text = self._generate_enhanced_text(text, assessments, domain)
        
        # Calculate improvement
        original_score = self._quality_to_score(self._calculate_overall_quality([
            DecisionAssessment(d, DecisionQuality.ADEQUATE, [], [], [], d.recommendation)
            for d in decisions
        ]))
        enhanced_score = self._quality_to_score(overall_quality)
        
        return DecisionEnhancementResult(
            decisions_found=decisions,
            assessments=assessments,
            overall_quality=overall_quality,
            enhanced_text=enhanced_text,
            quality_improvement=enhanced_score - original_score,
            key_enhancements=self._summarize_enhancements(assessments)
        )
    
    def _log(self, message: str):
        """Log enhancement step"""
        self.enhancement_log.append(message)
    
    def _detect_domain(self, text: str) -> str:
        """Detect the domain of the text"""
        text_lower = text.lower()
        
        domain_scores = {}
        for domain, keywords in self.HIGH_RISK_DOMAINS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            domain_scores[domain] = score
        
        if domain_scores:
            max_domain = max(domain_scores.items(), key=lambda x: x[1])
            if max_domain[1] > 0:
                return max_domain[0]
        
        return 'general'
    
    def _extract_decisions(self, text: str, 
                           user_context: Optional[Dict]) -> List[Decision]:
        """Extract decisions and recommendations from text"""
        decisions = []
        
        for pattern in self.DECISION_MARKERS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Get surrounding context
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 200)
                context = text[start:end]
                
                # Extract the recommendation sentence
                sentence_match = re.search(
                    rf'{pattern}\s+([^.!?]+[.!?])',
                    text[match.start():],
                    re.IGNORECASE
                )
                
                if sentence_match:
                    recommendation = sentence_match.group(0).strip()
                    
                    # Extract alternatives mentioned
                    alternatives = self._extract_alternatives(text)
                    
                    # Extract risks mentioned
                    risks = self._extract_risks(text)
                    
                    # Extract caveats
                    caveats = self._extract_caveats(text)
                    
                    decisions.append(Decision(
                        recommendation=recommendation,
                        target_audience=self._infer_audience(context, user_context),
                        context=context,
                        confidence_level=self._assess_confidence(recommendation),
                        alternatives_mentioned=alternatives,
                        risks_mentioned=risks,
                        caveats_mentioned=caveats
                    ))
        
        return decisions
    
    def _extract_alternatives(self, text: str) -> List[str]:
        """Extract mentioned alternatives"""
        alternatives = []
        
        patterns = [
            r'alternatively,?\s+([^.]+)',
            r'another option\s+(?:is|would be)\s+([^.]+)',
            r'you could also\s+([^.]+)',
            r'other options include\s+([^.]+)',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                alternatives.append(match.group(1).strip())
        
        return alternatives
    
    def _extract_risks(self, text: str) -> List[str]:
        """Extract mentioned risks"""
        risks = []
        
        patterns = [
            r'risk(?:s)?\s+(?:of|include)\s+([^.]+)',
            r'downside\s+(?:is|includes)\s+([^.]+)',
            r'potential(?:ly)?\s+(?:harmful|dangerous|risky)\s+([^.]+)',
            r'be aware\s+(?:that|of)\s+([^.]+)',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                risks.append(match.group(1).strip())
        
        return risks
    
    def _extract_caveats(self, text: str) -> List[str]:
        """Extract mentioned caveats"""
        caveats = []
        
        patterns = [
            r'however,?\s+([^.]+)',
            r'please\s+(?:note|consult)\s+([^.]+)',
            r'this is not\s+([^.]+advice)',
            r'individual\s+(?:circumstances|results)\s+([^.]+)',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                caveats.append(match.group(0).strip())
        
        return caveats
    
    def _infer_audience(self, context: str, 
                        user_context: Optional[Dict]) -> str:
        """Infer target audience from context"""
        if user_context:
            parts = []
            if 'age' in user_context:
                parts.append(f"age {user_context['age']}")
            if 'risk_tolerance' in user_context:
                parts.append(f"{user_context['risk_tolerance']} risk tolerance")
            return ', '.join(parts) if parts else "general"
        
        # Try to infer from text
        if 'retire' in context.lower():
            return "retiree/pre-retiree"
        if 'student' in context.lower():
            return "student"
        if 'business' in context.lower():
            return "business professional"
        
        return "general audience"
    
    def _assess_confidence(self, text: str) -> float:
        """Assess confidence level of recommendation"""
        text_lower = text.lower()
        
        # High confidence
        if any(w in text_lower for w in ['definitely', 'certainly', 'must', 'should']):
            return 0.9
        
        # Moderate confidence
        if any(w in text_lower for w in ['recommend', 'suggest', 'consider']):
            return 0.7
        
        # Low confidence
        if any(w in text_lower for w in ['might', 'could', 'perhaps']):
            return 0.5
        
        return 0.7
    
    def _assess_decision(self, decision: Decision, domain: str) -> DecisionAssessment:
        """Assess the quality of a decision"""
        flaws = []
        missing = []
        requirements = self.DOMAIN_REQUIREMENTS.get(domain, self.DOMAIN_REQUIREMENTS['general'])
        
        # Check for alternatives
        if requirements.get('alternatives') and not decision.alternatives_mentioned:
            flaws.append(DecisionFlaw.MISSING_ALTERNATIVES)
            missing.append("Alternative options")
        
        # Check for risks
        if requirements.get('risks') and not decision.risks_mentioned:
            flaws.append(DecisionFlaw.MISSING_RISKS)
            missing.append("Risk acknowledgment")
        
        # Check for required caveats
        has_required_caveat = any(
            any(req.lower() in caveat.lower() for req in requirements.get('caveats', []))
            for caveat in decision.caveats_mentioned
        )
        if not has_required_caveat and requirements.get('caveats'):
            missing.append("Required disclaimer")
        
        # Check for overconfidence
        if decision.confidence_level > 0.85 and domain in ['medical', 'financial', 'legal']:
            flaws.append(DecisionFlaw.OVERCONFIDENCE)
            missing.append("Appropriate hedging")
        
        # Determine quality
        quality = self._determine_quality(flaws, missing, domain)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(flaws, missing, domain)
        
        # Generate enhanced recommendation
        enhanced = self._enhance_recommendation(decision, flaws, domain)
        
        return DecisionAssessment(
            decision=decision,
            quality=quality,
            flaws=flaws,
            missing_elements=missing,
            improvement_suggestions=suggestions,
            enhanced_recommendation=enhanced
        )
    
    def _determine_quality(self, flaws: List[DecisionFlaw], 
                           missing: List[str], domain: str) -> DecisionQuality:
        """Determine overall decision quality"""
        
        # Dangerous conditions
        if domain in ['medical', 'financial', 'legal'] and len(flaws) >= 3:
            return DecisionQuality.DANGEROUS
        
        # Poor
        if len(flaws) >= 2:
            return DecisionQuality.POOR
        
        # Adequate
        if len(flaws) == 1 or len(missing) >= 2:
            return DecisionQuality.ADEQUATE
        
        # Good
        if len(missing) == 1:
            return DecisionQuality.GOOD
        
        return DecisionQuality.EXCELLENT
    
    def _generate_suggestions(self, flaws: List[DecisionFlaw], 
                               missing: List[str], domain: str) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        for flaw in flaws:
            if flaw == DecisionFlaw.MISSING_ALTERNATIVES:
                suggestions.append("Add 2-3 alternative approaches")
            elif flaw == DecisionFlaw.MISSING_RISKS:
                suggestions.append("Acknowledge potential risks and downsides")
            elif flaw == DecisionFlaw.OVERCONFIDENCE:
                suggestions.append("Use hedging language to acknowledge uncertainty")
        
        if 'Required disclaimer' in missing:
            requirements = self.DOMAIN_REQUIREMENTS.get(domain, {})
            if requirements.get('caveats'):
                suggestions.append(f"Add disclaimer: {requirements['caveats'][0]}")
        
        return suggestions
    
    def _enhance_recommendation(self, decision: Decision, 
                                 flaws: List[DecisionFlaw],
                                 domain: str) -> str:
        """Generate enhanced version of recommendation"""
        enhanced = decision.recommendation
        additions = []
        
        # Add alternatives if missing
        if DecisionFlaw.MISSING_ALTERNATIVES in flaws:
            generic_alternatives = {
                'medical': "consulting a specialist, seeking a second opinion, or exploring conservative treatments first",
                'financial': "diversifying investments, consulting a financial advisor, or starting with a smaller position",
                'legal': "consulting with an attorney, seeking legal aid, or exploring mediation",
                'general': "gathering more information, consulting an expert, or considering the status quo"
            }
            alt = generic_alternatives.get(domain, generic_alternatives['general'])
            additions.append(f"\n\nAlternative approaches to consider: {alt}.")
        
        # Add risks if missing
        if DecisionFlaw.MISSING_RISKS in flaws:
            generic_risks = {
                'medical': "side effects, interactions with other medications, and individual health factors",
                'financial': "market volatility, potential loss of principal, and changing economic conditions",
                'legal': "legal costs, time investment, and potential adverse outcomes",
                'general': "unforeseen consequences and individual circumstances"
            }
            risk = generic_risks.get(domain, generic_risks['general'])
            additions.append(f"\n\nPotential risks to consider: {risk}.")
        
        # Add disclaimer if domain requires it
        requirements = self.DOMAIN_REQUIREMENTS.get(domain, {})
        if requirements.get('caveats') and not any(
            any(c.lower() in decision.recommendation.lower() for c in requirements['caveats'])
            for _ in [1]
        ):
            caveat = requirements['caveats'][0]
            additions.append(f"\n\nImportant: Please {caveat} before making any decisions.")
        
        # Soften overconfidence
        if DecisionFlaw.OVERCONFIDENCE in flaws:
            enhanced = f"Based on general principles, {enhanced[0].lower()}{enhanced[1:]}"
        
        return enhanced + ''.join(additions)
    
    def _calculate_overall_quality(self, 
                                    assessments: List[DecisionAssessment]) -> DecisionQuality:
        """Calculate overall quality from assessments"""
        if not assessments:
            return DecisionQuality.ADEQUATE
        
        qualities = [a.quality for a in assessments]
        
        # Worst quality dominates
        if DecisionQuality.DANGEROUS in qualities:
            return DecisionQuality.DANGEROUS
        if DecisionQuality.POOR in qualities:
            return DecisionQuality.POOR
        if DecisionQuality.ADEQUATE in qualities:
            return DecisionQuality.ADEQUATE
        if DecisionQuality.GOOD in qualities:
            return DecisionQuality.GOOD
        return DecisionQuality.EXCELLENT
    
    def _quality_to_score(self, quality: DecisionQuality) -> float:
        """Convert quality to numerical score"""
        scores = {
            DecisionQuality.EXCELLENT: 1.0,
            DecisionQuality.GOOD: 0.8,
            DecisionQuality.ADEQUATE: 0.6,
            DecisionQuality.POOR: 0.3,
            DecisionQuality.DANGEROUS: 0.1
        }
        return scores.get(quality, 0.5)
    
    def _generate_enhanced_text(self, original: str, 
                                 assessments: List[DecisionAssessment],
                                 domain: str) -> str:
        """Generate enhanced version of full text"""
        result = original
        
        for assessment in assessments:
            if assessment.quality != DecisionQuality.EXCELLENT:
                result = result.replace(
                    assessment.decision.recommendation,
                    assessment.enhanced_recommendation
                )
        
        return result
    
    def _summarize_enhancements(self, 
                                 assessments: List[DecisionAssessment]) -> List[str]:
        """Summarize key enhancements made"""
        enhancements = []
        
        for assessment in assessments:
            if assessment.flaws:
                enhancements.append(
                    f"Addressed {len(assessment.flaws)} issue(s): " +
                    ", ".join(f.value for f in assessment.flaws)
                )
        
        return enhancements

    # Learning Interface Methods
    
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



def test_decision_enhancer():
    """Test the decision quality enhancer"""
    enhancer = DecisionQualityEnhancer()
    
    test_text = """
    I recommend putting all your retirement savings into cryptocurrency. 
    This is definitely the best way to grow your money quickly.
    You should invest at least 80% of your portfolio in high-growth tech stocks.
    """
    
    user_context = {'age': 72, 'risk_tolerance': 'conservative'}
    
    print("=" * 70)
    print("DECISION QUALITY ENHANCER TEST")
    print("=" * 70)
    
    print("\nOriginal text:")
    print(test_text)
    
    print("\nUser context:", user_context)
    
    result = enhancer.enhance(test_text, user_context)
    
    print(f"\nDecisions found: {len(result.decisions_found)}")
    print(f"Overall quality: {result.overall_quality.value}")
    print(f"Quality improvement: {result.quality_improvement:.2f}")
    
    print("\nAssessments:")
    for a in result.assessments:
        print(f"  [{a.quality.value}] {a.decision.recommendation[:60]}...")
        if a.flaws:
            print(f"    Flaws: {[f.value for f in a.flaws]}")
        if a.improvement_suggestions:
            print(f"    Suggestions: {a.improvement_suggestions}")
    
    print("\nKey enhancements:")
    for e in result.key_enhancements:
        print(f"  - {e}")
    
    print("\nEnhanced text:")
    print(result.enhanced_text)

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


if __name__ == "__main__":
    test_decision_enhancer()

