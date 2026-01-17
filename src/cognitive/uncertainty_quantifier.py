"""
BASE Uncertainty Quantifier

Ensures appropriate uncertainty is expressed in LLM outputs.
Key principle: Know what you don't know.

Patent Alignment: Novel Invention - Calibrated Uncertainty Expression
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
import re


class CertaintyLevel(Enum):
    """Levels of certainty"""
    HIGHLY_CERTAIN = "highly_certain"     # >95% confidence warranted
    MODERATELY_CERTAIN = "moderately_certain"  # 70-95% confidence
    UNCERTAIN = "uncertain"               # 40-70% confidence
    HIGHLY_UNCERTAIN = "highly_uncertain"  # <40% confidence
    UNQUANTIFIABLE = "unquantifiable"     # Cannot estimate


class CertaintyMismatch(Enum):
    """Types of certainty mismatch"""
    OVERCLAIMING = "overclaiming"         # Claims more certain than warranted
    UNDERCLAIMING = "underclaiming"       # Claims less certain than warranted
    MISSING_UNCERTAINTY = "missing_uncertainty"  # No uncertainty expressed
    APPROPRIATE = "appropriate"           # Uncertainty is appropriate


@dataclass
class UncertaintyClaim:
    """A claim with uncertainty analysis"""
    text: str
    expressed_certainty: CertaintyLevel
    warranted_certainty: CertaintyLevel
    mismatch: CertaintyMismatch
    confidence_in_assessment: float


@dataclass
class UncertaintyFix:
    """A fix to apply to a claim"""
    original: str
    fixed: str
    reason: str


@dataclass
class UncertaintyAnalysisResult:
    """Complete result of uncertainty analysis"""
    claims: List[UncertaintyClaim]
    fixes_needed: List[UncertaintyFix]
    overall_calibration: float  # How well calibrated is the uncertainty?
    improved_text: str
    summary: str


class UncertaintyQuantifier:
    """
    Ensures appropriate uncertainty in LLM outputs.
    
    Key capabilities:
    1. Detect expressed certainty levels
    2. Assess warranted certainty levels
    3. Fix mismatches
    4. Generate properly calibrated text
    """
    
    # Certainty markers in text
    CERTAINTY_MARKERS = {
        CertaintyLevel.HIGHLY_CERTAIN: [
            r'\b(?:definitely|certainly|absolutely|undoubtedly|100%|always|never)\b',
            r'\b(?:guaranteed|proven|established fact|undeniable|impossible)\b',
            r'\b(?:must|will certainly|cannot fail|without doubt)\b'
        ],
        CertaintyLevel.MODERATELY_CERTAIN: [
            r'\b(?:likely|probably|generally|usually|typically|most)\b',
            r'\b(?:should|would|confident|expected|tends to)\b',
            r'\b(?:evidence suggests|studies show|research indicates)\b'
        ],
        CertaintyLevel.UNCERTAIN: [
            r'\b(?:might|may|could|possibly|perhaps|potentially)\b',
            r'\b(?:unclear|uncertain|depends|varies|sometimes)\b',
            r'\b(?:limited evidence|some studies|in some cases)\b'
        ],
        CertaintyLevel.HIGHLY_UNCERTAIN: [
            r'\b(?:unknown|speculative|anecdotal|unverified)\b',
            r'\b(?:no evidence|inconclusive|contradictory)\b',
            r'\b(?:purely theoretical|mere hypothesis)\b'
        ]
    }
    
    # Claim types and their warranted certainty
    CLAIM_TYPE_CERTAINTY = {
        'factual_established': CertaintyLevel.HIGHLY_CERTAIN,      # E.g., "Water boils at 100°C"
        'factual_scientific': CertaintyLevel.MODERATELY_CERTAIN,   # E.g., "Smoking causes cancer"
        'statistical': CertaintyLevel.MODERATELY_CERTAIN,          # Numbers with error margins
        'predictive': CertaintyLevel.UNCERTAIN,                    # Future predictions
        'subjective': CertaintyLevel.UNCERTAIN,                    # Opinions, preferences
        'speculative': CertaintyLevel.HIGHLY_UNCERTAIN,            # Theories, hypotheses
        'individual_outcome': CertaintyLevel.UNCERTAIN,            # "This will work for you"
    }
    
    # Hedging templates
    HEDGING_TEMPLATES = {
        CertaintyLevel.MODERATELY_CERTAIN: [
            "It is likely that {claim}",
            "Evidence suggests that {claim}",
            "In most cases, {claim}",
            "{claim}, though individual results may vary"
        ],
        CertaintyLevel.UNCERTAIN: [
            "It may be that {claim}",
            "Depending on circumstances, {claim}",
            "While not certain, {claim}",
            "Some evidence suggests that {claim}"
        ],
        CertaintyLevel.HIGHLY_UNCERTAIN: [
            "It is speculative whether {claim}",
            "There is limited evidence that {claim}",
            "It is unclear if {claim}",
            "Anecdotally, some report that {claim}"
        ]
    }
    
    def __init__(self):
        self.analysis_log: List[str] = []
    
    def analyze(self, text: str, domain: str = 'general') -> UncertaintyAnalysisResult:
        """
        Analyze and improve uncertainty calibration in text.
        
        Args:
            text: Text to analyze
            domain: Domain context
            
        Returns:
            UncertaintyAnalysisResult with analysis and improvements
        """
        self.analysis_log = []
        self._log("Starting uncertainty analysis")
        
        # Step 1: Extract claims with certainty markers
        claims = self._extract_claims(text)
        self._log(f"Found {len(claims)} claims to analyze")
        
        # Step 2: Assess warranted certainty for each claim
        for claim in claims:
            claim.warranted_certainty = self._assess_warranted_certainty(claim, domain)
            claim.mismatch = self._assess_mismatch(claim)
        
        # Step 3: Generate fixes for mismatched claims
        fixes = self._generate_fixes(claims)
        self._log(f"Generated {len(fixes)} fixes")
        
        # Step 4: Apply fixes to text
        improved_text = self._apply_fixes(text, fixes)
        
        # Step 5: Calculate calibration score
        calibration = self._calculate_calibration(claims)
        
        # Step 6: Generate summary
        summary = self._generate_summary(claims, fixes, calibration)
        
        return UncertaintyAnalysisResult(
            claims=claims,
            fixes_needed=fixes,
            overall_calibration=calibration,
            improved_text=improved_text,
            summary=summary
        )
    
    def _log(self, message: str):
        """Log analysis step"""
        self.analysis_log.append(message)
    
    def _extract_claims(self, text: str) -> List[UncertaintyClaim]:
        """Extract claims and their expressed certainty"""
        claims = []
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue
            
            # Detect expressed certainty
            expressed = self._detect_certainty_level(sentence)
            
            claims.append(UncertaintyClaim(
                text=sentence,
                expressed_certainty=expressed,
                warranted_certainty=CertaintyLevel.UNCERTAIN,  # Default, will be assessed
                mismatch=CertaintyMismatch.APPROPRIATE,  # Default
                confidence_in_assessment=0.7
            ))
        
        return claims
    
    def _detect_certainty_level(self, text: str) -> CertaintyLevel:
        """Detect the certainty level expressed in text"""
        text_lower = text.lower()
        
        for level, patterns in self.CERTAINTY_MARKERS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return level
        
        # Default to moderate if no markers found
        return CertaintyLevel.MODERATELY_CERTAIN
    
    def _assess_warranted_certainty(self, claim: UncertaintyClaim, 
                                     domain: str) -> CertaintyLevel:
        """Assess what certainty level is warranted"""
        text_lower = claim.text.lower()
        
        # Check claim type
        claim_type = self._identify_claim_type(text_lower)
        base_certainty = self.CLAIM_TYPE_CERTAINTY.get(
            claim_type, CertaintyLevel.UNCERTAIN
        )
        
        # Adjust for domain
        if domain in ['medical', 'legal', 'financial']:
            # High-stakes domains warrant more caution
            if base_certainty == CertaintyLevel.HIGHLY_CERTAIN:
                return CertaintyLevel.MODERATELY_CERTAIN
            elif base_certainty == CertaintyLevel.MODERATELY_CERTAIN:
                return CertaintyLevel.UNCERTAIN
        
        # Check for individual-specific claims
        if any(w in text_lower for w in ['you will', 'you should', 'for you']):
            # Individual outcomes are inherently uncertain
            return CertaintyLevel.UNCERTAIN
        
        # Check for predictions
        if any(w in text_lower for w in ['will be', 'going to', 'in the future']):
            return CertaintyLevel.UNCERTAIN
        
        return base_certainty
    
    def _identify_claim_type(self, text: str) -> str:
        """Identify the type of claim"""
        # Statistical
        if re.search(r'\d+%|\d+\s+out\s+of', text):
            return 'statistical'
        
        # Predictive
        if any(w in text for w in ['will', 'going to', 'predict', 'forecast']):
            return 'predictive'
        
        # Subjective
        if any(w in text for w in ['best', 'should', 'recommend', 'better']):
            return 'subjective'
        
        # Speculative
        if any(w in text for w in ['theory', 'hypothesis', 'speculate']):
            return 'speculative'
        
        return 'factual_scientific'  # Default
    
    def _assess_mismatch(self, claim: UncertaintyClaim) -> CertaintyMismatch:
        """Assess if there's a certainty mismatch"""
        expressed = claim.expressed_certainty
        warranted = claim.warranted_certainty
        
        certainty_order = [
            CertaintyLevel.HIGHLY_UNCERTAIN,
            CertaintyLevel.UNCERTAIN,
            CertaintyLevel.MODERATELY_CERTAIN,
            CertaintyLevel.HIGHLY_CERTAIN
        ]
        
        expressed_idx = certainty_order.index(expressed) if expressed in certainty_order else 2
        warranted_idx = certainty_order.index(warranted) if warranted in certainty_order else 2
        
        if expressed_idx > warranted_idx:
            return CertaintyMismatch.OVERCLAIMING
        elif expressed_idx < warranted_idx:
            return CertaintyMismatch.UNDERCLAIMING
        else:
            return CertaintyMismatch.APPROPRIATE
    
    def _generate_fixes(self, claims: List[UncertaintyClaim]) -> List[UncertaintyFix]:
        """Generate fixes for mismatched claims"""
        fixes = []
        
        for claim in claims:
            if claim.mismatch == CertaintyMismatch.OVERCLAIMING:
                # Need to add hedging
                fixed = self._add_hedging(claim.text, claim.warranted_certainty)
                fixes.append(UncertaintyFix(
                    original=claim.text,
                    fixed=fixed,
                    reason=f"Overclaiming: expressed {claim.expressed_certainty.value}, "
                           f"warranted {claim.warranted_certainty.value}"
                ))
            elif claim.mismatch == CertaintyMismatch.UNDERCLAIMING:
                # Could strengthen, but we generally don't need to
                pass
        
        return fixes
    
    def _add_hedging(self, text: str, target_certainty: CertaintyLevel) -> str:
        """Add appropriate hedging to text"""
        # Remove high-certainty markers
        hedged = text
        for pattern in self.CERTAINTY_MARKERS[CertaintyLevel.HIGHLY_CERTAIN]:
            hedged = re.sub(pattern, '', hedged, flags=re.IGNORECASE)
        
        hedged = hedged.strip()
        
        # If text is very short after removal, use a template
        if len(hedged) < 20:
            return hedged
        
        # Add appropriate hedging based on target certainty
        if target_certainty == CertaintyLevel.UNCERTAIN:
            # Add "may" or "might"
            if not any(w in hedged.lower() for w in ['may', 'might', 'could']):
                hedged = f"It may be that {hedged[0].lower()}{hedged[1:]}"
        elif target_certainty == CertaintyLevel.MODERATELY_CERTAIN:
            # Add "likely" or "probably"
            if not any(w in hedged.lower() for w in ['likely', 'probably']):
                hedged = f"It is likely that {hedged[0].lower()}{hedged[1:]}"
        elif target_certainty == CertaintyLevel.HIGHLY_UNCERTAIN:
            hedged = f"There is limited evidence to suggest that {hedged[0].lower()}{hedged[1:]}"
        
        return hedged
    
    def _apply_fixes(self, text: str, fixes: List[UncertaintyFix]) -> str:
        """Apply fixes to text"""
        result = text
        for fix in fixes:
            result = result.replace(fix.original, fix.fixed)
        return result
    
    def _calculate_calibration(self, claims: List[UncertaintyClaim]) -> float:
        """Calculate overall calibration score"""
        if not claims:
            return 1.0
        
        appropriate = sum(1 for c in claims if c.mismatch == CertaintyMismatch.APPROPRIATE)
        return appropriate / len(claims)
    
    def _generate_summary(self, claims: List[UncertaintyClaim],
                          fixes: List[UncertaintyFix],
                          calibration: float) -> str:
        """Generate human-readable summary"""
        overclaiming = sum(1 for c in claims if c.mismatch == CertaintyMismatch.OVERCLAIMING)
        
        lines = [
            f"Uncertainty Analysis Summary",
            f"----------------------------",
            f"Claims analyzed: {len(claims)}",
            f"Calibration score: {calibration:.2f}",
            f"Overclaiming instances: {overclaiming}",
            f"Fixes applied: {len(fixes)}"
        ]
        
        if overclaiming > 0:
            lines.append(f"\nWarning: Text expresses more certainty than warranted in {overclaiming} places.")
        
        return "\n".join(lines)

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



def test_uncertainty_quantifier():
    """Test the uncertainty quantifier"""
    quantifier = UncertaintyQuantifier()
    
    test_text = """
    This treatment will definitely cure your condition. 
    Studies show it has a 100% success rate. 
    You will absolutely see results within a week.
    Everyone who uses this product gets amazing results.
    The stock will certainly go up next month.
    """
    
    print("=" * 70)
    print("UNCERTAINTY QUANTIFIER TEST")
    print("=" * 70)
    
    print("\nOriginal text:")
    print(test_text)
    
    result = quantifier.analyze(test_text, domain='medical')
    
    print(f"\n{result.summary}")
    
    print("\nClaims analysis:")
    for claim in result.claims:
        print(f"  [{claim.mismatch.value}] {claim.text[:50]}...")
        print(f"    Expressed: {claim.expressed_certainty.value}")
        print(f"    Warranted: {claim.warranted_certainty.value}")
    
    print("\nFixes applied:")
    for fix in result.fixes_needed:
        print(f"  - {fix.reason}")
        print(f"    Original: {fix.original[:50]}...")
        print(f"    Fixed: {fix.fixed[:50]}...")
    
    print("\nImproved text:")
    print(result.improved_text)



if __name__ == "__main__":
    test_uncertainty_quantifier()






