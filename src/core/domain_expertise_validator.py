"""
BASE Domain Expertise Validator

Uses Multi-Track LLM Challenger to validate domain expertise when BASE
doesn't have direct access to domain-specific databases.

This module bridges the gap between:
- BASE pattern-based detection (domain-agnostic)
- Domain-specific knowledge validation (via external LLMs)

Patent Alignment:
- NOVEL-23: Multi-Track Challenger
- NOVEL-8: Cross-LLM Governance Orchestration
- PPA1-Inv20: Human-Machine Hybrid Arbitration
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import asyncio
import json


class ValidationConfidence(Enum):
    """Confidence levels from domain validation."""
    HIGH = "high"           # 3+ LLMs agree
    MEDIUM = "medium"       # 2 LLMs agree
    LOW = "low"             # No consensus
    UNAVAILABLE = "unavailable"  # No LLMs available


@dataclass
class DomainValidationQuery:
    """A specific query to validate domain expertise."""
    query_id: str
    domain: str
    question: str           # The domain-specific question to ask LLMs
    context: str            # Context from the original response
    expected_answer_type: str  # "boolean", "explanation", "list"


@dataclass
class LLMDomainResponse:
    """Response from a single LLM for domain validation."""
    provider: str
    model: str
    answer: Any
    confidence: float
    reasoning: str
    execution_time_ms: float
    error: Optional[str] = None


@dataclass
class DomainValidationResult:
    """Result of domain validation via Multi-Track LLM."""
    query: DomainValidationQuery
    responses: List[LLMDomainResponse]
    consensus: Optional[Any]
    consensus_confidence: ValidationConfidence
    agreement_ratio: float      # 0.0 - 1.0
    domain_issues: List[str]    # Issues identified by LLMs
    recommendations: List[str]
    total_time_ms: float
    
    def to_dict(self) -> Dict:
        return {
            "domain": self.query.domain,
            "question": self.query.question,
            "consensus": self.consensus,
            "confidence": self.consensus_confidence.value,
            "agreement_ratio": self.agreement_ratio,
            "domain_issues": self.domain_issues,
            "recommendations": self.recommendations,
            "llm_count": len(self.responses),
            "total_time_ms": self.total_time_ms
        }


class DomainExpertiseValidator:
    """
    Validates domain-specific content using Multi-Track LLM.
    
    BASE detects patterns → DomainExpertiseValidator asks LLMs 
    "Is this domain content accurate?"
    """
    
    # Domain-specific validation prompts
    DOMAIN_VALIDATION_PROMPTS = {
        "medical": """You are a medical expert reviewer. Evaluate the following medical content for accuracy.

CONTENT TO EVALUATE:
{content}

SPECIFIC QUESTION:
{question}

Respond in JSON format:
{{
    "is_accurate": true/false,
    "confidence": 0.0-1.0,
    "issues": ["list of medical accuracy issues"],
    "missing_considerations": ["what should have been considered"],
    "reasoning": "explain your assessment"
}}

Be rigorous. Flag anything that could be clinically dangerous or misleading.""",

        "legal": """You are a legal expert reviewer. Evaluate the following legal content for accuracy.

CONTENT TO EVALUATE:
{content}

SPECIFIC QUESTION:
{question}

Respond in JSON format:
{{
    "is_accurate": true/false,
    "confidence": 0.0-1.0,
    "issues": ["list of legal accuracy issues"],
    "missing_case_law": ["relevant cases not cited"],
    "jurisdiction_concerns": ["jurisdiction-specific issues"],
    "reasoning": "explain your assessment"
}}

Be rigorous. Flag incorrect legal standards or missing precedent.""",

        "financial": """You are a financial expert reviewer. Evaluate the following financial content for accuracy.

CONTENT TO EVALUATE:
{content}

SPECIFIC QUESTION:
{question}

Respond in JSON format:
{{
    "is_accurate": true/false,
    "confidence": 0.0-1.0,
    "issues": ["list of financial accuracy issues"],
    "risk_concerns": ["understated or missing risks"],
    "regulatory_issues": ["regulatory compliance concerns"],
    "reasoning": "explain your assessment"
}}

Be rigorous. Flag unrealistic claims or missing risk disclosures.""",

        "code": """You are a software engineering expert reviewer. Evaluate the following code/technical content for accuracy.

CONTENT TO EVALUATE:
{content}

SPECIFIC QUESTION:
{question}

Respond in JSON format:
{{
    "is_accurate": true/false,
    "confidence": 0.0-1.0,
    "issues": ["list of technical accuracy issues"],
    "missing_edge_cases": ["edge cases not handled"],
    "security_concerns": ["potential security issues"],
    "reasoning": "explain your assessment"
}}

Be rigorous. Flag incomplete implementations or security vulnerabilities.""",

        "general": """You are an expert fact-checker. Evaluate the following content for accuracy.

CONTENT TO EVALUATE:
{content}

SPECIFIC QUESTION:
{question}

Respond in JSON format:
{{
    "is_accurate": true/false,
    "confidence": 0.0-1.0,
    "issues": ["list of accuracy issues"],
    "missing_nuance": ["important nuances missing"],
    "reasoning": "explain your assessment"
}}

Be rigorous. Flag overstatements or missing context."""
    }
    
    def __init__(self, multi_track_challenger=None):
        """
        Initialize domain expertise validator.
        
        Args:
            multi_track_challenger: Instance of MultiTrackChallenger (optional)
        """
        self.multi_track = multi_track_challenger
        self._validation_history: List[DomainValidationResult] = []
    
    def set_multi_track_challenger(self, challenger):
        """Set the Multi-Track Challenger instance."""
        self.multi_track = challenger
    
    async def validate_domain_content(self,
                                       content: str,
                                       domain: str,
                                       specific_questions: List[str] = None) -> List[DomainValidationResult]:
        """
        Validate domain-specific content using Multi-Track LLM.
        
        Args:
            content: The content to validate
            domain: Domain context (medical, legal, code, financial, general)
            specific_questions: Specific questions to ask about the content
        
        Returns:
            List of validation results
        """
        import time
        start_time = time.time()
        
        # Generate questions if not provided
        if not specific_questions:
            specific_questions = self._generate_validation_questions(content, domain)
        
        results = []
        
        for question in specific_questions:
            query = DomainValidationQuery(
                query_id=f"val_{domain}_{len(self._validation_history)}",
                domain=domain,
                question=question,
                context=content[:1000],  # Truncate for efficiency
                expected_answer_type="explanation"
            )
            
            # Try Multi-Track LLM validation
            if self.multi_track:
                result = await self._validate_with_multi_track(query, content)
            else:
                # Fallback to pattern-based validation
                result = self._validate_with_patterns(query, content)
            
            results.append(result)
            self._validation_history.append(result)
        
        return results
    
    async def _validate_with_multi_track(self, 
                                          query: DomainValidationQuery,
                                          content: str) -> DomainValidationResult:
        """Validate using Multi-Track LLM Challenger."""
        import time
        start_time = time.time()
        
        # Get domain-specific prompt
        prompt_template = self.DOMAIN_VALIDATION_PROMPTS.get(
            query.domain, 
            self.DOMAIN_VALIDATION_PROMPTS["general"]
        )
        
        prompt = prompt_template.format(
            content=content[:2000],
            question=query.question
        )
        
        try:
            # Use multi-track challenger
            verdict = await self.multi_track.challenge_parallel(
                claim=query.question,
                response=content,
                domain=query.domain
            )
            
            # Process track results
            responses = []
            all_issues = []
            all_recommendations = []
            
            for track_result in verdict.track_results:
                if track_result.error:
                    continue
                
                # Parse the raw response
                parsed = self._parse_llm_response(track_result.raw_response)
                
                responses.append(LLMDomainResponse(
                    provider=track_result.provider.value,
                    model=track_result.model_name,
                    answer=parsed.get("is_accurate", None),
                    confidence=parsed.get("confidence", 0.5),
                    reasoning=parsed.get("reasoning", ""),
                    execution_time_ms=track_result.execution_time_ms
                ))
                
                all_issues.extend(parsed.get("issues", []))
                
                # Domain-specific recommendations
                if query.domain == "medical":
                    all_recommendations.extend(parsed.get("missing_considerations", []))
                elif query.domain == "legal":
                    all_recommendations.extend(parsed.get("missing_case_law", []))
                elif query.domain == "code":
                    all_recommendations.extend(parsed.get("missing_edge_cases", []))
            
            # Calculate consensus
            consensus, confidence, agreement = self._calculate_consensus(responses)
            
            return DomainValidationResult(
                query=query,
                responses=responses,
                consensus=consensus,
                consensus_confidence=confidence,
                agreement_ratio=agreement,
                domain_issues=list(set(all_issues)),
                recommendations=list(set(all_recommendations))[:5],
                total_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return DomainValidationResult(
                query=query,
                responses=[],
                consensus=None,
                consensus_confidence=ValidationConfidence.UNAVAILABLE,
                agreement_ratio=0.0,
                domain_issues=[f"Validation error: {str(e)}"],
                recommendations=["Manual review recommended"],
                total_time_ms=(time.time() - start_time) * 1000
            )
    
    def _validate_with_patterns(self, 
                                 query: DomainValidationQuery,
                                 content: str) -> DomainValidationResult:
        """Fallback validation using patterns when LLMs unavailable."""
        import time
        start_time = time.time()
        
        issues = []
        recommendations = []
        
        content_lower = content.lower()
        
        # Domain-specific pattern checks
        if query.domain == "medical":
            if "diagnosis is certain" in content_lower or "100%" in content_lower:
                issues.append("Overconfident diagnosis detected")
                recommendations.append("Consider differential diagnosis")
            if "start immediately" in content_lower:
                issues.append("Immediate treatment without full evaluation")
                recommendations.append("Recommend proper evaluation before treatment")
        
        elif query.domain == "legal":
            if "clearly established" in content_lower and "cite" not in content_lower:
                issues.append("Legal standard claimed without citation")
                recommendations.append("Add supporting case law citations")
            if "summary judgment" in content_lower and "plaintiff" in content_lower:
                issues.append("Plaintiff summary judgment is rare")
                recommendations.append("Review summary judgment standards")
        
        elif query.domain == "code":
            if "fully implemented" in content_lower and "test" not in content_lower:
                issues.append("Completion claimed without test evidence")
                recommendations.append("Add test coverage")
            if "handles all" in content_lower:
                issues.append("Absolute claim about edge case handling")
                recommendations.append("Enumerate handled edge cases")
        
        return DomainValidationResult(
            query=query,
            responses=[],
            consensus=len(issues) == 0,
            consensus_confidence=ValidationConfidence.LOW,
            agreement_ratio=0.0,
            domain_issues=issues,
            recommendations=recommendations,
            total_time_ms=(time.time() - start_time) * 1000
        )
    
    def _generate_validation_questions(self, content: str, domain: str) -> List[str]:
        """Generate domain-specific validation questions."""
        questions = []
        
        # Generic question
        questions.append(f"Is the {domain} content in this response accurate and complete?")
        
        # Domain-specific questions
        if domain == "medical":
            questions.append("Are there important differential diagnoses that should be considered?")
            questions.append("Are there any safety concerns with the recommendations?")
        elif domain == "legal":
            questions.append("Is the legal analysis consistent with current case law?")
            questions.append("Are there jurisdiction-specific considerations being missed?")
        elif domain == "code":
            questions.append("Are there edge cases or error conditions not handled?")
            questions.append("Are there security vulnerabilities in this implementation?")
        elif domain == "financial":
            questions.append("Are the risk disclosures adequate?")
            questions.append("Are there regulatory compliance concerns?")
        
        return questions[:3]  # Limit to 3 questions
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM JSON response."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {
            "is_accurate": None,
            "confidence": 0.5,
            "issues": [],
            "reasoning": response[:500]
        }
    
    def _calculate_consensus(self, 
                             responses: List[LLMDomainResponse]
                             ) -> tuple[Optional[bool], ValidationConfidence, float]:
        """Calculate consensus from LLM responses."""
        if not responses:
            return None, ValidationConfidence.UNAVAILABLE, 0.0
        
        # Count votes
        accurate_votes = sum(1 for r in responses if r.answer is True)
        inaccurate_votes = sum(1 for r in responses if r.answer is False)
        total = len(responses)
        
        # Determine consensus
        if accurate_votes > total / 2:
            consensus = True
            agreement = accurate_votes / total
        elif inaccurate_votes > total / 2:
            consensus = False
            agreement = inaccurate_votes / total
        else:
            consensus = None
            agreement = max(accurate_votes, inaccurate_votes) / total if total > 0 else 0.0
        
        # Determine confidence level
        if total >= 3 and agreement >= 0.8:
            confidence = ValidationConfidence.HIGH
        elif total >= 2 and agreement >= 0.6:
            confidence = ValidationConfidence.MEDIUM
        else:
            confidence = ValidationConfidence.LOW
        
        return consensus, confidence, agreement
    
    def get_validation_summary(self, domain: str = None) -> Dict:
        """Get summary of validation history."""
        relevant = self._validation_history
        if domain:
            relevant = [r for r in relevant if r.query.domain == domain]
        
        if not relevant:
            return {"total_validations": 0}
        
        return {
            "total_validations": len(relevant),
            "high_confidence": sum(1 for r in relevant if r.consensus_confidence == ValidationConfidence.HIGH),
            "medium_confidence": sum(1 for r in relevant if r.consensus_confidence == ValidationConfidence.MEDIUM),
            "low_confidence": sum(1 for r in relevant if r.consensus_confidence == ValidationConfidence.LOW),
            "average_agreement": sum(r.agreement_ratio for r in relevant) / len(relevant),
            "total_issues_found": sum(len(r.domain_issues) for r in relevant)
        }

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

