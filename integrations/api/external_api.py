"""
External Governance API (Front 2)
API for external LLM/service integration

Proprietary IP - 100% owned by Invitas Inc.
"""

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from onyx.db.engine import get_session
from onyx.auth.api_key import api_key_dep
from onyx.governance.fusion import CognitiveGovernanceEngine, GovernanceContext, ContextProfile, GovernanceDecision
from onyx.context.search.models import InferenceChunk


router = APIRouter(prefix="/governance/external", tags=["governance"])


class ExternalGovernanceRequest(BaseModel):
    """External governance evaluation request"""
    query: str
    response: str
    documents: Optional[List[Dict[str, Any]]] = None
    citations: Optional[Dict[int, str]] = None
    context_profile: Optional[str] = "general"
    api_key: Optional[str] = None


class ExternalGovernanceResponse(BaseModel):
    """External governance evaluation response"""
    accepted: bool
    confidence: float
    calibrated_posterior: Dict[str, float]
    temporal_state: str
    behavioral_state: str
    rag_quality: Optional[float] = None
    fact_check_coverage: Optional[float] = None
    reasoning_valid: Optional[bool] = None
    knowledge_aligned: Optional[bool] = None
    semantic_understood: Optional[bool] = None
    logical_valid: Optional[bool] = None
    rejection_reason: Optional[str] = None
    audit_hash: str
    recommendation: str  # "accept", "reject", "revise"


@router.post("/evaluate", response_model=ExternalGovernanceResponse)
def evaluate_external_governance(
    request: ExternalGovernanceRequest,
    _: None = Depends(api_key_dep),
    db_session: Session = Depends(get_session),
) -> ExternalGovernanceResponse:
    """
    Evaluate governance for external LLM/service
    
    This is Front 2: External API
    Can be called by any external service or LLM
    """
    # Convert context profile
    try:
        context_profile = ContextProfile(request.context_profile or "general")
    except ValueError:
        context_profile = ContextProfile.GENERAL
    
    # Initialize governance engine
    engine = CognitiveGovernanceEngine(
        db_session=db_session,
        context_profile=context_profile,
    )
    
    # Convert documents if provided
    documents = []
    if request.documents:
        for doc_dict in request.documents:
            chunk = InferenceChunk(
                document_id=doc_dict.get("document_id", ""),
                chunk_id=doc_dict.get("chunk_id", 0),
                content=doc_dict.get("content", ""),
                score=doc_dict.get("score", 0.0),
                source_type=doc_dict.get("source_type"),
                semantic_identifier=doc_dict.get("semantic_identifier"),
                blurb=doc_dict.get("blurb", ""),
                metadata=doc_dict.get("metadata", {}),
            )
            documents.append(chunk)
    
    # Build context
    context = GovernanceContext(
        profile=context_profile,
        user_id=None,  # External API doesn't have user context
        project_id=None,
        chat_session_id=None,
        query=request.query,
        retrieved_documents=documents,
        llm_response=request.response,
        citations=request.citations or {},
        metadata={},
    )
    
    # Evaluate
    decision = engine.evaluate_response(
        query=request.query,
        response=request.response,
        retrieved_documents=documents,
        citations=request.citations or {},
        context=context,
    )
    
    # Generate recommendation
    recommendation = "accept"
    if not decision.accepted:
        recommendation = "reject"
    elif decision.confidence < 0.7:
        recommendation = "revise"
    
    # Build response
    return ExternalGovernanceResponse(
        accepted=decision.accepted,
        confidence=decision.confidence,
        calibrated_posterior=decision.calibrated_posterior,
        temporal_state=decision.temporal_state.state.value,
        behavioral_state=decision.behavioral_state.pathway.value,
        rag_quality=decision.rag_quality.overall_score if documents else None,
        fact_check_coverage=decision.fact_check.coverage if documents else None,
        reasoning_valid=decision.reasoning_verification.valid if documents else None,
        knowledge_aligned=decision.knowledge_alignment.aligned if documents else None,
        semantic_understood=decision.semantic_analysis.understood,
        logical_valid=decision.logical_verification.valid,
        rejection_reason=decision.rejection_reason,
        audit_hash=decision.audit_hash,
        recommendation=recommendation,
    )


@router.get("/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy", "service": "external_governance_api"}


@router.get("/capabilities")
def get_capabilities() -> Dict[str, Any]:
    """Get governance capabilities"""
    return {
        "pathways": [
            "temporal_detection",
            "behavioral_detection",
            "rag_governance",
            "fact_checking",
            "reasoning_verification",
            "knowledge_graph",
            "semantic_understanding",
            "logical_reasoning",
            "episodic_memory",
            "meta_learning",
        ],
        "context_profiles": ["crisis", "regulated", "general", "low_stakes"],
        "acceptance_profiles": ["lean", "formal"],
    }










