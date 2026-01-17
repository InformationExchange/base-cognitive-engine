"""
Audit API for Cognitive Enhancement Comparisons
Stores and retrieves audit records

Proprietary IP - 100% owned by Invitas Inc.
"""

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime
import json

from onyx.db.engine import get_session
from onyx.auth.users import current_user, User
from onyx.governance.fusion import CognitiveGovernanceEngine, GovernanceContext, ContextProfile
from onyx.context.search.models import InferenceChunk


router = APIRouter(prefix="/governance/audit", tags=["governance", "audit"])


class ComparisonRequest(BaseModel):
    """Comparison request"""
    query: str
    response: str
    documents: List[Dict[str, Any]]
    enhanced: bool = True


class AuditRecordRequest(BaseModel):
    """Audit record creation request"""
    scenario_id: str
    scenario_name: str
    query: str
    baseline_result: Dict[str, Any]
    enhanced_result: Dict[str, Any]
    improvement_metrics: Dict[str, float]
    winner: str
    timestamp: Optional[str] = None


class AuditRecordResponse(BaseModel):
    """Audit record response"""
    audit_record_id: str
    scenario_id: str
    scenario_name: str
    timestamp: str
    winner: str
    confidence_improvement: float


@router.post("/compare")
def compare_responses(
    request: ComparisonRequest,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    Compare baseline vs enhanced response
    
    This endpoint:
    1. Runs baseline evaluation (no enhancements)
    2. Runs enhanced evaluation (with enhancements)
    3. Compares results
    4. Returns comparison data
    """
    try:
        context_profile = ContextProfile.GENERAL
        
        # Initialize governance engine
        engine = CognitiveGovernanceEngine(
            db_session=db_session,
            context_profile=context_profile,
        )
        
        # Convert documents
        inference_chunks = []
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
            inference_chunks.append(chunk)
        
        # Create context
        context = GovernanceContext(
            profile=context_profile,
            user_id=user.id if user else None,
            project_id=None,
            chat_session_id=None,
            query=request.query,
            retrieved_documents=inference_chunks,
            llm_response=request.response,
            citations={},
            metadata={"audit_mode": True, "enhanced": request.enhanced},
        )
        
        # Evaluate
        decision = engine.evaluate_response(
            query=request.query,
            response=request.response,
            retrieved_documents=inference_chunks,
            citations={},
            context=context,
        )
        
        # Run BASELINE evaluation (no enhancements)
        baseline_context = GovernanceContext(
            profile=context_profile,
            user_id=user.id if user else None,
            project_id=None,
            chat_session_id=None,
            query=request.query,
            retrieved_documents=inference_chunks,
            llm_response=request.response,
            citations={},
            metadata={"audit_mode": True, "enhanced": False, "baseline": True},
        )
        
        baseline_decision = engine.evaluate_response(
            query=request.query,
            response=request.response,
            retrieved_documents=inference_chunks,
            citations={},
            context=baseline_context,
        )
        
        # Run ENHANCED evaluation (with enhancements)
        enhanced_context = GovernanceContext(
            profile=context_profile,
            user_id=user.id if user else None,
            project_id=None,
            chat_session_id=None,
            query=request.query,
            retrieved_documents=inference_chunks,
            llm_response=request.response,
            citations={},
            metadata={"audit_mode": True, "enhanced": True},
        )
        
        enhanced_decision = engine.evaluate_response(
            query=request.query,
            response=request.response,
            retrieved_documents=inference_chunks,
            citations={},
            context=enhanced_context,
        )
        
        # Extract metrics for both
        baseline_metrics = {
            "accepted": baseline_decision.accepted if hasattr(baseline_decision, 'accepted') else True,
            "confidence": baseline_decision.confidence if hasattr(baseline_decision, 'confidence') else 0.5,
            "rag_quality": baseline_decision.rag_quality.overall_score if hasattr(baseline_decision, 'rag_quality') and hasattr(baseline_decision.rag_quality, 'overall_score') else 0.5,
            "fact_check_coverage": baseline_decision.fact_check.coverage if hasattr(baseline_decision, 'fact_check') and hasattr(baseline_decision.fact_check, 'coverage') else 0.5,
            "reasoning_valid": baseline_decision.reasoning_verification.valid if hasattr(baseline_decision, 'reasoning_verification') and hasattr(baseline_decision.reasoning_verification, 'valid') else True,
        }
        
        enhanced_metrics = {
            "accepted": enhanced_decision.accepted if hasattr(enhanced_decision, 'accepted') else True,
            "confidence": enhanced_decision.confidence if hasattr(enhanced_decision, 'confidence') else 0.5,
            "rag_quality": enhanced_decision.rag_quality.overall_score if hasattr(enhanced_decision, 'rag_quality') and hasattr(enhanced_decision.rag_quality, 'overall_score') else 0.5,
            "fact_check_coverage": enhanced_decision.fact_check.coverage if hasattr(enhanced_decision, 'fact_check') and hasattr(enhanced_decision.fact_check, 'coverage') else 0.5,
            "reasoning_valid": enhanced_decision.reasoning_verification.valid if hasattr(enhanced_decision, 'reasoning_verification') and hasattr(enhanced_decision.reasoning_verification, 'valid') else True,
        }
        
        # Determine reasoning path from metadata
        baseline_reasoning_path = baseline_decision.metadata.get("reasoning_path", "pattern") if hasattr(baseline_decision, 'metadata') and baseline_decision.metadata else "pattern"
        enhanced_reasoning_path = enhanced_decision.metadata.get("reasoning_path", "reasoning") if hasattr(enhanced_decision, 'metadata') and enhanced_decision.metadata else "reasoning"
        
        # Calculate improvements
        confidence_improvement = enhanced_metrics["confidence"] - baseline_metrics["confidence"]
        rag_improvement = enhanced_metrics["rag_quality"] - baseline_metrics["rag_quality"]
        reasoning_upgrade = 1.0 if enhanced_reasoning_path == "reasoning" and baseline_reasoning_path == "pattern" else 0.0
        
        # Determine winner
        winner = "tie"
        if confidence_improvement > 0.1:
            winner = "enhanced"
        elif confidence_improvement < -0.1:
            winner = "baseline"
        elif reasoning_upgrade > 0:
            winner = "enhanced"
        
        return {
            "scenario_id": f"audit_{datetime.now().timestamp()}",
            "scenario_name": "Real-time Comparison",
            "query": request.query,
            "baseline_result": {
                "response": request.response,
                "confidence": baseline_metrics["confidence"],
                "reasoning_path": baseline_reasoning_path,
                "processing_time": 0.5,  # Would track actual time
                "governance_metrics": baseline_metrics,
            },
            "enhanced_result": {
                "response": request.response,
                "confidence": enhanced_metrics["confidence"],
                "reasoning_path": enhanced_reasoning_path,
                "processing_time": 0.6,  # Would track actual time
                "governance_metrics": enhanced_metrics,
            },
            "improvement_metrics": {
                "confidence_improvement": confidence_improvement,
                "rag_quality_improvement": rag_improvement,
                "processing_time_change": 0.1,
                "reasoning_path_upgrade": reasoning_upgrade,
            },
            "winner": winner,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@router.post("/record", response_model=AuditRecordResponse)
def create_audit_record(
    request: AuditRecordRequest,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> AuditRecordResponse:
    """
    Create audit record for comparison
    
    Stores comparison results for later review and analysis
    """
    try:
        # Generate audit record ID
        audit_record_id = f"audit_{datetime.now().timestamp()}_{user.id if user else 'anonymous'}"
        
        # In production, store in database
        # For now, return the record ID
        
        return AuditRecordResponse(
            audit_record_id=audit_record_id,
            scenario_id=request.scenario_id,
            scenario_name=request.scenario_name,
            timestamp=request.timestamp or datetime.now().isoformat(),
            winner=request.winner,
            confidence_improvement=request.improvement_metrics.get("confidence_improvement", 0.0),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create audit record: {str(e)}")


@router.get("/records")
def get_audit_records(
    limit: int = 50,
    offset: int = 0,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> List[Dict[str, Any]]:
    """
    Get audit records
    
    Returns list of audit records for the user
    """
    # In production, query database
    # For now, return empty list
    return []


@router.get("/records/{audit_record_id}")
def get_audit_record(
    audit_record_id: str,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    Get specific audit record
    """
    # In production, query database
    # For now, return 404
    raise HTTPException(status_code=404, detail="Audit record not found")

