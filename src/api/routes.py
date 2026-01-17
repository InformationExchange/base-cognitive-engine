"""
BAIS Cognitive Governance Engine v16.2
API Routes

RESTful API for the governance engine.
Supports dual-mode: FULL (ML) and LITE (Statistical)
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime

# Import config for capabilities
try:
    from core.config import get_config
except ImportError:
    get_config = None

router = APIRouter()

# Request/Response Models

class Document(BaseModel):
    content: str
    score: float = 1.0
    source: Optional[str] = None


class EvaluateRequest(BaseModel):
    query: str = Field(..., description="The user query")
    documents: List[Document] = Field(default=[], description="Source documents")
    response: Optional[str] = Field(None, description="Pre-generated LLM response")
    generate_response: bool = Field(True, description="Generate response if not provided")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class FeedbackRequest(BaseModel):
    session_id: str
    was_correct: bool
    feedback: Optional[str] = None


class AlgorithmSwitchRequest(BaseModel):
    algorithm: str = Field(..., description="Algorithm name: oco, bayesian, thompson, ucb, exp3")


class ThresholdAdjustRequest(BaseModel):
    domain: str
    adjustment: float = Field(..., ge=-20, le=20)


# Global engine reference (set by main.py)
_engine = None


def set_engine(engine):
    global _engine
    _engine = engine


def get_engine():
    if _engine is None:
        raise HTTPException(500, "Engine not initialized")
    return _engine


# Routes

@router.get("/")
async def root():
    """API information."""
    config = get_config() if get_config else None
    mode_info = config.get_capabilities_summary() if config else {"mode": "unknown"}
    
    return {
        "service": "Invitas BAIS Cognitive Governance Engine",
        "version": "16.2.0",
        "mode": mode_info.get('mode_description', 'unknown'),
        "description": "Clinical-grade AI governance with true learning",
        "features": [
            "Dual-mode: FULL (ML 90%) / LITE (Statistical 70%)",
            "Pluggable learning algorithms (OCO, Bayesian, Thompson, UCB, EXP3)",
            "Context-aware adaptive thresholds",
            "State machine with hysteresis",
            "Full patent compliance (PPA 1, 2, 3)",
            "Persistent learning across restarts"
        ],
        "endpoints": {
            "/evaluate": "POST - Evaluate a query/response",
            "/feedback": "POST - Provide outcome feedback",
            "/status": "GET - Engine status",
            "/capabilities": "GET - Mode and capabilities",
            "/learning": "GET - Learning report",
            "/algorithm": "GET/POST - View/switch algorithm",
            "/thresholds": "GET - View learned thresholds",
            "/state": "GET - State machine status",
            "/health": "GET - Health check"
        }
    }


@router.get("/health")
async def health():
    """Health check endpoint."""
    engine = get_engine()
    status = engine.state_machine.get_health_assessment()
    config = get_config() if get_config else None
    
    return {
        "status": "healthy" if status['health'] in ['healthy', 'good'] else "degraded",
        "health_score": status['score'],
        "operational_state": engine.state_machine.state.value,
        "version": "16.2.0",
        "mode": config.get_mode_description() if config else "unknown"
    }


@router.get("/capabilities")
async def capabilities():
    """Get detailed capabilities and mode information."""
    engine = get_engine()
    config = get_config() if get_config else None
    
    if not config:
        return {"error": "Config not available"}
    
    return {
        "version": "16.2.0",
        **config.get_capabilities_summary(),
        "upgrade_available": not config.use_embeddings,
        "upgrade_instructions": "Use Dockerfile.full for FULL mode (ML-based, ~90% accuracy)" if not config.use_embeddings else None
    }


@router.post("/evaluate")
async def evaluate(request: EvaluateRequest):
    """
    Evaluate a query/response for governance.
    
    This is the main evaluation endpoint.
    """
    engine = get_engine()
    
    # Convert documents
    docs = [{"content": d.content, "score": d.score, "source": d.source} 
            for d in request.documents]
    
    # Run evaluation
    decision = await engine.evaluate(
        query=request.query,
        documents=docs,
        response=request.response,
        generate_response=request.generate_response,
        context=request.context
    )
    
    return decision.to_dict()


@router.post("/feedback")
async def feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    """
    Provide feedback on a past decision.
    
    This triggers learning updates.
    """
    engine = get_engine()
    
    result = engine.record_feedback(
        session_id=request.session_id,
        was_correct=request.was_correct,
        feedback=request.feedback
    )
    
    if 'error' in result:
        raise HTTPException(404, result['error'])
    
    return result


@router.get("/status")
async def status():
    """Get comprehensive engine status."""
    engine = get_engine()
    return engine.get_status()


@router.get("/learning")
async def learning():
    """Get detailed learning report."""
    engine = get_engine()
    return engine.get_learning_report()


@router.get("/algorithm")
async def get_algorithm():
    """Get current algorithm information."""
    engine = get_engine()
    optimizer = engine.threshold_optimizer
    
    return {
        "current_algorithm": optimizer.algorithm_name,
        "available_algorithms": ["oco", "bayesian", "thompson", "ucb", "exp3"],
        "statistics": optimizer.algorithm.get_statistics(),
        "performance": optimizer.algorithm_performance
    }


@router.post("/algorithm")
async def switch_algorithm(request: AlgorithmSwitchRequest):
    """Switch to a different learning algorithm."""
    engine = get_engine()
    optimizer = engine.threshold_optimizer
    
    valid_algorithms = ["oco", "bayesian", "thompson", "ucb", "exp3"]
    if request.algorithm.lower() not in valid_algorithms:
        raise HTTPException(400, f"Invalid algorithm. Choose from: {valid_algorithms}")
    
    result = optimizer.switch_algorithm(request.algorithm.lower())
    return result


@router.get("/thresholds")
async def get_thresholds():
    """Get learned thresholds by domain."""
    engine = get_engine()
    stats = engine.threshold_optimizer.algorithm.get_statistics()
    
    return {
        "algorithm": engine.threshold_optimizer.algorithm_name,
        "thresholds_by_domain": stats.get('domain_values', {}),
        "convergence_status": stats.get('convergence_status', {}),
        "context_adjustments": engine.threshold_optimizer.context_adjustments
    }


@router.get("/state")
async def get_state():
    """Get state machine status."""
    engine = get_engine()
    return engine.state_machine.get_status()


@router.get("/history")
async def get_history(
    limit: int = 50,
    domain: Optional[str] = None,
    only_with_feedback: bool = False
):
    """Get decision history."""
    engine = get_engine()
    
    decisions = engine.outcome_memory.get_recent_decisions(
        limit=limit,
        domain=domain,
        only_with_feedback=only_with_feedback
    )
    
    return {
        "count": len(decisions),
        "decisions": [d.to_dict() for d in decisions]
    }


@router.get("/accuracy")
async def get_accuracy(domain: Optional[str] = None):
    """Get accuracy statistics."""
    engine = get_engine()
    
    by_domain = engine.outcome_memory.get_accuracy_by_domain()
    trend = engine.outcome_memory.get_accuracy_trend(domain=domain, days=30)
    
    return {
        "by_domain": by_domain,
        "trend_30d": trend
    }


@router.get("/statistics")
async def get_statistics():
    """Get comprehensive statistics."""
    engine = get_engine()
    return engine.outcome_memory.get_statistics()


@router.post("/threshold/adjust")
async def adjust_threshold(request: ThresholdAdjustRequest):
    """Manually adjust a domain threshold."""
    engine = get_engine()
    optimizer = engine.threshold_optimizer
    
    # Get current threshold
    current = optimizer.algorithm.get_value(request.domain)
    new_value = current + request.adjustment
    new_value = max(30, min(90, new_value))
    
    # Update (this uses the algorithm's internal method)
    optimizer.algorithm._set_initial_value(request.domain, new_value)
    
    # Record event
    optimizer.outcome_memory.record_learning_event(
        event_type='manual_threshold_adjustment',
        domain=request.domain,
        old_value=current,
        new_value=new_value,
        trigger='api_request',
        metadata={'adjustment': request.adjustment}
    )
    
    return {
        "domain": request.domain,
        "old_threshold": current,
        "new_threshold": new_value,
        "adjustment": request.adjustment
    }


@router.post("/state/feedback")
async def state_feedback(too_sensitive: bool = False, too_slow: bool = False):
    """Provide feedback on state machine sensitivity."""
    engine = get_engine()
    
    engine.state_machine.adapt_thresholds({
        'too_sensitive': too_sensitive,
        'too_slow': too_slow
    })
    
    return {
        "feedback_applied": True,
        "new_thresholds": {
            'enter': {s.value: t for s, t in engine.state_machine.enter_thresholds.items()},
            'exit': {s.value: t for s, t in engine.state_machine.exit_thresholds.items()}
        }
    }


@router.get("/patents")
async def patents():
    """Get patent compliance information."""
    config = get_config() if get_config else None
    using_ml = config and config.use_embeddings
    
    return {
        "PPA1": {
            "title": "Multi-Modal Cognitive Fusion",
            "inventions_implemented": [
                "Inv-1: Context Detection",
                "Inv-2: Multi-Timescale Detection (Fast/Slow layers)",
                "Inv-3: Signal Fusion with Learned Weights"
            ]
        },
        "PPA2": {
            "title": "Acceptance Controller",
            "inventions_implemented": [
                "Inv-1: Must-Pass Predicates (Evidence, Completeness, Safety, Grounding)",
                "Inv-2: Acceptance Control with Pathway Routing",
                "Inv-3: Adaptive Threshold via OCO",
                "Inv-4: LCB Conservative Certificate"
            ]
        },
        "PPA3": {
            "title": "Behavioral Detector",
            "inventions_implemented": [
                "Inv-1: State Machine with Hysteresis",
                "Inv-2: Confirmation Bias Detection",
                "Inv-3: Reward-Seeking Detection",
                "Inv-4: Social Validation Detection",
                "Inv-5: Metric Gaming Detection"
            ]
        },
        "compliance_level": "95%" if using_ml else "80%",
        "detection_mode": "FULL (Neural ML)" if using_ml else "LITE (Statistical/Rule-based)",
        "note": None if using_ml else "Upgrade to FULL mode for ML-based semantic detection (~90% accuracy)"
    }

