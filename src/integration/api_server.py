"""
BAIS Cognitive Governance Engine - Production API Server

FastAPI application for serving BAIS governance capabilities.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# BAIS Core Imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.integrated_engine import IntegratedGovernanceEngine
from core.model_provider import get_api_key, get_best_reasoning_model, list_available_providers
from core.tenant_manager import get_tenant_manager, Tenant

# Configure logging
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================

class AuditRequest(BaseModel):
    """Request model for auditing an LLM response."""
    query: str = Field(..., description="The user's original query")
    response: str = Field(..., description="The LLM's response to audit")
    domain: str = Field(default="general", description="Domain context")
    documents: List[Dict[str, Any]] = Field(default=[], description="Source documents")


class AuditResponse(BaseModel):
    """Response model for audit results."""
    decision: str
    accuracy: float
    confidence: float
    issues: List[str]
    warnings: List[str]
    recommendation: str
    should_regenerate: bool
    improved_response: Optional[str] = None
    processing_time_ms: int
    clinical_status: Optional[str] = None


class VerifyRequest(BaseModel):
    """Request model for verifying a completion claim."""
    claim: str = Field(..., description="The claim to verify")
    evidence: List[str] = Field(default=[], description="Evidence supporting the claim")


class VerifyResponse(BaseModel):
    """Response model for verification results."""
    valid: bool
    confidence: float
    violations: List[str]
    clinical_status: str
    reasoning: Optional[str] = None


class CheckQueryRequest(BaseModel):
    """Request model for pre-checking a query."""
    query: str = Field(..., description="The query to check")


class CheckQueryResponse(BaseModel):
    """Response model for query check results."""
    safe: bool
    risk_level: str
    issues: List[str]


class ImproveRequest(BaseModel):
    """Request model for improving a response."""
    response: str = Field(..., description="The response to improve")
    issues: List[str] = Field(..., description="Issues to address")


class ImproveResponse(BaseModel):
    """Response model for improved response."""
    improved_response: str
    changes_made: List[str]


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    timestamp: str
    providers_available: List[str]


# =============================================================================
# Application State
# =============================================================================

class AppState:
    """Application state container."""
    engine: IntegratedGovernanceEngine = None
    startup_time: datetime = None


state = AppState()


# =============================================================================
# Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting BAIS Cognitive Governance Engine...")
    state.engine = IntegratedGovernanceEngine()
    state.startup_time = datetime.now(timezone.utc)
    logger.info("BAIS engine initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down BAIS engine...")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="BAIS Cognitive Governance Engine",
    description="AI Governance and Bias Detection API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Authentication
# =============================================================================

async def verify_api_key(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None)
) -> Tenant:
    """Verify API key from headers and return tenant."""
    # Extract API key from headers
    api_key = None
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization[7:]
    elif x_api_key:
        api_key = x_api_key
    
    # Skip auth in development
    if os.environ.get("ENVIRONMENT") == "development" and not api_key:
        tenant_manager = get_tenant_manager()
        return tenant_manager.get_tenant_by_api_key("dev-api-key-for-testing-only")
    
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    # Authenticate with tenant manager
    tenant_manager = get_tenant_manager()
    tenant = tenant_manager.authenticate(api_key)
    
    if not tenant:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Check rate limits
    if not tenant_manager.check_rate_limit(tenant.id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    return tenant


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    providers = list_available_providers()
    return HealthResponse(
        status="healthy",
        version="29.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        providers_available=providers
    )


@app.get("/live")
async def liveness_probe():
    """
    Kubernetes liveness probe.
    Returns 200 if service is running - used to detect if restart needed.
    """
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/ready")
async def readiness_probe():
    """
    Kubernetes readiness probe.
    Returns 200 if service is ready to accept traffic.
    """
    is_ready = state.engine is not None
    if not is_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/startup")
async def startup_probe():
    """
    Kubernetes startup probe.
    Returns 200 if service has completed initialization.
    """
    if state.startup_time is None:
        raise HTTPException(status_code=503, detail="Service still starting")
    uptime = (datetime.now(timezone.utc) - state.startup_time).total_seconds()
    return {
        "status": "started",
        "uptime_seconds": uptime,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/metrics")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.
    Returns metrics in Prometheus exposition format.
    """
    from fastapi.responses import PlainTextResponse
    
    uptime = 0
    if state.startup_time:
        uptime = (datetime.now(timezone.utc) - state.startup_time).total_seconds()
    
    # Gather metrics
    metrics_lines = [
        "# HELP bais_uptime_seconds Uptime of the BAIS service",
        "# TYPE bais_uptime_seconds gauge",
        f"bais_uptime_seconds {uptime}",
        "",
        "# HELP bais_info BAIS service information",
        "# TYPE bais_info gauge",
        'bais_info{version="29.0.0",environment="' + os.environ.get("ENVIRONMENT", "development") + '"} 1',
        "",
        "# HELP bais_engine_ready Whether the BAIS engine is ready",
        "# TYPE bais_engine_ready gauge",
        f"bais_engine_ready {1 if state.engine else 0}",
    ]
    
    # Add provider availability
    providers = list_available_providers()
    metrics_lines.extend([
        "",
        "# HELP bais_providers_available Number of available LLM providers",
        "# TYPE bais_providers_available gauge",
        f"bais_providers_available {len(providers)}"
    ])
    
    return PlainTextResponse(
        content="\n".join(metrics_lines),
        media_type="text/plain; version=0.0.4"
    )


@app.post("/governance/audit", response_model=AuditResponse)
async def audit_response(
    request: AuditRequest,
    tenant: Tenant = Depends(verify_api_key)
):
    """
    Audit an LLM response for bias, errors, and quality issues.
    
    This is the primary endpoint for BAIS governance.
    """
    start_time = time.time()
    
    try:
        # Record usage
        tenant_manager = get_tenant_manager()
        tenant_manager.record_usage(tenant.id, api_calls=1, audits=1)
        
        # Run governance evaluation (async)
        result = await state.engine.evaluate(
            query=request.query,
            response=request.response,
            documents=request.documents
        )
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Map GovernanceDecision attributes to AuditResponse
        issues = result.warnings if hasattr(result, 'warnings') else []
        recommendations = result.recommendations if hasattr(result, 'recommendations') else []
        
        return AuditResponse(
            decision="approved" if result.accepted else "rejected",
            accuracy=result.accuracy,
            confidence=result.confidence,
            issues=issues,
            warnings=issues,  # Use same warnings list
            recommendation="; ".join(recommendations) if recommendations else "",
            should_regenerate=not result.accepted and len(issues) > 2,
            improved_response=None,  # Not provided by GovernanceDecision
            processing_time_ms=processing_time_ms,
            clinical_status=result.pathway.value if hasattr(result, 'pathway') else None
        )
        
    except Exception as e:
        logger.error(f"Audit failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/governance/verify", response_model=VerifyResponse)
async def verify_completion(
    request: VerifyRequest,
    tenant: Tenant = Depends(verify_api_key)
):
    """
    Verify a completion claim against evidence.
    
    Uses LLM-based proof validation to determine if a claim is truly complete.
    """
    try:
        # Use the hybrid proof validator if available
        if hasattr(state.engine, 'hybrid_proof_validator'):
            result = state.engine.hybrid_proof_validator.validate(
                claim=request.claim,
                evidence=request.evidence
            )
            return VerifyResponse(
                valid=result.get("valid", False),
                confidence=result.get("confidence", 0),
                violations=result.get("violations", []),
                clinical_status=result.get("clinical_status", "unknown"),
                reasoning=result.get("reasoning")
            )
        else:
            # Fallback to basic verification
            return VerifyResponse(
                valid=len(request.evidence) > 0,
                confidence=0.5,
                violations=[],
                clinical_status="unknown"
            )
            
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/governance/check_query", response_model=CheckQueryResponse)
async def check_query(
    request: CheckQueryRequest,
    tenant: Tenant = Depends(verify_api_key)
):
    """
    Pre-check a query for manipulation or injection attempts.
    """
    try:
        # Check for common injection patterns
        query_lower = request.query.lower()
        issues = []
        risk_level = "low"
        
        # Injection patterns
        injection_patterns = [
            ("ignore previous", "Prompt injection attempt detected"),
            ("disregard instructions", "Prompt injection attempt detected"),
            ("you are now", "Role manipulation detected"),
            ("pretend you are", "Role manipulation detected"),
            ("forget everything", "Memory manipulation detected"),
        ]
        
        for pattern, issue in injection_patterns:
            if pattern in query_lower:
                issues.append(issue)
                risk_level = "high"
        
        # Length check
        if len(request.query) > 10000:
            issues.append("Query exceeds maximum length")
            risk_level = "medium" if risk_level == "low" else risk_level
        
        return CheckQueryResponse(
            safe=len(issues) == 0,
            risk_level=risk_level,
            issues=issues
        )
        
    except Exception as e:
        logger.error(f"Query check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/governance/improve", response_model=ImproveResponse)
async def improve_response(
    request: ImproveRequest,
    tenant: Tenant = Depends(verify_api_key)
):
    """
    Improve a response based on detected issues.
    """
    try:
        # Use the response improver if available
        if hasattr(state.engine, 'response_improver'):
            result = await state.engine.response_improver.improve(
                response=request.response,
                issues=request.issues
            )
            return ImproveResponse(
                improved_response=result.improved_text if hasattr(result, 'improved_text') else request.response,
                changes_made=result.changes if hasattr(result, 'changes') else []
            )
        else:
            # Return original if no improver available
            return ImproveResponse(
                improved_response=request.response,
                changes_made=[]
            )
            
    except Exception as e:
        logger.error(f"Improvement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/governance/statistics")
async def get_statistics(tenant: Tenant = Depends(verify_api_key)):
    """
    Get governance statistics for the current tenant.
    """
    try:
        tenant_manager = get_tenant_manager()
        usage_stats = tenant_manager.get_usage_stats(tenant.id)
        
        engine_stats = {}
        if hasattr(state.engine, 'get_statistics'):
            engine_stats = state.engine.get_statistics()
        
        return {
            "tenant": tenant.to_dict(),
            "usage": usage_stats,
            "engine": engine_stats,
            "uptime_seconds": (datetime.now(timezone.utc) - state.startup_time).total_seconds()
        }
    except Exception as e:
        logger.error(f"Statistics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/providers")
async def list_providers():
    """List available LLM providers."""
    providers = list_available_providers()
    return {
        "providers": providers,
        "count": len(providers)
    }


# =============================================================================
# Tenant Management Endpoints
# =============================================================================

class AddLLMConfigRequest(BaseModel):
    """Request to add LLM configuration."""
    provider: str
    api_key: str
    model_name: Optional[str] = None
    priority: int = 0


@app.post("/tenant/llm")
async def add_tenant_llm(
    request: AddLLMConfigRequest,
    tenant: Tenant = Depends(verify_api_key)
):
    """Add LLM provider configuration for the tenant."""
    tenant_manager = get_tenant_manager()
    success = tenant_manager.add_llm_config(
        tenant_id=tenant.id,
        provider=request.provider,
        api_key=request.api_key,
        model_name=request.model_name,
        priority=request.priority
    )
    
    if success:
        return {"status": "success", "message": f"Added {request.provider} configuration"}
    else:
        raise HTTPException(status_code=400, detail="Failed to add LLM configuration")


@app.delete("/tenant/llm/{provider}")
async def remove_tenant_llm(
    provider: str,
    tenant: Tenant = Depends(verify_api_key)
):
    """Remove LLM provider configuration for the tenant."""
    tenant_manager = get_tenant_manager()
    success = tenant_manager.remove_llm_config(tenant.id, provider)
    
    if success:
        return {"status": "success", "message": f"Removed {provider} configuration"}
    else:
        raise HTTPException(status_code=404, detail="Provider not found")


@app.get("/tenant/info")
async def get_tenant_info(tenant: Tenant = Depends(verify_api_key)):
    """Get current tenant information."""
    return tenant.to_dict()


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__}
    )


# =============================================================================
# Run Server (Development)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=os.environ.get("ENVIRONMENT") == "development"
    )

