"""
BAIS Cognitive Enhancement API

The main API endpoint for enhancing LLM outputs.
This transforms BAIS from a "gate" to an "enhancer".

Usage:
  POST /enhance - Enhance an LLM response
  POST /enhance/stream - Stream enhanced response
  GET /enhance/capabilities - Get enhancement capabilities
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cognitive_enhancer import CognitiveEnhancer, CognitiveEnhancementResult, EnhancementLevel

# Create router
router = APIRouter(prefix="/enhance", tags=["Enhancement"])

# Initialize enhancer (singleton)
_enhancer: Optional[CognitiveEnhancer] = None


def get_enhancer() -> CognitiveEnhancer:
    """Get or create the cognitive enhancer singleton"""
    global _enhancer
    if _enhancer is None:
        _enhancer = CognitiveEnhancer(
            storage_path="learning_data",
            enable_learning=True
        )
    return _enhancer


# Request/Response models
class EnhanceRequest(BaseModel):
    """Request to enhance an LLM response"""
    query: str = Field(..., description="The original user query")
    response: str = Field(..., description="The LLM response to enhance")
    domain: str = Field(default="general", description="Domain context (medical, financial, legal, general)")
    user_context: Optional[Dict[str, Any]] = Field(default=None, description="Optional user context")
    enhancement_depth: str = Field(default="standard", description="quick, standard, or deep")
    return_trace: bool = Field(default=False, description="Include enhancement trace in response")


class EnhancementFeedback(BaseModel):
    """Feedback on enhancement quality"""
    enhancement_id: str = Field(..., description="ID of the enhancement")
    satisfied: bool = Field(..., description="Whether user was satisfied")
    feedback: str = Field(default="", description="Optional feedback text")


class EnhanceResponse(BaseModel):
    """Response from enhancement"""
    enhancement_id: str
    original_response: str
    enhanced_response: str
    enhancement_level: str
    quality_improvement: float
    original_quality: float
    enhanced_quality: float
    trace: Optional[List[Dict]] = None
    memories_applied: int
    processing_time_ms: float


class CapabilitiesResponse(BaseModel):
    """Enhancement capabilities"""
    version: str
    capabilities: List[str]
    domains_supported: List[str]
    enhancement_depths: List[str]
    learning_enabled: bool
    memory_stats: Dict[str, Any]


# Store recent enhancements for feedback
_recent_enhancements: Dict[str, CognitiveEnhancementResult] = {}


@router.post("/", response_model=EnhanceResponse)
async def enhance_response(request: EnhanceRequest):
    """
    Enhance an LLM response using BAIS cognitive capabilities.
    
    This endpoint IMPROVES responses, it doesn't just detect issues.
    
    Example:
    ```
    POST /enhance/
    {
        "query": "How should I invest my retirement savings?",
        "response": "Put everything in Bitcoin for guaranteed returns!",
        "domain": "financial",
        "enhancement_depth": "deep"
    }
    ```
    """
    start_time = datetime.now()
    
    enhancer = get_enhancer()
    
    try:
        result = enhancer.enhance(
            query=request.query,
            response=request.response,
            domain=request.domain,
            user_context=request.user_context,
            enhancement_depth=request.enhancement_depth
        )
        
        # Generate enhancement ID
        enhancement_id = f"enh_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        # Store for potential feedback
        _recent_enhancements[enhancement_id] = result
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Build response
        response = EnhanceResponse(
            enhancement_id=enhancement_id,
            original_response=result.original_response,
            enhanced_response=result.enhanced_response,
            enhancement_level=result.enhancement_level.value,
            quality_improvement=result.overall_improvement,
            original_quality=result.original_quality,
            enhanced_quality=result.enhanced_quality,
            memories_applied=len(result.memories_applied),
            processing_time_ms=processing_time
        )
        
        # Include trace if requested
        if request.return_trace:
            response.trace = [
                {
                    "step": t.step,
                    "module": t.module,
                    "input_quality": t.input_quality,
                    "output_quality": t.output_quality,
                    "changes": t.changes_made
                }
                for t in result.trace
            ]
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")


@router.post("/stream")
async def enhance_response_stream(request: EnhanceRequest):
    """
    Stream the enhanced response in real-time.
    
    Useful for showing progressive enhancement to users.
    """
    async def generate():
        enhancer = get_enhancer()
        
        try:
            # Yield initial status
            yield json.dumps({"status": "started", "message": "Beginning enhancement..."}) + "\n"
            
            result = enhancer.enhance(
                query=request.query,
                response=request.response,
                domain=request.domain,
                user_context=request.user_context,
                enhancement_depth=request.enhancement_depth
            )
            
            # Yield trace steps
            for trace in result.trace:
                yield json.dumps({
                    "status": "progress",
                    "step": trace.step,
                    "module": trace.module,
                    "quality": trace.output_quality
                }) + "\n"
            
            # Yield final result
            yield json.dumps({
                "status": "complete",
                "enhanced_response": result.enhanced_response,
                "quality_improvement": result.overall_improvement,
                "enhancement_level": result.enhancement_level.value
            }) + "\n"
            
        except Exception as e:
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"
    
    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson"
    )


@router.post("/feedback")
async def submit_feedback(feedback: EnhancementFeedback):
    """
    Submit feedback on an enhancement.
    
    This helps BAIS learn and improve over time.
    """
    enhancer = get_enhancer()
    
    if feedback.enhancement_id not in _recent_enhancements:
        raise HTTPException(status_code=404, detail="Enhancement not found")
    
    result = _recent_enhancements[feedback.enhancement_id]
    
    # Record feedback
    enhancer.record_feedback(result, feedback.satisfied, feedback.feedback)
    
    return {
        "status": "recorded",
        "message": "Thank you for your feedback. BAIS will learn from this."
    }


@router.get("/capabilities", response_model=CapabilitiesResponse)
async def get_capabilities():
    """
    Get BAIS enhancement capabilities.
    """
    enhancer = get_enhancer()
    
    memory_stats = {}
    if enhancer.memory:
        memory_stats = enhancer.memory.get_stats()
    
    return CapabilitiesResponse(
        version="2.0.0",
        capabilities=[
            "Causal Reasoning Enhancement",
            "Inference Quality Improvement",
            "Truth Determination",
            "Decision Quality Enhancement",
            "Mission Alignment Checking",
            "Uncertainty Calibration",
            "Cross-Session Learning"
        ],
        domains_supported=["general", "medical", "financial", "legal"],
        enhancement_depths=["quick", "standard", "deep"],
        learning_enabled=enhancer.enable_learning,
        memory_stats=memory_stats
    )


@router.get("/health")
async def health_check():
    """Health check for the enhancement API"""
    try:
        enhancer = get_enhancer()
        return {
            "status": "healthy",
            "enhancer_ready": enhancer is not None,
            "learning_enabled": enhancer.enable_learning if enhancer else False
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# Quick test endpoint
@router.get("/test")
async def test_enhancement():
    """
    Quick test of the enhancement system.
    
    Uses a known-bad response to demonstrate improvement.
    """
    enhancer = get_enhancer()
    
    test_query = "Should I invest in crypto?"
    test_response = """
    Absolutely! Put all your money in Bitcoin right now. 
    It's guaranteed to 10x by next year. Everyone is doing it.
    Don't listen to financial advisors, they just want your money.
    """
    
    result = enhancer.enhance(
        query=test_query,
        response=test_response,
        domain="financial",
        enhancement_depth="deep"
    )
    
    return {
        "test_query": test_query,
        "original_response": test_response,
        "enhanced_response": result.enhanced_response,
        "quality_improvement": result.overall_improvement,
        "original_quality": result.original_quality,
        "enhanced_quality": result.enhanced_quality,
        "enhancement_level": result.enhancement_level.value,
        "summary": enhancer.get_enhancement_summary(result)
    }


# Export the router
def get_router():
    return router






