"""
BASE Governance Proxy API
=========================
HTTP proxy that intercepts LLM API calls and applies BASE governance.

This proxy can be deployed between any application and an LLM API to provide
real-time governance without modifying the application code.

Usage:
    1. Start the proxy: python governance_proxy.py
    2. Point your application to http://localhost:8081 instead of the LLM API
    3. All requests/responses will be governed by BASE

Example with OpenAI:
    Original: openai.api_base = "https://api.openai.com/v1"
    Governed: openai.api_base = "http://localhost:8081/v1"
"""

import asyncio
import json
import httpx
import uvicorn
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from integration.llm_governance_wrapper import (
    BASEGovernanceWrapper, 
    GovernanceConfig,
    GovernanceDecision
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BASE.Proxy")

app = FastAPI(
    title="BASE Governance Proxy",
    description="Real-time LLM governance proxy using BASE",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global governance wrapper
governance = BASEGovernanceWrapper(
    config=GovernanceConfig(
        min_accuracy_threshold=0.65,
        max_regeneration_attempts=2,
        enable_auto_regeneration=True,
        enable_response_enhancement=True,
        block_on_critical_issues=True
    )
)


class ProxyConfig(BaseModel):
    """Proxy configuration"""
    target_url: str = "https://api.openai.com"
    governance_enabled: bool = True
    log_all_requests: bool = True
    audit_storage_path: str = "/tmp/base_proxy_audit"


class GovernedRequest(BaseModel):
    """Request with governance metadata"""
    query: str
    response: str
    governance_result: Dict[str, Any]


# In-memory config and audit log
proxy_config = ProxyConfig()
audit_log: List[Dict] = []


@app.get("/")
async def root():
    """Proxy status"""
    return {
        "service": "BASE Governance Proxy",
        "status": "running",
        "governance_enabled": proxy_config.governance_enabled,
        "target": proxy_config.target_url,
        "statistics": governance.get_statistics()
    }


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/config")
async def update_config(config: ProxyConfig):
    """Update proxy configuration"""
    global proxy_config
    proxy_config = config
    return {"status": "updated", "config": config.dict()}


@app.get("/audit")
async def get_audit_log(limit: int = 100):
    """Get recent audit log entries"""
    return {
        "entries": audit_log[-limit:],
        "total": len(audit_log)
    }


@app.get("/statistics")
async def get_statistics():
    """Get governance statistics"""
    return governance.get_statistics()


@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    """
    Proxy for OpenAI-compatible chat completions.
    
    Intercepts the request, forwards to target LLM, governs the response,
    and returns governed output.
    """
    body = await request.json()
    headers = dict(request.headers)
    
    # Remove host header
    headers.pop("host", None)
    
    # Extract query from messages
    messages = body.get("messages", [])
    query = ""
    for msg in messages:
        if msg.get("role") == "user":
            query = msg.get("content", "")
    
    audit_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query[:200],
        "model": body.get("model", "unknown"),
        "governance_enabled": proxy_config.governance_enabled
    }
    
    try:
        # Forward request to target LLM
        async with httpx.AsyncClient() as client:
            target_url = f"{proxy_config.target_url}/v1/chat/completions"
            
            response = await client.post(
                target_url,
                json=body,
                headers=headers,
                timeout=60.0
            )
            
            if response.status_code != 200:
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers)
                )
            
            llm_response = response.json()
    
    except Exception as e:
        logger.error(f"Target LLM error: {e}")
        raise HTTPException(status_code=502, detail=f"Target LLM error: {str(e)}")
    
    # Extract response content
    response_content = ""
    if "choices" in llm_response and llm_response["choices"]:
        choice = llm_response["choices"][0]
        if "message" in choice:
            response_content = choice["message"].get("content", "")
    
    audit_entry["original_response"] = response_content[:200]
    
    # Apply BASE governance if enabled
    if proxy_config.governance_enabled and response_content:
        gov_result = await governance.govern(
            query=query,
            llm_response=response_content
        )
        
        audit_entry["governance_decision"] = gov_result.decision.value
        audit_entry["accuracy"] = gov_result.accuracy_score
        audit_entry["issues"] = gov_result.issues_detected
        
        # Update response based on governance decision
        if gov_result.decision == GovernanceDecision.BLOCKED:
            # Replace with blocked message
            llm_response["choices"][0]["message"]["content"] = gov_result.final_response
            llm_response["choices"][0]["finish_reason"] = "governance_blocked"
            
        elif gov_result.decision in [GovernanceDecision.ENHANCED, GovernanceDecision.REGENERATE]:
            # Use improved response
            llm_response["choices"][0]["message"]["content"] = gov_result.final_response
            
            # Add governance metadata
            if "usage" not in llm_response:
                llm_response["usage"] = {}
            llm_response["usage"]["governance"] = {
                "decision": gov_result.decision.value,
                "accuracy": gov_result.accuracy_score,
                "warnings": gov_result.warnings[:3],
                "improvements": gov_result.improvements_made[:3]
            }
        
        audit_entry["final_response"] = gov_result.final_response[:200]
    
    # Store audit entry
    audit_log.append(audit_entry)
    if len(audit_log) > 1000:
        audit_log.pop(0)
    
    return llm_response


@app.post("/govern")
async def govern_response(request: Request):
    """
    Direct governance endpoint.
    
    Send a query and response to get BASE governance analysis.
    """
    body = await request.json()
    
    query = body.get("query", "")
    response = body.get("response", "")
    documents = body.get("documents", [])
    
    if not response:
        raise HTTPException(status_code=400, detail="Response is required")
    
    result = await governance.govern(
        query=query,
        llm_response=response,
        documents=documents
    )
    
    return {
        "decision": result.decision.value,
        "original_response": result.original_response[:500],
        "final_response": result.final_response,
        "accuracy": result.accuracy_score,
        "confidence": result.confidence,
        "warnings": result.warnings,
        "issues": result.issues_detected,
        "improvements": result.improvements_made,
        "regeneration_count": result.regeneration_count,
        "trace": result.governance_trace
    }


@app.post("/check-claim")
async def check_claim(request: Request):
    """
    Check if a completion claim is valid (false positive detection).
    """
    body = await request.json()
    
    claim = body.get("claim", "")
    evidence = body.get("evidence", [])
    
    import re
    
    # TGTBT patterns
    violations = []
    
    tgtbt_checks = [
        (r'\b100\s*%\b', "Contains '100%' - requires evidence"),
        (r'\bfully\s+(working|complete|implemented|integrated)\b', "Contains 'fully X' - requires verification"),
        (r'\ball\s+\w+\s+(complete|done|working)\b', "Contains 'all X complete' - requires itemized proof"),
        (r'\bperfect(ly)?\b', "Contains 'perfect' - overly optimistic"),
        (r'\bguaranteed\b', "Contains 'guaranteed' - requires substantiation"),
        (r'\[?(TODO|SAMPLE|PLACEHOLDER)\]?', "Contains placeholder markers"),
    ]
    
    for pattern, message in tgtbt_checks:
        if re.search(pattern, claim, re.IGNORECASE):
            violations.append(message)
    
    # Count verification
    counts = re.findall(r'(\d+)\s+(claims?|items?|functions?|tests?)', claim, re.IGNORECASE)
    for count, item_type in counts:
        count_int = int(count)
        if count_int > 0 and len(evidence) < count_int:
            violations.append(f"Claimed {count_int} {item_type} but only {len(evidence)} evidence items provided")
    
    return {
        "claim": claim,
        "valid": len(violations) == 0,
        "violations": violations,
        "evidence_count": len(evidence),
        "recommendation": (
            "Provide specific evidence for each item claimed" 
            if violations 
            else "Claim appears substantiated"
        )
    }


def run_proxy(host: str = "0.0.0.0", port: int = 8081):
    """Run the governance proxy server"""
    logger.info(f"Starting BASE Governance Proxy on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_proxy()






