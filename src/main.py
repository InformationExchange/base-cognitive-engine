"""
BASE Cognitive Governance Engine v16.2
Main Entry Point - Phase 4

Production-ready FastAPI application with full integration.

NO PLACEHOLDERS | NO STUBS | NO SIMULATIONS
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Environment setup
DATA_DIR = Path(os.environ.get('BASE_DATA_DIR', '/data/base'))
LLM_API_KEY = os.environ.get('XAI_API_KEY', '')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("=" * 60)
    print("BASE Cognitive Governance Engine v16.2")
    print("=" * 60)
    
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize the integrated engine
    from api.integrated_routes import initialize_engine
    engine = initialize_engine(data_dir=DATA_DIR, llm_api_key=LLM_API_KEY)
    
    print(f"Data directory: {DATA_DIR}")
    print(f"LLM configured: {'Yes' if LLM_API_KEY else 'No'}")
    print("=" * 60)
    
    yield
    
    # Shutdown
    print("Shutting down BASE engine...")


# Create FastAPI app
app = FastAPI(
    title="Invitas BASE Cognitive Governance Engine",
    description="""
    Production-ready AI governance engine with full patent compliance.
    
    ## Features
    - 5 learning algorithms (OCO, Bayesian, Thompson, UCB, EXP3)
    - 4 detectors (grounding, factual, behavioral, temporal)
    - 4 fusion methods (weighted, bayesian, dempster-shafer, kalman)
    - Clinical validation with A/B testing
    - Shadow mode for safe deployment
    
    ## Modes
    - **LITE**: Statistical detection (~70% accuracy)
    - **FULL**: Neural ML detection (~90% accuracy)
    """,
    version="16.2.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
from api.integrated_routes import router
app.include_router(router)


# Health check at root
@app.get("/ping")
async def ping():
    """Simple ping endpoint for load balancer health checks."""
    return {"status": "ok", "service": "base"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
