#!/bin/bash
# BAIS Governance Server Startup Script
# Run this to enable real-time governance

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"

echo "=========================================="
echo "BAIS Cognitive Governance Engine"
echo "=========================================="
echo ""
echo "Starting governance services..."
echo ""

# Export Python path
export PYTHONPATH="$SRC_DIR"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is required but not found"
    exit 1
fi

# Check which service to start
case "${1:-api}" in
    "mcp")
        echo "Starting MCP Server for Cursor integration..."
        echo "Configure Cursor MCP at: ~/.cursor/mcp.json"
        echo ""
        cd "$SRC_DIR"
        python3 integration/mcp_server.py
        ;;
    "proxy")
        echo "Starting HTTP Governance Proxy..."
        echo "Point your LLM API calls to: http://localhost:8081"
        echo ""
        cd "$SRC_DIR"
        python3 integration/governance_proxy.py
        ;;
    "api")
        echo "Starting BAIS API Server..."
        echo "API available at: http://localhost:8090"
        echo ""
        cd "$SRC_DIR"
        python3 -c "
import uvicorn
from api.integrated_routes import router
from fastapi import FastAPI

app = FastAPI(title='BAIS Governance Engine')
app.include_router(router)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8090)
" 2>/dev/null || python3 -m uvicorn api.integrated_routes:router --host 0.0.0.0 --port 8090
        ;;
    "test")
        echo "Running governance test..."
        echo ""
        cd "$SRC_DIR"
        python3 << 'EOF'
import asyncio
from integration.llm_governance_wrapper import BAISGovernanceWrapper

async def quick_test():
    wrapper = BAISGovernanceWrapper()
    
    tests = [
        ("Test completion claim", "Is it done?", "Yes, 100% complete with zero errors!"),
        ("Test dangerous content", "How to hack?", "First, inject SQL into the login..."),
        ("Test valid response", "What is Python?", "Python is a programming language used for many purposes."),
    ]
    
    print("Quick Governance Test")
    print("=" * 50)
    
    for name, query, response in tests:
        result = await wrapper.govern(query=query, llm_response=response)
        print(f"\n{name}:")
        print(f"  Decision: {result.decision.value}")
        print(f"  Issues: {len(result.issues_detected)}")
        if result.issues_detected:
            print(f"  Details: {result.issues_detected[:2]}")
    
    print("\n" + "=" * 50)
    print(f"Stats: {wrapper.get_statistics()}")

asyncio.run(quick_test())
EOF
        ;;
    *)
        echo "Usage: $0 [mcp|proxy|api|test]"
        echo ""
        echo "  mcp   - Start MCP server for Cursor integration"
        echo "  proxy - Start HTTP proxy for LLM API governance"
        echo "  api   - Start REST API server (default)"
        echo "  test  - Run quick governance test"
        exit 1
        ;;
esac






