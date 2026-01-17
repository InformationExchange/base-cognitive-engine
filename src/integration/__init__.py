"""
BASE Integration Module
=======================
Real-time governance integration for LLMs and development tools.

Components:
- LLMGovernanceWrapper: Intercepts and governs LLM responses
- BASEMCPServer: MCP server for Cursor integration
- GovernanceProxy: HTTP proxy for LLM API governance
"""

from integration.llm_governance_wrapper import (
    BASEGovernanceWrapper,
    GovernanceConfig,
    GovernanceResult,
    GovernanceDecision,
    govern_llm_response
)

__all__ = [
    "BASEGovernanceWrapper",
    "GovernanceConfig", 
    "GovernanceResult",
    "GovernanceDecision",
    "govern_llm_response"
]






