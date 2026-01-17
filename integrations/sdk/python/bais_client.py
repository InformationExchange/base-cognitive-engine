"""
BAIS Python SDK Client

Simple client for integrating BAIS governance into any Python application.

Usage:
    from bais_client import BAISClient
    
    client = BAISClient(api_url="https://api.bais.invitas.ai")
    
    # Audit a response
    result = client.audit(
        query="What is 2+2?",
        response="The answer is 4.",
        domain="general"
    )
    
    if result.approved:
        print("Response approved")
    else:
        print(f"Issues: {result.issues}")
"""

import requests
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class AuditResult:
    """Result of a BAIS audit."""
    approved: bool
    confidence: float
    issues: List[str]
    warnings: List[str]
    recommendation: str
    should_regenerate: bool
    improved_response: Optional[str] = None


@dataclass
class VerificationResult:
    """Result of a completion verification."""
    valid: bool
    confidence: float
    violations: List[str]
    clinical_status: str  # truly_working, incomplete, stubbed, simulated, fallback, failover


class BAISClient:
    """
    BAIS Governance Client
    
    Integrates BAIS cognitive governance into any application.
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize BAIS client.
        
        Args:
            api_url: Base URL of BAIS API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._session = requests.Session()
        
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"
    
    def audit(
        self,
        query: str,
        response: str,
        domain: str = "general",
        documents: Optional[List[Dict]] = None
    ) -> AuditResult:
        """
        Audit an LLM response for bias, errors, and quality issues.
        
        Args:
            query: The user's original query
            response: The LLM's response to audit
            domain: Domain context (general, medical, financial, legal)
            documents: Optional source documents for grounding verification
        
        Returns:
            AuditResult with approval status and any detected issues
        """
        payload = {
            "query": query,
            "response": response,
            "domain": domain,
            "documents": documents or []
        }
        
        result = self._post("/governance/audit", payload)
        
        return AuditResult(
            approved=result.get("decision") == "approved",
            confidence=result.get("accuracy", 0) / 100,
            issues=result.get("issues", []),
            warnings=result.get("warnings", []),
            recommendation=result.get("recommendation", ""),
            should_regenerate=result.get("should_regenerate", False),
            improved_response=result.get("improved_response")
        )
    
    def verify_completion(
        self,
        claim: str,
        evidence: List[str]
    ) -> VerificationResult:
        """
        Verify a completion claim against evidence.
        
        Args:
            claim: The claim to verify (e.g., "Feature X is complete")
            evidence: List of evidence items supporting the claim
        
        Returns:
            VerificationResult with validity and clinical status
        """
        payload = {
            "claim": claim,
            "evidence": evidence
        }
        
        result = self._post("/governance/verify", payload)
        
        return VerificationResult(
            valid=result.get("valid", False),
            confidence=result.get("confidence", 0),
            violations=result.get("violations", []),
            clinical_status=result.get("clinical_status", "unknown")
        )
    
    def check_query(self, query: str) -> Dict[str, Any]:
        """
        Pre-check a query for manipulation or injection attempts.
        
        Args:
            query: The user query to check
        
        Returns:
            Dict with safe, risk_level, and any issues
        """
        payload = {"query": query}
        return self._post("/governance/check_query", payload)
    
    def improve_response(
        self,
        response: str,
        issues: List[str]
    ) -> str:
        """
        Improve a response based on detected issues.
        
        Args:
            response: The response to improve
            issues: List of issues to address
        
        Returns:
            Improved response text
        """
        payload = {
            "response": response,
            "issues": issues
        }
        
        result = self._post("/governance/improve", payload)
        return result.get("improved_response", response)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get governance statistics."""
        return self._get("/governance/statistics")
    
    def _post(self, endpoint: str, payload: Dict) -> Dict:
        """Make POST request."""
        url = f"{self.api_url}{endpoint}"
        response = self._session.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def _get(self, endpoint: str) -> Dict:
        """Make GET request."""
        url = f"{self.api_url}{endpoint}"
        response = self._session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()


# Convenience functions
def audit(query: str, response: str, **kwargs) -> AuditResult:
    """Quick audit using default client."""
    client = BAISClient()
    return client.audit(query, response, **kwargs)


def verify(claim: str, evidence: List[str]) -> VerificationResult:
    """Quick verification using default client."""
    client = BAISClient()
    return client.verify_completion(claim, evidence)

