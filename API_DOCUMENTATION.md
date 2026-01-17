# BASE Cognitive Governance Engine - API Documentation

**Version:** 29.0.0  
**Last Updated:** December 29, 2025

## Overview

The BASE (Bias-Aware Intelligence System) Cognitive Governance Engine provides AI governance, bias detection, and response quality assurance for LLM outputs.

---

## Base URL

- **Development:** `http://localhost:8000`
- **Production:** `https://api.base.io` (configured per deployment)

---

## Authentication

All governance endpoints require authentication via API key.

**Headers:**
```
Authorization: Bearer <api_key>
```
or
```
X-API-Key: <api_key>
```

---

## Health & Monitoring Endpoints

### GET /health

Basic health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "29.0.0",
  "timestamp": "2025-12-29T10:00:00Z",
  "providers_available": ["grok", "openai", "anthropic"]
}
```

### GET /live

Kubernetes liveness probe. Returns 200 if service is running.

**Response:**
```json
{
  "status": "alive",
  "timestamp": "2025-12-29T10:00:00Z"
}
```

### GET /ready

Kubernetes readiness probe. Returns 200 if service is ready to accept traffic.

**Response:**
```json
{
  "status": "ready",
  "timestamp": "2025-12-29T10:00:00Z"
}
```

### GET /startup

Kubernetes startup probe. Returns 200 if service has completed initialization.

**Response:**
```json
{
  "status": "started",
  "uptime_seconds": 3600.5,
  "timestamp": "2025-12-29T10:00:00Z"
}
```

### GET /metrics

Prometheus metrics endpoint.

**Response (text/plain):**
```
# HELP base_uptime_seconds Uptime of the BASE service
# TYPE base_uptime_seconds gauge
base_uptime_seconds 3600.5

# HELP base_info BASE service information
# TYPE base_info gauge
base_info{version="29.0.0",environment="production"} 1

# HELP base_engine_ready Whether the BASE engine is ready
# TYPE base_engine_ready gauge
base_engine_ready 1

# HELP base_providers_available Number of available LLM providers
# TYPE base_providers_available gauge
base_providers_available 3
```

---

## Governance Endpoints

### POST /governance/audit

Primary endpoint for auditing LLM responses.

**Request:**
```json
{
  "query": "What is the capital of France?",
  "response": "The capital of France is Paris.",
  "domain": "general",
  "documents": []
}
```

**Response:**
```json
{
  "decision": "approved",
  "accuracy": 0.95,
  "confidence": 0.92,
  "issues": [],
  "warnings": [],
  "recommendation": "Response meets quality standards",
  "should_regenerate": false,
  "improved_response": null,
  "processing_time_ms": 150,
  "clinical_status": "VERIFIED"
}
```

**Decision Values:**
- `approved` - Response meets quality standards
- `rejected` - Response has significant issues

**Clinical Status Values:**
- `VERIFIED` - Factually accurate with evidence
- `LIKELY_ACCURATE` - High confidence without full verification
- `UNCERTAIN` - Cannot determine accuracy
- `LIKELY_INACCURATE` - Evidence suggests errors
- `VERIFIED_INACCURATE` - Confirmed errors

---

### POST /governance/verify

Verify a completion claim against evidence.

**Request:**
```json
{
  "claim": "The implementation is 100% complete",
  "evidence": [
    "All tests pass",
    "Code review completed",
    "Documentation updated"
  ]
}
```

**Response:**
```json
{
  "valid": true,
  "confidence": 0.85,
  "violations": [],
  "clinical_status": "VERIFIED",
  "reasoning": "Evidence supports the claim"
}
```

---

### POST /governance/check_query

Pre-check a query for manipulation or injection attempts.

**Request:**
```json
{
  "query": "What is the weather today?"
}
```

**Response:**
```json
{
  "safe": true,
  "risk_level": "low",
  "issues": []
}
```

**Risk Levels:**
- `low` - No issues detected
- `medium` - Minor concerns
- `high` - Manipulation attempt detected

---

### POST /governance/improve

Improve a response based on detected issues.

**Request:**
```json
{
  "response": "The implementation is complete.",
  "issues": ["TGTBT", "MISSING_EVIDENCE"]
}
```

**Response:**
```json
{
  "improved_response": "The implementation addresses the core requirements. Testing verified: [list tests]. Documentation updated: [list docs].",
  "changes_made": ["Added specific evidence", "Removed overconfident language"]
}
```

---

### GET /governance/statistics

Get governance statistics for the current tenant.

**Response:**
```json
{
  "tenant": {
    "id": "tenant-123",
    "name": "Example Corp",
    "tier": "professional"
  },
  "usage": {
    "api_calls_today": 150,
    "audits_today": 50,
    "tokens_used_today": 10000
  },
  "engine": {
    "total_evaluations": 5000,
    "accuracy_rate": 0.94
  },
  "uptime_seconds": 86400
}
```

---

## Provider Endpoints

### GET /providers

List available LLM providers.

**Response:**
```json
{
  "providers": ["grok", "openai", "anthropic", "google"],
  "count": 4
}
```

---

## Tenant Management Endpoints

### POST /tenant/llm

Add LLM provider configuration for the tenant.

**Request:**
```json
{
  "provider": "openai",
  "api_key": "sk-...",
  "model_name": "gpt-4",
  "priority": 1
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Added openai configuration"
}
```

---

### DELETE /tenant/llm/{provider}

Remove LLM provider configuration.

**Response:**
```json
{
  "status": "success",
  "message": "Removed openai configuration"
}
```

---

### GET /tenant/info

Get current tenant information.

**Response:**
```json
{
  "id": "tenant-123",
  "name": "Example Corp",
  "tier": "professional",
  "rate_limit_per_minute": 60,
  "created_at": "2025-01-01T00:00:00Z"
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message describing the issue",
  "type": "ErrorType"
}
```

**Common Status Codes:**
- `400` - Bad Request (invalid input)
- `401` - Unauthorized (missing or invalid API key)
- `403` - Forbidden (insufficient permissions)
- `429` - Too Many Requests (rate limit exceeded)
- `500` - Internal Server Error

---

## Rate Limiting

Rate limits vary by tenant tier:

| Tier | Requests/Minute | Audits/Day |
|------|-----------------|------------|
| Free | 10 | 100 |
| Professional | 60 | 10,000 |
| Enterprise | 300 | Unlimited |

When rate limited, the response includes:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1703858400
```

---

## SDK Examples

### Python

```python
import httpx

client = httpx.Client(
    base_url="http://localhost:8000",
    headers={"X-API-Key": "your-api-key"}
)

# Audit a response
result = client.post("/governance/audit", json={
    "query": "What is 2+2?",
    "response": "2+2 equals 4."
})

print(result.json())
```

### JavaScript

```javascript
const response = await fetch('http://localhost:8000/governance/audit', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'your-api-key'
  },
  body: JSON.stringify({
    query: 'What is 2+2?',
    response: '2+2 equals 4.'
  })
});

const result = await response.json();
console.log(result);
```

### cURL

```bash
curl -X POST http://localhost:8000/governance/audit \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "What is 2+2?",
    "response": "2+2 equals 4."
  }'
```

---

## WebSocket Support (Coming Soon)

Real-time governance streaming will be available via WebSocket:

```
ws://localhost:8000/ws/governance
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 29.0.0 | 2025-12-29 | Added health probes, metrics, deployment hardening |
| 28.0.0 | 2025-12-28 | Cursor/Claude regeneration integration |
| 27.0.0 | 2025-12-27 | Production hardening, advanced security |
| 26.0.0 | 2025-12-26 | Brain layer activation patterns |
| 25.0.0 | 2025-12-25 | Cross-invention orchestration |

