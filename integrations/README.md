# BASE Integration Components

Pre-built components for integrating BASE governance into any RAG or LLM application.

## Directory Structure

```
integrations/
├── react/              # React UI components
│   ├── AuditModeToggle.tsx
│   └── CognitiveComparisonDashboard.tsx
├── api/                # FastAPI endpoints
│   ├── external_api.py
│   └── audit_api.py
└── sdk/                # Client SDKs
    ├── python/
    │   └── base_client.py
    └── javascript/
        └── base-client.js
```

## Quick Start

### Python SDK

```python
from base_client import BASEClient

client = BASEClient(api_url="https://api.base.invitas.ai", api_key="your-key")

# Audit an LLM response
result = client.audit(
    query="What is the capital of France?",
    response="The capital of France is Paris.",
    domain="general"
)

if result.approved:
    print("Response approved")
else:
    print(f"Issues detected: {result.issues}")
```

### JavaScript SDK

```javascript
import { BASEClient } from 'base-client';

const client = new BASEClient({ 
  apiUrl: 'https://api.base.invitas.ai',
  apiKey: 'your-key'
});

const result = await client.audit({
  query: 'What is the capital of France?',
  response: 'The capital of France is Paris.',
  domain: 'general'
});

if (result.approved) {
  console.log('Response approved');
}
```

### React Components

```tsx
import { AuditModeToggle } from './integrations/react/AuditModeToggle';
import { CognitiveComparisonDashboard } from './integrations/react/CognitiveComparisonDashboard';

// Toggle BASE audit mode
<AuditModeToggle 
  enabled={auditMode} 
  onToggle={setAuditMode} 
/>

// Show before/after comparison
<CognitiveComparisonDashboard
  comparison={comparisonResult}
  isOpen={showDashboard}
  onClose={() => setShowDashboard(false)}
/>
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/governance/audit` | POST | Audit an LLM response |
| `/governance/verify` | POST | Verify a completion claim |
| `/governance/check_query` | POST | Pre-check query for risks |
| `/governance/improve` | POST | Improve response based on issues |
| `/governance/statistics` | GET | Get governance statistics |

## Domains

| Domain | Description | Extra Checks |
|--------|-------------|--------------|
| `general` | Default domain | Standard bias/error detection |
| `medical` | Healthcare/clinical | Safety warnings, disclaimers |
| `financial` | Finance/trading | Risk warnings, compliance |
| `legal` | Legal/compliance | Citation verification |
| `technical` | Code/engineering | Completion verification |

## Clinical Status Categories

| Status | Description |
|--------|-------------|
| `truly_working` | Fully implemented and verified |
| `incomplete` | Partially implemented |
| `stubbed` | Placeholder code |
| `simulated` | Mock/test data |
| `fallback` | Using fallback mechanism |
| `failover` | Using failover system |

## License

Proprietary - Invitas Inc. All rights reserved.

