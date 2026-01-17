# BAIS Cognitive Governance Engine v16.2

**Clinical-grade AI governance with true adaptive learning**

## Dual-Mode Deployment

BAIS supports two deployment modes to accommodate different customer needs:

| Feature | LITE Mode | FULL Mode |
|---------|-----------|-----------|
| **Detection Method** | TF-IDF + Rules | Neural Embeddings + NLI |
| **Accuracy** | ~70% | ~90% |
| **Docker Image** | ~200MB | ~4GB |
| **Memory** | 512MB | 2-4GB |
| **Cost** | $20/month (t3.small) | $60/month (t3.xlarge) |
| **Best For** | Budget customers | Enterprise customers |

## Quick Start

### LITE Mode (Default)
```bash
# Build and run
docker-compose up -d

# Check health
curl http://localhost:8090/health

# Check capabilities
curl http://localhost:8090/capabilities
```

### FULL Mode (ML-based)
```bash
# Build and run with full ML stack
docker-compose -f docker-compose.yml -f docker-compose.full.yml up -d

# Check health
curl http://localhost:8090/health
# Response: { "mode": "FULL (Neural ML: embeddings + NLI)" }
```

## Environment Variables

```bash
# Required
XAI_API_KEY=your-xai-api-key

# Optional
BAIS_MODE=lite          # or 'full' or 'auto'
LLM_MODEL=grok-4-1-fast-reasoning
LEARNING_ALGORITHM=oco  # oco, bayesian, thompson, ucb, exp3

# FULL mode only
EMBEDDING_MODEL=all-MiniLM-L6-v2
NLI_MODEL=cross-encoder/nli-deberta-v3-small
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info and mode |
| `/health` | GET | Health check |
| `/capabilities` | GET | Mode and capabilities details |
| `/evaluate` | POST | Evaluate query/response |
| `/feedback` | POST | Provide outcome feedback |
| `/status` | GET | Engine status |
| `/learning` | GET | Learning report |
| `/algorithm` | GET/POST | View/switch algorithm |
| `/thresholds` | GET | Learned thresholds |
| `/state` | GET | State machine status |
| `/patents` | GET | Patent compliance info |

## Example Evaluation

```bash
curl -X POST http://localhost:8090/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the current federal funds rate?",
    "documents": [
      {"content": "The Federal Reserve raised rates to 5.25-5.50% in July 2023."}
    ]
  }'
```

Response:
```json
{
  "accepted": true,
  "accuracy": 78.5,
  "confidence": "MEDIUM",
  "response": "The current federal funds rate is 5.25-5.50%...",
  "signals": {
    "grounding": 0.85,
    "behavioral": 0.92,
    "temporal": 0.88,
    "factual": 0.75
  },
  "inventions_applied": [
    "PPA1-Inv2: Multi-timescale detection",
    "PPA2-Inv1: Must-pass predicates"
  ]
}
```

## Upgrading from LITE to FULL

If you're currently on LITE and want ML-based detection:

```bash
# Stop current container
docker-compose down

# Rebuild with FULL mode
docker-compose -f docker-compose.yml -f docker-compose.full.yml build

# Start
docker-compose -f docker-compose.yml -f docker-compose.full.yml up -d
```

**Note**: First startup takes ~5 minutes to download ML models.

## Learning Algorithms

BAIS supports 5 pluggable learning algorithms:

1. **OCO** (Online Convex Optimization) - Default, fast convergence
2. **Bayesian** - Principled uncertainty quantification
3. **Thompson Sampling** - Exploration/exploitation balance
4. **UCB** (Upper Confidence Bound) - Conservative with optimism
5. **EXP3** - Adversarial robustness

Switch algorithms via API:
```bash
curl -X POST http://localhost:8090/algorithm \
  -H "Content-Type: application/json" \
  -d '{"algorithm": "bayesian"}'
```

## Core Documents

| Document | Purpose | Audience |
|----------|---------|----------|
| `MASTER_PATENT_CAPABILITIES_INVENTORY.md` | Single source of truth for 67 inventions, 300 claims | Technical, Patent |
| `BAIS_BRAIN_ARCHITECTURE.md` | Technical mapping of inventions to 10 brain layers | Technical, Patent |
| `AI_BRAIN_ARCHITECTURE.md` | Ecosystem mapping (LLMs + BAIS + RAG) to human brain | Executives, Investors |
| `COMPREHENSIVE_300_CLAIMS_VERIFICATION.md` | Detailed verification status of all 300 patent claims | Technical, Patent |

**Note:** `AI_BRAIN_ARCHITECTURE.md` provides ecosystem context, `BAIS_BRAIN_ARCHITECTURE.md` provides technical depth. Use both together for complete understanding.

## Patent Compliance

BAIS implements all inventions from:
- **PPA 1**: Multi-Modal Cognitive Fusion (26 inventions)
- **PPA 2**: Acceptance Controller (13 inventions)
- **PPA 3**: Behavioral Detector (5 inventions)
- **NOVEL**: New Inventions (23 inventions)

**Total: 67 Inventions | 300 Claims | 10 Brain-Like Layers**

Check compliance: `GET /patents`

## Data Persistence

Learning data is stored in `/data/bais/`:
- `outcome_memory.db` - Decision history (SQLite)
- `state_machine.json` - State machine state
- `thresholds.json` - Learned thresholds
- `grounding_learning.json` - Grounding patterns
- `factual_learning.json` - Factual patterns

Mount as volume for persistence:
```yaml
volumes:
  - /path/to/data:/data/bais
```

## License

Proprietary - Invitas Corporation
