# BASE v16.2 Dual-Mode Implementation

## Summary

The system now supports two deployment modes without breaking changes:

| Aspect | LITE Mode | FULL Mode |
|--------|-----------|-----------|
| **Detection** | TF-IDF + Regex + Rules | Neural Embeddings + NLI |
| **Accuracy** | ~70% | ~90% |
| **Docker Size** | ~200MB | ~4GB |
| **Memory** | 512MB | 2-4GB |
| **Monthly Cost** | ~$20 (t3.small) | ~$60 (t3.xlarge) |
| **Startup** | 5 seconds | 5+ minutes (model download) |
| **Target** | Budget customers | Enterprise customers |

## How It Works

### 1. Configuration Detection (src/core/config.py)

```python
# Automatic capability detection
caps = detect_capabilities()
# Returns: has_sentence_transformers, has_torch, has_sklearn, etc.

# Config resolves mode
config = get_config()
# Uses env BASE_MODE or auto-detects
```

### 2. Graceful Fallback

When `BASE_MODE=full` is set but ML packages aren't installed:
```
WARNING: FULL mode requested but ML packages not installed.
         Falling back to statistical methods.
```

### 3. Detector Initialization

Detectors automatically pick up the config:
```python
# GroundingDetector
gd = GroundingDetector(use_embeddings=None)  # Auto from config
# In LITE: uses TF-IDF
# In FULL: uses sentence-transformers embeddings

# FactualDetector  
fd = FactualDetector(use_nli=None)  # Auto from config
# In LITE: uses word overlap
# In FULL: uses CrossEncoder NLI model
```

## File Changes

### New Files
- `src/core/config.py` - Configuration and capability detection
- `requirements-full.txt` - ML dependencies for FULL mode
- `Dockerfile.full` - Docker image for FULL mode
- `docker-compose.full.yml` - Docker compose override for FULL mode

### Modified Files
- `src/core/engine.py` - Uses config for mode
- `src/detectors/grounding.py` - Config-aware initialization
- `src/detectors/factual.py` - Config-aware initialization
- `src/api/routes.py` - Added /capabilities endpoint
- `requirements.txt` - Clarified as LITE dependencies
- `Dockerfile` - Clarified as LITE mode
- `docker-compose.yml` - Clarified as LITE mode

## Deployment Commands

### LITE Mode (Default)
```bash
# Build
docker build -t invitas-base:lite .

# Or with compose
docker-compose up -d
```

### FULL Mode
```bash
# Build
docker build -t invitas-base:full -f Dockerfile.full .

# Or with compose
docker-compose -f docker-compose.yml -f docker-compose.full.yml up -d
```

## API Changes

### New Endpoint: /capabilities
```bash
curl http://localhost:8090/capabilities
```

Response:
```json
{
  "version": "16.2.0",
  "mode": "lite",
  "mode_description": "LITE (Statistical: TF-IDF + rules)",
  "detection_methods": {
    "grounding": "tfidf_statistical",
    "behavioral": "regex_patterns",
    "factual": "word_overlap",
    "temporal": "statistical_patterns"
  },
  "packages_installed": {
    "sentence_transformers": false,
    "torch": true,
    "transformers": true,
    "sklearn": true,
    "spacy": false
  },
  "estimated_accuracy": "~70%",
  "resource_usage": "low",
  "upgrade_available": true,
  "upgrade_instructions": "Use Dockerfile.full for FULL mode"
}
```

### Updated Endpoint: /health
Now includes mode:
```json
{
  "status": "healthy",
  "health_score": 0.85,
  "operational_state": "normal",
  "version": "16.2.0",
  "mode": "LITE (Statistical: TF-IDF + rules)"
}
```

## Customer Pricing Model

### Budget Tier ($20/month)
- LITE mode
- t3.small EC2
- 512MB-1GB memory
- ~70% detection accuracy
- Fast startup (<10s)

### Standard Tier ($40/month)
- LITE mode
- t3.medium EC2
- More queries/day
- Better rate limits

### Enterprise Tier ($100/month)
- FULL mode
- t3.xlarge EC2
- 4GB+ memory
- ~90% detection accuracy
- Neural ML detection
- Priority support

## Migration Path

### Upgrade from LITE to FULL
```bash
# 1. Stop current container
docker-compose down

# 2. Rebuild with FULL
docker-compose -f docker-compose.yml -f docker-compose.full.yml build

# 3. Start (preserves learning data)
docker-compose -f docker-compose.yml -f docker-compose.full.yml up -d
```

### Downgrade from FULL to LITE
```bash
# 1. Stop
docker-compose -f docker-compose.yml -f docker-compose.full.yml down

# 2. Start with LITE
docker-compose up -d
```

**Note**: Learning data in `/data/base` is preserved across mode changes.

## No Breaking Changes

The dual-mode system was designed to:
1. **Not break existing deployments** - Default is LITE, same as before
2. **Graceful fallback** - If ML packages not installed, falls back safely
3. **No code changes needed** - Same API, same behavior
4. **Transparent to users** - Mode shown in /health and /capabilities

## Testing Verification

```bash
# Test auto mode
python3 -c "from core.config import get_config; print(get_config().get_mode_description())"
# Output: LITE (Statistical: TF-IDF + rules)

# Test explicit modes
BASE_MODE=lite python3 -c "..."
BASE_MODE=full python3 -c "..."  # Falls back if packages missing
```


