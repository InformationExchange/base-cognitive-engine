# BASE → BASE Rename Migration Plan

## Overview
Renaming BASE (Bias-Aware Intelligent System) to BASE (Bias-Aware System Engine)

## Phase 1: Code Changes (Before Repo Rename)

### 1.1 File Renames
```bash
# Files with 'base' in name → 'base'
base_client.py → base_client.py
base-client.js → base-client.js
base_v2_orchestrator.py → base_v2_orchestrator.py
start_base_governance.sh → start_base_governance.sh
# etc.
```

### 1.2 Code Reference Updates
- `BASE` → `BASE` (278 files)
- `base` → `base` 
- `Base` → `Base`
- Class names: `BASEMCPServer` → `BASEMCPServer`
- Variables: `base_result` → `base_result`

### 1.3 Docker Updates
```yaml
# docker-compose.yml
services:
  base-governance:  # was: base-governance
    image: base-cognitive-engine:latest  # was: base-governance
```

## Phase 2: GitHub Repository Rename

### 2.1 Rename Repository
- Old: `InformationExchange/base-cognitive-engine`
- New: `InformationExchange/base-cognitive-engine`

### 2.2 GitHub Handles Redirects
GitHub automatically redirects old URLs for 1 year:
- `https://github.com/.../base-cognitive-engine` → `https://github.com/.../base-cognitive-engine`

## Phase 3: Backwards Compatibility

### 3.1 Import Aliases (Python)
```python
# In __init__.py files, add aliases:
from .base_client import BASEClient
BASEClient = BASEClient  # Backwards compatibility alias
```

### 3.2 Docker Multi-Tag
```bash
# Tag both names during transition
docker tag base-cognitive-engine:latest base-governance:latest
docker push base-cognitive-engine:latest
docker push base-governance:latest  # Legacy tag
```

### 3.3 Environment Variable Compatibility
```python
# Accept both old and new env var names
BASE_PORT = os.environ.get("BASE_PORT") or os.environ.get("BASE_PORT", "8000")
```

## Phase 4: Cleanup (After 3-6 months)

- Remove backwards compatibility aliases
- Remove legacy Docker tags
- Update all documentation references

## Execution Commands

```bash
# Step 1: Rename files
find . -name "*base*" -exec rename 's/base/base/g' {} \;
find . -name "*BASE*" -exec rename 's/BASE/BASE/g' {} \;

# Step 2: Update code references
find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.yml" -o -name "*.json" -o -name "*.sh" -o -name "*.js" -o -name "*.ts" \) \
  -exec sed -i '' 's/BASE/BASE/g; s/base/base/g; s/Base/Base/g' {} \;

# Step 3: Commit changes
git add -A
git commit -m "refactor: Rename BASE to BASE

- BASE (Bias-Aware Intelligent System) → BASE (Bias-Aware System Engine)
- All code references updated
- File names updated
- Backwards compatibility aliases added
"

# Step 4: Push to current repo
git push origin main

# Step 5: Rename GitHub repo (via gh CLI or web UI)
gh repo rename base-cognitive-engine
```

## Risk Mitigation

1. **Git History**: Preserved (rename, not delete/recreate)
2. **URL Redirects**: GitHub handles for 1 year
3. **Docker Images**: Multi-tag strategy
4. **Import Aliases**: Backwards compatible
5. **Env Vars**: Accept both old and new names
