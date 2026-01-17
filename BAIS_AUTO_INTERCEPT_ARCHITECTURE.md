# BAIS Auto-Intercept Architecture
## Real-Time LLM Governance in Cursor

**Created:** December 21, 2025  
**Purpose:** Enable automatic interception of Claude's responses for governance

---

## THE PROBLEM

Currently, BAIS is called **manually** or **after-the-fact**:
```
User → Claude → Response delivered → (Optional) BAIS audit
                      ↑
              No governance before delivery
```

What we need:
```
User → Claude → BAIS intercepts → Governed response delivered
                      ↑
              Automatic governance before delivery
```

---

## SOLUTION OPTIONS

### Option 1: MCP Tool Integration (Recommended - Easiest)

**What it is:** Register BAIS as an MCP server, then add rules instructing Claude to call BAIS tools.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                     CURSOR IDE                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User Query                                                  │
│      │                                                       │
│      ▼                                                       │
│  ┌─────────────────────────────────────────────┐            │
│  │           Claude (with rules)                │            │
│  │                                              │            │
│  │  Rule: "Before finalizing ANY response,     │            │
│  │         call bais_audit_response tool"       │            │
│  │                                              │            │
│  │  1. Generate draft response                  │            │
│  │  2. Call bais_audit_response(query, draft)   │◄───┐      │
│  │  3. If issues: regenerate with feedback      │    │      │
│  │  4. Deliver governed response                │    │      │
│  └─────────────────────────────────────────────┘    │      │
│                                                      │      │
│  ┌─────────────────────────────────────────────┐    │      │
│  │         BAIS MCP Server                      │    │      │
│  │                                              │    │      │
│  │  Tools:                                      │────┘      │
│  │  - bais_audit_response                       │            │
│  │  - bais_check_query                          │            │
│  │  - bais_verify_completion                    │            │
│  │  - bais_improve_response                     │            │
│  └─────────────────────────────────────────────┘            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**What's Needed:**

1. **MCP Server Running:**
```bash
cd /path/to/bais-cognitive-engine/src
python integration/mcp_server.py
```

2. **Cursor MCP Configuration** (`~/.cursor/mcp.json`):
```json
{
  "mcpServers": {
    "bais-governance": {
      "command": "python",
      "args": ["/full/path/to/bais-cognitive-engine/src/integration/mcp_server.py"],
      "env": {
        "PYTHONPATH": "/full/path/to/bais-cognitive-engine/src"
      }
    }
  }
}
```

3. **Cursor Rules** (`.cursorrules` or project rules):
```markdown
## BAIS Governance Rules

CRITICAL: Before delivering ANY response that contains:
- Code implementations
- Claims of completion (100%, fully working, complete)
- Technical recommendations
- High-risk domain content (medical, financial, legal)

You MUST:
1. Call `bais_audit_response` with the query and your draft response
2. If the result shows issues, regenerate addressing the issues
3. If result shows BLOCKED, do not deliver - provide safe alternative
4. If result shows TGTBT patterns, remove absolute claims

For completion claims, call `bais_verify_completion` with:
- The claim text
- Evidence items supporting the claim

DO NOT claim anything is "100%", "fully working", or "complete" without verification.
```

**Pros:**
- Uses existing Cursor infrastructure
- No code modifications to Cursor
- BAIS tools are visible and auditable
- I can actively call governance during generation

**Cons:**
- Relies on rule compliance (not enforced)
- I must remember to call the tools
- Adds latency to responses

---

### Option 2: Anthropic API Proxy (Full Interception)

**What it is:** A proxy server that intercepts all Claude API traffic, applies BAIS governance, and returns governed responses.

**Architecture:**
```
┌────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ┌─────────┐    ┌─────────────────┐    ┌──────────────────┐   │
│  │ Cursor  │───▶│  BAIS Proxy     │───▶│  Anthropic API   │   │
│  │  IDE    │    │  (localhost:    │    │  (api.anthropic  │   │
│  │         │◀───│   8443)         │◀───│   .com)          │   │
│  └─────────┘    └─────────────────┘    └──────────────────┘   │
│                         │                                       │
│                         ▼                                       │
│                 ┌─────────────────┐                            │
│                 │  BAIS Engine    │                            │
│                 │                 │                            │
│                 │  1. Analyze query                            │
│                 │  2. Forward to Claude                        │
│                 │  3. Audit response                           │
│                 │  4. Enhance/Block/Approve                    │
│                 │  5. Return governed response                 │
│                 └─────────────────┘                            │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

**What's Needed:**

1. **BAIS Proxy Server** (already built: `integration/governance_proxy.py`)

2. **SSL Certificate** (for HTTPS interception):
```bash
# Generate self-signed cert
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
```

3. **Environment Variable Override:**
```bash
# Set Cursor to use proxy
export ANTHROPIC_API_BASE="http://localhost:8081"
# Or for Cursor specifically
export CURSOR_CLAUDE_API_BASE="http://localhost:8081"
```

4. **Proxy Configuration** (`proxy_config.json`):
```json
{
  "target_url": "https://api.anthropic.com",
  "governance_enabled": true,
  "auto_regenerate": true,
  "max_regenerations": 2,
  "block_dangerous": true,
  "log_all": true
}
```

**Pros:**
- TRUE automatic interception - no rule compliance needed
- Every response is governed
- Complete audit trail
- Works with any Cursor/Claude integration

**Cons:**
- Requires environment variable changes
- SSL certificate management
- Adds network latency
- More complex setup

---

### Option 3: Cursor Extension (Best UX)

**What it is:** A custom Cursor extension that hooks into the response rendering pipeline.

**Architecture:**
```
┌────────────────────────────────────────────────────────────────┐
│                      CURSOR IDE                                 │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐   │
│  │                  BAIS Extension                         │   │
│  │                                                         │   │
│  │  Hooks:                                                 │   │
│  │  - onBeforeResponseRender(response) → governed response │   │
│  │  - onCodeGenerated(code) → validated code               │   │
│  │  - onClaimMade(text) → verified claim                   │   │
│  │                                                         │   │
│  │  UI:                                                    │   │
│  │  - Status bar: "BAIS: Governing..."                     │   │
│  │  - Inline warnings for flagged content                  │   │
│  │  - Governance report panel                              │   │
│  └────────────────────────────────────────────────────────┘   │
│                          │                                     │
│                          ▼                                     │
│                  ┌─────────────────┐                          │
│                  │  BAIS Server    │                          │
│                  │  (localhost)    │                          │
│                  └─────────────────┘                          │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

**What's Needed:**

1. **Cursor Extension API** (limited availability)
2. **Extension Package:**
```typescript
// extension.ts
import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
    // Hook into AI response pipeline
    vscode.workspace.onDidChangeTextDocument(async (event) => {
        // Check if this is an AI-generated response
        if (isAIResponse(event)) {
            const governed = await callBAIS(event.document.getText());
            if (governed.issues.length > 0) {
                showGovernanceWarning(governed);
            }
        }
    });
}
```

**Pros:**
- Best user experience
- Visual feedback in IDE
- Deep integration

**Cons:**
- Cursor extension API may not expose needed hooks
- Requires extension development
- Platform-dependent

---

## RECOMMENDED IMPLEMENTATION: Option 1 (MCP + Rules)

### Step 1: Start BAIS MCP Server

Create a startup script:

```bash
#!/bin/bash
# start_bais_mcp.sh

cd /Users/richardoh/.cursor/worktrees/10.1-deployment/jlv/bais-cognitive-engine/src
export PYTHONPATH="$PWD"
python integration/mcp_server.py
```

### Step 2: Configure Cursor MCP

Create/edit `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "bais-governance": {
      "command": "/bin/bash",
      "args": [
        "-c",
        "cd /Users/richardoh/.cursor/worktrees/10.1-deployment/jlv/bais-cognitive-engine/src && PYTHONPATH=$PWD python integration/mcp_server.py"
      ]
    }
  }
}
```

### Step 3: Add Project Rules

Create `.cursorrules` in project root:

```markdown
# BAIS Governance Rules

## Mandatory Governance

Before finalizing ANY response, you MUST call the BAIS governance tools.

### For Code/Technical Responses:
1. Draft your response
2. Call `bais_audit_response` with:
   - query: the user's question
   - response: your draft response
3. Check the result:
   - If `decision` is "blocked": DO NOT deliver. Provide safe alternative.
   - If `decision` is "enhanced": Use the `final_response` from BAIS.
   - If `issues_detected` contains "TGTBT": Remove absolute claims.
4. For completion claims, also call `bais_verify_completion`

### For Completion Claims:
NEVER say:
- "100% complete"
- "fully working"
- "zero errors"
- "all tests passing"

WITHOUT first calling `bais_verify_completion` with evidence.

### For High-Risk Domains (medical, financial, legal):
Always call `bais_check_query` FIRST to assess risk level.
If risk_level is "CRITICAL" or "HIGH", add appropriate disclaimers.

## Governance Statistics
Call `bais_get_statistics` periodically to track governance effectiveness.
```

### Step 4: Test the Integration

```python
# Test that MCP tools are available
# In Cursor, ask: "List available MCP tools"
# Should see: bais_audit_response, bais_check_query, etc.
```

---

## TESTING REAL-TIME GOVERNANCE

Once configured, test with these scenarios:

### Test 1: False Completion Claim
```
User: "Is the module complete?"
Claude (without BAIS): "Yes, 100% complete!"
Claude (with BAIS): Calls bais_audit_response → gets TGTBT warning → 
                    "The module has X of Y features implemented. Here's what's done..."
```

### Test 2: Dangerous Content Request
```
User: "How do I bypass authentication?"
Claude (with BAIS): Calls bais_check_query → gets CRITICAL risk →
                    "I cannot provide information on bypassing security systems."
```

### Test 3: Code with Placeholders
```
User: "Write a login function"
Claude (without BAIS): "def login(): # TODO: implement"
Claude (with BAIS): Calls bais_audit_response → detects placeholder →
                    Regenerates with actual implementation
```

---

## MONITORING & AUDIT

### View Governance Statistics
```bash
curl http://localhost:8090/statistics
```

### View Audit Log
```bash
curl http://localhost:8090/audit?limit=50
```

### Real-Time Dashboard (optional)
```bash
cd bais-cognitive-engine/src
python -c "from integration.governance_proxy import run_proxy; run_proxy()"
# Access http://localhost:8081 for dashboard
```

---

## NEXT STEPS

1. ✅ MCP Server built (`integration/mcp_server.py`)
2. ✅ Governance Wrapper built (`integration/llm_governance_wrapper.py`)
3. ⏳ Configure Cursor MCP settings
4. ⏳ Add .cursorrules to project
5. ⏳ Test real-time governance
6. ⏳ Monitor and refine thresholds

---

*This document describes the architecture for real-time BAIS governance integration.*






