# BASE PROOF VERIFICATION ORCHESTRATION FLOW
## How Proof-Based Verification is Called and Triggered

**Created:** December 24, 2025  
**Status:** VERIFIED WORKING - Proof gaps detected automatically

---

## ORCHESTRATION PATHWAY

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ENTRY POINT: User Query + LLM Response                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  IntegratedGovernanceEngine.evaluate_and_improve()                           │
│  FILE: src/core/integrated_engine.py                                         │
│  LINE: ~1200                                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1-3: PERCEPTION & BEHAVIORAL DETECTORS                                │
│  ├── grounding_detector.analyze()      → Grounding score                    │
│  ├── factual_detector.analyze()        → Factual score                      │
│  ├── behavioral_detector.detect_all()  → TGTBT, biases (with past-tense!)   │
│  └── temporal_detector.detect()        → Temporal signals                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 6: EVIDENCE DEMAND (PROOF VERIFICATION) ⭐ AUTOMATIC                  │
│  FILE: src/core/integrated_engine.py LINE: 1386                             │
│                                                                              │
│  evidence_demand.run_full_verification(response, query, workspace_path)      │
│         │                                                                    │
│         ▼                                                                    │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  EnhancedEvidenceDemandLoop.run_full_verification() [OVERRIDDEN]      │  │
│  │  FILE: src/core/evidence_demand.py LINE: 1166                         │  │
│  │                                                                        │  │
│  │  STEP 1: Base class claim extraction (patterns)                       │  │
│  │          └── Extract completion, test, implementation claims          │  │
│  │                                                                        │  │
│  │  STEP 2: PROOF VERIFICATION (for each claim)                          │  │
│  │          └── _verify_claim_proof_sync()                               │  │
│  │              ├── Past-tense proposal detection                        │  │
│  │              ├── Explained failure detection                          │  │
│  │              ├── Goal alignment/substitution check                    │  │
│  │              └── Enumeration verification                             │  │
│  │                                                                        │  │
│  │  STEP 3: Update results with PROOF_GAP entries                        │  │
│  │          └── unverified_claims.append("PROOF_GAP: ...")              │  │
│  │                                                                        │  │
│  │  STEP 4: Track inventions (NOVEL-3+, GAP-1+)                         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  DECISION: Check if evidence_demand_critical                                 │
│  FILE: src/core/integrated_engine.py LINE: 1419-1452                        │
│                                                                              │
│  IF unverified_claims OR proof_gaps FOUND:                                   │
│     └── decision.warnings.append("[Evidence Demand] ...")                   │
│                                                                              │
│  IF placeholder_code + completion_claim:                                     │
│     └── decision.accepted = False                                           │
│     └── decision.pathway = REJECTED                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 7: MULTI-TRACK CHALLENGER (if high-stakes)                           │
│  └── Run parallel LLM analysis if domain is medical/financial/legal         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 8: RESPONSE IMPROVER (if issues detected)                            │
│  └── Apply corrections based on detected issues                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  RETURN: GovernanceDecision                                                  │
│  ├── accepted: True/False                                                   │
│  ├── pathway: VERIFIED/ASSISTED/REJECTED                                    │
│  ├── warnings: [...including proof gaps...]                                 │
│  ├── recommendations: [...]                                                 │
│  └── inventions_applied: [...NOVEL-3+, GAP-1+...]                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## PROOF VERIFICATION METHODS CALLED

| Method | File | Line | What It Checks |
|--------|------|------|----------------|
| `run_full_verification()` | evidence_demand.py | 1166 | Entry point (OVERRIDDEN) |
| `_verify_claim_proof_sync()` | evidence_demand.py | 1230 | Synchronous wrapper |
| `_detect_past_tense_proposals()` | evidence_demand.py | 956 | "was designed", "we implemented" |
| `_detect_explained_failures()` | evidence_demand.py | 968 | "minor issues", "being addressed" |
| `_check_goal_alignment()` | evidence_demand.py | 1062 | quality→performance substitution |

---

## WHEN PROOF VERIFICATION IS TRIGGERED

| Condition | Proof Check | Result |
|-----------|-------------|--------|
| Any LLM response | Always | Part of evaluate_and_improve() |
| Completion claim found | Enumeration check | "All 15 items" → verify each |
| Past-tense detected | Proposal check | "was designed" → demand proof |
| Minimizing language | Explained failure | "minor issues" → verify severity |
| Query has objective | Goal alignment | Check response addresses query |

---

## VERIFIED WORKING EXAMPLE

**Input:**
```
Query: Is the payment module complete?
Response: The payment module is complete. All 15 tests pass. 
         The system was designed with Redis. Future enhancements 
         will add crypto. Minor issues are being addressed.
```

**Proof Gaps Detected:**
1. `PROOF_GAP: Past-tense proposal: 'was designed' - verify implementation`
2. `PROOF_GAP: Enumeration claim without list - verify each item`
3. `PROOF_GAP: Explained failure: 'being addressed' - verify severity`
4. `PROOF_GAP: Past-tense proposal: 'Future enhancements' - verify...`

**Result:**
- `overall_status: unverified`
- `overall_confidence: 0.00`
- `inventions_applied: [..., NOVEL-3+, GAP-1+]`

---

## KEY FILES

| File | Purpose |
|------|---------|
| `src/core/integrated_engine.py` | Main engine - calls evidence_demand on line 1386 |
| `src/core/evidence_demand.py` | Enhanced evidence loop with proof verification |
| `src/detectors/behavioral.py` | TGTBT patterns including past-tense |

---

**STATUS:** ✅ ORCHESTRATED AND WORKING

The proof-based verification is automatically called whenever 
`IntegratedGovernanceEngine.evaluate_and_improve()` is invoked. No manual
trigger needed - it's part of the standard BASE evaluation pipeline.


