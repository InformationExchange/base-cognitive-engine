# Enhancement Architectural Alignment Document

## Phase 15 Enhancements: December 25, 2025

This document verifies that all proposed enhancements align with the BASE architecture, existing inventions, and proper orchestration pathways.

---

## Step 3: Architectural Alignment Verification

### Enhancement 1: Citation-Aware Metric Gaming Detection

| Attribute | Value |
|-----------|-------|
| **Problem** | BASE flags legitimate cited statistics as "metric_gaming" |
| **Root Cause** | Pattern detector lacks citation context awareness |
| **Solution** | Add citation proximity check before flagging |

#### Architectural Alignment

| Alignment Check | Status | Evidence |
|-----------------|--------|----------|
| **Existing Invention** | PPA1-Inv2: Bias Modeling Framework | Extends bias detection with context |
| **Brain Layer** | Layer 3: Limbic System (Behavioral) | Matches existing bias detection layer |
| **Module** | `domain_pattern_learning.py` | DomainPatternLearner class |
| **Integration Point** | `detect_patterns()` method | Returns PatternMatch with context |

#### Code Location

```
File: src/core/domain_pattern_learning.py
Class: DomainPatternLearner
Method: detect_patterns()
Enhancement: Add citation proximity check before flagging METRIC_GAMING
```

#### Signal Flow Integration

```
Response Text 
    ↓
Layer 3 (Behavioral): domain_pattern_learner.detect_patterns()
    ↓
[ENHANCEMENT] Check for nearby citations ("Source:", table headers)
    ↓
If citations present → reduce confidence or skip METRIC_GAMING flag
    ↓
Continue to Layer 4
```

---

### Enhancement 2: Content-Type Classification for Reasoning Analyzer

| Attribute | Value |
|-----------|-------|
| **Problem** | Reasoning analyzer flags factual statements as lacking alternatives |
| **Root Cause** | All statements treated as analytical/diagnostic |
| **Solution** | Add content-type classification (FACTUAL, ANALYTICAL, RECOMMENDATION) |

#### Architectural Alignment

| Alignment Check | Status | Evidence |
|-----------------|--------|----------|
| **Existing Invention** | NOVEL-14: Theory of Mind | Extends intent understanding |
| **Existing Invention** | PPA1-Inv7: Structured Reasoning Trees | Adds context-aware paths |
| **Brain Layer** | Layer 2: Prefrontal Cortex (Reasoning) | Matches reasoning layer |
| **Module** | `reasoning_chain_analyzer.py` | ReasoningChainAnalyzer class |
| **Integration Point** | `analyze()` method | Before applying alternative requirements |

#### Code Location

```
File: src/core/reasoning_chain_analyzer.py
Class: ReasoningChainAnalyzer
Method: analyze()
Enhancement: Add _classify_content_type() before checking alternatives
```

#### Content Type Rules

| Content Type | Alternative Required | Example |
|--------------|---------------------|---------|
| FACTUAL | No | "The capital of France is Paris" |
| ANALYTICAL | Yes | "The patient likely has condition X" |
| RECOMMENDATION | Yes | "You should use framework Y" |
| DIAGNOSTIC | Yes | "The diagnosis is Z" |

#### Signal Flow Integration

```
Response Text 
    ↓
Layer 2 (Reasoning): reasoning_chain_analyzer.analyze()
    ↓
[ENHANCEMENT] _classify_content_type(text)
    ↓
If FACTUAL → skip alternative requirement check
If ANALYTICAL/RECOMMENDATION → apply alternative requirement
    ↓
Continue to Layer 3
```

---

### Enhancement 3: LLMAwareLearning Integration

| Attribute | Value |
|-----------|-------|
| **Problem** | LLMAwareLearning module exists but not integrated into evaluate() |
| **Root Cause** | Module created but import/instantiation missing |
| **Solution** | Import and use in _run_detectors() for per-LLM effectiveness tracking |

#### Architectural Alignment

| Alignment Check | Status | Evidence |
|-----------------|--------|----------|
| **Existing Invention** | PPA1-Inv22: Feedback Loop | Extends feedback with LLM context |
| **Existing Invention** | PPA2-Inv27: OCO Threshold Adapter | Adds LLM-specific thresholds |
| **Existing Invention** | NOVEL-30: Dimensional Learning | Adds LLM-aware pattern learning |
| **Brain Layer** | Layer 4: Hippocampus (Memory & Learning) | Matches memory layer |
| **Module** | `llm_aware_learning.py` | LLMAwareLearning class |
| **Integration Point** | `integrated_engine.py` → `_run_detectors()` | After pattern detection |

#### Code Location

```
File: src/core/integrated_engine.py
Class: IntegratedGovernanceEngine
Method: __init__() - Add import and instantiation
Method: _run_detectors() - Add learning calls
Method: evaluate() - Record outcomes after decision
```

#### Integration Points

1. **Initialization** (`__init__`):
   ```python
   from core.llm_aware_learning import LLMAwareLearning
   self.llm_aware_learning = LLMAwareLearning()
   ```

2. **Pattern Application** (`_run_detectors`):
   ```python
   # Get LLM-specific learnings
   applicable = self.llm_aware_learning.get_applicable_learnings(llm_provider, domain)
   ```

3. **Outcome Recording** (`evaluate`):
   ```python
   # After decision, record outcome for learning
   self.llm_aware_learning.record_learning_outcome(learning_id, success, llm_provider)
   ```

#### Signal Flow Integration

```
User Input → evaluate()
    ↓
_run_detectors()
    ↓
Layer 4 (Memory): [ENHANCEMENT] llm_aware_learning.get_applicable_learnings()
    ↓
Apply LLM-specific patterns with adjusted confidence
    ↓
...processing...
    ↓
Final Decision
    ↓
[ENHANCEMENT] llm_aware_learning.record_learning_outcome()
    ↓
Persist learning for future use
```

---

## Step 4: Orchestration Planning

### Complete Signal Flow with Enhancements

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         USER INPUT                                                       │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: PERCEPTION (Sensory Cortex)                                                    │
│ • UP1: Grounding Detector                                                               │
│ • UP2: Factual Detector                                                                 │
│ • PPA1-Inv1: Multi-Modal Fusion                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 2: REASONING (Prefrontal Cortex)                                                  │
│ • PPA1-Inv7: Reasoning Trees                                                            │
│ • NOVEL-14: Theory of Mind                                                              │
│ • ReasoningChainAnalyzer                                                                │
│                                                                                         │
│ [ENHANCEMENT 2] ──► _classify_content_type()                                            │
│                     • FACTUAL → skip alternative check                                  │
│                     • ANALYTICAL → require alternatives                                 │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 3: BEHAVIORAL (Limbic System)                                                     │
│ • PPA1-Inv2: Bias Modeling                                                              │
│ • NOVEL-1: TGTBT Detection                                                              │
│ • DomainPatternLearner                                                                  │
│                                                                                         │
│ [ENHANCEMENT 1] ──► Citation-aware metric_gaming                                        │
│                     • Check for nearby citations                                        │
│                     • If cited → reduce confidence                                      │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 4: MEMORY (Hippocampus)                                                           │
│ • PPA1-Inv22: Feedback Loop                                                             │
│ • PPA2-Inv27: OCO Adapter                                                               │
│ • NOVEL-30: Dimensional Learning                                                        │
│                                                                                         │
│ [ENHANCEMENT 3] ──► LLMAwareLearning integration                                        │
│                     • Get LLM-specific learnings                                        │
│                     • Apply with adjusted confidence                                    │
│                     • Record outcomes after decision                                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYERS 5-10: Self-Awareness → Evidence → Challenge → Improvement → Orchestration → Output│
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         FINAL OUTPUT (GovernanceDecision)                               │
│                                                                                         │
│ [ENHANCEMENT 3 CONTINUED] ──► Record outcome for LLM-aware learning                     │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Testing Plan: User Level to Conclusions

### Test 1: Citation-Aware Metric Gaming

**Input:** "67 Inventions, 300 Claims (Source: BASE_BRAIN_ARCHITECTURE.md)"

**Expected Outcome:**
- Track A: Would flag as METRIC_GAMING
- Track B: Should NOT flag (citation present)

### Test 2: Content-Type Classification

**Input:** "The capital of France is Paris."

**Expected Outcome:**
- Track A: Would flag as lacking alternatives
- Track B: Should NOT flag (FACTUAL content type)

### Test 3: LLM-Aware Learning

**Input:** Run with Grok → record outcome → switch to Claude → verify learnings persist

**Expected Outcome:**
- Grok-specific learnings stored
- Claude uses universal learnings
- Transfer metrics tracked

---

## Verification Criteria

| Enhancement | Success Criteria |
|-------------|------------------|
| 1. Citation-aware metric_gaming | No false positive on cited statistics |
| 2. Content-type classification | Factual statements not flagged for alternatives |
| 3. LLMAwareLearning | Learning persists across LLM switches |

---

## Approval for Execution

All three enhancements have been verified to:
- ✅ Align to existing inventions
- ✅ Map to appropriate brain layers
- ✅ Have defined orchestration pathways
- ✅ Include end-to-end test plans

**Ready to proceed with Step 5: Execution**

