# BAIS Wiring Audit Report

**Date:** January 11, 2026  
**Engine Version:** 49.0.0  
**Auditor:** Phase 1 Automated Analysis

---

## Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total Module Slots** | 87 | 100% |
| **Modules Initialized** | 72 | 83% |
| **Modules Called in `_run_detectors`** | 32 | 37% |
| **Modules Never Called** | 40 | 46% |
| **Modules with Silent Failures** | 55 | 63% |

### Critical Finding
**Only 37% of wired modules are actually called during governance evaluation.** The remaining 63% are either:
- Called in other methods (e.g., `evaluate`, `_determine_pathway`)
- Never called at all
- Silently failing due to `except pass` patterns

---

## Module Status by Category

### Category 1: Core Detectors (Always Called) ✅

These 5 modules are always invoked and form the foundation:

| Module | Patent ID | Called In | Status |
|--------|-----------|-----------|--------|
| `grounding_detector` | PPA1-Inv1, UP1 | `_run_detectors` | ✅ Active |
| `factual_detector` | UP2 | `_run_detectors` | ✅ Active |
| `behavioral_detector` | PPA1-Inv14, PPA3-Inv2 | `_run_detectors` | ✅ Active |
| `temporal_detector` | PPA3-Inv1 | `_run_detectors` | ✅ Active |
| `neurosymbolic_module` | UP3, NOVEL-15 | `_run_detectors` | ✅ Active |

---

### Category 2: Conditionally Called in `_run_detectors` ⚠️

These modules are called but only under specific conditions:

| Module | Patent ID | Trigger Condition | Status |
|--------|-----------|-------------------|--------|
| `trigger_intelligence` | Phase 24 | Always (if exists) | ✅ |
| `contradiction_resolver` | PPA1-Inv8 | `is_complex OR has_logical_claims` | ⚠️ Conditional |
| `multi_framework` | PPA1-Inv19 | `is_high_stakes OR force_llm` | ⚠️ Conditional |
| `big5_detector` | PPA2-Big5 | Always (if exists) | ✅ |
| `theory_of_mind` | NOVEL-14 | `is_conversational OR is_high_stakes` | ⚠️ Conditional |
| `world_models` | NOVEL-16 | `is_complex OR risk_factors > 0` | ⚠️ Conditional |
| `zpd_manager` | PPA1-Inv12, NOVEL-4 | Always (if exists) | ✅ |
| `knowledge_graph` | PPA1-Inv6 | Always (if exists) | ✅ |
| `semantic_mode_selector` | NOVEL-48 | Always (if exists) | ✅ |
| `governance_mode_controller` | NOVEL-42/43/49 | Always (if exists) | ✅ |
| `skeptical_learning` | NOVEL-45 | `is_high_stakes` | ⚠️ Conditional |
| `realtime_assistance` | NOVEL-46 | Always (if exists) | ❌ **WRONG METHOD** |
| `functional_completeness` | NOVEL-50 | `is_code_response` | ⚠️ Conditional |
| `interface_compliance` | NOVEL-51 | `is_code_response` | ⚠️ Conditional |
| `domain_proof_engine` | NOVEL-52 | `has_claims OR is_high_stakes` | ⚠️ Conditional |
| `evidence_verification` | NOVEL-53 | `is_high_stakes` | ⚠️ Conditional |
| `dynamic_plugins` | NOVEL-54 | Always (if exists) | ✅ |
| `vibe_coding_verifier` | NOVEL-5 | `is_code_response` | ⚠️ Conditional |
| `voi_shortcircuit` | PPA2-Comp8 | Always (if exists) | ✅ |
| `platform_harmonizer` | PPA1-Inv9 | `context exists` | ⚠️ Conditional |
| `creative_reasoning` | NOVEL-17 | `is_novel OR force_llm` | ⚠️ Conditional |
| `dimensional_expander` | NOVEL-28 | Always (if exists) | ✅ |
| `dimension_correlator` | NOVEL-29 | `dimensional_analysis exists` | ⚠️ Conditional |
| `domain_pattern_learner` | Phase 15 | Always (if exists) | ✅ |
| `reasoning_analyzer` | NOVEL-14/15 | Always (if exists) | ✅ |
| `domain_validator` | Phase 15 | `domain in multi_track_domains` | ⚠️ Conditional |
| `llm_aware_learning` | Phase 15 | Always (if exists) | ✅ |

---

### Category 3: Called in Other Methods (Not `_run_detectors`)

| Module | Patent ID | Called In | Status |
|--------|-----------|-----------|--------|
| `llm_helper` | - | Multiple methods | ✅ Active |
| `signal_fusion` | PPA1-Inv1 | `evaluate` | ✅ Active |
| `clinical_validator` | - | `evaluate` | ✅ Active |
| `multi_track_challenger` | NOVEL-22/23 | `evaluate`, `_challenge` | ✅ Active |
| `llm_challenger` | NOVEL-22 | `_challenge` | ✅ Active |
| `predicate_policy` | PPA1-Inv21, PPA2-Comp4 | `_determine_pathway` | ✅ Active |
| `governance_rules` | NOVEL-18 | `evaluate` | ✅ Active |
| `llm_registry` | NOVEL-19 | Multiple | ✅ Active |
| `query_analyzer` | NOVEL-9 | `evaluate` | ✅ Active |
| `response_improver` | NOVEL-20 | `evaluate`, `improve` | ✅ Active |
| `self_awareness` | NOVEL-21 | `evaluate` | ✅ Active |
| `evidence_demand` | NOVEL-3, GAP-1 | `evaluate`, `_check_evidence` | ✅ Active |
| `proof_verifier` | - | `_verify_proof` | ✅ Active |
| `llm_proof_validator` | Phase 17 | `_verify_proof` | ✅ Active |
| `hybrid_proof_validator` | Phase 17 | `_verify_proof` | ✅ Active |
| `clinical_status_classifier` | NOVEL-32 | `evaluate` | ✅ Active |
| `corrective_action_engine` | NOVEL-33 | `evaluate` | ✅ Active |
| `audit_trail` | NOVEL-33 | `evaluate`, `_record_audit` | ✅ Active |
| `feedback_loop` | PPA1-Inv3, PPA1-Inv22 | `record_feedback` | ✅ Active |
| `threshold_optimizer` | PPA2-Inv27 | Multiple | ✅ Active |
| `learning_manager` | Phase 49 | Multiple | ✅ Active |

---

### Category 4: NEVER CALLED ❌

These modules are initialized but never invoked anywhere:

| Module | Patent ID | Reason | Priority |
|--------|-----------|--------|----------|
| `cognitive_enhancer` | UP5 | No call site | 🔴 High |
| `smart_gate` | NOVEL-10 | Referenced but not called | 🔴 High |
| `hybrid_orchestrator` | NOVEL-11 | No call site | 🔴 High |
| `conversational_orchestrator` | NOVEL-12 | No call site | 🟡 Medium |
| `cognitive_intervention` | PPA1-Inv4, PPA1-Inv17 | No call site | 🟡 Medium |
| `literacy_standards` | PPA1-Inv5 | No call site | 🟡 Medium |
| `entity_trust` | PPA1-Inv6, UP4 | No call site | 🟡 Medium |
| `adaptive_difficulty` | PPA1-Inv12 | Superseded by zpd_manager? | 🟢 Low |
| `behavioral_signals` | PPA1-Inv14 | No call site | 🟡 Medium |
| `human_arbitration` | PPA1-Inv20 | No call site | 🟡 Medium |
| `triangulator` | NOVEL-6, PPA1-Inv23 | No call site | 🔴 High |
| `bias_evolution` | PPA1-Inv13/24 | Superseded by bias_evolution_tracker? | 🟢 Low |
| `bias_evolution_tracker` | PPA1-Inv1 | No call site | 🟡 Medium |
| `temporal_bias_detector` | PPA1-Inv4 | No call site | 🟡 Medium |
| `ccp_calibrator` | PPA2-Comp6/9 | No call site | 🔴 High |
| `privacy_manager` | PPA1-Inv3/13 | No call site | 🟡 Medium |
| `cross_orchestrator` | Phase 25 | No call site | 🟡 Medium |
| `brain_layer_manager` | Phase 26 | No call site | 🟡 Medium |
| `production_manager` | Phase 27 | No call site | 🟡 Medium |
| `advanced_security` | Phase 27b | No call site | 🟡 Medium |
| `unified_learning` | Phase 16 | No call site | 🟡 Medium |
| `performance_tracker` | Phase 16 | No call site | 🟡 Medium |
| `agent_config` | Phase 14 | Referenced but not called | 🟢 Low |
| `governance_output` | NOVEL-47 | No call site | 🟡 Medium |

---

## Silent Failure Analysis

### Pattern: `try/except pass`

**55 modules** use this anti-pattern which silently swallows errors:

```python
# ANTI-PATTERN (found 55 times)
self.module = None
if ModuleClass:
    try:
        self.module = ModuleClass()
    except Exception:
        pass  # Silent failure!
```

### Recommendation
Replace with logging:

```python
# PROPER PATTERN
self.module = None
if ModuleClass:
    try:
        self.module = ModuleClass()
        logger.info(f"[{patent_id}] {module_name} initialized")
    except Exception as e:
        logger.warning(f"[{patent_id}] {module_name} failed: {e}")
```

---

## Patent Coverage Analysis

### Patents with ACTIVE Coverage ✅

| Patent ID | Invention | Module | Active |
|-----------|-----------|--------|--------|
| PPA1-Inv1 | Multi-Modal Fusion | `grounding_detector`, `signal_fusion` | ✅ |
| PPA1-Inv3 | Federated Convergence | `feedback_loop` | ✅ |
| PPA1-Inv8 | Contradiction Handling | `contradiction_resolver` | ⚠️ Conditional |
| PPA1-Inv14 | Behavioral Capture | `behavioral_detector` | ✅ |
| PPA1-Inv19 | Multi-Framework | `multi_framework` | ⚠️ Conditional |
| PPA1-Inv21 | Predicate Acceptance | `predicate_policy` | ✅ |
| PPA1-Inv22 | Feedback Loop | `feedback_loop` | ✅ |
| PPA3-Inv1 | Temporal Detection | `temporal_detector` | ✅ |
| PPA3-Inv2 | Behavioral Detection | `behavioral_detector` | ✅ |
| NOVEL-3 | Claim-Evidence | `evidence_demand` | ✅ |
| NOVEL-9 | Query Analyzer | `query_analyzer` | ✅ |
| NOVEL-14/15 | Theory of Mind / Neuro-Symbolic | `theory_of_mind`, `neurosymbolic_module` | ✅ |
| NOVEL-18 | Governance Rules | `governance_rules` | ✅ |
| NOVEL-19 | LLM Registry | `llm_registry` | ✅ |
| NOVEL-20 | Response Improver | `response_improver` | ✅ |
| NOVEL-21 | Self-Awareness | `self_awareness` | ✅ |
| NOVEL-22/23 | LLM/Multi-Track Challenger | `llm_challenger`, `multi_track_challenger` | ✅ |
| NOVEL-32 | Clinical Status | `clinical_status_classifier` | ✅ |
| NOVEL-33 | Corrective Action | `corrective_action_engine`, `audit_trail` | ✅ |
| NOVEL-40 | Task Completion | (external via MCP) | ✅ |
| NOVEL-41 | Enforcement Loop | (external via MCP) | ✅ |
| NOVEL-42/43/49 | Governance Mode | `governance_mode_controller` | ✅ |
| NOVEL-44 | Multi-Track Orchestrator | (via challenger) | ✅ |
| NOVEL-45 | Skeptical Learning | `skeptical_learning` | ⚠️ Conditional |
| NOVEL-48 | Semantic Mode | `semantic_mode_selector` | ✅ |
| NOVEL-50 | Functional Completeness | `functional_completeness` | ⚠️ Code only |
| NOVEL-51 | Interface Compliance | `interface_compliance` | ⚠️ Code only |
| NOVEL-52 | Domain Proof | `domain_proof_engine` | ⚠️ Conditional |
| NOVEL-53 | Evidence Verification | `evidence_verification` | ⚠️ Conditional |
| NOVEL-54 | Dynamic Plugins | `dynamic_plugins` | ✅ |
| PPA2-Big5 | OCEAN Traits | `big5_detector` | ✅ |
| PPA2-Inv27 | OCO Threshold | `threshold_optimizer` | ✅ |

### Patents with NO Active Coverage ❌

| Patent ID | Invention | Module | Issue |
|-----------|-----------|--------|-------|
| PPA1-Inv4 | Computational Intervention | `cognitive_intervention` | Never called |
| PPA1-Inv5 | ACRL Literacy | `literacy_standards` | Never called |
| PPA1-Inv6 | Knowledge Graphs | `entity_trust` | Never called |
| PPA1-Inv9 | Platform Harmonization | `platform_harmonizer` | Minimal use |
| PPA1-Inv12 | Adaptive Difficulty | `adaptive_difficulty` | Never called |
| PPA1-Inv13 | Federated Privacy | `privacy_manager` | Never called |
| PPA1-Inv17 | Cognitive Window | `cognitive_intervention` | Never called |
| PPA1-Inv20 | Human-Machine Hybrid | `human_arbitration` | Never called |
| PPA1-Inv23 | Triangulation | `triangulator` | Never called |
| PPA1-Inv24 | Neuroplasticity | `bias_evolution` | Never called |
| UP4 | Knowledge Graph Integration | `entity_trust` | Never called |
| UP5 | Cognitive Enhancement | `cognitive_enhancer` | Never called |
| NOVEL-4 | ZPD | `adaptive_difficulty` | Never called |
| NOVEL-5 | Vibe Coding | `vibe_coding_verifier` | Code-only trigger |
| NOVEL-6 | Triangulation | `triangulator` | Never called |
| NOVEL-7 | Neuroplasticity | `bias_evolution_tracker` | Never called |
| NOVEL-10 | Smart Gate | `smart_gate` | Referenced, not called |
| NOVEL-11 | Hybrid Orchestrator | `hybrid_orchestrator` | Never called |
| NOVEL-12 | Conversational | `conversational_orchestrator` | Never called |
| NOVEL-16 | World Models | `world_models` | Conditional only |
| NOVEL-17 | Creative Reasoning | `creative_reasoning` | Conditional only |
| NOVEL-46 | Real-Time Assist | `realtime_assistance` | **WRONG METHOD** |
| NOVEL-47 | Governance Output | `governance_output` | Never called |
| PPA2-Comp4 | Conformal Must-Pass | `predicate_policy` | Partial |
| PPA2-Comp5 | Crisis Mode | - | Not implemented |
| PPA2-Comp6 | Calibration | `ccp_calibrator` | Never called |
| PPA2-Comp8 | VOI Short-Circuit | `voi_shortcircuit` | Late in pipeline |
| PPA2-Comp9 | Calibrated Posterior | `ccp_calibrator` | Never called |
| PPA2-Inv26 | Lexicographic Gate | `predicate_policy` | Partial |
| PPA2-Inv28 | Cognitive Window | `cognitive_intervention` | Never called |

---

## Bug Report: Method Name Mismatches

### Critical Bug: `realtime_assistance`

```python
# Line 4360-4366 in integrated_engine.py
signals.realtime_assistance = self.realtime_assistance.analyze(
    query=query,
    response=response,
    domain=domain
)
```

**Issue:** `RealTimeAssistanceEngine` has method `.assist()`, not `.analyze()`

**Fix Required:**
```python
signals.realtime_assistance = self.realtime_assistance.assist(
    response=response,
    query=query,
    domain=domain
)
```

### Potential Bug: `evidence_verification`

```python
# Line 4407-4410
signals.evidence_verification = self.evidence_verification.verify(
    query=query,
    response=response
)
```

**Issue:** `EvidenceVerificationModule.verify()` takes a `VerificationRequest` object, not kwargs.

---

## Recommendations

### Immediate (This Week)

1. **Fix Method Mismatches** - realtime_assistance.analyze → assist
2. **Add Call Sites for HIGH Priority Never-Called Modules:**
   - `cognitive_enhancer`
   - `smart_gate`
   - `hybrid_orchestrator`
   - `triangulator`
   - `ccp_calibrator`

3. **Replace Silent Failures** - Add logging to all 55 `except pass` blocks

### Short-Term (2 Weeks)

4. **Wire remaining MEDIUM priority modules:**
   - `conversational_orchestrator`
   - `cognitive_intervention`
   - `literacy_standards`
   - `entity_trust`
   - `behavioral_signals`
   - `human_arbitration`
   - `bias_evolution_tracker`
   - `temporal_bias_detector`
   - `privacy_manager`
   - `governance_output`

### Long-Term (6 Weeks)

5. **Complete all 100% coverage per Phase Plan**
6. **Add MCP tools for all patents**
7. **Implement adaptive triggering for all modules**

---

## Appendix: Module Initialization Locations

| Line | Module | Patent |
|------|--------|--------|
| 1257 | learning_manager | Phase 49 |
| 1279 | threshold_optimizer | PPA2-Inv27 |
| 1287 | feedback_loop | PPA1-Inv3/22 |
| 1292 | grounding_detector | PPA1-Inv1 |
| 1295 | factual_detector | UP2 |
| 1298 | behavioral_detector | PPA1-Inv14 |
| 1301 | temporal_detector | PPA3-Inv1 |
| 1306 | neurosymbolic_module | UP3 |
| 1309 | llm_helper | - |
| 1317 | signal_fusion | PPA1-Inv1 |
| 1323 | clinical_validator | - |
| 1340 | governance_rules | NOVEL-18 |
| 1343 | llm_registry | NOVEL-19 |
| 1346 | query_analyzer | NOVEL-9 |
| 1350 | response_improver | NOVEL-20 |
| 1360 | self_awareness | NOVEL-21 |
| 1379 | evidence_demand | NOVEL-3 |
| 1386 | llm_proof_validator | Phase 17 |
| 1392 | hybrid_proof_validator | Phase 17 |
| 1400 | proof_verifier | - |
| 1406 | clinical_status_classifier | NOVEL-32 |
| 1412 | corrective_action_engine | NOVEL-33 |
| 1440 | audit_trail | NOVEL-33 |
| 1456 | multi_track_challenger | NOVEL-22/23 |
| 1462 | llm_challenger | NOVEL-22 |
| 1477 | contradiction_resolver | PPA1-Inv8 |
| 1485 | multi_framework | PPA1-Inv19 |
| 1493 | big5_detector | PPA2-Big5 |
| 1541 | dimensional_expander | NOVEL-28 |
| 1552 | dimension_correlator | NOVEL-29 |
| 1563 | dimensional_learning | NOVEL-30 |
| 1579 | domain_pattern_learner | Phase 15 |
| 1590 | reasoning_analyzer | NOVEL-14/15 |
| 1599 | domain_validator | Phase 15 |
| 1611 | llm_aware_learning | Phase 15 |
| 1623 | unified_learning | Phase 16 |
| 1634 | performance_tracker | Phase 16 |
| 1645 | cognitive_enhancer | UP5 |
| 1653 | smart_gate | NOVEL-10 |
| 1661 | hybrid_orchestrator | NOVEL-11 |
| 1669 | theory_of_mind | NOVEL-14 |
| 1677 | world_models | NOVEL-16 |
| 1685 | creative_reasoning | NOVEL-17 |
| 1693 | predicate_policy | PPA1-Inv21 |
| 1701 | conversational_orchestrator | NOVEL-12 |
| 1713 | cognitive_intervention | PPA1-Inv4 |
| 1721 | literacy_standards | PPA1-Inv5 |
| 1729 | entity_trust | PPA1-Inv6 |
| 1737 | adaptive_difficulty | PPA1-Inv12 |
| 1745 | zpd_manager | PPA1-Inv12 |
| 1754 | knowledge_graph | PPA1-Inv6 |
| 1767 | governance_mode_controller | NOVEL-42 |
| 1776 | skeptical_learning | NOVEL-45 |
| 1787 | realtime_assistance | NOVEL-46 |
| 1796 | governance_output | NOVEL-47 |
| 1805 | semantic_mode_selector | NOVEL-48 |
| 1814 | functional_completeness | NOVEL-50 |
| 1823 | interface_compliance | NOVEL-51 |
| 1832 | domain_proof_engine | NOVEL-52 |
| 1841 | evidence_verification | NOVEL-53 |
| 1850 | dynamic_plugins | NOVEL-54 |
| 1865 | vibe_coding_verifier | NOVEL-5 |
| 1874 | platform_harmonizer | PPA1-Inv9 |
| 1883 | voi_shortcircuit | PPA2-Comp8 |
| 1892 | behavioral_signals | PPA1-Inv14 |
| 1900 | human_arbitration | PPA1-Inv20 |
| 1908 | triangulator | NOVEL-6 |
| 1916 | bias_evolution | PPA1-Inv24 |
| 1925 | bias_evolution_tracker | PPA1-Inv1 |
| 1940 | temporal_bias_detector | PPA1-Inv4 |
| 1956 | ccp_calibrator | PPA2-Comp9 |
| 1968 | privacy_manager | PPA1-Inv13 |
| 1982 | trigger_intelligence | Phase 24 |
| 1995 | cross_orchestrator | Phase 25 |
| 2007 | brain_layer_manager | Phase 26 |
| 2019 | production_manager | Phase 27 |
| 2031 | advanced_security | Phase 27b |

---

*Audit complete. Proceed to Phase 2 for Layer 1 wiring.*

