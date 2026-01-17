# BAIS Remediation Plan: Enforcement & Learning Completion

**Version:** 1.0  
**Date:** January 3, 2026  
**Target:** Fix PARTIAL items from BAIS effectiveness audit  
**Estimated Effort:** 8-12 hours

---

## Executive Summary

This plan addresses the two PARTIAL categories from the BAIS effectiveness audit:

| Category | Current State | Target State | Effort |
|----------|---------------|--------------|--------|
| **Enforcement** | Architecture exists, loop incomplete | Fully functional enforcement loop | 3-4 hours |
| **Learning** | 12.5% full coverage (3/24 modules) | 100% coverage (24/24 modules) | 5-8 hours |

---

## Part 1: ENFORCEMENT COMPLETION

### 1.1 Current State

```
EnforcementLoop Status:
├── enforce() method: ✅ Exists
├── blocking logic: ✅ Exists
├── remediation prompts: ✅ Exists
├── max_attempts limit: ✅ Exists
├── get_statistics(): ✅ Exists
├── record_outcome(): ❌ Missing
├── learn_from_feedback(): ❌ Missing
├── serialize_state(): ❌ Missing
├── deserialize_state(): ❌ Missing
└── Integration with BAISv2Orchestrator: ⚠️ Partial
```

### 1.2 Enforcement Tasks

| Task ID | Description | Priority | Effort |
|---------|-------------|----------|--------|
| **E1** | Add `record_outcome()` to EnforcementLoop | HIGH | 30 min |
| **E2** | Add `learn_from_feedback()` to EnforcementLoop | HIGH | 30 min |
| **E3** | Add `serialize_state()` / `deserialize_state()` | MEDIUM | 30 min |
| **E4** | Fix GovernanceOutput instantiation | HIGH | 30 min |
| **E5** | Complete BAISv2Orchestrator integration | HIGH | 1 hour |
| **E6** | Add enforcement loop unit tests | MEDIUM | 1 hour |

### 1.3 Enforcement Implementation Details

#### E1: Add `record_outcome()` to EnforcementLoop

```python
# File: src/core/enforcement_loop.py

def record_outcome(self, result: EnforcedResult, user_feedback: Optional[str] = None) -> None:
    """
    Record enforcement outcome for learning.
    
    Args:
        result: The enforcement result
        user_feedback: Optional user feedback on quality
    """
    self._stats['total_enforcements'] += 1
    
    if result.success:
        self._stats['successful_completions'] += 1
    else:
        self._stats['escalated_to_user'] += 1
    
    # Track attempt distribution
    attempts_key = f'attempts_{result.attempts}'
    self._stats.setdefault(attempts_key, 0)
    self._stats[attempts_key] += 1
    
    # Store for learning
    self._outcome_history.append({
        'timestamp': datetime.now().isoformat(),
        'success': result.success,
        'attempts': result.attempts,
        'feedback': user_feedback
    })
```

#### E2: Add `learn_from_feedback()` to EnforcementLoop

```python
def learn_from_feedback(self, feedback: Dict[str, Any]) -> None:
    """
    Learn from user feedback to improve enforcement.
    
    Args:
        feedback: Dictionary with:
            - was_correct: bool - Was the enforcement decision correct?
            - should_have_blocked: bool - Should it have blocked when it didn't?
            - suggestions: str - User suggestions
    """
    was_correct = feedback.get('was_correct', True)
    should_have_blocked = feedback.get('should_have_blocked', False)
    
    if not was_correct:
        self._stats['false_decisions'] += 1
        
        if should_have_blocked:
            # Tighten enforcement - lower threshold for blocking
            self._learning_params['block_threshold'] *= 0.95
            self._stats['threshold_tightened'] += 1
        else:
            # Loosen enforcement - raise threshold
            self._learning_params['block_threshold'] *= 1.05
            self._stats['threshold_loosened'] += 1
    else:
        self._stats['correct_decisions'] += 1
    
    # Ensure threshold stays in valid range
    self._learning_params['block_threshold'] = max(0.3, min(0.9, 
        self._learning_params['block_threshold']))
```

#### E4: Fix GovernanceOutput Instantiation

```python
# File: src/core/governance_output.py
# Current: __init__ requires 14 parameters
# Fix: Add default factory method

@classmethod
def create_empty(cls) -> 'GovernanceOutput':
    """Create an empty GovernanceOutput with defaults."""
    return cls(
        request_id=str(uuid.uuid4()),
        mode=BAISMode.AUDIT_ONLY,
        status=OutputStatus.PENDING,
        action_required=ActionRequired.NONE,
        original_query="",
        original_response="",
        issues=[],
        warnings=[],
        evidence_report=None,
        enhancements=[],
        modified=False,
        confidence_boost=0.0,
        output_content="",
        metadata={}
    )

# Also add no-arg __init__ with defaults
def __init__(self, ...):
    # ... existing code ...
    pass

# Add classmethod for common patterns
@classmethod  
def from_audit(cls, query: str, response: str, issues: List) -> 'GovernanceOutput':
    """Factory for audit-only results."""
    ...
```

---

## Part 2: LEARNING INTERFACE COMPLETION

### 2.1 Current State

```
Learning Coverage:
├── Modules with 5/5 methods: 3 (12.5%)
├── Modules with 1/5 methods: 17 (70.8%)
├── Modules with 0/5 methods: 4 (16.7%)
└── Total modules to fix: 20
```

### 2.2 Learning Interface Standard

All modules must implement these 5 methods:

```python
class LearningInterface(Protocol):
    """Standard learning interface for all BAIS modules."""
    
    def record_outcome(self, outcome: Dict[str, Any]) -> None:
        """Record an outcome for this module's operation."""
        ...
    
    def learn_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """Update internal parameters based on feedback."""
        ...
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return statistics about this module's performance."""
        ...
    
    def serialize_state(self) -> Dict[str, Any]:
        """Serialize current state for persistence."""
        ...
    
    def deserialize_state(self, state: Dict[str, Any]) -> None:
        """Restore state from serialized data."""
        ...
```

### 2.3 Learning Tasks by Module

#### Phase L1: BAIS v2.0 Modules (9 modules)

| Module | Missing Methods | Effort |
|--------|-----------------|--------|
| TaskCompletionEnforcer | 5/5 | 30 min |
| EnforcementLoop | 4/5 | 20 min |
| GovernanceModeController | 4/5 | 20 min |
| EvidenceClassifier | 5/5 | 30 min |
| MultiTrackOrchestrator | 4/5 | 20 min |
| SkepticalLearningManager | 4/5 | 20 min |
| RealTimeAssistanceEngine | 4/5 | 20 min |
| GovernanceOutput | 5/5 | 30 min |
| SemanticModeSelector | 4/5 | 20 min |

**Phase L1 Total:** ~3.5 hours

#### Phase L2: Original Modules (11 modules)

| Module | Missing Methods | Effort |
|--------|-----------------|--------|
| QueryAnalyzer | 4/5 | 20 min |
| ReasoningChainAnalyzer | 4/5 | 20 min |
| BehavioralBiasDetector | 4/5 | 20 min |
| ResponseImprover | 4/5 | 20 min |
| CognitiveEnhancer | 4/5 | 20 min |
| SmartGate | 4/5 | 20 min |
| HybridOrchestrator | 4/5 | 20 min |
| LLMRegistry | 4/5 | 20 min |
| LLMChallenger | 4/5 | 20 min |
| MultiTrackChallenger | 4/5 | 20 min |
| EvidenceDemandLoop | 4/5 | 20 min |

**Phase L2 Total:** ~3.5 hours

### 2.4 Learning Implementation Template

For each module, add this boilerplate (customized per module):

```python
class ModuleName:
    def __init__(self):
        # ... existing init ...
        
        # Learning state
        self._outcome_history: List[Dict] = []
        self._learning_params: Dict[str, float] = {
            'base_threshold': 0.5,
            'learning_rate': 0.1
        }
        self._stats: Dict[str, int] = {
            'total_operations': 0,
            'successful': 0,
            'failed': 0,
            'feedback_received': 0
        }
    
    def record_outcome(self, outcome: Dict[str, Any]) -> None:
        """Record outcome for learning."""
        self._stats['total_operations'] += 1
        
        success = outcome.get('success', True)
        if success:
            self._stats['successful'] += 1
        else:
            self._stats['failed'] += 1
        
        self._outcome_history.append({
            'timestamp': datetime.now().isoformat(),
            **outcome
        })
        
        # Keep history bounded
        if len(self._outcome_history) > 1000:
            self._outcome_history = self._outcome_history[-500:]
    
    def learn_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """Learn from feedback to improve performance."""
        self._stats['feedback_received'] += 1
        
        was_correct = feedback.get('was_correct', True)
        if not was_correct:
            # Adjust parameters based on feedback type
            adjustment = feedback.get('adjustment', 0.05)
            direction = feedback.get('direction', 'tighten')
            
            if direction == 'tighten':
                self._learning_params['base_threshold'] *= (1 - adjustment)
            else:
                self._learning_params['base_threshold'] *= (1 + adjustment)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return module statistics."""
        total = self._stats['total_operations']
        return {
            **self._stats,
            'success_rate': self._stats['successful'] / max(1, total),
            'learning_params': self._learning_params.copy(),
            'history_size': len(self._outcome_history)
        }
    
    def serialize_state(self) -> Dict[str, Any]:
        """Serialize state for persistence."""
        return {
            'outcome_history': self._outcome_history[-100:],  # Last 100
            'learning_params': self._learning_params.copy(),
            'stats': self._stats.copy()
        }
    
    def deserialize_state(self, state: Dict[str, Any]) -> None:
        """Restore state from serialized data."""
        self._outcome_history = state.get('outcome_history', [])
        self._learning_params.update(state.get('learning_params', {}))
        self._stats.update(state.get('stats', {}))
```

---

## Part 3: EXECUTION PLAN

### 3.1 Phased Execution

| Phase | Tasks | Duration | Dependency |
|-------|-------|----------|------------|
| **Phase 1** | E4 (Fix GovernanceOutput) | 30 min | None |
| **Phase 2** | E1-E3 (EnforcementLoop learning) | 1.5 hours | Phase 1 |
| **Phase 3** | E5 (BAISv2Orchestrator integration) | 1 hour | Phase 2 |
| **Phase 4** | L1 (v2.0 module learning) | 3.5 hours | Phase 1 |
| **Phase 5** | L2 (original module learning) | 3.5 hours | Phase 4 |
| **Phase 6** | E6 + Testing | 1 hour | All above |

**Total Estimated Time:** 11 hours

### 3.2 Verification Criteria

After each phase, verify:

```python
# Verification script
def verify_module(module_path: str, class_name: str) -> Dict:
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    instance = cls()  # Must instantiate without error
    
    required = ['record_outcome', 'learn_from_feedback', 'get_statistics',
                'serialize_state', 'deserialize_state']
    
    results = {}
    for method in required:
        results[method] = hasattr(instance, method) and callable(getattr(instance, method))
    
    # Test each method
    instance.record_outcome({'success': True, 'test': True})
    instance.learn_from_feedback({'was_correct': True})
    stats = instance.get_statistics()
    state = instance.serialize_state()
    instance.deserialize_state(state)
    
    return {
        'methods': results,
        'all_present': all(results.values()),
        'functional': True
    }
```

### 3.3 Success Metrics

| Metric | Before | Target |
|--------|--------|--------|
| Modules with 5/5 learning | 3 (12.5%) | 24 (100%) |
| EnforcementLoop complete | No | Yes |
| GovernanceOutput instantiates | No | Yes |
| BAISv2Orchestrator integrated | Partial | Full |
| All unit tests pass | N/A | 100% |

---

## Part 4: DETAILED TASK BREAKDOWN

### Task E4: Fix GovernanceOutput (HIGHEST PRIORITY)

**Why First:** GovernanceOutput failed to instantiate, blocking other v2.0 functionality.

**File:** `src/core/governance_output.py`

**Changes:**
1. Add default parameter values to `__init__`
2. Add `create_empty()` factory method
3. Add `from_audit()`, `from_enforcement()` factory methods
4. Initialize learning state in `__init__`
5. Add all 5 learning methods

**Verification:**
```python
from core.governance_output import GovernanceOutput
output = GovernanceOutput()  # Should not raise
output = GovernanceOutput.create_empty()  # Should work
```

### Task E1-E3: Complete EnforcementLoop Learning

**File:** `src/core/enforcement_loop.py`

**Changes:**
1. Add `_outcome_history` list
2. Add `_learning_params` dict
3. Implement `record_outcome()`
4. Implement `learn_from_feedback()`
5. Implement `serialize_state()`
6. Implement `deserialize_state()`

### Task E5: Complete BAISv2Orchestrator Integration

**File:** `src/core/bais_v2_orchestrator.py`

**Changes:**
1. Ensure all v2.0 components are wired
2. Add proper error handling for component failures
3. Add mode-specific routing
4. Connect SemanticModeSelector to auto-detect mode
5. Add statistics aggregation across components

---

## Part 5: TESTING PLAN

### 5.1 Unit Tests

| Test | Description |
|------|-------------|
| `test_enforcement_loop_learning` | Verify all 5 learning methods |
| `test_governance_output_instantiation` | Verify factory methods |
| `test_v2_modules_learning` | Verify all v2.0 modules have learning |
| `test_original_modules_learning` | Verify all original modules have learning |
| `test_state_persistence` | Serialize/deserialize round trip |

### 5.2 Integration Tests

| Test | Description |
|------|-------------|
| `test_enforcement_loop_integration` | Full enforcement cycle |
| `test_bais_v2_orchestrator` | Complete governance flow |
| `test_mode_switching` | Switch between AUDIT/ENFORCE/ASSIST |
| `test_learning_persistence` | Save/load learning state |

### 5.3 BAIS Self-Test

After completion, run:
```bash
python3 -c "
from core.enforcement_loop import EnforcementLoop
from core.governance_output import GovernanceOutput

loop = EnforcementLoop()
output = GovernanceOutput.create_empty()

# Verify learning interface
for component in [loop, output]:
    assert hasattr(component, 'record_outcome')
    assert hasattr(component, 'learn_from_feedback')
    assert hasattr(component, 'get_statistics')
    assert hasattr(component, 'serialize_state')
    assert hasattr(component, 'deserialize_state')
    
print('✅ All components have learning interface')
"
```

---

## Part 6: ROLLBACK PLAN

If issues arise:

1. **Phase 1-3 Issues:** Revert `governance_output.py`, `enforcement_loop.py`
2. **Phase 4 Issues:** Revert individual v2.0 module files
3. **Phase 5 Issues:** Revert individual original module files
4. **Integration Issues:** Check `bais_v2_orchestrator.py` wiring

Git commands:
```bash
git stash  # Save current state
git checkout HEAD -- src/core/governance_output.py  # Revert specific file
git stash pop  # Restore other changes
```

---

## Approval

| Approver | Date | Status |
|----------|------|--------|
| User | | Pending |

---

*Plan generated: January 3, 2026*
*Based on BAIS Effectiveness Audit results*

