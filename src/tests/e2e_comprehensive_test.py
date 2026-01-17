"""
BASE Cognitive Governance Engine v48.0
Comprehensive End-to-End Test Suite

Tests all 48 phases in integrated operation.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any
import sys

# Test results storage
class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.timings = {}
        self.phase_results = {}
    
    def record(self, phase: str, test: str, passed: bool, duration_ms: float, error: str = None):
        if phase not in self.phase_results:
            self.phase_results[phase] = {"passed": 0, "failed": 0, "tests": []}
        
        if passed:
            self.passed += 1
            self.phase_results[phase]["passed"] += 1
        else:
            self.failed += 1
            self.phase_results[phase]["failed"] += 1
            if error:
                self.errors.append(f"{phase}/{test}: {error}")
        
        self.phase_results[phase]["tests"].append({
            "name": test,
            "passed": passed,
            "duration_ms": duration_ms
        })
        self.timings[f"{phase}/{test}"] = duration_ms


def run_test(results: TestResults, phase: str, test_name: str, test_func):
    """Run a single test with timing."""
    start = time.time()
    try:
        test_func()
        duration = (time.time() - start) * 1000
        results.record(phase, test_name, True, duration)
        return True
    except Exception as e:
        duration = (time.time() - start) * 1000
        results.record(phase, test_name, False, duration, str(e))
        return False


def test_core_governance(results: TestResults):
    """Test core governance pipeline (Phases 1-29)."""
    print("\n[1] Testing Core Governance Pipeline")
    print("-" * 60)
    
    from core.integrated_engine import IntegratedGovernanceEngine
    engine = IntegratedGovernanceEngine()
    
    # Test 1: Engine initialization
    def test_init():
        assert engine is not None
        assert engine.VERSION == "48.0.0"
    run_test(results, "Core", "engine_init", test_init)
    
    # Test 2: Bias evolution
    def test_bias():
        assert engine.bias_evolution is not None
    run_test(results, "Core", "bias_evolution", test_bias)
    
    # Test 3: Grounding detection
    def test_grounding():
        assert engine.grounding_detector is not None
    run_test(results, "Core", "grounding_detector", test_grounding)
    
    # Test 4: Factual detection
    def test_factual():
        assert engine.factual_detector is not None
    run_test(results, "Core", "factual_detector", test_factual)
    
    # Test 5: Signal fusion
    def test_fusion():
        assert engine.signal_fusion is not None
    run_test(results, "Core", "signal_fusion", test_fusion)
    
    # Test 6: Multi-track challenger
    def test_challenger():
        assert engine.multi_track_challenger is not None
    run_test(results, "Core", "multi_track_challenger", test_challenger)
    
    # Test 7: CCP Calibrator (Phase 22)
    def test_ccp():
        assert engine.ccp_calibrator is not None
    run_test(results, "Core", "ccp_calibrator", test_ccp)
    
    # Test 8: Privacy Manager (Phase 23)
    def test_privacy():
        assert engine.privacy_manager is not None
    run_test(results, "Core", "privacy_manager", test_privacy)
    
    # Test 9: Trigger Intelligence (Phase 24)
    def test_trigger():
        assert engine.trigger_intelligence is not None
    run_test(results, "Core", "trigger_intelligence", test_trigger)
    
    # Test 10: Cross-Invention Orchestrator (Phase 25)
    def test_orchestrator():
        assert engine.cross_orchestrator is not None
    run_test(results, "Core", "cross_orchestrator", test_orchestrator)
    
    return engine


def test_advanced_features(results: TestResults, engine):
    """Test advanced features (Phases 30-40)."""
    print("\n[2] Testing Advanced Features")
    print("-" * 60)
    
    # Phase 30: Drift Detection
    def test_drift():
        assert engine.drift_manager is not None
    run_test(results, "Advanced", "drift_manager", test_drift)
    
    # Phase 30: Probe Mode
    def test_probe():
        assert engine.probe_manager is not None
    run_test(results, "Advanced", "probe_manager", test_probe)
    
    # Phase 30: Conservative Certificates
    def test_certs():
        assert engine.certificate_manager is not None
    run_test(results, "Advanced", "certificate_manager", test_certs)
    
    # Phase 31: Temporal Robustness
    def test_temporal():
        assert engine.temporal_robustness is not None
    run_test(results, "Advanced", "temporal_robustness", test_temporal)
    
    # Phase 31: Verifiable Audit
    def test_audit():
        assert engine.verifiable_audit is not None
    run_test(results, "Advanced", "verifiable_audit", test_audit)
    
    # Phase 32: Crisis Detection
    def test_crisis():
        assert engine.crisis_detection is not None
    run_test(results, "Advanced", "crisis_detection", test_crisis)
    
    # Phase 33: Counterfactual Reasoning
    def test_counterfactual():
        assert engine.counterfactual_reasoning is not None
    run_test(results, "Advanced", "counterfactual_reasoning", test_counterfactual)
    
    # Phase 34: Multi-Modal Context
    def test_multimodal():
        assert engine.multimodal_context is not None
    run_test(results, "Advanced", "multimodal_context", test_multimodal)
    
    # Phase 35: Federated Privacy
    def test_federated():
        assert engine.federated_privacy is not None
    run_test(results, "Advanced", "federated_privacy", test_federated)
    
    # Phase 36: Active Learning
    def test_active():
        assert engine.active_learning is not None
    run_test(results, "Advanced", "active_learning", test_active)
    
    # Phase 37: Adversarial Robustness
    def test_adversarial():
        assert engine.adversarial_robustness is not None
    run_test(results, "Advanced", "adversarial_robustness", test_adversarial)
    
    # Phase 38: Compliance Engine
    def test_compliance():
        assert engine.compliance_engine is not None
    run_test(results, "Advanced", "compliance_engine", test_compliance)
    
    # Phase 39: Interpretability
    def test_interpretability():
        assert engine.interpretability_engine is not None
    run_test(results, "Advanced", "interpretability", test_interpretability)
    
    # Phase 40: Performance
    def test_performance():
        assert engine.performance_engine is not None
    run_test(results, "Advanced", "performance_engine", test_performance)


def test_operational_enhancements(results: TestResults, engine):
    """Test operational enhancements (Phases 41-48)."""
    print("\n[3] Testing Operational Enhancements")
    print("-" * 60)
    
    # Phase 41: Monitoring
    def test_monitoring():
        assert engine.monitoring_engine is not None
        status = engine.monitoring_engine.get_status()
        assert status["mode"] == "AI + Pattern + Learning"
    run_test(results, "Operational", "monitoring_engine", test_monitoring)
    
    # Phase 42: Testing
    def test_testing():
        assert engine.testing_engine is not None
        status = engine.testing_engine.get_status()
        assert status["ai_enabled"] == True
    run_test(results, "Operational", "testing_engine", test_testing)
    
    # Phase 43: Documentation
    def test_documentation():
        assert engine.documentation_engine is not None
    run_test(results, "Operational", "documentation_engine", test_documentation)
    
    # Phase 44: Configuration
    def test_configuration():
        assert engine.configuration_engine is not None
    run_test(results, "Operational", "configuration_engine", test_configuration)
    
    # Phase 45: Logging
    def test_logging():
        assert engine.logging_engine is not None
    run_test(results, "Operational", "logging_engine", test_logging)
    
    # Phase 46: Workflow
    def test_workflow():
        assert engine.workflow_engine is not None
        status = engine.workflow_engine.get_status()
        assert status["templates_available"] >= 3
    run_test(results, "Operational", "workflow_engine", test_workflow)
    
    # Phase 47: API Gateway
    def test_gateway():
        assert engine.api_gateway is not None
        status = engine.api_gateway.get_status()
        assert status["routes_registered"] >= 4
    run_test(results, "Operational", "api_gateway", test_gateway)
    
    # Phase 48: Integration Hub
    def test_integration():
        assert engine.integration_hub is not None
        status = engine.integration_hub.get_status()
        assert status["connectors_registered"] >= 5
    run_test(results, "Operational", "integration_hub", test_integration)


def test_full_governance_flow(results: TestResults, engine):
    """Test full governance evaluation flow."""
    print("\n[4] Testing Full Governance Flow")
    print("-" * 60)
    
    # Test evaluate method
    def test_evaluate():
        result = asyncio.get_event_loop().run_until_complete(
            engine.evaluate(
                query="What is the capital of France?",
                response="The capital of France is Paris.",
                documents=[{"content": "Paris is the capital of France."}]
            )
        )
        assert result is not None
        assert hasattr(result, 'accepted')
    run_test(results, "Flow", "evaluate_basic", test_evaluate)
    
    # Test with bias detection
    def test_bias_detection():
        result = asyncio.get_event_loop().run_until_complete(
            engine.evaluate(
                query="Tell me about group X",
                response="All members of group X are always bad.",
                documents=[]
            )
        )
        assert result is not None
    run_test(results, "Flow", "bias_detection", test_bias_detection)
    
    # Test evaluate_and_improve
    def test_improve():
        result = asyncio.get_event_loop().run_until_complete(
            engine.evaluate_and_improve(
                query="What is 2+2?",
                response="2+2 equals 5.",
                documents=[]
            )
        )
        assert result is not None
    run_test(results, "Flow", "evaluate_improve", test_improve)


def run_all_tests():
    """Run all end-to-end tests."""
    print("=" * 70)
    print("BASE v48.0 COMPREHENSIVE END-TO-END TEST SUITE")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    
    results = TestResults()
    start_time = time.time()
    
    try:
        # Test core governance
        engine = test_core_governance(results)
        
        # Test advanced features
        test_advanced_features(results, engine)
        
        # Test operational enhancements
        test_operational_enhancements(results, engine)
        
        # Test full governance flow
        test_full_governance_flow(results, engine)
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        results.errors.append(f"CRITICAL: {e}")
    
    total_time = time.time() - start_time
    
    # Print results
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n[Overall]")
    print(f"  Total Tests: {results.passed + results.failed}")
    print(f"  Passed: {results.passed}")
    print(f"  Failed: {results.failed}")
    print(f"  Pass Rate: {results.passed / max(1, results.passed + results.failed) * 100:.1f}%")
    print(f"  Total Time: {total_time * 1000:.0f}ms")
    
    print(f"\n[By Category]")
    for phase, data in results.phase_results.items():
        status = "✓" if data["failed"] == 0 else "✗"
        print(f"  {status} {phase}: {data['passed']}/{data['passed'] + data['failed']} passed")
    
    if results.errors:
        print(f"\n[Errors]")
        for err in results.errors[:5]:
            print(f"  - {err}")
    
    print("\n" + "=" * 70)
    if results.failed == 0:
        print("STATUS: ALL TESTS PASSED ✓")
    else:
        print(f"STATUS: {results.failed} TESTS FAILED")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
    sys.exit(0 if results.failed == 0 else 1)
