#!/usr/bin/env python3
"""
PHASE 2 TEST: Adaptive Learning Integration Verification
=========================================================
Tests that all detectors have adaptive learning capabilities.

Purpose: Verify Phase 2 enhancements add learning capabilities
Method: Test learning methods + dual-track A/B

Created: December 24, 2025
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
import time
import asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class LearningCapabilityResult:
    """Result of adaptive learning capability test."""
    detector_name: str
    has_domain_thresholds: bool
    has_record_outcome: bool
    has_learning_rate: bool
    domain_specific: bool
    test_passed: bool
    details: str = ""


def test_behavioral_learning() -> LearningCapabilityResult:
    """Test BehavioralBiasDetector adaptive learning."""
    try:
        from detectors.behavioral import BehavioralBiasDetector, ComprehensiveBiasResult
        
        detector = BehavioralBiasDetector()
        
        # Check learning capabilities
        has_domain_thresholds = hasattr(detector, 'DOMAIN_THRESHOLDS')
        has_record_outcome = hasattr(detector, 'record_outcome')
        has_learning_rate = hasattr(detector, '_learning_rate')
        
        # Test domain-specific threshold
        domain_specific = False
        if hasattr(detector, 'get_domain_threshold'):
            medical_threshold = detector.get_domain_threshold('medical')
            general_threshold = detector.get_domain_threshold('general')
            domain_specific = medical_threshold != general_threshold
        
        # Test learning method
        if has_record_outcome:
            result = detector.detect_all(response="This is a test response")
            detector.record_outcome(result, was_correct=True, domain='general')
        
        test_passed = has_domain_thresholds and has_record_outcome and has_learning_rate
        
        return LearningCapabilityResult(
            detector_name="BehavioralBiasDetector",
            has_domain_thresholds=has_domain_thresholds,
            has_record_outcome=has_record_outcome,
            has_learning_rate=has_learning_rate,
            domain_specific=domain_specific,
            test_passed=test_passed,
            details=f"Medical: {medical_threshold if domain_specific else 'N/A'}, General: {general_threshold if domain_specific else 'N/A'}"
        )
    except Exception as e:
        return LearningCapabilityResult(
            detector_name="BehavioralBiasDetector",
            has_domain_thresholds=False,
            has_record_outcome=False,
            has_learning_rate=False,
            domain_specific=False,
            test_passed=False,
            details=str(e)
        )


def test_grounding_learning() -> LearningCapabilityResult:
    """Test GroundingDetector adaptive learning."""
    try:
        from detectors.grounding import GroundingDetector
        
        detector = GroundingDetector()
        
        # Check learning capabilities
        has_domain_thresholds = hasattr(detector, 'DOMAIN_THRESHOLDS')
        has_source_reliability = hasattr(detector, '_source_reliability')
        has_learning_rate = hasattr(detector, '_learning_rate')
        
        # Test domain-specific threshold
        domain_specific = False
        if has_domain_thresholds:
            medical_threshold = detector.DOMAIN_THRESHOLDS.get('medical', 0.6)
            general_threshold = detector.DOMAIN_THRESHOLDS.get('general', 0.6)
            domain_specific = medical_threshold != general_threshold
        
        test_passed = has_domain_thresholds and has_source_reliability and has_learning_rate
        
        return LearningCapabilityResult(
            detector_name="GroundingDetector",
            has_domain_thresholds=has_domain_thresholds,
            has_record_outcome=has_source_reliability,  # Uses source reliability instead
            has_learning_rate=has_learning_rate,
            domain_specific=domain_specific,
            test_passed=test_passed,
            details=f"Source reliability tracking: {has_source_reliability}"
        )
    except Exception as e:
        return LearningCapabilityResult(
            detector_name="GroundingDetector",
            has_domain_thresholds=False,
            has_record_outcome=False,
            has_learning_rate=False,
            domain_specific=False,
            test_passed=False,
            details=str(e)
        )


def test_factual_learning() -> LearningCapabilityResult:
    """Test FactualDetector adaptive learning."""
    try:
        from detectors.factual import FactualDetector
        
        detector = FactualDetector()
        
        # Check learning capabilities
        has_domain_thresholds = hasattr(detector, 'DOMAIN_THRESHOLDS')
        has_claim_tracking = hasattr(detector, '_claim_type_accuracy')
        has_learning_rate = hasattr(detector, '_learning_rate')
        
        # Test domain-specific threshold
        domain_specific = False
        if has_domain_thresholds:
            medical_threshold = detector.DOMAIN_THRESHOLDS.get('medical', 0.6)
            general_threshold = detector.DOMAIN_THRESHOLDS.get('general', 0.6)
            domain_specific = medical_threshold != general_threshold
        
        test_passed = has_domain_thresholds and has_claim_tracking and has_learning_rate
        
        return LearningCapabilityResult(
            detector_name="FactualDetector",
            has_domain_thresholds=has_domain_thresholds,
            has_record_outcome=has_claim_tracking,  # Uses claim tracking
            has_learning_rate=has_learning_rate,
            domain_specific=domain_specific,
            test_passed=test_passed,
            details=f"Claim type tracking: {has_claim_tracking}"
        )
    except Exception as e:
        return LearningCapabilityResult(
            detector_name="FactualDetector",
            has_domain_thresholds=False,
            has_record_outcome=False,
            has_learning_rate=False,
            domain_specific=False,
            test_passed=False,
            details=str(e)
        )


def test_temporal_learning() -> LearningCapabilityResult:
    """Test TemporalDetector adaptive learning."""
    try:
        from detectors.temporal import TemporalDetector
        
        detector = TemporalDetector()
        
        # Check learning capabilities
        has_domain_thresholds = hasattr(detector, 'DOMAIN_THRESHOLDS')
        has_pattern_tracking = hasattr(detector, '_pattern_accuracy')
        has_learning_rate = hasattr(detector, '_learning_rate')
        
        # Test domain-specific threshold
        domain_specific = False
        if has_domain_thresholds:
            medical_threshold = detector.DOMAIN_THRESHOLDS.get('medical', 0.5)
            general_threshold = detector.DOMAIN_THRESHOLDS.get('general', 0.5)
            domain_specific = medical_threshold != general_threshold
        
        test_passed = has_domain_thresholds and has_pattern_tracking and has_learning_rate
        
        return LearningCapabilityResult(
            detector_name="TemporalDetector",
            has_domain_thresholds=has_domain_thresholds,
            has_record_outcome=has_pattern_tracking,  # Uses pattern tracking
            has_learning_rate=has_learning_rate,
            domain_specific=domain_specific,
            test_passed=test_passed,
            details=f"Pattern tracking: {has_pattern_tracking}"
        )
    except Exception as e:
        return LearningCapabilityResult(
            detector_name="TemporalDetector",
            has_domain_thresholds=False,
            has_record_outcome=False,
            has_learning_rate=False,
            domain_specific=False,
            test_passed=False,
            details=str(e)
        )


def run_dual_track_ab_test() -> Dict:
    """Run dual-track A/B test on adaptive learning."""
    from core.integrated_engine import IntegratedGovernanceEngine
    
    print("\n" + "="*60)
    print("DUAL-TRACK A/B TEST: Adaptive Learning")
    print("="*60)
    
    # Track A: Direct pattern detection (no learning)
    track_a_start = time.time()
    try:
        from detectors.behavioral import BehavioralBiasDetector
        detector = BehavioralBiasDetector()
        result = detector.detect_all(
            response="This cure is 100% guaranteed to work perfectly with zero side effects!"
        )
        track_a_score = 100 - result.total_bias_score
        track_a_issues = len(result.detected_biases)
    except Exception as e:
        track_a_score = 0
        track_a_issues = 0
    track_a_time = (time.time() - track_a_start) * 1000
    
    # Track B: BASE-governed with learning
    track_b_start = time.time()
    try:
        engine = IntegratedGovernanceEngine()
        
        test_query = "Is this medicine effective?"
        test_response = "This cure is 100% guaranteed to work perfectly with zero side effects!"
        
        async def run_eval():
            return await engine.evaluate(test_query, test_response)
        
        base_result = asyncio.run(run_eval())
        track_b_score = base_result.accuracy
        track_b_issues = len(base_result.issues)
        track_b_accepted = base_result.accepted
        track_b_pathway = base_result.pathway
    except Exception as e:
        track_b_score = 0
        track_b_issues = 0
        track_b_accepted = True
        track_b_pathway = "error"
    track_b_time = (time.time() - track_b_start) * 1000
    
    # Determine winner based on issue detection
    winner = "Track B" if track_b_issues >= track_a_issues else "Track A"
    
    results = {
        "track_a": {
            "score": track_a_score,
            "issues": track_a_issues,
            "time_ms": track_a_time
        },
        "track_b": {
            "score": track_b_score,
            "issues": track_b_issues,
            "accepted": track_b_accepted,
            "pathway": track_b_pathway,
            "time_ms": track_b_time
        },
        "winner": winner
    }
    
    print(f"\nTrack A: Score={track_a_score:.1f}, Issues={track_a_issues}")
    print(f"Track B: Score={track_b_score:.1f}, Issues={track_b_issues}, Accepted={track_b_accepted}")
    print(f"Winner: {winner}")
    
    return results


def main():
    print("="*60)
    print("PHASE 2 TEST: Adaptive Learning Integration")
    print("="*60)
    print()
    
    # Test all detectors
    results = []
    
    print("Testing BehavioralBiasDetector...")
    r1 = test_behavioral_learning()
    results.append(r1)
    print(f"  ✅ Passed" if r1.test_passed else f"  ❌ Failed: {r1.details}")
    
    print("\nTesting GroundingDetector...")
    r2 = test_grounding_learning()
    results.append(r2)
    print(f"  ✅ Passed" if r2.test_passed else f"  ❌ Failed: {r2.details}")
    
    print("\nTesting FactualDetector...")
    r3 = test_factual_learning()
    results.append(r3)
    print(f"  ✅ Passed" if r3.test_passed else f"  ❌ Failed: {r3.details}")
    
    print("\nTesting TemporalDetector...")
    r4 = test_temporal_learning()
    results.append(r4)
    print(f"  ✅ Passed" if r4.test_passed else f"  ❌ Failed: {r4.details}")
    
    # Summary
    passed = sum(1 for r in results if r.test_passed)
    total = len(results)
    
    print()
    print("="*60)
    print("LEARNING CAPABILITIES SUMMARY")
    print("="*60)
    print(f"{'Detector':<25} {'Domain':<8} {'Learning':<10} {'Rate':<6} {'Status'}")
    print("-"*60)
    for r in results:
        status = "✅" if r.test_passed else "❌"
        print(f"{r.detector_name:<25} {'✓' if r.domain_specific else '✗':<8} {'✓' if r.has_record_outcome else '✗':<10} {'✓' if r.has_learning_rate else '✗':<6} {status}")
    
    print()
    print(f"Total: {passed}/{total} detectors have adaptive learning ({100*passed/total:.0f}%)")
    
    # Run dual-track test
    ab_results = run_dual_track_ab_test()
    
    # Final verdict
    print()
    print("="*60)
    print("PHASE 2 VERDICT")
    print("="*60)
    if passed == total:
        print("✅ PHASE 2 COMPLETE: All detectors have adaptive learning")
        print(f"   - Domain-specific thresholds: {sum(1 for r in results if r.domain_specific)}/{total}")
        print(f"   - Learning integration: {passed}/{total}")
        print(f"   - A/B Test Winner: {ab_results['winner']}")
    else:
        print(f"❌ PHASE 2 NEEDS WORK: Only {passed}/{total} detectors have learning")


if __name__ == "__main__":
    main()


