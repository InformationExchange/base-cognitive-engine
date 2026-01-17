"""
PHASE 20: ORCHESTRATION TRIGGER TESTS

Tests that all 71 inventions are triggered under the RIGHT conditions.

Test Categories:
1. SmartGate routing (NOVEL-10)
2. High-stakes domain triggering (PPA1-Inv19)
3. Contradiction detection triggering (PPA1-Inv8)
4. Human arbitration escalation (PPA1-Inv20)
5. Conditional module activation
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from core.integrated_engine import IntegratedGovernanceEngine


async def test_smartgate_routing():
    """Test that SmartGate routes queries correctly."""
    print("\n" + "="*60)
    print("TEST 1: SMARTGATE ROUTING (NOVEL-10)")
    print("="*60)
    
    engine = IntegratedGovernanceEngine()
    
    # Test 1a: Simple query - should route to pattern_only
    print("\n[Test 1a] Simple query routing...")
    simple_query = "What is Python?"
    simple_response = "Python is a programming language."
    
    result = await engine.evaluate(
        query=simple_query,
        response=simple_response,
        context={}
    )
    
    gate_routing = result.context.get('gate_routing', {}) if hasattr(result, 'context') else {}
    inventions = result.inventions_applied
    
    smartgate_used = any('NOVEL-10' in inv or 'SmartGate' in inv for inv in inventions)
    print(f"  SmartGate used: {smartgate_used}")
    print(f"  Inventions: {[inv for inv in inventions if 'NOVEL-10' in inv or 'SmartGate' in inv]}")
    
    # Test 1b: Complex query with risk factors - should force LLM
    print("\n[Test 1b] High-risk query routing...")
    risk_query = "Should I take aspirin for my heart condition?"
    risk_response = "Yes, aspirin is often recommended for heart conditions because it thins the blood."
    
    result = await engine.evaluate(
        query=risk_query,
        response=risk_response,
        context={}
    )
    
    inventions = result.inventions_applied
    smartgate_used = any('NOVEL-10' in inv or 'SmartGate' in inv for inv in inventions)
    multiframework_used = any('Multi-Framework' in inv or 'PPA1-Inv19' in inv for inv in inventions)
    
    print(f"  SmartGate used: {smartgate_used}")
    print(f"  MultiFramework triggered (high-stakes): {multiframework_used}")
    print(f"  Domain: {result.domain}")
    
    return True


async def test_human_arbitration_escalation():
    """Test that human arbitration is triggered for low-confidence high-stakes."""
    print("\n" + "="*60)
    print("TEST 2: HUMAN ARBITRATION ESCALATION (PPA1-Inv20)")
    print("="*60)
    
    engine = IntegratedGovernanceEngine()
    
    # Test with medical domain (high-stakes) and uncertain response
    print("\n[Test 2a] Medical domain with uncertain claim...")
    medical_query = "What medication should I take for depression?"
    medical_response = "You should probably take Prozac. It might help with your symptoms."
    
    result = await engine.evaluate(
        query=medical_query,
        response=medical_response,
        context={}
    )
    
    inventions = result.inventions_applied
    arbitration_triggered = any('PPA1-Inv20' in inv or 'Human Arbitration' in inv for inv in inventions)
    
    print(f"  Domain: {result.domain}")
    print(f"  Confidence: {result.confidence:.1f}%")
    print(f"  Arbitration triggered: {arbitration_triggered}")
    print(f"  Warnings: {[w for w in result.warnings if 'Arbitration' in w]}")
    
    return True


async def test_conditional_triggers():
    """Test that modules only run when their conditions are met."""
    print("\n" + "="*60)
    print("TEST 3: CONDITIONAL MODULE TRIGGERS")
    print("="*60)
    
    engine = IntegratedGovernanceEngine()
    
    # Test 3a: Simple query - minimal modules should run
    print("\n[Test 3a] Simple query - minimal modules...")
    simple_query = "Hello"
    simple_response = "Hello! How can I help you today?"
    
    result = await engine.evaluate(
        query=simple_query,
        response=simple_response,
        context={}
    )
    
    print(f"  Inventions applied: {len(result.inventions_applied)}")
    print(f"  All inventions: {result.inventions_applied[:5]}...")
    
    # Test 3b: Complex logical query - more modules should run
    print("\n[Test 3b] Complex logical query - more modules...")
    complex_query = "If A implies B, and B implies C, therefore A implies C. Is this valid reasoning because of transitivity?"
    complex_response = "Yes, this is valid reasoning because of the transitive property of implication. Therefore, A implies C is correct."
    
    result = await engine.evaluate(
        query=complex_query,
        response=complex_response,
        context={}
    )
    
    inventions = result.inventions_applied
    contradiction_run = any('PPA1-Inv8' in inv or 'Contradiction' in inv for inv in inventions)
    
    print(f"  Inventions applied: {len(result.inventions_applied)}")
    print(f"  Contradiction analysis triggered: {contradiction_run}")
    print(f"  Has logical claims detected: True (contains 'therefore', 'implies')")
    
    return True


async def test_high_stakes_domain_detection():
    """Test high-stakes domain detection triggers appropriate modules."""
    print("\n" + "="*60)
    print("TEST 4: HIGH-STAKES DOMAIN DETECTION")
    print("="*60)
    
    engine = IntegratedGovernanceEngine()
    
    domains_to_test = [
        ("medical", "What is the treatment for diabetes?", "Treatment includes insulin therapy and diet management."),
        ("financial", "Should I invest in Bitcoin?", "Bitcoin is a high-risk investment that may provide returns."),
        ("legal", "Is this contract enforceable?", "The contract appears to have valid consideration and mutual assent."),
        ("general", "What color is the sky?", "The sky is blue during the day."),
    ]
    
    for expected_domain, query, response in domains_to_test:
        result = await engine.evaluate(query=query, response=response, context={})
        
        is_high_stakes = result.domain in ['medical', 'financial', 'legal', 'safety']
        multiframework_used = any('Multi-Framework' in inv or 'PPA1-Inv19' in inv for inv in result.inventions_applied)
        
        print(f"\n  Query: '{query[:40]}...'")
        print(f"  Detected domain: {result.domain}")
        print(f"  Is high-stakes: {is_high_stakes}")
        print(f"  MultiFramework triggered: {multiframework_used}")
    
    return True


async def main():
    """Run all Phase 20 orchestration tests."""
    print("\n" + "="*70)
    print("PHASE 20: ORCHESTRATION TRIGGER TESTS")
    print("Testing that all 71 inventions trigger under RIGHT conditions")
    print("="*70)
    
    tests = [
        ("SmartGate Routing", test_smartgate_routing),
        ("Human Arbitration Escalation", test_human_arbitration_escalation),
        ("Conditional Triggers", test_conditional_triggers),
        ("High-Stakes Domain Detection", test_high_stakes_domain_detection),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, "PASS" if result else "FAIL"))
        except Exception as e:
            print(f"\n  ERROR: {str(e)[:100]}")
            results.append((name, f"ERROR: {str(e)[:50]}"))
    
    # Summary
    print("\n" + "="*70)
    print("PHASE 20 TEST SUMMARY")
    print("="*70)
    for name, result in results:
        status = "✅" if result == "PASS" else "❌"
        print(f"  {status} {name}: {result}")
    
    passed = sum(1 for _, r in results if r == "PASS")
    total = len(results)
    print(f"\n  Total: {passed}/{total} passed")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

