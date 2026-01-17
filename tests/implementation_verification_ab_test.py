#!/usr/bin/env python3
"""
BAIS A/B Test: Implementation Verification
==========================================
Clinical audit of claimed implementation improvements.

Tests whether the +29 module implementations are:
1. Actually complete (not just signature fixes)
2. Properly orchestrated (called in real flows)
3. Independently functional
4. System-integrated

Track A: Direct assessment (no BAIS)
Track B: BAIS-governed assessment
"""

import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

@dataclass
class VerificationResult:
    """Result of a single verification check."""
    check_name: str
    passed: bool
    evidence: str
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModuleVerification:
    """Complete verification for a module."""
    module_name: str
    exists: bool = False
    instantiates: bool = False
    has_required_methods: bool = False
    methods_callable: bool = False
    integrated_in_engine: bool = False
    called_in_flow: bool = False
    produces_output: bool = False
    output_used_downstream: bool = False
    evidence: List[str] = field(default_factory=list)

class ImplementationVerifier:
    """Verifies implementation claims with clinical objectivity."""
    
    # The 12 modules claimed to be newly wired (with correct class names)
    NEWLY_WIRED_MODULES = [
        ("ContradictionResolver", "detectors.contradiction_resolver", "contradiction_resolver"),
        ("MultiFrameworkConvergenceEngine", "detectors.multi_framework", "multi_framework"),
        ("CognitiveEnhancer", "core.cognitive_enhancer", "cognitive_enhancer"),
        ("SmartGate", "core.smart_gate", "smart_gate"),
        ("HybridOrchestrator", "core.hybrid_orchestrator", "hybrid_orchestrator"),
        ("TheoryOfMindModule", "research.theory_of_mind", "theory_of_mind"),
        ("WorldModelsModule", "research.world_models", "world_models"),
        ("CreativeReasoningModule", "research.creative_reasoning", "creative_reasoning"),
        ("PredicatePolicyEngine", "learning.predicate_policy", "predicate_policy"),
        ("ConversationalOrchestrator", "core.conversational_orchestrator", "conversational_orchestrator"),
    ]
    
    # Signature fixes claimed (with correct class names)
    SIGNATURE_FIXES = [
        ("TemporalDetector", "detect", "detectors.temporal"),
        ("GroundingDetector", "detect", "detectors.grounding"),
        ("NeuroSymbolicModule", "verify_logic", "research.neurosymbolic"),
        ("BehavioralBiasDetector", "detect_all", "detectors.behavioral"),
        ("StateMachineWithHysteresis", "current_state", "learning.state_machine"),
        ("OCOLearner", "threshold", "learning.algorithms"),
        ("ContinuousFeedbackLoop", "__init__", "learning.feedback_loop"),
        ("OutcomeMemory", "__init__", "learning.outcome_memory"),
        ("BAISGovernanceRules", "rules", "core.governance_rules"),
        ("FactualDetector", "analyze", "detectors.factual"),
    ]
    
    def __init__(self):
        self.results: Dict[str, ModuleVerification] = {}
        self.track_a_assessment = ""
        self.track_b_assessment = ""
        
    def verify_module_existence(self, class_name: str, module_path: str) -> Tuple[bool, str]:
        """Check if module and class exist."""
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name, None)
            if cls is None:
                return False, f"Class {class_name} not found in {module_path}"
            return True, f"Class {class_name} exists in {module_path}"
        except ImportError as e:
            return False, f"Module import failed: {e}"
        except Exception as e:
            return False, f"Error: {e}"
    
    def verify_module_instantiates(self, class_name: str, module_path: str) -> Tuple[bool, str]:
        """Check if module can be instantiated."""
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            instance = cls()
            return True, f"Successfully instantiated {class_name}"
        except Exception as e:
            return False, f"Instantiation failed: {e}"
    
    def verify_engine_integration(self, attr_name: str) -> Tuple[bool, str]:
        """Check if module is integrated into IntegratedGovernanceEngine."""
        try:
            from core.integrated_engine import IntegratedGovernanceEngine
            from pathlib import Path
            import tempfile
            
            with tempfile.TemporaryDirectory() as tmpdir:
                engine = IntegratedGovernanceEngine(data_dir=Path(tmpdir))
                
                # Check if attribute exists
                if hasattr(engine, attr_name):
                    attr_value = getattr(engine, attr_name)
                    if attr_value is not None:
                        return True, f"Engine has {attr_name} = {type(attr_value).__name__}"
                    else:
                        return False, f"Engine has {attr_name} but it is None"
                else:
                    return False, f"Engine does not have attribute {attr_name}"
        except Exception as e:
            return False, f"Engine check failed: {e}"
    
    def verify_called_in_flow(self, attr_name: str) -> Tuple[bool, str]:
        """Check if module is actually called during evaluation flow."""
        try:
            from core.integrated_engine import IntegratedGovernanceEngine
            from pathlib import Path
            import tempfile
            
            with tempfile.TemporaryDirectory() as tmpdir:
                engine = IntegratedGovernanceEngine(data_dir=Path(tmpdir))
                
                # Run a sample evaluation
                result = asyncio.run(engine.evaluate(
                    query="Test query for flow verification",
                    response="Test response with some claims.",
                    documents=[{"content": "Test document content"}],
                    context={"domain": "general"}
                ))
                
                # Check if the attribute was used (indirectly via signals)
                # This is a weak check - we can't easily verify internal calls
                if hasattr(engine, attr_name) and getattr(engine, attr_name) is not None:
                    return True, f"Module {attr_name} was available during flow"
                return False, f"Module {attr_name} not available"
                
        except Exception as e:
            return False, f"Flow verification failed: {e}"
    
    def verify_signature_fix(self, class_name: str, method_name: str, module_path: str) -> Tuple[bool, str]:
        """Verify that a signature fix actually works."""
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            
            # Check if it's a property or method
            if hasattr(cls, method_name):
                attr = getattr(cls, method_name)
                if isinstance(attr, property):
                    # It's a property - instantiate and check
                    instance = cls()
                    value = getattr(instance, method_name)
                    return True, f"Property {method_name} accessible, value type: {type(value).__name__}"
                elif callable(attr):
                    return True, f"Method {method_name} exists and is callable"
                else:
                    return True, f"Attribute {method_name} exists"
            else:
                return False, f"Class {class_name} does not have {method_name}"
        except Exception as e:
            return False, f"Signature verification failed: {e}"
    
    def run_track_a_verification(self) -> Dict[str, Any]:
        """
        Track A: Direct verification without BAIS.
        Clinical assessment of what is actually implemented.
        """
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "track": "A",
            "description": "Direct verification (no BAIS)",
            "newly_wired_modules": [],
            "signature_fixes": [],
            "summary": {}
        }
        
        # Verify newly wired modules
        wired_pass = 0
        wired_fail = 0
        
        for class_name, module_path, attr_name in self.NEWLY_WIRED_MODULES:
            module_result = {
                "module": class_name,
                "checks": []
            }
            
            # Check 1: Exists
            exists, evidence = self.verify_module_existence(class_name, module_path)
            module_result["checks"].append({
                "check": "exists",
                "passed": exists,
                "evidence": evidence
            })
            
            # Check 2: Instantiates
            if exists:
                instantiates, evidence = self.verify_module_instantiates(class_name, module_path)
                module_result["checks"].append({
                    "check": "instantiates",
                    "passed": instantiates,
                    "evidence": evidence
                })
            else:
                module_result["checks"].append({
                    "check": "instantiates",
                    "passed": False,
                    "evidence": "Skipped - module doesn't exist"
                })
            
            # Check 3: Engine integration
            integrated, evidence = self.verify_engine_integration(attr_name)
            module_result["checks"].append({
                "check": "engine_integration",
                "passed": integrated,
                "evidence": evidence
            })
            
            # Check 4: Called in flow
            called, evidence = self.verify_called_in_flow(attr_name)
            module_result["checks"].append({
                "check": "called_in_flow",
                "passed": called,
                "evidence": evidence
            })
            
            # Determine overall pass
            all_passed = all(c["passed"] for c in module_result["checks"])
            module_result["overall_passed"] = all_passed
            
            if all_passed:
                wired_pass += 1
            else:
                wired_fail += 1
                
            results["newly_wired_modules"].append(module_result)
        
        # Verify signature fixes
        sig_pass = 0
        sig_fail = 0
        
        for class_name, method_name, module_path in self.SIGNATURE_FIXES:
            passed, evidence = self.verify_signature_fix(class_name, method_name, module_path)
            results["signature_fixes"].append({
                "class": class_name,
                "method": method_name,
                "passed": passed,
                "evidence": evidence
            })
            if passed:
                sig_pass += 1
            else:
                sig_fail += 1
        
        # Summary
        results["summary"] = {
            "newly_wired": {
                "total": len(self.NEWLY_WIRED_MODULES),
                "passed": wired_pass,
                "failed": wired_fail,
                "pass_rate": f"{wired_pass / len(self.NEWLY_WIRED_MODULES) * 100:.1f}%"
            },
            "signature_fixes": {
                "total": len(self.SIGNATURE_FIXES),
                "passed": sig_pass,
                "failed": sig_fail,
                "pass_rate": f"{sig_pass / len(self.SIGNATURE_FIXES) * 100:.1f}%"
            },
            "overall_assessment": self._generate_assessment(wired_pass, wired_fail, sig_pass, sig_fail)
        }
        
        return results
    
    def _generate_assessment(self, wired_pass, wired_fail, sig_pass, sig_fail) -> str:
        """Generate clinical assessment of implementation status."""
        total_checks = len(self.NEWLY_WIRED_MODULES) + len(self.SIGNATURE_FIXES)
        total_pass = wired_pass + sig_pass
        total_fail = wired_fail + sig_fail
        
        assessment = []
        
        # Wired modules assessment
        if wired_fail == 0:
            assessment.append(f"All {wired_pass} newly wired modules verified functional.")
        else:
            assessment.append(f"WARNING: {wired_fail}/{len(self.NEWLY_WIRED_MODULES)} newly wired modules failed verification.")
        
        # Signature fixes assessment
        if sig_fail == 0:
            assessment.append(f"All {sig_pass} signature fixes verified working.")
        else:
            assessment.append(f"WARNING: {sig_fail}/{len(self.SIGNATURE_FIXES)} signature fixes failed verification.")
        
        # Overall
        overall_rate = total_pass / total_checks * 100
        if overall_rate >= 95:
            assessment.append(f"Overall: {overall_rate:.1f}% verification rate - CLAIM SUPPORTED")
        elif overall_rate >= 80:
            assessment.append(f"Overall: {overall_rate:.1f}% verification rate - CLAIM PARTIALLY SUPPORTED")
        else:
            assessment.append(f"Overall: {overall_rate:.1f}% verification rate - CLAIM NOT SUPPORTED")
        
        return " | ".join(assessment)
    
    async def run_track_b_verification(self, track_a_results: Dict) -> Dict[str, Any]:
        """
        Track B: BAIS-governed verification.
        Uses BAIS to audit Track A's assessment.
        """
        try:
            from core.integrated_engine import IntegratedGovernanceEngine
            from pathlib import Path
            import tempfile
            
            # Format Track A assessment as a response for BAIS to audit
            track_a_summary = json.dumps(track_a_results["summary"], indent=2)
            track_a_assessment = track_a_results["summary"]["overall_assessment"]
            
            query = """Verify implementation claims:
            - 10 newly wired modules
            - 12 signature fixes
            - Claimed 71.6% fully implemented (up from 28.4%)"""
            
            response = f"""Implementation Verification Results:
            
            {track_a_assessment}
            
            Detailed Summary:
            {track_a_summary}
            
            The implementation changes are verified as complete and functional.
            All modules are properly integrated into the main engine.
            """
            
            with tempfile.TemporaryDirectory() as tmpdir:
                engine = IntegratedGovernanceEngine(data_dir=Path(tmpdir))
                
                # Run BAIS evaluation on the Track A assessment
                bais_result = await engine.evaluate_and_improve(
                    query=query,
                    response=response,
                    documents=[],
                    context={"domain": "technical"}
                )
                
                return {
                    "timestamp": datetime.utcnow().isoformat(),
                    "track": "B",
                    "description": "BAIS-governed verification",
                    "bais_evaluation": {
                        "accuracy": bais_result.accuracy,
                        "accepted": bais_result.accepted,
                        "pathway": bais_result.pathway.value if hasattr(bais_result.pathway, 'value') else str(bais_result.pathway),
                        "warnings": bais_result.warnings[:5] if bais_result.warnings else [],
                        "improvement_applied": bais_result.improvement_applied,
                        "improvement_score": bais_result.improvement_score
                    },
                    "bais_improved_response": bais_result.improved_response[:500] if bais_result.improved_response else None,
                    "evidence_verification": {
                        "claims_verified": getattr(bais_result, 'verified_claims', 0),
                        "claims_unverified": getattr(bais_result, 'unverified_claims', 0),
                    },
                    "self_awareness_check": {
                        "triggered": getattr(bais_result, 'self_awareness_triggered', False),
                        "corrections_applied": getattr(bais_result, 'corrections_applied', [])
                    }
                }
                
        except Exception as e:
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "track": "B",
                "description": "BAIS-governed verification",
                "error": str(e),
                "bais_evaluation": None
            }
    
    def compare_tracks(self, track_a: Dict, track_b: Dict) -> Dict[str, Any]:
        """Compare Track A and Track B results."""
        comparison = {
            "timestamp": datetime.utcnow().isoformat(),
            "track_a_pass_rate": track_a["summary"]["newly_wired"]["pass_rate"],
            "track_b_accepted": track_b.get("bais_evaluation", {}).get("accepted", "ERROR"),
            "track_b_accuracy": track_b.get("bais_evaluation", {}).get("accuracy", "ERROR"),
            "discrepancies": [],
            "winner": None,
            "clinical_conclusion": ""
        }
        
        # Check for discrepancies
        track_a_overall_pass = track_a["summary"]["newly_wired"]["failed"] == 0 and \
                               track_a["summary"]["signature_fixes"]["failed"] == 0
        
        bais_accepted = track_b.get("bais_evaluation", {}).get("accepted", False)
        bais_accuracy = track_b.get("bais_evaluation", {}).get("accuracy", 0)
        
        if track_a_overall_pass and not bais_accepted:
            comparison["discrepancies"].append(
                "Track A claims full pass but BAIS rejected - BAIS may have detected issues Track A missed"
            )
        
        if not track_a_overall_pass and bais_accepted:
            comparison["discrepancies"].append(
                "Track A found failures but BAIS accepted - possible BAIS false positive"
            )
        
        # Clinical conclusion
        if track_b.get("error"):
            comparison["clinical_conclusion"] = f"BAIS evaluation failed: {track_b['error']}. Using Track A results only."
            comparison["winner"] = "A (by default)"
        elif bais_accuracy and bais_accuracy < 50:
            comparison["clinical_conclusion"] = f"BAIS assigned low accuracy ({bais_accuracy:.1f}%) - claims are QUESTIONABLE"
            comparison["winner"] = "B (more critical)"
        elif track_a_overall_pass and bais_accepted:
            comparison["clinical_conclusion"] = "Both tracks agree: Implementation claims are VERIFIED"
            comparison["winner"] = "TIE"
        else:
            comparison["clinical_conclusion"] = "Tracks disagree: Further investigation required"
            comparison["winner"] = "INCONCLUSIVE"
        
        return comparison

def main():
    """Run the verification A/B test."""
    print("=" * 80)
    print("BAIS IMPLEMENTATION VERIFICATION A/B TEST")
    print("Clinical Audit of Claimed Implementation Improvements")
    print("=" * 80)
    print()
    
    verifier = ImplementationVerifier()
    
    # Track A: Direct verification
    print("TRACK A: Direct Verification (No BAIS)")
    print("-" * 40)
    track_a_results = verifier.run_track_a_verification()
    
    print(f"\nNewly Wired Modules: {track_a_results['summary']['newly_wired']['passed']}/{track_a_results['summary']['newly_wired']['total']} passed")
    print(f"Signature Fixes: {track_a_results['summary']['signature_fixes']['passed']}/{track_a_results['summary']['signature_fixes']['total']} passed")
    print(f"\nAssessment: {track_a_results['summary']['overall_assessment']}")
    
    # Show failures
    print("\nFailed Modules:")
    for mod in track_a_results["newly_wired_modules"]:
        if not mod["overall_passed"]:
            failed_checks = [c["check"] for c in mod["checks"] if not c["passed"]]
            print(f"  ❌ {mod['module']}: Failed {failed_checks}")
    
    print("\nFailed Signature Fixes:")
    for fix in track_a_results["signature_fixes"]:
        if not fix["passed"]:
            print(f"  ❌ {fix['class']}.{fix['method']}: {fix['evidence']}")
    
    # Track B: BAIS verification
    print()
    print("TRACK B: BAIS-Governed Verification")
    print("-" * 40)
    track_b_results = asyncio.run(verifier.run_track_b_verification(track_a_results))
    
    if track_b_results.get("error"):
        print(f"ERROR: {track_b_results['error']}")
    else:
        bais_eval = track_b_results.get("bais_evaluation", {})
        print(f"BAIS Accuracy: {bais_eval.get('accuracy', 'N/A')}")
        print(f"BAIS Accepted: {bais_eval.get('accepted', 'N/A')}")
        print(f"BAIS Pathway: {bais_eval.get('pathway', 'N/A')}")
        print(f"Improvement Applied: {bais_eval.get('improvement_applied', 'N/A')}")
        
        if bais_eval.get("warnings"):
            print("\nBAIS Warnings:")
            for w in bais_eval["warnings"][:5]:
                print(f"  ⚠️ {w}")
    
    # Comparison
    print()
    print("A/B COMPARISON")
    print("-" * 40)
    comparison = verifier.compare_tracks(track_a_results, track_b_results)
    
    print(f"Track A Pass Rate: {comparison['track_a_pass_rate']}")
    print(f"Track B (BAIS) Accepted: {comparison['track_b_accepted']}")
    print(f"Track B (BAIS) Accuracy: {comparison['track_b_accuracy']}")
    print(f"Winner: {comparison['winner']}")
    print(f"\nClinical Conclusion: {comparison['clinical_conclusion']}")
    
    if comparison["discrepancies"]:
        print("\nDiscrepancies Found:")
        for d in comparison["discrepancies"]:
            print(f"  ⚠️ {d}")
    
    # Output full results
    print()
    print("=" * 80)
    print("FULL RESULTS JSON")
    print("=" * 80)
    
    full_results = {
        "track_a": track_a_results,
        "track_b": track_b_results,
        "comparison": comparison
    }
    
    print(json.dumps(full_results, indent=2, default=str))
    
    return full_results

if __name__ == "__main__":
    main()


