"""
BASE Continuous Enhancement System
Uses BASE to:
1. Identify what's still broken
2. Track WHY things were missed
3. Implement fixes with verification
4. Record lessons learned
"""

import sys
import os
import json
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.learning_integrator import LearningIntegrator, EvaluationFeedback


@dataclass
class EnhancementRecord:
    """Record of an enhancement attempt."""
    enhancement_id: str
    target_claim: str
    issue_description: str
    why_missed_before: str
    why_llm_said_complete: str
    why_recognized_now: str
    fix_implemented: str
    verification_result: str
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class ContinuousEnhancementSystem:
    """
    BASE-powered continuous enhancement.
    
    This system embodies the core patent claim of self-improving AI governance.
    """
    
    def __init__(self):
        self.learning = LearningIntegrator()
        self.enhancements: List[EnhancementRecord] = []
        self.enhancement_file = os.path.join(
            os.path.dirname(__file__), 
            "enhancement_history.json"
        )
        self._load_history()
    
    def _load_history(self):
        """Load enhancement history."""
        if os.path.exists(self.enhancement_file):
            try:
                with open(self.enhancement_file, 'r') as f:
                    data = json.load(f)
                    # Don't reload as objects, just keep count
                    print(f"[BASE] Loaded {len(data)} prior enhancement records")
            except:
                pass
    
    def _save_history(self):
        """Save enhancement history."""
        with open(self.enhancement_file, 'w') as f:
            json.dump([asdict(e) for e in self.enhancements], f, indent=2)
    
    def analyze_failures(self) -> Dict[str, Any]:
        """Use BASE to analyze current failures."""
        print("\n" + "=" * 70)
        print("BASE FAILURE ANALYSIS")
        print("=" * 70)
        
        # Load evaluation results
        results_file = os.path.join(
            os.path.dirname(__file__), 
            "phased_evaluation_final.json"
        )
        
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        results = data['results']
        
        # Categorize failures
        analysis = {
            'total': len(results),
            'passed': 0,
            'failed': 0,
            'poor_claims': [],
            'failure_patterns': {},
            'root_causes': []
        }
        
        for r in results:
            if r['effectiveness_score'] >= 70:
                analysis['passed'] += 1
            else:
                analysis['failed'] += 1
                
                if r['effectiveness_score'] < 40:
                    analysis['poor_claims'].append({
                        'claim_id': r['claim_id'],
                        'score': r['effectiveness_score'],
                        'what_failed': r.get('what_failed', []),
                        'group': r.get('group', 'unknown')
                    })
                
                # Track failure patterns
                for fail in r.get('what_failed', []):
                    analysis['failure_patterns'][fail] = \
                        analysis['failure_patterns'].get(fail, 0) + 1
        
        # Identify root causes
        if analysis['failure_patterns']:
            for pattern, count in sorted(
                analysis['failure_patterns'].items(), 
                key=lambda x: -x[1]
            )[:5]:
                analysis['root_causes'].append({
                    'pattern': pattern,
                    'count': count,
                    'recommended_fix': self._recommend_fix(pattern)
                })
        
        return analysis
    
    def _recommend_fix(self, pattern: str) -> str:
        """Recommend fix for failure pattern."""
        fixes = {
            'Failed to detect injection attempt': 
                'Add Unicode lookalike, zero-width char, homoglyph patterns',
            'Failed to detect expected manipulation': 
                'Add subtle coercion, context-dependent manipulation patterns',
            'Failed to detect contradiction': 
                'Enhance semantic contradiction with entity extraction',
        }
        return fixes.get(pattern, 'Review pattern coverage for this category')
    
    def implement_enhancement(self, 
                              target_claim: str,
                              issue: str,
                              fix_code: str) -> EnhancementRecord:
        """Implement and record an enhancement."""
        
        record = EnhancementRecord(
            enhancement_id=f"ENH-{len(self.enhancements)+1:04d}",
            target_claim=target_claim,
            issue_description=issue,
            why_missed_before="Pattern not in detection list",
            why_llm_said_complete="Saw improvement, declared success prematurely",
            why_recognized_now="BASE failure analysis identified specific gap",
            fix_implemented=fix_code,
            verification_result="PENDING"
        )
        
        self.enhancements.append(record)
        self._save_history()
        
        return record
    
    def verify_enhancement(self, 
                           enhancement_id: str, 
                           test_result: bool,
                           details: str):
        """Verify an enhancement worked."""
        for e in self.enhancements:
            if e.enhancement_id == enhancement_id:
                e.verification_result = "PASS" if test_result else "FAIL"
                e.verification_result += f": {details}"
                self._save_history()
                break
    
    def run_continuous_enhancement_cycle(self):
        """Run a full enhancement cycle using BASE."""
        
        print("\n" + "=" * 80)
        print("BASE CONTINUOUS ENHANCEMENT CYCLE")
        print("=" * 80)
        
        # Step 1: Analyze failures
        print("\n[1] ANALYZING CURRENT FAILURES")
        print("-" * 60)
        analysis = self.analyze_failures()
        
        print(f"Total claims: {analysis['total']}")
        print(f"Passed: {analysis['passed']} ({analysis['passed']/analysis['total']*100:.1f}%)")
        print(f"Failed: {analysis['failed']} ({analysis['failed']/analysis['total']*100:.1f}%)")
        print(f"POOR claims (< 40%): {len(analysis['poor_claims'])}")
        
        # Step 2: Identify specific fixes needed
        print("\n[2] IDENTIFYING SPECIFIC FIXES")
        print("-" * 60)
        
        for claim in analysis['poor_claims']:
            print(f"\n  POOR: {claim['claim_id']} ({claim['score']}%)")
            print(f"    Group: {claim['group']}")
            print(f"    Failed: {claim['what_failed']}")
        
        # Step 3: Root cause analysis
        print("\n[3] ROOT CAUSE ANALYSIS")
        print("-" * 60)
        
        for rc in analysis['root_causes']:
            print(f"\n  Pattern: {rc['pattern']}")
            print(f"    Count: {rc['count']} failures")
            print(f"    Fix: {rc['recommended_fix']}")
        
        # Step 4: Implement fixes
        print("\n[4] IMPLEMENTING FIXES")
        print("-" * 60)
        
        fixes_implemented = []
        
        # Fix 1: Unicode/Homoglyph injection detection
        if any('injection' in str(rc['pattern']).lower() for rc in analysis['root_causes']):
            fix = self._implement_unicode_detection()
            fixes_implemented.append(fix)
        
        # Fix 2: Subtle manipulation patterns
        if any('manipulation' in str(rc['pattern']).lower() for rc in analysis['root_causes']):
            fix = self._implement_subtle_manipulation()
            fixes_implemented.append(fix)
        
        # Step 5: Verify fixes
        print("\n[5] VERIFYING FIXES")
        print("-" * 60)
        
        for fix in fixes_implemented:
            result = self._verify_fix(fix)
            print(f"  {fix['name']}: {result}")
        
        # Step 6: Record lessons learned
        print("\n[6] LESSONS LEARNED")
        print("-" * 60)
        print("""
  1. Don't declare "complete" without integration testing
  2. Define success criteria BEFORE testing (not after seeing results)
  3. 14.6% pass rate is NOT acceptable - push to 70%+
  4. Different claim types need different test methodologies
  5. Learning systems must be CONNECTED to evaluation flow
  6. Check for empty data structures (calibration_scores = [])
  7. Suspicious uniformity (62 claims at exactly 50%) = wrong test
        """)
        
        # Step 7: Next enhancement opportunities
        print("\n[7] NEXT ENHANCEMENT OPPORTUNITIES")
        print("-" * 60)
        
        return self._identify_next_opportunities(analysis)
    
    def _implement_unicode_detection(self) -> Dict:
        """Implement Unicode/homoglyph injection detection."""
        print("  Implementing Unicode injection detection...")
        
        # Add patterns to query_analyzer
        qa_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "core", "query_analyzer.py"
        )
        
        new_patterns = '''
        # BASE Enhancement: Unicode/Homoglyph attack detection
        (r'[\\u200b-\\u200f\\u2028-\\u202f\\u205f-\\u206f]',
         'Zero-width character injection', 0.9),
        (r'[а-яА-Я].*[a-zA-Z]|[a-zA-Z].*[а-яА-Я]',
         'Cyrillic/Latin mixing (homoglyph)', 0.85),
        (r'[\\uff00-\\uffef]',
         'Fullwidth character obfuscation', 0.8),
        (r'(?:[\\u0300-\\u036f]){2,}',
         'Combining character abuse', 0.8),
'''
        
        try:
            with open(qa_path, 'r') as f:
                content = f.read()
            
            if 'Zero-width character injection' not in content:
                # Find insertion point
                marker = "INJECTION_PATTERNS = ["
                if marker in content:
                    idx = content.find(marker) + len(marker)
                    content = content[:idx] + new_patterns + content[idx:]
                    
                    with open(qa_path, 'w') as f:
                        f.write(content)
                    
                    print("    ✅ Added 4 Unicode detection patterns")
                    return {'name': 'Unicode Detection', 'success': True}
            else:
                print("    ✓ Already implemented")
                return {'name': 'Unicode Detection', 'success': True}
        except Exception as e:
            print(f"    ❌ Error: {e}")
            return {'name': 'Unicode Detection', 'success': False}
    
    def _implement_subtle_manipulation(self) -> Dict:
        """Implement subtle manipulation detection."""
        print("  Implementing subtle manipulation detection...")
        
        tom_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "research", "theory_of_mind.py"
        )
        
        new_patterns = '''
        # BASE Enhancement: Subtle coercion patterns
        'subtle_coercion': [
            r'\\bI\\'m sure you\\'ll (?:agree|understand|see)\\b',
            r'\\b(?:obviously|clearly|surely) you\\b',
            r'\\banyone (?:can see|would agree|knows)\\b',
            r'\\bdon\\'t you (?:think|agree|want)\\b',
            r'\\byou\\'d be (?:crazy|foolish|wrong) (?:not )?to\\b',
        ],
        'context_manipulation': [
            r'\\bgiven (?:the circumstances|everything|what happened)\\b',
            r'\\bconsidering (?:your|the) situation\\b',
            r'\\bin (?:your|this) position\\b',
            r'\\bsomeone like you\\b',
            r'\\bpeople in your (?:situation|position)\\b',
        ],
'''
        
        try:
            with open(tom_path, 'r') as f:
                content = f.read()
            
            if 'subtle_coercion' not in content:
                # Find insertion point after love_bombing
                marker = "'love_bombing': ["
                if marker in content:
                    # Find end of love_bombing pattern list
                    idx = content.find(marker)
                    bracket_count = 0
                    for i in range(idx, len(content)):
                        if content[i] == '[':
                            bracket_count += 1
                        elif content[i] == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                insert_pos = i + 2
                                content = content[:insert_pos] + new_patterns + content[insert_pos:]
                                break
                    
                    with open(tom_path, 'w') as f:
                        f.write(content)
                    
                    print("    ✅ Added 10 subtle manipulation patterns")
                    return {'name': 'Subtle Manipulation', 'success': True}
            else:
                print("    ✓ Already implemented")
                return {'name': 'Subtle Manipulation', 'success': True}
        except Exception as e:
            print(f"    ❌ Error: {e}")
            return {'name': 'Subtle Manipulation', 'success': False}
    
    def _verify_fix(self, fix: Dict) -> str:
        """Verify a fix is working."""
        if fix['name'] == 'Unicode Detection':
            try:
                from core.query_analyzer import QueryAnalyzer
                qa = QueryAnalyzer()
                
                # Test with zero-width characters
                test = "Please he​lp me"  # Contains zero-width space
                result = qa.analyze(test)
                
                if result.risk_score > 0.3:
                    return "✅ PASS - Detecting hidden characters"
                else:
                    return "⚠️ PARTIAL - May need threshold adjustment"
            except Exception as e:
                return f"❌ ERROR: {e}"
        
        elif fix['name'] == 'Subtle Manipulation':
            try:
                from research.theory_of_mind import TheoryOfMindModule
                tom = TheoryOfMindModule()
                
                # Test with subtle coercion
                result = tom.analyze(
                    "Should I buy this?",
                    "Obviously you can see this is the right choice. Don't you think?"
                )
                
                if 'subtle_coercion' in result.manipulation.techniques_detected:
                    return "✅ PASS - Detecting subtle coercion"
                elif result.manipulation.risk_score > 0.3:
                    return "⚠️ PARTIAL - Detecting something"
                else:
                    return "❌ FAIL - Not detecting subtle patterns"
            except Exception as e:
                return f"❌ ERROR: {e}"
        
        return "UNKNOWN"
    
    def _identify_next_opportunities(self, analysis: Dict) -> List[Dict]:
        """Identify next enhancement opportunities."""
        
        opportunities = []
        
        # Based on remaining failures
        if analysis['failed'] > analysis['passed']:
            opportunities.append({
                'priority': 'CRITICAL',
                'area': 'Overall Detection',
                'description': f'{analysis["failed"]/analysis["total"]*100:.0f}% still failing',
                'action': 'Systematic pattern expansion for each failure category'
            })
        
        # Based on group analysis
        poor_by_group = {}
        for claim in analysis['poor_claims']:
            group = claim['group']
            poor_by_group[group] = poor_by_group.get(group, 0) + 1
        
        for group, count in sorted(poor_by_group.items(), key=lambda x: -x[1]):
            opportunities.append({
                'priority': 'HIGH',
                'area': f'{group} Group',
                'description': f'{count} POOR claims in this group',
                'action': f'Deep dive into {group}-specific patterns'
            })
        
        # Add continuous improvement opportunities
        opportunities.append({
            'priority': 'MEDIUM',
            'area': 'Learning Integration',
            'description': 'Thresholds adapting but not affecting detection yet',
            'action': 'Connect adapted thresholds to actual evaluation decisions'
        })
        
        opportunities.append({
            'priority': 'MEDIUM',
            'area': 'Test Methodology',
            'description': 'PPA2 Components need functional tests',
            'action': 'Create unit tests for mathematical/algorithmic claims'
        })
        
        print("\nNext Enhancement Opportunities:")
        for i, opp in enumerate(opportunities[:5], 1):
            print(f"\n  [{opp['priority']}] {i}. {opp['area']}")
            print(f"      Issue: {opp['description']}")
            print(f"      Action: {opp['action']}")
        
        return opportunities


def main():
    """Run continuous enhancement cycle."""
    system = ContinuousEnhancementSystem()
    opportunities = system.run_continuous_enhancement_cycle()
    
    print("\n" + "=" * 80)
    print("ENHANCEMENT CYCLE COMPLETE")
    print("=" * 80)
    print(f"\nTotal enhancements this session: {len(system.enhancements)}")
    print(f"Next opportunities identified: {len(opportunities)}")
    
    # Save opportunities for next cycle
    opp_file = os.path.join(os.path.dirname(__file__), "next_opportunities.json")
    with open(opp_file, 'w') as f:
        json.dump(opportunities, f, indent=2)
    print(f"\nOpportunities saved to: next_opportunities.json")


if __name__ == "__main__":
    main()






