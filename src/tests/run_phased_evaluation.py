"""
Phased Clinical Evaluation - Runs in batches to prevent memory/resource issues
Saves progress after each phase so can resume if interrupted
"""

import sys
import os
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.comprehensive_clinical_evaluation import ComprehensiveClinicalEvaluation

# Progress file
PROGRESS_FILE = os.path.join(os.path.dirname(__file__), "evaluation_progress.json")

def load_progress():
    """Load progress from file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"completed_phases": [], "results": [], "started": datetime.now().isoformat()}

def save_progress(progress):
    """Save progress to file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2, default=str)

def run_phase(eval_instance, patent: str, batch_size: int = 20):
    """Run a single phase in batches."""
    print(f"\n{'=' * 70}")
    print(f"PHASE: {patent}")
    print(f"{'=' * 70}")
    
    claims = eval_instance.claims.get(patent, [])
    total = len(claims)
    results = []
    
    # Process in batches
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_claims = claims[batch_start:batch_end]
        
        print(f"\n[Batch {batch_start//batch_size + 1}] Processing claims {batch_start+1}-{batch_end} of {total}")
        
        for i, claim in enumerate(batch_claims):
            claim_num = batch_start + i + 1
            print(f"  [{claim_num}/{total}] {claim['id']}...", end=" ", flush=True)
            
            try:
                result = eval_instance.test_claim(claim)
                emoji = {"EXCELLENT": "✅", "GOOD": "👍", "FAIR": "⚠️", "POOR": "❌", "CRITICAL": "💥"}
                print(f"{emoji.get(result.effectiveness_rating, '?')} {result.effectiveness_score:.0f}%")
                results.append(result)
            except Exception as e:
                print(f"❌ Error: {str(e)[:30]}")
            
            # Brief pause to avoid rate limiting
            time.sleep(0.3)
        
        # Save progress after each batch
        print(f"  [Batch complete - {len(results)} results so far]")
    
    return results

def main():
    """Run phased evaluation with progress saving."""
    print("=" * 80)
    print("BAIS PHASED CLINICAL EVALUATION")
    print("Running in batches with progress saving")
    print("=" * 80)
    
    # Load any existing progress
    progress = load_progress()
    completed = progress.get("completed_phases", [])
    
    # Initialize evaluator
    eval_instance = ComprehensiveClinicalEvaluation()
    
    # Define phases
    phases = ['PPA1', 'PPA2', 'PPA3', 'UTILITY', 'NOVEL']
    
    all_results = progress.get("results", [])
    
    for patent in phases:
        if patent in completed:
            print(f"\n✓ {patent} already completed, skipping...")
            continue
        
        print(f"\n>>> Starting {patent} phase...")
        
        try:
            results = run_phase(eval_instance, patent, batch_size=15)
            
            # Convert to dict for JSON serialization
            for r in results:
                all_results.append({
                    "claim_id": r.claim_id,
                    "patent": r.patent,
                    "group": r.group,
                    "effectiveness_score": r.effectiveness_score,
                    "effectiveness_rating": r.effectiveness_rating,
                    "bais_issues_found": r.bais_issues_found,
                    "bais_would_block": r.bais_would_block,
                    "what_worked": r.what_worked,
                    "what_failed": r.what_failed,
                    "recommendations": r.recommendations,
                    "bais_self_check_passed": r.bais_self_check_passed
                })
            
            completed.append(patent)
            progress["completed_phases"] = completed
            progress["results"] = all_results
            save_progress(progress)
            
            print(f"\n✅ {patent} complete - Progress saved")
            
        except KeyboardInterrupt:
            print(f"\n⚠️ Interrupted during {patent} - Progress saved")
            progress["completed_phases"] = completed
            progress["results"] = all_results
            save_progress(progress)
            return
        except Exception as e:
            print(f"\n❌ Error in {patent}: {e}")
            save_progress(progress)
    
    # Generate final summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    total = len(all_results)
    if total > 0:
        avg_score = sum(r["effectiveness_score"] for r in all_results) / total
        passed = sum(1 for r in all_results if r["effectiveness_score"] >= 70)
        blocked = sum(1 for r in all_results if r["bais_would_block"])
        
        print(f"\nTotal Claims Tested: {total}")
        print(f"Average Effectiveness: {avg_score:.1f}%")
        print(f"Pass Rate (≥70%): {(passed/total)*100:.1f}%")
        print(f"Would Block: {blocked}")
        
        # Rating breakdown
        ratings = {}
        for r in all_results:
            rating = r["effectiveness_rating"]
            ratings[rating] = ratings.get(rating, 0) + 1
        
        print("\nRating Breakdown:")
        for rating, count in sorted(ratings.items()):
            pct = (count / total) * 100
            emoji = {"EXCELLENT": "✅", "GOOD": "👍", "FAIR": "⚠️", "POOR": "❌", "CRITICAL": "💥"}.get(rating, "?")
            print(f"  {emoji} {rating}: {count} ({pct:.1f}%)")
        
        # By patent
        print("\nBy Patent:")
        for patent in phases:
            patent_results = [r for r in all_results if r["patent"] == patent or r["claim_id"].startswith(patent)]
            if patent_results:
                patent_avg = sum(r["effectiveness_score"] for r in patent_results) / len(patent_results)
                patent_passed = sum(1 for r in patent_results if r["effectiveness_score"] >= 70)
                print(f"  {patent}: {len(patent_results)} claims, {patent_avg:.1f}% avg, {patent_passed} passed")
    
    # Save final results
    final_path = os.path.join(os.path.dirname(__file__), "phased_evaluation_final.json")
    with open(final_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_claims": total,
            "results": all_results
        }, f, indent=2)
    print(f"\n✅ Final results saved: {final_path}")

if __name__ == "__main__":
    main()






