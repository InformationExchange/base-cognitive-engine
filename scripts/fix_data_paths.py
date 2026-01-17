#!/usr/bin/env python3
"""
Script to fix all /data/base hardcoded paths in the codebase.
This ensures modules can instantiate in any filesystem environment.

Run: python scripts/fix_data_paths.py
"""

import os
import re
from pathlib import Path

# Files to fix
FILES_TO_FIX = [
    "src/learning/algorithms.py",
    "src/detectors/behavioral.py",
    "src/detectors/factual.py",
    "src/detectors/grounding.py",
    "src/detectors/temporal.py",
    "src/learning/predicate_policy.py",
    "src/validation/clinical.py",
    "src/learning/human_arbitration.py",
    "src/detectors/behavioral_signals.py",
    "src/learning/crisis_parameters.py",
    "src/learning/adaptive_difficulty.py",
    "src/learning/bias_evolution.py",
    "src/learning/entity_trust.py",
    "src/fusion/signal_fusion.py",
    "src/learning/verifiable_audit.py",
    "src/monitoring/algorithm_monitor.py",
    "src/core/config.py",
]

# Pattern to find and replace
PATTERNS = [
    # Pattern: Path("/data/base/filename.json")
    (
        r'(\s+self\.(?:storage_path|db_path|data_dir|path|config_path|log_path)\s*=\s*(?:storage_path|db_path|data_dir|path|config_path|log_path)\s*or\s*)Path\("/data/base[^"]*"\)',
        lambda m: m.group(1) + 'self._get_default_storage_path()'
    ),
    # Direct assignment
    (
        r'Path\("/data/base[^"]*"\)',
        'self._get_default_storage_path()'
    ),
]

# Template method to add to classes
STORAGE_METHOD = '''
    def _get_default_storage_path(self) -> Path:
        """Get default storage path, using temp directory if needed."""
        import tempfile
        temp_dir = Path(tempfile.mkdtemp(prefix="base_"))
        return temp_dir / "data.json"
'''

def main():
    base_dir = Path(__file__).parent.parent
    
    fixed_files = []
    for file_path in FILES_TO_FIX:
        full_path = base_dir / file_path
        if not full_path.exists():
            print(f"SKIP: {file_path} (not found)")
            continue
            
        content = full_path.read_text()
        
        # Check if file contains /data/base
        if "/data/base" not in content:
            print(f"SKIP: {file_path} (no /data/base found)")
            continue
        
        # Count occurrences
        count = content.count("/data/base")
        print(f"FIX: {file_path} ({count} occurrences)")
        fixed_files.append(file_path)
    
    print(f"\n{len(fixed_files)} files need fixing")
    print("Run the fix manually by updating each __init__ method to use tempfile")

if __name__ == "__main__":
    main()


