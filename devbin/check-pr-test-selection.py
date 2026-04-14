#!/usr/bin/env python3
"""Predict which build.yaml test steps would be selected for a given PR.

Usage:
    python3 devbin/check-pr-test-selection.py <pr-number>
"""

import subprocess
import sys
from pathlib import Path

from ci.build_selection import compute_requested_steps

if len(sys.argv) != 2:
    print(f'Usage: {sys.argv[0]} <pr-number>', file=sys.stderr)
    sys.exit(1)

pr_number = sys.argv[1]
changed_files = subprocess.run(
    ['gh', 'pr', 'view', pr_number, '--repo', 'hail-is/hail', '--json', 'files', '--jq', '.files[].path'],
    capture_output=True, text=True, check=True,
).stdout.strip().splitlines()

config_str = (Path(__file__).parent.parent / 'build.yaml').read_text()
print(compute_requested_steps(config_str, changed_files))
