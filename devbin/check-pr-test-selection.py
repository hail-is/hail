#!/usr/bin/env python3
"""Predict which build.yaml test steps would be selected for a given PR.

Usage:
    python3 devbin/check-pr-test-selection.py <pr-number>
"""

import subprocess
import sys
from pathlib import Path

import yaml

from ci.build_selection import compute_requested_steps, expand_build_steps

if len(sys.argv) != 2:
    print(f'Usage: {sys.argv[0]} <pr-number>', file=sys.stderr)
    sys.exit(1)

pr_number = sys.argv[1]
changed_files = subprocess.run(
    ['gh', 'pr', 'view', pr_number, '--repo', 'hail-is/hail', '--json', 'files', '--jq', '.files[].path'],
    capture_output=True, text=True, check=True,
).stdout.strip().splitlines()

config_str = (Path(__file__).parent.parent / 'build.yaml').read_text()
selection = compute_requested_steps(config_str, changed_files)
print(selection)

config = yaml.safe_load(config_str)
cleanup_names = {s['name'] for s in config['steps'] if s.get('cleanupFor')}
expanded = expand_build_steps(config_str, selection.requested_steps, include_cleanups=True)
cleanup = [s for s in expanded if s in cleanup_names]
if cleanup:
    print(f'cleanup_steps={cleanup}')
