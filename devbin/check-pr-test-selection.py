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

config = yaml.safe_load(config_str)
all_cleanup_names = {s['name'] for s in config['steps'] if s.get('cleanupFor')}
run_if_requested = {s['name'] for s in config['steps'] if s.get('runIfRequested', False)}
expanded = expand_build_steps(config_str, selection.requested_steps, include_cleanups=True)
expanded_set = set(expanded)
requested_set = set(selection.requested_steps)

selected_cleanups = sorted(s for s in expanded if s in all_cleanup_names)
unselected_cleanups = sorted(all_cleanup_names - expanded_set)
excluded_watched = sorted(
    s['name'] for s in config['steps']
    if s.get('watchedPaths')
    and s['name'] not in expanded_set
    and s['name'] not in run_if_requested
)
dep_only = {
    s['name'] for s in config['steps']
    if not s.get('watchedPaths')
    and not s.get('runIfRequested', False)
    and s['name'] not in all_cleanup_names
}
included_deps = sorted(expanded_set & dep_only - requested_set)
excluded_deps = sorted(dep_only - expanded_set)

print(f'changed_files={changed_files}')
print(selection)
if included_deps:
    print(f'included_test_dependencies={included_deps}')
if selected_cleanups:
    print(f'included_cleanup_steps={selected_cleanups}')
if excluded_watched:
    print(f'excluded_test_steps_with_watched_paths={excluded_watched}')
if excluded_deps:
    print(f'excluded_test_dependencies={excluded_deps}')
if unselected_cleanups:
    print(f'excluded_cleanup_steps={unselected_cleanups}')
