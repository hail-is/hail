#!/usr/bin/env python3
"""Show watchedPaths configuration for steps in build.yaml.

Usage:
    python3 devbin/show-watched-paths.py              # all steps with watchedPaths
    python3 devbin/show-watched-paths.py step1 step2  # specific steps only
"""

import sys
from pathlib import Path

import yaml

config = yaml.safe_load((Path(__file__).parent.parent / 'build.yaml').read_text())
filter_names = set(sys.argv[1:])

full_retest_paths = config.get('fullRetestPaths', [])

if not filter_names:
    print('- fullRetestPaths:')
    for p in full_retest_paths:
        print(f'    - {p}')
    print()

for step in config['steps']:
    name = step['name']
    paths = step.get('watchedPaths')
    if not paths:
        continue
    if filter_names and name not in filter_names:
        continue
    print(f'- {name}:')
    if desc := step.get('description'):
        print(f'    - description: {desc}')
    print('    - watchedPaths:')
    for p in paths:
        print(f'        - {p}')
