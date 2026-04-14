#!/usr/bin/env python3
"""Predict which build.yaml test steps would be selected for a given PR.

Usage:
    python3 devbin/check-pr-test-selection.py <pr-number>
    python3 devbin/check-pr-test-selection.py 15394
"""

import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'ci'))

from ci.build_selection import compute_requested_steps  # noqa: E402


def get_pr_files(pr_number: int) -> list[str]:
    result = subprocess.run(
        ['gh', 'pr', 'view', str(pr_number), '--repo', 'hail-is/hail', '--json', 'files', '--jq', '.files[].path'],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().splitlines()


def main():
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <pr-number>', file=sys.stderr)
        sys.exit(1)

    pr_number = int(sys.argv[1])
    changed_files = get_pr_files(pr_number)

    print(f'PR #{pr_number} — {len(changed_files)} changed file(s):')
    for f in changed_files:
        print(f'  {f}')
    print()

    config_str = (repo_root / 'build.yaml').read_text()
    result = compute_requested_steps(config_str, changed_files)

    print(f'Selected steps ({len(result.requested_steps)}):')
    for step in result.requested_steps:
        print(f'  {step}')

    if result.full_retest_triggers:
        print(f'\nFull-retest triggers ({len(result.full_retest_triggers)}) — unmapped files that forced all steps:')
        for f in result.full_retest_triggers:
            print(f'  {f}')
    else:
        print('\nNo full-retest triggers — all changed files were cleanly mapped.')


if __name__ == '__main__':
    main()
