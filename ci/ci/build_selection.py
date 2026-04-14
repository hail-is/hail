"""Change-based test selection helpers.

This module is intentionally free of cloud/environment imports so it can be
used in unit tests without a running deployment.
"""

import fnmatch
from typing import List, Set

import yaml


def _path_matches(changed_file: str, pattern: str) -> bool:
    """Match a changed file path against a watchedPaths/unwatchedPaths pattern.

    Patterns ending in '/' match any file under that directory.
    Otherwise uses fnmatch glob matching.
    """
    if pattern.endswith('/'):
        return changed_file.startswith(pattern) or changed_file == pattern.rstrip('/')
    return fnmatch.fnmatch(changed_file, pattern)


def compute_requested_steps(config_str: str, changed_files: List[str]) -> List[str]:
    """Return the minimal set of step names to run given a list of changed file paths.

    - alwaysRunSteps are always included when changed_files is non-empty.
    - Files matching unwatchedPaths contribute nothing beyond the always-run set.
    - Files matching a step's watchedPaths add that step.
    - Files matching no step's watchedPaths (fallback) add all watched steps.
    - Returns [] when changed_files is empty.
    """
    if not changed_files:
        return []

    config = yaml.safe_load(config_str)
    unwatched: List[str] = config.get('unwatchedPaths', [])
    always_run: List[str] = config.get('alwaysRunSteps', [])
    watched_steps = [s for s in config['steps'] if s.get('watchedPaths')]

    target: Set[str] = set(always_run)
    all_watched_names = {s['name'] for s in watched_steps}

    for f in changed_files:
        if any(_path_matches(f, p) for p in unwatched):
            continue
        matched = {s['name'] for s in watched_steps if any(_path_matches(f, p) for p in s['watchedPaths'])}
        if matched:
            target |= matched
        else:
            target |= all_watched_names  # unknown file → run everything

    return sorted(target)
