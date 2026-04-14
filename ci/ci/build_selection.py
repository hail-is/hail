"""Change-based test selection helpers.

This module is intentionally free of cloud/environment imports so it can be
used in unit tests without a running deployment.
"""

import fnmatch
from dataclasses import dataclass, field
from typing import List, Set

import yaml


@dataclass
class BuildSelectionResult:
    """Result of change-based step selection.

    Attributes:
        requested_steps: Sorted list of step names that should run.
        full_retest_triggers: Sorted list of changed files that matched no
            watchedPaths pattern (and therefore caused all watched steps to be
            included as a fallback).
    """

    requested_steps: List[str] = field(default_factory=list)
    full_retest_triggers: List[str] = field(default_factory=list)


def _path_matches(changed_file: str, pattern: str) -> bool:
    """Match a changed file path against a watchedPaths/unwatchedPaths pattern.

    Patterns ending in '/' match any file under that directory.
    Otherwise uses case-insensitive fnmatch glob matching.
    """
    if pattern.endswith('/'):
        cf = changed_file.lower()
        p = pattern.lower()
        return cf.startswith(p) or cf == p.rstrip('/')
    return fnmatch.fnmatch(changed_file.lower(), pattern.lower())


def compute_requested_steps(config_str: str, changed_files: List[str]) -> BuildSelectionResult:
    """Return step selection results given a list of changed file paths.

    - alwaysRunSteps are always included when changed_files is non-empty.
    - Files matching unwatchedPaths contribute nothing beyond the always-run set.
    - Files matching a step's watchedPaths add that step.
    - Files matching no step's watchedPaths (fallback) add all watched steps and
      are recorded in full_retest_triggers for observability.
    - Returns an empty BuildSelectionResult when changed_files is empty.
    """
    if not changed_files:
        return BuildSelectionResult()

    config = yaml.safe_load(config_str)
    unwatched: List[str] = config.get('unwatchedPaths', [])
    always_run: List[str] = config.get('alwaysRunSteps', [])
    watched_steps = [s for s in config['steps'] if s.get('watchedPaths')]

    target: Set[str] = set(always_run)
    all_watched_names = {s['name'] for s in watched_steps}
    fallthrough: List[str] = []

    for f in changed_files:
        if any(_path_matches(f, p) for p in unwatched):
            continue
        matched = {s['name'] for s in watched_steps if any(_path_matches(f, p) for p in s['watchedPaths'])}
        if matched:
            target |= matched
        else:
            target |= all_watched_names  # unknown file → run everything
            fallthrough.append(f)

    return BuildSelectionResult(requested_steps=sorted(target), full_retest_triggers=sorted(fallthrough))
