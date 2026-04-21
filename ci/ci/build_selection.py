"""Change-based test selection helpers.

This module is intentionally free of cloud/environment imports so it can be
used in unit tests without a running deployment.
"""

import fnmatch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import yaml


@dataclass
class BuildSelectionResult:
    """Result of change-based step selection.

    Attributes:
        requested_steps: Sorted list of step names that should run.
        full_retest_triggers: Sorted list of changed files that matched a
            fullRetestPaths pattern (and therefore caused all watched steps to
            be included).
    """

    requested_steps: List[str] = field(default_factory=list)
    full_retest_triggers: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f'requested_steps={self.requested_steps}\nfull_retest_triggers={self.full_retest_triggers}'


def _transitive_closure(
    names: Set[str],
    deps_map: Dict[str, List[str]],
    run_if_requested: Set[str],
    excluded: Set[str],
    cloud: Optional[str],
    clouds_map: Dict[str, Optional[List[str]]],
) -> Set[str]:
    """Transitively expand names over dependsOn, skipping runIfRequested and wrong-cloud steps."""
    visited: Set[str] = set()
    frontier = list(names)
    while frontier:
        cur = frontier.pop()
        if cur in visited or cur in excluded:
            continue
        if cloud is not None:
            step_clouds = clouds_map.get(cur)
            if step_clouds is not None and cloud not in step_clouds:
                continue
        visited.add(cur)
        for dep in deps_map.get(cur, []):
            if dep not in run_if_requested:
                frontier.append(dep)
    return visited


def expand_build_steps(
    config_str: str,
    requested_step_names: Sequence[str],
    cloud: Optional[str] = None,
    excluded_step_names: Sequence[str] = (),
    include_cleanups: bool = False,
) -> List[str]:
    """Expand a list of requested step names into the full set that should run.

    - Transitively closes requested_step_names over dependsOn (skipping
      runIfRequested deps and steps not runnable in cloud).
    - When include_cleanups=True, also adds cleanup steps (those declaring
      cleanupFor) whose target is transitively reachable from the expanded set.
    - Returns a sorted list of all step names to include.

    This is a pure function with no cloud/environment imports, suitable for
    unit tests and for use before BuildConfiguration is constructed.
    """
    config = yaml.safe_load(config_str)
    steps = config.get('steps', [])

    excluded = set(excluded_step_names)
    run_if_requested: Set[str] = {s['name'] for s in steps if s.get('runIfRequested', False)}
    deps_map: Dict[str, List[str]] = {s['name']: s.get('dependsOn', []) for s in steps}
    clouds_map: Dict[str, Optional[List[str]]] = {s['name']: s.get('clouds') for s in steps}

    expanded = _transitive_closure(
        set(requested_step_names),
        deps_map,
        run_if_requested,
        excluded,
        cloud,
        clouds_map,
    )

    if include_cleanups:
        # Include cleanup steps whose cleanupFor target is in the expanded set.
        cleanup_names = {s['name'] for s in steps if s.get('cleanupFor') in expanded}
        expanded |= cleanup_names

    return sorted(expanded)


def _path_matches(changed_file: str, pattern: str) -> bool:
    """Match a changed file path against a watchedPaths/unwatchedPaths pattern.

    Patterns ending in '/' match any file under that directory.
    Otherwise uses case-insensitive fnmatch glob matching.
    """
    if pattern.endswith('/'):
        cf = changed_file.lower()
        p = pattern.lower()
        return cf.startswith(p)
    return fnmatch.fnmatch(changed_file.lower(), pattern.lower())


def compute_requested_steps(config_str: str, changed_files: List[str]) -> BuildSelectionResult:
    """Return step selection results given a list of changed file paths.

    - alwaysRunSteps are always included when changed_files is non-empty.
    - Files matching unwatchedPaths are skipped (explicit no-op list, e.g. doc extensions).
    - Files matching fullRetestPaths add all watched steps and are recorded in
      full_retest_triggers for observability.
    - Files matching a step's watchedPaths add that step.
    - Files matching none of the above contribute nothing (safe default).
    - Returns an empty BuildSelectionResult when changed_files is empty.
    """
    if not changed_files:
        return BuildSelectionResult()

    config = yaml.safe_load(config_str)
    unwatched: List[str] = config.get('unwatchedPaths', [])
    full_retest: List[str] = config.get('fullRetestPaths', [])
    always_run: List[str] = config.get('alwaysRunSteps', [])
    watched_steps = [s for s in config['steps'] if s.get('watchedPaths')]

    target: Set[str] = set(always_run)
    all_watched_names = {s['name'] for s in watched_steps}
    full_retest_triggers: List[str] = []

    for f in changed_files:
        if any(_path_matches(f, p) for p in unwatched):
            continue
        if any(_path_matches(f, p) for p in full_retest):
            target |= all_watched_names
            full_retest_triggers.append(f)
            continue
        matched = {s['name'] for s in watched_steps if any(_path_matches(f, p) for p in s['watchedPaths'])}
        target |= matched  # unknown file → nothing

    return BuildSelectionResult(
        requested_steps=sorted(target),
        full_retest_triggers=sorted(full_retest_triggers),
    )
