"""Change-based test selection helpers.

Derives step selection from each step's `inputs` declarations in build.yaml.
Steps that directly import changed repository files are seeds; all transitive
dependents in the dependsOn DAG are included.

This module is intentionally free of cloud/environment imports so it can be
used in tests and tooling without a running deployment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import yaml


@dataclass
class BuildSelectionResult:
    """Result of change-based step selection.

    Attributes:
        requested_steps: Sorted list of step names that should run.
    """

    requested_steps: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f'requested_steps={self.requested_steps}'


def _repo_input_local_path(from_path: str) -> Optional[str]:
    """Return the local path from a /repo/... input, or None if not a repo input.

    '/repo'   -> ''       (matches everything)
    '/repo/x' -> 'x'
    other     -> None
    """
    if from_path == '/repo':
        return ''
    if from_path.startswith('/repo/'):
        return from_path[len('/repo/') :]
    return None


def _file_matches_input(changed_file: str, local_path: str) -> bool:
    """True if changed_file is at or under local_path."""
    if local_path == '':
        return True
    return changed_file == local_path or changed_file.startswith(local_path + '/')


def _in_scope(scopes: Optional[List[str]], scope: str) -> bool:
    return scopes is None or scope in scopes


def _in_cloud(clouds: Optional[List[str]], cloud: Optional[str]) -> bool:
    if cloud is None or clouds is None:
        return True
    return cloud in clouds


def compute_requested_steps(
    config_str: str,
    changed_files: List[str],
    scope: str = 'test',
    cloud: Optional[str] = None,
) -> BuildSelectionResult:
    """Return step selection results given a list of changed file paths.

    Finds steps directly affected (those with /repo/ inputs matching changed
    files), then follows dependsOn edges forward to find all transitive
    descendants. Only steps runnable in `scope` and `cloud` are returned; this
    prevents deploy-only or wrong-cloud steps from pulling in unrelated
    upstream deps via BuildConfiguration.visit_dependent.

    Returns an empty BuildSelectionResult when changed_files is empty.
    """
    if not changed_files:
        return BuildSelectionResult()

    config = yaml.safe_load(config_str)
    steps = config.get('steps', [])

    scopes_map: Dict[str, Optional[List[str]]] = {s['name']: s.get('scopes') for s in steps}
    clouds_map: Dict[str, Optional[List[str]]] = {s['name']: s.get('clouds') for s in steps}

    def runnable(name: str) -> bool:
        return _in_scope(scopes_map.get(name), scope) and _in_cloud(clouds_map.get(name), cloud)

    # Build reverse adjacency: step_name -> list of steps that depend on it
    reverse: Dict[str, List[str]] = {}
    for step in steps:
        for dep in step.get('dependsOn', []):
            reverse.setdefault(dep, []).append(step['name'])

    # Find seed steps: those whose /repo/ inputs match any changed file,
    # restricted to steps runnable in the target scope and cloud.
    seed_steps: Set[str] = set()
    for step in steps:
        if not runnable(step['name']):
            continue
        inputs = step.get('inputs') or []
        for inp in inputs:
            local_path = _repo_input_local_path(inp.get('from', ''))
            if local_path is None:
                continue
            if any(_file_matches_input(f, local_path) for f in changed_files):
                seed_steps.add(step['name'])
                break

    # BFS forward from seeds to find all runnable descendants
    result: Set[str] = set(seed_steps)
    frontier = list(seed_steps)
    while frontier:
        cur = frontier.pop()
        for dependent in reverse.get(cur, []):
            if dependent not in result and runnable(dependent):
                result.add(dependent)
                frontier.append(dependent)

    return BuildSelectionResult(requested_steps=sorted(result))
