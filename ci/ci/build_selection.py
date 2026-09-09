from typing import Dict, List, Optional, Sequence, Set

import yaml


def _repo_input_local_path(from_path: str, repo_prefix: str = '/repo') -> Optional[str]:
    """Return the repo-relative path for a repo input, or None if not a repo input.

    '/repo'   -> ''       (matches everything)
    '/repo/x' -> 'x'
    other     -> None
    """
    if from_path == repo_prefix:
        return ''
    if from_path.startswith(repo_prefix + '/'):
        return from_path[len(repo_prefix) + 1 :]
    return None


def _file_matches_input(changed_file: str, local_path: str) -> bool:
    """True if changed_file is at or under local_path."""
    if local_path == '':
        return True
    return changed_file == local_path or changed_file.startswith(local_path + '/')


def _valid_step(step: dict, actual_scope: str, actual_cloud: Optional[str]) -> bool:
    step_scopes = step.get('scopes')
    step_clouds = step.get('clouds')
    return (
        not step.get('runIfRequested')
        and (step_scopes is None or actual_scope in step_scopes)
        and (step_clouds is None or actual_cloud is None or actual_cloud in step_clouds)
    )


def _find_affected_steps(
    steps: list,
    changed_files: List[str],
    repo_prefix: str = '/repo',
) -> Set[str]:
    """Return steps with repo inputs that overlap with the changed files."""
    affected_steps: Set[str] = set()
    for step in steps:
        inputs = step.get('inputs') or []
        for inp in inputs:
            local_path = _repo_input_local_path(inp.get('from', ''), repo_prefix)
            if local_path is None:
                continue
            if any(_file_matches_input(f, local_path) for f in changed_files):
                affected_steps.add(step['name'])
                break
    return affected_steps


def _expand_to_descendants(
    affected_steps: Set[str],
    descendants_map: Dict[str, List[str]],
) -> Set[str]:
    """Return affected_steps plus all steps that transitively depend on them."""
    result: Set[str] = set(affected_steps)
    steps_to_review = set(affected_steps)
    while steps_to_review:
        cur = steps_to_review.pop()
        for dependent in descendants_map.get(cur, []):
            if dependent not in result:
                result.add(dependent)
                steps_to_review.add(dependent)
    return result


def _ancestors_closure(seeds: Set[str], deps_map: Dict[str, List[str]]) -> Set[str]:
    """Return seeds plus all steps reachable by following dependsOn backwards."""
    visited: Set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        for dep in deps_map.get(name, []):
            visit(dep)

    for name in seeds:
        visit(name)
    return visited


def _descendants_closure(seeds: Set[str], ordered_steps: Sequence[Dict]) -> Set[str]:
    """Return seeds plus any step whose ordering predecessors (dependsOn or after)
    intersect the selected set.  ordered_steps must be in topological order so that
    a single linear sweep suffices — no fixed-point loop needed."""
    selected = set(seeds)
    for step in ordered_steps:
        name = step['name']
        if step.get('runIfRequested') or name in selected:
            continue
        predecessors = step.get('dependsOn', []) + step.get('after', [])
        if any(p in selected for p in predecessors):
            selected.add(name)
    return selected


def select_steps(seeds: Set[str], ordered_steps: Sequence[Dict]) -> Set[str]:
    """Return the full set of step names that should run given a seed set.

    Algorithm:
      1. Forward pass  — expand seeds to include steps whose ordering predecessors
         (dependsOn *or* after) are selected.  after edges are used here
         so that cleanup steps follow the steps they clean up.
      2. Backward pass — for every selected step, pull in hard dependsOn ancestors.
         after edges are NOT followed here, so a cleanup step never forces its
         ordering predecessors to re-run (important for tactical retries).
    """
    run_if_requested = {s['name'] for s in ordered_steps if s.get('runIfRequested')}
    eligible = [s for s in ordered_steps if not s.get('runIfRequested')]

    # Hard-dep map: dependsOn only, excluding runIfRequested steps.
    deps_map: Dict[str, List[str]] = {
        s['name']: [d for d in s.get('dependsOn', []) if d not in run_if_requested] for s in eligible
    }

    # runIfRequested steps may appear as explicit seeds but are never pulled in
    # transitively, so handle them separately.
    seeded_run_if_requested = seeds & run_if_requested
    eligible_seeds = seeds - run_if_requested

    with_descendants = _descendants_closure(eligible_seeds, eligible)
    return _ancestors_closure(with_descendants, deps_map) | seeded_run_if_requested


def compute_requested_steps(
    config_str: str,
    changed_files: List[str],
    scope: str,
    cloud: Optional[str] = None,
) -> Set[str]:
    """Select which build steps should be requested, given the changed files in a PR.

    set((directly affected steps + their descendants) + alwaysRunSteps)
    """
    config = yaml.safe_load(config_str)
    repo_prefix: str = config.get('repoPrefix', '/repo')
    always_run_steps: Set[str] = set(config.get('alwaysRunSteps', []))

    if not changed_files:
        return always_run_steps

    steps = [s for s in config.get('steps', []) if _valid_step(s, scope, cloud)]

    descendants_map: Dict[str, List[str]] = {}
    for step in steps:
        for dep in step.get('dependsOn', []):
            descendants_map.setdefault(dep, []).append(step['name'])

    affected_steps = _find_affected_steps(steps, changed_files, repo_prefix)
    affected_and_descendants = _expand_to_descendants(affected_steps, descendants_map)
    requested_steps = affected_and_descendants | always_run_steps

    return requested_steps
