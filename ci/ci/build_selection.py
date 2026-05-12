from typing import Dict, List, Optional, Set

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
