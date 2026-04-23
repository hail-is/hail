import pytest

from ci.build_selection import (
    BuildSelectionResult,
    _expand_to_descendants,
    _file_matches_input,
    _find_seed_steps,
    _in_cloud,
    _in_scope,
    _repo_input_local_path,
    compute_requested_steps,
)


@pytest.mark.parametrize(
    'from_path, expected',
    [
        ('/repo', ''),
        ('/repo/ci', 'ci'),
        ('/repo/hail/python/hail', 'hail/python/hail'),
        ('/other', None),
        ('/repo-suffix', None),
        ('repo', None),
        ('', None),
    ],
)
def test_repo_input_local_path(from_path, expected):
    assert _repo_input_local_path(from_path) == expected


@pytest.mark.parametrize(
    'changed_file, local_path, expected',
    [
        # empty local_path matches everything
        ('anything.py', '', True),
        ('ci/foo.py', '', True),
        # exact file match
        ('ci/foo.py', 'ci/foo.py', True),
        # file under directory
        ('ci/foo.py', 'ci', True),
        ('ci/sub/foo.py', 'ci', True),
        # sibling directory should not match
        ('ci_extra/foo.py', 'ci', False),
        # plain prefix without path separator is not a match
        ('cibuild.py', 'ci', False),
        # different top-level path
        ('hail/foo.py', 'ci', False),
    ],
)
def test_file_matches_input(changed_file, local_path, expected):
    assert _file_matches_input(changed_file, local_path) == expected


@pytest.mark.parametrize(
    'scopes, scope, expected',
    [
        (None, 'test', True),
        (None, 'dev', True),
        (['test'], 'test', True),
        (['test', 'dev'], 'test', True),
        (['dev'], 'test', False),
        ([], 'test', False),
    ],
)
def test_in_scope(scopes, scope, expected):
    assert _in_scope(scopes, scope) == expected


@pytest.mark.parametrize(
    'clouds, cloud, expected',
    [
        (None, 'gcp', True),
        (None, None, True),
        (['gcp'], 'gcp', True),
        (['gcp', 'azure'], 'gcp', True),
        (['gcp'], 'azure', False),
        (['gcp'], None, True),
        ([], 'gcp', False),
    ],
)
def test_in_cloud(clouds, cloud, expected):
    assert _in_cloud(clouds, cloud) == expected


def _always_runnable(_name: str) -> bool:
    return True


def _never_runnable(_name: str) -> bool:
    return False


@pytest.mark.parametrize(
    'steps, changed_files, runnable, expected_seeds',
    [
        # step with matching /repo/ci input
        (
            [{'name': 'check_ci', 'inputs': [{'from': '/repo/ci', 'to': '/io/ci'}]}],
            ['ci/foo.py'],
            _always_runnable,
            {'check_ci'},
        ),
        # /repo matches any changed file
        (
            [{'name': 'check_all', 'inputs': [{'from': '/repo', 'to': '/io/repo'}]}],
            ['anything.py'],
            _always_runnable,
            {'check_all'},
        ),
        # changed file doesn't match the step's input path
        (
            [{'name': 'check_ci', 'inputs': [{'from': '/repo/ci', 'to': '/io/ci'}]}],
            ['hail/foo.py'],
            _always_runnable,
            set(),
        ),
        # step excluded because not runnable
        (
            [{'name': 'check_ci', 'inputs': [{'from': '/repo/ci', 'to': '/io/ci'}]}],
            ['ci/foo.py'],
            _never_runnable,
            set(),
        ),
        # step with no inputs is never a seed
        (
            [{'name': 'deploy_auth'}],
            ['ci/foo.py'],
            _always_runnable,
            set(),
        ),
        # non-/repo input (artifact from another step) is ignored
        (
            [{'name': 'test_ci', 'inputs': [{'from': '/io/wheel', 'to': '/wheel'}]}],
            ['ci/foo.py'],
            _always_runnable,
            set(),
        ),
        # multiple steps — only the matching one is a seed
        (
            [
                {'name': 'check_ci', 'inputs': [{'from': '/repo/ci', 'to': '/io/ci'}]},
                {'name': 'check_hail', 'inputs': [{'from': '/repo/hail', 'to': '/io/hail'}]},
            ],
            ['ci/foo.py'],
            _always_runnable,
            {'check_ci'},
        ),
    ],
)
def test_find_seed_steps(steps, changed_files, runnable, expected_seeds):
    assert _find_seed_steps(steps, changed_files, runnable) == expected_seeds


@pytest.mark.parametrize(
    'seeds, reverse, runnable, expected',
    [
        # seeds with no dependents
        ({'A'}, {}, _always_runnable, {'A'}),
        # single-level descendant
        ({'A'}, {'A': ['B']}, _always_runnable, {'A', 'B'}),
        # two-level chain
        ({'A'}, {'A': ['B'], 'B': ['C']}, _always_runnable, {'A', 'B', 'C'}),
        # descendant filtered out by runnable
        ({'A'}, {'A': ['B']}, _never_runnable, {'A'}),
        # diamond: A -> B, A -> C, B -> D, C -> D
        (
            {'A'},
            {'A': ['B', 'C'], 'B': ['D'], 'C': ['D']},
            _always_runnable,
            {'A', 'B', 'C', 'D'},
        ),
        # empty seeds
        (set(), {'A': ['B']}, _always_runnable, set()),
        # multiple seeds
        ({'A', 'X'}, {'A': ['B'], 'X': ['Y']}, _always_runnable, {'A', 'B', 'X', 'Y'}),
    ],
)
def test_expand_to_descendants(seeds, reverse, runnable, expected):
    assert _expand_to_descendants(seeds, reverse, runnable) == expected


_SIMPLE_CONFIG = """
steps:
  - name: merge_code
    kind: buildImage
    scopes: [test, dev]
  - name: check_ci
    kind: runImage
    scopes: [test, dev]
    inputs:
      - from: /repo/ci
        to: /io/ci
    dependsOn: [merge_code]
  - name: check_hail
    kind: runImage
    scopes: [test, dev]
    inputs:
      - from: /repo/hail
        to: /io/hail
    dependsOn: [merge_code]
  - name: deploy_ci
    kind: deploy
    scopes: [deploy]
    dependsOn: [check_ci]
"""

_CLOUD_CONFIG = """
steps:
  - name: merge_code
    kind: buildImage
    scopes: [test]
  - name: test_gcp
    kind: runImage
    scopes: [test]
    clouds: [gcp]
    inputs:
      - from: /repo/ci
        to: /io/ci
    dependsOn: [merge_code]
  - name: test_azure
    kind: runImage
    scopes: [test]
    clouds: [azure]
    inputs:
      - from: /repo/ci
        to: /io/ci
    dependsOn: [merge_code]
"""


@pytest.mark.parametrize(
    'config_str, changed_files, scope, cloud, expected_steps',
    [
        # empty changed_files -> nothing runs
        (_SIMPLE_CONFIG, [], 'test', None, []),
        # ci change seeds check_ci; deploy_ci excluded (wrong scope)
        (_SIMPLE_CONFIG, ['ci/foo.py'], 'test', None, ['check_ci']),
        # hail change seeds check_hail
        (_SIMPLE_CONFIG, ['hail/foo.py'], 'test', None, ['check_hail']),
        # unrelated change matches nothing
        (_SIMPLE_CONFIG, ['docs/index.md'], 'test', None, []),
        # deploy scope: check_ci is scoped to [test,dev] so it's not a seed,
        # and deploy_ci has no /repo inputs, so nothing is selected
        (_SIMPLE_CONFIG, ['ci/foo.py'], 'deploy', None, []),
        # gcp cloud filter — only test_gcp returned
        (_CLOUD_CONFIG, ['ci/foo.py'], 'test', 'gcp', ['test_gcp']),
        # azure cloud filter
        (_CLOUD_CONFIG, ['ci/foo.py'], 'test', 'azure', ['test_azure']),
        # no cloud filter -> both cloud-specific steps returned
        (_CLOUD_CONFIG, ['ci/foo.py'], 'test', None, ['test_azure', 'test_gcp']),
    ],
)
def test_compute_requested_steps(config_str, changed_files, scope, cloud, expected_steps):
    result = compute_requested_steps(config_str, changed_files, scope=scope, cloud=cloud)
    assert result == BuildSelectionResult(requested_steps=expected_steps)
