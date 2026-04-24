import pytest

from ci.build_selection import (
    _expand_to_descendants,
    _file_matches_input,
    _find_affected_steps,
    _in_cloud,
    _in_scope,
    _repo_input_local_path,
    _valid_step,
    compute_requested_steps,
)


@pytest.mark.parametrize(
    'from_path, repo_prefix, expected',
    [
        ('/repo', '/repo', ''),
        ('/repo/ci', '/repo', 'ci'),
        ('/repo/hail/python/hail', '/repo', 'hail/python/hail'),
        ('/other', '/repo', None),
        ('/repo-suffix', '/repo', None),
        ('repo', '/repo', None),
        ('', '/repo', None),
        # custom prefix
        ('/src', '/src', ''),
        ('/src/ci', '/src', 'ci'),
        ('/repo/ci', '/src', None),
        ('/src-suffix', '/src', None),
    ],
)
def test_repo_input_local_path(from_path, repo_prefix, expected):
    assert _repo_input_local_path(from_path, repo_prefix) == expected


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


@pytest.mark.parametrize(
    'step, scope, cloud, expected',
    [
        ({'scopes': None, 'clouds': None}, 'test', 'gcp', True),
        ({'scopes': ['test'], 'clouds': None}, 'test', 'gcp', True),
        ({'scopes': ['dev'], 'clouds': None}, 'test', 'gcp', False),
        ({'scopes': None, 'clouds': ['gcp']}, 'test', 'gcp', True),
        ({'scopes': None, 'clouds': ['gcp']}, 'test', 'azure', False),
        ({'scopes': ['deploy'], 'clouds': ['gcp']}, 'test', 'gcp', False),
        ({}, 'test', None, True),
    ],
)
def test_valid_step(step, scope, cloud, expected):
    assert _valid_step(step, scope, cloud) == expected


@pytest.mark.parametrize(
    'steps, changed_files, repo_prefix, expected_affected_steps',
    [
        # step with matching /repo/ci input
        (
            [{'name': 'check_ci', 'inputs': [{'from': '/repo/ci', 'to': '/io/ci'}]}],
            ['ci/foo.py'],
            '/repo',
            {'check_ci'},
        ),
        # /repo matches any changed file
        (
            [{'name': 'check_all', 'inputs': [{'from': '/repo', 'to': '/io/repo'}]}],
            ['anything.py'],
            '/repo',
            {'check_all'},
        ),
        # changed file doesn't match the step's input path
        (
            [{'name': 'check_ci', 'inputs': [{'from': '/repo/ci', 'to': '/io/ci'}]}],
            ['hail/foo.py'],
            '/repo',
            set(),
        ),
        # step with no inputs is never affected
        (
            [{'name': 'deploy_auth'}],
            ['ci/foo.py'],
            '/repo',
            set(),
        ),
        # non-/repo input (artifact from another step) is ignored
        (
            [{'name': 'test_ci', 'inputs': [{'from': '/io/wheel', 'to': '/wheel'}]}],
            ['ci/foo.py'],
            '/repo',
            set(),
        ),
        # multiple steps — only the matching one is affected
        (
            [
                {'name': 'check_ci', 'inputs': [{'from': '/repo/ci', 'to': '/io/ci'}]},
                {'name': 'check_hail', 'inputs': [{'from': '/repo/hail', 'to': '/io/hail'}]},
            ],
            ['ci/foo.py'],
            '/repo',
            {'check_ci'},
        ),
        # custom repo_prefix: /src/ci matches when prefix is /src
        (
            [{'name': 'check_ci', 'inputs': [{'from': '/src/ci', 'to': '/io/ci'}]}],
            ['ci/foo.py'],
            '/src',
            {'check_ci'},
        ),
        # custom repo_prefix: /repo/ci does NOT match when prefix is /src
        (
            [{'name': 'check_ci', 'inputs': [{'from': '/repo/ci', 'to': '/io/ci'}]}],
            ['ci/foo.py'],
            '/src',
            set(),
        ),
    ],
)
def test_find_affected_steps(steps, changed_files, repo_prefix, expected_affected_steps):
    assert _find_affected_steps(steps, changed_files, repo_prefix) == expected_affected_steps


@pytest.mark.parametrize(
    'affected_steps, descendants, expected',
    [
        # no dependents
        ({'A'}, {}, {'A'}),
        # single-level descendant
        ({'A'}, {'A': ['B']}, {'A', 'B'}),
        # two-level chain
        ({'A'}, {'A': ['B'], 'B': ['C']}, {'A', 'B', 'C'}),
        # diamond: A -> B, A -> C, B -> D, C -> D
        ({'A'}, {'A': ['B', 'C'], 'B': ['D'], 'C': ['D']}, {'A', 'B', 'C', 'D'}),
        # no affected steps
        (set(), {'A': ['B']}, set()),
        # multiple affected steps
        ({'A', 'X'}, {'A': ['B'], 'X': ['Y']}, {'A', 'B', 'X', 'Y'}),
    ],
)
def test_expand_to_descendants(affected_steps, descendants, expected):
    assert _expand_to_descendants(affected_steps, descendants) == expected


_SIMPLE_CONFIG = """
alwaysRunSteps:
  - merge_code

steps:
  - name: merge_code
    kind: runImage
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
alwaysRunSteps:
  - merge_code

steps:
  - name: merge_code
    kind: runImage
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


_CUSTOM_PREFIX_CONFIG = """
repoPrefix: /src

alwaysRunSteps:
  - merge_code

steps:
  - name: merge_code
    kind: runImage
  - name: check_ci
    kind: runImage
    scopes: [test]
    inputs:
      - from: /src/ci
        to: /io/ci
    dependsOn: [merge_code]
"""


@pytest.mark.parametrize(
    'config_str, changed_files, scope, cloud, expected_steps',
    [
        # empty changed_files -> nothing runs (not even merge_code)
        (_SIMPLE_CONFIG, [], 'test', None, []),
        # ci change affects check_ci; merge_code always included
        (_SIMPLE_CONFIG, ['ci/foo.py'], 'test', None, ['check_ci', 'merge_code']),
        # hail change affects check_hail; merge_code always included
        (_SIMPLE_CONFIG, ['hail/foo.py'], 'test', None, ['check_hail', 'merge_code']),
        # README.md under ci/ affects check_ci just like any other file there
        (_SIMPLE_CONFIG, ['ci/README.md'], 'test', None, ['check_ci', 'merge_code']),
        # multiple ci/ files both affect check_ci
        (_SIMPLE_CONFIG, ['ci/README.md', 'ci/foo.py'], 'test', None, ['check_ci', 'merge_code']),
        # deploy scope: check_ci scoped to [test,dev] is not affected; merge_code always included
        (_SIMPLE_CONFIG, ['ci/foo.py'], 'deploy', None, ['merge_code']),
        # gcp cloud filter — test_gcp + merge_code
        (_CLOUD_CONFIG, ['ci/foo.py'], 'test', 'gcp', ['merge_code', 'test_gcp']),
        # azure cloud filter
        (_CLOUD_CONFIG, ['ci/foo.py'], 'test', 'azure', ['merge_code', 'test_azure']),
        # no cloud filter -> both cloud-specific steps + merge_code
        (_CLOUD_CONFIG, ['ci/foo.py'], 'test', None, ['merge_code', 'test_azure', 'test_gcp']),
        # custom repoPrefix in config: ci change affects check_ci
        (_CUSTOM_PREFIX_CONFIG, ['ci/foo.py'], 'test', None, ['check_ci', 'merge_code']),
    ],
)
def test_compute_requested_steps(config_str, changed_files, scope, cloud, expected_steps):
    result = compute_requested_steps(config_str, changed_files, scope=scope, cloud=cloud)
    assert result == expected_steps
