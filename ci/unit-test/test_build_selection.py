import pathlib

import pytest
import yaml

from ci.build_selection import (
    _expand_to_descendants,
    _file_matches_input,
    _find_affected_steps,
    _repo_input_local_path,
    _valid_step,
    compute_requested_steps,
    select_steps,
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
    'step, scope, cloud, expected',
    [
        ({'scopes': None, 'clouds': None}, 'test', 'gcp', True),
        ({'scopes': None, 'clouds': None}, 'test', None, True),  # no cloud filter
        ({'scopes': ['test'], 'clouds': None}, 'test', 'gcp', True),
        ({'scopes': ['dev'], 'clouds': None}, 'test', 'gcp', False),
        ({'scopes': None, 'clouds': ['gcp']}, 'test', 'gcp', True),
        ({'scopes': None, 'clouds': ['gcp']}, 'test', 'azure', False),
        ({'scopes': None, 'clouds': ['gcp']}, 'test', None, True),  # unfiltered cloud matches any
        ({'scopes': ['deploy'], 'clouds': ['gcp']}, 'test', 'gcp', False),
        ({'runIfRequested': True}, 'test', 'gcp', False),
        ({'runIfRequested': True}, 'test', None, False),
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


_RUN_IF_REQUESTED_CONFIG = """
alwaysRunSteps:
  - merge_code

steps:
  - name: merge_code
    kind: runImage
  - name: check_ci
    kind: runImage
    scopes: [test]
    inputs:
      - from: /repo/ci
        to: /io/ci
    dependsOn: [merge_code]
  - name: create_initial_user
    kind: runImage
    runIfRequested: true
    dependsOn: [merge_code, check_ci]
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
        # empty changed_files -> nothing runs except alwaysRunSteps (ie merge_code)
        (_SIMPLE_CONFIG, [], 'test', None, {'merge_code'}),
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
        # runIfRequested step is never auto-selected as a descendant
        (_RUN_IF_REQUESTED_CONFIG, ['ci/foo.py'], 'test', None, ['check_ci', 'merge_code']),
        # custom repoPrefix in config: ci change affects check_ci
        (_CUSTOM_PREFIX_CONFIG, ['ci/foo.py'], 'test', None, ['check_ci', 'merge_code']),
    ],
)
def test_compute_requested_steps(config_str, changed_files, scope, cloud, expected_steps):
    result = compute_requested_steps(config_str, changed_files, scope=scope, cloud=cloud)
    assert result == set(expected_steps)


# ---------------------------------------------------------------------------
# select_steps: after graph traversal
# ---------------------------------------------------------------------------


# Test 2: after edges are NOT followed in the backward (ancestors) pass.
#
#   A  <--after--  B  <--dependsOn--  C
#
# Seed = {C}.  Expected: {C, B}.  A must NOT be pulled in.
def test_select_steps_after_not_followed_backwards():
    steps = [
        {'name': 'A'},
        {'name': 'B', 'after': ['A']},
        {'name': 'C', 'dependsOn': ['B']},
    ]
    assert select_steps({'C'}, steps) == {'C', 'B'}


# Test 3: after edges ARE followed in the forward (descendants) pass.
#
#   A  <--after--  B
#
# Seed = {A}.  Expected: {A, B}.  B should be pulled in as a cleanup follower.
def test_select_steps_after_followed_forwards():
    steps = [
        {'name': 'A'},
        {'name': 'B', 'after': ['A']},
    ]
    assert select_steps({'A'}, steps) == {'A', 'B'}


# Combined test: the realistic cancel_all / test_batch_invariants scenario.
#
#   infra
#     ^-- dependsOn -- cancel_all (after: test_batch, test_ci)
#                          ^-- dependsOn -- test_batch_invariants
#   test_batch
#   test_ci
#
# Scenario A (forward): seed = {test_batch}.
#   cancel_all follows test_batch via after → included.
#   test_batch_invariants follows cancel_all via dependsOn → included.
#   infra is pulled in as an ancestor of cancel_all.
#   test_ci is NOT included (not a seed, not downstream of test_batch).
#
# Scenario B (backward / tactical retry): seed = {test_batch_invariants}.
#   cancel_all is a hard ancestor → included.
#   infra is a hard ancestor of cancel_all → included.
#   test_batch and test_ci are NOT included — they're only in cancel_all's
#   after, which the backward pass does not follow.
def test_select_steps_forwards_not_backwards():
    steps = [
        {'name': 'infra'},
        {'name': 'test_batch'},
        {'name': 'test_ci'},
        {'name': 'cancel_all', 'dependsOn': ['infra'], 'after': ['test_batch', 'test_ci']},
        {'name': 'test_batch_invariants', 'dependsOn': ['cancel_all', 'infra']},
    ]

    # Scenario A: running test_batch brings in cancel_all and test_batch_invariants.
    assert select_steps({'test_batch'}, steps) == {
        'test_batch',
        'cancel_all',
        'infra',
        'test_batch_invariants',
    }

    # Scenario B: tactical retry of test_batch_invariants does NOT re-add test_batch or test_ci.
    assert select_steps({'test_batch_invariants'}, steps) == {
        'test_batch_invariants',
        'cancel_all',
        'infra',
    }


# ---------------------------------------------------------------------------
# Unit tests against the real build.yaml
# ---------------------------------------------------------------------------

_BUILD_YAML = (pathlib.Path(__file__).parents[2] / 'build.yaml').read_text()
_BUILD_STEPS = [s for s in yaml.safe_load(_BUILD_YAML)['steps'] if _valid_step(s, 'test', 'gcp')]


def test_batch_file_change_selects_batch_steps():
    # compute_requested_steps returns seeds based on file inputs — it does not
    # follow the full graph.  test_batch_invariants is reachable only after the
    # subsequent select_steps expansion, so it need not appear here.
    result = compute_requested_steps(_BUILD_YAML, ['batch/batch/driver/main.py'], scope='test', cloud='gcp')
    assert 'test_batch' in result
    assert 'cancel_all_running_test_batches' in result


@pytest.mark.xfail(
    reason='dependsOn temporarily duplicated onto after-edges in build.yaml until the deployed CI '
    'understands `after` (see commit "restore depends... to unblock build"); unxfail once removed',
    strict=False,
)
def test_tactical_retry_of_test_hailtop_python_does_not_pull_in_test_batch_or_test_ci():
    # Simulates a tactical retry seeded with only test_hailtop_python.
    # The cleanup chain (cancel_all → test_batch_invariants → delete_gcp) is pulled
    # in via forward after-edges, but test_batch and test_ci are not — they share
    # the same cancel_all sink but have no dependsOn relationship with test_hailtop_python.
    result = select_steps({'test_hailtop_python'}, _BUILD_STEPS)

    # cleanup chain pulled in via after
    assert 'cancel_all_running_test_batches' in result
    assert 'test_batch_invariants' in result
    assert 'delete_gcp_batch_instances' in result

    # unrelated test suites are not
    assert 'test_batch' not in result
    assert 'test_ci' not in result


@pytest.mark.xfail(
    reason='dependsOn temporarily duplicated onto after-edges in build.yaml until the deployed CI '
    'understands `after` (see commit "restore depends... to unblock build"); unxfail once removed',
    strict=False,
)
def test_tactical_retry_of_test_batch_does_not_pull_in_test_ci():
    # compute_requested_steps correctly includes test_ci via the hard dependsOn
    # chain batch_image → deploy_batch → deploy_ci → test_ci, so we can't assert
    # its absence there.  But a tactical retry seeded with only test_batch (a
    # batch-specific failure) must not pull test_ci in — they're unrelated.
    result = select_steps({'test_batch'}, _BUILD_STEPS)

    # downstream cleanup steps are pulled in via `after`
    assert 'cancel_all_running_test_batches' in result
    assert 'delete_test_billing_projects' in result

    # unrelated test suites are not
    assert 'test_ci' not in result


@pytest.mark.xfail(
    reason='dependsOn temporarily duplicated onto after-edges in build.yaml until the deployed CI '
    'understands `after` (see commit "restore depends... to unblock build"); unxfail once removed',
    strict=False,
)
def test_tactical_retry_of_test_batch_invariants_does_not_pull_in_test_suites():
    # Simulates a tactical retry where only test_batch_invariants needs to re-run.
    # The backward (ancestors) pass must not follow `after` edges, so the test
    # suites that cancel_all_running_test_batches and delete_gcp_batch_instances
    # merely order themselves after are not pulled back in.
    result = select_steps({'test_batch_invariants'}, _BUILD_STEPS)

    # hard upstream prereqs are pulled in
    assert 'cancel_all_running_test_batches' in result
    assert 'deploy_batch' in result
    assert 'default_ns' in result

    # downstream cleanup is pulled in via `after`
    assert 'delete_gcp_batch_instances' in result

    # test suites are not — they're only in `after` edges, not hard deps
    assert 'test_batch' not in result
    assert 'test_ci' not in result
    assert 'test_hailtop_python' not in result
    assert 'test_hail_python_service_backend_gcp' not in result
