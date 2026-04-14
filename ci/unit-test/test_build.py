from ci.build_selection import compute_requested_steps

MINIMAL_CONFIG = """
unwatchedPaths:
  - dev-docs/
  - '*.md'
alwaysRunSteps:
  - merge_code
steps:
  - kind: runImage
    name: test_batch
    watchedPaths:
      - batch/
      - hail/python/hailtop/
  - kind: runImage
    name: test_auth
    watchedPaths:
      - auth/
  - kind: runImage
    name: merge_code
"""


def test_all_unwatched_files_returns_always_run_only():
    result = compute_requested_steps(MINIMAL_CONFIG, ['dev-docs/foo.md'])
    assert result == ['merge_code']


def test_unwatched_glob_pattern_returns_always_run_only():
    result = compute_requested_steps(MINIMAL_CONFIG, ['README.md'])
    assert result == ['merge_code']


def test_matched_file_returns_specific_step():
    result = compute_requested_steps(MINIMAL_CONFIG, ['batch/server.py'])
    assert result == ['merge_code', 'test_batch']


def test_matched_file_returns_multiple_steps():
    # hailtop is in both test_batch and any step watching it; here only test_batch watches hailtop
    result = compute_requested_steps(MINIMAL_CONFIG, ['hail/python/hailtop/batch/client.py'])
    assert result == ['merge_code', 'test_batch']


def test_unknown_file_triggers_all_watched_steps():
    result = compute_requested_steps(MINIMAL_CONFIG, ['some/new/thing.py'])
    assert result == ['merge_code', 'test_auth', 'test_batch']


def test_mixed_matched_and_unknown_triggers_all():
    # one unknown file → fallback to all watched steps
    result = compute_requested_steps(MINIMAL_CONFIG, ['batch/x.py', 'some/new/thing.py'])
    assert result == ['merge_code', 'test_auth', 'test_batch']


def test_empty_changed_files_returns_empty():
    result = compute_requested_steps(MINIMAL_CONFIG, [])
    assert result == []


def test_directory_pattern_matches_nested_file():
    config = """
steps:
  - kind: runImage
    name: test_hail
    watchedPaths:
      - hail/src/
"""
    result = compute_requested_steps(config, ['hail/src/main/scala/is/hail/Foo.scala'])
    assert result == ['test_hail']


def test_directory_pattern_does_not_match_sibling():
    config = """
steps:
  - kind: runImage
    name: test_hail
    watchedPaths:
      - hail/src/
"""
    # hail/python/ does not start with hail/src/
    result = compute_requested_steps(config, ['hail/python/hail/foo.py'])
    # no watchedPaths match → fallback to all watched steps
    assert result == ['test_hail']


def test_multiple_matched_files_each_different_step():
    result = compute_requested_steps(MINIMAL_CONFIG, ['batch/x.py', 'auth/y.py'])
    assert result == ['merge_code', 'test_auth', 'test_batch']


def test_mixed_unwatched_and_matched():
    # docs-only file contributes nothing beyond always-run; batch file adds test_batch
    result = compute_requested_steps(MINIMAL_CONFIG, ['dev-docs/guide.md', 'batch/handler.py'])
    assert result == ['merge_code', 'test_batch']


def test_always_run_step_always_selected():
    # merge_code is in alwaysRunSteps — it always appears when changed_files is non-empty
    result = compute_requested_steps(MINIMAL_CONFIG, ['batch/server.py'])
    assert 'merge_code' in result


def test_always_run_step_not_in_empty_result():
    # alwaysRunSteps do not appear when changed_files is empty
    result = compute_requested_steps(MINIMAL_CONFIG, [])
    assert 'merge_code' not in result


def test_result_is_sorted():
    result = compute_requested_steps(MINIMAL_CONFIG, ['batch/x.py', 'auth/y.py'])
    assert result == sorted(result)
