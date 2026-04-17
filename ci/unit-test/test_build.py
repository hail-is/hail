from ci.build_selection import BuildSelectionResult, compute_requested_steps, expand_build_steps

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
    assert result.requested_steps == ['merge_code']
    assert result.full_retest_triggers == []


def test_unwatched_glob_pattern_returns_always_run_only():
    result = compute_requested_steps(MINIMAL_CONFIG, ['README.md'])
    assert result.requested_steps == ['merge_code']
    assert result.full_retest_triggers == []


def test_unwatched_glob_matches_nested_md():
    result = compute_requested_steps(MINIMAL_CONFIG, ['nested/README.md'])
    assert result.requested_steps == ['merge_code']
    assert result.full_retest_triggers == []


def test_unwatched_glob_matches_uppercase_extension():
    result = compute_requested_steps(MINIMAL_CONFIG, ['README.MD'])
    assert result.requested_steps == ['merge_code']
    assert result.full_retest_triggers == []


def test_matched_file_returns_specific_step():
    result = compute_requested_steps(MINIMAL_CONFIG, ['batch/server.py'])
    assert result.requested_steps == ['merge_code', 'test_batch']
    assert result.full_retest_triggers == []


def test_matched_file_returns_multiple_steps():
    # hailtop is in both test_batch and any step watching it; here only test_batch watches hailtop
    result = compute_requested_steps(MINIMAL_CONFIG, ['hail/python/hailtop/batch/client.py'])
    assert result.requested_steps == ['merge_code', 'test_batch']
    assert result.full_retest_triggers == []


def test_unknown_file_triggers_all_watched_steps():
    result = compute_requested_steps(MINIMAL_CONFIG, ['some/new/thing.py'])
    assert result.requested_steps == ['merge_code', 'test_auth', 'test_batch']
    assert result.full_retest_triggers == ['some/new/thing.py']


def test_mixed_matched_and_unknown_triggers_all():
    # one unknown file → fallback to all watched steps
    result = compute_requested_steps(MINIMAL_CONFIG, ['batch/x.py', 'some/new/thing.py'])
    assert result.requested_steps == ['merge_code', 'test_auth', 'test_batch']
    assert result.full_retest_triggers == ['some/new/thing.py']


def test_empty_changed_files_returns_empty():
    result = compute_requested_steps(MINIMAL_CONFIG, [])
    assert result == BuildSelectionResult()


def test_directory_pattern_matches_nested_file():
    config = """
steps:
  - kind: runImage
    name: test_hail
    watchedPaths:
      - hail/src/
"""
    result = compute_requested_steps(config, ['hail/src/main/scala/is/hail/Foo.scala'])
    assert result.requested_steps == ['test_hail']
    assert result.full_retest_triggers == []


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
    assert result.requested_steps == ['test_hail']
    assert result.full_retest_triggers == ['hail/python/hail/foo.py']


def test_multiple_matched_files_each_different_step():
    result = compute_requested_steps(MINIMAL_CONFIG, ['batch/x.py', 'auth/y.py'])
    assert result.requested_steps == ['merge_code', 'test_auth', 'test_batch']
    assert result.full_retest_triggers == []


def test_mixed_unwatched_and_matched():
    # docs-only file contributes nothing beyond always-run; batch file adds test_batch
    result = compute_requested_steps(MINIMAL_CONFIG, ['dev-docs/guide.md', 'batch/handler.py'])
    assert result.requested_steps == ['merge_code', 'test_batch']
    assert result.full_retest_triggers == []


def test_always_run_step_always_selected():
    # merge_code is in alwaysRunSteps — it always appears when changed_files is non-empty
    result = compute_requested_steps(MINIMAL_CONFIG, ['batch/server.py'])
    assert 'merge_code' in result.requested_steps


def test_always_run_step_not_in_empty_result():
    # alwaysRunSteps do not appear when changed_files is empty
    result = compute_requested_steps(MINIMAL_CONFIG, [])
    assert 'merge_code' not in result.requested_steps


def test_no_duplicates_when_multiple_files_match_same_step():
    # batch/x.py and hail/python/hailtop/foo.py both match test_batch — it should appear only once
    result = compute_requested_steps(MINIMAL_CONFIG, ['batch/x.py', 'hail/python/hailtop/foo.py'])
    assert result.requested_steps.count('test_batch') == 1


def test_result_is_sorted():
    result = compute_requested_steps(MINIMAL_CONFIG, ['batch/x.py', 'auth/y.py'])
    assert result.requested_steps == sorted(result.requested_steps)


def test_full_retest_triggers_are_sorted():
    result = compute_requested_steps(MINIMAL_CONFIG, ['z/unknown.py', 'a/unknown.py'])
    assert result.full_retest_triggers == sorted(result.full_retest_triggers)


def test_multiple_unknown_files_all_recorded_in_triggers():
    result = compute_requested_steps(MINIMAL_CONFIG, ['x/foo.py', 'y/bar.py', 'z/baz.py'])
    assert result.full_retest_triggers == ['x/foo.py', 'y/bar.py', 'z/baz.py']


def test_unwatched_files_not_recorded_in_triggers():
    # Files matching unwatchedPaths are skipped entirely — not counted as fallthrough triggers
    result = compute_requested_steps(MINIMAL_CONFIG, ['dev-docs/guide.md', 'README.md'])
    assert result.full_retest_triggers == []


def test_mixed_unwatched_and_unknown_only_unknown_in_triggers():
    # Unwatched file is silently skipped; unknown file causes full-retest
    result = compute_requested_steps(MINIMAL_CONFIG, ['dev-docs/guide.md', 'new/thing.py'])
    assert result.full_retest_triggers == ['new/thing.py']


def test_build_selection_result_defaults():
    r = BuildSelectionResult()
    assert r.requested_steps == []
    assert r.full_retest_triggers == []


def test_build_selection_result_equality():
    assert BuildSelectionResult(['a', 'b'], ['x']) == BuildSelectionResult(['a', 'b'], ['x'])
    assert BuildSelectionResult(['a'], []) != BuildSelectionResult(['b'], [])


# ---------------------------------------------------------------------------
# expand_build_steps tests
# ---------------------------------------------------------------------------

EXPAND_CONFIG = """
steps:
  - name: setup
  - name: test_a
    watchedPaths: [a/]
    dependsOn: [setup]
  - name: test_b
    watchedPaths: [b/]
    dependsOn: [setup]
  - name: cleanup
    cleanupFor: setup
    dependsOn: [test_a, test_b]
"""


def test_expand_includes_transitive_deps():
    result = expand_build_steps(EXPAND_CONFIG, ['test_a'])
    assert 'setup' in result
    assert 'test_a' in result


def test_expand_no_cleanup_by_default():
    result = expand_build_steps(EXPAND_CONFIG, ['test_a'])
    assert 'cleanup' not in result


def test_expand_cleanup_included_when_requested():
    result = expand_build_steps(EXPAND_CONFIG, ['test_a'], include_cleanups=True)
    assert 'cleanup' in result


def test_expand_cleanup_not_included_when_setup_not_reachable():
    config = """
steps:
  - name: other
    watchedPaths: [other/]
  - name: setup
  - name: cleanup
    cleanupFor: setup
    dependsOn: [other]
"""
    result = expand_build_steps(config, ['other'], include_cleanups=True)
    assert 'cleanup' not in result


def test_expand_chained_cleanups_both_included():
    config = """
steps:
  - name: setup
  - name: test_a
    watchedPaths: [a/]
    dependsOn: [setup]
  - name: cleanup1
    cleanupFor: setup
    dependsOn: [test_a]
  - name: cleanup2
    cleanupFor: setup
    dependsOn: [cleanup1]
"""
    result = expand_build_steps(config, ['test_a'], include_cleanups=True)
    assert 'cleanup1' in result
    assert 'cleanup2' in result


def test_expand_excluded_steps_not_included():
    result = expand_build_steps(EXPAND_CONFIG, ['test_a', 'test_b'], excluded_step_names=['test_b'])
    assert 'test_b' not in result
    assert 'test_a' in result


def test_expand_run_if_requested_dep_not_pulled_in():
    config = """
steps:
  - name: infra
    runIfRequested: true
  - name: test_a
    dependsOn: [infra]
"""
    result = expand_build_steps(config, ['test_a'])
    assert 'test_a' in result
    assert 'infra' not in result


def test_expand_cloud_filter_excludes_wrong_cloud():
    config = """
steps:
  - name: gcp_only
    clouds: [gcp]
  - name: test_a
    dependsOn: [gcp_only]
"""
    result = expand_build_steps(config, ['test_a'], cloud='azure')
    assert 'gcp_only' not in result


def test_expand_result_is_sorted():
    result = expand_build_steps(EXPAND_CONFIG, ['test_b', 'test_a'])
    assert result == sorted(result)


def test_expand_empty_requested_returns_empty():
    result = expand_build_steps(EXPAND_CONFIG, [])
    assert result == []
