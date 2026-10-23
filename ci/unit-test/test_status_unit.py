"""Unit tests for the CI status endpoint payloads and health checks."""

from unittest.mock import AsyncMock, MagicMock, patch

from ci.constants import GITHUB_STATUS_CONTEXT
from ci.github import (
    DO_NOT_TEST,
    MAX_CONCURRENT_PR_BATCHES,
    PR,
    FailedMerge,
    FQBranch,
    GitHubApiError,
    GraphQLRateLimit,
    LastUpdate,
    MergeFailureBatch,
    PendingMerge,
    Repo,
    RestRateLimit,
    WatchedBranch,
)
from ci.status import (
    LOOP_STALE_MS,
    UPDATE_STALE_MS,
    WEBHOOK_GAP_MS,
    developer_branch_status,
    health_problems,
    latest_github_api_error,
    operator_branch_status,
)
from ci.utils import GithubStatus

NOW = 100 * 24 * 60 * 60 * 1000
TARGET_SHA = 'target-sha-0'


def make_branch() -> WatchedBranch:
    wb = WatchedBranch(0, FQBranch(Repo('hail-is', 'hail'), 'main'), deployable=False, mergeable=True, developers=[])
    wb.sha = TARGET_SHA
    return wb


def add_mergeable_pr(wb: WatchedBranch, number: int) -> PR:
    with patch('ci.github.TRACKED_PRS'):
        pr = PR(
            number=number,
            title=f'PR {number}',
            body='',
            source_branch=MagicMock(),
            source_sha=f'source-sha-{number}',
            target_branch=wb,
            author='testuser',
            assignees=set(),
            reviewers=set(),
            labels=set(),
            developers=[],
        )
    pr.review_state = 'approved'
    pr.build_state = 'success'
    pr.last_known_github_status = {GITHUB_STATUS_CONTEXT: GithubStatus.SUCCESS}
    pr.batch = MagicMock(attributes={'target_sha': wb.sha})
    wb.prs[number] = pr
    return pr


def authorizing_db(authorized: bool = True):
    db = MagicMock()
    db.execute_and_fetchone = AsyncMock(return_value={'sha': 'x'} if authorized else None)
    return db


def add_unbuilt_pr(wb, number):
    pr = add_mergeable_pr(wb, number)
    pr.batch = None
    pr.build_state = None
    return pr


async def test_build_slots_and_queue():
    wb = make_branch()
    running = add_mergeable_pr(wb, 1)
    running.build_state = None
    add_mergeable_pr(wb, 2)  # finished build
    add_unbuilt_pr(wb, 3).pending_build_reason = 'new commit 1a2b3c4d'
    add_unbuilt_pr(wb, 4).labels = {DO_NOT_TEST}
    wb.merge_candidate = running

    status = await developer_branch_status(wb, authorizing_db(), frozen=False)

    assert status['build_slots'] == {
        'used': 1,
        'max': MAX_CONCURRENT_PR_BATCHES,
        'running': [1],
        'on_deck': 1,
        'queued': [{'pr': 3, 'reason': 'new commit 1a2b3c4d'}],
    }
    assert status['merge_candidate'] == 1
    assert status['n_prs'] == 4


async def test_unauthorized_prs_are_not_queued_or_in_merge_queue():
    wb = make_branch()
    add_unbuilt_pr(wb, 1)

    status = await developer_branch_status(wb, authorizing_db(authorized=False), frozen=False)

    assert status['build_slots']['queued'] == []
    assert status['merge_queue'] == []


async def test_merge_queue_has_blockers_and_backoff():
    wb = make_branch()
    add_mergeable_pr(wb, 1)
    out_of_date = add_mergeable_pr(wb, 2)
    out_of_date.batch = MagicMock(attributes={'target_sha': 'old-target'})
    add_mergeable_pr(wb, 3).review_state = 'pending'
    add_mergeable_pr(wb, 4).last_known_github_status = {'ci-test': GithubStatus.FAILURE}
    wb.failed_merges[wb._merge_attempt(out_of_date)] = FailedMerge(2, 123)

    status = await developer_branch_status(wb, authorizing_db(), frozen=False)

    assert status['merge_queue'] == [
        {'pr': 1, 'blockers': [], 'failures': 0, 'backoff_until_ms': None},
        {'pr': 2, 'blockers': ['not built against current main'], 'failures': 2, 'backoff_until_ms': 123},
    ]


async def test_merge_pause_pending_merge_and_last_update():
    wb = make_branch()
    wb.pending_merge = PendingMerge(5, 'uuid', 'src', 'tgt', requested_at_ms=42)
    wb.last_update = LastUpdate(1_000, 9_200, error='boom')

    status = await developer_branch_status(wb, authorizing_db(), frozen=True)

    assert status['merge_paused_reason'] == 'frozen'
    assert status['pending_merge'] == {'pr': 5, 'requested_at_ms': 42}
    # the error is operator-only
    assert status['last_update'] == {'started_ms': 1_000, 'finished_ms': 9_200, 'duration_s': 8.2}


def test_operator_branch_status():
    wb = make_branch()
    wb.updating = True
    wb.last_update_started_ms = 7
    wb.github_changed, wb.batch_changed, wb.state_changed = True, False, True
    wb.last_update = LastUpdate(1, 2, error='ValueError: x')
    wb.rest_rate_limit = RestRateLimit(4000, 5000, '2026-01-01T00:00:00+00:00')
    wb.graphql_rate_limit = GraphQLRateLimit(3, 4800, 5000, '2026-01-01T00:00:00Z')
    wb.deploy_batch = MergeFailureBatch(RuntimeError('checkout failed'), {})

    status = operator_branch_status(wb)

    assert status['update_started_ms'] == 7
    assert status['dirty'] == ['github', 'state']
    assert status['last_update_error'] == 'ValueError: x'
    assert status['rate_limit'] == {
        'rest': {'remaining': 4000, 'limit': 5000, 'reset_at': '2026-01-01T00:00:00+00:00'},
        'graphql': {'last_update_cost': 3, 'remaining': 4800, 'limit': 5000, 'reset_at': '2026-01-01T00:00:00Z'},
    }
    assert status['deploy_exception'] == 'RuntimeError: checkout failed'


def test_update_start_hidden_when_not_updating():
    wb = make_branch()
    wb.last_update_started_ms = 7
    assert operator_branch_status(wb)['update_started_ms'] is None


def test_latest_github_api_error():
    wb1, wb2 = make_branch(), make_branch()
    assert latest_github_api_error([wb1, wb2]) is None
    wb1.last_github_api_error = GitHubApiError(5, 'merge 1', 405, 'nope')
    wb2.last_github_api_error = GitHubApiError(9, 'post status 2', 403, 'forbidden')
    assert latest_github_api_error([wb1, wb2]) == {
        'branch': 'main',
        'at_ms': 9,
        'operation': 'post status 2',
        'status': 403,
        'message': 'forbidden',
    }


def problems(wbs, **kwargs):
    args = {
        'now_ms': NOW,
        'started_at_ms': NOW,
        'last_webhook_ms': {},
        'loop_last_success_ms': {},
        'expected_loops': [],
        **kwargs,
    }
    return health_problems(wbs, **args)


def healthy_branch():
    wb = make_branch()
    wb.last_update = LastUpdate(NOW - 2000, NOW - 1000, None)
    return wb


def test_healthy():
    assert not problems(
        [healthy_branch()],
        last_webhook_ms={'push': NOW},
        expected_loops=['loop'],
        loop_last_success_ms={'loop': NOW},
    )


def test_fresh_start_is_healthy():
    assert not problems([make_branch()], expected_loops=['loop'])


def test_stale_update():
    wb = healthy_branch()
    wb.last_update = LastUpdate(0, NOW - UPDATE_STALE_MS - 1, None)
    assert problems([wb]) == ['main: no update finished in 15 minutes']


def test_never_finished_update_is_stale_after_startup():
    assert problems([make_branch()], started_at_ms=NOW - UPDATE_STALE_MS - 1) == [
        'main: no update finished in 15 minutes'
    ]


def test_failed_update():
    wb = healthy_branch()
    wb.last_update = LastUpdate(NOW - 2000, NOW - 1000, 'boom')
    assert problems([wb]) == ['main: last update failed']


def test_github_api_error_is_unhealthy_only_while_recent():
    wb = healthy_branch()
    wb.last_github_api_error = GitHubApiError(NOW - 1000, 'merge 1', 403, 'forbidden')
    assert problems([wb]) == ['main: GitHub API error during merge 1']
    wb.last_github_api_error = GitHubApiError(0, 'merge 1', 403, 'forbidden')
    assert not problems([wb])


def test_low_rate_limits():
    wb = healthy_branch()
    wb.rest_rate_limit = RestRateLimit(499, 5000, '')
    wb.graphql_rate_limit = GraphQLRateLimit(1, 500, 5000, '')
    assert problems([wb]) == ['main: GitHub REST rate limit low (499/5000)']


def test_webhook_gap():
    assert problems([], last_webhook_ms={'push': NOW - WEBHOOK_GAP_MS - 1}) == [
        'no GitHub webhook received in 24 hours'
    ]
    assert problems([], started_at_ms=NOW - WEBHOOK_GAP_MS - 1) == ['no GitHub webhook received in 24 hours']


def test_stale_background_loop():
    assert problems([], expected_loops=['a', 'b'], loop_last_success_ms={'a': NOW - LOOP_STALE_MS - 1, 'b': NOW}) == [
        'background loop a has not succeeded in 5 minutes'
    ]
