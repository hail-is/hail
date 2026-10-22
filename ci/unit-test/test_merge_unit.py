"""Unit tests for WatchedBranch.try_to_merge against a fake GitHub async merge API.

Each call to try_to_merge is one pass of CI's update loop.
"""

import http
import re
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import gidgethub
import gidgethub.sansio
import pytest

from ci.constants import GITHUB_STATUS_CONTEXT
from ci.github import PR, STACKED_PR, WIP, FQBranch, LastUpdate, Repo, RestRateLimit, WatchedBranch
from ci.utils import GithubStatus

TARGET_SHA = 'target-sha-0'

_MERGE_URL = re.compile(r'/repos/[^/]+/[^/]+/pulls/(\d+)/merge-async(?:/(.+))?$')


class FakeGH:
    """Fake of GitHub's async merge endpoints.

    Every merge PUT is accepted with `put_status` and a fresh uuid, or rejected with 409 if
    `put_conflict` is set. Polls report the status set in `merge_status` for that uuid, defaulting
    to pending.
    """

    def __init__(self, put_status: str = 'pending', put_conflict: bool = False):
        self.merge_requests: List[int] = []
        self.merge_status: Dict[str, str] = {}
        self.put_status = put_status
        self.put_conflict = put_conflict

    async def put(self, url: str, *, data: Any, extra_headers: Any = None) -> Any:  # pylint: disable=unused-argument
        pr_number = int(_parse(url)[0])
        self.merge_requests.append(pr_number)
        if self.put_conflict:
            raise gidgethub.HTTPException(http.HTTPStatus.CONFLICT, 'conflict')
        uuid = f'{pr_number}-{len(self.merge_requests)}'
        return {'status': self.put_status, 'details': {'uuid': uuid}}

    async def getitem(self, url: str, *, extra_headers: Any = None) -> Any:  # pylint: disable=unused-argument
        _, uuid = _parse(url)
        return {'status': self.merge_status.get(uuid, 'pending')}

    def last_uuid(self) -> str:
        return f'{self.merge_requests[-1]}-{len(self.merge_requests)}'


def _parse(url: str):
    m = _MERGE_URL.match(url)
    assert m is not None, f'unexpected GitHub API call: {url}'
    return m.group(1), m.group(2)


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


async def test_failed_async_merge_does_not_starve_lower_priority_prs():
    wb = make_branch()
    add_mergeable_pr(wb, 1)  # older, so higher priority
    add_mergeable_pr(wb, 2)
    gh = FakeGH()

    await wb.try_to_merge(gh)
    assert gh.merge_requests == [1]

    gh.merge_status[gh.last_uuid()] = 'failed'
    await wb.try_to_merge(gh)
    assert gh.merge_requests == [1, 2]


async def test_failed_merge_is_retried_with_backoff(monkeypatch):
    now = [1_000_000]
    monkeypatch.setattr('ci.github.time_msecs', lambda: now[0])
    wb = make_branch()
    add_mergeable_pr(wb, 1)
    gh = FakeGH()

    await wb.try_to_merge(gh)
    for backoff in [60, 600, 3600, 3600]:
        gh.merge_status[gh.last_uuid()] = 'failed'
        await wb.try_to_merge(gh)
        n_requests = len(gh.merge_requests)

        now[0] += backoff * 1000 - 1
        await wb.try_to_merge(gh)
        assert len(gh.merge_requests) == n_requests

        now[0] += 1
        await wb.try_to_merge(gh)
        assert len(gh.merge_requests) == n_requests + 1


async def test_new_commit_is_not_held_back_by_earlier_failure():
    wb = make_branch()
    pr = add_mergeable_pr(wb, 1)
    gh = FakeGH()

    await wb.try_to_merge(gh)
    gh.merge_status[gh.last_uuid()] = 'failed'
    await wb.try_to_merge(gh)
    assert gh.merge_requests == [1]

    pr.source_sha = 'source-sha-1-v2'
    await wb.try_to_merge(gh)
    assert gh.merge_requests == [1, 1]


async def test_enqueued_merge_is_waited_on_like_a_pending_one():
    wb = make_branch()
    add_mergeable_pr(wb, 1)
    add_mergeable_pr(wb, 2)
    gh = FakeGH(put_status='enqueued')

    await wb.try_to_merge(gh)
    gh.merge_status[gh.last_uuid()] = 'enqueued'
    await wb.try_to_merge(gh)
    assert gh.merge_requests == [1]


async def test_in_flight_merge_without_uuid_is_dropped_once_pr_is_no_longer_mergeable():
    wb = make_branch()
    pr = add_mergeable_pr(wb, 1)
    gh = FakeGH(put_conflict=True)

    await wb.try_to_merge(gh)
    assert gh.merge_requests == [1]

    pr.labels.add(WIP)
    await wb.try_to_merge(gh)
    assert gh.merge_requests == [1]
    assert wb.pending_merge is None


def test_mergeable_pr_has_no_blockers():
    pr = add_mergeable_pr(make_branch(), 1)
    assert not pr.merge_blockers()
    assert pr.is_mergeable()


@pytest.mark.parametrize(
    'review_state, blocker',
    [('pending', 'not approved'), ('changes_requested', 'changes requested'), (None, 'not approved')],
)
def test_review_blockers(review_state, blocker):
    pr = add_mergeable_pr(make_branch(), 1)
    pr.review_state = review_state
    assert pr.merge_blockers() == [blocker]
    assert not pr.is_mergeable()


def test_status_blockers():
    pr = add_mergeable_pr(make_branch(), 1)
    pr.last_known_github_status = {}
    assert pr.merge_blockers() == ['no status checks reported']

    pr.last_known_github_status = {
        'ci': GithubStatus.SUCCESS,
        'b-check': GithubStatus.FAILURE,
        'a-check': GithubStatus.FAILURE,
        'slow-check': GithubStatus.PENDING,
    }
    assert pr.merge_blockers() == ['checks failing: a-check, b-check', 'checks pending: slow-check']
    assert not pr.is_mergeable()


def test_build_blockers():
    wb = make_branch()
    pr = add_mergeable_pr(wb, 1)
    wb.sha = 'target-sha-1'
    assert pr.merge_blockers() == ['not built against current main']

    pr.batch = None
    assert pr.merge_blockers() == ['not built']
    assert not pr.is_mergeable()


def test_label_blockers():
    pr = add_mergeable_pr(make_branch(), 1)
    pr.labels |= {WIP, STACKED_PR, 'prio:high'}
    assert pr.merge_blockers() == ['label: WIP', 'label: stacked PR']
    assert not pr.is_mergeable()


def test_blockers_accumulate():
    pr = add_mergeable_pr(make_branch(), 1)
    pr.review_state = 'pending'
    pr.batch = None
    pr.labels.add(WIP)
    assert pr.merge_blockers() == ['not approved', 'not built', 'label: WIP']


@pytest.mark.parametrize(
    'frozen, deploy_batch, deploy_state, reason',
    [
        (False, None, None, None),
        (False, MagicMock(), None, 'deploying'),
        (False, MagicMock(), 'success', None),
        (False, MagicMock(), 'failure', None),
        (True, None, None, 'frozen'),
        (True, MagicMock(), None, 'frozen'),
    ],
)
def test_merge_paused_reason(frozen, deploy_batch, deploy_state, reason):
    wb = make_branch()
    wb.deploy_batch = deploy_batch
    wb.deploy_state = deploy_state
    assert wb.merge_paused_reason(frozen) == reason


async def test_update_only_merges_when_not_paused():
    wb = make_branch()
    wb.deploy_batch = MagicMock()

    async def heal_then_update(frozen: bool):
        wb.github_changed = wb.batch_changed = False
        wb.state_changed = True
        await wb._update(MagicMock(), MagicMock(), MagicMock(), frozen)

    with (
        patch.object(wb, '_heal', new=AsyncMock()),
        patch.object(wb, 'try_to_merge', new=AsyncMock()) as try_to_merge,
    ):
        await heal_then_update(frozen=False)
        try_to_merge.assert_not_awaited()

        wb.deploy_state = 'success'
        await heal_then_update(frozen=True)
        try_to_merge.assert_not_awaited()

        await heal_then_update(frozen=False)
        try_to_merge.assert_awaited_once()


# Bookkeeping: recorded for display only, so each test also checks the existing outcome is unchanged


async def test_update_records_timing_and_no_error(monkeypatch):
    monkeypatch.setattr('ci.github.time_msecs', iter([100, 250]).__next__)
    wb = make_branch()
    wb.github_changed = wb.batch_changed = wb.state_changed = False

    await wb._update(MagicMock(), MagicMock(), MagicMock(), frozen=False)

    assert wb.last_update_started_ms == 100
    assert wb.last_update == LastUpdate(100, 250, None)


async def test_update_records_error_and_still_raises():
    wb = make_branch()
    wb.batch_changed = wb.state_changed = False

    with (
        patch.object(wb, '_update_github', new=AsyncMock(side_effect=ValueError('boom'))),
        pytest.raises(ValueError, match='boom'),
    ):
        await wb._update(MagicMock(), MagicMock(), MagicMock(), frozen=False)

    assert wb.last_update is not None
    assert wb.last_update.error == 'ValueError: boom'
    assert not wb.updating


async def test_update_skipped_while_updating_records_nothing():
    wb = make_branch()
    wb.updating = True

    await wb._update(MagicMock(), MagicMock(), MagicMock(), frozen=False)

    assert wb.last_update_started_ms is None
    assert wb.last_update is None


async def test_rejected_merge_records_github_api_error():
    wb = make_branch()
    pr = add_mergeable_pr(wb, 1)
    gh = FakeGH()
    gh.put = AsyncMock(side_effect=gidgethub.HTTPException(http.HTTPStatus.FORBIDDEN, 'forbidden'))

    assert await pr.merge(gh) == ('failed', None)

    assert wb.last_github_api_error is not None
    assert wb.last_github_api_error.operation == 'merge 1'
    assert wb.last_github_api_error.status == 403


async def test_merge_already_in_flight_is_not_an_api_error():
    wb = make_branch()
    pr = add_mergeable_pr(wb, 1)

    assert await pr.merge(FakeGH(put_conflict=True)) == ('pending', None)

    assert wb.last_github_api_error is None


async def test_failed_pending_merge_poll_records_github_api_error():
    wb = make_branch()
    add_mergeable_pr(wb, 1)
    gh = FakeGH()
    await wb.try_to_merge(gh)
    gh.getitem = AsyncMock(side_effect=gidgethub.HTTPException(http.HTTPStatus.BAD_GATEWAY, 'bad gateway'))

    await wb.try_to_merge(gh)

    assert gh.merge_requests == [1]
    assert wb.pending_merge is not None
    assert wb.last_github_api_error is not None
    assert wb.last_github_api_error.operation == 'poll pending merge 1'
    assert wb.last_github_api_error.status == 502


async def test_update_github_snapshots_rest_rate_limit():
    wb = make_branch()
    gh = MagicMock()
    gh.getitem = AsyncMock(return_value={'object': {'sha': TARGET_SHA}})

    async def no_prs(_url):
        return
        yield

    gh.getiter = no_prs
    gh.rate_limit = gidgethub.sansio.RateLimit(limit=5000, remaining=4123, reset_epoch=0)

    await wb._update_github(gh)

    assert wb.rest_rate_limit == RestRateLimit(4123, 5000, '1970-01-01T00:00:00+00:00')
