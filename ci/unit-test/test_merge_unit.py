"""Unit tests for WatchedBranch.try_to_merge against a fake GitHub async merge API.

Each call to try_to_merge is one pass of CI's update loop.
"""

import http
import re
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import gidgethub

from ci.constants import GITHUB_STATUS_CONTEXT
from ci.github import PR, WIP, FQBranch, Repo, WatchedBranch
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
