"""Unit tests for fetching PR data from GitHub GraphQL, and applying it to PR.

These mock out the GitHub API client so no network access is needed.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ci.events import CIEvent
from ci.github import (
    PR,
    FQBranch,
    GitHubGraphQLError,
    GitHubPr,
    GraphQLRateLimit,
    Repo,
    Review,
    ReviewRequest,
    UnreadablePrInfo,
    WatchedBranch,
    _fetch_branch_prs,
    _fetch_pr_github_data,
    _GraphQLCost,
    _list_open_pr_numbers,
    _PrGithubData,
)
from ci.utils import GithubStatus

# ── Mock helpers ─────────────────────────────────────────────────────────────


class MockGH:
    """Records calls and returns pre-canned responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0
        self.queries: list = []

    async def post(self, url: str, *, data: Any) -> Any:  # pylint: disable=unused-argument
        assert self.call_count < len(self._responses), (
            f'Unexpected API call #{self.call_count + 1} (only {len(self._responses)} response(s) queued)'
        )
        response = self._responses[self.call_count]
        self.call_count += 1
        self.queries.append(data['query'])
        return response


def graphql_response(pr_data_by_number: dict, *, rate_limit=None) -> dict:
    # A first page carries the PR's own fields, number included
    data: dict = {
        "repository": {
            f"pr_{n}": {**data, "number": n} if data is not None and "title" in data else data
            for n, data in pr_data_by_number.items()
        }
    }
    if rate_limit is not None:
        data["rateLimit"] = rate_limit
    return {"data": data}


def rate_limit(cost, remaining, reset_at='2026-10-22T12:00:00Z'):
    return {"cost": cost, "remaining": remaining, "limit": 5000, "resetAt": reset_at}


def pr_fields(
    *,
    title='Test PR',
    body='',
    state='OPEN',
    base='main',
    head_sha='head-sha',
    head_ref='feature',
    head_repo=None,
    author=None,
    assignees=(),
    labels=(),
):
    """The PR's own fields, which the query only asks for on a PR's first page."""
    return {
        "title": title,
        "body": body,
        "state": state,
        "baseRefName": base,
        "headRefName": head_ref,
        "headRefOid": head_sha,
        "headRepository": head_repo if head_repo is not None else {"name": "hail", "owner": {"login": "someone"}},
        "author": author if author is not None else {"login": "testuser"},
        "assignees": {"nodes": [{"login": a} for a in assignees]},
        "labels": {"nodes": [{"name": label} for label in labels]},
    }


def review_fields(review_requests=(), reviews=(), requested_events=()):
    """The review fragments the query only asks for on a PR's first page."""
    return {
        "reviewRequests": {"nodes": list(review_requests)},
        "latestOpinionatedReviews": {"nodes": list(reviews)},
        "timelineItems": {"nodes": list(requested_events)},
    }


def user(login):
    return {"__typename": "User", "login": login}


def team(slug):
    return {"__typename": "Team", "slug": slug}


def pr_response(
    review_decision, check_nodes, *, has_next_page=False, end_cursor=None, first_page=True, pr=None, **review
):
    """Minimal pullRequest fragment matching the shape _fetch_pr_github_data expects."""
    return {
        **({"reviewDecision": review_decision, **(pr or pr_fields()), **review_fields(**review)} if first_page else {}),
        "commits": {
            "nodes": [
                {
                    "commit": {
                        "statusCheckRollup": {
                            "contexts": {
                                "nodes": check_nodes,
                                "pageInfo": {"hasNextPage": has_next_page, "endCursor": end_cursor},
                            }
                        }
                    }
                }
            ]
        },
    }


def pr_response_no_rollup(review_decision):
    """PR with no CI runs yet — statusCheckRollup is null."""
    return {
        "reviewDecision": review_decision,
        **pr_fields(),
        **review_fields(),
        "commits": {"nodes": [{"commit": {"statusCheckRollup": None}}]},
    }


def check_run(name, conclusion, *, is_required=True):
    return {"__typename": "CheckRun", "name": name, "conclusion": conclusion, "isRequired": is_required}


def status_context(context, state, *, is_required=True):
    return {"__typename": "StatusContext", "context": context, "state": state, "isRequired": is_required}


def gh_pr(number=123, **fields):
    return GitHubPr.from_graphql({"number": number, **pr_fields(**fields), **review_fields()})


def gh_data(review_decision, check_nodes, review_requests=(), reviews=(), *, number=123):
    return _PrGithubData(gh_pr(number), review_decision, check_nodes, list(review_requests), list(reviews))


def make_pr(number=123):
    """Construct a minimal PR with a MagicMock target_branch, bypassing prometheus metrics."""
    tb = MagicMock()
    tb.state_changed = False
    tb.batch_changed = False
    with patch('ci.github.TRACKED_PRS'):
        pr = PR(
            number=number,
            title='Test PR',
            body='',
            source_branch=MagicMock(),
            source_sha='deadbeef',
            target_branch=tb,
            author='testuser',
            assignees=set(),
            reviewers=set(),
            labels=set(),
            developers=[],
        )
    tb.state_changed = False  # reset; constructor sets it
    return pr


# ── _fetch_pr_github_data ─────────────────────────────────────────────────────


async def test_single_pr_no_pagination():
    nodes = [check_run('ci-test', 'SUCCESS'), status_context('lint', 'SUCCESS', is_required=False)]
    gh = MockGH([graphql_response({123: pr_response('APPROVED', nodes)})])

    result = (await _fetch_pr_github_data(gh, 'hail-is', 'hail', [123])).prs

    assert gh.call_count == 1
    review_decision, check_nodes = result[123].review_decision, result[123].check_nodes
    assert review_decision == 'APPROVED'
    assert check_nodes == nodes


async def test_multiple_prs_in_single_request():
    nodes_a = [check_run('ci-test', 'SUCCESS')]
    nodes_b = [check_run('ci-test', 'FAILURE')]
    gh = MockGH([
        graphql_response({
            1: pr_response('APPROVED', nodes_a),
            2: pr_response('REVIEW_REQUIRED', nodes_b),
        })
    ])

    result = (await _fetch_pr_github_data(gh, 'hail-is', 'hail', [1, 2])).prs

    assert gh.call_count == 1
    assert result[1] == gh_data('APPROVED', nodes_a, number=1)
    assert result[2] == gh_data('REVIEW_REQUIRED', nodes_b, number=2)


async def test_pagination_fires_multiple_requests_and_accumulates_nodes():
    page1 = [check_run('check-a', 'SUCCESS')]
    page2 = [check_run('check-b', 'FAILURE')]
    page3 = [check_run('check-c', 'SUCCESS')]

    gh = MockGH([
        graphql_response({42: pr_response('APPROVED', page1, has_next_page=True, end_cursor='cursor-1')}),
        # reviewDecision is ignored on subsequent pages (already captured from page 1)
        graphql_response({42: pr_response(None, page2, has_next_page=True, end_cursor='cursor-2', first_page=False)}),
        graphql_response({42: pr_response(None, page3, has_next_page=False, first_page=False)}),
    ])

    result = (await _fetch_pr_github_data(gh, 'hail-is', 'hail', [42])).prs

    assert gh.call_count == 3
    review_decision, check_nodes = result[42].review_decision, result[42].check_nodes
    assert review_decision == 'APPROVED'
    assert check_nodes == page1 + page2 + page3


async def test_pagination_only_re_requests_prs_that_need_it():
    """PRs that finish on page 1 are dropped from subsequent requests; only lagging PRs continue."""
    page1_pr1 = [check_run('check-a', 'SUCCESS')]
    page2_pr1 = [check_run('check-b', 'SUCCESS')]
    page3_pr1 = [check_run('check-c', 'SUCCESS')]
    nodes_pr2 = [check_run('check-a', 'FAILURE')]

    # Request 1: both PRs. PR 2 finishes; PR 1 needs 2 more pages.
    # Requests 2 and 3: only PR 1 — mock would error if PR 2 appeared again.
    gh = MockGH([
        graphql_response({
            1: pr_response('APPROVED', page1_pr1, has_next_page=True, end_cursor='cur-1'),
            2: pr_response('REVIEW_REQUIRED', nodes_pr2, has_next_page=False),
        }),
        graphql_response({1: pr_response(None, page2_pr1, has_next_page=True, end_cursor='cur-2', first_page=False)}),
        graphql_response({1: pr_response(None, page3_pr1, has_next_page=False, first_page=False)}),
    ])

    result = (await _fetch_pr_github_data(gh, 'hail-is', 'hail', [1, 2])).prs

    assert gh.call_count == 3
    assert result[1] == gh_data('APPROVED', page1_pr1 + page2_pr1 + page3_pr1, number=1)
    assert result[2] == gh_data('REVIEW_REQUIRED', nodes_pr2, number=2)


async def test_null_review_decision_becomes_api_none():
    gh = MockGH([graphql_response({5: pr_response(None, [])})])

    result = (await _fetch_pr_github_data(gh, 'hail-is', 'hail', [5])).prs

    assert result[5].review_decision == 'API_NONE'


async def test_null_status_rollup_returns_empty_nodes():
    gh = MockGH([graphql_response({7: pr_response_no_rollup('REVIEW_REQUIRED')})])

    result = (await _fetch_pr_github_data(gh, 'hail-is', 'hail', [7])).prs

    review_decision, check_nodes = result[7].review_decision, result[7].check_nodes
    assert review_decision == 'REVIEW_REQUIRED'
    assert check_nodes == []


async def test_empty_pr_list_makes_no_request():
    gh = MockGH([])

    result = (await _fetch_pr_github_data(gh, 'hail-is', 'hail', [])).prs

    assert result == {}
    assert gh.call_count == 0


async def test_review_data_parsed_with_latest_request_time():
    gh = MockGH([
        graphql_response({
            9: pr_response(
                'REVIEW_REQUIRED',
                [],
                review_requests=[
                    {"asCodeOwner": True, "requestedReviewer": team('appsec')},
                    {"asCodeOwner": False, "requestedReviewer": user('carol')},
                    {"asCodeOwner": False, "requestedReviewer": None},  # deleted account
                ],
                reviews=[
                    {"author": {"login": "dave"}, "state": "APPROVED"},
                    {"author": None, "state": "APPROVED"},  # deleted account
                ],
                requested_events=[
                    {"createdAt": "2026-10-01T00:00:00Z", "requestedReviewer": user('carol')},
                    {"createdAt": "2026-10-05T00:00:00Z", "requestedReviewer": user('carol')},  # re-requested
                    {"createdAt": "2026-10-02T00:00:00Z", "requestedReviewer": team('appsec')},
                    {"createdAt": "2026-10-03T00:00:00Z", "requestedReviewer": user('appsec')},  # not the team
                    {},  # some other timeline item type
                ],
            )
        })
    ])

    result = (await _fetch_pr_github_data(gh, 'hail-is', 'hail', [9])).prs

    assert result[9].review_requests == [
        ReviewRequest('appsec', True, True, '2026-10-02T00:00:00Z'),
        ReviewRequest('carol', False, False, '2026-10-05T00:00:00Z'),
    ]
    assert result[9].reviews == [Review('dave', 'APPROVED')]


async def test_hidden_codeowner_request_kept_without_reviewer():
    # Without the org Members permission, GitHub nulls out team reviewers and reports an error alongside the data
    response = graphql_response({
        9: pr_response(
            'REVIEW_REQUIRED',
            [],
            review_requests=[
                {"asCodeOwner": True, "requestedReviewer": None},
                {"asCodeOwner": False, "requestedReviewer": None},
            ],
            requested_events=[
                {"createdAt": "2026-10-01T00:00:00Z", "requestedReviewer": None},
                {"createdAt": "2026-10-04T00:00:00Z", "requestedReviewer": None},
                {"createdAt": "2026-10-06T00:00:00Z", "requestedReviewer": user('carol')},
            ],
        )
    })
    response["errors"] = [{"message": "Resource not accessible by personal access token"}] * 2
    gh = MockGH([response])

    with patch('ci.github.log') as log:
        result = (await _fetch_pr_github_data(gh, 'hail-is', 'hail', [9])).prs

    assert result[9].review_requests == [ReviewRequest(None, False, True, '2026-10-04T00:00:00Z')]
    log.warning.assert_called_once()
    assert 'Resource not accessible by personal access token' in log.warning.call_args.args[0]


async def test_review_request_without_event_has_no_time():
    gh = MockGH([
        graphql_response({
            9: pr_response(
                'REVIEW_REQUIRED', [], review_requests=[{"asCodeOwner": False, "requestedReviewer": user('x')}]
            )
        })
    ])

    result = (await _fetch_pr_github_data(gh, 'hail-is', 'hail', [9])).prs

    assert result[9].review_requests == [ReviewRequest('x', False, False, None)]


async def test_review_fields_only_requested_on_first_page():
    gh = MockGH([
        graphql_response({42: pr_response('APPROVED', [], has_next_page=True, end_cursor='cursor-1')}),
        graphql_response({42: pr_response(None, [], first_page=False)}),
    ])

    await _fetch_pr_github_data(gh, 'hail-is', 'hail', [42])

    assert 'reviewRequests' in gh.queries[0]
    assert 'reviewRequests' not in gh.queries[1]
    assert all('rateLimit' in q for q in gh.queries)


async def test_rate_limit_sums_cost_and_keeps_last_remaining():
    gh = MockGH([
        graphql_response(
            {42: pr_response('APPROVED', [], has_next_page=True, end_cursor='c')}, rate_limit=rate_limit(2, 4990)
        ),
        graphql_response(
            {42: pr_response(None, [], first_page=False)}, rate_limit=rate_limit(1, 4989, '2026-10-22T13:00:00Z')
        ),
    ])

    cost = _GraphQLCost()
    await _fetch_pr_github_data(gh, 'hail-is', 'hail', [42], cost)

    assert cost.rate_limit() == GraphQLRateLimit(3, 4989, 5000, '2026-10-22T13:00:00Z')


async def test_rate_limit_is_none_without_requests():
    cost = _GraphQLCost()
    await _fetch_pr_github_data(MockGH([]), 'hail-is', 'hail', [], cost)
    assert cost.rate_limit() is None


# ── PR._apply_github_data ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    'review_decision,expected_state',
    [
        ('APPROVED', 'approved'),
        ('CHANGES_REQUESTED', 'changes_requested'),
        ('REVIEW_REQUIRED', 'pending'),
        ('API_NONE', 'pending'),
    ],
)
def test_review_decision_mapping(review_decision, expected_state):
    pr = make_pr()
    with patch('ci.github.TRACKED_PRS'):
        pr._apply_github_data(gh_data(review_decision, []))
    assert pr.review_state == expected_state
    assert pr.target_branch.state_changed is True


def test_review_state_unchanged_does_not_set_state_changed():
    pr = make_pr()
    with patch('ci.github.TRACKED_PRS'):
        pr._apply_github_data(gh_data('APPROVED', []))
        pr.target_branch.state_changed = False
        pr._apply_github_data(gh_data('APPROVED', []))
    assert pr.target_branch.state_changed is False


def test_required_check_run_appears_in_status():
    pr = make_pr()
    with patch('ci.github.TRACKED_PRS'):
        pr._apply_github_data(gh_data('APPROVED', [check_run('ci-test', 'SUCCESS', is_required=True)]))
    assert pr.last_known_github_status == {'ci-test': GithubStatus.SUCCESS}


def test_required_status_context_appears_in_status():
    pr = make_pr()
    with patch('ci.github.TRACKED_PRS'):
        pr._apply_github_data(gh_data('APPROVED', [status_context('ci-status', 'FAILURE', is_required=True)]))
    assert pr.last_known_github_status == {'ci-status': GithubStatus.FAILURE}


def test_non_required_checks_are_ignored():
    pr = make_pr()
    nodes = [
        check_run('required', 'SUCCESS', is_required=True),
        check_run('optional', 'FAILURE', is_required=False),
        status_context('optional-status', 'FAILURE', is_required=False),
    ]
    with patch('ci.github.TRACKED_PRS'):
        pr._apply_github_data(gh_data('APPROVED', nodes))
    assert list(pr.last_known_github_status.keys()) == ['required']


def test_github_status_unchanged_does_not_set_state_changed():
    pr = make_pr()
    nodes = [check_run('ci-test', 'SUCCESS', is_required=True)]
    with patch('ci.github.TRACKED_PRS'):
        pr._apply_github_data(gh_data('APPROVED', nodes))
        pr.target_branch.state_changed = False
        pr._apply_github_data(gh_data('APPROVED', nodes))
    assert pr.target_branch.state_changed is False


def test_check_run_failure_status():
    pr = make_pr()
    with patch('ci.github.TRACKED_PRS'):
        pr._apply_github_data(gh_data('APPROVED', [check_run('ci-test', 'FAILURE', is_required=True)]))
    assert pr.last_known_github_status['ci-test'] == GithubStatus.FAILURE


def test_check_run_pending_conclusion():
    pr = make_pr()
    with patch('ci.github.TRACKED_PRS'):
        pr._apply_github_data(gh_data('APPROVED', [check_run('ci-test', 'ACTION_REQUIRED', is_required=True)]))
    assert pr.last_known_github_status['ci-test'] == GithubStatus.PENDING


def test_review_data_stored_without_setting_state_changed():
    pr = make_pr()
    requests = [ReviewRequest('appsec', True, True, '2026-10-02T00:00:00Z')]
    reviews = [Review('dave', 'APPROVED')]
    with patch('ci.github.TRACKED_PRS'):
        pr._apply_github_data(gh_data('APPROVED', []))
        pr.target_branch.state_changed = False
        pr._apply_github_data(gh_data('APPROVED', [], requests, reviews))
    assert pr.review_requests == requests
    assert pr.reviews == reviews
    assert pr.target_branch.state_changed is False


# ── GitHubPr / PR fields ──────────────────────────────────────────────────────


def test_pr_fields_match_the_rest_listing():
    """The PR state the REST listing gave (title, body, user, assignees, requested_reviewers, labels, head) now comes
    from GraphQL. Teams were never in REST's requested_reviewers, and REST reported a deleted author as 'ghost'."""
    node = {
        "number": 7,
        **pr_fields(
            title='Fix it',
            body='#assign services',
            head_sha='abc123',
            head_ref='fix-it',
            head_repo={"name": "hail-fork", "owner": {"login": "alice"}},
            author={"login": "alice"},
            assignees=['bob'],
            labels=['bug', 'prio:high'],
        ),
        **review_fields(
            review_requests=[
                {"asCodeOwner": False, "requestedReviewer": user('carol')},
                {"asCodeOwner": False, "requestedReviewer": {"__typename": "Bot", "login": "a-bot"}},
                {"asCodeOwner": True, "requestedReviewer": team('services')},
                {"asCodeOwner": True, "requestedReviewer": None},
            ]
        ),
    }
    tb = MagicMock(developers=[])
    with patch('ci.github.TRACKED_PRS'):
        pr = PR.from_github(GitHubPr.from_graphql(node), tb)

    assert pr.number == 7
    assert pr.title == 'Fix it'
    assert pr.body == '#assign services'
    assert pr.author == 'alice'
    assert pr.assignees == {'bob'}
    assert pr.reviewers == {'carol', 'a-bot'}
    assert pr.labels == {'bug', 'prio:high'}
    assert pr.source_branch == FQBranch(Repo('alice', 'hail-fork'), 'fix-it')
    assert pr.source_sha == 'abc123'


def test_deleted_author_is_ghost():
    node = {"number": 1, **pr_fields(), **review_fields()}
    node["author"] = None
    assert GitHubPr.from_graphql(node).author == 'ghost'


def test_update_from_github_new_commit_resets_build():
    pr = make_pr()
    pr.sha = 'merged-sha'
    pr.batch = MagicMock()
    with patch('ci.github.TRACKED_PRS'):
        pr.update_from_github(gh_pr(head_sha='new-sha', labels=['bug']))
    assert pr.source_sha == 'new-sha'
    assert pr.sha is None
    assert pr.batch is None
    assert pr.labels == {'bug'}
    assert pr.target_branch.state_changed is True
    assert pr.target_branch.batch_changed is True


def test_update_from_github_unchanged_does_not_set_state_changed():
    pr = make_pr()
    pr.target_branch.batch_changed = False  # the constructor sets it
    with patch('ci.github.TRACKED_PRS'):
        pr.update_from_github(gh_pr(head_sha='deadbeef'))
    assert pr.target_branch.state_changed is False
    assert pr.target_branch.batch_changed is False


# ── Unreadable PRs ────────────────────────────────────────────────────────────


async def test_pr_with_deleted_head_repository_is_skipped():
    gh = MockGH([
        graphql_response({
            1: pr_response('APPROVED', [], pr={**pr_fields(head_sha='sha-1'), "headRepository": None}),
            2: pr_response('APPROVED', []),
        })
    ])

    with patch('ci.github.log'):
        fetched = await _fetch_pr_github_data(gh, 'hail-is', 'hail', [1, 2])

    assert list(fetched.prs) == [2]
    assert fetched.unreadable[1].source_sha == 'sha-1'
    assert fetched.unreadable[1].message.startswith('PR skipped: head repository')


def null_pr_error(number, error_type, message='boom'):
    """GitHub's error for a nulled pr_N alias, in the shape a live query for a missing PR returns."""
    return {"type": error_type, "path": ["repository", f"pr_{number}"], "message": message}


@pytest.mark.parametrize('error_type', ['NOT_FOUND', 'FORBIDDEN'])
async def test_pr_github_says_is_gone_or_hidden_is_skipped_with_its_message(error_type):
    response = graphql_response({1: None, 2: pr_response('APPROVED', [])})
    response["errors"] = [null_pr_error(1, error_type, 'Could not resolve to a PullRequest with the number of 1.')]
    gh = MockGH([response])

    with patch('ci.github.log'):
        fetched = await _fetch_pr_github_data(gh, 'hail-is', 'hail', [1, 2])

    assert list(fetched.prs) == [2]
    assert fetched.unreadable[1] == UnreadablePrInfo(
        None, 'PR skipped: Could not resolve to a PullRequest with the number of 1.'
    )


@pytest.mark.parametrize(
    'errors',
    [
        [],  # nulled with no error
        [null_pr_error(1, 'SERVICE_UNAVAILABLE')],
        [{"message": "timeout", "path": ["repository", "pr_1", "commits"]}],  # error on a field below the PR
    ],
)
async def test_pr_nulled_for_any_other_reason_is_unexplained(errors):
    response = graphql_response({1: None, 2: pr_response('APPROVED', [])})
    response["errors"] = errors
    gh = MockGH([response])

    with patch('ci.github.log'):
        fetched = await _fetch_pr_github_data(gh, 'hail-is', 'hail', [1, 2])

    assert list(fetched.prs) == [2]
    assert fetched.unreadable == {}
    assert list(fetched.unexplained) == [1]
    assert fetched.unexplained[1].message.startswith('PR kept at its previous state: ')


async def test_pr_nulled_on_a_later_page_is_unexplained():
    gh = MockGH([
        graphql_response({1: pr_response('APPROVED', [check_run('a', 'SUCCESS')], has_next_page=True, end_cursor='c')}),
        graphql_response({1: None}),
    ])

    with patch('ci.github.log'):
        fetched = await _fetch_pr_github_data(gh, 'hail-is', 'hail', [1])

    assert fetched.prs == {}
    assert list(fetched.unexplained) == [1]


async def test_no_data_raises_with_githubs_message():
    message = 'Something went wrong while executing your query. This may be the result of a timeout.'
    gh = MockGH([{"data": None, "errors": [{"message": message}]}])

    with pytest.raises(GitHubGraphQLError, match='may be the result of a timeout'):
        await _fetch_pr_github_data(gh, 'hail-is', 'hail', [1])


async def test_unreadable_pr_is_not_paginated():
    gh = MockGH([
        graphql_response({
            1: pr_response(
                'APPROVED', [], has_next_page=True, end_cursor='c', pr={**pr_fields(), "headRepository": None}
            ),
        })
    ])

    with patch('ci.github.log'):
        fetched = await _fetch_pr_github_data(gh, 'hail-is', 'hail', [1])

    assert gh.call_count == 1
    assert fetched.prs == {}


def make_watched_branch() -> WatchedBranch:
    return WatchedBranch(0, FQBranch(Repo('hail-is', 'hail'), 'main'), deployable=False, mergeable=True, developers=[])


async def test_pr_read_error_recorded_once_per_pr_and_sha():
    wb = make_watched_branch()
    with patch('ci.github.record_event', new_callable=AsyncMock) as record:
        await wb._record_pr_read_errors(MagicMock(), {1: UnreadablePrInfo('sha-a', 'gone')})
        await wb._record_pr_read_errors(MagicMock(), {1: UnreadablePrInfo('sha-a', 'gone')})
        assert record.call_count == 1
        assert record.call_args.args[1] == CIEvent.UPDATE_FAILED
        assert record.call_args.kwargs == {
            'target_branch': 'hail-is/hail:main',
            'pr_number': 1,
            'source_sha': 'sha-a',
            'message': 'gone',
        }

        await wb._record_pr_read_errors(MagicMock(), {1: UnreadablePrInfo('sha-b', 'gone')})  # new commit
        assert record.call_count == 2

        await wb._record_pr_read_errors(MagicMock(), {})  # readable again
        await wb._record_pr_read_errors(MagicMock(), {1: UnreadablePrInfo('sha-b', 'gone')})
        assert record.call_count == 3


# ── Listing (phase 1) and the whole branch fetch ──────────────────────────────


def list_response(numbers, *, end_cursor=None, rate_limit=None):
    data: dict = {
        "repository": {
            "pullRequests": {
                "nodes": [{"number": n} for n in numbers],
                "pageInfo": {"hasNextPage": end_cursor is not None, "endCursor": end_cursor},
            }
        }
    }
    if rate_limit is not None:
        data["rateLimit"] = rate_limit
    return {"data": data}


class MockGHWithVariables(MockGH):
    def __init__(self, responses):
        super().__init__(responses)
        self.variables: list = []

    async def post(self, url: str, *, data: Any) -> Any:
        self.variables.append(data.get('variables'))
        return await super().post(url, data=data)


async def test_list_open_pr_numbers_paginates():
    gh = MockGHWithVariables([list_response([3, 2], end_cursor='c1'), list_response([1])])

    numbers = await _list_open_pr_numbers(gh, 'hail-is', 'hail', 'main', _GraphQLCost())

    assert numbers == [3, 2, 1]
    assert gh.variables[0] == {"owner": "hail-is", "name": "hail", "branch": "main"}
    assert gh.variables[1] == {"owner": "hail-is", "name": "hail", "branch": "main", "after": "c1"}
    assert 'states: OPEN' in gh.queries[0]
    assert '$after' not in gh.queries[0]
    assert 'after: $after' in gh.queries[1]


async def test_fetch_branch_prs_drops_prs_merged_or_retargeted_since_the_listing():
    gh = MockGH([
        list_response([4, 3, 2, 1]),
        graphql_response({
            4: pr_response('APPROVED', []),
            3: pr_response('APPROVED', [], pr=pr_fields(state='MERGED')),
            2: pr_response('APPROVED', [], pr=pr_fields(base='release')),
            1: pr_response('APPROVED', []),
        }),
    ])

    fetched = await _fetch_branch_prs(gh, 'hail-is', 'hail', 'main')

    assert list(fetched.prs) == [4, 1]
    assert fetched.unreadable == {}


async def test_fetch_branch_prs_keeps_listing_order_and_sums_cost_over_both_phases():
    gh = MockGH([
        list_response([2, 1], rate_limit=rate_limit(1, 4999)),
        graphql_response(
            {2: pr_response('APPROVED', []), 1: pr_response('APPROVED', [])},
            rate_limit=rate_limit(3, 4996, '2026-10-22T13:00:00Z'),
        ),
    ])

    fetched = await _fetch_branch_prs(gh, 'hail-is', 'hail', 'main')

    assert list(fetched.prs) == [2, 1]
    assert fetched.rate_limit == GraphQLRateLimit(4, 4996, 5000, '2026-10-22T13:00:00Z')


async def test_fetch_branch_prs_with_no_open_prs_makes_one_request():
    gh = MockGH([list_response([])])

    fetched = await _fetch_branch_prs(gh, 'hail-is', 'hail', 'main')

    assert fetched.prs == {}
    assert gh.call_count == 1


async def test_update_keeps_a_pr_github_nulled_and_updates_the_rest():
    wb = make_watched_branch()
    with patch('ci.github.TRACKED_PRS'):
        for number in (1, 2):
            wb.prs[number] = PR.from_github(gh_pr(number, title='old', head_sha=f'old-{number}'), wb)
    kept = wb.prs[1]
    kept.review_state = 'approved'

    response = graphql_response({
        2: pr_response('APPROVED', [], pr=pr_fields(title='new', head_sha='old-2')),
        1: None,
    })
    response["errors"] = [null_pr_error(1, 'SERVICE_UNAVAILABLE')]
    replies = iter([list_response([2, 1]), response])

    gh = MagicMock(rate_limit=None)
    gh.getitem = AsyncMock(return_value={'object': {'sha': 'target-sha'}})

    async def post(_url, *, data):  # pylint: disable=unused-argument
        return next(replies)

    gh.post = post

    with (
        patch('ci.github.TRACKED_PRS'),
        patch('ci.github.log'),
        patch('ci.github.record_event', new_callable=AsyncMock) as record,
    ):
        await wb._update_github(MagicMock(), gh)

    assert list(wb.prs) == [2, 1]
    assert wb.prs[1] is kept
    assert (kept.title, kept.source_sha, kept.review_state) == ('old', 'old-1', 'approved')
    assert wb.prs[2].title == 'new'
    assert record.call_args.args[1] == CIEvent.UPDATE_FAILED
    assert record.call_args.kwargs['pr_number'] == 1
