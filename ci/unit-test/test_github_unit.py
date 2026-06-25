"""Unit tests for _fetch_pr_github_data and PR._apply_github_data.

These mock out the GitHub API client so no network access is needed.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ci.github import PR, _fetch_pr_github_data
from ci.utils import GithubStatus

# ── Mock helpers ─────────────────────────────────────────────────────────────


class MockGH:
    """Records calls and returns pre-canned responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0

    async def post(self, url: str, *, data: Any) -> Any:  # pylint: disable=unused-argument
        assert self.call_count < len(self._responses), (
            f'Unexpected API call #{self.call_count + 1} (only {len(self._responses)} response(s) queued)'
        )
        response = self._responses[self.call_count]
        self.call_count += 1
        return response


def graphql_response(pr_data_by_number: dict) -> dict:
    return {"data": {"repository": {f"pr_{n}": data for n, data in pr_data_by_number.items()}}}


def pr_response(review_decision, check_nodes, *, has_next_page=False, end_cursor=None):
    """Minimal pullRequest fragment matching the shape _fetch_pr_github_data expects."""
    return {
        "reviewDecision": review_decision,
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
        "commits": {"nodes": [{"commit": {"statusCheckRollup": None}}]},
    }


def check_run(name, conclusion, *, is_required=True):
    return {"__typename": "CheckRun", "name": name, "conclusion": conclusion, "isRequired": is_required}


def status_context(context, state, *, is_required=True):
    return {"__typename": "StatusContext", "context": context, "state": state, "isRequired": is_required}


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

    result = await _fetch_pr_github_data(gh, 'hail-is', 'hail', [123])

    assert gh.call_count == 1
    review_decision, check_nodes = result[123]
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

    result = await _fetch_pr_github_data(gh, 'hail-is', 'hail', [1, 2])

    assert gh.call_count == 1
    assert result[1] == ('APPROVED', nodes_a)
    assert result[2] == ('REVIEW_REQUIRED', nodes_b)


async def test_pagination_fires_multiple_requests_and_accumulates_nodes():
    page1 = [check_run('check-a', 'SUCCESS')]
    page2 = [check_run('check-b', 'FAILURE')]
    page3 = [check_run('check-c', 'SUCCESS')]

    gh = MockGH([
        graphql_response({42: pr_response('APPROVED', page1, has_next_page=True, end_cursor='cursor-1')}),
        # reviewDecision is ignored on subsequent pages (already captured from page 1)
        graphql_response({42: pr_response(None, page2, has_next_page=True, end_cursor='cursor-2')}),
        graphql_response({42: pr_response(None, page3, has_next_page=False)}),
    ])

    result = await _fetch_pr_github_data(gh, 'hail-is', 'hail', [42])

    assert gh.call_count == 3
    review_decision, check_nodes = result[42]
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
        graphql_response({1: pr_response(None, page2_pr1, has_next_page=True, end_cursor='cur-2')}),
        graphql_response({1: pr_response(None, page3_pr1, has_next_page=False)}),
    ])

    result = await _fetch_pr_github_data(gh, 'hail-is', 'hail', [1, 2])

    assert gh.call_count == 3
    assert result[1] == ('APPROVED', page1_pr1 + page2_pr1 + page3_pr1)
    assert result[2] == ('REVIEW_REQUIRED', nodes_pr2)


async def test_null_review_decision_becomes_api_none():
    gh = MockGH([graphql_response({5: pr_response(None, [])})])

    result = await _fetch_pr_github_data(gh, 'hail-is', 'hail', [5])

    assert result[5][0] == 'API_NONE'


async def test_null_status_rollup_returns_empty_nodes():
    gh = MockGH([graphql_response({7: pr_response_no_rollup('REVIEW_REQUIRED')})])

    result = await _fetch_pr_github_data(gh, 'hail-is', 'hail', [7])

    review_decision, check_nodes = result[7]
    assert review_decision == 'REVIEW_REQUIRED'
    assert check_nodes == []


async def test_empty_pr_list_makes_no_request():
    gh = MockGH([])

    result = await _fetch_pr_github_data(gh, 'hail-is', 'hail', [])

    assert result == {}
    assert gh.call_count == 0


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
        pr._apply_github_data(review_decision, [])
    assert pr.review_state == expected_state
    assert pr.target_branch.state_changed is True


def test_review_state_unchanged_does_not_set_state_changed():
    pr = make_pr()
    with patch('ci.github.TRACKED_PRS'):
        pr._apply_github_data('APPROVED', [])
        pr.target_branch.state_changed = False
        pr._apply_github_data('APPROVED', [])
    assert pr.target_branch.state_changed is False


def test_required_check_run_appears_in_status():
    pr = make_pr()
    with patch('ci.github.TRACKED_PRS'):
        pr._apply_github_data('APPROVED', [check_run('ci-test', 'SUCCESS', is_required=True)])
    assert pr.last_known_github_status == {'ci-test': GithubStatus.SUCCESS}


def test_required_status_context_appears_in_status():
    pr = make_pr()
    with patch('ci.github.TRACKED_PRS'):
        pr._apply_github_data('APPROVED', [status_context('ci-status', 'FAILURE', is_required=True)])
    assert pr.last_known_github_status == {'ci-status': GithubStatus.FAILURE}


def test_non_required_checks_are_ignored():
    pr = make_pr()
    nodes = [
        check_run('required', 'SUCCESS', is_required=True),
        check_run('optional', 'FAILURE', is_required=False),
        status_context('optional-status', 'FAILURE', is_required=False),
    ]
    with patch('ci.github.TRACKED_PRS'):
        pr._apply_github_data('APPROVED', nodes)
    assert list(pr.last_known_github_status.keys()) == ['required']


def test_github_status_unchanged_does_not_set_state_changed():
    pr = make_pr()
    nodes = [check_run('ci-test', 'SUCCESS', is_required=True)]
    with patch('ci.github.TRACKED_PRS'):
        pr._apply_github_data('APPROVED', nodes)
        pr.target_branch.state_changed = False
        pr._apply_github_data('APPROVED', nodes)
    assert pr.target_branch.state_changed is False


def test_check_run_failure_status():
    pr = make_pr()
    with patch('ci.github.TRACKED_PRS'):
        pr._apply_github_data('APPROVED', [check_run('ci-test', 'FAILURE', is_required=True)])
    assert pr.last_known_github_status['ci-test'] == GithubStatus.FAILURE


def test_check_run_pending_conclusion():
    pr = make_pr()
    with patch('ci.github.TRACKED_PRS'):
        pr._apply_github_data('APPROVED', [check_run('ci-test', 'ACTION_REQUIRED', is_required=True)])
    assert pr.last_known_github_status['ci-test'] == GithubStatus.PENDING
