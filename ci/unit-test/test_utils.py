import pytest

from ci.github import GithubStatus, github_status


def test_github_check_statuses():
    for state in ["PENDING", "EXPECTED", "ACTION_REQUIRED", "STALE"]:
        assert github_status(state) == GithubStatus.PENDING
    for state in ["FAILURE", "ERROR", "TIMED_OUT", "CANCELLED", "STARTUP_FAILURE", "SKIPPED"]:
        assert github_status(state) == GithubStatus.FAILURE
    for state in ["SUCCESS", "NEUTRAL"]:
        assert github_status(state) == GithubStatus.SUCCESS
    for state in ["foo", "bar"]:
        with pytest.raises(ValueError):
            github_status(state)
