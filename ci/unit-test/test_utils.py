import pytest

from ci.utils import GithubStatus, github_status, record_success


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


async def test_record_success_records_only_successful_calls(monkeypatch):
    monkeypatch.setattr('ci.utils.time_msecs', lambda: 42)
    last_success: dict = {}

    async def flaky_loop(fail: bool):
        if fail:
            raise RuntimeError('loop failed')
        return 'ok'

    wrapped = record_success(flaky_loop, last_success)
    assert wrapped.__name__ == 'flaky_loop'

    with pytest.raises(RuntimeError):
        await wrapped(True)
    assert not last_success

    assert await wrapped(False) == 'ok'
    assert last_success == {'flaky_loop': 42}
