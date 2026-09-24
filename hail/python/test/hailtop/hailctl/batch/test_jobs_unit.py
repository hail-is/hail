from typing import List
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from hailtop.hailctl.batch import cli

runner = CliRunner()


def invoke_jobs(args: List[str]):
    mock_batch = MagicMock()
    mock_batch.jobs.return_value = iter([])
    with patch('hailtop.batch_client.client.BatchClient') as MockClient:
        MockClient.return_value.__enter__.return_value.get_batch.return_value = mock_batch
        result = runner.invoke(cli.app, ['jobs', '123', *args], catch_exceptions=False)
    return result, mock_batch


@pytest.mark.parametrize(
    'args, expected_q, expected_last_job_id',
    [
        # fmt: off
        ([], None, None),
        (['--state', 'bad'], 'state=bad', None),
        (['--state', 'live'], 'state=live', None),
        (['--state', 'done'], 'state=done', None),
        (['--state', 'failed'], 'state=failed', None),
        (['--exit-code', '1'], 'exit_code=1', None),
        (['--name', 'my-job'], 'name=my-job', None),
        (['--state', 'bad', '--exit-code', '1'], 'state=bad\nexit_code=1', None),
        (['--state', 'bad', '--name', 'step'], 'state=bad\nname=step', None),
        (['--exit-code', '137', '--name', 'oom'], 'exit_code=137\nname=oom', None),
        (['--last-job-id', '50'], None, 50),
        (['--state', 'bad', '--last-job-id', '99'], 'state=bad', 99),
        # fmt: on
    ],
)
def test_jobs_query(args, expected_q, expected_last_job_id):
    result, mock_batch = invoke_jobs(args)
    assert result.exit_code == 0, result.output
    mock_batch.jobs.assert_called_once_with(q=expected_q, version=2, last_job_id=expected_last_job_id)


@pytest.mark.parametrize('state', ['garbage', 'FAILED', 'Bad', ''])
def test_invalid_state(state):
    result, _ = invoke_jobs(['--state', state])
    assert result.exit_code == 2
    assert 'Invalid state' in result.output


@pytest.mark.parametrize(
    'args, expected_break_at',
    [
        (['--limit', '3'], 3),
        (['--limit', '1'], 1),
    ],
)
def test_limit(args, expected_break_at):
    jobs = [{'job_id': i} for i in range(10)]
    mock_batch = MagicMock()
    mock_batch.jobs.return_value = iter(jobs)
    with patch('hailtop.batch_client.client.BatchClient') as MockClient:
        MockClient.return_value.__enter__.return_value.get_batch.return_value = mock_batch
        result = runner.invoke(cli.app, ['jobs', '123', *args], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    # output should contain exactly expected_break_at jobs
    import yaml  # pylint: disable=import-outside-toplevel

    parsed = yaml.safe_load(result.output)
    assert len(parsed) == expected_break_at
