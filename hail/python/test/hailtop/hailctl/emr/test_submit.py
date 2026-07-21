from unittest.mock import Mock, patch

import pytest

from hailtop.hailctl.emr import submit


def test_step_waiter_failure_raises_system_exit_even_when_describe_also_fails(tmp_path, monkeypatch):
    """describe_step failure must not mask the SystemExit(1) from the waiter failure."""
    script = tmp_path / 'job.py'
    script.write_text('print("hello")')

    mock_client = Mock()
    mock_waiter = Mock()
    mock_waiter.wait.side_effect = RuntimeError('waiter timed out')
    mock_client.get_waiter.return_value = mock_waiter
    mock_client.add_job_flow_steps.return_value = {'StepIds': ['s-FAKE']}
    mock_client.describe_step.side_effect = RuntimeError('network error')

    monkeypatch.setattr('hailtop.hailctl.emr.emr.resolve_region', lambda region: 'us-east-1')
    monkeypatch.setattr('hailtop.hailctl.emr.emr.emr_client', lambda region: mock_client)
    monkeypatch.setattr('hailtop.hailctl.emr.emr.upload_to_s3', Mock())

    with pytest.raises(SystemExit):
        submit.submit('j-FAKE', str(script), 's3://bkt/tmp/', None, [], wait=True)


def test_spark_submit_step_args():
    args = submit.spark_submit_step_args('s3://bkt/scripts/x.py', ['--foo', 'bar'])
    assert args[0] == 'spark-submit'
    assert args[-3:] == ['s3://bkt/scripts/x.py', '--foo', 'bar']


def test_spark_submit_step_args_no_passthrough():
    args = submit.spark_submit_step_args('s3://bkt/scripts/x.py', [])
    assert args == ['spark-submit', '--deploy-mode', 'client', 's3://bkt/scripts/x.py']
