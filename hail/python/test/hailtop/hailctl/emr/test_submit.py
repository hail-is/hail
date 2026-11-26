from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from hailtop.hailctl.emr import emr
from hailtop.hailctl.emr.cli import app
from hailtop.hailctl.emr.submit import spark_submit_step_args

runner = CliRunner()


@pytest.fixture
def aws(monkeypatch):
    client = Mock()
    client.add_job_flow_steps.return_value = {'StepIds': ['s-1']}
    upload = Mock()
    monkeypatch.setattr(emr, 'emr_client', lambda region: client)
    monkeypatch.setattr(emr, 'upload_to_s3', upload)
    return client, upload


def test_step_args():
    assert spark_submit_step_args('s3://b/x.py', ['--a', '1']) == [
        'spark-submit',
        '--deploy-mode',
        'client',
        's3://b/x.py',
        '--a',
        '1',
    ]


def test_submit_s3_script_does_not_upload(aws):
    client, upload = aws
    res = runner.invoke(app, ['submit', 'j-1', 's3://b/x.py', '--no-wait', '--', '--a', '1'])
    assert res.exit_code == 0, res.output
    upload.assert_not_called()
    step = client.add_job_flow_steps.call_args.kwargs['Steps'][0]
    assert step['HadoopJarStep']['Args'] == ['spark-submit', '--deploy-mode', 'client', 's3://b/x.py', '--a', '1']


def test_submit_local_script_uploads(aws, tmp_path):
    _, upload = aws
    script = tmp_path / 'job.py'
    script.write_text('print(1)')
    res = runner.invoke(app, ['submit', 'j-1', str(script), '--upload-to', 's3://b/scripts/', '--no-wait'])
    assert res.exit_code == 0, res.output
    assert upload.call_args.args[0] == 's3://b/scripts/job.py'


def test_submit_local_script_requires_upload_to(aws, tmp_path):
    script = tmp_path / 'job.py'
    script.write_text('print(1)')
    res = runner.invoke(app, ['submit', 'j-1', str(script), '--no-wait'])
    assert res.exit_code != 0
