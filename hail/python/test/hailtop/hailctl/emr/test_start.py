import json
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from hailtop.hailctl.emr import emr
from hailtop.hailctl.emr.cli import app

WHEEL = 's3://my-bucket/wheels/hail-0.2.140-py3-none-any.whl'
runner = CliRunner()


@pytest.fixture
def aws(monkeypatch):
    client = Mock()
    client.run_job_flow.return_value = {'JobFlowId': 'j-ABC123'}
    upload = Mock()
    monkeypatch.setattr(emr, 'emr_client', lambda region: client)
    monkeypatch.setattr(emr, 'upload_to_s3', upload)
    return client, upload


def test_dry_run_prints_request_and_does_not_call_aws(aws):
    client, upload = aws
    res = runner.invoke(app, ['start', 'c', '--wheel', WHEEL, '--dry-run'])
    assert res.exit_code == 0, res.output
    request = json.loads(res.stdout)
    assert request['ReleaseLabel'] == 'emr-spark-8.1.0'
    assert request['ServiceRole'] == 'EMR_DefaultRole'
    assert request['JobFlowRole'] == 'EMR_EC2_DefaultRole'
    assert request['AutoTerminationPolicy'] == {'IdleTimeout': 3600}
    assert request['BootstrapActions'][0]['ScriptBootstrapAction']['Args'] == [WHEEL]
    assert request['BootstrapActions'][0]['ScriptBootstrapAction']['Path'].startswith(
        's3://my-bucket/wheels/bootstrap/install-hail-emr-'
    )
    client.run_job_flow.assert_not_called()
    upload.assert_not_called()


def test_start_uploads_bootstrap_then_creates_cluster(aws):
    client, upload = aws
    res = runner.invoke(app, ['start', 'c', '--wheel', WHEEL, '-w', '3'])
    assert res.exit_code == 0, res.output
    assert 'Started cluster j-ABC123.' in res.stdout
    upload.assert_called_once()
    kwargs = client.run_job_flow.call_args.kwargs
    assert kwargs['Instances']['InstanceGroups'][1]['InstanceCount'] == 3
    assert upload.call_args.args[0] == kwargs['BootstrapActions'][0]['ScriptBootstrapAction']['Path']


def test_s3a_and_hail_cloud_configuration(aws):
    res = runner.invoke(app, ['start', 'c', '--wheel', WHEEL, '--dry-run'])
    configs = {c['Classification']: c for c in json.loads(res.stdout)['Configurations']}
    assert configs['core-site']['Properties']['fs.s3a.committer.name'] == 'magicv2'
    assert configs['spark-defaults']['Properties']['spark.executorEnv.HAIL_CLOUD'] == 'aws'


def test_optional_network_options(aws):
    res = runner.invoke(
        app,
        [
            'start',
            'c',
            '--wheel',
            WHEEL,
            '--subnet-id',
            'subnet-1',
            '--service-access-security-group',
            'sg-1',
            '--dry-run',
        ],
    )
    instances = json.loads(res.stdout)['Instances']
    assert instances['Ec2SubnetId'] == 'subnet-1'
    assert instances['ServiceAccessSecurityGroup'] == 'sg-1'
    res = runner.invoke(app, ['start', 'c', '--wheel', WHEEL, '--dry-run'])
    assert 'Ec2SubnetId' not in json.loads(res.stdout)['Instances']


@pytest.mark.parametrize('wheel', ['gs://b/hail-1.whl', 's3://b/other.whl', 'hail-1.whl', 's3:///hail-1.whl'])
def test_rejects_bad_wheel(aws, wheel):
    res = runner.invoke(app, ['start', 'c', '--wheel', wheel, '--dry-run'])
    assert res.exit_code != 0


@pytest.mark.parametrize('timeout', ['59', '604801'])
def test_rejects_bad_idle_timeout(aws, timeout):
    res = runner.invoke(app, ['start', 'c', '--wheel', WHEEL, '--idle-timeout', timeout, '--dry-run'])
    assert res.exit_code != 0


def test_stop_and_list(aws):
    client, _ = aws
    assert runner.invoke(app, ['stop', 'j-1']).exit_code == 0
    client.terminate_job_flows.assert_called_once_with(JobFlowIds=['j-1'])
    client.get_paginator.return_value.paginate.return_value = [
        {'Clusters': [{'Id': 'j-1', 'Status': {'State': 'WAITING'}, 'Name': 'c'}]}
    ]
    res = runner.invoke(app, ['list'])
    assert res.stdout == 'j-1\tWAITING\tc\n'


def test_missing_region_is_a_user_facing_error(monkeypatch):
    import boto3

    monkeypatch.setattr(boto3.session.Session, 'region_name', None, raising=False)
    with pytest.raises(Exception, match='Could not determine the AWS region'):
        emr.emr_client(None)
