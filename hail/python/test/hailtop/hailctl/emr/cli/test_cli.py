import json

import pytest
from typer.testing import CliRunner

from hailtop.hailctl.emr import cli

runner = CliRunner()


def test_start_requires_name_and_tmpdir():
    res = runner.invoke(cli.app, ['start'])
    assert res.exit_code != 0


def test_start_dry_run_makes_no_aws_calls(emr_client_mock, upload_mock):
    res = runner.invoke(cli.app, ['start', 'c1', '--s3-scratch', 's3://bkt/tmp/', '--dry-run'])
    assert res.exit_code == 0
    assert emr_client_mock.run_job_flow.call_count == 0
    assert upload_mock.call_count == 0  # dry run must not write to S3


def test_start_dry_run_does_not_check_iam_roles(check_default_roles_mock):
    res = runner.invoke(cli.app, ['start', 'c1', '--s3-scratch', 's3://bkt/tmp/', '--dry-run'])
    assert res.exit_code == 0
    assert check_default_roles_mock.call_count == 0


def test_start_calls_run_job_flow_with_hail_config(emr_client_mock, upload_mock):
    emr_client_mock.run_job_flow.return_value = {'JobFlowId': 'j-123'}
    res = runner.invoke(cli.app, ['start', 'c1', '--s3-scratch', 's3://bkt/tmp/'])
    assert res.exit_code == 0, res.stdout
    # bootstrap script uploaded to the scratch bucket
    assert upload_mock.call_count == 1
    kwargs = emr_client_mock.run_job_flow.call_args.kwargs
    assert kwargs['ReleaseLabel'] == 'emr-7.3.0'
    spark_defaults = next(c for c in kwargs['Configurations'] if c['Classification'] == 'spark-defaults')
    assert spark_defaults['Properties']['spark.executorEnv.HAIL_CLOUD'] == 'aws'
    ba = kwargs['BootstrapActions'][0]['ScriptBootstrapAction']
    assert ba['Path'].startswith('s3://bkt/tmp/')
    assert ba['Path'].endswith('install-hail-emr.sh')


def test_start_unknown_release_errors(emr_client_mock):
    res = runner.invoke(cli.app, ['start', 'c1', '--s3-scratch', 's3://bkt/tmp/', '--release-label', 'emr-6.15.0'])
    assert res.exit_code != 0
    assert emr_client_mock.run_job_flow.call_count == 0


def test_start_json_overlay_merges(emr_client_mock):
    emr_client_mock.run_job_flow.return_value = {'JobFlowId': 'j-1'}
    res = runner.invoke(
        cli.app,
        ['start', 'c1', '--s3-scratch', 's3://bkt/tmp/', '--run-job-flow-json', '{"Name": "override"}'],
    )
    assert res.exit_code == 0, res.stdout
    assert emr_client_mock.run_job_flow.call_args.kwargs['Name'] == 'override'


def test_start_json_overlay_custom_roles_skip_default_role_preflight(emr_client_mock, check_default_roles_mock):
    emr_client_mock.run_job_flow.return_value = {'JobFlowId': 'j-1'}
    res = runner.invoke(
        cli.app,
        [
            'start',
            'c1',
            '--s3-scratch',
            's3://bkt/tmp/',
            '--run-job-flow-json',
            '{"ServiceRole":"custom-service","JobFlowRole":"custom-profile"}',
        ],
    )
    assert res.exit_code == 0, res.output
    assert check_default_roles_mock.call_count == 0
    kwargs = emr_client_mock.run_job_flow.call_args.kwargs
    assert kwargs['ServiceRole'] == 'custom-service'
    assert kwargs['JobFlowRole'] == 'custom-profile'


@pytest.mark.parametrize(
    ('overlay', 'expected_call'),
    [
        (
            '{"ServiceRole":"custom-service"}',
            {'check_service_role': False, 'check_job_flow_role': True},
        ),
        (
            '{"JobFlowRole":"custom-profile"}',
            {'check_service_role': True, 'check_job_flow_role': False},
        ),
    ],
)
def test_start_json_overlay_mixed_roles_check_remaining_default(
    overlay, expected_call, emr_client_mock, check_default_roles_mock
):
    emr_client_mock.run_job_flow.return_value = {'JobFlowId': 'j-1'}
    res = runner.invoke(
        cli.app,
        ['start', 'c1', '--s3-scratch', 's3://bkt/tmp/', '--run-job-flow-json', overlay],
    )
    assert res.exit_code == 0, res.output
    check_default_roles_mock.assert_called_once_with(**expected_call)


def test_stop_calls_terminate(emr_client_mock):
    res = runner.invoke(cli.app, ['stop', 'j-123'])
    assert res.exit_code == 0
    emr_client_mock.terminate_job_flows.assert_called_once_with(JobFlowIds=['j-123'])


def test_list_filters_to_active_clusters_and_paginates(emr_client_mock):
    paginator = emr_client_mock.get_paginator.return_value
    paginator.paginate.return_value = [
        {'Clusters': [{'Id': 'j-1', 'Status': {'State': 'WAITING'}, 'Name': 'c1'}]},
        {'Clusters': [{'Id': 'j-2', 'Status': {'State': 'RUNNING'}, 'Name': 'c2'}]},
    ]
    res = runner.invoke(cli.app, ['list'])
    assert res.exit_code == 0, res.stdout
    emr_client_mock.get_paginator.assert_called_once_with('list_clusters')
    paginator.paginate.assert_called_once_with(ClusterStates=cli.ACTIVE_CLUSTER_STATES)
    # every page is consumed, not just the first
    assert 'j-1' in res.stdout and 'j-2' in res.stdout


def test_list_all_includes_terminated(emr_client_mock):
    paginator = emr_client_mock.get_paginator.return_value
    paginator.paginate.return_value = []
    res = runner.invoke(cli.app, ['list', '--all'])
    assert res.exit_code == 0, res.stdout
    paginator.paginate.assert_called_once_with()


def test_start_rejects_unknown_flags(emr_client_mock):
    # A typo like --core-instance-cout must not be silently ignored, which would
    # start a cluster with a different shape than the user asked for.
    res = runner.invoke(
        cli.app, ['start', 'c1', '--s3-scratch', 's3://bkt/tmp/', '--dry-run', '--core-instance-cout', '5']
    )
    assert res.exit_code != 0
    assert emr_client_mock.run_job_flow.call_count == 0


def test_start_rejects_custom_roles_without_no_use_default_roles(emr_client_mock):
    # Otherwise the custom role is silently dropped and the cluster quietly runs
    # under EMR_DefaultRole.
    res = runner.invoke(cli.app, ['start', 'c1', '--s3-scratch', 's3://bkt/tmp/', '--service-role', 'my-role'])
    assert res.exit_code != 0
    assert emr_client_mock.run_job_flow.call_count == 0


def test_start_invalid_json_overlay_errors(emr_client_mock):
    res = runner.invoke(
        cli.app,
        ['start', 'c1', '--s3-scratch', 's3://bkt/tmp/', '--run-job-flow-json', '{not json'],
    )
    assert res.exit_code != 0
    assert emr_client_mock.run_job_flow.call_count == 0


@pytest.mark.parametrize('overlay', ['[]', 'null', '1', '"value"'])
def test_start_json_overlay_requires_object(overlay, emr_client_mock):
    res = runner.invoke(
        cli.app,
        ['start', 'c1', '--s3-scratch', 's3://bkt/tmp/', '--dry-run', '--run-job-flow-json', overlay],
    )
    assert res.exit_code != 0
    assert '--run-job-flow-json must contain a JSON object' in res.output
    assert emr_client_mock.run_job_flow.call_count == 0


def test_start_json_overlay_revalidates_release(emr_client_mock):
    res = runner.invoke(
        cli.app,
        [
            'start',
            'c1',
            '--s3-scratch',
            's3://bkt/tmp/',
            '--dry-run',
            '--run-job-flow-json',
            '{"ReleaseLabel":"emr-6.15.0"}',
        ],
    )
    assert res.exit_code != 0
    assert emr_client_mock.run_job_flow.call_count == 0


@pytest.mark.parametrize(
    'scratch',
    [
        'gs://bkt/tmp/',
        's3://',
        's3:///tmp/',
        's3:// bad/tmp/',
        's3://bkt/tmp/?x=1',
        's3://user@bucket/key',
        's3://bucket:443/key',
        r's3://bucket\evil/key',
        's3://[broken/key',
    ],
)
def test_start_rejects_invalid_s3_scratch(scratch, emr_client_mock, upload_mock):
    res = runner.invoke(cli.app, ['start', 'c1', '--s3-scratch', scratch, '--dry-run'])
    assert res.exit_code != 0
    assert '--s3-scratch must be a valid S3 URI' in res.output
    assert emr_client_mock.run_job_flow.call_count == 0
    assert upload_mock.call_count == 0


def test_start_rejects_non_s3_log_uri(emr_client_mock, upload_mock):
    res = runner.invoke(
        cli.app,
        ['start', 'c1', '--s3-scratch', 's3://bkt/tmp/', '--log-uri', 'gs://bkt/logs/', '--dry-run'],
    )
    assert res.exit_code != 0
    assert '--log-uri must be a valid S3 URI' in res.output
    assert emr_client_mock.run_job_flow.call_count == 0
    assert upload_mock.call_count == 0


def test_start_json_overlay_revalidates_log_uri(emr_client_mock, upload_mock):
    res = runner.invoke(
        cli.app,
        [
            'start',
            'c1',
            '--s3-scratch',
            's3://bkt/tmp/',
            '--dry-run',
            '--run-job-flow-json',
            '{"LogUri":"gs://bkt/logs/"}',
        ],
    )
    assert res.exit_code != 0
    assert 'final RunJobFlow LogUri must be a valid S3 URI' in res.output
    assert emr_client_mock.run_job_flow.call_count == 0
    assert upload_mock.call_count == 0


def test_start_json_overlay_rejects_null_log_uri(emr_client_mock, upload_mock):
    res = runner.invoke(
        cli.app,
        [
            'start',
            'c1',
            '--s3-scratch',
            's3://bkt/tmp/',
            '--dry-run',
            '--run-job-flow-json',
            '{"LogUri":null}',
        ],
    )
    assert res.exit_code != 0
    assert 'final RunJobFlow LogUri must be a string' in res.output
    assert emr_client_mock.run_job_flow.call_count == 0
    assert upload_mock.call_count == 0


def test_start_json_overlay_requires_spark(emr_client_mock):
    res = runner.invoke(
        cli.app,
        [
            'start',
            'c1',
            '--s3-scratch',
            's3://bkt/tmp/',
            '--dry-run',
            '--run-job-flow-json',
            '{"Applications":[{"Name":"Hadoop"}]}',
        ],
    )
    assert res.exit_code != 0
    assert 'Applications list must include Spark' in res.output
    assert emr_client_mock.run_job_flow.call_count == 0


def test_start_json_overlay_instance_fleets_replace_default_groups():
    res = runner.invoke(
        cli.app,
        [
            'start',
            'c1',
            '--s3-scratch',
            's3://bkt/tmp/',
            '--dry-run',
            '--run-job-flow-json',
            '{"Instances":{"InstanceFleets":[]}}',
        ],
    )
    assert res.exit_code == 0, res.output
    instances = json.loads(res.output)['Instances']
    assert 'InstanceGroups' not in instances
    assert instances['InstanceFleets'] == []


def test_start_json_overlay_rejects_groups_and_fleets_together(emr_client_mock):
    res = runner.invoke(
        cli.app,
        [
            'start',
            'c1',
            '--s3-scratch',
            's3://bkt/tmp/',
            '--dry-run',
            '--run-job-flow-json',
            '{"Instances":{"InstanceGroups":[],"InstanceFleets":[]}}',
        ],
    )
    assert res.exit_code != 0
    assert 'cannot contain both InstanceGroups' in res.output
    assert 'InstanceFleets' in res.output
    assert emr_client_mock.run_job_flow.call_count == 0


def test_submit_invokes_submit(monkeypatch):
    called = {}

    def fake_submit(cluster_id, script, remote_tmpdir, region, pass_through_args, wait=True):
        called['cluster_id'] = cluster_id
        called['script'] = script
        return 's-1'

    monkeypatch.setattr('hailtop.hailctl.emr.submit.submit', fake_submit)
    res = runner.invoke(cli.app, ['submit', 'j-123', 'script.py', '--s3-scratch', 's3://bkt/tmp/'])
    assert res.exit_code == 0, res.stdout
    assert called['cluster_id'] == 'j-123'
    assert called['script'] == 'script.py'


def test_submit_rejects_non_s3_scratch(monkeypatch):
    called = False

    def fake_submit(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr('hailtop.hailctl.emr.submit.submit', fake_submit)
    res = runner.invoke(cli.app, ['submit', 'j-123', 'script.py', '--s3-scratch', 'gs://bkt/tmp/'])
    assert res.exit_code != 0
    assert '--s3-scratch must be a valid S3 URI' in res.output
    assert not called
