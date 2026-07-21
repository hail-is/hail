from typer.testing import CliRunner

from hailtop.hailctl.emr import cli

runner = CliRunner()


def test_start_requires_name_and_tmpdir():
    res = runner.invoke(cli.app, ['start'])
    assert res.exit_code != 0


def test_start_dry_run_makes_no_aws_calls(emr_client_mock, upload_mock):
    res = runner.invoke(
        cli.app, ['start', 'c1', '--s3-scratch', 's3://bkt/tmp/', '--dry-run']
    )
    assert res.exit_code == 0
    assert emr_client_mock.run_job_flow.call_count == 0
    assert upload_mock.call_count == 0  # dry run must not write to S3


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
    res = runner.invoke(
        cli.app, ['start', 'c1', '--s3-scratch', 's3://bkt/tmp/', '--release-label', 'emr-6.15.0']
    )
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


def test_stop_calls_terminate(emr_client_mock):
    res = runner.invoke(cli.app, ['stop', 'j-123'])
    assert res.exit_code == 0
    emr_client_mock.terminate_job_flows.assert_called_once_with(JobFlowIds=['j-123'])


def test_list_calls_list_clusters(emr_client_mock):
    emr_client_mock.list_clusters.return_value = {'Clusters': []}
    res = runner.invoke(cli.app, ['list'])
    assert res.exit_code == 0
    assert emr_client_mock.list_clusters.call_count == 1


def test_start_invalid_json_overlay_errors(emr_client_mock):
    res = runner.invoke(
        cli.app,
        ['start', 'c1', '--s3-scratch', 's3://bkt/tmp/', '--run-job-flow-json', '{not json'],
    )
    assert res.exit_code != 0
    assert emr_client_mock.run_job_flow.call_count == 0


def test_submit_invokes_submit(monkeypatch):
    called = {}

    def fake_submit(cluster_id, script, remote_tmpdir, region, pass_through_args, wait=True):
        called['cluster_id'] = cluster_id
        called['script'] = script
        return 's-1'

    monkeypatch.setattr('hailtop.hailctl.emr.submit.submit', fake_submit)
    res = runner.invoke(
        cli.app, ['submit', 'j-123', 'script.py', '--s3-scratch', 's3://bkt/tmp/']
    )
    assert res.exit_code == 0, res.stdout
    assert called['cluster_id'] == 'j-123'
    assert called['script'] == 'script.py'
