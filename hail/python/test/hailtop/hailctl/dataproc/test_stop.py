from hailtop.hailctl.dataproc import cli


def test_stop(runner, gcloud_run):
    runner.invoke(cli.app, ['stop', 'test-cluster'])
    assert gcloud_run.call_args[0][0][:3] == ["dataproc", "clusters", "delete"]


def test_cluster_name_required(runner, gcloud_run):
    res = runner.invoke(cli.app, ['stop'])
    assert res.exit_code == 2
    assert "Missing argument 'NAME'" in res.stderr
    assert gcloud_run.call_count == 0


def test_dry_run(runner, gcloud_run):
    res = runner.invoke(cli.app, ['stop', 'test-cluster', '--dry-run'])
    assert res.exit_code == 0
    assert gcloud_run.call_count == 0


def test_cluster_project(runner, gcloud_run):
    runner.invoke(cli.app, ['stop', '--project=foo', 'test-cluster'])
    assert "--project=foo" in gcloud_run.call_args[0][0]


def test_cluster_region(runner, gcloud_run):
    runner.invoke(cli.app, ['stop', '--region=europe-north1', 'test-cluster'])
    assert "--region=europe-north1" in gcloud_run.call_args[0][0]


def test_async(runner, gcloud_run):
    runner.invoke(cli.app, ['stop', 'test-cluster', '--async'])
    assert "--async" in gcloud_run.call_args[0][0]
