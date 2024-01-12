from typer.testing import CliRunner

from hailtop.hailctl.dataproc import cli

runner = CliRunner(mix_stderr=False)


def test_stop(gcloud_run):
    runner.invoke(cli.app, ["stop", "test-cluster"])
    assert gcloud_run.call_args[0][0][:3] == ["dataproc", "clusters", "delete"]


def test_cluster_name_required(gcloud_run):
    res = runner.invoke(cli.app, ["stop"])
    assert res.exit_code == 2
    assert "Missing argument 'NAME'" in res.stderr
    assert gcloud_run.call_count == 0


def test_dry_run(gcloud_run):
    res = runner.invoke(cli.app, ["stop", "test-cluster", "--dry-run"])
    assert res.exit_code == 0
    assert gcloud_run.call_count == 0


def test_cluster_project(gcloud_run):
    runner.invoke(cli.app, ["stop", "--project=foo", "test-cluster"])
    assert "--project=foo" in gcloud_run.call_args[0][0]


def test_cluster_region(gcloud_run):
    runner.invoke(cli.app, ["stop", "--region=europe-north1", "test-cluster"])
    assert "--region=europe-north1" in gcloud_run.call_args[0][0]


def test_async(gcloud_run):
    runner.invoke(cli.app, ["stop", "test-cluster", "--async"])
    assert "--async" in gcloud_run.call_args[0][0]
