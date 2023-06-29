from typer.testing import CliRunner

from hailtop.hailctl.dataproc import cli


runner = CliRunner(mix_stderr=False)


def test_list(gcloud_run):
    runner.invoke(cli.app, ['list'])
    assert gcloud_run.call_args[0][0] == ["dataproc", "clusters", "list"]


def test_clusters_project(gcloud_run):
    runner.invoke(cli.app, ['list', '--project=foo'])
    assert "--project=foo" in gcloud_run.call_args[0][0]


def test_clusters_region(gcloud_run):
    runner.invoke(cli.app, ['list', '--region=europe-north1'])
    assert "--region=europe-north1" in gcloud_run.call_args[0][0]
