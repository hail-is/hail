from hailtop.hailctl.dataproc import cli


def test_list(runner, gcloud_run):
    runner.invoke(cli.app, ['list'])
    assert gcloud_run.call_args[0][0] == ["dataproc", "clusters", "list"]


def test_clusters_project(runner, gcloud_run):
    runner.invoke(cli.app, ['list', '--project=foo'])
    assert "--project=foo" in gcloud_run.call_args[0][0]


def test_clusters_region(runner, gcloud_run):
    runner.invoke(cli.app, ['list', '--region=europe-north1'])
    assert "--region=europe-north1" in gcloud_run.call_args[0][0]
