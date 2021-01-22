import pytest

from hailtop import hailctl


def test_stop(gcloud_run):
    hailctl.main(["dataproc", "stop", "test-cluster"])
    assert gcloud_run.call_args[0][0][:7] == ["gcloud", "--project=hailctl-dataproc-tests", "--zone=us-central1-b", "dataproc", "--region=us-central1", "clusters", "delete"]


def test_cluster_name_required(capsys, gcloud_run):
    with pytest.raises(SystemExit):
        hailctl.main(["dataproc", "stop"])

    assert "Missing argument 'CLUSTER_NAME'" in capsys.readouterr().err
    assert gcloud_run.call_count == 0


def test_dry_run(gcloud_run):
    hailctl.main(["dataproc", "stop", "test-cluster", "--dry-run"])
    assert gcloud_run.call_count == 0


def test_cluster_project(gcloud_run):
    hailctl.main(["dataproc", "stop", "test-cluster", "--project=foo"])
    assert "--project=foo" in gcloud_run.call_args[0][0]


def test_cluster_region(gcloud_run):
    hailctl.main(["dataproc", "stop", "test-cluster", "--zone=europe-north1-a"])
    assert "--region=europe-north1" in gcloud_run.call_args[0][0]


def test_async(gcloud_run):
    hailctl.main(["dataproc", "stop", "test-cluster", "--async"])
    assert "--async" in gcloud_run.call_args[0][0]
