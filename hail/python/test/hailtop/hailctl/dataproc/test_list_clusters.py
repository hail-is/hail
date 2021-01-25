import pytest

from hailtop import hailctl


def test_list(gcloud_run):
    hailctl.main(["dataproc", "list"])
    assert gcloud_run.call_args[0][0] == ["gcloud", "--project=hailctl-dataproc-tests", "dataproc", "--region=us-central1", "clusters", "list"]


def test_clusters_project(gcloud_run):
    hailctl.main(["dataproc", "--project=foo", "list"])
    assert "--project=foo" in gcloud_run.call_args[0][0]


def test_clusters_region(gcloud_run):
    hailctl.main(["dataproc", "--zone=europe-north1-a", "list"])
    assert "--region=europe-north1" in gcloud_run.call_args[0][0]
