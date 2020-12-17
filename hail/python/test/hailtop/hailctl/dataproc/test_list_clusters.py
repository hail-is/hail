import pytest

from hailtop import hailctl


def test_list(gcloud_run):
    hailctl.main(["dataproc", "list"])
    assert gcloud_run.call_args[0][0] == ["dataproc", "clusters", "list"]


def test_clusters_project(gcloud_run):
    hailctl.main(["dataproc", "list", "--", "--project=foo"])
    assert "--project=foo" in gcloud_run.call_args[0][0]


def test_clusters_region(gcloud_run):
    hailctl.main(["dataproc", "list", "--", "--region=europe-north1"])
    assert "--region=europe-north1" in gcloud_run.call_args[0][0]
