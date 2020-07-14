import pytest

from hailtop.hailctl.dataproc import cli


def test_submit(gcloud_run):
    cli.main(["submit", "test-cluster", "a-script.py"])
    gcloud_args = gcloud_run.call_args[0][0]
    assert gcloud_args[:5] == ["dataproc", "jobs", "submit", "pyspark", "a-script.py"]
    assert "--cluster=test-cluster" in gcloud_args


def test_cluster_and_script_required(gcloud_run):
    with pytest.raises(SystemExit):
        cli.main(["submit"])

    assert gcloud_run.call_count == 0

    with pytest.raises(SystemExit):
        cli.main(["submit", "test-cluster"])

    assert gcloud_run.call_count == 0


def test_dry_run(gcloud_run):
    cli.main(["submit", "test-cluster", "a-script.py", "--dry-run"])
    assert gcloud_run.call_count == 0


def test_script_args(gcloud_run):
    cli.main(["submit", "test-cluster", "a-script.py", "--foo", "bar"])
    gcloud_args = gcloud_run.call_args[0][0]
    job_args = gcloud_args[gcloud_args.index("--") + 1:]
    assert job_args == ["--foo", "bar"]


def test_files(gcloud_run):
    cli.main(["submit", "test-cluster", "a-script.py", "--files=some-file.txt"])
    assert "--" not in gcloud_run.call_args[0][0]  # make sure arg is passed to gcloud and not job
    assert "--files=some-file.txt" in gcloud_run.call_args[0][0]


def test_properties(gcloud_run):
    cli.main(["submit", "test-cluster", "a-script.py", "--properties=spark:spark.task.maxFailures=3"])
    assert "--" not in gcloud_run.call_args[0][0]  # make sure arg is passed to gcloud and not job
    assert "--properties=spark:spark.task.maxFailures=3" in gcloud_run.call_args[0][0]


def test_gcloud_configuration(gcloud_run):
    cli.main(["submit", "test-cluster", "a-script.py", "--gcloud_configuration=some-config"])
    assert "--" not in gcloud_run.call_args[0][0]  # make sure arg is passed to gcloud and not job
    assert "--configuration=some-config" in gcloud_run.call_args[0][0]
