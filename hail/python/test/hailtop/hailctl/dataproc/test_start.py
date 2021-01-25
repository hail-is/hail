import pytest

from hailtop import hailctl


def test_cluster_name_required(capsys, gcloud_run):
    with pytest.raises(SystemExit):
        hailctl.main(["dataproc", "start"])

    assert "Missing argument 'CLUSTER_NAME'" in capsys.readouterr().err
    assert gcloud_run.call_count == 0


def test_dry_run(gcloud_run):
    hailctl.main(["dataproc", "--dry-run", "start", "test-cluster"])
    assert gcloud_run.call_count == 0


def test_cluster_project(gcloud_run):
    hailctl.main(["dataproc", "--project", "foo", "start", "test-cluster"])
    assert "--project=foo" in gcloud_run.call_args[0][0]


def test_cluster_zone(gcloud_run):
    hailctl.main(["dataproc", "--zone=us-central1-b", "start", "test-cluster"])
    assert "--region=us-central1" in gcloud_run.call_args[0][0]


def test_creator_label(gcloud_run, gcloud_config):
    gcloud_config["account"] = "test-user@hail.is"
    hailctl.main(["dataproc", "start", "my-cluster"])
    assert "--labels=creator=test-user_hail_is" in gcloud_run.call_args[0][0]

    gcloud_config["account"] = None
    hailctl.main(["dataproc", "start", "my-cluster"])
    assert not any(arg.startswith("--labels=") and "creator=" in arg for arg in gcloud_run.call_args[0][0])


def test_workers_configuration(gcloud_run):
    hailctl.main(["dataproc", "start", "--num-workers=4", "test-cluster"])
    assert "--num-workers=4" in gcloud_run.call_args[0][0]


@pytest.mark.parametrize("workers_arg", [
    "--num-secondary-workers=8",
    "--num-preemptible-workers=8"
])
def test_secondary_workers_configuration(gcloud_run, workers_arg):
    hailctl.main(["dataproc", "start", workers_arg, "test-cluster"])
    assert "--num-secondary-workers=8" in gcloud_run.call_args[0][0]


@pytest.mark.parametrize("machine_arg", [
    "--master-machine-type=n1-highmem-16",
    "--worker-machine-type=n1-standard-32",
])
def test_machine_type_configuration(gcloud_run, machine_arg):
    hailctl.main(["dataproc", "start", machine_arg, "test-cluster"])
    assert machine_arg in gcloud_run.call_args[0][0]


@pytest.mark.parametrize("machine_arg", [
    "--master-boot-disk-size=250",
    "--worker-boot-disk-size=200",
    "--secondary-worker-boot-disk-size=100"
])
def test_boot_disk_size_configuration(gcloud_run, machine_arg):
    hailctl.main(["dataproc", "start", machine_arg, "test-cluster"])
    assert f"{machine_arg}GB" in gcloud_run.call_args[0][0]


def test_vep_defaults_to_highmem_master_machine(gcloud_run):
    hailctl.main(["dataproc", "start", "test-cluster", "--vep=GRCh37"])
    assert "--master-machine-type=n1-highmem-8" in gcloud_run.call_args[0][0]


def test_vep_defaults_to_larger_worker_boot_disk(gcloud_run):
    hailctl.main(["dataproc", "start", "test-cluster", "--vep=GRCh37"])
    assert "--worker-boot-disk-size=200GB" in gcloud_run.call_args[0][0]
    assert "--secondary-worker-boot-disk-size=200GB" in gcloud_run.call_args[0][0]


@pytest.mark.parametrize("requester_pays_arg", [
    "--requester-pays-allow-all",
    "--requester-pays-allow-buckets=example-bucket",
    "--requester-pays-allow-annotation-db",
])
def test_requester_pays_project_configuration(gcloud_run, gcloud_config, requester_pays_arg):
    gcloud_config["project"] = "foo-project"

    hailctl.main(["dataproc", "start", "test-cluster", requester_pays_arg])
    properties = next(arg for arg in gcloud_run.call_args[0][0] if arg.startswith("--properties="))
    assert "spark:spark.hadoop.fs.gs.requester.pays.project.id=foo-project" in properties

    hailctl.main(["dataproc", "--project=bar-project", "start", "test-cluster", requester_pays_arg])
    properties = next(arg for arg in gcloud_run.call_args[0][0] if arg.startswith("--properties="))
    assert "spark:spark.hadoop.fs.gs.requester.pays.project.id=bar-project" in properties


@pytest.mark.parametrize("requester_pays_arg,expected_mode", [
    ("--requester-pays-allow-all", "AUTO"),
    ("--requester-pays-allow-buckets=example-bucket", "CUSTOM"),
    ("--requester-pays-allow-annotation-db", "CUSTOM"),
])
def test_requester_pays_mode_configuration(gcloud_run, gcloud_config, requester_pays_arg, expected_mode):
    hailctl.main(["dataproc", "start", "test-cluster", requester_pays_arg])
    properties = next(arg for arg in gcloud_run.call_args[0][0] if arg.startswith("--properties="))
    assert f"spark:spark.hadoop.fs.gs.requester.pays.mode={expected_mode}" in properties


def test_requester_pays_buckets_configuration(gcloud_run, gcloud_config):
    hailctl.main(["dataproc", "start", "test-cluster", "--requester-pays-allow-buckets=foo,bar"])
    properties = next(arg for arg in gcloud_run.call_args[0][0] if arg.startswith("--properties="))
    assert f"spark:spark.hadoop.fs.gs.requester.pays.buckets=foo,bar" in properties


@pytest.mark.parametrize("scheduled_deletion_arg", [
    "--max-idle=30m",
    "--max-age=1h",
])
def test_scheduled_deletion_configuration(gcloud_run, scheduled_deletion_arg):
    hailctl.main(["dataproc", "start", scheduled_deletion_arg, "test-cluster"])
    assert scheduled_deletion_arg in gcloud_run.call_args[0][0]


def test_master_tags(gcloud_run):
    hailctl.main(["dataproc", "start", "test-cluster", "--master-tags=foo"])
    assert gcloud_run.call_count == 2
    assert gcloud_run.call_args_list[0][0][0][:7] == ["gcloud", "--project=hailctl-dataproc-tests", "dataproc", "--region=us-central1", "clusters", "create", "test-cluster"]
    assert gcloud_run.call_args_list[1][0][0] == ["gcloud", "--project=hailctl-dataproc-tests", "compute", "--zone=us-central1-b", "instances", "add-tags", "test-cluster-m", "--tags", "foo"]


def test_master_tags_project(gcloud_run):
    hailctl.main(["dataproc", "--project=some-project", "start", "test-cluster", "--master-tags=foo"])
    assert gcloud_run.call_count == 2
    assert "--project=some-project" in gcloud_run.call_args_list[1][0][0]


def test_master_tags_zone(gcloud_run):
    hailctl.main(["dataproc", "--zone=us-east1-d", "start", "test-cluster", "--master-tags=foo"])
    assert gcloud_run.call_count == 2
    assert "--zone=us-east1-d" in gcloud_run.call_args_list[1][0][0]


def test_master_tags_dry_run(gcloud_run):
    hailctl.main(["dataproc", "--dry-run", "start", "test-cluster", "--master-tags=foo"])
    assert gcloud_run.call_count == 0
