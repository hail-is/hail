import pytest
from typer.testing import CliRunner

from hailtop.hailctl.dataproc import cli

from ..helpers import fails_test


runner = CliRunner(mix_stderr=False)


def test_cluster_name_required(gcloud_run):
    res = runner.invoke(cli.app, ['start'])
    assert "Missing argument 'NAME'" in res.stderr
    assert gcloud_run.call_count == 0


@fails_test
def test_dry_run(gcloud_run):
    res = runner.invoke(cli.app, ['start', 'test-cluster', '--dry-run'])
    assert res.exit_code == 0
    assert gcloud_run.call_count == 0


@fails_test
def test_cluster_project(gcloud_run):
    runner.invoke(cli.app, ['start', '--project', 'foo', 'test-cluster'])
    assert "--project=foo" in gcloud_run.call_args[0][0]


@pytest.mark.parametrize("location_arg", [
    "--region=europe-north1",
    "--zone=us-central1-b",
])
@fails_test
def test_cluster_location(gcloud_run, location_arg):
    runner.invoke(cli.app, ['start', location_arg, 'test-cluster'])
    assert location_arg in gcloud_run.call_args[0][0]


@fails_test
def test_creator_label(gcloud_run, gcloud_config):
    gcloud_config["account"] = "test-user@hail.is"
    runner.invoke(cli.app, ['start', 'my-cluster'])
    assert "--labels=creator=test-user_hail_is" in gcloud_run.call_args[0][0]

    gcloud_config["account"] = None
    runner.invoke(cli.app, ['start', 'my-cluster'])
    assert not any(arg.startswith("--labels=") and "creator=" in arg for arg in gcloud_run.call_args[0][0])


@fails_test
def test_workers_configuration(gcloud_run):
    runner.invoke(cli.app, ['start', '--num-workers=4', 'test-cluster'])
    assert "--num-workers=4" in gcloud_run.call_args[0][0]


@pytest.mark.parametrize("workers_arg", [
    "--num-secondary-workers=8",
    "--num-preemptible-workers=8"
])
@fails_test
def test_secondary_workers_configuration(gcloud_run, workers_arg):
    runner.invoke(cli.app, ['start', workers_arg, 'test-cluster'])
    assert "--num-secondary-workers=8" in gcloud_run.call_args[0][0]


@pytest.mark.parametrize("machine_arg", [
    "--master-machine-type=n1-highmem-16",
    "--worker-machine-type=n1-standard-32",
])
@fails_test
def test_machine_type_configuration(gcloud_run, machine_arg):
    runner.invoke(cli.app, ['start', machine_arg, 'test-cluster'])
    assert machine_arg in gcloud_run.call_args[0][0]


@pytest.mark.parametrize("machine_arg", [
    "--master-boot-disk-size=250",
    "--worker-boot-disk-size=200",
    "--secondary-worker-boot-disk-size=100"
])
@fails_test
def test_boot_disk_size_configuration(gcloud_run, machine_arg):
    runner.invoke(cli.app, ['start', machine_arg, 'test-cluster'])
    assert f"{machine_arg}GB" in gcloud_run.call_args[0][0]


@fails_test
def test_vep_defaults_to_highmem_master_machine(gcloud_run):
    runner.invoke(cli.app, ['start', 'test-cluster', '--vep=GRCh37'])
    assert "--master-machine-type=n1-highmem-8" in gcloud_run.call_args[0][0]


@fails_test
def test_vep_defaults_to_larger_worker_boot_disk(gcloud_run):
    runner.invoke(cli.app, ['start', 'test-cluster', '--vep=GRCh37'])
    assert "--worker-boot-disk-size=200GB" in gcloud_run.call_args[0][0]
    assert "--secondary-worker-boot-disk-size=200GB" in gcloud_run.call_args[0][0]


@pytest.mark.parametrize("requester_pays_arg", [
    "--requester-pays-allow-all",
    "--requester-pays-allow-buckets=example-bucket",
    "--requester-pays-allow-annotation-db",
])
@fails_test
def test_requester_pays_project_configuration(gcloud_run, gcloud_config, requester_pays_arg):
    gcloud_config["project"] = "foo-project"

    runner.invoke(cli.app, ['start', 'test-cluster', requester_pays_arg])
    properties = next(arg for arg in gcloud_run.call_args[0][0] if arg.startswith("--properties="))
    assert "spark:spark.hadoop.fs.gs.requester.pays.project.id=foo-project" in properties

    runner.invoke(cli.app, ['start', '--project=bar-project', 'test-cluster', requester_pays_arg])
    properties = next(arg for arg in gcloud_run.call_args[0][0] if arg.startswith("--properties="))
    assert "spark:spark.hadoop.fs.gs.requester.pays.project.id=bar-project" in properties


@pytest.mark.parametrize("requester_pays_arg,expected_mode", [
    ("--requester-pays-allow-all", "AUTO"),
    ("--requester-pays-allow-buckets=example-bucket", "CUSTOM"),
    ("--requester-pays-allow-annotation-db", "CUSTOM"),
])
@fails_test
def test_requester_pays_mode_configuration(gcloud_run, requester_pays_arg, expected_mode):
    runner.invoke(cli.app, ['start', 'test-cluster', requester_pays_arg])
    properties = next(arg for arg in gcloud_run.call_args[0][0] if arg.startswith("--properties="))
    assert f"spark:spark.hadoop.fs.gs.requester.pays.mode={expected_mode}" in properties


@fails_test
def test_requester_pays_buckets_configuration(gcloud_run):
    runner.invoke(cli.app, ['start', 'test-cluster', '--requester-pays-allow-buckets=foo,bar'])
    properties = next(arg for arg in gcloud_run.call_args[0][0] if arg.startswith("--properties="))
    assert f"spark:spark.hadoop.fs.gs.requester.pays.buckets=foo,bar" in properties


@pytest.mark.parametrize("scheduled_deletion_arg", [
    "--max-idle=30m",
    "--max-age=1h",
])
@fails_test
def test_scheduled_deletion_configuration(gcloud_run, scheduled_deletion_arg):
    runner.invoke(cli.app, ['start', scheduled_deletion_arg, 'test-cluster'])
    assert scheduled_deletion_arg in gcloud_run.call_args[0][0]


@fails_test
def test_master_tags(gcloud_run):
    runner.invoke(cli.app, ['start', 'test-cluster', '--master-tags=foo'])
    assert gcloud_run.call_count == 2
    assert gcloud_run.call_args_list[0][0][0][:4] == ["dataproc", "clusters", "create", "test-cluster"]
    assert gcloud_run.call_args_list[1][0][0] == ["compute", "instances", "add-tags", "test-cluster-m", "--tags", "foo"]


@fails_test
def test_master_tags_project(gcloud_run):
    runner.invoke(cli.app, ['start', 'test-cluster', '--master-tags=foo', '--project=some-project'])
    assert gcloud_run.call_count == 2
    assert "--project=some-project" in gcloud_run.call_args_list[1][0][0]


@fails_test
def test_master_tags_zone(gcloud_run):
    runner.invoke(cli.app, ['start', 'test-cluster', '--master-tags=foo', '--zone=us-east1-d'])
    assert gcloud_run.call_count == 2
    assert "--zone=us-east1-d" in gcloud_run.call_args_list[1][0][0]


def test_master_tags_dry_run(gcloud_run):
    runner.invoke(cli.app, ['start', 'test-cluster', '--master-tags=foo', '--dry-run'])
    assert gcloud_run.call_count == 0
