import pytest
from typer.testing import CliRunner

from hailtop.hailctl.dataproc import cli


runner = CliRunner(mix_stderr=False)


def test_stop(gcloud_run):
    runner.invoke(cli.app, ['modify', 'test-cluster', '--num-workers=2'])
    assert gcloud_run.call_args[0][0][:3] == ["dataproc", "clusters", "update"]


def test_beta(gcloud_run):
    runner.invoke(cli.app, ['--beta', 'modify', 'test-cluster', '--num-workers=2'])
    assert gcloud_run.call_args[0][0][:4] == ["beta", "dataproc", "clusters", "update"]


def test_cluster_name_required(gcloud_run):
    res = runner.invoke(cli.app, ['modify'])
    assert "Missing argument 'NAME'" in res.stderr
    assert gcloud_run.call_count == 0


def test_cluster_project(gcloud_run):
    runner.invoke(cli.app, ['modify', '--project=foo', 'test-cluster', '--num-workers=2'])
    assert "--project=foo" in gcloud_run.call_args[0][0]


def test_cluster_region(gcloud_run):
    runner.invoke(cli.app, ['modify', '--region=europe-north1', 'test-cluster', '--num-workers=2'])
    assert "--region=europe-north1" in gcloud_run.call_args[0][0]


def test_modify_dry_run(gcloud_run):
    runner.invoke(cli.app, ['modify', 'test-cluster', '--num-workers=2', '--dry-run'])
    assert gcloud_run.call_count == 0


@pytest.mark.parametrize(
    "workers_arg",
    [
        "--num-workers=2",
        "--n-workers=2",
        "-w2",
    ],
)
def test_modify_workers(gcloud_run, workers_arg):
    runner.invoke(cli.app, ['modify', 'test-cluster', workers_arg])
    assert "--num-workers=2" in gcloud_run.call_args[0][0]


@pytest.mark.parametrize(
    "workers_arg",
    [
        "--num-secondary-workers=2",
        "--num-preemptible-workers=2",
        "--n-pre-workers=2",
        "-p2",
    ],
)
def test_modify_secondary_workers(gcloud_run, workers_arg):
    runner.invoke(cli.app, ['modify', 'test-cluster', workers_arg])
    assert "--num-secondary-workers=2" in gcloud_run.call_args[0][0]


def test_modify_max_idle(gcloud_run):
    runner.invoke(cli.app, ['modify', 'test-cluster', '--max-idle=1h'])
    assert "--max-idle=1h" in gcloud_run.call_args[0][0]


@pytest.mark.parametrize(
    "workers_arg",
    [
        "--num-workers=2",
        "--num-secondary-workers=2",
    ],
)
def test_graceful_decommission_timeout(gcloud_run, workers_arg):
    runner.invoke(cli.app, ['modify', 'test-cluster', workers_arg, '--graceful-decommission-timeout=1h'])
    assert workers_arg in gcloud_run.call_args[0][0]
    assert "--graceful-decommission-timeout=1h" in gcloud_run.call_args[0][0]


def test_graceful_decommission_timeout_no_resize(gcloud_run):
    res = runner.invoke(cli.app, ['modify', 'test-cluster', '--graceful-decommission-timeout=1h'])
    assert res.exit_code == 1
    assert gcloud_run.call_count == 0


def test_modify_wheel_remote_wheel(gcloud_run):
    runner.invoke(cli.app, ['modify', 'test-cluster', '--wheel=gs://some-bucket/hail.whl'])
    assert gcloud_run.call_count == 1
    gcloud_args = gcloud_run.call_args[0][0]
    assert gcloud_args[:3] == ["compute", "ssh", "test-cluster-m"]

    remote_command = gcloud_args[gcloud_args.index("--") + 1]
    assert remote_command == (
        "sudo gsutil cp gs://some-bucket/hail.whl /tmp/ && "
        + "sudo /opt/conda/default/bin/pip uninstall -y hail && "
        + "sudo /opt/conda/default/bin/pip install --no-dependencies /tmp/hail.whl && "
        + "unzip /tmp/hail.whl && "
        + "requirements_file=$(mktemp) && "
        + "grep 'Requires-Dist: ' hail*dist-info/METADATA | sed 's/Requires-Dist: //' | sed 's/ (//' | sed 's/)//' | grep -v 'pyspark' >$requirements_file &&"
        + "/opt/conda/default/bin/pip install -r $requirements_file"
    )


def test_modify_wheel_local_wheel(gcloud_run):
    runner.invoke(cli.app, ['modify', 'test-cluster', '--wheel=./local-hail.whl'])
    assert gcloud_run.call_count == 2

    copy_gcloud_args = gcloud_run.call_args_list[0][0][0]
    assert copy_gcloud_args[:2] == ["compute", "scp"]
    assert copy_gcloud_args[-2:] == ["./local-hail.whl", "test-cluster-m:/tmp/"]

    install_gcloud_args = gcloud_run.call_args_list[1][0][0]
    assert install_gcloud_args[:3] == ["compute", "ssh", "test-cluster-m"]

    remote_command = install_gcloud_args[install_gcloud_args.index("--") + 1]
    assert remote_command == (
        "sudo /opt/conda/default/bin/pip uninstall -y hail && "
        + "sudo /opt/conda/default/bin/pip install --no-dependencies /tmp/local-hail.whl && "
        + "unzip /tmp/local-hail.whl && "
        + "requirements_file=$(mktemp) && "
        + "grep 'Requires-Dist: ' hail*dist-info/METADATA | sed 's/Requires-Dist: //' | sed 's/ (//' | sed 's/)//' | grep -v 'pyspark' >$requirements_file &&"
        + "/opt/conda/default/bin/pip install -r $requirements_file"
    )


@pytest.mark.parametrize(
    "wheel_arg",
    [
        "--wheel=gs://some-bucket/hail.whl",
        "--wheel=./hail.whl",
    ],
)
def test_modify_wheel_zone(gcloud_run, gcloud_config, wheel_arg):
    gcloud_config["compute/zone"] = "us-central1-b"

    runner.invoke(cli.app, ['modify', 'test-cluster', wheel_arg, '--zone=us-east1-d'])
    for call_args in gcloud_run.call_args_list:
        assert "--zone=us-east1-d" in call_args[0][0]


@pytest.mark.parametrize(
    "wheel_arg",
    [
        "--wheel=gs://some-bucket/hail.whl",
        "--wheel=./hail.whl",
    ],
)
def test_modify_wheel_default_zone(gcloud_run, gcloud_config, wheel_arg):
    gcloud_config["compute/zone"] = "us-central1-b"

    runner.invoke(cli.app, ['modify', 'test-cluster', wheel_arg])
    for call_args in gcloud_run.call_args_list:
        assert "--zone=us-central1-b" in call_args[0][0]


@pytest.mark.parametrize(
    "wheel_arg",
    [
        "--wheel=gs://some-bucket/hail.whl",
        "--wheel=./hail.whl",
    ],
)
def test_modify_wheel_zone_required(gcloud_run, gcloud_config, wheel_arg):
    gcloud_config["compute/zone"] = None

    res = runner.invoke(cli.app, ['modify', 'test-cluster', wheel_arg])
    assert res.exit_code == 1
    assert gcloud_run.call_count == 0


@pytest.mark.parametrize(
    "wheel_arg",
    [
        "--wheel=gs://some-bucket/hail.whl",
        "--wheel=./hail.whl",
    ],
)
def test_modify_wheel_dry_run(gcloud_run, wheel_arg):
    runner.invoke(cli.app, ['modify', 'test-cluster', wheel_arg, '--dry-run'])
    assert gcloud_run.call_count == 0


def test_wheel_and_update_hail_version_mutually_exclusive(gcloud_run):
    res = runner.invoke(cli.app, ['modify', 'test-cluster', '--wheel=./hail.whl', '--update-hail-version'])
    assert res.exit_code == 1
    assert res.exception
    assert 'argument --update-hail-version: not allowed with argument --wheel' in res.exception.args[0]
    assert gcloud_run.call_count == 0


def test_update_hail_version(gcloud_run, monkeypatch, deploy_metadata):
    monkeypatch.setattr("hailtop.hailctl.dataproc.modify.get_deploy_metadata", lambda: deploy_metadata)

    runner.invoke(cli.app, ['modify', 'test-cluster', '--update-hail-version'])
    assert gcloud_run.call_count == 1
    gcloud_args = gcloud_run.call_args[0][0]
    assert gcloud_args[:3] == ["compute", "ssh", "test-cluster-m"]

    remote_command = gcloud_args[gcloud_args.index("--") + 1]
    assert remote_command == (
        "sudo gsutil cp gs://hail-common/hailctl/dataproc/test-version/hail-test-version-py3-none-any.whl /tmp/ && "
        + "sudo /opt/conda/default/bin/pip uninstall -y hail && "
        + "sudo /opt/conda/default/bin/pip install --no-dependencies /tmp/hail-test-version-py3-none-any.whl && "
        + "unzip /tmp/hail-test-version-py3-none-any.whl && "
        + "requirements_file=$(mktemp) && "
        + "grep 'Requires-Dist: ' hail*dist-info/METADATA | sed 's/Requires-Dist: //' | sed 's/ (//' | sed 's/)//' | grep -v 'pyspark' >$requirements_file &&"
        + "/opt/conda/default/bin/pip install -r $requirements_file"
    )
