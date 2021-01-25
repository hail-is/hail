import pytest

from hailtop import hailctl


def test_stop(gcloud_run):
    hailctl.main(["dataproc", "modify", "test-cluster", "--extra-gcloud-update-args=--num-workers=2"])
    assert gcloud_run.call_args[0][0] == ["gcloud", "--project=hailctl-dataproc-tests", "dataproc", "--region=us-central1", "clusters", "update", "test-cluster", "--num-workers=2"]


def test_beta(gcloud_run):
    hailctl.main(["dataproc", "--beta", "modify", "test-cluster", "--extra-gcloud-update-args=--num-workers=2"])
    assert gcloud_run.call_args[0][0] == ["gcloud", "--project=hailctl-dataproc-tests", "beta", "dataproc", "--region=us-central1", "clusters", "update", "test-cluster", "--num-workers=2"]


def test_cluster_name_required(capsys, gcloud_run):
    with pytest.raises(SystemExit):
        hailctl.main(["dataproc", "modify"])

    assert "Missing argument 'CLUSTER_NAME'" in capsys.readouterr().err
    assert gcloud_run.call_count == 0


def test_cluster_project(gcloud_run):
    hailctl.main(["dataproc", "--project=foo", "modify", "test-cluster", "--extra-gcloud-update-args=--num-workers=2"])
    assert "--project=foo" in gcloud_run.call_args[0][0]


def test_cluster_region(gcloud_run):
    hailctl.main(["dataproc", "--zone=europe-north1-a", "modify", "test-cluster", "--extra-gcloud-update-args=--num-workers=2"])
    assert "--region=europe-north1" in gcloud_run.call_args[0][0]


def test_modify_dry_run(gcloud_run):
    hailctl.main(["dataproc", "--dry-run", "modify", "test-cluster", "--extra-gcloud-update-args=--num-workers=2"])
    assert gcloud_run.call_count == 0


def test_modify_wheel_remote_wheel(gcloud_run):
    hailctl.main(["dataproc", "modify", "test-cluster", "--wheel=gs://some-bucket/hail.whl"])
    assert gcloud_run.call_count == 1
    gcloud_args = gcloud_run.call_args[0][0]
    assert gcloud_args[:5] == ["gcloud", "--project=hailctl-dataproc-tests", "compute", "--zone=us-central1-b", "ssh"]

    remote_command = gcloud_args[gcloud_args.index("--") + 1]
    assert remote_command == ("sudo gsutil cp gs://some-bucket/hail.whl /tmp/ && " +
        "sudo /opt/conda/default/bin/pip uninstall -y hail && " +
        "sudo /opt/conda/default/bin/pip install --no-dependencies /tmp/hail.whl && " +
        "unzip /tmp/hail.whl && " +
        "grep 'Requires-Dist: ' hail*dist-info/METADATA | sed 's/Requires-Dist: //' | sed 's/ (//' | sed 's/)//' | grep -v 'pyspark' | xargs /opt/conda/default/bin/pip install")


def test_modify_wheel_local_wheel(gcloud_run):
    hailctl.main(["dataproc", "modify", "test-cluster", "--wheel=./local-hail.whl"])
    assert gcloud_run.call_count == 2

    copy_gcloud_args = gcloud_run.call_args_list[0][0][0]
    assert copy_gcloud_args[:5] == ["gcloud", "--project=hailctl-dataproc-tests", "compute", "--zone=us-central1-b", "scp"]
    assert copy_gcloud_args[-2:] == ["./local-hail.whl", "test-cluster-m:/tmp/"]

    install_gcloud_args = gcloud_run.call_args_list[1][0][0]
    assert install_gcloud_args[:6] == ["gcloud", "--project=hailctl-dataproc-tests", "compute", "--zone=us-central1-b", "ssh", "test-cluster-m"]

    remote_command = install_gcloud_args[install_gcloud_args.index("--") + 1]
    assert remote_command == ("sudo /opt/conda/default/bin/pip uninstall -y hail && " +
        "sudo /opt/conda/default/bin/pip install --no-dependencies /tmp/local-hail.whl && " +
        "unzip /tmp/local-hail.whl && " +
        "grep 'Requires-Dist: ' hail*dist-info/METADATA | sed 's/Requires-Dist: //' | sed 's/ (//' | sed 's/)//' | grep -v 'pyspark' | xargs /opt/conda/default/bin/pip install")


@pytest.mark.parametrize("wheel_arg", [
    "--wheel=gs://some-bucket/hail.whl",
    "--wheel=./hail.whl",
])
def test_modify_wheel_zone(gcloud_run, gcloud_config, wheel_arg):
    gcloud_config["compute/zone"] = "us-central1-b"

    hailctl.main(["dataproc", "--zone=us-east1-d", "modify", "test-cluster", wheel_arg])
    for call_args in gcloud_run.call_args_list:
        assert "--zone=us-east1-d" in call_args[0][0]


@pytest.mark.parametrize("wheel_arg", [
    "--wheel=gs://some-bucket/hail.whl",
    "--wheel=./hail.whl",
])
def test_modify_wheel_default_zone(gcloud_run, gcloud_config, wheel_arg):
    gcloud_config["compute/zone"] = "us-central1-b"

    hailctl.main(["dataproc", "modify", "test-cluster", wheel_arg])
    for call_args in gcloud_run.call_args_list:
        assert "--zone=us-central1-b" in call_args[0][0]


@pytest.mark.parametrize("wheel_arg", [
    "--wheel=gs://some-bucket/hail.whl",
    "--wheel=./hail.whl",
])
def test_modify_wheel_zone_required(gcloud_run, gcloud_config, wheel_arg):
    gcloud_config["compute/zone"] = None

    with pytest.raises(Exception):
        hailctl.main(["dataproc", "modify", "test-cluster", wheel_arg])
        assert gcloud_run.call_count == 0


@pytest.mark.parametrize("wheel_arg", [
    "--wheel=gs://some-bucket/hail.whl",
    "--wheel=./hail.whl",
])
def test_modify_wheel_dry_run(gcloud_run, wheel_arg):
    hailctl.main(["dataproc", "--dry-run", "modify", "test-cluster", wheel_arg])
    assert gcloud_run.call_count == 0


def test_wheel_and_update_hail_version_mutually_exclusive(gcloud_run, capsys):
    with pytest.raises(SystemExit):
        hailctl.main(["dataproc", "modify", "test-cluster", "--wheel=./hail.whl", "--update-hail-version"])

    assert gcloud_run.call_count == 0
    assert "at most one of --wheel and --update-hail-version allowed" in capsys.readouterr().err


def test_update_hail_version(gcloud_run, monkeypatch, deploy_metadata):
    monkeypatch.setattr("hailtop.hailctl.dataproc.modify.get_deploy_metadata", lambda: deploy_metadata)

    hailctl.main(["dataproc", "modify", "test-cluster", "--update-hail-version"])
    assert gcloud_run.call_count == 1
    gcloud_args = gcloud_run.call_args[0][0]
    assert gcloud_args[:6] == ["gcloud", "--project=hailctl-dataproc-tests", "compute", "--zone=us-central1-b", "ssh", "test-cluster-m"]

    remote_command = gcloud_args[gcloud_args.index("--") + 1]
    assert remote_command == (
        "sudo gsutil cp gs://hail-common/hailctl/dataproc/test-version/hail-test-version-py3-none-any.whl /tmp/ && " +
        "sudo /opt/conda/default/bin/pip uninstall -y hail && " +
        "sudo /opt/conda/default/bin/pip install --no-dependencies /tmp/hail-test-version-py3-none-any.whl && " +
        "unzip /tmp/hail-test-version-py3-none-any.whl && " +
        "grep 'Requires-Dist: ' hail*dist-info/METADATA | sed 's/Requires-Dist: //' | sed 's/ (//' | sed 's/)//' | grep -v 'pyspark' | xargs /opt/conda/default/bin/pip install"
    )
