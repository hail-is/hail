from unittest.mock import Mock

import pytest

from hailtop.hailctl.dataproc import cli


@pytest.fixture
def subprocess():
    return Mock()


@pytest.fixture(autouse=True)
def patch_subprocess(monkeypatch, subprocess):
    """Automatically mock subprocess module."""
    monkeypatch.setattr("hailtop.hailctl.dataproc.connect.subprocess", subprocess)
    monkeypatch.setattr(
        "hailtop.hailctl.dataproc.connect.get_chrome_path",
        Mock(return_value="chromium")
    )
    yield
    monkeypatch.undo()


def test_cluster_and_service_required(gcloud_run):
    with pytest.raises(SystemExit):
        cli.main(["connect"])

    assert gcloud_run.call_count == 0

    with pytest.raises(SystemExit):
        cli.main(["connect", "notebook"])

    assert gcloud_run.call_count == 0


def test_dry_run(gcloud_run, subprocess):
    cli.main(["connect", "test-cluster", "notebook", "--dry-run"])
    assert gcloud_run.call_count == 0
    assert subprocess.Popen.call_count == 0


def test_connect(gcloud_run, subprocess):
    cli.main(["connect", "test-cluster", "notebook"])

    gcloud_args = gcloud_run.call_args[0][0]
    assert gcloud_args[:2] == ["compute", "ssh"]
    assert gcloud_args[2][(gcloud_args[2].find("@") + 1):] == "test-cluster-m"

    assert "--ssh-flag=-D 10000" in gcloud_args
    assert "--ssh-flag=-N" in gcloud_args
    assert "--ssh-flag=-f" in gcloud_args
    assert "--ssh-flag=-n" in gcloud_args

    popen_args = subprocess.Popen.call_args[0][0]
    assert popen_args[0] == "chromium"
    assert popen_args[1].startswith("http://localhost")

    assert "--proxy-server=socks5://localhost:10000" in popen_args
    assert "--host-resolver-rules=MAP * 0.0.0.0 , EXCLUDE localhost" in popen_args
    assert "--proxy-bypass-list=<-loopback>" in popen_args
    assert any(arg.startswith("--user-data-dir=") for arg in popen_args)


@pytest.mark.parametrize("service,expected_port_and_path", [
    ("spark-ui", "18080/?showIncomplete=true"),
    ("ui", "18080/?showIncomplete=true"),
    ("spark-history", "18080"),
    ("hist", "18080"),
    ("notebook", "8123"),
    ("nb", "8123"),
])
def test_service_port_and_path(gcloud_run, subprocess, service, expected_port_and_path):
    cli.main(["connect", "test-cluster", service])

    popen_args = subprocess.Popen.call_args[0][0]
    assert popen_args[1] == f"http://localhost:{expected_port_and_path}"


def test_hailctl_chrome(subprocess, monkeypatch):
    monkeypatch.setattr(
        "hailtop.hailctl.dataproc.connect.get_chrome_path",
        Mock(side_effect=Exception("Unable to find chrome"))
    )
    monkeypatch.setenv("HAILCTL_CHROME", "/path/to/chrome.exe")

    cli.main(["connect", "test-cluster", "notebook"])
    popen_args = subprocess.Popen.call_args[0][0]
    assert popen_args[0] == "/path/to/chrome.exe"


def test_port(gcloud_run):
    cli.main(["connect", "test-cluster", "notebook", "--port=8000"])
    assert "--ssh-flag=-D 8000" in gcloud_run.call_args[0][0]


def test_connect_zone(gcloud_run, gcloud_config):
    gcloud_config["compute/zone"] = "us-central1-b"

    cli.main(["connect", "test-cluster", "notebook", "--zone=us-east1-d"])

    assert "--zone=us-east1-d" in gcloud_run.call_args[0][0]


def test_connect_default_zone(gcloud_run, gcloud_config):
    gcloud_config["compute/zone"] = "us-west1-a"

    cli.main(["connect", "test-cluster", "notebook"])

    assert "--zone=us-west1-a" in gcloud_run.call_args[0][0]


def test_connect_zone_required(gcloud_run, gcloud_config):
    gcloud_config["compute/zone"] = None

    with pytest.raises(Exception):
        cli.main(["connect", "test-cluster", "notebook"])

    assert gcloud_run.call_count == 0


def test_connect_project(gcloud_run):
    cli.main(["connect", "test-cluster", "notebook", "--project=test-project"])

    assert "--project=test-project" in gcloud_run.call_args[0][0]
