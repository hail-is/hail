import json
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from hailtop.hailctl.dataproc import cli

runner = CliRunner()

JUPYTERLAB_URL = "https://abc123-dot-us-central1.dataproc.googleusercontent.com/gateway/default/jupyter/lab/"
SPARK_HISTORY_URL = "https://abc123-dot-us-central1.dataproc.googleusercontent.com/sparkhistory/"

DESCRIBE_OUTPUT = {
    "config": {
        "endpointConfig": {
            "httpPorts": {
                "JupyterLab": JUPYTERLAB_URL,
                "Spark History Server": SPARK_HISTORY_URL,
            },
        },
    },
}


@pytest.fixture
def webbrowser():
    return Mock()


@pytest.fixture(autouse=True)
def patch_webbrowser(monkeypatch, webbrowser, gcloud_output):
    """Automatically mock the webbrowser module and the gateway url lookup."""
    monkeypatch.setattr("hailtop.hailctl.dataproc.connect.webbrowser", webbrowser)
    gcloud_output.return_value = json.dumps(DESCRIBE_OUTPUT)
    yield
    monkeypatch.undo()


def test_cluster_and_service_required(gcloud_run):
    res = runner.invoke(cli.app, ['connect'])
    assert res.exit_code == 2
    assert gcloud_run.call_count == 0

    res = runner.invoke(cli.app, ['connect', 'notebook'])
    assert res.exit_code == 2
    assert gcloud_run.call_count == 0


def test_dry_run(gcloud_output, webbrowser):
    res = runner.invoke(cli.app, ['connect', 'test-cluster', 'notebook', '--dry-run'])
    assert res.exit_code == 0
    assert gcloud_output.call_count == 0
    assert webbrowser.open.call_count == 0


@pytest.mark.parametrize("service", ["notebook", "nb", "spark-ui", "ui", "spark-history", "hist"])
def test_connect_describes_cluster(gcloud_output, service):
    runner.invoke(cli.app, ['connect', 'test-cluster', service])

    gcloud_args = gcloud_output.call_args[0][0]
    assert gcloud_args[:4] == ["dataproc", "clusters", "describe", "test-cluster"]
    assert "--format=json(config.endpointConfig.httpPorts)" in gcloud_args


@pytest.mark.parametrize(
    "service,expected_url",
    [
        ("notebook", JUPYTERLAB_URL),
        ("nb", JUPYTERLAB_URL),
        ("spark-ui", SPARK_HISTORY_URL + "?showIncomplete=true"),
        ("ui", SPARK_HISTORY_URL + "?showIncomplete=true"),
        ("spark-history", SPARK_HISTORY_URL),
        ("hist", SPARK_HISTORY_URL),
    ],
)
def test_connect_opens_gateway_url(webbrowser, service, expected_url):
    runner.invoke(cli.app, ['connect', 'test-cluster', service])
    webbrowser.open.assert_called_once_with(expected_url)


def test_connect_region_from_config(gcloud_output, gcloud_config):
    gcloud_config["dataproc/region"] = "europe-north1"
    gcloud_config["compute/zone"] = None

    runner.invoke(cli.app, ['connect', 'test-cluster', 'notebook'])
    assert "--region=europe-north1" in gcloud_output.call_args[0][0]


def test_connect_region_from_zone(gcloud_output, gcloud_config):
    gcloud_config["dataproc/region"] = None

    runner.invoke(cli.app, ['connect', 'test-cluster', 'notebook', '--zone=us-east1-d'])
    assert "--region=us-east1" in gcloud_output.call_args[0][0]


def test_connect_region_required(gcloud_output, gcloud_config):
    gcloud_config["dataproc/region"] = None
    gcloud_config["compute/zone"] = None

    res = runner.invoke(cli.app, ['connect', 'test-cluster', 'notebook'])
    assert res.exit_code == 1
    assert gcloud_output.call_count == 0


def test_connect_gateway_not_enabled(gcloud_output, webbrowser):
    gcloud_output.return_value = "{}"

    res = runner.invoke(cli.app, ['connect', 'test-cluster', 'notebook'])
    assert res.exit_code == 1
    assert webbrowser.open.call_count == 0


def test_connect_project(gcloud_output):
    runner.invoke(cli.app, ['connect', 'test-cluster', 'notebook', '--project=test-project'])
    assert "--project=test-project" in gcloud_output.call_args[0][0]
