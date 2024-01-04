from unittest.mock import Mock
from typer.testing import CliRunner

from hailtop.hailctl.dataproc import cli


runner = CliRunner(mix_stderr=False)


def test_required_gcloud_version_met(gcloud_run, monkeypatch):
    monkeypatch.setattr(
        "hailtop.hailctl.dataproc.gcloud.get_version", Mock(return_value=cli.MINIMUM_REQUIRED_GCLOUD_VERSION)
    )

    runner.invoke(cli.app, ['list'])
    assert gcloud_run.call_count == 1


def test_required_gcloud_version_unmet(gcloud_run, monkeypatch):
    monkeypatch.setattr("hailtop.hailctl.dataproc.gcloud.get_version", Mock(return_value=(200, 0, 0)))

    res = runner.invoke(cli.app, ['list'])
    assert res.exit_code == 1
    assert res.exception
    assert "hailctl dataproc requires Google Cloud SDK (gcloud) version" in res.exception.args[0]

    assert gcloud_run.call_count == 0


def test_unable_to_determine_version(gcloud_run, monkeypatch):
    monkeypatch.setattr("hailtop.hailctl.dataproc.gcloud.get_version", Mock(side_effect=ValueError))

    runner.invoke(cli.app, ['list'])
    assert gcloud_run.call_count == 1
