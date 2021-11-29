from unittest.mock import Mock

import pytest

from hailtop.hailctl.dataproc import cli
from hailtop.hailctl.dataproc import list_clusters


def test_required_gcloud_version_met(monkeypatch):
    monkeypatch.setattr("hailtop.hailctl.dataproc.gcloud.get_version", Mock(return_value=cli.MINIMUM_REQUIRED_GCLOUD_VERSION))

    mock_list = Mock()
    async def async_return_mock_list(*args, **kwargs):
        return mock_list(*args, **kwargs)
    monkeypatch.setattr(list_clusters, "main", async_return_mock_list)
    cli.main(["list"])

    assert mock_list.called


def test_required_gcloud_version_unmet(monkeypatch, capsys):
    monkeypatch.setattr("hailtop.hailctl.dataproc.gcloud.get_version", Mock(return_value=(200, 0, 0)))

    mock_list = Mock()
    async def async_return_mock_list(*args, **kwargs):
        return mock_list(*args, **kwargs)
    monkeypatch.setattr(list_clusters, "main", async_return_mock_list)
    with pytest.raises(SystemExit):
        cli.main(["list"])

    assert "hailctl dataproc requires Google Cloud SDK (gcloud) version" in capsys.readouterr().err

    assert not mock_list.called


def test_unable_to_determine_version(monkeypatch):
    monkeypatch.setattr("hailtop.hailctl.dataproc.gcloud.get_version", Mock(side_effect=ValueError))

    mock_list = Mock()
    async def async_return_mock_list(*args, **kwargs):
        return mock_list(*args, **kwargs)
    monkeypatch.setattr(list_clusters, "main", async_return_mock_list)
    cli.main(["list"])

    assert mock_list.called
