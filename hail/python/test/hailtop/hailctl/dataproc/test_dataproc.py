from unittest.mock import Mock

import pytest

from hailtop import hailctl
from hailtop.hailctl.dataproc.dataproc import MINIMUM_REQUIRED_GCLOUD_VERSION


def test_required_gcloud_version_met(monkeypatch):
    monkeypatch.setattr("hailtop.hailctl.dataproc.gcloud.get_version", Mock(return_value=MINIMUM_REQUIRED_GCLOUD_VERSION))

    mock_list = Mock()
    monkeypatch.setattr(hailctl.dataproc.list_clusters.list_clusters, "callback", mock_list)
    hailctl.main(["dataproc", "list"])

    assert mock_list.called


def test_required_gcloud_version_unmet(monkeypatch, capsys):
    monkeypatch.setattr("hailtop.hailctl.dataproc.gcloud.get_version", Mock(return_value=(200, 0, 0)))

    mock_list = Mock()
    monkeypatch.setattr(hailctl.dataproc.list_clusters.list_clusters, "callback", mock_list)
    with pytest.raises(SystemExit):
        hailctl.main(["dataproc", "list"])

    assert "hailctl dataproc requires Google Cloud SDK (gcloud) version" in capsys.readouterr().err

    assert not mock_list.called


def test_unable_to_determine_version(monkeypatch):
    monkeypatch.setattr("hailtop.hailctl.dataproc.gcloud.get_version", Mock(side_effect=ValueError))

    mock_list = Mock()
    monkeypatch.setattr(hailctl.dataproc.list_clusters.list_clusters, "callback", mock_list)
    hailctl.main(["dataproc", "list"])

    assert mock_list.called
