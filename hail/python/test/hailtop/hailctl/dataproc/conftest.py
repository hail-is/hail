from unittest.mock import Mock

import pytest


@pytest.fixture
def gcloud_config(request):
    """Fixture for gcloud configuration values."""
    return {
        "account": "test@hail.is",
        "project": "hailctl-dataproc-tests",
        "dataproc/region": "us-central1",
        "compute/zone": "us-central1-b",
    }


@pytest.fixture
def gcloud_run():
    return Mock()


@pytest.fixture(autouse=True)
def patch_gcloud(monkeypatch, gcloud_run, gcloud_config):
    """Automatically replace gcloud functions with mocks."""
    monkeypatch.setattr("hailtop.hailctl.dataproc.gcloud.run", gcloud_run)

    def mock_gcloud_get_config(setting):
        return gcloud_config.get(setting, None)

    monkeypatch.setattr("hailtop.hailctl.dataproc.gcloud.get_config", mock_gcloud_get_config)

    yield

    monkeypatch.undo()
