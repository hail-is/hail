from unittest.mock import Mock

import pytest


@pytest.fixture
def emr_client_mock():
    return Mock()


@pytest.fixture
def upload_mock():
    # Replaces emr.upload_to_s3; records (dest_uri, data) calls without touching S3.
    return Mock()


@pytest.fixture(autouse=True)
def patch_aws(monkeypatch, emr_client_mock, upload_mock):
    monkeypatch.setattr('hailtop.hailctl.emr.emr.emr_client', lambda region: emr_client_mock)
    monkeypatch.setattr('hailtop.hailctl.emr.emr.upload_to_s3', upload_mock)
    monkeypatch.setattr('hailtop.hailctl.emr.emr.resolve_region', lambda region: 'us-east-1')
    yield
    monkeypatch.undo()
