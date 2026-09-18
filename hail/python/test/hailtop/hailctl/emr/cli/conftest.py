from unittest.mock import Mock

import pytest

from hailtop.hailctl.emr.artifact import HailArtifactManifest


@pytest.fixture
def artifact_manifest():
    return HailArtifactManifest(
        schema_version=1,
        hail_git_revision='a' * 40,
        hail_pip_version='0.2.140',
        spark_version='4.1.2',
        scala_version='2.13.18',
        python_version='3.12',
        wheel_uri='s3://artifact-bucket/hail.whl',
        wheel_sha256='b' * 64,
        wheelhouse_uri='s3://artifact-bucket/wheelhouse.tar.gz',
        wheelhouse_sha256='c' * 64,
    )


@pytest.fixture
def emr_client_mock():
    client = Mock()
    client.describe_release_label.return_value = {
        'ReleaseLabel': 'emr-spark-8.1.0',
        'Applications': [{'Name': 'Spark', 'Version': '4.1.1'}],
    }
    return client


@pytest.fixture
def upload_mock():
    return Mock()


@pytest.fixture
def check_default_roles_mock():
    return Mock()


@pytest.fixture
def check_private_subnet_mock():
    return Mock()


@pytest.fixture
def check_custom_roles_mock():
    return Mock()


@pytest.fixture(autouse=True)
def patch_aws(
    monkeypatch,
    artifact_manifest,
    emr_client_mock,
    upload_mock,
    check_default_roles_mock,
    check_private_subnet_mock,
    check_custom_roles_mock,
):
    monkeypatch.setattr(
        'hailtop.hailctl.emr.artifact.load_artifact_manifest',
        lambda path: artifact_manifest,
    )
    monkeypatch.setattr('hailtop.hailctl.emr.emr.emr_client', lambda region: emr_client_mock)
    monkeypatch.setattr('hailtop.hailctl.emr.emr.upload_to_s3', upload_mock)
    monkeypatch.setattr('hailtop.hailctl.emr.emr.resolve_region', lambda region: 'us-east-1')
    monkeypatch.setattr('hailtop.hailctl.emr.emr.check_default_roles', check_default_roles_mock)
    monkeypatch.setattr('hailtop.hailctl.emr.emr.check_custom_roles', check_custom_roles_mock)
    monkeypatch.setattr('hailtop.hailctl.emr.emr.check_private_subnet', check_private_subnet_mock)
    yield
    monkeypatch.undo()
