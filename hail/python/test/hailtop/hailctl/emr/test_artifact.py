import json

import pytest

from hailtop.hailctl.emr.artifact import HailArtifactManifest, load_artifact_manifest, major_minor


def valid_manifest_dict():
    return {
        'schema_version': 1,
        'hail_git_revision': 'a' * 40,
        'hail_pip_version': '0.2.140',
        'spark_version': '4.1.2',
        'scala_version': '2.13.18',
        'python_version': '3.12',
        'wheel_uri': 's3://artifact-bucket/hail.whl',
        'wheel_sha256': 'b' * 64,
        'wheelhouse_uri': 's3://artifact-bucket/wheelhouse.tar.gz',
        'wheelhouse_sha256': 'c' * 64,
    }


def test_load_artifact_manifest(tmp_path):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(valid_manifest_dict()))
    manifest = load_artifact_manifest(str(path))
    assert manifest.spark_version == '4.1.2'
    assert manifest.bootstrap_args() == [
        's3://artifact-bucket/hail.whl',
        'b' * 64,
        's3://artifact-bucket/wheelhouse.tar.gz',
        'c' * 64,
    ]


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('schema_version', 2),
        ('hail_git_revision', 'short'),
        ('wheel_sha256', 'short'),
        ('wheelhouse_sha256', 'short'),
        ('wheel_uri', 'https://example/hail.whl'),
        ('wheelhouse_uri', 'gs://bucket/wheelhouse.tar.gz'),
        ('spark_version', 'not-a-version'),
    ],
)
def test_manifest_rejects_invalid_fields(field, value):
    data = valid_manifest_dict()
    data[field] = value
    with pytest.raises(ValueError):
        HailArtifactManifest.from_dict(data)


def test_manifest_rejects_missing_and_unknown_fields():
    data = valid_manifest_dict()
    del data['scala_version']
    with pytest.raises(ValueError, match='missing field'):
        HailArtifactManifest.from_dict(data)
    data = valid_manifest_dict()
    data['unexpected'] = 'value'
    with pytest.raises(ValueError, match='unknown field'):
        HailArtifactManifest.from_dict(data)


def test_major_minor_ignores_patch_and_vendor_suffix():
    assert major_minor('4.1.2') == '4.1'
    assert major_minor('4.1.1-amzn-0') == '4.1'
