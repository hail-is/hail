from unittest.mock import patch

from hailtop.hailctl.emr import emr


def test_resolve_region_prefers_explicit(monkeypatch):
    monkeypatch.setenv('AWS_DEFAULT_REGION', 'us-west-2')
    with patch('hailtop.hailctl.emr.emr.configuration_of', return_value='eu-west-1'):
        assert emr.resolve_region('us-east-1') == 'us-east-1'


def test_resolve_region_falls_back_to_config(monkeypatch):
    monkeypatch.delenv('AWS_DEFAULT_REGION', raising=False)
    monkeypatch.delenv('AWS_REGION', raising=False)
    with patch('hailtop.hailctl.emr.emr.configuration_of', return_value='eu-west-1'):
        assert emr.resolve_region(None) == 'eu-west-1'


def test_resolve_region_falls_back_to_env(monkeypatch):
    monkeypatch.setenv('AWS_DEFAULT_REGION', 'us-west-2')
    with patch('hailtop.hailctl.emr.emr.configuration_of', return_value=None):
        assert emr.resolve_region(None) == 'us-west-2'


def test_resolve_region_none_when_unset(monkeypatch):
    monkeypatch.delenv('AWS_DEFAULT_REGION', raising=False)
    monkeypatch.delenv('AWS_REGION', raising=False)
    with patch('hailtop.hailctl.emr.emr.configuration_of', return_value=None):
        assert emr.resolve_region(None) is None


def test_upload_to_s3_writes_through_router_fs():
    from unittest.mock import AsyncMock, MagicMock

    fake_fs = MagicMock()
    fake_fs.write = AsyncMock()
    # RouterAsyncFS() is used as an async context manager: `async with RouterAsyncFS() as fs`.
    fake_ctx = MagicMock()
    fake_ctx.__aenter__ = AsyncMock(return_value=fake_fs)
    fake_ctx.__aexit__ = AsyncMock(return_value=False)
    with patch('hailtop.hailctl.emr.emr.RouterAsyncFS', return_value=fake_ctx):
        emr.upload_to_s3('s3://bkt/key.sh', b'hello')
    fake_fs.write.assert_awaited_once_with('s3://bkt/key.sh', b'hello')
