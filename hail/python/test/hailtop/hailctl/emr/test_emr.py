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


def _fake_iam(existing_roles):
    from unittest.mock import MagicMock

    from botocore.exceptions import ClientError

    iam = MagicMock()

    def get_role(RoleName):  # noqa: N803 (boto3 kwarg name)
        if RoleName in existing_roles:
            return {'Role': {'RoleName': RoleName, 'Arn': f'arn:aws:iam::123:role/{RoleName}'}}
        raise ClientError({'Error': {'Code': 'NoSuchEntity', 'Message': 'not found'}}, 'GetRole')

    iam.get_role.side_effect = get_role
    return iam


def test_check_default_roles_present_prints_message(capsys):
    iam = _fake_iam({'EMR_DefaultRole', 'EMR_EC2_DefaultRole'})
    emr.check_default_roles(iam)
    out = capsys.readouterr().out
    assert 'Using existing EMR default roles' in out
    assert 'EMR_DefaultRole' in out and 'EMR_EC2_DefaultRole' in out


def test_check_default_roles_missing_raises():
    import pytest

    iam = _fake_iam({'EMR_DefaultRole'})  # EMR_EC2_DefaultRole missing
    with pytest.raises(ValueError, match='Missing EMR default IAM role.*EMR_EC2_DefaultRole'):
        emr.check_default_roles(iam)


def test_check_default_roles_propagates_unexpected_error():
    import pytest
    from botocore.exceptions import ClientError

    from unittest.mock import MagicMock

    iam = MagicMock()
    iam.get_role.side_effect = ClientError(
        {'Error': {'Code': 'AccessDenied', 'Message': 'nope'}}, 'GetRole'
    )
    with pytest.raises(ClientError):
        emr.check_default_roles(iam)


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
