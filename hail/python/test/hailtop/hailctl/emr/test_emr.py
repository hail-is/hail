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


def _fake_iam(existing_roles, existing_instance_profiles):
    from unittest.mock import MagicMock

    from botocore.exceptions import ClientError

    iam = MagicMock()

    def get_role(RoleName):
        if RoleName in existing_roles:
            return {'Role': {'RoleName': RoleName, 'Arn': f'arn:aws:iam::123:role/{RoleName}'}}
        raise ClientError({'Error': {'Code': 'NoSuchEntity', 'Message': 'not found'}}, 'GetRole')

    def get_instance_profile(InstanceProfileName):
        if InstanceProfileName in existing_instance_profiles:
            return {
                'InstanceProfile': {
                    'InstanceProfileName': InstanceProfileName,
                    'Roles': [{'RoleName': InstanceProfileName}],
                }
            }
        raise ClientError({'Error': {'Code': 'NoSuchEntity', 'Message': 'not found'}}, 'GetInstanceProfile')

    iam.get_role.side_effect = get_role
    iam.get_instance_profile.side_effect = get_instance_profile
    return iam


def test_check_default_roles_present_prints_message(capsys):
    iam = _fake_iam({'EMR_DefaultRole', 'EMR_EC2_DefaultRole'}, {'EMR_EC2_DefaultRole'})
    emr.check_default_roles(iam)
    out = capsys.readouterr().out
    assert 'Using existing EMR defaults' in out
    assert 'EMR_DefaultRole' in out and 'EMR_EC2_DefaultRole' in out


def test_check_default_roles_can_check_only_service_role(capsys):
    iam = _fake_iam({'EMR_DefaultRole'}, set())
    emr.check_default_roles(iam, check_service_role=True, check_job_flow_role=False)
    assert iam.get_instance_profile.call_count == 0
    assert 'EMR_DefaultRole' in capsys.readouterr().out


def test_check_default_roles_can_check_only_job_flow_role(capsys):
    iam = _fake_iam({'EMR_EC2_DefaultRole'}, {'EMR_EC2_DefaultRole'})
    emr.check_default_roles(iam, check_service_role=False, check_job_flow_role=True)
    assert iam.get_role.call_count == 1
    assert 'EMR_EC2_DefaultRole' in capsys.readouterr().out


def test_check_default_roles_missing_role_raises():
    import pytest

    iam = _fake_iam({'EMR_DefaultRole'}, {'EMR_EC2_DefaultRole'})
    with pytest.raises(ValueError, match=r'Missing EMR default IAM resource.*role.*EMR_EC2_DefaultRole'):
        emr.check_default_roles(iam)


def test_check_default_roles_missing_instance_profile_raises():
    import pytest

    iam = _fake_iam({'EMR_DefaultRole', 'EMR_EC2_DefaultRole'}, set())
    with pytest.raises(ValueError, match=r'Missing EMR default IAM resource.*instance profile.*EMR_EC2_DefaultRole'):
        emr.check_default_roles(iam)


def test_check_default_roles_rejects_profile_without_expected_role():
    from unittest.mock import MagicMock

    import pytest

    iam = MagicMock()
    iam.get_role.return_value = {'Role': {'RoleName': 'present'}}
    iam.get_instance_profile.return_value = {
        'InstanceProfile': {'InstanceProfileName': 'EMR_EC2_DefaultRole', 'Roles': []}
    }
    with pytest.raises(ValueError, match=r'instance profile.*containing role EMR_EC2_DefaultRole'):
        emr.check_default_roles(iam)


def test_check_default_roles_warns_when_read_permissions_are_denied(capsys):
    from unittest.mock import MagicMock

    from botocore.exceptions import ClientError

    iam = MagicMock()
    iam.get_role.side_effect = ClientError({'Error': {'Code': 'AccessDenied', 'Message': 'nope'}}, 'GetRole')
    emr.check_default_roles(iam)
    assert 'Warning: unable to preflight' in capsys.readouterr().out


def test_check_default_roles_warns_when_instance_profile_read_is_denied(capsys):
    from unittest.mock import MagicMock

    from botocore.exceptions import ClientError

    iam = MagicMock()
    iam.get_role.return_value = {'Role': {'RoleName': 'present'}}
    iam.get_instance_profile.side_effect = ClientError(
        {'Error': {'Code': 'AccessDenied', 'Message': 'nope'}}, 'GetInstanceProfile'
    )
    emr.check_default_roles(iam)
    assert 'Warning: unable to preflight' in capsys.readouterr().out


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


def test_check_release_label_accepts_expected_spark(capsys):
    from unittest.mock import Mock

    from hailtop.hailctl.emr import start

    client = Mock()
    client.describe_release_label.return_value = {
        'ReleaseLabel': 'emr-spark-8.1.0',
        'Applications': [{'Name': 'Spark', 'Version': '4.1.1-amzn-0'}],
    }
    emr.check_release_label('us-east-1', start.release_config('emr-spark-8.1.0'), client=client)
    client.describe_release_label.assert_called_once_with(ReleaseLabel='emr-spark-8.1.0')
    assert 'Spark 4.1.1-amzn-0' in capsys.readouterr().out


def test_check_release_label_rejects_wrong_label_or_spark():
    from unittest.mock import Mock

    import pytest

    from hailtop.hailctl.emr import start

    expected = start.release_config('emr-spark-8.1.0')
    client = Mock()
    client.describe_release_label.return_value = {
        'ReleaseLabel': 'emr-spark-8.0.0',
        'Applications': [{'Name': 'Spark', 'Version': '4.0.2'}],
    }
    with pytest.raises(ValueError, match='unexpected label'):
        emr.check_release_label('us-east-1', expected, client=client)

    client.describe_release_label.return_value = {
        'ReleaseLabel': 'emr-spark-8.1.0',
        'Applications': [{'Name': 'Spark', 'Version': '4.0.2'}],
    }
    with pytest.raises(ValueError, match='reports Spark'):
        emr.check_release_label('us-east-1', expected, client=client)

    client.describe_release_label.return_value = {'ReleaseLabel': 'emr-spark-8.1.0', 'Applications': []}
    with pytest.raises(ValueError, match='does not report a Spark'):
        emr.check_release_label('us-east-1', expected, client=client)


def _private_subnet_ec2():
    from unittest.mock import Mock

    ec2 = Mock()
    ec2.describe_subnets.return_value = {
        'Subnets': [{'SubnetId': 'subnet-1', 'VpcId': 'vpc-1', 'MapPublicIpOnLaunch': False}]
    }
    response_keys = {
        'enableDnsSupport': 'EnableDnsSupport',
        'enableDnsHostnames': 'EnableDnsHostnames',
    }
    ec2.describe_vpc_attribute.side_effect = lambda **kwargs: {response_keys[kwargs['Attribute']]: {'Value': True}}
    ec2.describe_route_tables.return_value = {
        'RouteTables': [
            {
                'RouteTableId': 'rtb-1',
                'Routes': [
                    {
                        'DestinationCidrBlock': '0.0.0.0/0',
                        'NatGatewayId': 'nat-1',
                        'State': 'active',
                    }
                ],
            }
        ]
    }
    ec2.describe_vpc_endpoints.return_value = {
        'VpcEndpoints': [
            {
                'ServiceName': 'aws.api.us-east-1.emr-service-cell01',
                'State': 'available',
                'PrivateDnsEnabled': True,
                'Groups': [{'GroupId': 'sg-service'}],
            },
            {
                'ServiceName': 'com.amazonaws.us-east-1.s3',
                'State': 'available',
                'RouteTableIds': ['rtb-1'],
            },
        ]
    }
    ec2.describe_vpcs.return_value = {'Vpcs': [{'VpcId': 'vpc-1', 'CidrBlock': '10.77.0.0/16'}]}
    ec2.describe_security_groups.return_value = {
        'SecurityGroups': [
            {
                'GroupId': 'sg-service',
                'VpcId': 'vpc-1',
                'IpPermissions': [
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 443,
                        'ToPort': 443,
                        'IpRanges': [{'CidrIp': '10.77.0.0/16'}],
                    }
                ],
            },
            {'GroupId': 'sg-primary', 'VpcId': 'vpc-1', 'IpPermissions': []},
            {'GroupId': 'sg-core', 'VpcId': 'vpc-1', 'IpPermissions': []},
        ]
    }
    return ec2


def test_check_private_subnet_accepts_nat_and_required_endpoints(capsys):
    ec2 = _private_subnet_ec2()
    emr.check_private_subnet('us-east-1', 'subnet-1', 'sg-service', 'sg-primary', 'sg-core', ec2=ec2)
    assert 'private subnet subnet-1' in capsys.readouterr().out


def test_check_private_subnet_rejects_public_subnet():
    import pytest

    ec2 = _private_subnet_ec2()
    ec2.describe_subnets.return_value['Subnets'][0]['MapPublicIpOnLaunch'] = True
    with pytest.raises(ValueError, match='maps public IP'):
        emr.check_private_subnet('us-east-1', 'subnet-1', 'sg-service', 'sg-primary', 'sg-core', ec2=ec2)


def test_check_private_subnet_rejects_missing_nat():
    import pytest

    ec2 = _private_subnet_ec2()
    ec2.describe_route_tables.return_value['RouteTables'][0]['Routes'] = []
    with pytest.raises(ValueError, match='NAT default route'):
        emr.check_private_subnet('us-east-1', 'subnet-1', 'sg-service', 'sg-primary', 'sg-core', ec2=ec2)


def test_check_private_subnet_rejects_missing_emr_or_s3_endpoint():
    import pytest

    ec2 = _private_subnet_ec2()
    ec2.describe_vpc_endpoints.return_value['VpcEndpoints'] = [
        {
            'ServiceName': 'com.amazonaws.us-east-1.s3',
            'State': 'available',
            'RouteTableIds': ['rtb-1'],
        }
    ]
    with pytest.raises(ValueError, match='EMR service endpoint'):
        emr.check_private_subnet('us-east-1', 'subnet-1', 'sg-service', 'sg-primary', 'sg-core', ec2=ec2)

    ec2 = _private_subnet_ec2()
    ec2.describe_vpc_endpoints.return_value['VpcEndpoints'] = [
        {
            'ServiceName': 'aws.api.us-east-1.emr-service-cell01',
            'State': 'available',
            'PrivateDnsEnabled': True,
            'Groups': [{'GroupId': 'sg-service'}],
        }
    ]
    with pytest.raises(ValueError, match='S3 gateway endpoint'):
        emr.check_private_subnet('us-east-1', 'subnet-1', 'sg-service', 'sg-primary', 'sg-core', ec2=ec2)


def test_check_private_subnet_rejects_endpoint_security_group_misconfiguration():
    import pytest

    ec2 = _private_subnet_ec2()
    ec2.describe_vpc_endpoints.return_value['VpcEndpoints'][0]['Groups'] = [{'GroupId': 'sg-other'}]
    with pytest.raises(ValueError, match='is not attached to security group'):
        emr.check_private_subnet('us-east-1', 'subnet-1', 'sg-service', 'sg-primary', 'sg-core', ec2=ec2)

    ec2 = _private_subnet_ec2()
    ec2.describe_security_groups.return_value['SecurityGroups'][0]['IpPermissions'] = []
    with pytest.raises(ValueError, match='must allow TCP 443'):
        emr.check_private_subnet('us-east-1', 'subnet-1', 'sg-service', 'sg-primary', 'sg-core', ec2=ec2)


def test_check_custom_roles_accepts_service_role_and_profile(capsys):
    from unittest.mock import MagicMock

    iam = MagicMock()
    iam.get_role.return_value = {'Role': {'RoleName': 'custom-service'}}
    iam.get_instance_profile.return_value = {
        'InstanceProfile': {
            'InstanceProfileName': 'custom-profile',
            'Roles': [{'RoleName': 'custom-ec2-role'}],
        }
    }
    emr.check_custom_roles('custom-service', 'custom-profile', iam=iam)
    assert 'custom-service' in capsys.readouterr().out


def test_check_custom_roles_rejects_missing_resources():
    import pytest

    iam = _fake_iam(set(), set())
    with pytest.raises(ValueError, match='service role'):
        emr.check_custom_roles('missing-service', 'missing-profile', iam=iam)

    iam = _fake_iam({'custom-service'}, set())
    with pytest.raises(ValueError, match='instance profile'):
        emr.check_custom_roles('custom-service', 'missing-profile', iam=iam)


def test_check_custom_roles_warns_when_read_permissions_are_denied(capsys):
    from unittest.mock import MagicMock

    from botocore.exceptions import ClientError

    iam = MagicMock()
    iam.get_role.side_effect = ClientError({'Error': {'Code': 'AccessDenied', 'Message': 'nope'}}, 'GetRole')
    emr.check_custom_roles('custom-service', 'custom-profile', iam=iam)
    assert 'Warning: unable to preflight custom EMR IAM resources' in capsys.readouterr().out
