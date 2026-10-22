import os
from typing import Optional

from hailtop.aiotools.router_fs import RouterAsyncFS
from hailtop.config import ConfigVariable, configuration_of
from hailtop.utils import async_to_blocking


def resolve_region(explicit_region: Optional[str]) -> Optional[str]:
    """Resolve the AWS region for EMR operations.

    Order: explicit argument, then the emr/region config variable, then the
    AWS_DEFAULT_REGION / AWS_REGION environment variables. Returns None if
    unset so that botocore can resolve it from the user's AWS config.
    """
    if explicit_region is not None:
        return explicit_region
    config_region = configuration_of(ConfigVariable.EMR_REGION, None, None)
    if config_region is not None:
        return config_region
    return os.environ.get('AWS_DEFAULT_REGION') or os.environ.get('AWS_REGION')


def emr_client(region: Optional[str]):
    import boto3  # pylint: disable=import-outside-toplevel

    return boto3.client('emr', region_name=region)


DEFAULT_SERVICE_ROLE = 'EMR_DefaultRole'
DEFAULT_JOB_FLOW_ROLE = 'EMR_EC2_DefaultRole'


def _is_access_denied(exc: Exception) -> bool:
    code = getattr(exc, 'response', {}).get('Error', {}).get('Code')
    return code in {'AccessDenied', 'AccessDeniedException', 'UnauthorizedOperation'}


def _role_exists(iam, role_name: str) -> bool:
    from botocore.exceptions import ClientError  # pylint: disable=import-outside-toplevel

    try:
        iam.get_role(RoleName=role_name)
        return True
    except ClientError as exc:
        if exc.response.get('Error', {}).get('Code') == 'NoSuchEntity':
            return False
        raise


def _instance_profile_contains_role(iam, instance_profile_name: str, role_name: str) -> bool:
    from botocore.exceptions import ClientError  # pylint: disable=import-outside-toplevel

    try:
        response = iam.get_instance_profile(InstanceProfileName=instance_profile_name)
    except ClientError as exc:
        if exc.response.get('Error', {}).get('Code') == 'NoSuchEntity':
            return False
        raise
    roles = response.get('InstanceProfile', {}).get('Roles', [])
    return any(role.get('RoleName') == role_name for role in roles)


def check_default_roles(iam=None, *, check_service_role: bool = True, check_job_flow_role: bool = True) -> None:
    """Verify the EMR default IAM roles exist, printing a clear message.

    `aws emr create-default-roles` prints only the roles it *creates*, so it
    returns an empty list (``[]``) when they already exist -- which reads as if
    nothing happened. This preflight instead reports explicitly that the roles
    are present, and raises an actionable error listing any that are missing.

    IAM is global, so no region is needed.
    """
    if iam is None:
        import boto3  # pylint: disable=import-outside-toplevel

        iam = boto3.client('iam')

    roles = []
    if check_service_role:
        roles.append(DEFAULT_SERVICE_ROLE)
    if check_job_flow_role:
        roles.append(DEFAULT_JOB_FLOW_ROLE)
    try:
        missing_roles = [role for role in roles if not _role_exists(iam, role)]
        missing_instance_profiles = []
        if check_job_flow_role and not _instance_profile_contains_role(
            iam, DEFAULT_JOB_FLOW_ROLE, DEFAULT_JOB_FLOW_ROLE
        ):
            missing_instance_profiles.append(f'{DEFAULT_JOB_FLOW_ROLE} containing role {DEFAULT_JOB_FLOW_ROLE}')
    except Exception as exc:
        if _is_access_denied(exc):
            print(f'Warning: unable to preflight EMR default IAM resources: {exc}')
            return
        raise

    if missing_roles or missing_instance_profiles:
        missing = []
        if missing_roles:
            missing.append(f"role(s): {', '.join(missing_roles)}")
        if missing_instance_profiles:
            missing.append(f"instance profile(s): {', '.join(missing_instance_profiles)}")
        raise ValueError(
            f"Missing EMR default IAM resource(s): {'; '.join(missing)}. "
            f"Create them once with `aws emr create-default-roles`, or pass "
            f"--no-use-default-roles together with --service-role and --instance-profile."
        )
    resources = []
    if roles:
        resources.append(f"role(s) {', '.join(roles)}")
    if check_job_flow_role:
        resources.append(f'instance profile {DEFAULT_JOB_FLOW_ROLE}')
    print(f"Using existing EMR defaults: {'; '.join(resources)}.")


def upload_to_s3(dest_uri: str, data: bytes) -> None:
    """Write bytes to an s3:// URI through Hail's RouterAsyncFS.

    S3 file I/O goes through the same FS abstraction the rest of hailtop uses,
    rather than a raw boto3 S3 client.
    """

    async def _upload() -> None:
        async with RouterAsyncFS() as fs:
            await fs.write(dest_uri, data)

    async_to_blocking(_upload())


def check_release_label(region: Optional[str], expected_release, client=None) -> None:
    from .artifact import major_minor  # pylint: disable=import-outside-toplevel

    if client is None:
        client = emr_client(region)
    response = client.describe_release_label(ReleaseLabel=expected_release.release_label)
    actual_label = response.get('ReleaseLabel')
    if actual_label != expected_release.release_label:
        raise ValueError(
            f'EMR resolved release {expected_release.release_label!r} to unexpected label {actual_label!r}'
        )
    applications = response.get('Applications', [])
    spark_version = next(
        (application.get('Version') for application in applications if application.get('Name') == 'Spark'),
        None,
    )
    if not isinstance(spark_version, str):
        raise ValueError(f'EMR release {actual_label!r} does not report a Spark application version')
    if major_minor(spark_version) != major_minor(expected_release.spark_version):
        raise ValueError(
            f'EMR release {actual_label!r} reports Spark {spark_version}, expected {expected_release.spark_version}'
        )
    print(f'Using EMR release {actual_label} with Spark {spark_version}.')


def check_private_subnet(
    region: str,
    subnet_id: str,
    service_access_security_group: str,
    primary_security_group: str,
    core_security_group: str,
    ec2=None,
) -> None:
    if ec2 is None:
        import boto3  # pylint: disable=import-outside-toplevel

        ec2 = boto3.client('ec2', region_name=region)

    subnet_response = ec2.describe_subnets(SubnetIds=[subnet_id])
    subnets = subnet_response.get('Subnets', [])
    if len(subnets) != 1:
        raise ValueError(f'could not resolve exactly one subnet for {subnet_id!r}')
    subnet = subnets[0]
    if subnet.get('MapPublicIpOnLaunch'):
        raise ValueError(f'subnet {subnet_id} maps public IP addresses; a private subnet is required')
    vpc_id = subnet['VpcId']

    dns_attributes = {
        'enableDnsSupport': 'EnableDnsSupport',
        'enableDnsHostnames': 'EnableDnsHostnames',
    }
    for attribute, response_key in dns_attributes.items():
        response = ec2.describe_vpc_attribute(VpcId=vpc_id, Attribute=attribute)
        if not response.get(response_key, {}).get('Value'):
            raise ValueError(f'VPC {vpc_id} must have {attribute} enabled')

    route_tables = ec2.describe_route_tables(Filters=[{'Name': 'association.subnet-id', 'Values': [subnet_id]}]).get(
        'RouteTables', []
    )
    if not route_tables:
        route_tables = ec2.describe_route_tables(
            Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}, {'Name': 'association.main', 'Values': ['true']}]
        ).get('RouteTables', [])
    if len(route_tables) != 1:
        raise ValueError(f'could not resolve the route table for private subnet {subnet_id}')
    route_table = route_tables[0]
    routes = route_table.get('Routes', [])
    public_default = any(
        route.get('State') == 'active'
        and route.get('DestinationCidrBlock') == '0.0.0.0/0'
        and str(route.get('GatewayId', '')).startswith('igw-')
        for route in routes
    )
    if public_default:
        raise ValueError(f'subnet {subnet_id} has a default route to an internet gateway')
    nat_default = any(
        route.get('State') == 'active'
        and route.get('DestinationCidrBlock') == '0.0.0.0/0'
        and str(route.get('NatGatewayId', '')).startswith('nat-')
        for route in routes
    )
    if not nat_default:
        raise ValueError(f'subnet {subnet_id} needs an active NAT default route for bootstrap egress')

    endpoints = ec2.describe_vpc_endpoints(
        Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}, {'Name': 'vpc-endpoint-state', 'Values': ['available']}]
    ).get('VpcEndpoints', [])
    emr_service = f'aws.api.{region}.emr-service-cell01'
    emr_endpoints = [endpoint for endpoint in endpoints if endpoint.get('ServiceName') == emr_service]
    if not emr_endpoints:
        raise ValueError(f'VPC {vpc_id} needs an available EMR service endpoint for {emr_service}')
    if not any(endpoint.get('PrivateDnsEnabled') for endpoint in emr_endpoints):
        raise ValueError(f'EMR service endpoint {emr_service} must have private DNS enabled')
    endpoint_security_groups = {
        group.get('GroupId') for endpoint in emr_endpoints for group in endpoint.get('Groups', [])
    }
    if service_access_security_group not in endpoint_security_groups:
        raise ValueError(
            f'EMR service endpoint {emr_service} is not attached to security group {service_access_security_group}'
        )

    vpcs = ec2.describe_vpcs(VpcIds=[vpc_id]).get('Vpcs', [])
    if len(vpcs) != 1:
        raise ValueError(f'could not resolve VPC {vpc_id}')
    vpc_cidr = vpcs[0]['CidrBlock']
    required_security_groups = [service_access_security_group, primary_security_group, core_security_group]
    security_groups = ec2.describe_security_groups(GroupIds=required_security_groups).get('SecurityGroups', [])
    groups_by_id = {group.get('GroupId'): group for group in security_groups}
    for group_id in required_security_groups:
        if groups_by_id.get(group_id, {}).get('VpcId') != vpc_id:
            raise ValueError(f'security group {group_id} is not in VPC {vpc_id}')
    service_group = groups_by_id[service_access_security_group]
    https_from_vpc = any(
        permission.get('IpProtocol') == 'tcp'
        and permission.get('FromPort') == 443
        and permission.get('ToPort') == 443
        and any(ip_range.get('CidrIp') == vpc_cidr for ip_range in permission.get('IpRanges', []))
        for permission in service_group.get('IpPermissions', [])
    )
    if not https_from_vpc:
        raise ValueError(
            f'service access security group {service_access_security_group} must allow TCP 443 from {vpc_cidr}'
        )

    s3_service = f'com.amazonaws.{region}.s3'
    s3_endpoints = [endpoint for endpoint in endpoints if endpoint.get('ServiceName') == s3_service]
    if not s3_endpoints:
        raise ValueError(f'VPC {vpc_id} needs an available S3 gateway endpoint for {s3_service}')
    route_table_id = route_table['RouteTableId']
    if not any(route_table_id in endpoint.get('RouteTableIds', []) for endpoint in s3_endpoints):
        raise ValueError(f'S3 gateway endpoint {s3_service} is not associated with route table {route_table_id}')

    print(
        f'Using private subnet {subnet_id} in VPC {vpc_id} with NAT, S3, and EMR service endpoints; '
        f'service access security group {service_access_security_group}.'
    )


def check_custom_roles(service_role: str, instance_profile: str, iam=None) -> None:
    from botocore.exceptions import ClientError  # pylint: disable=import-outside-toplevel

    if iam is None:
        import boto3  # pylint: disable=import-outside-toplevel

        iam = boto3.client('iam')
    try:
        if not _role_exists(iam, service_role):
            raise ValueError(f'EMR service role {service_role!r} does not exist')
        response = iam.get_instance_profile(InstanceProfileName=instance_profile)
    except ClientError as exc:
        if exc.response.get('Error', {}).get('Code') == 'NoSuchEntity':
            raise ValueError(f'EMR EC2 instance profile {instance_profile!r} does not exist') from exc
        if _is_access_denied(exc):
            print(f'Warning: unable to preflight custom EMR IAM resources: {exc}')
            return
        raise
    roles = response.get('InstanceProfile', {}).get('Roles', [])
    if not roles:
        raise ValueError(f'EMR EC2 instance profile {instance_profile!r} does not contain an IAM role')
    print(
        f"Using custom EMR service role {service_role} and EC2 instance profile {instance_profile} "
        f"with role(s) {', '.join(role['RoleName'] for role in roles)}."
    )
