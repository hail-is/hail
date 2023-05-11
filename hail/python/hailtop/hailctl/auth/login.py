from typing import Optional
import json

from hailtop.config import get_deploy_config, DeployConfig, get_user_identity_config_path
from hailtop.auth import hail_credentials, IdentityProvider, AzureFlow, GoogleFlow
from hailtop.httpx import client_session, ClientSession


async def auth_flow(deploy_config: DeployConfig, default_ns: str, session: ClientSession):
    resp = await session.get_read_json(deploy_config.url('auth', '/api/v1alpha/oauth2-client'))
    idp = IdentityProvider(resp['idp'])
    client_secret_config = resp['oauth2_client']
    if idp == IdentityProvider.GOOGLE:
        credentials = GoogleFlow.perform_installed_app_login_flow(client_secret_config)
    else:
        assert idp == IdentityProvider.MICROSOFT
        credentials = AzureFlow.perform_installed_app_login_flow(client_secret_config)

    with open(get_user_identity_config_path(), 'w', encoding='utf-8') as f:
        f.write(json.dumps({'idp': idp.value, 'credentials': credentials}))

    # Confirm that the logged in user is registered with the hail service
    async with hail_credentials(namespace=default_ns) as c:
        headers_with_auth = await c.auth_headers()
        async with client_session(headers=headers_with_auth) as auth_session:
            userinfo = await auth_session.get_read_json(deploy_config.url('auth', '/api/v1alpha/userinfo'))

    username = userinfo['username']
    if default_ns == 'default':
        print(f'Logged in as {username}.')
    else:
        print(f'Logged into namespace {default_ns} as {username}.')


async def async_login(namespace: Optional[str]):
    deploy_config = get_deploy_config()
    namespace = namespace or deploy_config.default_namespace()
    async with hail_credentials(namespace=namespace, authorize_target=False) as credentials:
        headers = await credentials.auth_headers()
        async with client_session(headers=headers) as session:
            await auth_flow(deploy_config, namespace, session)
