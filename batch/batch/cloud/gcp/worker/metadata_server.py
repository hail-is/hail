from aiohttp import web

from hailtop.aiocloud import aiogoogle

from ....globals import HTTP_CLIENT_MAX_SIZE


class AppKeys:
    USER_CREDENTIALS = web.AppKey('credentials', aiogoogle.GoogleServiceAccountCredentials)
    GCE_METADATA_SERVER_CLIENT = web.AppKey('ms_client', aiogoogle.GoogleMetadataServerClient)


async def root(_):
    return web.Response(text='computeMetadata/\n')


async def project_id(request: web.Request):
    metadata_server_client = request.app[AppKeys.GCE_METADATA_SERVER_CLIENT]
    return web.Response(text=await metadata_server_client.project())


async def numeric_project_id(request: web.Request):
    metadata_server_client = request.app[AppKeys.GCE_METADATA_SERVER_CLIENT]
    return web.Response(text=await metadata_server_client.numeric_project_id())


async def service_accounts(request: web.Request):
    gsa_email = request.app[AppKeys.USER_CREDENTIALS].email
    return web.Response(text=f'default\n{gsa_email}\n')


async def user_service_account(request: web.Request):
    gsa_email = request.app[AppKeys.USER_CREDENTIALS].email
    recursive = request.query.get('recursive')
    # https://cloud.google.com/compute/docs/metadata/querying-metadata
    # token is not included in the recursive version, presumably as that
    # is not simple metadata but requires requesting an access token
    if recursive == 'true':
        return web.json_response(
            {
                'aliases': ['default'],
                'email': gsa_email,
                'scopes': ['https://www.googleapis.com/auth/cloud-platform'],
            },
        )
    return web.Response(text='aliases\nemail\nscopes\ntoken\n')


async def user_email(request: web.Request):
    return web.Response(text=request.app[AppKeys.USER_CREDENTIALS].email)


async def user_token(request: web.Request):
    access_token = await request.app[AppKeys.USER_CREDENTIALS]._get_access_token()
    return web.json_response({
        'access_token': access_token.token,
        'expires_in': access_token.expires_in,
        'token_type': 'Bearer',
    })


@web.middleware
async def middleware(request: web.Request, handler):
    credentials = request.app[AppKeys.USER_CREDENTIALS]
    gsa = request.match_info.get('gsa')
    if gsa and gsa not in (credentials.email, 'default'):
        raise web.HTTPBadRequest()

    response = await handler(request)
    response.enable_compression()

    # `gcloud` does not properly respect `charset`, which aiohttp automatically
    # sets so we have to explicitly erase it
    # See https://github.com/googleapis/google-auth-library-python/blob/b935298aaf4ea5867b5778bcbfc42408ba4ec02c/google/auth/compute_engine/_metadata.py#L170
    if 'application/json' in response.headers['Content-Type']:
        response.headers['Content-Type'] = 'application/json'
    response.headers['Metadata-Flavor'] = 'Google'
    response.headers['Server'] = 'Metadata Server for VM'
    response.headers['X-XSS-Protection'] = '0'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    return response


def create_app(
    credentials: aiogoogle.GoogleServiceAccountCredentials,
    metadata_server_client: aiogoogle.GoogleMetadataServerClient,
) -> web.Application:
    app = web.Application(
        client_max_size=HTTP_CLIENT_MAX_SIZE,
        middlewares=[middleware],
    )
    app[AppKeys.USER_CREDENTIALS] = credentials
    app[AppKeys.GCE_METADATA_SERVER_CLIENT] = metadata_server_client

    app.add_routes([
        web.get('/', root),
        web.get('/computeMetadata/v1/project/project-id', project_id),
        web.get('/computeMetadata/v1/project/numeric-project-id', numeric_project_id),
        web.get('/computeMetadata/v1/instance/service-accounts/', service_accounts),
        web.get('/computeMetadata/v1/instance/service-accounts/{gsa}/', user_service_account),
        web.get('/computeMetadata/v1/instance/service-accounts/{gsa}/email', user_email),
        web.get('/computeMetadata/v1/instance/service-accounts/{gsa}/token', user_token),
    ])

    async def close_credentials(_):
        await credentials.close()

    app.on_cleanup.append(close_credentials)
    return app
