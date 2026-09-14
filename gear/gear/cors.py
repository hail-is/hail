from functools import wraps

from aiohttp import web

from hailtop.config import get_deploy_config

deploy_config = get_deploy_config()

# Services whose browser-facing pages are allowed to make cross-origin, credentialed calls to
# each other's APIs, e.g. the monitoring cost-analysis dashboard fetching from batch.
HAIL_SERVICES = ['auth', 'batch', 'batch-driver', 'ci', 'monitoring']

ALLOWED_ORIGINS = {deploy_config.external_origin(service) for service in HAIL_SERVICES}


def cors_allow_hail_services(fun):
    @wraps(fun)
    async def wrapped(request: web.Request, *args, **kwargs):
        origin = request.headers.get('Origin')
        try:
            response = await fun(request, *args, **kwargs)
        except web.HTTPException as exc:
            if origin in ALLOWED_ORIGINS:
                exc.headers['Access-Control-Allow-Origin'] = origin
                exc.headers['Access-Control-Allow-Credentials'] = 'true'
                exc.headers['Vary'] = 'Origin'
            raise
        if origin in ALLOWED_ORIGINS:
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Credentials'] = 'true'
            response.headers['Vary'] = 'Origin'
        return response

    return wrapped
