from aiohttp import web


@web.middleware
async def global_security_headers_middleware(request: web.Request, handler):
    response = await handler(request)

    response.headers['Strict-Transport-Security'] = 'max-age=63072000; includeSubDomains;'
    response.headers['X-Content-Type-Options'] = 'nosniff'

    return response
