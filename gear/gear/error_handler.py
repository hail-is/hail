from aiohttp import web


@web.middleware
async def error_handler_middleware(request: web.Request, handler):
    """Error handler middleware that catches exceptions and generates basic responses.

    This middleware wraps the handler in a try/catch block to catch any exceptions
    that would otherwise bypass the middleware chain. It returns simple text responses
    for common HTTP errors while allowing subsequent middlewares (like security headers)
    to still process the response.
    """
    try:
        response = await handler(request)
        return response
    except web.HTTPMove as e:
        response = web.Response(text=e.text, status=e.status)
        response.headers.add('Location', str(e.location))
        return response
    except web.HTTPMethodNotAllowed as e:
        response = web.Response(text=e.text, status=e.status)
        response.headers.add('Allow', ','.join(e.allowed_methods))
        return response
    except web.HTTPException as e:
        response = web.Response(text=e.text, status=e.status)
        return response
    except Exception:
        response = web.Response(text="500 Internal Server Error", status=500)
        return response
