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
    except web.HTTPException as e:
        # For aiohttp HTTP exceptions, create a simple text response
        # This response then goes through the remaining middleware chain
        return web.Response(text=e.text, status=e.status)
    except Exception:
        # For unexpected exceptions, generate a 500 error response
        # This response then goes through the remaining middleware chain
        response = web.Response(text="500 Internal Server Error", status=500)
        return response
