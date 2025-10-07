from aiohttp import web


@web.middleware
async def error_handler_middleware(request: web.Request, handler):
    """Error handler middleware that catches exceptions and ensures security headers are applied.

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
        # This ensures the response goes through the middleware chain
        if e.status == 404:
            response = web.Response(text="404 Not Found", status=404)
        elif e.status == 307:
            response = web.Response(text="307 Temporary Redirect", status=307)
        elif e.status == 500:
            response = web.Response(text="500 Internal Server Error", status=500)
        elif e.status == 403:
            response = web.Response(text="403 Forbidden", status=403)
        elif e.status == 401:
            response = web.Response(text="401 Unauthorized", status=401)
        elif e.status == 400:
            response = web.Response(text="400 Bad Request", status=400)
        else:
            # For other HTTP status codes, use the exception's reason or a generic message
            reason = e.reason or f"{e.status} Error"
            response = web.Response(text=reason, status=e.status)

        return response
    except Exception:
        # For unexpected exceptions, return a 500 error
        # This ensures all errors go through the middleware chain
        response = web.Response(text="500 Internal Server Error", status=500)
        return response
