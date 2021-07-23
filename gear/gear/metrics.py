from aiohttp import web
import prometheus_client as pc  # type: ignore
from prometheus_async.aio import time as prom_async_time  # type: ignore

REQUEST_TIME = pc.Summary('http_request_latency_seconds', 'Endpoint latency in seconds', ['endpoint', 'verb'])
REQUEST_COUNT = pc.Counter('http_request_count', 'Number of HTTP requests', ['endpoint', 'verb', 'status'])
CONCURRENT_REQUESTS = pc.Gauge('http_concurrent_requests', 'Number of in progress HTTP requests', ['endpoint', 'verb'])


@web.middleware
async def monitor_endpoints_middleware(request, handler):
    if request.match_info.route.resource:
        # Use the path template given to @route.<METHOD>, not the fully resolved one
        endpoint = request.match_info.route.resource.canonical
    else:
        endpoint = ''
    verb = request.method
    CONCURRENT_REQUESTS.labels(endpoint=endpoint, verb=verb).inc()
    try:
        response = await prom_async_time(REQUEST_TIME.labels(endpoint=endpoint, verb=verb), handler(request))
        REQUEST_COUNT.labels(endpoint=endpoint, verb=verb, status=response.status).inc()
        return response
    except web.HTTPException as e:
        REQUEST_COUNT.labels(endpoint=endpoint, verb=verb, status=e.status).inc()
        raise e
    finally:
        CONCURRENT_REQUESTS.labels(endpoint=endpoint, verb=verb).dec()
