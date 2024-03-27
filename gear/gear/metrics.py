import json
from typing import Any, Optional

import prometheus_client as pc  # type: ignore
from aiohttp import web
from aiohttp.typedefs import JSONEncoder
from prometheus_async.aio import time as prom_async_time
from prometheus_client.context_managers import Timer  # type: ignore

REQUEST_TIME = pc.Summary('http_request_latency_seconds', 'Endpoint latency in seconds', ['endpoint', 'verb'])
REQUEST_COUNT = pc.Counter('http_request_count', 'Number of HTTP requests', ['endpoint', 'verb', 'status'])
CONCURRENT_REQUESTS = pc.Gauge('http_concurrent_requests', 'Number of in progress HTTP requests', ['endpoint', 'verb'])

SQL_TRANSACTIONS = pc.Counter('sql_transactions', 'Number of SQL transactions')
SQL_QUERY_COUNT = pc.Counter('sql_query_count', 'Number of SQL Queries', ['query_name'])
SQL_QUERY_LATENCY = pc.Summary('sql_query_latency_seconds', 'SQL Query latency in seconds', ['query_name'])
DB_CONNECTION_QUEUE_SIZE = pc.Gauge('sql_connection_queue_size', 'Number of coroutines waiting for a connection')

ACTIVE_WEBSOCKETS = pc.Gauge('wss_active_connections', 'Number of active websocket connections')
WEBSOCKET_MESSAGES = pc.Counter('wss_messages', 'Number of websocket messages', ['sent'])
WEBSOCKET_ERRORS = pc.Counter('wss_errors', 'Number of errors that occured during websocket communication')


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
        response = await prom_async_time(REQUEST_TIME.labels(endpoint=endpoint, verb=verb), handler(request))  # type: ignore
        REQUEST_COUNT.labels(endpoint=endpoint, verb=verb, status=response.status).inc()
        return response
    except web.HTTPException as e:
        REQUEST_COUNT.labels(endpoint=endpoint, verb=verb, status=e.status).inc()
        raise e
    finally:
        CONCURRENT_REQUESTS.labels(endpoint=endpoint, verb=verb).dec()


class PrometheusSQLTimer:
    def __init__(self, query_name: str):
        self.query_name = query_name
        self.sql_query_latency_manager: Optional[Timer] = None

    async def __aenter__(self):
        SQL_QUERY_COUNT.labels(query_name=self.query_name).inc()
        self.sql_query_latency_manager = SQL_QUERY_LATENCY.labels(query_name=self.query_name).time()
        self.sql_query_latency_manager.__enter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        assert self.sql_query_latency_manager
        self.sql_query_latency_manager.__exit__(exc_type, exc, tb)


class MetricsWebSocketResponse(web.WebSocketResponse):
    def __init__(self):
        super().__init__()
        ACTIVE_WEBSOCKETS.inc()

    async def send_str(self, data: str, compress: Optional[bool] = None) -> None:
        await super().send_str(data, compress)
        WEBSOCKET_MESSAGES.labels(sent=False).inc()

    async def send_bytes(self, data: bytes, compress: Optional[bool] = None) -> None:
        await super().send_bytes(data, compress)
        WEBSOCKET_MESSAGES.labels(sent=False).inc()

    async def send_json(
        self,
        data: Any,
        compress: Optional[bool] = None,
        *,
        dumps: JSONEncoder = json.dumps,
    ) -> None:
        await super().send_json(data, compress, dumps=dumps)
        WEBSOCKET_MESSAGES.labels(sent=False).inc()

    async def __anext__(self):
        try:
            val = await super().__anext__()
            WEBSOCKET_MESSAGES.labels(sent=False).inc()
            return val
        except StopAsyncIteration:
            ACTIVE_WEBSOCKETS.dec()
            raise
        except Exception:
            ACTIVE_WEBSOCKETS.dec()
            WEBSOCKET_ERRORS.inc()
            raise
