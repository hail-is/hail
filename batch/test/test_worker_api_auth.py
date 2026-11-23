from typing import AsyncIterator, Dict, Optional

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from batch.worker_api_auth import INSTANCE_TOKEN_HEADER, instance_token_middleware, worker_api_headers


class TokenHolder:
    def __init__(self, token: Optional[str]):
        self.token = token


@pytest.fixture
def token_holder() -> TokenHolder:
    return TokenHolder('the-token')


@pytest.fixture
async def client(token_holder: TokenHolder) -> AsyncIterator[TestClient]:
    async def ok(request: web.Request) -> web.Response:  # pylint: disable=unused-argument
        return web.Response(text='ok')

    app = web.Application(middlewares=[instance_token_middleware(lambda: token_holder.token)])
    app.add_routes([
        web.post('/api/v1alpha/kill', ok),
        web.get('/api/v1alpha/batches/{batch_id}/jobs/{job_id}/status', ok),
        web.get('/healthcheck', ok),
    ])
    async with TestClient(TestServer(app)) as c:
        yield c


@pytest.mark.parametrize(
    'headers',
    [{}, {INSTANCE_TOKEN_HEADER: 'wrong'}, {INSTANCE_TOKEN_HEADER: ''}, {'Authorization': 'Bearer the-token'}],
)
@pytest.mark.parametrize(
    'method, path',
    [('POST', '/api/v1alpha/kill'), ('GET', '/api/v1alpha/batches/1/jobs/1/status'), ('GET', '/healthcheck')],
)
async def test_rejects_missing_or_wrong_token(client: TestClient, method: str, path: str, headers: Dict[str, str]):
    resp = await client.request(method, path, headers=headers)
    assert resp.status == 401


async def test_accepts_the_instance_token(client: TestClient):
    resp = await client.post('/api/v1alpha/kill', headers=worker_api_headers('the-token'))
    assert resp.status == 200


@pytest.mark.parametrize('headers', [{}, {INSTANCE_TOKEN_HEADER: ''}, {INSTANCE_TOKEN_HEADER: 'the-token'}])
async def test_unavailable_before_activation(client: TestClient, token_holder: TokenHolder, headers: Dict[str, str]):
    token_holder.token = None
    resp = await client.post('/api/v1alpha/kill', headers=headers)
    assert resp.status == 503


async def test_healthcheck_accepts_the_instance_token(client: TestClient):
    resp = await client.get('/healthcheck', headers=worker_api_headers('the-token'))
    assert resp.status == 200


async def test_unknown_paths_need_the_token(client: TestClient):
    resp = await client.get('/no/such/path')
    assert resp.status == 401
