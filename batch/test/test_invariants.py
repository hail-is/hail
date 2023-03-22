import aiohttp
import logging

import aiohttp
import pytest

import hailtop.utils as utils
from hailtop.auth import service_auth_headers
from hailtop.config import get_deploy_config
from hailtop.httpx import client_session

pytestmark = pytest.mark.asyncio

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


async def test_invariants():
    deploy_config = get_deploy_config()
    url = deploy_config.url('batch-driver', '/check_invariants')
    headers = service_auth_headers(deploy_config, 'batch-driver')
    async with client_session(timeout=aiohttp.ClientTimeout(total=60)) as session:

        resp = await utils.request_retry_transient_errors(session, 'GET', url, headers=headers)
        data = await resp.json()

        assert data['check_incremental_error'] is None, data
        assert data['check_resource_aggregation_error'] is None, data
