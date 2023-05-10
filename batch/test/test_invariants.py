import logging

import aiohttp
import pytest

from hailtop.utils import retry_transient_errors
from hailtop.auth import hail_credentials
from hailtop.config import get_deploy_config
from hailtop.httpx import client_session

pytestmark = pytest.mark.asyncio

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


async def test_invariants():
    deploy_config = get_deploy_config()
    url = deploy_config.url('batch-driver', '/check_invariants')
    headers = await hail_credentials().auth_headers()
    async with client_session(timeout=aiohttp.ClientTimeout(total=60)) as session:

        data = await retry_transient_errors(session.get_return_json, url, headers=headers)

        assert data['check_incremental_error'] is None, data
        assert data['check_resource_aggregation_error'] is None, data
