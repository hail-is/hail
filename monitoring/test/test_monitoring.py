import asyncio
import logging

import pytest

import hailtop.utils as utils
from hailtop.auth import hail_credentials
from hailtop.config import get_deploy_config
from hailtop.httpx import client_session

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_billing_monitoring():
    deploy_config = get_deploy_config()
    monitoring_deploy_config_url = deploy_config.url('monitoring', '/api/v1alpha/billing')
    headers = await hail_credentials().auth_headers()
    async with client_session() as session:

        async def wait_forever():
            data = None
            while data is None:
                resp = await utils.request_retry_transient_errors(
                    session, 'GET', f'{monitoring_deploy_config_url}', headers=headers
                )
                data = await resp.json()
                await asyncio.sleep(5)
            return data

        data = await asyncio.wait_for(wait_forever(), timeout=30 * 60)
        assert data['cost_by_service'] is not None, data
