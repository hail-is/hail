import asyncio
import logging

import pytest

from hailtop.auth import hail_credentials
from hailtop.config import get_deploy_config
from hailtop.httpx import client_session
from hailtop.utils import retry_transient_errors

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_billing_monitoring():
    deploy_config = get_deploy_config()
    monitoring_deploy_config_url = deploy_config.url('monitoring', '/api/v1alpha/billing')
    async with hail_credentials() as credentials:
        async with client_session() as session:

            async def wait_forever():
                data = None
                while data is None:
                    headers = await credentials.auth_headers()
                    data = await retry_transient_errors(
                        session.get_read_json, monitoring_deploy_config_url, headers=headers
                    )
                    await asyncio.sleep(5)
                return data

            data = await asyncio.wait_for(wait_forever(), timeout=30 * 60)
            assert data['cost_by_service'] is not None, data
