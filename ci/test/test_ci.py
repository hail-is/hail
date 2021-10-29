import json
import logging
import asyncio
import pytest

from hailtop.config import get_deploy_config
from hailtop.auth import service_auth_headers
from hailtop.httpx import client_session
import hailtop.utils as utils

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_deploy():
    deploy_config = get_deploy_config()
    ci_deploy_status_url = deploy_config.url('ci', '/api/v1alpha/deploy_status')
    headers = service_auth_headers(deploy_config, 'ci')
    async with client_session() as session:

        async def wait_forever():
            deploy_state = None
            failure_information = None
            while deploy_state is None:
                resp = await utils.request_retry_transient_errors(
                    session, 'GET', f'{ci_deploy_status_url}', headers=headers
                )
                deploy_statuses = await resp.json()
                log.info(f'deploy_statuses:\n{json.dumps(deploy_statuses, indent=2)}')
                assert len(deploy_statuses) == 1, deploy_statuses
                deploy_status = deploy_statuses[0]
                deploy_state = deploy_status['deploy_state']
                failure_information = deploy_status.get('failure_information')
                await asyncio.sleep(5)
            log.info(f'returning {deploy_status} {failure_information}')
            return deploy_state, failure_information

        deploy_state, failure_information = await wait_forever()
        assert deploy_state == 'success', str(failure_information)
