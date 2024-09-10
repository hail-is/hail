import asyncio
import json
import logging

import pytest

from ci.github import GithubStatus, github_status
from hailtop.auth import hail_credentials
from hailtop.config import get_deploy_config
from hailtop.httpx import client_session
from hailtop.utils import retry_transient_errors

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


deploy_config = get_deploy_config()


async def test_deploy():
    ci_deploy_status_url = deploy_config.url('ci', '/api/v1alpha/deploy_status')
    async with hail_credentials() as creds:
        async with client_session() as session:

            async def wait_forever():
                deploy_state = None
                deploy_status = None
                failure_information = None
                while deploy_state is None:
                    headers = await creds.auth_headers()
                    deploy_statuses = await retry_transient_errors(
                        session.get_read_json, ci_deploy_status_url, headers=headers
                    )
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


async def test_envoy_config_debug_endpoint():
    for proxy in ('gateway', 'internal-gateway'):
        url = deploy_config.url('ci', f'/envoy-config/{proxy}')
        async with hail_credentials() as creds:
            async with client_session() as session:
                headers = await creds.auth_headers()
                await retry_transient_errors(session.get_read_json, url, headers=headers)


def test_github_check_statuses():
    for state in ["PENDING", "EXPECTED", "ACTION_REQUIRED", "STALE"]:
        assert github_status(state) == GithubStatus.PENDING
    for state in ["FAILURE", "ERROR", "TIMED_OUT", "CANCELLED", "STARTUP_FAILURE", "SKIPPED"]:
        assert github_status(state) == GithubStatus.FAILURE
    for state in ["SUCCESS", "NEUTRAL"]:
        assert github_status(state) == GithubStatus.SUCCESS
    for state in ["foo", "bar"]:
        with pytest.raises(ValueError):
            github_status(state)
