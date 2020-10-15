import json
import logging
import asyncio
import pytest
import aiohttp

from hailtop.config import get_deploy_config
from hailtop.auth import service_auth_headers
from hailtop.tls import in_cluster_ssl_client_session, get_context_specific_ssl_client_session
import hailtop.utils as utils

pytestmark = pytest.mark.asyncio

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

commit = "{'author': { 'name': 'John Compitello', 'email': 'johnc@broadinstitute.org', 'date': '2020-09-11T13:56:57Z'}, 'committer': {'name': 'GitHub','email': 'noreply@github.com','date': '2020-09-11T13:56:57Z'}, 'message': 'emitI: NDArrayShape and NDArrayReindex (#9431)\n\n* Move NDArrayShape into emitI, add function to get shape PCode from PNDArrayCode\n\n* Cleaned up NDArrayReindex a bit, I'm going to want to find a new construct interface at some point though\n\n* Move all the emitDeforestedArray calls to emitI\n\n* Delete the old unused emitDeforestedNDArray method from regular emit', 'tree': {'sha': 'fc3f7c697b9ffff9b44546c459cfc7eebbd2fcf5', 'url': '[https://api.github.com/repos/hail-is/hail/git/trees/fc3f7c697b9ffff9b44546c459cfc7eebbd2fcf5](https://api.github.com/repos/hail-is/hail/git/trees/fc3f7c697b9ffff9b44546c459cfc7eebbd2fcf5)'}, 'url': '[https://api.github.com/repos/hail-is/hail/git/commits/ef7262d01f2bde422aaf09b6f84091ac0e439b1d](https://api.github.com/repos/hail-is/hail/git/commits/ef7262d01f2bde422aaf09b6f84091ac0e439b1d)', 'comment_count': 0, 'verification': {'verified': True, 'reason': 'valid', 'signature': '-----BEGIN PGP SIGNATURE-----\n\nwsBcBAABCAAQBQJfW4IpCRBK7hj4Ov3rIwAAdHIIAHhCCkG1NoSIENxP/iPVhWjZ\n7PYRJn6nYyPR7fSyXsQVLOr/p/k4bFwDarjz1DglVCk+nlDF777hnYhGIsHKmdud\nd5kke5lE7wUTUtgYwAGDFOROdIcq8ihkezoB2JREpJ9mdilRtvNfS1cl5YUbdwT+\nGuRXqTnvLaVqhNrDpCthIU+9LHAhLRn44uNRlel824ttFWp5GTWUaeYUZEmZ3cLQ\n7sgshhy8LSVqgh81sgUyBhymuTJdPDW6nev3FlFx+eCbt8M3sQArn/KmA04mIHY1\nNR41txCRBHzn7f0DPLDV05yW9zhPRYO6sc9/fbuc1jeAK1iQDOF7ZZA8co8ZcmM=\n=7aoX\n-----END PGP SIGNATURE-----\n', 'payload': 'tree fc3f7c697b9ffff9b44546c459cfc7eebbd2fcf5\nparent b24b393755d58b712c5029a3b4f8c6cf3e1b9b25\nauthor John Compitello [johnc@broadinstitute.org](mailto:johnc@broadinstitute.org) 1599832617 -0400\ncommitter GitHub [noreply@github.com](mailto:noreply@github.com) 1599832617 -0400\n\nemitI: NDArrayShape and NDArrayReindex (#9431)\n\n* Move NDArrayShape into emitI, add function to get shape PCode from PNDArrayCode\n\n* Cleaned up NDArrayReindex a bit, I'm going to want to find a new construct interface at some point though\n\n* Move all the emitDeforestedArray calls to emitI\n\n* Delete the old unused emitDeforestedNDArray method from regular emit'}}, 'url': '[https://api.github.com/repos/hail-is/hail/commits/ef7262d01f2bde422aaf09b6f84091ac0e439b1d](https://api.github.com/repos/hail-is/hail/commits/ef7262d01f2bde422aaf09b6f84091ac0e439b1d)', 'html_url': '[https://github.com/hail-is/hail/commit/ef7262d01f2bde422aaf09b6f84091ac0e439b1d](https://github.com/hail-is/hail/commit/ef7262d01f2bde422aaf09b6f84091ac0e439b1d)', 'comments_url': '[https://api.github.com/repos/hail-is/hail/commits/ef7262d01f2bde422aaf09b6f84091ac0e439b1d/comments](https://api.github.com/repos/hail-is/hail/commits/ef7262d01f2bde422aaf09b6f84091ac0e439b1d/comments)', 'author': {'login': 'johnc1231', 'id': 13773586, 'node_id': 'MDQ6VXNlcjEzNzczNTg2', 'avatar_url': '[https://avatars3.githubusercontent.com/u/13773586?v=4](https://avatars3.githubusercontent.com/u/13773586?v=4)', 'gravatar_id': '', 'url': '[https://api.github.com/users/johnc1231](https://api.github.com/users/johnc1231)', 'html_url': '[https://github.com/johnc1231](https://github.com/johnc1231)', 'followers_url': '[https://api.github.com/users/johnc1231/followers](https://api.github.com/users/johnc1231/followers)', 'following_url': '[https://api.github.com/users/johnc1231/following{/other_user}](https://api.github.com/users/johnc1231/following%7B/other_user%7D)', 'gists_url': '[https://api.github.com/users/johnc1231/gists{/gist_id}](https://api.github.com/users/johnc1231/gists%7B/gist_id%7D)', 'starred_url': '[https://api.github.com/users/johnc1231/starred{/owner}{/repo}](https://api.github.com/users/johnc1231/starred%7B/owner%7D%7B/repo%7D)', 'subscriptions_url': '[https://api.github.com/users/johnc1231/subscriptions](https://api.github.com/users/johnc1231/subscriptions)', 'organizations_url': '[https://api.github.com/users/johnc1231/orgs](https://api.github.com/users/johnc1231/orgs)', 'repos_url': '[https://api.github.com/users/johnc1231/repos](https://api.github.com/users/johnc1231/repos)', 'events_url': '[https://api.github.com/users/johnc1231/events{/privacy}](https://api.github.com/users/johnc1231/events%7B/privacy%7D)', 'received_events_url': '[https://api.github.com/users/johnc1231/received_events](https://api.github.com/users/johnc1231/received_events)', 'type': 'User', 'site_admin': False}, 'committer': {'login': 'web-flow', 'id': 19864447, 'node_id': 'MDQ6VXNlcjE5ODY0NDQ3', 'avatar_url': '[https://avatars3.githubusercontent.com/u/19864447?v=4](https://avatars3.githubusercontent.com/u/19864447?v=4)', 'gravatar_id': '', 'url': '[https://api.github.com/users/web-flow](https://api.github.com/users/web-flow)', 'html_url': '[https://github.com/web-flow](https://github.com/web-flow)', 'followers_url': '[https://api.github.com/users/web-flow/followers](https://api.github.com/users/web-flow/followers)', 'following_url': '[https://api.github.com/users/web-flow/following{/other_user}](https://api.github.com/users/web-flow/following%7B/other_user%7D)', 'gists_url': '[https://api.github.com/users/web-flow/gists{/gist_id}](https://api.github.com/users/web-flow/gists%7B/gist_id%7D)', 'starred_url': '[https://api.github.com/users/web-flow/starred{/owner}{/repo}](https://api.github.com/users/web-flow/starred%7B/owner%7D%7B/repo%7D)', 'subscriptions_url': '[https://api.github.com/users/web-flow/subscriptions](https://api.github.com/users/web-flow/subscriptions)', 'organizations_url': '[https://api.github.com/users/web-flow/orgs](https://api.github.com/users/web-flow/orgs)', 'repos_url': '[https://api.github.com/users/web-flow/repos](https://api.github.com/users/web-flow/repos)', 'events_url': '[https://api.github.com/users/web-flow/events{/privacy}](https://api.github.com/users/web-flow/events%7B/privacy%7D)', 'received_events_url': '[https://api.github.com/users/web-flow/received_events](https://api.github.com/users/web-flow/received_events)', 'type': 'User', 'site_admin': False}"

print("hi")


async def test_submit():
    print("hello")
    deploy_config = get_deploy_config()
    headers = service_auth_headers(deploy_config, 'benchmark')
    create_benchmark_url = deploy_config.url('benchmark', '/api/v1alpha/benchmark/create_benchmark')
    # testing locally: where you've used in_cluster_ssl_client_session you need to use get_context_specific_ssl_client_session
    async with get_context_specific_ssl_client_session(
            raise_for_status=True,
            timeout=aiohttp.ClientTimeout(total=60)) as session:
        resp = await utils.request_retry_transient_errors(
                session, 'POST', f'{create_benchmark_url}', headers=headers, data={'commit': f'{commit}'})
        #return resp

        batch_id = int(resp.text())
        batch_url = deploy_config.url('benchmark', f'/api/v1alpha/benchmark/{batch_id}')
    # async with get_context_specific_ssl_client_session(
    #         raise_for_status=True,
    #         timeout=aiohttp.ClientTimeout(total=60)) as session:

        async def wait_forever():
            batch_status = None
            while batch_status is None:
                resp2 = await utils.request_retry_transient_errors(
                    session, 'GET', f'{batch_url}', headers=headers)
                batch_status = await resp2.json()
                await asyncio.sleep(5)
            return batch_status

        batch_status = await asyncio.wait_for(wait_forever(), timeout=30 * 60)
        assert batch_status == 'success'
        print(f'{batch_status}')

    # async with in_cluster_ssl_client_session(
    #         raise_for_status=True,
    #         timeout=aiohttp.ClientTimeout(total=60)) as session:
    #     async def wait_forever():
    #         batch_status = None
    #         while batch_status is None:
    #             resp = await utils.request_retry_transient_errors(
    #                 session, 'POST', f'{create_benchmark_url}', headers=headers, params={'', ''})
    #             batch_status = await resp.json()
    #             await asyncio.sleep(5)
    #         return batch_status
    #
    #     batch_status = await asyncio.wait_for(wait_forever(), timeout=30 * 60)
    #     assert batch_status == 'success'
