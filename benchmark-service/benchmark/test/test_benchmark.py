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

commit = {
    "url": "https://api.github.com/repos/octocat/Hello-World/commits/6dcb09b5b57875f334f61aebed695e2e4193db5e",
    "sha": "6dcb09b5b57875f334f61aebed695e2e4193db5e",
    "node_id": "MDY6Q29tbWl0NmRjYjA5YjViNTc4NzVmMzM0ZjYxYWViZWQ2OTVlMmU0MTkzZGI1ZQ==",
    "html_url": "https://github.com/octocat/Hello-World/commit/6dcb09b5b57875f334f61aebed695e2e4193db5e",
    "comments_url": "https://api.github.com/repos/octocat/Hello-World/commits/6dcb09b5b57875f334f61aebed695e2e4193db5e/comments",
    "commit": {
        "url": "https://api.github.com/repos/octocat/Hello-World/git/commits/6dcb09b5b57875f334f61aebed695e2e4193db5e",
        "author": {
            "name": "Monalisa Octocat",
            "email": "support@github.com",
            "date": "2011-04-14T16:00:49Z"
        },
        "committer": {
            "name": "Monalisa Octocat",
            "email": "support@github.com",
            "date": "2011-04-14T16:00:49Z"
        },
        "message": "Fix all the bugs",
        "tree": {
            "url": "https://api.github.com/repos/octocat/Hello-World/tree/6dcb09b5b57875f334f61aebed695e2e4193db5e",
            "sha": "6dcb09b5b57875f334f61aebed695e2e4193db5e"
        },
        "comment_count": 0,
        "verification": {
            "verified": False,
            "reason": "unsigned",
            "signature": False,
            "payload": False
        }
    },
    "author": {
        "login": "octocat",
        "id": 1,
        "node_id": "MDQ6VXNlcjE=",
        "avatar_url": "https://github.com/images/error/octocat_happy.gif",
        "gravatar_id": "",
        "url": "https://api.github.com/users/octocat",
        "html_url": "https://github.com/octocat",
        "followers_url": "https://api.github.com/users/octocat/followers",
        "following_url": "https://api.github.com/users/octocat/following{/other_user}",
        "gists_url": "https://api.github.com/users/octocat/gists{/gist_id}",
        "starred_url": "https://api.github.com/users/octocat/starred{/owner}{/repo}",
        "subscriptions_url": "https://api.github.com/users/octocat/subscriptions",
        "organizations_url": "https://api.github.com/users/octocat/orgs",
        "repos_url": "https://api.github.com/users/octocat/repos",
        "events_url": "https://api.github.com/users/octocat/events{/privacy}",
        "received_events_url": "https://api.github.com/users/octocat/received_events",
        "type": "User",
        "site_admin": False
    },
    "committer": {
        "login": "octocat",
        "id": 1,
        "node_id": "MDQ6VXNlcjE=",
        "avatar_url": "https://github.com/images/error/octocat_happy.gif",
        "gravatar_id": "",
        "url": "https://api.github.com/users/octocat",
        "html_url": "https://github.com/octocat",
        "followers_url": "https://api.github.com/users/octocat/followers",
        "following_url": "https://api.github.com/users/octocat/following{/other_user}",
        "gists_url": "https://api.github.com/users/octocat/gists{/gist_id}",
        "starred_url": "https://api.github.com/users/octocat/starred{/owner}{/repo}",
        "subscriptions_url": "https://api.github.com/users/octocat/subscriptions",
        "organizations_url": "https://api.github.com/users/octocat/orgs",
        "repos_url": "https://api.github.com/users/octocat/repos",
        "events_url": "https://api.github.com/users/octocat/events{/privacy}",
        "received_events_url": "https://api.github.com/users/octocat/received_events",
        "type": "User",
        "site_admin": False
    },
    "parents": [
        {
            "url": "https://api.github.com/repos/octocat/Hello-World/commits/6dcb09b5b57875f334f61aebed695e2e4193db5e",
            "sha": "6dcb09b5b57875f334f61aebed695e2e4193db5e"
        }
    ],
    "stats": {
        "additions": 104,
        "deletions": 4,
        "total": 108
    },
    "files": [
        {
            "filename": "file1.txt",
            "additions": 10,
            "deletions": 2,
            "changes": 12,
            "status": "modified",
            "raw_url": "https://github.com/octocat/Hello-World/raw/7ca483543807a51b6079e54ac4cc392bc29ae284/file1.txt",
            "blob_url": "https://github.com/octocat/Hello-World/blob/7ca483543807a51b6079e54ac4cc392bc29ae284/file1.txt",
            "patch": "@@ -29,7 +29,7 @@\n....."
        }
    ]
}


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
                session, 'POST', f'{create_benchmark_url}', headers=headers, json={'commit': commit})
        # return resp

        # batch_id = await resp.json()
        resp_text = await resp.text()
        batch_info = json.loads(resp_text)
        batch_id = batch_info['batch_status']['id']
        batch_url = deploy_config.url('benchmark', f'/api/v1alpha/benchmark/batches/{batch_id}')
    # async with get_context_specific_ssl_client_session(
    #         raise_for_status=True,
    #         timeout=aiohttp.ClientTimeout(total=60)) as session:

        async def wait_forever():
            batch_status = None
            complete = None
            while complete is None or complete is False:
                resp2 = await utils.request_retry_transient_errors(
                    session, 'GET', f'{batch_url}', headers=headers)
                batch_status = await resp2.json()
                log.info(f'batch_status:\n{json.dumps(batch_status, indent=2)}')
                print(f'status: {batch_status}')
                # assert batch_status['batch_status']['n_succeeded'] > 0
                await asyncio.sleep(5)
                complete = batch_status['batch_status']['complete']
            return batch_status

        batch_status = await asyncio.wait_for(wait_forever(), timeout=30 * 60)
        assert batch_status['batch_status']['n_succeeded'] > 0
        # assert batch_status == 'success'
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
