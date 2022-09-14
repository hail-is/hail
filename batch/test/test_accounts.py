import asyncio
import os
import secrets
from typing import Any, AsyncGenerator, Awaitable, Callable, Optional

import aiohttp
import pytest

from hailtop.auth import session_id_encode_to_str
from hailtop.batch_client.aioclient import Batch, BatchClient
from hailtop.utils import secret_alnum_string

from .billing_projects import get_billing_project_prefix

pytestmark = pytest.mark.asyncio

DOCKER_ROOT_IMAGE = os.environ['DOCKER_ROOT_IMAGE']


@pytest.fixture
async def make_client() -> AsyncGenerator[Callable[[str], Awaitable[BatchClient]], Any]:
    _bcs = []

    async def factory(project: str):
        bc = await BatchClient.create(project, token_file=os.environ['HAIL_TEST_TOKEN_FILE'])
        _bcs.append(bc)
        return bc

    yield factory
    for bc in _bcs:
        await bc.close()


@pytest.fixture
async def dev_client() -> AsyncGenerator[BatchClient, Any]:
    bc = await BatchClient.create(
        'billing-project-not-needed-but-required-by-BatchClient', token_file=os.environ['HAIL_TEST_DEV_TOKEN_FILE']
    )
    yield bc
    await bc.close()


@pytest.fixture
async def random_billing_project_name(dev_client: BatchClient) -> AsyncGenerator[str, Any]:
    billing_project_prefix = get_billing_project_prefix()

    name = f'{billing_project_prefix}_{secret_alnum_string(5)}'
    try:
        yield name
    finally:
        try:
            r = await dev_client.get_billing_project(name)
        except aiohttp.ClientResponseError as e:
            assert e.status == 403, e
        else:
            assert r['status'] != 'deleted', r
            try:
                async for batch in dev_client.list_batches(f'billing_project:{name}'):
                    await batch.delete()
            finally:
                try:
                    if r['status'] == 'open':
                        await dev_client.close_billing_project(name)
                finally:
                    if r['status'] != 'deleted':
                        await dev_client.delete_billing_project(name)


@pytest.fixture
async def new_billing_project(dev_client: BatchClient, random_billing_project_name: str):
    yield await dev_client.create_billing_project(random_billing_project_name)


async def test_bad_token():
    token = session_id_encode_to_str(secrets.token_bytes(32))
    bc = await BatchClient.create('test', _token=token)
    try:
        bb = bc.create_batch()
        bb.create_job(DOCKER_ROOT_IMAGE, ['false'])
        b = await bb.submit()
        assert False, str(await b.debug_info())
    except aiohttp.ClientResponseError as e:
        assert e.status == 401
    finally:
        await bc.close()


async def test_get_billing_project(make_client: Callable[[str], Awaitable[BatchClient]]):
    c = await make_client('billing-project-not-needed-but-required-by-BatchClient')
    r = await c.get_billing_project('test')
    assert r['billing_project'] == 'test', r
    assert {'test', 'test-dev'}.issubset(set(r['users'])), r
    assert r['status'] == 'open', r


async def test_list_billing_projects(make_client: Callable[[str], Awaitable[BatchClient]]):
    c = await make_client('billing-project-not-needed-but-required-by-BatchClient')
    r = await c.list_billing_projects()
    test_bps = [p for p in r if p['billing_project'] == 'test']
    assert len(test_bps) == 1, r
    bp = test_bps[0]
    assert bp['billing_project'] == 'test', bp
    assert {'test', 'test-dev'}.issubset(set(bp['users'])), bp
    assert bp['status'] == 'open', bp


async def test_unauthorized_billing_project_modification(
    make_client: Callable[[str], Awaitable[BatchClient]], new_billing_project: str
):
    project = new_billing_project
    client = await make_client('billing-project-not-needed-but-required-by-BatchClient')
    try:
        await client.create_billing_project(project)
    except aiohttp.ClientResponseError as e:
        assert e.status == 401, e
    else:
        assert False, 'expected error'

    try:
        await client.add_user('test', project)
    except aiohttp.ClientResponseError as e:
        assert e.status == 401, e
    else:
        assert False, 'expected error'

    try:
        await client.remove_user('test', project)
    except aiohttp.ClientResponseError as e:
        assert e.status == 401, e
    else:
        assert False, 'expected error'

    try:
        await client.close_billing_project(project)
    except aiohttp.ClientResponseError as e:
        assert e.status == 401, e
    else:
        assert False, 'expected error'

    try:
        await client.reopen_billing_project(project)
    except aiohttp.ClientResponseError as e:
        assert e.status == 401, e
    else:
        assert False, 'expected error'


async def test_create_billing_project(dev_client: BatchClient, new_billing_project: str):
    project = new_billing_project
    # test idempotent
    await dev_client.create_billing_project(project)

    r = await dev_client.list_billing_projects()
    assert project in {bp['billing_project'] for bp in r}


async def test_close_reopen_billing_project(dev_client: BatchClient, new_billing_project: str):
    project = new_billing_project

    await dev_client.close_billing_project(project)
    # test idempotent
    await dev_client.close_billing_project(project)
    r = await dev_client.list_billing_projects()
    assert [bp for bp in r if bp['billing_project'] == project and bp['status'] == 'open'] == [], r

    await dev_client.reopen_billing_project(project)
    # test idempotent
    await dev_client.reopen_billing_project(project)
    r = await dev_client.list_billing_projects()
    assert [bp['billing_project'] for bp in r if bp['billing_project'] == project and bp['status'] == 'open'] == [
        project
    ], r


async def test_close_billing_project_with_open_batch_errors(
    make_client: Callable[[str], Awaitable[BatchClient]], dev_client: BatchClient, new_billing_project: str
):
    project = new_billing_project
    await dev_client.add_user("test", project)
    client = await make_client(project)
    b = await client.create_batch()._open_batch()

    try:
        await dev_client.close_billing_project(project)
    except aiohttp.ClientResponseError as e:
        assert e.status == 403, str((e, await b.debug_info()))
    else:
        assert False, str(await b.debug_info())
    await client._patch(f'/api/v1alpha/batches/{b.id}/close')


async def test_close_nonexistent_billing_projet(dev_client: BatchClient):
    try:
        await dev_client.close_billing_project("nonexistent_project")
    except aiohttp.ClientResponseError as e:
        assert e.status == 404, e
    else:
        assert False, 'expected error'


async def test_add_user_with_nonexistent_billing_project(dev_client: BatchClient):
    try:
        await dev_client.add_user("test", "nonexistent_project")
    except aiohttp.ClientResponseError as e:
        assert e.status == 404, e
    else:
        assert False, 'expected error'


async def test_remove_user_with_nonexistent_billing_project(dev_client: BatchClient):
    try:
        await dev_client.remove_user("test", "nonexistent_project")
    except aiohttp.ClientResponseError as e:
        assert e.status == 404, e
    else:
        assert False, 'expected error'


async def test_delete_billing_project_only_when_closed(dev_client: BatchClient, new_billing_project: str):
    project = new_billing_project
    try:
        await dev_client.delete_billing_project(project)
    except aiohttp.ClientResponseError as e:
        assert e.status == 403, e
    else:
        assert False, 'expected error'

    await dev_client.close_billing_project(project)
    await dev_client.delete_billing_project(project)
    # test idempotent
    await dev_client.delete_billing_project(project)

    try:
        await dev_client.get_billing_project(project)
    except aiohttp.ClientResponseError as e:
        assert e.status == 403, e
    else:
        assert False, 'expected error'

    try:
        await dev_client.reopen_billing_project(project)
    except aiohttp.ClientResponseError as e:
        assert e.status == 403, e
    else:
        assert False, 'expected error'


async def test_add_and_delete_user(dev_client: BatchClient, new_billing_project: str):
    project = new_billing_project
    r = await dev_client.add_user('test', project)
    # test idempotent
    r = await dev_client.add_user('test', project)
    assert r['user'] == 'test'
    assert r['billing_project'] == project

    bp = await dev_client.get_billing_project(project)
    assert r['user'] in bp['users'], bp

    r = await dev_client.remove_user('test', project)
    # test idempotent
    r = await dev_client.remove_user('test', project)

    assert r['user'] == 'test'
    assert r['billing_project'] == project

    bp = await dev_client.get_billing_project(project)
    assert r['user'] not in bp['users']


async def test_edit_billing_limit_dev(dev_client: BatchClient, new_billing_project: str):
    project = new_billing_project
    r = await dev_client.add_user('test', project)
    assert r['user'] == 'test'
    assert r['billing_project'] == project

    limit: Optional[int] = 5
    r = await dev_client.edit_billing_limit(project, limit)
    assert r['limit'] == limit
    r = await dev_client.get_billing_project(project)
    assert r['limit'] == limit

    limit = None
    r = await dev_client.edit_billing_limit(project, limit)
    assert r['limit'] is None
    r = await dev_client.get_billing_project(project)
    assert r['limit'] is None

    try:
        bad_limit = 'foo'
        r = await dev_client.edit_billing_limit(project, bad_limit)
    except aiohttp.ClientResponseError as e:
        assert e.status == 400, e
    else:
        r = await dev_client.get_billing_project(project)
        assert r['limit'] is None, r
        assert False, 'expected error'

    try:
        limit = -1
        r = await dev_client.edit_billing_limit(project, limit)
    except aiohttp.ClientResponseError as e:
        assert e.status == 400, e
    else:
        r = await dev_client.get_billing_project(project)
        assert r['limit'] is None, r
        assert False, 'expected error'


async def test_edit_billing_limit_nondev(
    make_client: Callable[[str], Awaitable[BatchClient]], dev_client: BatchClient, new_billing_project: str
):
    project = new_billing_project
    r = await dev_client.add_user('test', project)
    assert r['user'] == 'test'
    assert r['billing_project'] == project

    client = await make_client(project)

    try:
        limit = 5
        await client.edit_billing_limit(project, limit)
    except aiohttp.ClientResponseError as e:
        assert e.status == 401, e
    else:
        r = await client.get_billing_project(project)
        assert r['limit'] is None, r
        assert False, 'expected error'


async def test_billing_project_accrued_costs(
    make_client: Callable[[str], Awaitable[BatchClient]], dev_client: BatchClient, new_billing_project: str
):
    project = new_billing_project
    r = await dev_client.add_user('test', project)
    assert r['user'] == 'test'
    assert r['billing_project'] == project
    r = await dev_client.get_billing_project(project)
    assert r['limit'] is None
    assert r['accrued_cost'] == 0

    client = await make_client(project)

    def approx_equal(x, y, tolerance=1e-10):
        return abs(x - y) <= tolerance

    bb = client.create_batch()
    j1_1 = bb.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'head'])
    j1_2 = bb.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'head'])
    b1 = await bb.submit()

    bb = client.create_batch()
    j2_1 = bb.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'head'])
    j2_2 = bb.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'head'])
    b2 = await bb.submit()

    await b1.wait()
    await b2.wait()

    # Mitigation for https://github.com/hail-is/hail-production-issues/issues/3
    await asyncio.sleep(15)
    b1_status = await b1.status()
    b2_status = await b2.status()

    b1_expected_cost = (await j1_1.status())['cost'] + (await j1_2.status())['cost']
    assert approx_equal(b1_expected_cost, b1_status['cost']), str(
        (b1_expected_cost, b1_status['cost'], await b1.debug_info(), await b2.debug_info())
    )

    b2_expected_cost = (await j2_1.status())['cost'] + (await j2_2.status())['cost']
    assert approx_equal(b2_expected_cost, b2_status['cost']), str(
        (b2_expected_cost, b2_status['cost'], await b1.debug_info(), await b2.debug_info())
    )

    cost_by_batch = b1_status['cost'] + b2_status['cost']
    cost_by_billing_project = (await dev_client.get_billing_project(project))['accrued_cost']

    assert approx_equal(cost_by_batch, cost_by_billing_project), str(
        (cost_by_batch, cost_by_billing_project, await b1.debug_info(), await b2.debug_info())
    )


async def test_billing_limit_zero(
    make_client: Callable[[str], Awaitable[BatchClient]], dev_client: BatchClient, new_billing_project: str
):
    project = new_billing_project
    r = await dev_client.add_user('test', project)
    assert r['user'] == 'test'
    assert r['billing_project'] == project
    r = await dev_client.get_billing_project(project)
    assert r['limit'] is None
    assert r['accrued_cost'] == 0

    limit = 0
    r = await dev_client.edit_billing_limit(project, limit)
    assert r['limit'] == limit

    client = await make_client(project)

    try:
        bb = client.create_batch()
        b = await bb.submit()
    except aiohttp.ClientResponseError as e:
        assert e.status == 403 and 'has exceeded the budget' in e.message
    else:
        assert False, str(await b.debug_info())


async def test_billing_limit_tiny(
    make_client: Callable[[str], Awaitable[BatchClient]], dev_client: BatchClient, new_billing_project: str
):
    project = new_billing_project
    r = await dev_client.add_user('test', project)
    assert r['user'] == 'test'
    assert r['billing_project'] == project
    r = await dev_client.get_billing_project(project)
    assert r['limit'] is None
    assert r['accrued_cost'] == 0

    limit = 0.00001
    r = await dev_client.edit_billing_limit(project, limit)
    assert r['limit'] == limit

    client = await make_client(project)

    batch = client.create_batch()
    j1 = batch.create_job(DOCKER_ROOT_IMAGE, command=['sleep', '30'])
    j2 = batch.create_job(DOCKER_ROOT_IMAGE, command=['sleep', '30'], parents=[j1])
    j3 = batch.create_job(DOCKER_ROOT_IMAGE, command=['sleep', '30'], parents=[j2])
    j4 = batch.create_job(DOCKER_ROOT_IMAGE, command=['sleep', '30'], parents=[j3])
    j5 = batch.create_job(DOCKER_ROOT_IMAGE, command=['sleep', '30'], parents=[j4])
    j6 = batch.create_job(DOCKER_ROOT_IMAGE, command=['sleep', '30'], parents=[j5])
    j7 = batch.create_job(DOCKER_ROOT_IMAGE, command=['sleep', '30'], parents=[j6])
    j8 = batch.create_job(DOCKER_ROOT_IMAGE, command=['sleep', '30'], parents=[j7])
    j9 = batch.create_job(DOCKER_ROOT_IMAGE, command=['sleep', '30'], parents=[j8])
    batch.create_job(DOCKER_ROOT_IMAGE, command=['sleep', '5'], parents=[j9])
    batch = await batch.submit()
    batch_status = await batch.wait()
    assert batch_status['state'] == 'cancelled', str(await batch.debug_info())


async def search_batches(client, expected_batch_id, q):
    found = False
    batches = [x async for x in client.list_batches(q=q, last_batch_id=expected_batch_id + 1, limit=200)]
    for batch in batches:
        if batch.id == expected_batch_id:
            found = True
            break
    return found, [b.id for b in batches]


async def test_user_can_access_batch_made_by_other_user_in_shared_billing_project(
    make_client: Callable[[str], Awaitable[BatchClient]], dev_client: BatchClient, new_billing_project: str
):
    project = new_billing_project

    r = await dev_client.add_user("test", project)
    assert r['user'] == 'test'
    assert r['billing_project'] == project

    r = await dev_client.add_user("test-dev", project)
    assert r['user'] == 'test-dev'
    assert r['billing_project'] == project

    user1_client = await make_client(project)
    b = user1_client.create_batch()
    j = b.create_job(DOCKER_ROOT_IMAGE, command=['sleep', '30'])
    b_handle = await b.submit()

    user2_client = dev_client
    user2_batch = await user2_client.get_batch(b_handle.id)
    user2_job = await user2_client.get_job(j.batch_id, j.job_id)

    await user2_job.attempts()
    await user2_job.log()
    await user2_job.status()

    # list batches results for user1
    found, batches = await search_batches(user1_client, b_handle.id, q='')
    assert found, str((b_handle.id, batches, await b_handle.debug_info()))

    found, batches = await search_batches(user1_client, b_handle.id, q=f'billing_project:{project}')
    assert found, str((b_handle.id, batches, await b_handle.debug_info()))

    found, batches = await search_batches(user1_client, b_handle.id, q='user:test')
    assert found, str((b_handle.id, batches, await b_handle.debug_info()))

    found, batches = await search_batches(user1_client, b_handle.id, q='billing_project:foo')
    assert not found, str((b_handle.id, batches, await b_handle.debug_info()))

    found, batches = await search_batches(user1_client, b_handle.id, q=None)
    assert found, str((b_handle.id, batches, await b_handle.debug_info()))

    found, batches = await search_batches(user1_client, b_handle.id, q='user:test-dev')
    assert not found, str((b_handle.id, batches, await b_handle.debug_info()))

    # list batches results for user2
    found, batches = await search_batches(user2_client, b_handle.id, q='')
    assert found, str((b_handle.id, batches, await b_handle.debug_info()))

    found, batches = await search_batches(user2_client, b_handle.id, q=f'billing_project:{project}')
    assert found, str((b_handle.id, batches, await b_handle.debug_info()))

    found, batches = await search_batches(user2_client, b_handle.id, q='user:test')
    assert found, str((b_handle.id, batches, await b_handle.debug_info()))

    found, batches = await search_batches(user2_client, b_handle.id, q='billing_project:foo')
    assert not found, str((b_handle.id, batches, await b_handle.debug_info()))

    found, batches = await search_batches(user2_client, b_handle.id, q=None)
    assert not found, str((b_handle.id, batches, await b_handle.debug_info()))

    found, batches = await search_batches(user2_client, b_handle.id, q='user:test-dev')
    assert not found, str((b_handle.id, batches, await b_handle.debug_info()))

    await user2_batch.status()
    await user2_batch.cancel()
    await user2_batch.delete()

    # make sure deleted batches don't show up
    found, batches = await search_batches(user1_client, b_handle.id, q='')
    assert not found, str((b_handle.id, batches, await b_handle.debug_info()))


async def test_batch_cannot_be_accessed_by_users_outside_the_billing_project(
    make_client: Callable[[str], Awaitable[BatchClient]], dev_client: BatchClient, new_billing_project: str
):
    project = new_billing_project

    r = await dev_client.add_user("test", project)
    assert r['user'] == 'test'
    assert r['billing_project'] == project

    user1_client = await make_client(project)
    bb = user1_client.create_batch()
    j = bb.create_job(DOCKER_ROOT_IMAGE, command=['sleep', '30'])
    b = await bb.submit()

    user2_client = dev_client
    user2_batch = Batch(user2_client, b.id, b.attributes, b.n_jobs, b.token)

    try:
        try:
            await user2_client.get_batch(b.id)
        except aiohttp.ClientResponseError as e:
            assert e.status == 404, str((e, await b.debug_info()))
        else:
            assert False, str(await b.debug_info())

        try:
            await user2_client.get_job(j.batch_id, j.job_id)
        except aiohttp.ClientResponseError as e:
            assert e.status == 404, str((e, await b.debug_info()))
        else:
            assert False, str(await b.debug_info())

        try:
            await user2_client.get_job_log(j.batch_id, j.job_id)
        except aiohttp.ClientResponseError as e:
            assert e.status == 404, str((e, await b.debug_info()))
        else:
            assert False, str(await b.debug_info())

        try:
            await user2_client.get_job_attempts(j.batch_id, j.job_id)
        except aiohttp.ClientResponseError as e:
            assert e.status == 404, str((e, await b.debug_info()))
        else:
            assert False, str(await b.debug_info())

        try:
            await user2_batch.status()
        except aiohttp.ClientResponseError as e:
            assert e.status == 404, str((e, await b.debug_info()))
        else:
            assert False, str(await b.debug_info())

        try:
            await user2_batch.cancel()
        except aiohttp.ClientResponseError as e:
            assert e.status == 404, str((e, await b.debug_info()))
        else:
            assert False, str(await b.debug_info())

        try:
            await user2_batch.delete()
        except aiohttp.ClientResponseError as e:
            assert e.status == 404, str((e, await b.debug_info()))
        else:
            assert False, str(await b.debug_info())

        # list batches results for user2
        found, batches = await search_batches(user2_client, b.id, q='')
        assert not found, str((b.id, batches, await b.debug_info()))

        found, batches = await search_batches(user2_client, b.id, q=f'billing_project:{project}')
        assert not found, str((b.id, batches, await b.debug_info()))

        found, batches = await search_batches(user2_client, b.id, q='user:test')
        assert not found, str((b.id, batches, await b.debug_info()))

        found, batches = await search_batches(user2_client, b.id, q=None)
        assert not found, str((b.id, batches, await b.debug_info()))

        found, batches = await search_batches(user2_client, b.id, q='user:test-dev')
        assert not found, str((b.id, batches, await b.debug_info()))

    finally:
        await b.delete()


async def test_deleted_open_batches_do_not_prevent_billing_project_closure(
    make_client: Callable[[str], Awaitable[BatchClient]],
    dev_client: BatchClient,
    random_billing_project_name: Callable[[], str],
):
    project = await dev_client.create_billing_project(random_billing_project_name)
    try:
        await dev_client.add_user('test', project)
        client = await make_client(project)
        open_batch = await client.create_batch()._open_batch()
        await open_batch.delete()
    finally:
        await dev_client.close_billing_project(project)
