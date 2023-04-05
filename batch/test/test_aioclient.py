import os

import pytest

from hailtop.batch_client.aioclient import BatchClient

DOCKER_ROOT_IMAGE = os.environ['DOCKER_ROOT_IMAGE']

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def client():
    bc = await BatchClient.create('test')
    yield bc
    await bc.close()


async def test_job(client: BatchClient):
    bb = client.create_batch()
    j = bb.create_job(DOCKER_ROOT_IMAGE, ['echo', 'test'])
    b = await bb.submit()
    status = await j.wait()
    assert 'attributes' not in status, str((status, await b.debug_info()))
    assert status['state'] == 'Success', str((status, await b.debug_info()))
    assert j._get_exit_code(status, 'main') == 0, str((status, await b.debug_info()))
    job_log = await j.log()
    assert job_log['main'] == 'test\n', str((job_log, await b.debug_info()))
    assert await j.is_complete(), str(await b.debug_info())
