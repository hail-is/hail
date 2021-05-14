import os
import pytest
from hailtop.batch_client.aioclient import BatchClient

DOCKER_ROOT_IMAGE = os.environ.get('DOCKER_ROOT_IMAGE', 'gcr.io/hail-vdc/ubuntu:18.04')

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def client():
    bc = BatchClient('test')
    yield bc
    await bc.close()


async def test_job(client):
    b = await client.create_batch()
    j = b.create_job(DOCKER_ROOT_IMAGE, ['echo', 'test'])
    await b.submit()
    status = await j.wait()
    assert 'attributes' not in status, (status, await j.log())
    assert status['state'] == 'Success', (status, await j.log())
    assert j._get_exit_code(status, 'main') == 0, (status, await j.log())
    assert (await j.log())['main'] == 'test\n'
    assert await j.is_complete()
