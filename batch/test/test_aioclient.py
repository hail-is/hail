import pytest

from hailtop.batch_client.aioclient import BatchClient

from .utils import DOCKER_ROOT_IMAGE, create_batch

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def client():
    bc = await BatchClient.create('test')
    yield bc
    await bc.close()


async def test_job(client: BatchClient):
    b = create_batch(client)
    j = b.create_job(DOCKER_ROOT_IMAGE, ['echo', 'test'])
    await b.submit()
    status = await j.wait()
    assert 'attributes' not in status, str((status, await b.debug_info()))
    assert status['state'] == 'Success', str((status, await b.debug_info()))
    assert j._get_exit_code(status, 'main') == 0, str((status, await b.debug_info()))
    job_log = await j.log()
    assert job_log['main'] == 'test\n', str((job_log, await b.debug_info()))
    assert await j.is_complete(), str(await b.debug_info())
