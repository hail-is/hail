import pytest
from hailtop.batch_client.aioclient import BatchClient

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def client():
    bc = await BatchClient('test')
    yield bc
    await bc.close()


async def test_job(client):
    b = client.create_batch()
    j = b.create_job('ubuntu:18.04', ['echo', 'test'])
    await b.submit()
    status = await j.wait()
    assert 'attributes' not in status, (status, await j.log())
    assert status['state'] == 'Success', (status, await j.log())
    assert j._get_exit_code(status, 'main') == 0, (status, await j.log())
    assert (await j.log())['main'] == 'test\n'
    assert await j.is_complete()
