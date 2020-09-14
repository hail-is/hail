import asyncio
from hailtop.batch_client.aioclient import BatchClient
from ci.ci.environment import CI_UTILS_IMAGE


async def hello():
   batch_client = await BatchClient('ci')
   hello_batch = batch_client.create_batch()
   job = hello_batch.create_job(CI_UTILS_IMAGE,
                                command='echo "hello world"')
   hello_batch.submit()


async def submit():
   batch_client = await BatchClient('ci')
   hello_batch = batch_client.create_batch()
