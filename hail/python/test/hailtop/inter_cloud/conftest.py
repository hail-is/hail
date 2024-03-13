import asyncio
import functools
import os
import secrets
from typing import AsyncIterator, Dict, Tuple

import pytest

from hailtop.aiotools.router_fs import AsyncFS, RouterAsyncFS
from hailtop.utils import bounded_gather2


@pytest.fixture(scope='module')
async def router_filesystem() -> AsyncIterator[Tuple[asyncio.Semaphore, AsyncFS, Dict[str, str]]]:
    token = secrets.token_hex(16)

    async with RouterAsyncFS() as fs:
        file_base = f'/tmp/{token}/'
        await fs.mkdir(file_base)

        gs_bucket = os.environ['HAIL_TEST_GCS_BUCKET']
        gs_base = f'gs://{gs_bucket}/tmp/{token}/'

        s3_bucket = os.environ['HAIL_TEST_S3_BUCKET']
        s3_base = f's3://{s3_bucket}/tmp/{token}/'

        azure_account = os.environ['HAIL_TEST_AZURE_ACCOUNT']
        azure_container = os.environ['HAIL_TEST_AZURE_CONTAINER']
        azure_base = f'https://{azure_account}.blob.core.windows.net/{azure_container}/tmp/{token}/'

        bases = {'file': file_base, 'gs': gs_base, 's3': s3_base, 'azure-https': azure_base}

        sema = asyncio.Semaphore(50)
        async with sema:
            yield (sema, fs, bases)
            await bounded_gather2(
                sema,
                functools.partial(fs.rmtree, sema, file_base),
                functools.partial(fs.rmtree, sema, gs_base),
                functools.partial(fs.rmtree, sema, s3_base),
                functools.partial(fs.rmtree, sema, azure_base),
            )

        assert not await fs.isdir(file_base)
        assert not await fs.isdir(gs_base)
        assert not await fs.isdir(s3_base)
        assert not await fs.isdir(azure_base)
