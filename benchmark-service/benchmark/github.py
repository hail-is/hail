import asyncio
import aiohttp
import gidgethub.aiohttp
import logging
import argparse
import json
from hailtop.utils import retry_long_running
import hailtop.batch_client.aioclient as bc
from google.cloud import storage
import os
from .benchmark_configuration import BENCHMARK_BUCKET_NAME
from .utils import submit_batch

log = logging.getLogger('benchmark')
START_POINT = '2020-09-24T00:00:00Z'

#bucket_name = 'hail-benchmarks'
#main_commit_sha = 'ef7262d01f2bde422aaf09b6f84091ac0e439b1d'


async def get_new_commits(github_client, batch_client, gs_reader):
    #async with aiohttp.ClientSession() as session:
    #request_string = f'/repos/hail-is/hail/commits?sha={main_commit_sha}&since={START_POINT}'
    request_string = f'/repos/hail-is/hail/commits?since={START_POINT}'

    data = await github_client.getitem(request_string)
    new_commits = []
    for commit in data:
        sha = commit.get('sha')
        bc = await batch_client()
        batches = [b async for b in bc.list_batches(q=f'sha={sha} running')]

        # def has_results_file():
        #     name = f'gs://{bucket_name}/benchmark-test/{sha}'
        #     storage_client = storage.Client()
        #     bucket = storage_client.bucket(bucket_name)
        #     stats = storage.Blob(bucket=bucket, name=name).exists(storage_client)
        #     return stats
        file_path = f'gs://{BENCHMARK_BUCKET_NAME}/benchmark-test/{sha}'
        has_results_file = gs_reader.file_exists(file_path)

        if not batches and not has_results_file:
            new_commits.append(commit)
    return new_commits

#
# async def submit_batch(commit, batch_client):
#     sha = commit.get('sha')
#     batch = batch_client.create_batch()
#     job = batch.create_job(image='ubuntu:18.04',
#                            command=['/bin/bash', '-c', 'touch /io/test; sleep 300'],
#                            output_files=[('/io/test', f'gs://{bucket_name}/benchmark-test/{sha}')])
#     await batch.submit(disable_progress_bar=True)
#     log.info(f'submitting batch for commit {sha}')
#     # global START_POINT
#     # START_POINT = commit.get('commit').get('date')
#     return job.batch_id


async def query_github(github_client, batch_client, gs_reader):
    new_commits = await get_new_commits(github_client, batch_client, gs_reader)
    log.info('got new commits')
    for commit in new_commits:
        await submit_batch(commit, batch_client)
        sha = commit.get('sha')
        log.info(f'submitted a batch for commit {sha}')
