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
from hailtop.hail_logging import configure_logging

configure_logging()
log = logging.getLogger('benchmark')
START_POINT = '2020-09-24T00:00:00Z'

running_commit_shas = {}
result_commit_shas = {}
bucket_name = 'hail-benchmarks'
main_commit_sha = 'ef7262d01f2bde422aaf09b6f84091ac0e439b1d'


async def get_new_commits(github_client, batch_client):
    async with aiohttp.ClientSession() as session:
        # gh = gidgethub.aiohttp.GitHubAPI(session, 'hail-is/hail',
        #                                  oauth_token=os.getenv("GH_AUTH"))

        request_string = f'/repos/hail-is/hail/commits?sha={main_commit_sha}&since={START_POINT}'

        data = await github_client.getitem(request_string)
        list_of_shas = []
        new_commits = []
        for commit in data:
            sha = commit.get('sha')
            list_of_shas.append(sha)
            batches = [b async for b in batch_client.list_batches(q=f'sha={sha} running')]

            def has_results_file():
                name = f'gs://{bucket_name}/benchmark-test/{sha}'
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                stats = storage.Blob(bucket=bucket, name=name).exists(storage_client)
                return stats

            if not batches and not has_results_file():
                new_commits.append(commit)
        return new_commits


async def submit_batch(commit, batch_client):
    sha = commit.get('sha')
    batch = batch_client.create_batch()
    job = batch.create_job(image='ubuntu:18.04',
                           command=['/bin/bash', '-c', 'touch /io/test; sleep 300'],
                           output_files=[('/io/test', f'gs://{bucket_name}/benchmark-test/{sha}')])
    await batch.submit(disable_progress_bar=True)
    log.info(f'submitting batch for commit {sha}')
    global START_POINT
    START_POINT = commit.get('commit').get('date')
    return job.batch_id


async def query_github(github_client, batch_client):
    new_commits = await get_new_commits(github_client, batch_client)
    log.info('got new commits')
    for commit in new_commits:
        await submit_batch(commit, batch_client)
        sha = commit.get('sha')
        log.info(f'submitted a batch for commit {sha}')
        break
