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

# Figures out what commits have results already, which commits have running batches,
#   and which commits we need to submit benchmarks for.
# Checks whether a results file for each commit exists in Google Storage or is in a currently running Batch.
# If not, then we submit a batch for that commit.

# you're using a file as a key-value store of whether a job has been completed.
# To get whether a batch has been run, use list_batches(q=f'sha={sha} running')

# get a list and then iterate through it asking if the commits have batches or not?
# query for running batches. And then get the commits for those batches from the result that is returned
# We want to write a result file once it's successful. In the batch. But that will come later.
# I think this will use the lower level batch client. Best to use the async version hailtop.batch_client.aioclient
# results file is in google storage
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
    global START_POINT
    START_POINT = commit.get('commit').get('date')
    log.info(f'LOOK HERE: {job.batch_id}')
    return job.batch_id


async def query_github(github_client, batch_client):
    new_commits = await get_new_commits(github_client, batch_client)
    log.info('got new commits')
    for commit in new_commits:
        await submit_batch(commit, batch_client)
        sha = commit.get('sha')
        log.info(f'submitted a batch for commit {sha}')
        break

# async def github_polling_loop():
#     while True:
#         await query_github()
#         log.info(f'successfully queried github')
#         await asyncio.sleep(60)
#
#
# async def main():
#     #asyncio.ensure_future(retry_long_running('github-polling-loop', github_polling_loop))
#     global batch_client
#     batch_client = await bc.BatchClient(billing_project='hail')
#     await retry_long_running('github-polling-loop', github_polling_loop)
#
#
# if __name__ == '__main__':
#     # parser = argparse.ArgumentParser()
#     # args = parser.parse_args()
#     # message = args.message
#
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(main())
#     loop.run_until_complete(loop.shutdown_asyncgens())
#     loop.close()

