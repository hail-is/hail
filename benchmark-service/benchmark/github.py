import asyncio
import aiohttp
import gidgethub.aiohttp
import logging
import argparse
import json
from hailtop.utils import retry_long_running
import hailtop.batch_client.aioclient as bc

log = logging.getLogger('benchmark')
oauth_token = '22a4707d382b1a887640e11a0ff8eeb2b05cfd09'
START_POINT = '2020-08-24T00:00:00Z'

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


async def main():
    async with aiohttp.ClientSession() as session:
        gh = gidgethub.aiohttp.GitHubAPI(session, 'hail-is/hail',
                                         oauth_token=oauth_token)

        request_string = f'/repos/hail-is/hail/commits?sha=ef7262d01f2bde422aaf09b6f84091ac0e439b1d&since={START_POINT}'

        data = await gh.getitem(request_string)
        # parse the resulting data to get a list of commit shas
        list_of_shas = []
        new_commits = []
        running_commits = {}
        batch_client = await bc.BatchClient(billing_project='hail')
        for commit in data:
            sha = commit.get('sha')
            list_of_shas.append(sha)
            # if (sha not in running_commits) and (sha not in result_commit_shas):
            #     new_commits.append(commit)
            batches = batch_client.list_batches(q=f'sha={sha} running')
            p = await batches.asend(None)
            print(p)
            # if batches is not None:
            #     running_commits.add(commit)
        print(list_of_shas)


# async def query_github():
#     ...
#
#
# async def github_polling_loop():
#     while True:
#         await query_github()
#         log.info(f'successfully queried github')
#         await asyncio.sleep(60)
#
#
# async def main():
#     asyncio.ensure_future(retry_long_running('github-polling-loop', github_polling_loop))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    #
    # args = parser.parse_args()
    # message = args.message

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()

