import os
import subprocess
import requests

from ...utils import sync_sleep_and_backoff


def init_parser(parser):
    parser.add_argument('name', type=str, help='Cluster name.')
    parser.add_argument('storage_account', type=str, help='Storage account in which cluster\'s container exists.')
    parser.add_argument('http_password', type=str, help='Web password for the cluster.')
    parser.add_argument('script', type=str, nargs='?', help='Path to script.')


async def main(args, pass_through_args):  # pylint: disable=unused-argument
    print("Submitting to cluster '{}'...".format(args.name))

    subprocess.check_call(
        ['az', 'storage', 'copy',
         '--source', args.script,
         '--destination', f'https://{args.storage_account}.blob.core.windows.net/{args.name}/{os.path.basename(args.script)}'])
    resp = requests.post(
        f'https://{args.name}.azurehdinsight.net/livy/batches',
        headers={'Content-Type': 'application/json', 'X-Requested-By': 'admin'},
        json={'file': f'wasbs://{args.name}@{args.storage_account}.blob.core.windows.net/{os.path.basename(args.script)}',
              'conf': {
                  # NB: Only the local protocol is permitted, the file protocol is banned #security
                  'spark.jars': 'local:/usr/bin/anaconda/envs/py37/lib/python3.7/site-packages/hail/backend/hail-all-spark.jar',
                  'spark.pyspark.driver.python': '/usr/bin/anaconda/envs/py37/bin/python3',
              },
              'args': pass_through_args},
        auth=requests.auth.HTTPBasicAuth('admin', args.http_password),
        timeout=60)
    batch = resp.json()
    resp.raise_for_status()
    batch_id = batch['id']

    delay = 0.01
    while True:
        resp = requests.get(
            f'https://{args.name}.azurehdinsight.net/livy/batches/{batch_id}',
            auth=requests.auth.HTTPBasicAuth('admin', args.http_password),
            timeout=60)
        batch = resp.json()
        resp.raise_for_status()
        if batch.get('appId'):
            print(f'Job submitted. View logs at: https://{args.name}.azurehdinsight.net/yarnui/hn/cluster/app/{batch["appId"]}')
            break
        delay = sync_sleep_and_backoff(delay)
