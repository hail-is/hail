import os
import subprocess
from typing import List


def submit(
    name: str,
    storage_account: str,
    http_password: str,
    script: str,
    pass_through_args: List[str],
):
    import requests  # pylint: disable=import-outside-toplevel
    import requests.auth  # pylint: disable=import-outside-toplevel

    from ...utils import sync_sleep_before_try  # pylint: disable=import-outside-toplevel

    print("Submitting to cluster '{}'...".format(name))

    subprocess.check_call(
        [
            'az',
            'storage',
            'copy',
            '--source',
            script,
            '--destination',
            f'https://{storage_account}.blob.core.windows.net/{name}/{os.path.basename(script)}',
        ]
    )
    resp = requests.post(
        f'https://{name}.azurehdinsight.net/livy/batches',
        headers={'Content-Type': 'application/json', 'X-Requested-By': 'admin'},
        json={
            'file': f'wasbs://{name}@{storage_account}.blob.core.windows.net/{os.path.basename(script)}',
            'conf': {
                # NB: Only the local protocol is permitted, the file protocol is banned #security
                'spark.jars': 'local:/usr/bin/anaconda/envs/py37/lib/python3.7/site-packages/hail/backend/hail-all-spark.jar',
                'spark.pyspark.driver.python': '/usr/bin/anaconda/envs/py37/bin/python3',
            },
            'args': pass_through_args,
        },
        auth=requests.auth.HTTPBasicAuth('admin', http_password),
        timeout=60,
    )
    batch = resp.json()
    resp.raise_for_status()
    batch_id = batch['id']

    tries = 1
    while True:
        resp = requests.get(
            f'https://{name}.azurehdinsight.net/livy/batches/{batch_id}',
            auth=requests.auth.HTTPBasicAuth('admin', http_password),
            timeout=60,
        )
        batch = resp.json()
        resp.raise_for_status()
        if batch.get('appId'):
            print(
                f'Job submitted. View logs at: https://{name}.azurehdinsight.net/yarnui/hn/cluster/app/{batch["appId"]}'
            )
            break
        tries += 1
        sync_sleep_before_try(tries, base_delay_ms=10)
