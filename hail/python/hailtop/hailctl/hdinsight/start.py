import re
import os
from enum import Enum
import sys
import time
import json
import subprocess
from shlex import quote as shq

from typing import List, Optional


def exec(*args):
    subprocess.check_call(args)


class VepVersion(str, Enum):
    GRCH37 = 'GRCh37'
    GRCH38 = 'GRCh38'


def start(
    cluster_name: str,
    storage_account: str,
    resource_group: str,
    http_password: Optional[str],
    sshuser_password: Optional[str],
    location: str,
    num_workers: int,
    install_hail_uri: str,
    install_native_deps_uri: str,
    wheel_uri: str,
    vep: Optional[VepVersion],
    vep_loftee_uri: Optional[str],
    vep_homo_sapiens_uri: Optional[str],
    vep_config_uri: Optional[str],
    install_vep_uri: str,
    pass_through_args: List[str],
):
    import requests  # pylint: disable=import-outside-toplevel
    import requests.auth  # pylint: disable=import-outside-toplevel
    from ...utils import secret_alnum_string  # pylint: disable=import-outside-toplevel
    from ... import pip_version  # pylint: disable=import-outside-toplevel

    print(f'Starting the cluster {cluster_name}')

    if sshuser_password is None:
        sshuser_password = secret_alnum_string(12) + '_aA0'

    if http_password is None:
        http_password = secret_alnum_string(12) + '_aA0'

    exec(
        'az',
        'hdinsight',
        'create',
        '--name',
        cluster_name,
        '--resource-group',
        resource_group,
        '--type',
        'spark',
        '--component-version',
        'Spark=3.0',
        '--http-password',
        http_password,
        '--http-user',
        'admin',
        '--location',
        location,
        '--workernode-count',
        str(num_workers),
        '--ssh-password',
        sshuser_password,
        '--ssh-user',
        'sshuser',
        '--storage-account',
        storage_account,
        '--storage-container',
        cluster_name,
        '--version',
        '4.0',
        *pass_through_args,
    )

    print(f'Installing Hail on {cluster_name}')
    wheel_pip_version_match = re.match('[^-]*-([^-]*)-.*.whl', os.path.basename(wheel_uri))
    assert wheel_pip_version_match
    (wheel_pip_version,) = wheel_pip_version_match.groups()
    exec(
        'az',
        'hdinsight',
        'script-action',
        'execute',
        '-g',
        resource_group,
        '-n',
        'installhail',
        '--cluster-name',
        cluster_name,
        '--script-uri',
        install_hail_uri,
        '--roles',
        'headnode',
        'workernode',
        '--persist-on-success',
        '--script-parameters',
        f'{wheel_uri} {wheel_pip_version} {cluster_name}',
    )

    print(f"Installing Hail's native dependencies on {cluster_name}")
    exec(
        'az',
        'hdinsight',
        'script-action',
        'execute',
        '-g',
        resource_group,
        '-n',
        'installnativedeps',
        '--cluster-name',
        cluster_name,
        '--script-uri',
        install_native_deps_uri,
        '--roles',
        'headnode',
        'workernode',
        '--persist-on-success',
    )

    if vep:
        if vep == 'GRCh38':
            image = 'konradjk/vep95_loftee:0.2'
        elif vep == 'GRCh37':
            image = 'konradjk/vep85_loftee:1.0.3'
        else:
            print(f'unknown reference genome {vep}')
            sys.exit(1)

        if vep_config_uri is None:
            vep_config_uri = f'https://raw.githubusercontent.com/hail-is/hail/{pip_version()}/hail/python/hailtop/hailctl/hdinsight/resources/vep-{vep}.json'

        if vep_loftee_uri is None or vep_homo_sapiens_uri is None:
            raise ValueError("Both `vep_loftee_uri` and `vep_homo_sapiens_uri` must be specified if `vep` is specified")

        print(f'Loading VEP into ABS container {cluster_name}')
        for uri in [vep_loftee_uri, vep_homo_sapiens_uri]:
            exec(
                'az',
                'storage',
                'copy',
                '--recursive',
                '--source',
                uri,
                '--destination',
                f'https://{storage_account}.blob.core.windows.net/{cluster_name}/',
            )

        print(f'Installing VEP on {cluster_name}')
        exec(
            'az',
            'hdinsight',
            'script-action',
            'execute',
            '-g',
            resource_group,
            '-n',
            'installvep',
            '--cluster-name',
            cluster_name,
            '--script-uri',
            install_vep_uri,
            '--roles',
            'headnode',
            'workernode',
            '--persist-on-success',
            '--script-parameters',
            f'/{os.path.basename(vep_loftee_uri)} /{os.path.basename(vep_homo_sapiens_uri)} {vep} {image} {vep_config_uri}',
        )

    def put_jupyter(command):
        # I figured this out after looking at
        # https://docs.microsoft.com/en-us/azure/hdinsight/hdinsight-hadoop-manage-ambari-rest-api#restart-a-service-component
        # and doing some trial and error
        assert http_password
        requests.put(
            f'https://{shq(cluster_name)}.azurehdinsight.net/api/v1/clusters/{shq(cluster_name)}/services/JUPYTER/',
            headers={'Content-Type': 'application/json', 'X-Requested-By': 'ambari'},
            json=command,
            auth=requests.auth.HTTPBasicAuth('admin', http_password),
            timeout=60,
        )

    stop = json.dumps({
        "RequestInfo": {"context": "put services into STOPPED state"},
        "Body": {"ServiceInfo": {"state": "INSTALLED"}},
    })
    start = json.dumps({
        "RequestInfo": {"context": "put services into STARTED state"},
        "Body": {"ServiceInfo": {"state": "STARTED"}},
    })

    print('Restarting Jupyter ...')
    put_jupyter(stop)
    time.sleep(10)
    put_jupyter(start)

    print(f"""Your cluster is ready.
Web username: admin
Web password: {http_password}
Jupyter URL: https://{cluster_name}.azurehdinsight.net/jupyter/tree

SSH username: sshuser
SSH password: {sshuser_password}
SSH domain name: {cluster_name}-ssh.azurehdinsight.net

Use the "Python3 (ipykernel)" kernel.
""")
