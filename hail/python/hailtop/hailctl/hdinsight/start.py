import re
import os
import sys
import time
import json
import subprocess
import requests
from shlex import quote as shq

from ... import pip_version
from ...utils import secret_alnum_string


def exec(*args):
    subprocess.check_call(args)


def init_parser(parser):
    parser.add_argument('cluster_name', type=str, help='Cluster name.')
    parser.add_argument('storage_account', type=str, help='Storage account in which to create a container for ephemeral cluster data.')
    parser.add_argument('resource_group', type=str, help='Resource group in which to place cluster.')
    parser.add_argument('--http-password', type=str, help='Password for web access. If unspecified one will be generated.')
    parser.add_argument('--sshuser-password', type=str, help='Password for ssh access. If unspecified one will be generated.')
    parser.add_argument('--location', type=str, default='eastus', help='Azure location in which to place the cluster.')
    parser.add_argument('--num-workers', type=str, default='2', help='Initial number of workers.')
    parser.add_argument('--install-hail-uri',
                        type=str,
                        default=f'https://raw.githubusercontent.com/hail-is/hail/{pip_version()}/hail/python/hailtop/hailctl/hdinsight/resources/install-hail.sh',
                        help='A custom install hail bash script to use. Must be accessible by the cluster\'s head nodes. http(s) and wasb(s) protocols are both acceptable')
    parser.add_argument('--install-native-deps-uri',
                        type=str,
                        default=f'https://raw.githubusercontent.com/hail-is/hail/{pip_version()}/hail/python/hailtop/hailctl/hdinsight/resources/install-native-deps.sh',
                        help='A custom install hail bash script to use. Must be accessible by the cluster\'s nodes. http(s) and wasb(s) protocols are both acceptable')
    parser.add_argument('--wheel-uri',
                        type=str,
                        default=f'https://storage.googleapis.com/hail-common/azure-hdinsight-wheels/hail-{pip_version()}-py3-none-any.whl',
                        help='A custom wheel file to use. Must be accessible by the cluster\'s head nodes. only http(s) protocol is acceptable')
    parser.add_argument('--vep',
                        help='Install VEP for the specified reference genome.',
                        required=False,
                        choices=['GRCh37', 'GRCh38'])
    parser.add_argument('--vep-loftee-uri',
                        type=str,
                        default=None,
                        help='(REQUIRED FOR VEP) A folder file containing the VEP loftee data files. There are tarred, requester-pays copies available at gs://hail-REGION-vep/loftee-beta/GRCh38.tar and gs://hail-REGION-vep/loftee-beta/GRCh37.tar where REGION is one of us, eu, uk, and aus-sydney. Must be accessible by the cluster\'s head nodes. Must be an Azure blob storage URI like https://account.blob.core.windows.net/container/foo. See the Azure-specific VEP instructions in the Hail documentation.')
    parser.add_argument('--vep-homo-sapiens-uri',
                        type=str,
                        default=None,
                        help='(REQUIRED FOR VEP) A folder file containing the VEP homo sapiens data files. There are tarred, requester-pays copies available at gs://hail-REGION-vep/homo-sapiens/95_GRCh38.tar and gs://hail-REGION-vep/homo-sapiens/85_GRCh37.tar where REGION is one of us, eu, uk, and aus-sydney. Must be accessible by the cluster\'s head nodes. Must be an Azure blob storage URI like https://account.blob.core.windows.net/container/foo. See the Azure-specific VEP instructions in the Hail documentation.')
    parser.add_argument('--vep-config-uri',
                        type=str,
                        default=None,
                        help='A VEP config to use. Must be accessible by the cluster\'s head nodes. Only http(s) protocol is acceptable.')
    parser.add_argument('--install-vep-uri',
                        type=str,
                        default=f'https://raw.githubusercontent.com/hail-is/hail/{pip_version()}/hail/python/hailtop/hailctl/hdinsight/resources/install-vep.sh',
                        help='A custom VEP install script to use. Must be accessible by the cluster\'s nodes. http(s) and wasb(s) protocols are both acceptable')


async def main(args, pass_through_args):
    print(f'Starting the cluster {args.cluster_name}')

    sshuser_password = args.sshuser_password
    if sshuser_password is None:
        sshuser_password = secret_alnum_string(12) + '_aA0'

    http_password = args.http_password
    if http_password is None:
        http_password = secret_alnum_string(12) + '_aA0'

    exec('az', 'hdinsight', 'create',
         '--name', args.cluster_name,
         '--resource-group', args.resource_group,
         '--type', 'spark',
         '--component-version', 'Spark=3.0',
         '--http-password', http_password,
         '--http-user', 'admin',
         '--location', args.location,
         '--workernode-count', args.num_workers,
         '--ssh-password', sshuser_password,
         '--ssh-user', 'sshuser',
         '--storage-account', args.storage_account,
         '--storage-container', args.cluster_name,
         '--version', '4.0',
         *pass_through_args)

    print(f'Installing Hail on {args.cluster_name}')
    wheel_pip_version_match = re.match('[^-]*-([^-]*)-.*.whl', os.path.basename(args.wheel_uri))
    assert wheel_pip_version_match
    wheel_pip_version, = wheel_pip_version_match.groups()
    exec('az', 'hdinsight', 'script-action', 'execute', '-g', args.resource_group, '-n', 'installhail',
         '--cluster-name', args.cluster_name,
         '--script-uri', args.install_hail_uri,
         '--roles', 'headnode', 'workernode',
         '--persist-on-success',
         '--script-parameters', f'{args.wheel_uri} {wheel_pip_version} {args.cluster_name}')

    print(f'Installing Hail\'s native dependencies on {args.cluster_name}')
    exec('az', 'hdinsight', 'script-action', 'execute', '-g', args.resource_group, '-n', 'installnativedeps',
         '--cluster-name', args.cluster_name,
         '--script-uri', args.install_native_deps_uri,
         '--roles', 'headnode', 'workernode',
         '--persist-on-success')

    if args.vep:
        if args.vep == 'GRCh38':
            image = 'konradjk/vep95_loftee:0.2'
        elif args.vep == 'GRCh37':
            image = 'konradjk/vep85_loftee:1.0.3'
        else:
            print(f'unknown reference genome {args.vep}')
            sys.exit(1)

        vep_config_uri = args.vep_config_uri
        if vep_config_uri is None:
            vep_config_uri = f'https://raw.githubusercontent.com/hail-is/hail/{pip_version()}/hail/python/hailtop/hailctl/hdinsight/resources/vep-{args.vep}.json'

        print(f'Loading VEP into ABS container {args.cluster_name}')
        for uri in [args.vep_loftee_uri, args.vep_homo_sapiens_uri]:
            exec('az', 'storage', 'copy', '--recursive', '--source', uri, '--destination', f'https://{args.storage_account}.blob.core.windows.net/{args.cluster_name}/')

        print(f'Installing VEP on {args.cluster_name}')
        exec('az', 'hdinsight', 'script-action', 'execute', '-g', args.resource_group, '-n', 'installvep',
             '--cluster-name', args.cluster_name,
             '--script-uri', args.install_vep_uri,
             '--roles', 'headnode', 'workernode',
             '--persist-on-success',
             '--script-parameters', f'/{os.path.basename(args.vep_loftee_uri)} /{os.path.basename(args.vep_homo_sapiens_uri)} {args.vep} {image} {vep_config_uri}')

    def put_jupyter(command):
        # I figured this out after looking at
        # https://docs.microsoft.com/en-us/azure/hdinsight/hdinsight-hadoop-manage-ambari-rest-api#restart-a-service-component
        # and doing some trial and error
        requests.put(
            f'https://{shq(args.cluster_name)}.azurehdinsight.net/api/v1/clusters/{shq(args.cluster_name)}/services/JUPYTER/',
            headers={'Content-Type': 'application/json', 'X-Requested-By': 'ambari'},
            json=command,
            auth=requests.auth.HTTPBasicAuth('admin', args.http_password),
            timeout=60,
        )

    stop = json.dumps({
        "RequestInfo": {"context": "put services into STOPPED state"},
        "Body": {"ServiceInfo": {"state" : "INSTALLED"}}
    })
    start = json.dumps({
        "RequestInfo": {"context": "put services into STARTED state"},
        "Body": {"ServiceInfo": {"state" : "STARTED"}}
    })

    print('Restarting Jupyter ...')
    put_jupyter(stop)
    time.sleep(10)
    put_jupyter(start)

    print(f'''Your cluster is ready.
Web username: admin
Web password: {http_password}
Jupyter URL: https://{args.cluster_name}.azurehdinsight.net/jupyter/tree

SSH username: sshuser
SSH password: {sshuser_password}
SSH domain name: {args.cluster_name}-ssh.azurehdinsight.net

Use the "Python3 (ipykernel)" kernel.
''')
