import re
import os
import json
import subprocess
from shlex import quote as shq


def bash(*args):
    subprocess.check_call(args)


from ... import pip_version


def init_parser(parser):
    parser.add_argument('cluster_name', type=str, help='Cluster name.')
    parser.add_argument('http_password', type=str, help='Password for web access.')
    parser.add_argument('sshuser_password', type=str, help='Password for ssh access.')
    parser.add_argument('storage_account', type=str, help='Storage account in which to create a container for ephemeral cluster data.')
    parser.add_argument('resource_group', type=str, help='Resource group in which to place cluster.')
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


async def main(args, pass_through_args):
    print(f'Starting the cluster {args.cluster_name}')
    bash('az', 'hdinsight', 'create',
         '--name', args.cluster_name,
         '--resource-group', args.resource_group,
         '--type', 'spark',
         '--component-version', 'Spark=3.0',
         '--http-password', args.http_password,
         '--http-user', 'admin',
         '--location', args.location,
         '--workernode-count', args.num_workers,
         '--ssh-password', args.sshuser_password,
         '--ssh-user', 'sshuser',
         '--storage-account', args.storage_account,
         '--storage-container', args.cluster_name,
         '--version', '4.0',
         *pass_through_args)

    print(f'Installing Hail on {args.cluster_name}')
    wheel_pip_version, = re.match('[^-]*-([^-]*)-.*.whl', os.path.basename(args.wheel_uri)).groups()
    bash('az', 'hdinsight', 'script-action', 'execute', '-g', args.resource_group, '-n', 'installhail',
         '--cluster-name', args.cluster_name,
         '--script-uri', args.install_hail_uri,
         '--roles', 'headnode',
         '--persist-on-success',
         '--script-parameters', f'{args.wheel_uri} {wheel_pip_version} {args.cluster_name}')

    print(f'Installing Hail\'s native dependencies on {args.cluster_name}')
    bash('az', 'hdinsight', 'script-action', 'execute', '-g', args.resource_group, '-n', 'installnativedeps',
         '--cluster-name', args.cluster_name,
         '--script-uri', args.install_native_deps_uri,
         '--roles', 'headnode', 'workernode',
         '--persist-on-success')

    stop = json.dumps({
        "RequestInfo": {"context": "put services into STOPPED state"},
        "Body": {"ServiceInfo": {"state" : "INSTALLED"}}
    })
    start = json.dumps({
        "RequestInfo": {"context": "put services into STARTED state"},
        "Body": {"ServiceInfo": {"state" : "STARTED"}}
    })
    stop_curl, start_curl = [f'''curl -u admin:{shq(args.http_password)} \
    -H 'X-Requested-By: ambari' \
    -X PUT \
    -d {shq(command)} https://{shq(args.cluster_name)}.azurehdinsight.net/api/v1/clusters/{shq(args.cluster_name)}/services/JUPYTER/'''
                             for command in [stop, start]]

    print('Restarting Jupyter. You will be prompted for the sshuser password.')
    bash('ssh',
         f'sshuser@{args.cluster_name}-ssh.azurehdinsight.net',
         f'set -ex ; {stop_curl} ; sleep 10 ; {start_curl}')
    print(f'''Your cluster is ready.
https://{args.cluster_name}.azurehdinsight.net/jupyter/tree
Web username: admin
SSH username: sshuser
''')
