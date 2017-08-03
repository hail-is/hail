import os
import signal
import argparse
from subprocess import Popen, check_call, check_output, CalledProcessError

def main(main_parser):

    parser = argparse.ArgumentParser(parents=[main_parser])

    parser.add_argument('service', type=str, nargs='?', default='notebook', choices=['spark-ui', 'spark-ui1', 'spark-ui2', 'spark-history', 'notebook'])
    parser.add_argument('--port', '-p', default='10000', type=str, help='Local port to use for SSH tunnel to master node.')
    parser.add_argument('--zone', '-z', default='us-central1-b', type=str, help='Compute zone for Google cluster.')

    args = parser.parse_args()

    # Dataproc port mapping
    dataproc_ports = {
        'spark-ui': 4040,
        'spark-ui1': 4041,
        'spark-ui2': 4042,
        'spark-history': 18080,
        'notebook': 8123
    }
    connect_port = dataproc_ports[args.service]

    # open SSH tunnel to master node
    cmd = [
        'gcloud',
        'compute',
        'ssh',
        '{}-m'.format(args.name),
        '--zone={}'.format(args.zone),
        '--ssh-flag=-D {}'.format(args.port),
        '--ssh-flag=-N',
        '--ssh-flag=-f',
        '--ssh-flag=-n'
    ]
    with open(os.devnull, 'w') as f:
        check_call(cmd, stdout=f, stderr=f)

    # open Chrome with SOCKS proxy configuration
    cmd = [
        r'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        'http://localhost:{}'.format(connect_port),
        '--proxy-server=socks5://localhost:{}'.format(args.port),
        '--host-resolver-rules=MAP * 0.0.0.0 , EXCLUDE localhost',
        '--user-data-dir=/tmp/'
    ]
    with open(os.devnull, 'w') as f:
        Popen(cmd, stdout=f, stderr=f)
