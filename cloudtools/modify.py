import os
import sys
from subprocess import Popen, check_call


def init_parser(parser):
    parser.add_argument('name', type=str, help='Cluster name.')
    parser.add_argument('--jar', type=str, help='New JAR.')
    parser.add_argument('--zip', type=str, help='New ZIP.')
    parser.add_argument('--zone', '-z', default='us-central1-b', type=str,
                        help='Compute zone for Dataproc cluster (default: %(default)s).')

def main(args):
    if (args.jar is not None):
        _scp_and_sudo_move(args.jar, args.name, '/home/hail/hail.jar', args.zone)
    if (args.zip is not None):
        _scp_and_sudo_move(args.zip, args.name, '/home/hail/hail.zip', args.zone)

# user doesn't have access to /home/hail/ so we copy then use sudo
def _scp_and_sudo_move(source, destination_host, destination, zone):
    if source.startswith("gs://"):
        cmd = [
            'gcloud',
            'compute',
            'ssh',
            '{}-m'.format(destination_host),
            '--zone={}'.format(zone),
            '--',
            'sudo gsutil cp {} {}'.format(source, destination)
        ]
        check_call(cmd, stdout=sys.stdout, stderr=sys.stderr)
    else:
        cmd = [
            'gcloud',
            'compute',
            'scp',
            '--zone={}'.format(zone),
            source,
            '{}-m:/tmp/foo'.format(destination_host)
        ]
        check_call(cmd, stdout=sys.stdout, stderr=sys.stderr)
        cmd = [
            'gcloud',
            'compute',
            'ssh',
            '{}-m'.format(destination_host),
            '--zone={}'.format(zone),
            '--',
            'sudo mv /tmp/foo {}'.format(destination)
        ]
        check_call(cmd, stdout=sys.stdout, stderr=sys.stderr)
