import os
import platform
import shutil
import subprocess
import tempfile

from . import gcloud


def init_parser(parser):
    parser.add_argument('name', type=str, help='Cluster name.')
    parser.add_argument('service', type=str,
                        choices=['notebook', 'nb', 'spark-ui', 'ui', 'spark-history', 'hist'],
                        help='Web service to launch.')
    parser.add_argument('--project', help='Google Cloud project for the cluster (defaults to currently set project).')
    parser.add_argument('--port', '-p', default='10000', type=str,
                        help='Local port to use for SSH tunnel to leader (master) node (default: %(default)s).')
    parser.add_argument('--zone', '-z', type=str, help='Compute zone for Dataproc cluster.')
    parser.add_argument('--dry-run', action='store_true', help="Print gcloud dataproc command, but don't run it.")


def get_chrome_path():
    system = platform.system()

    if system == 'Darwin':
        return '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

    if system == 'Linux':
        for c in ['chromium', 'chromium-browser']:
            chrome = shutil.which(c)
            if chrome:
                return chrome

        raise EnvironmentError("cannot find 'chromium' or 'chromium-browser' on path")

    if system == 'Windows':
        return '/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe'

    raise ValueError(f"unsupported system: {system}, set environment variable HAILCTL_CHROME to a chrome executable")


async def main(args, pass_through_args):  # pylint: disable=unused-argument
    # shortcut mapping
    shortcut = {
        'ui': 'spark-ui',
        'hist': 'spark-history',
        'nb': 'notebook'
    }

    service = args.service
    service = shortcut.get(service, service)

    # Dataproc port mapping
    dataproc_port_and_path = {
        'spark-ui': '18080/?showIncomplete=true',
        'spark-history': '18080',
        'notebook': '8123'
    }
    connect_port_and_path = dataproc_port_and_path[service]

    zone = args.zone if args.zone else gcloud.get_config("compute/zone")
    if not zone:
        raise RuntimeError("Could not determine compute zone. Use --zone argument to hailctl, or use `gcloud config set compute/zone <my-zone>` to set a default.")

    account = gcloud.get_config("account")
    if account:
        account = account[0:account.find('@')]
        ssh_login = '{}@{}-m'.format(account, args.name)
    else:
        ssh_login = '{}-m'.format(args.name)

    cmd = ['compute',
           'ssh',
           ssh_login,
           '--zone={}'.format(zone),
           '--ssh-flag=-D {}'.format(args.port),
           '--ssh-flag=-N',
           '--ssh-flag=-f',
           '--ssh-flag=-n']

    if args.project:
        cmd.append(f"--project={args.project}")

    print('gcloud command:')
    print(' '.join(cmd[:4]) + ' \\\n    ' + ' \\\n    '.join([f"'{x}'" for x in cmd[4:]]))

    if not args.dry_run:
        print("Connecting to cluster '{}'...".format(args.name))

        # open SSH tunnel to master node
        gcloud.run(cmd)

        chrome = os.environ.get('HAILCTL_CHROME') or get_chrome_path()

        # open Chrome with SOCKS proxy configuration
        with open(os.devnull, 'w') as f:
            subprocess.Popen([
                chrome,
                'http://localhost:{}'.format(connect_port_and_path),
                '--proxy-server=socks5://localhost:{}'.format(args.port),
                '--host-resolver-rules=MAP * 0.0.0.0 , EXCLUDE localhost',
                '--proxy-bypass-list=<-loopback>',  # https://chromium.googlesource.com/chromium/src/+/da790f920bbc169a6805a4fb83b4c2ab09532d91
                '--user-data-dir={}'.format(tempfile.gettempdir())
            ], stdout=f, stderr=f)
