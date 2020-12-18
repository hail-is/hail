import os
import platform
import shutil
import subprocess
import tempfile
import click

from . import gcloud
from .dataproc import dataproc


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


@dataproc.command(
    help="Connect to a running Dataproc cluster")
@click.argument('cluster_name')
@click.argument('service',
                type=click.Choice(['notebook', 'nb', 'spark-ui', 'ui', 'spark-history', 'hist'],
                                  case_sensitive=False))
@click.option('--project',
              metavar='GCP_PROJECT',
              help='Google Cloud project for the cluster.')
@click.option('--port', '-p',
              metavar='PORT',
              default='10000',
              type=int,
              help="Local port to use for SSH tunnel to leader (master) node",
              show_default=True)
@click.option('--zone', '-z',
              metavar='GCP_ZONE',
              help='Compute zone for Dataproc cluster.')
@click.option('--dry-run', is_flag=True,
              help="Print gcloud dataproc command, but don't run it")
def connect(cluster_name, service, *, project, port, zone, dry_run):
    # shortcut mapping
    shortcut = {
        'ui': 'spark-ui',
        'hist': 'spark-history',
        'nb': 'notebook'
    }

    service = shortcut.get(service, service)

    # Dataproc port mapping
    dataproc_port_and_path = {
        'spark-ui': '18080/?showIncomplete=true',
        'spark-history': '18080',
        'notebook': '8123'
    }
    connect_port_and_path = dataproc_port_and_path[service]

    zone = zone if zone else gcloud.get_config("compute/zone")
    if not zone:
        raise RuntimeError("Could not determine compute zone. Use --zone argument to hailctl, or use `gcloud config set compute/zone <my-zone>` to set a default.")

    account = gcloud.get_config("account")
    if account:
        account = account[0:account.find('@')]
        ssh_login = '{}@{}-m'.format(account, cluster_name)
    else:
        ssh_login = '{}-m'.format(cluster_name)

    cmd = ['compute',
           'ssh',
           ssh_login,
           '--zone={}'.format(zone),
           '--ssh-flag=-D {}'.format(port),
           '--ssh-flag=-N',
           '--ssh-flag=-f',
           '--ssh-flag=-n']

    if project:
        cmd.append(f"--project={project}")

    print('gcloud command:')
    print(' '.join(cmd[:4]) + ' \\\n    ' + ' \\\n    '.join([f"'{x}'" for x in cmd[4:]]))

    if not dry_run:
        print("Connecting to cluster '{}'...".format(cluster_name))

        # open SSH tunnel to master node
        gcloud.run(cmd)

        chrome = os.environ.get('HAILCTL_CHROME') or get_chrome_path()

        # open Chrome with SOCKS proxy configuration
        with open(os.devnull, 'w') as f:
            subprocess.Popen([
                chrome,
                'http://localhost:{}'.format(connect_port_and_path),
                '--proxy-server=socks5://localhost:{}'.format(port),
                '--host-resolver-rules=MAP * 0.0.0.0 , EXCLUDE localhost',
                '--proxy-bypass-list=<-loopback>',  # https://chromium.googlesource.com/chromium/src/+/da790f920bbc169a6805a4fb83b4c2ab09532d91
                '--user-data-dir={}'.format(tempfile.gettempdir())
            ], stdout=f, stderr=f)
