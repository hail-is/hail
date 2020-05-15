import subprocess as sp
import os

import shutil
import tempfile


def init_parser(parser):
    parser.add_argument('name', type=str, help='Cluster name.')
    parser.add_argument('service', type=str,
                        choices=['notebook', 'nb', 'spark-ui', 'ui', 'spark-ui1', 'ui1',
                                 'spark-ui2', 'ui2', 'spark-history', 'hist'],
                        help='Web service to launch.')
    parser.add_argument('--port', '-p', default='10000', type=str,
                        help='Local port to use for SSH tunnel to master node (default: %(default)s).')
    parser.add_argument('--zone', '-z', type=str, help='Compute zone for Dataproc cluster.')
    parser.add_argument('--dry-run', action='store_true', help="Print gcloud dataproc command, but don't run it.")


def main(args, pass_through_args):  # pylint: disable=unused-argument
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

    if args.zone:
        zone = args.zone
    else:
        zone = sp.check_output(["gcloud", "config", "get-value", "compute/zone"], stderr=sp.DEVNULL).decode().strip()

    if not zone:
        raise RuntimeError("Could not determine compute zone. Use --zone argument to hailctl, or use `gcloud config set compute/zone <my-zone>` to set a default.")

    cmd = ['gcloud',
           'compute',
           'ssh',
           '{}-m'.format(args.name),
           '--zone={}'.format(zone),
           '--ssh-flag=-D {}'.format(args.port),
           '--ssh-flag=-N',
           '--ssh-flag=-f',
           '--ssh-flag=-n']

    print('gcloud command:')
    print(' '.join(cmd[:4]) + ' \\\n    ' + ' \\\n    '.join([f"'{x}'" for x in cmd[4:]]))

    if not args.dry_run:
        print("Connecting to cluster '{}'...".format(args.name))

        # open SSH tunnel to master node
        sp.check_call(
            cmd,
            stderr=sp.STDOUT
        )

        import platform
        system = platform.system()

        chrome = os.environ.get('HAILCTL_CHROME')
        if system == 'Darwin':
            chrome = chrome or r'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
        elif system == 'Linux':
            for c in ['chromium', 'chromium-browser']:
                chrome = chrome or shutil.which(c)
            if chrome is None:
                raise EnvironmentError("cannot find 'chromium' or 'chromium-browser' on path")
        elif system == 'Windows':
            chrome = chrome or r'/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe'

        if not chrome:
            raise ValueError(f"unsupported system: {system}, set environment variable HAILCTL_CHROME to a chrome executable")

        # open Chrome with SOCKS proxy configuration
        with open(os.devnull, 'w') as f:
            sp.Popen([
                chrome,
                'http://localhost:{}'.format(connect_port_and_path),
                '--proxy-server=socks5://localhost:{}'.format(args.port),
                '--host-resolver-rules=MAP * 0.0.0.0 , EXCLUDE localhost',
                '--proxy-bypass-list=<-loopback>',  # https://chromium.googlesource.com/chromium/src/+/da790f920bbc169a6805a4fb83b4c2ab09532d91
                '--user-data-dir={}'.format(tempfile.gettempdir())
            ], stdout=f, stderr=f)
