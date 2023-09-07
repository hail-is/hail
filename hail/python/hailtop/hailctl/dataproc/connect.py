from enum import Enum
import os
import platform
import shutil
import subprocess
import tempfile

from typing import Optional, List


from . import gcloud


class DataprocConnectService(str, Enum):
    NOTEBOOK = 'notebook'
    NB = 'nb'
    SPARK_UI = 'spark-ui'
    UI = 'ui'
    SPARK_HISTORY = 'spark-history'
    HIST = 'hist'

    def shortcut(self):
        if self == self.UI:
            return self.SPARK_UI
        if self == self.HIST:
            return self.SPARK_HISTORY
        if self == self.NB:
            return self.NOTEBOOK

        return self


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


def connect(
    name: str,
    service: DataprocConnectService,
    project: Optional[str],
    port: str,
    zone: Optional[str],
    dry_run: bool,
    pass_through_args: List[str],
):
    from hailtop.utils import secret_alnum_string  # pylint: disable=import-outside-toplevel

    service = service.shortcut()

    # Dataproc port mapping
    dataproc_port_and_path = {
        DataprocConnectService.SPARK_UI: '18080/?showIncomplete=true',
        DataprocConnectService.SPARK_HISTORY: '18080',
        DataprocConnectService.NOTEBOOK: '8123',
    }
    connect_port_and_path = dataproc_port_and_path[service]

    zone = zone if zone else gcloud.get_config("compute/zone")
    if not zone:
        raise RuntimeError(
            "Could not determine compute zone. Use --zone argument to hailctl, or use `gcloud config set compute/zone <my-zone>` to set a default."
        )

    account = gcloud.get_config("account")
    if account:
        account = account[0 : account.find('@')]
        ssh_login = '{}@{}-m'.format(account, name)
    else:
        ssh_login = '{}-m'.format(name)

    cmd = [
        'compute',
        'ssh',
        ssh_login,
        '--zone={}'.format(zone),
        '--ssh-flag=-D {}'.format(port),
        '--ssh-flag=-N',
        '--ssh-flag=-f',
        '--ssh-flag=-n',
        *pass_through_args,
    ]

    if project:
        cmd.append(f"--project={project}")

    print('gcloud command:')
    print(' '.join(cmd[:4]) + ' \\\n    ' + ' \\\n    '.join([f"'{x}'" for x in cmd[4:]]))

    if not dry_run:
        print("Connecting to cluster '{}'...".format(name))

        # open SSH tunnel to master node
        gcloud.run(cmd)

        chrome = os.environ.get('HAILCTL_CHROME') or get_chrome_path()

        # open Chrome with SOCKS proxy configuration
        with subprocess.Popen(
            [  # pylint: disable=consider-using-with
                chrome,
                'http://localhost:{}'.format(connect_port_and_path),
                '--proxy-server=socks5://localhost:{}'.format(port),
                '--host-resolver-rules=MAP * 0.0.0.0 , EXCLUDE localhost',
                '--proxy-bypass-list=<-loopback>',  # https://chromium.googlesource.com/chromium/src/+/da790f920bbc169a6805a4fb83b4c2ab09532d91
                '--user-data-dir={}'.format(
                    os.path.join(tempfile.gettempdir(), 'hailctl-dataproc-connect-' + secret_alnum_string(6))
                ),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ):
            pass
