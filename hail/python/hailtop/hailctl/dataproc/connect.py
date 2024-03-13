import os
import platform
import shutil
import subprocess
import tempfile
from enum import Enum
from typing import List, Optional

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


def get_datadir_path():
    from hailtop.utils import secret_alnum_string  # pylint: disable=import-outside-toplevel

    system = platform.system()
    release = platform.uname().release
    is_wsl = system == 'Linux' and ('Microsoft' in release or 'microsoft' in release)

    if not is_wsl:
        return os.path.join(tempfile.mkdtemp('hailctl-dataproc-connect-'))
    return 'C:\\Temp\\hailctl-' + secret_alnum_string(5)


def get_chrome_path():
    system = platform.system()

    if system == 'Darwin':
        return '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

    release = platform.uname().release
    is_wsl = 'Microsoft' in release or 'microsoft' in release

    if system == 'Linux' and not is_wsl:
        for c in ['chromium', 'chromium-browser', 'chrome.exe']:
            chrome = shutil.which(c)
            if chrome:
                return chrome

        raise EnvironmentError("cannot find 'chromium', 'chromium-browser', or 'chrome.exe' on path")

    if system == 'Windows' or (system == 'Linux' and is_wsl):
        # https://stackoverflow.com/questions/40674914/google-chrome-path-in-windows-10
        fnames = [
            '/mnt/c/Program Files/Google/Chrome/Application/chrome.exe',
            '/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe',
            '/mnt/c/Program Files(x86)/Google/Chrome/Application/chrome.exe',
            '/mnt/c/ProgramFiles(x86)/Google/Chrome/Application/chrome.exe',
        ]

        for fname in fnames:
            if os.path.exists(fname):
                return fname

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
        data_dir = os.environ.get('HAILCTL_CHROME_DATA_DIR') or get_datadir_path()

        # open Chrome with SOCKS proxy configuration
        with subprocess.Popen(
            [  # pylint: disable=consider-using-with
                chrome,
                f'http://{name}-m:{connect_port_and_path}',
                f'--proxy-server=socks5://localhost:{port}',
                f'--user-data-dir={data_dir}',
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ):
            pass
