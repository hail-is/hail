import os
import sys
import re
import time
import click


def check_for_update():
    try:
        check_file = os.path.expanduser('~') + '/.hail_version_check'
        if os.path.exists(check_file):
            last_modified = os.stat(check_file).st_ctime_ns

            delta = time.time() - last_modified / 10 ** 9
            assert delta > 0
            day = 60 * 60 * 24
            check_for_update = delta / day > 1
        else:
            check_for_update = True

        if check_for_update:
            open(check_file, 'w').close()  # touch the file

            import subprocess as sp  # pylint: disable=import-outside-toplevel
            try:
                pip_out = sp.check_output(['pip', 'search', 'hail'], stderr=sp.STDOUT)
            except Exception:  # pylint: disable=broad-except
                pip_out = sp.check_output(['pip3', 'search', 'hail'], stderr=sp.STDOUT)

            latest = re.search(r'hail \((\d+)\.(\d+)\.(\d+).*', pip_out.decode()).groups()
            installed = re.search(r'(\d+)\.(\d+)\.(\d+).*', hailctl.version()).groups()

            def int_version(version):
                return tuple(map(int, version))

            def fmt_version(version):
                return '.'.join(version)

            if int_version(latest) > int_version(installed):
                sys.stderr.write(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
                                 f'You have Hail {fmt_version(installed)} installed, '
                                 f'but a newer version {fmt_version(latest)} exists.\n'
                                 f'  To upgrade to the latest version, please run:\n\n'
                                 f'    pip3 install -U hail\n\n'
                                 f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n')
    except Exception:  # pylint: disable=broad-except
        pass


@click.group(
    help="Command line interface and utilities for working with Hail")
def hailctl():
    check_for_update()


def main(*args, **kwargs):
    try:
        hailctl(*args, **kwargs)
    except SystemExit as e:
        if e.code != 0:
            raise
