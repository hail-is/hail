import os
import sys

import argparse
import re
import time

from hailtop import version


def print_help():
    main_parser = argparse.ArgumentParser(prog='hailctl',
                                          description='Manage and monitor Hail deployments.')
    subs = main_parser.add_subparsers()

    subs.add_parser('dataproc',
                    help='Manage Google Dataproc clusters configured for Hail.',
                    description='Manage Google Dataproc clusters configured for Hail.')
    subs.add_parser('describe',
                    help='Describe Hail Matrix Table and Table files.',
                    description='Describe Hail Matrix Table and Table files.')
    subs.add_parser('hdinsight',
                    help='Manage Azure HDInsight clusters configured for Hail.',
                    description='Manage Azure HDInsight clusters configured for Hail.')
    subs.add_parser('auth',
                    help='Manage Hail credentials.',
                    description='Manage Hail credentials.')
    subs.add_parser('dev',
                    help='Manage Hail development utilities.',
                    description='Manage Hail development utilities.')
    subs.add_parser('version',
                    help='Print version information and exit.',
                    description='Print version information and exit.')
    subs.add_parser('batch',
                    help='Manage batches running on the batch service managed by the Hail team.',
                    description='Manage batches running on the batch service managed by the Hail team.')
    subs.add_parser('curl',
                    help='Issue authenticated curl requests to Hail infrastructure.',
                    description='Issue authenticated curl requests to Hail infrastructure.')
    subs.add_parser('config',
                    help='Manage Hail configuration.',
                    description='Manage Hail configuration.')

    main_parser.print_help()


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
            # touch the file
            open(check_file, 'w', encoding='utf-8').close()  # pylint: disable=consider-using-with

            import subprocess as sp  # pylint: disable=import-outside-toplevel
            try:
                pip_out = sp.check_output(['pip', 'search', 'hail'], stderr=sp.STDOUT)
            except Exception:  # pylint: disable=broad-except
                pip_out = sp.check_output(['pip3', 'search', 'hail'], stderr=sp.STDOUT)

            latest_match = re.search(r'hail \((\\d+)\.(\\d+)\.(\\d+).*', pip_out.decode())
            assert latest_match
            latest = latest_match.groups()

            installed_match = re.search(r'(\d+)\.(\d+)\.(\d+).*', version())
            assert installed_match
            installed = installed_match.groups()

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


def print_version():
    print(version())


def main():
    check_for_update()

    if len(sys.argv) == 1:
        print_help()
        sys.exit(0)
    else:
        module = sys.argv[1]
        args = sys.argv[2:]
        if module == 'version':
            print_version()
        elif module == 'dataproc':
            from hailtop.hailctl.dataproc import cli as dataproc_cli  # pylint: disable=import-outside-toplevel
            dataproc_cli.main(args)
        elif module == 'describe':
            from hailtop.hailctl.describe import main as describe_main  # pylint: disable=import-outside-toplevel
            describe_main(args)
        elif module == 'hdinsight':
            from hailtop.hailctl.hdinsight import cli as hdinsight_cli  # pylint: disable=import-outside-toplevel
            hdinsight_cli.main(args)
        elif module == 'auth':
            from hailtop.hailctl.auth import cli as auth_cli  # pylint: disable=import-outside-toplevel
            auth_cli.main(args)
        elif module == 'dev':
            from hailtop.hailctl.dev import cli as dev_cli  # pylint: disable=import-outside-toplevel
            dev_cli.main(args)
        elif module == 'batch':
            from hailtop.hailctl.batch import cli as batch_cli  # pylint: disable=import-outside-toplevel
            batch_cli.main(args)
        elif module == 'curl':
            from hailtop.hailctl.curl import main as curl_main  # pylint: disable=import-outside-toplevel
            curl_main(args)
        elif module == 'config':
            from hailtop.hailctl.config import cli as config_cli  # pylint: disable=import-outside-toplevel
            config_cli.main(args)
        elif module in ('-h', '--help', 'help'):
            print_help()
        else:
            sys.stderr.write(f"ERROR: no such module: {module!r}")
            print_help()
            sys.exit(1)


if __name__ == '__main__':
    main()
