import os
import sys

import argparse
import re
import time

from . import version
from . import dataproc
from . import auth
from . import dev
from . import batch
from . import curl
from . import config


def parser():
    root_parser = argparse.ArgumentParser(
        prog='hailctl',
        description='Manage and monitor Hail deployments.')
    # we have to set dest becuase of a rendering bug in argparse
    # https://bugs.python.org/issue29298
    subparsers = root_parser.add_subparsers(
        title='hailctl subcommand',
        dest='hailctl subcommand',
        required=True)

    version.init_parser(subparsers)
    dataproc.init_parser(subparsers)
    auth.init_parser(subparsers)
    dev.init_parser(subparsers)
    batch.init_parser(subparsers)
    curl.init_parser(subparsers)
    config.init_parser(subparsers)

    return root_parser


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


def main(args=None):
    check_for_update()

    p = parser()
    args, unknown_args = p.parse_known_args(args=args)
    print(dir(args))
    if 'allow_unknown_args' in args:
        args.unknown_args = unknown_args
    elif unknown_args:
        p.print_usage(file=sys.stderr)
        print(f'hailctl: error: unrecognized arguments: {" ".join(unknown_args)}', file=sys.stderr)
        sys.exit(1)

    if args.module.startswith('hailctl version'):
        version.main(args)
    elif args.module.startswith('hailctl dataproc'):
        dataproc.main(args)
    elif args.module.startswith('hailctl auth'):
        auth.main(args)
    elif args.module.startswith('hailctl dev'):
        dev.main(args)
    elif args.module.startswith('hailctl batch'):
        batch.main(args)
    elif args.module.startswith('hailctl curl'):
        curl.main(args)
    else:
        assert args.module.startswith('hailctl config')
        config.main(args)
