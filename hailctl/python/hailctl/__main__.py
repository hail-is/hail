import argparse
import hailctl
import sys


def print_help():
    main_parser = argparse.ArgumentParser(prog='hailctl',
                                          description='Manage and monitor Hail deployments.')
    subs = main_parser.add_subparsers()

    subs.add_parser('dataproc',
                    help='Manage Google Dataproc clusters configured for Hail.',
                    description='Manage Google Dataproc clusters configured for Hail.')
    main_parser.add_argument('--version', action='store_true', help='print version information and exit')
    main_parser.print_help()


def main():
    modules = {
        'dataproc': hailctl.dataproc.cli,
    }

    if '--version' in sys.argv:
        print(hailctl.version())
        sys.exit(0)

    if len(sys.argv) == 1:
        print_help()
        sys.exit(0)
    else:
        main_module = sys.argv[1]
        args = sys.argv[2:]
        module = modules.get(main_module)
        if not module:
            # no module by this name
            print_help()
            sys.exit(0)
        module.main(args)


if __name__ == '__main__':
    main()
