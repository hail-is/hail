import argparse
import sys

from . import dataproc


def print_help():
    main_parser = argparse.ArgumentParser(prog='hailctl',
                                          description='Manage and monitor Hail deployments.')
    subs = main_parser.add_subparsers()

    subs.add_parser('dataproc',
                    help='Manage Google Dataproc clusters configured for Hail.',
                    description='Manage Google Dataproc clusters configured for Hail.')
    main_parser.print_help()


def main():
    jmp = {
        'dataproc': dataproc.cli,
    }

    if len(sys.argv) == 1:
        print_help()
        sys.exit(0)
    else:
        main_module = sys.argv[1]
        args = sys.argv[2:]
        mod = jmp.get(main_module)
        if not mod:
            # no module by this name
            print_help()
            sys.exit(0)
        mod.main(args)


if __name__ == '__main__':
    main()
