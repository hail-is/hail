import argparse
import sys
from cloudtools import start
from cloudtools import submit
from cloudtools import connect
from cloudtools import diagnose
from cloudtools import stop
from cloudtools import list_clusters
from cloudtools import modify


def main():
    main_parser = argparse.ArgumentParser(description='Deploy and monitor Google Dataproc clusters to use with Hail.')
    subs = main_parser.add_subparsers()

    start_parser = subs.add_parser('start',
                                   help='Start a Dataproc cluster configured for Hail.',
                                   description='Start a Dataproc cluster configured for Hail.')
    submit_parser = subs.add_parser('submit',
                                    help='Submit a Python script to a running Dataproc cluster.',
                                    description='Submit a Python script to a running Dataproc cluster.')
    connect_parser = subs.add_parser('connect',
                                     help='Connect to a running Dataproc cluster.',
                                     description='Connect to a running Dataproc cluster.')
    diagnose_parser = subs.add_parser('diagnose',
                                      help='Diagnose problems in a Dataproc cluster.',
                                      description='Diagnose problems in a Dataproc cluster.')
    stop_parser = subs.add_parser('stop',
                                  help='Shut down a Dataproc cluster.',
                                  description='Shut down a Dataproc cluster.')

    list_parser = subs.add_parser('list',
                                  help='List active Dataproc clusters.',
                                  description='List active Dataproc clusters.')

    modify_parser = subs.add_parser('modify',
                                    help='Modify active Dataproc clusters.',
                                    description='Modify active Dataproc clusters.')

    start_parser.set_defaults(module='start')
    start.init_parser(start_parser)

    submit_parser.set_defaults(module='submit')
    submit.init_parser(submit_parser)

    connect_parser.set_defaults(module='connect')
    connect.init_parser(connect_parser)

    diagnose_parser.set_defaults(module='diagnose')
    diagnose.init_parser(diagnose_parser)

    stop_parser.set_defaults(module='stop')
    stop.init_parser(stop_parser)

    list_parser.set_defaults(module='list')

    modify_parser.set_defaults(module='modify')
    modify.init_parser(modify_parser)

    if len(sys.argv) == 1:
        main_parser.print_help()
        sys.exit(0)

    args = main_parser.parse_args()

    if args.module == 'start':
        start.main(args)

    elif args.module == 'submit':
        submit.main(args)

    elif args.module == 'connect':
        connect.main(args)

    elif args.module == 'diagnose':
        diagnose.main(args)

    elif args.module == 'stop':
        stop.main(args)

    elif args.module == 'list':
        list_clusters.main(args)

    elif args.module == 'modify':
        modify.main(args)


if __name__ == '__main__':
    main()
