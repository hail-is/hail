import sys

import asyncio
import argparse

from . import start

def parser():
    main_parser = argparse.ArgumentParser(
        prog='hailctl dataproc',
        description='Manage and monitor Hail HDInsight clusters.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = main_parser.add_subparsers()

    start_parser = subparsers.add_parser(
        'start',
        help='Start an HDInsight cluster configured for Hail.',
        description='Start an HDInsight cluster configured for Hail.')

    start_parser.set_defaults(module='start')
    start.init_parser(start_parser)

    return main_parser


def main(args):
    p = parser()
    if not args:
        p.print_help()
        sys.exit(0)
    jmp = {
        'start': start,
        # 'submit': submit,
        # 'connect': connect,
        # 'diagnose': diagnose,
        # 'stop': stop,
        # 'list': list_clusters,
        # 'modify': modify,
        # 'describe': describe,
    }

    args, pass_through_args = p.parse_known_args(args=args)
    if "module" not in args:
        p.error('positional argument required')

    asyncio.get_event_loop().run_until_complete(
        jmp[args.module].main(args, pass_through_args))
'''
curl -u admin:LongPassword1 -H 'X-Requested-By: ambari' -X PUT -d '{
    "RequestInfo": {"context": "put services into STOPPED state"},
    "Body": {"ServiceInfo": {"state" : "INSTALLED"}}
}' https://$cluster_name.azurehdinsight.net/api/v1/clusters/dkingtest25/services/JUPYTER/

sleep 10

curl -u admin:LongPassword1 -H 'X-Requested-By: ambari' -X PUT -d '{
    "RequestInfo": {"context": "put services into STARTED state"},
    "Body": {"ServiceInfo": {"state" : "STARTED"}}
}' https://$cluster_name.azurehdinsight.net/api/v1/clusters/dkingtest25/services/JUPYTER/

'''
