import subprocess
import sys


def init_parser(parser):
    parser.add_argument('name', type=str, help='Cluster name.')
    parser.add_argument('storage_account', type=str, help='Storage account in which cluster\'s container exists.')
    parser.add_argument('resource_group', type=str, help='Resource group in which cluster exists.')
    parser.add_argument('--extra-hdinsight-delete-args', nargs='+', help='Storage account in which cluster\'s container exists.')
    parser.add_argument('--extra-storage-delete-args', nargs='+', help='Storage account in which cluster\'s container exists.')


async def main(args, pass_through_args):
    print("Stopping cluster '{}'...".format(args.name))

    if len(pass_through_args) > 0:
        print('Received too many arguments, did you intend to use --extra-hdinsight-delete-args '
              f'or --extra-storage-delete-args? Excess arguments were {pass_through_args}')
        sys.exit(1)

    subprocess.check_call(
        ['az', 'hdinsight', 'delete',
         '--name', args.name,
         '--resource-group', args.resource_group,
         *(args.extra_hdinsight_delete_args or [])])
    subprocess.check_call(
        ['az', 'storage', 'container', 'delete',
         '--name', args.name,
         '--account-name', args.storage_account,
         *(args.extra_storage_delete_args or [])])
