import asyncio

from hailtop.auth import async_delete_user


def init_parser(parser):
    parser.add_argument("username", type=str,
                        help="User name to delete.")
    parser.add_argument("--namespace", "-n", type=str,
                        help="Specify namespace for auth server.  (default: from deploy configuration)")
    parser.add_argument("--wait", default=False, action='store_true',
                        help="Wait for the creation of the user to finish")


def main(args, pass_through_args):  # pylint: disable=unused-argument
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_delete_user(args.username, args.namespace, wait=args.wait))
