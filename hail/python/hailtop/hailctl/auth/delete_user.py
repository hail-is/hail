import asyncio

from hailtop.utils import sleep_and_backoff
from hailtop.auth import async_delete_user, async_get_user


class DeleteUserException(Exception):
    pass


def init_parser(parser):
    parser.add_argument("username", type=str,
                        help="User name to delete.")
    parser.add_argument("--namespace", "-n", type=str,
                        help="Specify namespace for auth server.  (default: from deploy configuration)")
    parser.add_argument("--wait", default=False, action='store_true',
                        help="Wait for the creation of the user to finish")


async def async_main(args):
    try:
        await async_delete_user(args.username, args.namespace)

        if not args.wait:
            return

        async def _poll():
            delay = 5
            while True:
                user = await async_get_user(args.username, args.namespace)
                if user['state'] == 'deleted':
                    print(f"Deleted user '{args.username}'")
                    return
                assert user['state'] == 'deleting'
                delay = await sleep_and_backoff(delay)

        await _poll()
    except Exception as e:
        raise DeleteUserException(f"Error while deleting user '{args.username}'") from e


def main(args, pass_through_args):  # pylint: disable=unused-argument
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_main(args))
