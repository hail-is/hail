from typing import Optional

from hailtop.utils import sleep_before_try
from hailtop.auth import async_delete_user, async_get_user


class DeleteUserException(Exception):
    pass


async def polling_delete_user(
    username: str,
    namespace: Optional[str],
    wait: bool,
):
    try:
        await async_delete_user(username, namespace)

        if not wait:
            return

        async def _poll():
            tries = 1
            while True:
                user = await async_get_user(username, namespace)
                if user['state'] == 'deleted':
                    print(f"Deleted user '{username}'")
                    return
                assert user['state'] == 'deleting'
                tries += 1
                await sleep_before_try(tries, base_delay_ms = 5_000)

        await _poll()
    except Exception as e:
        raise DeleteUserException(f"Error while deleting user '{username}'") from e
