from typing import Optional

from hailtop.utils import sleep_before_try
from hailtop.auth import async_create_user, async_get_user


class CreateUserException(Exception):
    pass


async def polling_create_user(
    username: str,
    login_id: str,
    developer: bool,
    service_account: bool,
    hail_identity: Optional[str],
    hail_credentials_secret_name: Optional[str],
    *,
    wait: bool = False,
):
    try:
        await async_create_user(username, login_id, developer, service_account, hail_identity, hail_credentials_secret_name)

        if not wait:
            return

        async def _poll():
            tries = 0
            while True:
                user = await async_get_user(username)
                if user['state'] == 'active':
                    print(f"Created user '{username}'")
                    return
                assert user['state'] == 'creating'
                tries += 1
                await sleep_before_try(tries, base_delay_ms = 5_000)

        await _poll()
    except Exception as e:
        raise CreateUserException(f"Error while creating user '{username}'") from e
