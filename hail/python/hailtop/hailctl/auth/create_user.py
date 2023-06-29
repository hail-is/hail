from typing import Optional

from hailtop.utils import sleep_and_backoff
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
    namespace: Optional[str] = None,
    wait: bool = False,
):
    try:
        await async_create_user(username, login_id, developer, service_account, hail_identity, hail_credentials_secret_name, namespace=namespace)

        if not wait:
            return

        async def _poll():
            delay = 5
            while True:
                user = await async_get_user(username, namespace)
                if user['state'] == 'active':
                    print(f"Created user '{username}'")
                    return
                assert user['state'] == 'creating'
                delay = await sleep_and_backoff(delay)

        await _poll()
    except Exception as e:
        raise CreateUserException(f"Error while creating user '{username}'") from e
