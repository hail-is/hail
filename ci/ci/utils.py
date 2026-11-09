import datetime
import functools
import secrets
import string
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

from gear import Database
from hailtop.utils import time_msecs


def generate_token(size=12):
    assert size > 0
    alpha = string.ascii_lowercase
    alnum = string.ascii_lowercase + string.digits
    return secrets.choice(alpha) + ''.join([secrets.choice(alnum) for _ in range(size - 1)])


async def add_deployed_services(
    db: Database,
    namespace: str,
    services: List[str],
    expiration_time: Optional[datetime.datetime],
):
    expiration = expiration_time.strftime('%Y-%m-%d %H:%M:%S') if expiration_time else None
    await db.execute_insertone(
        """
INSERT INTO active_namespaces (`namespace`, `expiration_time`)
VALUES (%s, %s) as new_ns
ON DUPLICATE KEY UPDATE expiration_time = new_ns.expiration_time
        """,
        (namespace, expiration),
    )
    await db.execute_many(
        """
INSERT INTO deployed_services (`namespace`, `service`)
VALUES (%s, %s)
ON DUPLICATE KEY UPDATE namespace = namespace;
""",
        [(namespace, service) for service in services],
    )


class GithubStatus(Enum):
    SUCCESS = 'success'
    PENDING = 'pending'
    FAILURE = 'failure'


def github_status(state: Optional[str]) -> GithubStatus:
    """
    Converts a state for a commit status (https://docs.github.com/en/graphql/reference/enums#statusstate)
    or a conclusion for a check (https://docs.github.com/en/graphql/reference/enums#checkconclusionstate)
    from the GraphQL API to a GithubStatus.
    """
    if state is None:
        return GithubStatus.PENDING
    if state in {"PENDING", "EXPECTED", "ACTION_REQUIRED", "STALE"}:
        return GithubStatus.PENDING
    if state in {"FAILURE", "ERROR", "TIMED_OUT", "CANCELLED", "STARTUP_FAILURE", "SKIPPED"}:
        return GithubStatus.FAILURE
    if state in {"SUCCESS", "NEUTRAL"}:
        return GithubStatus.SUCCESS
    raise ValueError(f"Unexpected value for GithubStatus: {state}")


def record_success(f: Callable[..., Awaitable[Any]], last_success_ms: Dict[str, int]) -> Callable[..., Awaitable[Any]]:
    """Wrap f to record the time of each successful call under f's name. Exceptions pass through."""

    @functools.wraps(f)
    async def wrapped(*args, **kwargs):
        result = await f(*args, **kwargs)
        last_success_ms[f.__name__] = time_msecs()
        return result

    return wrapped
