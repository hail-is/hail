import asyncio
from typing import Any, Dict, Mapping, Optional

import pymysql
from aiohttp import web

from gear import Database

from ..exceptions import QueryError
from .query.job_list import JobListLimits, parse_job_list_params
from .query.job_list_sql import JobGroupNotFound, JobListTimeout, get_job_list

JOB_LIST_LIMITS = JobListLimits()
# Requests holding a database connection at once, per pod. The pool (10 by default) is shared with job submission.
JOB_LIST_CONCURRENCY = 4
JOB_LIST_QUEUE_WAIT_SECS = 2.0
RETRY_AFTER_SECS = 1


def is_transient_db_error(e: BaseException) -> bool:
    """A database error worth retrying: a timeout, a lost or refused connection, lock trouble.

    Checks ``__context__`` too: when the connection drops, rolling back on it fails, and that error replaces the
    original one.
    """
    seen = set()
    exc: Optional[BaseException] = e
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, (pymysql.err.OperationalError, pymysql.err.InterfaceError)):
            return True
        if isinstance(exc, pymysql.err.InternalError) and exc.args and exc.args[0] == 1205:
            return True
        exc = exc.__context__
    return False


def _unavailable(reason: str, text: str) -> web.HTTPServiceUnavailable:
    return web.HTTPServiceUnavailable(reason=reason, text=text, headers={'Retry-After': str(RETRY_AFTER_SECS)})


async def job_list_response(
    db: Database,
    batch_id: int,
    query: Mapping[str, str],
    semaphore: asyncio.Semaphore,
    *,
    limits: JobListLimits = JOB_LIST_LIMITS,
    queue_wait_secs: float = JOB_LIST_QUEUE_WAIT_SECS,
) -> Dict[str, Any]:
    """The /job-list response, or an HTTP error: 400 for a bad request, 404 for an unknown or uncommitted group,
    503 with Retry-After when the database is busy, slow or unreachable."""
    try:
        params = parse_job_list_params(query, limits)
    except QueryError as e:
        raise e.http_response() from e

    # The wait isn't counted against the query time limit, which is database time.
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=queue_wait_secs)
    except asyncio.TimeoutError as e:
        raise _unavailable('Busy', 'Too many job list requests at once; retry shortly.') from e
    try:
        return await get_job_list(db, batch_id, params, limits)
    except QueryError as e:
        raise e.http_response() from e
    except JobGroupNotFound as e:
        raise web.HTTPNotFound() from e
    except JobListTimeout as e:
        text = 'The job list query took too long; retry shortly.'
        if params.filter is not None:
            text += ' If it keeps happening, narrow the filter.'
        raise _unavailable('Timeout', text) from e
    except Exception as e:
        # Not gear's unbounded retries: the client's bounded ones, so a request never overruns its deadline.
        if is_transient_db_error(e):
            raise _unavailable('Database unavailable', 'The database is unavailable; retry shortly.') from e
        raise
    finally:
        semaphore.release()
