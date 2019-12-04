import asyncio
import secrets

from gear import Database
from hailtop.utils import time_msecs

import logging

log = logging.getLogger('database')


class CallError(Exception):
    def __init__(self, rv):
        super().__init__(rv)
        self.rv = rv


async def check_call_procedure(db, sql, args=None):
    rv = await db.execute_and_fetchone(sql, args)
    if rv['rc'] != 0:
        raise CallError(rv)
    return rv


class LeasedDatabase:
    def __init__(self):
        self.db = Database()
        self.lease = asyncio.Event()
        self.token = secrets.token_urlsafe(16)

    async def acquire_lease(self):
        while True:
            try:
                self.lease.clear()
                now = time_msecs()
                log.info(f'acquiring lease token={self.token} now={now}')
                await check_call_procedure(
                    self.db,
                    'CALL acquire_lease(%s, %s, %s);',
                    (self.token, now, now + 30 * 1000))
                self.lease.set()
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except CallError:
                log.exception('in acquiring lease')
            finally:
                await asyncio.sleep(15)

    async def async_init(self, autocommit=True, maxsize=10):
        await self.db.async_init(autocommit=autocommit, maxsize=maxsize)
        self.lease.clear()
        asyncio.ensure_future(self.acquire_lease())

    async def just_execute(self, sql, args=None):
        await self.lease.wait()
        await self.db.just_execute(sql, args)

    async def execute_and_fetchone(self, sql, args=None):
        await self.lease.wait()
        return await self.db.execute_and_fetchone(sql, args)

    async def execute_and_fetchall(self, sql, args=None):
        await self.lease.wait()
        async for row in self.db.execute_and_fetchall(sql, args):
            yield row

    async def execute_insertone(self, sql, args=None):
        await self.lease.wait()
        return self.db.execute_insertone(sql, args)

    async def execute_update(self, sql, args=None):
        await self.lease.wait()
        return self.db.execute_update(sql, args)

    async def async_close(self):
        await self.db.async_close()
