class CallError(Exception):
    def __init__(self, rv):
        super().__init__(rv)
        self.rv = rv


async def check_call_procedure(db, sql, args=None):
    rv = await db.execute_and_fetchone(sql, args)
    if rv['rc'] != 0:
        raise CallError(rv)
    return rv
