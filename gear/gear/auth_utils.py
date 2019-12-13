import secrets
import base64


async def insert_user(dbpool, spec):
    assert all(k in spec for k in ('state', 'username'))

    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                f'''
INSERT INTO users ({', '.join(spec.keys())})
VALUES ({', '.join([f'%({k})s' for k in spec.keys()])})
''',
                spec)
            return cursor.lastrowid


# 2592000s = 30d
async def create_session(dbpool, user_id, max_age_secs=2592000):
    session_id = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('ascii')
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('INSERT INTO sessions (session_id, user_id, max_age_secs) VALUES (%s, %s, %s);',
                                 (session_id, user_id, max_age_secs))
    return session_id
