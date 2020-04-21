import secrets
import base64


async def insert_user(db, spec):
    assert all(k in spec for k in ('state', 'username'))

    return await db.execute_insertone(
        f'''
INSERT INTO users ({', '.join(spec.keys())})
VALUES ({', '.join([f'%({k})s' for k in spec.keys()])})
''',
        spec)


# 2592000s = 30d
async def create_session(db, user_id, max_age_secs=2592000):
    session_id = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('ascii')
    await db.just_execute('INSERT INTO sessions (session_id, user_id, max_age_secs) VALUES (%s, %s, %s);',
                          (session_id, user_id, max_age_secs))
    return session_id
