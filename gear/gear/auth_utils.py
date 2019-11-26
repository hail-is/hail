import secrets
import base64


async def insert_user(dbpool, spec):
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                '''
INSERT INTO user_data (username, user_id, developer, gsa_email, bucket_name, gsa_key_secret_name, jwt_secret_name,
    service_account)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
''',
                (spec['username'], spec['user_id'], spec.get('developer', 0), spec['gsa_email'], spec['bucket_name'], spec['gsa_key_secret_name'], spec['jwt_secret_name'],
                 spec.get('service_account')))

        async with conn.cursor() as cursor:
            await cursor.execute('SELECT * FROM user_data WHERE user_id = %s', (spec['user_id'],))
            return await cursor.fetchone()


# 2592000s = 30d
async def create_session(dbpool, user_id, max_age_secs=2592000):
    session_id = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('ascii')
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('INSERT INTO sessions (session_id, user_id, max_age_secs) VALUES (%s, %s, %s);',
                                 (session_id, user_id, max_age_secs))
    return session_id
