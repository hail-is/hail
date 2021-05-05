import sys
import argparse
from hailtop.utils import async_to_blocking
from gear import Database, transaction


async def insert_user_if_not_exists(db, username, email):
    @transaction(db)
    async def insert(tx):
        row = await db.execute_and_fetchone('SELECT id, state FROM users where username = %s;', (username,))
        if row:
            if row['state'] == 'active':
                return None
            return row['id']

        return await db.execute_insertone(
            '''
    INSERT INTO users (state, username, email, is_developer, is_service_account)
    VALUES (%s, %s, %s, %s, %s);
    ''',
            ('creating', username, email, 1, 0),
        )

    return await insert(db)


async def main():
    parser = argparse.ArgumentParser(description='Create initial Hail as a service account.')

    parser.add_argument('username', help='The username of the initial user.')
    parser.add_argument('email', help='The email of the initial user.')

    args = parser.parse_args()

    db = Database()
    await db.async_init(maxsize=50)

    await insert_user_if_not_exists(db, args.username, args.email)


async_to_blocking(main())
