import asyncio
import os
from gear import Database
from gear.database import get_sql_config


MYSQL_CONFIG_FILE = os.environ.get('MYSQL_CONFIG_FILE')


async def delete_foreign_key_constraint(db, db_name, table_name, referenced_table_name, referenced_column_names):
    # https://dev.mysql.com/doc/refman/8.0/en/information-schema-key-column-usage-table.html

    assert referenced_column_names
    referenced_column_str = '(' + " OR ".join(['REFERENCED_COLUMN_NAME = %s' for _ in referenced_column_names]) + ')'

    query = f'''
SELECT CONSTRAINT_NAME as constraint_name
FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
WHERE REFERENCED_TABLE_SCHEMA = %s AND
  TABLE_NAME = %s AND
  REFERENCED_TABLE_NAME = %s AND
  {referenced_column_str};
'''

    query_args = [db_name, table_name, referenced_table_name] + referenced_column_names

    records = [record async for record in db.select_and_fetchall(query, query_args)]
    assert len(records) == len(referenced_column_names)

    constraint_names = {record['constraint_name'] for record in records}
    assert len(constraint_names) == 1
    constraint_name = list(constraint_names)[0]

    await db.just_execute(f'ALTER TABLE {table_name} DROP FOREIGN KEY `{constraint_name}`;')


async def main():
    db = Database()
    await db.async_init(config_file=MYSQL_CONFIG_FILE)

    try:
        sql_config = get_sql_config(maybe_config_file=MYSQL_CONFIG_FILE)
        db_name = sql_config.db
        assert db_name is not None

        await delete_foreign_key_constraint(db, db_name, 'attempt_resources', 'resources', ['resource'])
    finally:
        await db.async_close()


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
