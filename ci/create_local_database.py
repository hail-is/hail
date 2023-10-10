import asyncio
import os
import tempfile
from typing import List

import typer
import yaml
from create_database import create_migration_tables, migrate

from gear import Database


def main(service: str, database_name: str):
    asyncio.run(async_main(service, database_name))


async def async_main(service: str, database_name: str):
    migrations = read_migrations_from_build_yaml(service)

    db = Database()
    await db.async_init()
    await db.just_execute(f'CREATE DATABASE IF NOT EXISTS `{database_name}`;')

    os.environ['HAIL_SQL_DATABASE'] = database_name
    os.environ['HAIL_SCOPE'] = 'dev'
    os.environ['HAIL_NAMESPACE'] = 'local'

    if 'HAIL_CLOUD' not in os.environ:
        os.environ['HAIL_CLOUD'] = 'gcp'

    # Pick up the `HAIL_SQL_DATABASE` change
    await db.async_close()
    db = Database()
    await db.async_init()

    await create_migration_tables(db, database_name)
    with tempfile.NamedTemporaryFile() as mysql_cnf:
        mysql_cnf.write(
            f'''
[client]
host = 127.0.0.1
user = root
password = pw
database = {database_name}
'''.encode()
        )
        mysql_cnf.flush()
        for i, m in enumerate(migrations):
            await migrate(database_name, db, mysql_cnf.name, i, m)


def read_migrations_from_build_yaml(service: str) -> List[dict]:
    with open('build.yaml', 'r', encoding='utf-8') as f:
        build = yaml.safe_load(f)

    for step in build['steps']:
        if step['name'] == f'{service}_database':
            return [
                {**m, 'script': m['script'].replace('/io/', f'{service}/'), 'online': True} for m in step['migrations']
            ]
    raise ValueError(f'No database for service {service}')


if __name__ == '__main__':
    typer.run(main)
