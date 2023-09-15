import asyncio
import base64
import json
import os
import secrets
import string
import sys
from shlex import quote as shq
from typing import Optional

import orjson

from gear import Database, resolve_test_db_endpoint
from hailtop.auth.sql_config import SQLConfig, create_secret_data_from_config
from hailtop.utils import check_shell, check_shell_output

create_database_config = None


def generate_token(size=12):
    assert size > 0
    alpha = string.ascii_lowercase
    alnum = string.ascii_lowercase + string.digits
    return secrets.choice(alpha) + ''.join([secrets.choice(alnum) for _ in range(size - 1)])


async def write_user_config(namespace: str, database_name: str, user: str, config: SQLConfig):
    with open('/sql-config/server-ca.pem', 'r', encoding='utf-8') as f:
        server_ca = f.read()
    client_cert: Optional[str]
    client_key: Optional[str]
    if config.using_mtls():
        with open('/sql-config/client-cert.pem', 'r', encoding='utf-8') as f:
            client_cert = f.read()
        with open('/sql-config/client-key.pem', 'r', encoding='utf-8') as f:
            client_key = f.read()
    else:
        client_cert = None
        client_key = None
    secret = create_secret_data_from_config(config, server_ca, client_cert, client_key)
    files = secret.keys()
    for fname, data in secret.items():
        with open(os.path.basename(fname), 'w', encoding='utf-8') as f:
            f.write(data)
    secret_name = f'sql-{database_name}-{user}-config'
    print(f'creating secret {secret_name}')
    from_files = ' '.join(f'--from-file={f}' for f in files)
    await check_shell(
        f'''
kubectl -n {shq(namespace)} create secret generic \
        {shq(secret_name)} \
        {from_files} \
        --save-config --dry-run=client \
        -o yaml \
        | kubectl -n {shq(namespace)} apply -f -
'''
    )


async def create_database():
    with open('/sql-config/sql-config.json', 'r', encoding='utf-8') as f:
        sql_config = SQLConfig.from_json(f.read())

    assert create_database_config
    namespace = create_database_config['namespace']
    database_name = create_database_config['database_name']
    scope = create_database_config['scope']
    _name = create_database_config['_name']

    db = Database()
    await db.async_init()

    if scope == 'deploy':
        assert _name == database_name

        # create if not exists
        rows = db.execute_and_fetchall(f"SHOW DATABASES LIKE '{database_name}';")
        rows = [row async for row in rows]
        if len(rows) > 0:
            assert len(rows) == 1
            return

    async def create_user_if_doesnt_exist(admin_or_user, mysql_username, mysql_password):
        if admin_or_user == 'admin':
            allowed_operations = 'ALL'
        else:
            assert admin_or_user == 'user'
            allowed_operations = 'SELECT, INSERT, UPDATE, DELETE, EXECUTE, CREATE TEMPORARY TABLES'

        existing_user = await db.execute_and_fetchone('SELECT 1 FROM mysql.user WHERE user=%s', (mysql_username,))
        if existing_user is not None:
            await db.just_execute(
                f'''
                GRANT {allowed_operations} ON `{_name}`.* TO '{mysql_username}'@'%';
                '''
            )
            return

        await db.just_execute(
            f'''
            CREATE USER '{mysql_username}'@'%' IDENTIFIED BY '{mysql_password}';
            GRANT {allowed_operations} ON `{_name}`.* TO '{mysql_username}'@'%';
            '''
        )

        await write_user_config(
            namespace,
            database_name,
            admin_or_user,
            SQLConfig(
                host=sql_config.host,
                port=sql_config.port,
                instance=sql_config.instance,
                connection_name=sql_config.connection_name,
                user=mysql_username,
                password=mysql_password,
                db=_name,
                ssl_ca=sql_config.ssl_ca,
                ssl_cert=sql_config.ssl_cert,
                ssl_key=sql_config.ssl_key,
                ssl_mode=sql_config.ssl_mode,
            ),
        )

    admin_username = create_database_config['admin_username']
    user_username = create_database_config['user_username']

    with open(create_database_config['admin_password_file'], encoding='utf-8') as f:
        admin_password = f.read()
    with open(create_database_config['user_password_file'], encoding='utf-8') as f:
        user_password = f.read()

    await db.just_execute(f'CREATE DATABASE IF NOT EXISTS `{_name}`')
    await create_user_if_doesnt_exist('admin', admin_username, admin_password)
    await create_user_if_doesnt_exist('user', user_username, user_password)


did_shutdown = False


async def shutdown():
    global did_shutdown

    if did_shutdown:
        return

    assert create_database_config
    shutdowns = create_database_config['shutdowns']
    if shutdowns:
        for s in shutdowns:
            assert s['kind'] == 'Deployment'
            await check_shell(
                f'''
kubectl -n {s["namespace"]} delete --ignore-not-found=true deployment {s["name"]}
'''
            )

    did_shutdown = True


async def migrate(database_name, db, mysql_cnf_file, i, migration):
    print(f'applying migration {i} {migration}')

    # version to migrate to
    # the 0th migration migrates from 1 to 2
    to_version = i + 2

    name = migration['name']
    script = migration['script']
    online = migration.get('online', False)

    out, _ = await check_shell_output(f'sha1sum {script} | cut -d " " -f1')
    script_sha1 = out.decode('utf-8').strip()
    print(f'script_sha1 {script_sha1}')

    row = await db.execute_and_fetchone(f'SELECT version FROM `{database_name}_migration_version`;')
    current_version = row['version']

    if current_version + 1 == to_version:
        if not online:
            await shutdown()

        # migrate
        if script.endswith('.py'):
            await check_shell(f'python3 {script}')
        else:
            await check_shell(
                f'''
mysql --defaults-extra-file={mysql_cnf_file} <{script}
'''
            )

        await db.just_execute(
            f'''
UPDATE `{database_name}_migration_version`
SET version = %s;

INSERT INTO `{database_name}_migrations` (version, name, script_sha1)
VALUES (%s, %s, %s);
''',
            (to_version, to_version, name, script_sha1),
        )
    else:
        assert current_version >= to_version

        # verify checksum
        row = await db.execute_and_fetchone(
            f'SELECT * FROM `{database_name}_migrations` WHERE version = %s;', (to_version,)
        )
        assert row is not None
        assert name == row['name'], row
        assert script_sha1 == row['script_sha1'], row


async def async_main():
    global create_database_config
    create_database_config = json.load(sys.stdin)

    await create_database()

    namespace = create_database_config['namespace']
    scope = create_database_config['scope']
    cloud = create_database_config['cloud']
    database_name = create_database_config['database_name']

    admin_secret_name = f'sql-{database_name}-admin-config'
    out, _ = await check_shell_output(
        f'''
kubectl -n {namespace} get -o json secret {shq(admin_secret_name)}
'''
    )
    admin_secret = json.loads(out)

    admin_sql_config = SQLConfig.from_json(base64.b64decode(admin_secret['data']['sql-config.json']).decode())
    if namespace != 'default' and admin_sql_config.host.endswith('.svc.cluster.local'):
        admin_sql_config = await resolve_test_db_endpoint(admin_sql_config)

    with open('/sql-config.json', 'wb') as f:
        f.write(orjson.dumps(admin_sql_config.to_dict()))

    with open('/sql-config.cnf', 'wb') as f:
        f.write(admin_sql_config.to_cnf().encode('utf-8'))

    os.environ['HAIL_DATABASE_CONFIG_FILE'] = '/sql-config.json'
    os.environ['HAIL_SCOPE'] = scope
    os.environ['HAIL_CLOUD'] = cloud
    os.environ['HAIL_NAMESPACE'] = namespace

    db = Database()
    await db.async_init()

    await create_migration_tables(db, database_name)
    migrations = create_database_config['migrations']
    for i, m in enumerate(migrations):
        await migrate(database_name, db, '/sql-config.cnf', i, m)


async def create_migration_tables(db: Database, database_name: str):
    rows = db.execute_and_fetchall(f"SHOW TABLES LIKE '{database_name}_migration_version';")
    rows = [row async for row in rows]
    if len(rows) == 0:
        await db.just_execute(
            f'''
CREATE TABLE `{database_name}_migration_version` (
  `version` BIGINT NOT NULL
) ENGINE = InnoDB;
INSERT INTO `{database_name}_migration_version` (`version`) VALUES (1);

CREATE TABLE `{database_name}_migrations` (
  `version` BIGINT NOT NULL,
  `name` VARCHAR(100),
  `script_sha1` VARCHAR(40),
  PRIMARY KEY (`version`)
) ENGINE = InnoDB;
'''
        )


if __name__ == '__main__':
    assert len(sys.argv) == 1
    asyncio.run(async_main())
