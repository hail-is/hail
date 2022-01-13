import os
import sys
import base64
import string
import json
import secrets
import asyncio
from shlex import quote as shq
from hailtop.utils import check_shell, check_shell_output
from hailtop.auth.sql_config import create_secret_data_from_config, SQLConfig
from gear import Database

assert len(sys.argv) == 1
create_database_config = json.load(sys.stdin)


def generate_token(size=12):
    assert size > 0
    alpha = string.ascii_lowercase
    alnum = string.ascii_lowercase + string.digits
    return secrets.choice(alpha) + ''.join([secrets.choice(alnum) for _ in range(size - 1)])


async def write_user_config(namespace: str, database_name: str, user: str, config: SQLConfig):
    with open('/sql-config/server-ca.pem', 'r') as f:
        server_ca = f.read()
    with open('/sql-config/client-cert.pem', 'r') as f:
        client_cert = f.read()
    with open('/sql-config/client-key.pem', 'r') as f:
        client_key = f.read()
    secret = create_secret_data_from_config(config, server_ca, client_cert, client_key)
    files = secret.keys()
    for fname, data in secret.items():
        with open(os.path.basename(fname), 'w') as f:
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
    with open('/sql-config/sql-config.json', 'r') as f:
        sql_config = SQLConfig.from_json(f.read())

    cloud = create_database_config.get('cloud', 'gcp')
    namespace = create_database_config['namespace']
    database_name = create_database_config['database_name']
    cant_create_database = create_database_config['cant_create_database']

    if cant_create_database:
        assert sql_config.db is not None

        await write_user_config(namespace, database_name, 'admin', sql_config)
        await write_user_config(namespace, database_name, 'user', sql_config)
        return

    scope = create_database_config['scope']
    _name = create_database_config['_name']
    admin_username = create_database_config['admin_username']
    user_username = create_database_config['user_username']

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

    with open(create_database_config['admin_password_file']) as f:
        admin_password = f.read()

    with open(create_database_config['user_password_file']) as f:
        user_password = f.read()

    await db.just_execute(
        f'''
CREATE DATABASE IF NOT EXISTS `{_name}`;

CREATE USER IF NOT EXISTS '{admin_username}'@'%' IDENTIFIED BY '{admin_password}';
GRANT ALL ON `{_name}`.* TO '{admin_username}'@'%';

CREATE USER IF NOT EXISTS '{user_username}'@'%' IDENTIFIED BY '{user_password}';
GRANT SELECT, INSERT, UPDATE, DELETE, EXECUTE ON `{_name}`.* TO '{user_username}'@'%';
'''
    )

    # Azure MySQL requires that usernames follow username@servername format
    if cloud == 'azure':
        config_admin_username = admin_username + '@' + sql_config.instance
        config_user_username = user_username + '@' + sql_config.instance
    else:
        assert cloud == 'gcp'
        config_admin_username = admin_username
        config_user_username = user_username
    await write_user_config(
        namespace,
        database_name,
        'admin',
        SQLConfig(
            host=sql_config.host,
            port=sql_config.port,
            instance=sql_config.instance,
            connection_name=sql_config.connection_name,
            user=config_admin_username,
            password=admin_password,
            db=_name,
            ssl_ca=sql_config.ssl_ca,
            ssl_cert=sql_config.ssl_cert,
            ssl_key=sql_config.ssl_key,
            ssl_mode=sql_config.ssl_mode,
        ),
    )

    await write_user_config(
        namespace,
        database_name,
        'user',
        SQLConfig(
            host=sql_config.host,
            port=sql_config.port,
            instance=sql_config.instance,
            connection_name=sql_config.connection_name,
            user=config_user_username,
            password=user_password,
            db=_name,
            ssl_ca=sql_config.ssl_ca,
            ssl_cert=sql_config.ssl_cert,
            ssl_key=sql_config.ssl_key,
            ssl_mode=sql_config.ssl_mode,
        ),
    )


did_shutdown = False


async def shutdown():
    global did_shutdown

    if did_shutdown:
        return

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


async def migrate(database_name, db, i, migration):
    print(f'applying migration {i} {migration}')

    # version to migrate to
    # the 0th migration migrates from 1 to 2
    to_version = i + 2

    name = migration['name']
    script = migration['script']

    out, _ = await check_shell_output(f'sha1sum {script} | cut -d " " -f1')
    script_sha1 = out.decode('utf-8').strip()
    print(f'script_sha1 {script_sha1}')

    row = await db.execute_and_fetchone(f'SELECT version FROM `{database_name}_migration_version`;')
    current_version = row['version']

    if current_version + 1 == to_version:
        await shutdown()

        # migrate
        if script.endswith('.py'):
            await check_shell(f'python3 {script}')
        else:
            await check_shell(
                f'''
mysql --defaults-extra-file=/sql-config.cnf <{script}
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
        assert name == row['name']
        assert script_sha1 == row['script_sha1']


async def async_main():
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

    with open('/sql-config.json', 'wb') as f:
        f.write(base64.b64decode(admin_secret['data']['sql-config.json']))

    with open('/sql-config.cnf', 'wb') as f:
        f.write(base64.b64decode(admin_secret['data']['sql-config.cnf']))

    os.environ['HAIL_DATABASE_CONFIG_FILE'] = '/sql-config.json'
    os.environ['HAIL_SCOPE'] = scope
    os.environ['HAIL_CLOUD'] = cloud

    db = Database()
    await db.async_init()

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

    migrations = create_database_config['migrations']
    for i, m in enumerate(migrations):
        await migrate(database_name, db, i, m)


loop = asyncio.get_event_loop()
loop.run_until_complete(async_main())
