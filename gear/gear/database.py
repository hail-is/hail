import json
import aiomysql


async def create_database_pool(autocommit=True):
    with open('/sql-config/sql-config.json', 'r') as f:
        sql_config = json.loads(f.read())
    return await aiomysql.create_pool(host=sql_config['host'],
                                      port=sql_config['port'],
                                      db=sql_config['db'],
                                      user=sql_config['user'],
                                      password=sql_config['password'],
                                      charset='utf8',
                                      cursorclass=aiomysql.cursors.DictCursor,
                                      autocommit=autocommit)
