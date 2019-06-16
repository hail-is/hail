import json
import asyncio
import aiomysql
from asyncinit import asyncinit


def run_synchronous(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


@asyncinit
class Database:
    @classmethod
    def create_synchronous(cls, config_file):
        db = object.__new__(cls)
        run_synchronous(cls.__init__(db, config_file))
        return db

    async def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config = json.loads(f.read().strip())

        self.host = config['host']
        self.port = config['port']
        self.user = config['user']
        self.db = config['db']
        self.password = config['password']
        self.charset = 'utf8'

        self.pool = await aiomysql.create_pool(host=self.host,
                                               port=self.port,
                                               db=self.db,
                                               user=self.user,
                                               password=self.password,
                                               charset=self.charset,
                                               cursorclass=aiomysql.cursors.DictCursor,
                                               autocommit=True)


def make_where_statement(items):
    template = []
    values = []
    for k, v in items.items():
        if isinstance(v, list):
            if len(v) == 0:
                template.append("FALSE")
            else:
                template.append(f'`{k.replace("`", "``")}` IN %s')
                values.append(v)
        elif v is None:
            template.append(f'`{k.replace("`", "``")}` IS NULL')
        elif v == "NOT NULL":
            template.append(f'`{k.replace("`", "``")}` IS NOT NULL')
        else:
            template.append(f'`{k.replace("`", "``")}` = %s')
            values.append(v)

    template = " AND ".join(template)
    return template, values


class Table:  # pylint: disable=R0903
    def __init__(self, db, name):
        self.name = name
        self._db = db

    async def new_record(self, **items):
        async with self._db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                names = ", ".join([f'`{name.replace("`", "``")}`' for name in items])
                values_template = ", ".join(["%s" for _ in items.values()])
                sql = f"INSERT INTO `{self.name}` ({names}) VALUES ({values_template})"
                await cursor.execute(sql, tuple(items.values()))
                return cursor.lastrowid  # This returns 0 unless an autoincrement field is in the table

    async def update_record(self, where_items, set_items):
        async with self._db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                if len(set_items) != 0:
                    where_template, where_values = make_where_statement(where_items)
                    set_template = ", ".join([f'`{k.replace("`", "``")}` = %s' for k, v in set_items.items()])
                    set_values = set_items.values()
                    sql = f"UPDATE `{self.name}` SET {set_template} WHERE {where_template}"
                    await cursor.execute(sql, (*set_values, *where_values))

    async def get_records(self, where_items, select_fields=None):
        assert select_fields is None or len(select_fields) != 0
        async with self._db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                where_template, where_values = make_where_statement(where_items)
                select_fields = ",".join(select_fields) if select_fields is not None else "*"
                sql = f"SELECT {select_fields} FROM `{self.name}` WHERE {where_template}"
                await cursor.execute(sql, tuple(where_values))
                return await cursor.fetchall()

    async def get_all_records(self):
        async with self._db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(f"SELECT * FROM `{self.name}`")
                return await cursor.fetchall()

    async def has_record(self, where_items):
        async with self._db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                where_template, where_values = make_where_statement(where_items)
                sql = f"SELECT COUNT(1) FROM `{self.name}` WHERE {where_template}"
                await cursor.execute(sql, where_values)
                result = await cursor.fetchone()
                return result['COUNT(1)'] >= 1

    async def delete_record(self, where_items):
        async with self._db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                where_template, where_values = make_where_statement(where_items)
                sql = f"DELETE FROM `{self.name}` WHERE {where_template}"
                await cursor.execute(sql, tuple(where_values))
