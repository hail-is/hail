import pymysql
import os
import base64
from globals import v1

SQL_HOST_DEF = os.environ.get('SQL_HOST')


class Table:
    @staticmethod
    def getSecret(b64str):
        return base64.b64decode(b64str).decode('utf-8')

    @staticmethod
    def getSecrets():
        secrets = {}

        res = v1.read_namespaced_secret('create-users', 'default')
        data = res.data

        if SQL_HOST_DEF is not None:
            host = SQL_HOST_DEF
        else:
            host = Table.getSecret(data['host'])

        secrets['user'] = Table.getSecret(data['user'])
        secrets['password'] = Table.getSecret(data['password'])
        secrets['db'] = Table.getSecret(data['db'])
        secrets['host'] = host

        return secrets

    def __init__(self):
        secrets = Table.getSecrets()

        self.cnx = pymysql.connect(**secrets,
                                   cursorclass=pymysql.cursors.DictCursor)

    def __del__(self):
        self.cnx.close()

    def get(self, user_id):
        with self.cnx.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM user_data WHERE user_id=%s", (user_id,))
            return cursor.fetchone()

    def insert(self, user_id, gsa_email, ksa_name, bucket_name, gsa_key_secret_name):
        with self.cnx.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO user_data
                    (user_id, gsa_email, ksa_name, bucket_name,
                    gsa_key_secret_name)
                    VALUES (%s, %s, %s, %s, %s)
                """, (user_id, gsa_email, ksa_name, bucket_name,
                      gsa_key_secret_name))
            self.cnx.commit()

            assert cursor.rowcount == 1

    def delete(self, user_id):
        with self.cnx.cursor() as cursor:
            cursor.execute(
                "DELETE FROM user_data WHERE user_id=%s", (user_id,))
            self.cnx.commit()

            return cursor.rowcount == 1
