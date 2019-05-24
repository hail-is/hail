import pymysql
import os
from secrets import get_secrets

HOST = os.environ.get('SQL_HOST')


class Table:
    def __init__(self):
        conn_props = get_secrets('create-users', 'default')
        self.cnx = pymysql.connect(
            db=conn_props['db'],
            password=conn_props['password'],
            user=conn_props['user'],
            host=HOST if HOST is not None else conn_props['host'],
            cursorclass=pymysql.cursors.DictCursor)

    def __del__(self):
        self.cnx.close()

    def get(self, user_id):
        with self.cnx.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM user_data WHERE user_id=%s", (user_id,))
            return cursor.fetchone()

    def insert(self, username, email, user_id, namespace_name, gsa_email, ksa_name, bucket_name,
               gsa_key_secret_name, jwt_secret_name, developer = False, service_account = False):
        with self.cnx.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO user_data
                    (   username,
                        email,
                        user_id,
                        developer,
                        service_account,
                        namespace_name,
                        gsa_email,
                        ksa_name,
                        bucket_name,
                        gsa_key_secret_name,
                        jwt_secret_name
                    )
                    VALUES (    
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s
                    )
                """, (
                        username,
                        email,
                        user_id,
                        developer,
                        service_account,
                        namespace_name,
                        gsa_email,
                        ksa_name,
                        bucket_name,
                        gsa_key_secret_name,
                        jwt_secret_name,
                    ))
            self.cnx.commit()

            assert cursor.rowcount == 1

    def delete(self, user_id):
        with self.cnx.cursor() as cursor:
            cursor.execute(
                "DELETE FROM user_data WHERE user_id=%s", (user_id,))
            self.cnx.commit()

            return cursor.rowcount == 1
