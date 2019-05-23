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

    def get(self, username):
        with self.cnx.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM username WHERE username=%s", (user_id,))
            return cursor.fetchone()

    def insert(self, username, email, namespace_name, gsa_email, ksa_name, bucket_name,
               gsa_key_secret_name, jwt_secret_name, developer = 0, service_account = 0):
        with self.cnx.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO user_data
                    (   username,
                        email,
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
                        %d,
                        %d,
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
