import pymysql
import os
from utils import get_secrets

HOST = os.environ.get('SQL_HOST')


class Table:
    def __init__(self):
        self.conn_props = get_secrets('create-users', 'default')

        if HOST:
            self.conn_props['host'] = HOST

        self.cnx = pymysql.connect(
            db=self.conn_props['db'],
            password=self.conn_props['password'],
            user=self.conn_props['user'],
            host=self.conn_props['host'],
            cursorclass=pymysql.cursors.DictCursor)

    def __del__(self):
        self.cnx.close()

    def get(self, user_id):
        with self.cnx.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM user_data WHERE user_id=%s", (user_id,))
            return cursor.fetchone()

    def create_user_db(self, db_name, admin_role, user_role, admin_pass, user_pass):
        cnx = pymysql.connect(
            password=self.conn_props['password'],
            user=self.conn_props['user'],
            host=self.conn_props['host'],
            cursorclass=pymysql.cursors.DictCursor)

        with cnx.cursor() as cursor:
            rows = cursor.execute("SHOW DATABASES LIKE %s", (db_name, ))

            if rows != 0:
                print(f"{db_name} exists, skipping")
                return True

            edb = cnx.escape_string(db_name)

            aff = cursor.execute(
                f"CREATE DATABASE `{edb}`"
            )

            if aff == 0:
                return True

            cursor.execute(
                "CREATE USER %s@'%%' IDENTIFIED BY %s;",
                (admin_role, admin_pass,)
            )
            cursor.execute(
                f"GRANT ALL ON `{edb}`.* " + "TO %s@'%%';", (admin_role,)
            )
            cursor.execute(
                "CREATE USER %s@'%%' IDENTIFIED BY %s;", (
                    user_role, user_pass,)
            )
            cursor.execute(
                f"GRANT SELECT, INSERT, UPDATE, DELETE ON `{edb}`.* " +
                "TO %s@'%%';", (user_role,)
            )

            cnx.commit()

        cnx.close()

    def insert(self, username, email, user_id, namespace_name, gsa_email, ksa_name, bucket_name,
               gsa_key_secret_name, jwt_secret_name, developer=False, service_account=False):
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
