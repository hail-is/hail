from table import Table
import pymysql
import os
from utils import get_secrets

HOST = os.environ.get('SQL_HOST')


class MigrateTable(Table):
    def __init__(self):
        conn_props = get_secrets('create-users', 'default')
        print(conn_props)
        self.cnx = pymysql.connect(
            password=conn_props['password'],
            user=conn_props['user'],
            host=HOST if HOST is not None else conn_props['host'],
            cursorclass=pymysql.cursors.DictCursor)

    def up(self):
        with self.cnx.cursor() as cursor:
            cursor.execute("CREATE DATABASE IF NOT EXISTS users;")
            cursor.execute("USE users;")
            cursor.execute(
                """
                    CREATE TABLE IF NOT EXISTS user_data (
                        id INT NOT NULL AUTO_INCREMENT,
                        username VARCHAR(255) NOT NULL,
                        user_id VARCHAR(255) NOT NULL,
                        email VARCHAR(255) NOT NULL,
                        developer TINYINT,
                        service_account TINYINT,
                        namespace_name VARCHAR(255),
                        gsa_email VARCHAR(255) NOT NULL,
                        ksa_name VARCHAR(255),
                        bucket_name VARCHAR(255) NOT NULL,
                        gsa_key_secret_name VARCHAR(255) NOT NULL,
                        jwt_secret_name VARCHAR(255) NOT NULL,
                        PRIMARY KEY (id),
                        INDEX email (email),
                        INDEX username (username),
                        UNIQUE KEY user_id (user_id)
                    ) ENGINE=INNODB;
                """
            )

            self.cnx.commit()


if __name__ == "__main__":
    migrations = MigrateTable()
    migrations.up()

    print("Success")
