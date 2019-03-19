from table import Table
import mysql.connector


class MigrateTable(Table):
    def __init__(self):
        secrets = Table.getSecrets()

        if not secrets:
            raise "Couldn't read secrets"

        self.cnx = mysql.connector.connect(
            host=secrets['host'],
            user=secrets['user'],
            password=secrets['password']
        )

    def up(self):
        cursor = self.cnx.cursor()

        cursor.execute("CREATE DATABASE IF NOT EXISTS users;")
        cursor.execute("USE users;")
        cursor.execute(
            """
                CREATE TABLE IF NOT EXISTS user_data (
                    id INT NOT NULL AUTO_INCREMENT,
                    user_id VARCHAR(255) NOT NULL,
                    gsa_email VARCHAR(255) NOT NULL,
                    ksa_name VARCHAR(255) NOT NULL,
                    bucket_name VARCHAR(255) NOT NULL,
                    PRIMARY KEY (id),
                    UNIQUE INDEX auth0_id (user_id)
                ) ENGINE=INNODB;
            """
        )

        self.cnx.commit()
        cursor.close()


if __name__ == "__main__":
    migrations = MigrateTable()
    migrations.up()

    print("Success")
