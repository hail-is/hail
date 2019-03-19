from table import Table
import mysql.connector


class MigrateTable(Table):
    def __init__(self):
        secrets = Table.getSecrets()

        if not secrets:
            raise "Couldn't read user-secrets"

        self.cnx = mysql.connector.connect(
            host=secrets['host'],
            user=secrets['user'],
            password=secrets['password']
        )

        cursor = self.cnx.cursor()

        cursor.execute(
            """
                CREATE DATABASE IF NOT EXISTS users;
            """
        )

        cursor.execute("Use users")

        cursor.execute(
            """
                CREATE TABLE IF NOT EXISTS users (
                    id INT NOT NULL AUTO_INCREMENT,
                    user_id VARCHAR(255) NOT NULL,
                    gsa_name VARCHAR(255) NOT NULL,
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

    print("Success")
    
