import mysql.connector
import os
import base64
from globals import k8s, kube_client

SQL_HOST_DEF = os.environ.get('SQL_HOST')


class Table:
    @staticmethod
    def getSecret(b64str):
        return base64.b64decode(b64str).decode('utf-8')

    @staticmethod
    def getSecrets():
        secrets = {}

        try:
            res = k8s.read_namespaced_secret('create-users', 'default')
            data = res.data

            if SQL_HOST_DEF is not None:
                host = SQL_HOST_DEF
            else:
                host = Table.getSecret(data['host'])

            secrets['user'] = Table.getSecret(data['user'])
            secrets['password'] = Table.getSecret(data['password'])
            secrets['database'] = Table.getSecret(data['db'])
            secrets['host'] = host
        except kube_client.rest.ApiException as e:
            print(e)

        return secrets

    def __init__(self):
        secrets = Table.getSecrets()

        if not secrets:
            raise "Couldn't read secret"

        self.cnx = mysql.connector.connect(**secrets)

    def get(self, user_id):
        cursor = self.cnx.cursor(dictionary=True)

        cursor.execute("SELECT * FROM user_data WHERE user_id=%s", (user_id,))
        res = cursor.fetchone()
        cursor.close()

        return res

    def insert(self, user_id, gsa_projectId, gsa_email, ksa_name, bucket_name):
        cursor = self.cnx.cursor()
        cursor.execute(
            """
            INSERT INTO user_data
                (user_id, gsa_projectId, gsa_email, ksa_name, bucket_name)
                VALUES (%s, %s, %s, %s, %s)
            """, (user_id, gsa_projectId, gsa_email, ksa_name, bucket_name))
        self.cnx.commit()

        cnt = cursor.rowcount

        cursor.close()

        return cnt == 1


    def delete(self, user_id):
        cursor = self.cnx.cursor()
        cursor.execute("DELETE FROM user_data WHERE user_id=%s", (user_id,))

        self.cnx.commit()

        cnt = cursor.rowcount

        cursor.close()

        return cnt == 1
