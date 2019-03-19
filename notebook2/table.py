import base64
import mysql.connector
import kubernetes as kube
import os

if 'BATCH_USE_KUBE_CONFIG' in os.environ:
    kube.config.load_kube_config()
else:
    kube.config.load_incluster_config()
k8s = kube.client.CoreV1Api()

SQL_HOST_DEF = os.environ.get('SQL_HOST')


class Table:
    @staticmethod
    def getSecret(b64str):
        return base64.b64decode(b64str).decode('utf-8')

    @staticmethod
    def getSecrets():
        secrets = {}

        try:
            res = k8s.read_namespaced_secret('get-users', 'default')
            data = res.data

            if SQL_HOST_DEF is not None:
                host = SQL_HOST_DEF
            else:
                host = Table.getSecret(data['host'])

            secrets['user'] = Table.getSecret(data['user'])
            secrets['password'] = Table.getSecret(data['password'])
            secrets['db'] = Table.getSecret(data['db'])
            secrets['host'] = host
        except kube.client.rest.ApiException as e:
            print(e)

        return secrets

    def __init__(self):
        secrets = Table.getSecrets()

        if not secrets:
            raise "Couldn't read user-secrets"

        self.cnx = mysql.connector.connect(**secrets)

    def get(self, user_id):
        cursor = self.cnx.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT id, gsa_name, ksa_name, bucket_name
            FROM users
            WHERE user_id=%s
            """, (user_id,))

        res = cursor.fetchone()
        cursor.close()

        if res is None:
            return {
                'id': None,
                'gsa_name': None,
                'ksa_name': None,
                'bucket_name': None,
            }

        return res