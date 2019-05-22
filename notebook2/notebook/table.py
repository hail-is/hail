import base64
import pymysql
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
    def get_secret(b64str):
        return base64.b64decode(b64str).decode('utf-8')

    @staticmethod
    def get_secrets():
        secrets = {}

        res = k8s.read_namespaced_secret('get-users', 'default')
        data = res.data

        if SQL_HOST_DEF is not None:
            host = SQL_HOST_DEF
        else:
            host = Table.get_secret(data['host'])

        secrets['user'] = Table.get_secret(data['user'])
        secrets['password'] = Table.get_secret(data['password'])
        secrets['database'] = Table.get_secret(data['db'])
        secrets['host'] = host

        return secrets

    def __init__(self):
        self.connection_params = Table.get_secrets()

    def acquire_connection(self):
        return pymysql.connect(**self.connection_params,
                               cursorclass=pymysql.cursors.DictCursor)

    def get(self, user_id):
        with self.acquire_connection() as cursor:
            cursor.execute(
                """
                SELECT id, gsa_email, ksa_name, bucket_name,
                gsa_key_secret_name, user_jwt_secret_name
                FROM user_data
                WHERE user_id=%s
                """, (user_id,))

            res = cursor.fetchone()
            assert res is not None

            return res
