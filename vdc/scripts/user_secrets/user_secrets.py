import shortuuid
import mysql.connector
import base64

from google.cloud import storage
from globals import k8s, kube_client, gcloud_service
shortuuid.set_alphabet("0123456789abcdefghijkmnopqrstuvwxyz")


class Table:
    @staticmethod
    def getSecret(b64str):
        return base64.b64decode(b64str).decode('utf-8')

    @staticmethod
    def getSecrets():
        secrets = {}
        try:
            res = k8s.read_namespaced_secret('user-secrets', 'default')
            data = res.data

            secrets['user'] = Table.getSecret(data['user'])
            secrets['password'] = Table.getSecret(data['password'])
            secrets['db'] = Table.getSecret(data['db'])
            secrets['host'] = Table.getSecret(data['host'])
        except kube_client.rest.ApiException as e:
            print(e)

        return secrets

    def __init__(self):
        secrets = Table.getSecrets()

        if not secrets:
            raise "Couldn't read user-secrets"

        self.cnx = mysql.connector.connect(**secrets)
        cursor = self.cnx.cursor()
        cursor.execute(
            """
                CREATE TABLE IF NOT EXISTS users (
                    user_id VARCHAR(255) NOT NULL,
                    gsa_name VARCHAR(255) NOT NULL,
                    ksa_name VARCHAR(255) NOT NULL,
                    bucket_name VARCHAR(255) NOT NULL,
                    PRIMARY KEY (user_id)
                ) ENGINE=INNODB;
            """
        )
        self.cnx.commit()
        cursor.close()

    def get(self, user_id):
        cursor = self.cnx.cursor(dictionary=True)

        cursor.execute("SELECT * FROM users WHERE user_id=%s", (user_id,))
        res = cursor.fetchone()

        cursor.close()

        return res

    def insert(self, user_id, gsa_name, ksa_name, bucket_name):
        cursor = self.cnx.cursor()
        cursor.execute(
            """
            INSERT INTO users
                (user_id, gsa_name, ksa_name, bucket_name)
                VALUES (%s, %s, %s, %s)
            """, (user_id, gsa_name, ksa_name, bucket_name))
        self.cnx.commit()

        cnt = cursor.rowcount

        cursor.close()

        return cnt == 1


def make_service_id(username):
    if len(username) > 4:
        user_basename = username[:4]
    else:
        user_basename = username

    return f'{user_basename}-{shortuuid.uuid()}'


def make_google_service_account(sa_name, username, google_project):
    return gcloud_service.projects().serviceAccounts().create(name=f'projects/{google_project}', body={
        "accountId": sa_name, "serviceAccount": {
            "displayName": f'user-{username}'
        }
    }).execute()


def make_kube_service_acccount(sa_name, username):
    return k8s.create_namespaced_service_account(
        namespace='default',
        body=kube_client.V1ServiceAccount(
            api_version='v1',
            metadata=kube_client.V1ObjectMeta(
                name=sa_name,
                annotations={
                    "kubernetes.io/service-account.name": sa_name,
                    "username": username,
                    "type": "user"
                }
            )
        )
    )


def make_bucket(sa_name, username):
    bucket = storage.Client().bucket(sa_name)
    bucket.labels = {
        'type': 'user',
        'username': username
    }
    bucket.create()

    return bucket


def make_all(user_id, username, google_project = 'hail-vdc'):
    out = {}

    sa_name = make_service_id(username)

    gs_response = make_google_service_account(sa_name, username, google_project)
    out['gsa_name'] = gs_response['name']

    make_kube_service_acccount(sa_name, username)
    out['ksa_name'] = sa_name

    make_bucket(sa_name, username)
    out['bucket_name'] = sa_name

    return out


def make_all_idempotent(user_id, username, google_project = 'hail-vdc'):
    table = Table()
    existing = table.get(user_id)

    if existing is None:
        res = make_all(user_id, username, google_project)
        success = table.insert(user_id, **res)

        if success is False:
            raise f"Couldn't insert entries for {user_id}"
        return res

    else:
        return existing


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) == 1:
        sys.exit(f"\nUsage: {sys.argv[0]} <user_id> <user_name>\n")

    print(json.dumps((make_all_idempotent(sys.argv[1], sys.argv[2]))))
