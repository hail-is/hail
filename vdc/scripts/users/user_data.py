import shortuuid

import google
from google.cloud import storage
from googleapiclient.errors import HttpError
from table import Table
from base64 import b64decode

from globals import v1, kube_client, gcloud_service
from secrets import get_secret
from hailjwt import JWTClient

shortuuid.set_alphabet("0123456789abcdefghijkmnopqrstuvwxyz")

SECRET_KEY = get_secret('notebook-secrets', 'default', 'secret-key')
jwtclient = JWTClient(SECRET_KEY)


def create_service_id():
    return f'user-{shortuuid.uuid()}'


def create_google_service_account(sa_name, google_project):
    return gcloud_service.projects().serviceAccounts().create(
        name=f'projects/{google_project}',
        body={
            "accountId": sa_name, "serviceAccount": {
                "displayName": "user"
            }
        }).execute()


def delete_google_service_account(gsa_email, google_project):
    return gcloud_service.projects().serviceAccounts().delete(
        name=f'projects/{google_project}'
        f'/serviceAccounts/{gsa_email}'
    ).execute()


def create_kube_service_acccount(namespace):
    return v1.create_namespaced_service_account(
        namespace=namespace,
        body=kube_client.V1ServiceAccount(
            api_version='v1',
            metadata=kube_client.V1ObjectMeta(
                generate_name='user-',
                annotations={
                    "type": "user"
                }
            )
        )
    )


def delete_kube_service_acccount(ksa_name, namespace):
    return v1.delete_namespaced_service_account(name=ksa_name,
                                                namespace=namespace, body={})


def store_gsa_key_in_kube(gsa_email, google_project, kube_namespace):
    key = gcloud_service.projects().serviceAccounts().keys().create(
        name=f'projects/{google_project}/serviceAccounts/{gsa_email}', body={}
    ).execute()

    key['privateKeyData'] = b64decode(key['privateKeyData']).decode("utf-8")

    return v1.create_namespaced_secret(
        namespace=kube_namespace,
        body=kube_client.V1Secret(
            api_version='v1',
            string_data=key,
            metadata=kube_client.V1ObjectMeta(
                generate_name='gsa-key-',
                annotations={
                    "type": "user",
                    "gsa_email": gsa_email
                }
            )
        )
    )


def delete_kube_secret(secret_name, kube_namespace):
    return v1.delete_namespaced_secret(secret_name, kube_namespace)


def create_user_kube_secret(user_data, kube_namespace):
    jwt = jwtclient.encode(user_data)

    return v1.create_namespaced_secret(
        namespace=kube_namespace,
        body=kube_client.V1Secret(
            api_version='v1',
            string_data={'jwt': jwt.decode('utf-8')},
            metadata=kube_client.V1ObjectMeta(
                generate_name='user-jwt-',
                annotations={
                    "type": "user",
                }
            )
        )
    )


def create_bucket(sa_name, gsa_email):
    bucket = storage.Client().bucket(sa_name)
    bucket.labels = {
        'type': 'user',
    }

    bucket.create()

    acl = bucket.acl
    acl.user(gsa_email).grant_owner()
    acl.save()

    return bucket


def delete_bucket(bucket_name):
    return storage.Client().get_bucket(bucket_name).delete()


def create_all(google_project, kube_namespace):
    out = {}

    ksa_response = create_kube_service_acccount(kube_namespace)
    out['ksa_name'] = ksa_response.metadata.name

    sa_name = create_service_id()

    gs_response = create_google_service_account(sa_name, google_project)
    out['gsa_email'] = gs_response['email']

    create_bucket(sa_name, out['gsa_email'])
    out['bucket_name'] = sa_name

    ksa_secret_resp = store_gsa_key_in_kube(out['gsa_email'],
                                            google_project, kube_namespace)
    out['gsa_key_secret_name'] = ksa_secret_resp.metadata.name

    ksa_secret_resp = create_user_kube_secret(out, kube_namespace)
    out['user_jwt_secret_name'] = ksa_secret_resp.metadata.name

    return out


def delete_all(user_obj, google_project='hail-vdc', kube_namespace='default'):
    modified = 0

    try:
        delete_bucket(user_obj['bucket_name'])
        modified += 1
    except google.api_core.exceptions.NotFound:
        pass

    try:
        delete_google_service_account(user_obj['gsa_email'], google_project)
        modified += 1
    except HttpError as e:
        if e.resp.status != 404:
            raise e

    try:
        delete_kube_service_acccount(user_obj['ksa_name'], kube_namespace)
        modified += 1
    except kube_client.rest.ApiException as e:
        if e.status != 404:
            raise e

    try:
        delete_kube_secret(user_obj['gsa_key_secret_name'], kube_namespace)
        delete_kube_secret(user_obj['user_jwt_secret_name'], kube_namespace)
        modified += 1
    except kube_client.rest.ApiException as e:
        if e.status != 404:
            raise e

    if modified == 0:
        return 404


def create_all_idempotent(user_id, google_project='hail-vdc',
                          kube_namespace='default'):
    table = Table()
    existing = table.get(user_id)

    if existing is None:
        res = create_all(google_project, kube_namespace)
        table.insert(user_id, **res)

        return res

    else:
        return existing


def delete_all_idempotent(user_id, google_project='hail-vdc',
                          kube_namespace='default'):
    table = Table()
    existing = table.get(user_id)

    if existing is None:
        return 404

    delete_all(existing, google_project, kube_namespace)
    table.delete(user_id)


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) == 1:
        sys.exit(
            f"\nUsage: {sys.argv[0]} user_id <kube_namespace> <create|delete>")

    user_id = sys.argv[1]
    kube_namespace = sys.argv[2]
    op = sys.argv[3]

    if op == 'create':
        print(json.dumps((create_all_idempotent(user_id, 'hail-vdc',
                                                kube_namespace))))
    elif op == 'delete':
        error = delete_all_idempotent(user_id, 'hail-vdc', kube_namespace)

        if error == 404:
            print(f"\nNothing to change for {user_id}\n")
        else:
            print(f"\nSuccessfully deleted {user_id}\n")
    else:
        print(f"Unknown operation {op}")
