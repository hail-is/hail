import shortuuid

import google
from google.cloud import storage
from googleapiclient.errors import HttpError
from table import Table
from base64 import b64decode
import re
import sys
import secrets
import json

from globals import v1, kube_client, gcloud_service
from utils import get_secret_bin
from hailjwt import JWTClient

shortuuid.set_alphabet("0123456789abcdefghijkmnopqrstuvwxyz")

SECRET_KEY = get_secret_bin('jwt-secret-key', 'default', 'secret-key')
jwtclient = JWTClient(SECRET_KEY)

table = Table()


def create_service_id(username):
    return f'{username}-{shortuuid.uuid()[0:5]}'


def create_google_service_account(username, google_project):
    return gcloud_service.projects().serviceAccounts().create(
        name=f'projects/{google_project}',
        body={
            "accountId": username, "serviceAccount": {
                "displayName": "user"
            }
        }).execute()


def delete_google_service_account(gsa_email, google_project):
    return gcloud_service.projects().serviceAccounts().delete(
        name=f'projects/{google_project}'
        f'/serviceAccounts/{gsa_email}'
    ).execute()


def create_kube_service_acccount(username, namespace):
    return v1.create_namespaced_service_account(
        namespace=namespace,
        body=kube_client.V1ServiceAccount(
            api_version='v1',
            metadata=kube_client.V1ObjectMeta(
                name=username,
                labels={
                    "type": "user"
                }
            )
        )
    )


def delete_kube_service_acccount(ksa_name, namespace):
    return v1.delete_namespaced_service_account(name=ksa_name,
                                                namespace=namespace, body={})


def delete_kube_namespace(namespace):
    return v1.delete_namespace(name=namespace, body={})


def store_gsa_key_in_kube(gsa_key_name, gsa_email, google_project,
                          kube_namespace):
    key = gcloud_service.projects().serviceAccounts().keys().create(
        name=f'projects/{google_project}/serviceAccounts/{gsa_email}', body={}
    ).execute()

    key['privateKeyData'] = b64decode(key['privateKeyData']).decode("utf-8")

    create_namespaced_secret(kube_namespace,
                             secret_name=gsa_key_name,
                             string_data=key,
                             labels={
                                 "type": "user"
                             }, annotations={
                                 "gsa_email": gsa_email
                             })


def create_namespaced_secret(kube_namespace, secret_name, string_data,
                             labels=None, annotations=None):
    return v1.create_namespaced_secret(
        namespace=kube_namespace,
        body=kube_client.V1Secret(
            api_version='v1',
            string_data=string_data,
            metadata=kube_client.V1ObjectMeta(
                name=secret_name,
                labels=labels,
                annotations=annotations
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
            string_data={'jwt': jwt},
            metadata=kube_client.V1ObjectMeta(
                name=user_data['jwt_secret_name'],
                labels={
                    "type": "user",
                }
            )
        )
    )


def create_bucket(bucket_name, gsa_email):
    bucket = storage.Client().bucket(bucket_name)
    bucket.labels = {
        'type': 'user',
    }

    bucket.create()
    grant_bucket_read_write(bucket, gsa_email)

    return bucket


def grant_bucket_owner(bucket, email):
    acl = bucket.acl
    acl.user(email).grant_owner()
    acl.save()

    bucket.default_object_acl.user(email).grant_owner()
    bucket.default_object_acl.save()


def grant_bucket_read_write(bucket, email):
    acl = bucket.acl
    acl.user(email).grant_write()
    acl.user(email).grant_read()
    acl.save()

    bucket.acl.reload()

    bucket.default_object_acl.user(email).grant_read()
    bucket.default_object_acl.save()


def create_kube_namespace(username):
    return v1.create_namespace(
        body=kube_client.V1Namespace(
            api_version='v1',
            metadata=kube_client.V1ObjectMeta(
                name=username,
                labels={
                    "type": "user"
                }
            )
        )
    )


def create_rbac(namespace, sa_name):
    api = kube_client.RbacAuthorizationV1Api()

    rule_name = "admin"
    api.create_namespaced_role(
        namespace=namespace,
        body=kube_client.V1Role(
            metadata=kube_client.V1ObjectMeta(
                name=rule_name,
                labels={
                    "type": "user"
                }
            ),
            rules=[
                kube_client.V1PolicyRule(
                    verbs=["*"],
                    api_groups=["*"],
                    resources=["*"]
                )
            ]
        )
    )

    api.create_namespaced_role_binding(
        namespace=namespace,
        body=kube_client.V1RoleBinding(
            metadata=kube_client.V1ObjectMeta(
                name=f"{namespace}-admin",
                labels={
                    "type": "user"
                }
            ),
            role_ref=kube_client.V1RoleRef(
                api_group="",
                kind="Role",
                name=rule_name
            )
        )
    )


def delete_bucket(bucket_name):
    return storage.Client().get_bucket(bucket_name).delete(force=True)


def create_all(user_id, username, google_project, kube_namespace, email=None,
               service_account=False, developer=False):

    if not service_account and not email:
        print("\nMust provide valid email unless you are a service_account\n")
        sys.exit(1)

    if service_account and developer:
        print("\nCannot be both a 'service_account' and a 'developer'\n")
        sys.exit(1)

    out = {
        'email': email,
        'user_id': user_id,
    }

    random_str = shortuuid.uuid()[0:5]
    sa_name = f'{username}-{random_str}'

    if developer:
        out['developer'] = True
        out['service_account'] = False

        namespace_response = create_kube_namespace(username)
        out['namespace_name'] = namespace_response.metadata.name

        ksa_response = create_kube_service_acccount(username, kube_namespace)
        out['ksa_name'] = ksa_response.metadata.name

        (sql_admin_secret, sql_user_secret) = create_user_db(
            out['namespace_name'], username)

        out['sql_admin_secret'] = sql_admin_secret
        out['sql_user_secret'] = sql_user_secret

        create_rbac(out['namespace_name'], out['ksa_name'])
    else:
        out['developer'] = False

        if service_account:
            out['service_account'] = True
        else:
            username = sa_name
            out['service_account'] = False

        out['namespace_name'] = None
        out['ksa_name'] = None

    out['username'] = username

    gs_response = create_google_service_account(sa_name, google_project)
    out['gsa_email'] = gs_response['email']

    bucket_name = f'hail-{sa_name}'
    bucket = create_bucket(bucket_name, out['gsa_email'])

    if email is not None:
        grant_bucket_read_write(bucket, email)

    out['bucket_name'] = bucket_name

    gsa_key_secret_name = f"{username}-gsa-key"
    store_gsa_key_in_kube(gsa_key_secret_name, out['gsa_email'],
                          google_project, kube_namespace)
    out['gsa_key_secret_name'] = gsa_key_secret_name

    return out


def delete_all(user_obj, google_project, kube_namespace):
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
        delete_kube_secret(user_obj['gsa_key_secret_name'], kube_namespace)
        delete_kube_secret(user_obj['jwt_secret_name'], kube_namespace)
        modified += 1
    except kube_client.rest.ApiException as e:
        if e.status != 404:
            raise e

    if user_obj['ksa_name']:
        try:
            delete_kube_service_acccount(user_obj['ksa_name'], kube_namespace)
            modified += 1
        except kube_client.rest.ApiException as e:
            if e.status != 404:
                raise e

    if user_obj['namespace_name']:
        try:
            delete_kube_namespace(user_obj['namespace_name'])
            modified += 1
        except kube_client.rest.ApiException as e:
            if e.status != 404:
                raise e

    if modified == 0:
        return 404


def sanitize_username(username):
    username = re.sub('^[^0-9a-zA-Z]', '', username)
    username = re.sub('[^0-9a-zA-Z\-]+', '', username)

    length = len(username)

    if length > 15:
        username = username[:15]

    return username


def email_to_username(email):
    return sanitize_username(email.split('@')[0])


def create_user_db(kube_namespace, username):
    # Create secrets
    data = {
        'db_name': username,
        'admin_role': f"{username}-admin",
        'user_role': f"{username}-user",
        'admin_pass': secrets.token_urlsafe(16),
        'user_pass': secrets.token_urlsafe(16),
    }
    admin_secret = f"sql-{username}-{data['admin_role']}"
    user_secret = f"sql-{username}-{data['user_role']}"

    table.create_user_db(**data)

    create_namespaced_secret(kube_namespace=kube_namespace,
                             secret_name=admin_secret,
                             string_data={
                                 "sql-config.json": json.dumps({
                                     "host": "10.80.0.3",
                                     "port": 3306,
                                     "user": data['admin_role'],
                                     "password": data['admin_pass'],
                                     "instance": "db-gh0um",
                                     "connection_name": "hail-vdc:us-central1:db-gh0um",
                                     "db": data['db_name']
                                 }),
                                 'sql-config.cnf':
                                 f"[client]\nhost=10.80.0.3" +
                                 f"\nuser={data['admin_role']}" +
                                 f"\npassword={data['admin_pass']}"
                                 f"\ndatabase={data['db_name']}\n"
                             })

    create_namespaced_secret(kube_namespace=kube_namespace,
                             secret_name=user_secret,
                             string_data={
                                 "sql-config.json": json.dumps({
                                     "host": "10.80.0.3",
                                     "port": 3306,
                                     "user": data['user_role'],
                                     "password": data['user_pass'],
                                     "instance": "db-gh0um",
                                     "connection_name": "hail-vdc:us-central1:db-gh0um",
                                     "db": data['db_name']
                                 }),
                                 'sql-config.cnf':
                                 f"[client]\nhost=10.80.0.3" +
                                 f"\nuser={data['user_role']}" +
                                 f"\npassword={data['user_pass']}"
                                 f"\ndatabase={data['db_name']}\n"
                             })

    return admin_secret, user_secret


def create_all_idempotent(user_id, kube_namespace, username=None, email=None,
                          **kwargs):
    existing = table.get(user_id)

    if existing is None:
        if email is None and username is None:
            raise("Must provide either 'username' or 'email'")

        if username is not None:
            username = sanitize_username(username)
        else:
            username = email_to_username(email)

        user = create_all(user_id=user_id, username=username, email=email,
                          kube_namespace=kube_namespace, **kwargs)
        user['jwt_secret_name'] = f"{user['username']}-jwt"

        table.insert(**user)
        res = table.get(user_id)
        user['id'] = res['id']

        if user['namespace_name'] is None:
            del user['namespace_name']
        if user['ksa_name'] is None:
            del user['ksa_name']
        if user['email'] is None:
            del user['email']

        create_user_kube_secret(user, kube_namespace)

        return user

    else:
        return existing


def delete_all_idempotent(user_id, google_project, kube_namespace):

    existing = table.get(user_id)

    if existing is None:
        return 404

    delete_all(existing, google_project, kube_namespace)
    table.delete(user_id)


if __name__ == "__main__":
    import yaml

    with open(sys.argv[1], 'r') as file:
        users = yaml.safe_load(file)

    op = sys.argv[2]

    for user in users:
        user_identifier = user.get('username', user.get('email'))

        if op == 'create':
            print(f"\n\nCreating {user_identifier}:")
            print(json.dumps((create_all_idempotent(**user))), "\n")
        elif op == 'delete':
            print(f"\n\nDeleting {user_identifier}:")

            req = {
                'user_id': user['user_id'],
                'kube_namespace': user['kube_namespace'],
                'google_project': user['google_project'],
            }

            error = delete_all_idempotent(**req)

            if error == 404:
                print(f"\nNothing to change for {user_identifier}\n")
            else:
                print(f"\nSuccessfully deleted {user_identifier}\n")
        else:
            print(f"Unknown operation {op}")
