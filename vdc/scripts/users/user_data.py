import shortuuid

import google
from google.cloud import storage
from globals import k8s, kube_client, gcloud_service
from table import Table

shortuuid.set_alphabet("0123456789abcdefghijkmnopqrstuvwxyz")


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
    return gcloud_service.projects().serviceAccounts().delete(name='projects/-/serviceAccounts/' + gsa_email).execute()


def create_kube_service_acccount(namespace):
    return k8s.create_namespaced_service_account(
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
    return k8s.delete_namespaced_service_account(name=ksa_name, namespace=namespace, body={})


def create_bucket(sa_name, gsa_email):
    bucket = storage.Client().bucket(sa_name)
    bucket.labels = {
        'type': 'user',
    }

    bucket.create()

    acl = bucket.acl
    acl.user(gsa_email).grant_owner()

    return bucket


def delete_bucket(bucket_name):
    return storage.Client().get_bucket(bucket_name).delete()


def create_all(google_project, kube_namespace):
    out = {}

    ksa_response = create_kube_service_acccount(kube_namespace)
    out['ksa_name'] = ksa_response.metadata.name

    sa_name = create_service_id()

    gs_response = create_google_service_account(sa_name, google_project)

    out['gsa_projectId'] = gs_response['projectId']
    out['gsa_email'] = gs_response['email']

    create_bucket(sa_name, out['gsa_email'])
    out['bucket_name'] = sa_name

    return out


def delete_all(user_obj, google_project='hail-vdc', kube_namespace='default'):
    try:
        delete_bucket(user_obj['bucket_name'])
        delete_google_service_account(user_obj['gsa_email'], google_project)
        delete_kube_service_acccount(user_obj['ksa_name'], kube_namespace)
    except google.api_core.exceptions.NotFound:
        return 404


def create_all_idempotent(user_id, google_project='hail-vdc', kube_namespace='default'):
    table = Table()
    existing = table.get(user_id)

    if existing is None:
        res = create_all(google_project, kube_namespace)
        success = table.insert(user_id, **res)

        if success is False:
            raise f"Couldn't insert entries for {user_id}"
        return res

    else:
        return existing


def delete_all_idempotent(user_id, google_project='hail-vdc', kube_namespace='default'):
    status = delete_all(existing, google_project, kube_namespace)
    found = Table().delete(user_id)

    if status == 404 and found is False:
        return 404


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) == 1:
        sys.exit(f"\nUsage: {sys.argv[0]} user_id <create|delete>\n")

    user_id = sys.argv[1]
    create = sys.argv[2] if len(sys.argv) == 3 else "create"

    if create == 'create':
        print(json.dumps((create_all_idempotent(user_id))))
    elif create == 'delete':
        error = delete_all_idempotent(user_id)

        if error == 404:
            print(f"\nNothing to change for {user_id}\n")
        else:
            print(f"\nSuccessfully deleted {user_id}\n")
    else:
        print(f"Unknown operation {create}")
