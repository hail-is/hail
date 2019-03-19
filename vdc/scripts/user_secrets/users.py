import shortuuid

from google.cloud import storage
from globals import k8s, kube_client, gcloud_service
from table import Table

shortuuid.set_alphabet("0123456789abcdefghijkmnopqrstuvwxyz")


def make_service_id():
    return f'user-{shortuuid.uuid()}'


def make_google_service_account(sa_name, google_project):
    return gcloud_service.projects().serviceAccounts().create(
            name=f'projects/{google_project}',
            body={
                "accountId": sa_name, "serviceAccount": {
                    "displayName": "user"
                }
            }).execute()


def make_kube_service_acccount():
    return k8s.create_namespaced_service_account(
        namespace='default',
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


def make_bucket(sa_name):
    bucket = storage.Client().bucket(sa_name)
    bucket.labels = {
        'type': 'user',
    }
    bucket.create()

    return bucket


def make_all(google_project='hail-vdc'):
    out = {}

    ksa_response = make_kube_service_acccount()
    out['ksa_name'] = ksa_response.metadata.name

    sa_name = make_service_id()

    gs_response = make_google_service_account(sa_name, google_project)
    out['gsa_name'] = gs_response['name']

    make_bucket(sa_name)
    out['bucket_name'] = sa_name

    return out


def make_all_idempotent(user_id, google_project='hail-vdc'):
    table = Table()
    existing = table.get(user_id)

    if existing is None:
        res = make_all(google_project)
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
        sys.exit(f"\nUsage: {sys.argv[0]} <user_id>\n")

    print(json.dumps((make_all_idempotent(sys.argv[1]))))
