import shortuuid
from google.cloud import storage
from globals import k8s, kube_client, gcloud_service
shortuuid.set_alphabet("0123456789abcdefghijkmnopqrstuvwxyz")


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


def make_all(username, google_project = 'hail-vdc'):
    out = {}

    sa_name = make_service_id(username)

    gs_response = make_google_service_account(sa_name, username, google_project)
    out['gsa_name'] = gs_response['name']

    make_kube_service_acccount(sa_name, username)
    out['ksa_name'] = sa_name

    make_bucket(sa_name, username)
    out['bucket_name'] = sa_name

    return out


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) == 1:
        sys.exit(f"\nUsage: {sys.argv[0]} <user_name>\n")
    
    print(json.dumps((make_all(sys.argv[1]))))
