import kubernetes as kube
import os
from googleapiclient import discovery
from gcp import get_key

# Look for an environment variable containing the credentials for Google Cloud Platform
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = get_key()

# Build a Python representation of the REST API
service = discovery.build('iam', 'v1')

# Define the Project ID of your project
project_id = 'projects/hail-vdc'

"""Until this point, the code is general to any API
From this point on, it is specific to the IAM API"""

# Create the request using the appropriate 'serviceAccounts' API
# You can substitute serviceAccounts by any other available API
request = service.projects().serviceAccounts().list(name=project_id)

# Execute the request that was built in the previous step
response = request.execute()

# Process the data from the response obtained with the request execution
accounts = response['accounts']
for account in accounts:
    print(account['email'])


if 'BATCH_USE_KUBE_CONFIG' in os.environ:
    kube.config.load_kube_config()
else:
    kube.config.load_incluster_config()

kClient = kube.client
k8s = kube.client.CoreV1Api()

# {
#     "kind": "Secret",
#     "apiVersion": "v1",
#     "metadata": {
#         "name": @username@,
#         "annotations": {
#             "kubernetes.io/service-account.name": @username@
#         }
#     },
#     "type": "kubernetes.io/service-account-token"
# }


def try_raise_kube(read_fn, write_fn):
    try:
        print(read_fn())
    except kube.client.rest.ApiException as e:
        if e.status != 404:
            raise e

        return write_fn()


def main():
    namespace = 'user-a'
    kube_sa_name = f'{namespace}-kube-sa'
    googla_sa_name = f'{namespace}-goog-sa'

    try_raise_kube(
        lambda: k8s.read_namespace(namespace),
        lambda: k8s.create_namespace(
            kClient.V1Namespace(
                metadata=kube.client.V1ObjectMeta(name=namespace)
            )
        )
    )

    try_raise_kube(
        lambda: k8s.read_namespaced_service_account(kube_sa_name, namespace),
        lambda: k8s.create_namespaced_service_account(
            namespace=namespace,
            body=kClient.V1ServiceAccount(
                api_version='v1',
                metadata=kClient.V1ObjectMeta(
                    name=kube_sa_name,
                    annotations={
                        "kubernetes.io/service-account.name": kube_sa_name,
                    }
                )
            )
        )
    )
    

    # try: 
    #     k8s.read_namespace(namespace)
    # except kube.client.rest.ApiException as e:
    #     if e.status != 404:
    #         raise e

    #     k8s.create_namespace(
    #     #     kClient.V1Namespace(
    #     #         metadata=kube.client.V1ObjectMeta(name=namespace)
    #     #     )
    #     # )

        

    # try:
        

        

    #     # k8s.create_namespace(
    #     #     kClient.V1Namespace(
    #     #         metadata=kube.client.V1ObjectMeta(name=namespace)
    #     #     )
    #     # )

    #     # TODO: specify allowed secrets
        # k8s.create_namespaced_service_account(
        #     namespace=namespace,
        #     body=kClient.V1ServiceAccount(
        #         api_version='v1',
        #         metadata=kClient.V1ObjectMeta(
        #             name=f'{namespace}-kubernetes-sa',
        #             annotations={
        #                 "kubernetes.io/service-account.name": f'{namespace}-kubernetes-sa'
        #             }
        #         )
        #     )
        # )

    #     # k8s.create_namespaced_secret(
    #     #     namespace=namespace,
    #     #     body=kClient.V1Secret(
    #     #         metadata=kube.client.V1Secret(name=f'{namespace}-gcloud-sa', data={'gcloud_service_account': 'blah'})
    #     #     )
    #     # )
    # except kube.client.rest.ApiException as e:
    #     print("Exception when calling CoreV1Api->create_namespaced_secret: %s\n" % e)


if __name__ == "__main__":
    main()
        