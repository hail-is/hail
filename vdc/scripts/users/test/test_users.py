import unittest
from user_data import create_all, create_all_idempotent, delete_all_idempotent
from google.cloud import storage
import uuid

from globals import k8s, gcloud_service

user_id = f'test-user-{uuid.uuid4().hex}'
google_project = 'hail-vdc'
kube_namespace = 'default'

class TestCreatea(unittest.TestCase):
    def test_create_all(self):
        data = create_all(google_project, kube_namespace)

        ksa_name = data['ksa_name']
        gsa_name = f"projects/{google_project}/serviceAccounts/{data['gsa_email']}"
        bucket_name = data['bucket_name']

        try:
            k8s.read_namespaced_service_account(name=ksa_name, namespace=kube_namespace)
            k8s.delete_namespaced_service_account(name=ksa_name, namespace=kube_namespace, body={})
        except Exception:
            self.fail(f"Couldn't read created kubernetes service account")

        try:
            gcloud_service.projects().serviceAccounts().get(name=gsa_name)
            gcloud_service.projects().serviceAccounts().delete(name=gsa_name)
        except Exception:
            self.fail(f"Couldn't read created google service account")

        try:
            bucket = storage.Client().get_bucket(bucket_name)
            bucket.delete()
        except Exception:
            self.fail("Couldn't read created bucket")

        print("Created and deleted", data)

    def test_create_all_idempotent(self):
        data = create_all_idempotent(user_id, google_project=google_project, kube_namespace=kube_namespace)

        ksa_name = data['ksa_name']
        gsa_name = f"projects/-/serviceAccounts/{data['gsa_email']}"
        bucket_name = data['bucket_name']

        try:
            k8s.read_namespaced_service_account(name=ksa_name, namespace='default')
        except Exception:
            self.fail(f"Couldn't read created kubernetes service account")

        try:
            gcloud_service.projects().serviceAccounts().get(name=gsa_name)
        except Exception:
            self.fail(f"Couldn't read created google service account")

        try:
            bucket = storage.Client().get_bucket(bucket_name)
        except Exception:
            self.fail("Couldn't read created bucket")

        print("Created", data)

    def test_delete_all_idempotent(self):
        delete_all_idempotent(user_id, google_project=google_project, kube_namespace=kube_namespace)

        print(f"Deleted {user_id}")


if __name__ == "__main__":
    unittest.main()
