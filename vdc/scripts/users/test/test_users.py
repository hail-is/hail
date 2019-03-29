import unittest
from user_data import (create_all, create_all_idempotent,
                       delete_all_idempotent, delete_all)

from google.cloud import storage
import uuid
from globals import v1, gcloud_service

user_id = f'test-user-{uuid.uuid4().hex}'
google_project = 'hail-vdc'
kube_namespace = 'test'


class TestCreate(unittest.TestCase):
    def test_create_and_delete_all(self):
        data = create_all(google_project, kube_namespace)
        delete_all(data, google_project, kube_namespace)

    def test_create_all_idempotent(self):
        create_all_idempotent(
            user_id, google_project=google_project,
            kube_namespace=kube_namespace)

    def test_delete_all_idempotent(self):
        delete_all_idempotent(user_id, google_project, kube_namespace)

    def test_delete_partial_v1_sa(self):
        data = create_all(google_project, kube_namespace)

        try:
            v1.delete_namespaced_service_account(
                name=data['ksa_name'], namespace=kube_namespace, body={})
        except Exception:
            self.fail(f"Couldn't delete kubernetes service account")

        delete_all(data, google_project, kube_namespace)

    def test_delete_partial_gcloud_sa(self):
        data = create_all(google_project, kube_namespace)

        try:
            gsa_name = f"projects/-/serviceAccounts/{data['gsa_email']}"
            gcloud_service.projects().serviceAccounts().delete(name=gsa_name)
        except Exception:
            self.fail(f"Couldn't delete created google service account")

        delete_all(data, google_project, kube_namespace)

    def test_delete_partial_bucket(self):
        data = create_all(google_project, kube_namespace)

        try:
            bucket = storage.Client().get_bucket(data['bucket_name'])
            bucket.delete()
        except Exception:
            self.fail("Couldn't delete created bucket")

        delete_all(data, google_project, kube_namespace)


if __name__ == "__main__":
    unittest.main()
