import unittest
from user_data import make_all
from google.cloud import storage

from globals import k8s, gcloud_service


class TestSecrets(unittest.TestCase):
    def test_make_all(self):
        data = make_all()

        ksa_name = data['ksa_name']
        gsa_name = data['gsa_name']
        bucket_name = data['bucket_name']

        try:
            k8s.read_namespaced_service_account(name=ksa_name, namespace='default')
            k8s.delete_namespaced_service_account(name=ksa_name, namespace='default', body={})
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


if __name__ == "__main__":
    unittest.main()
