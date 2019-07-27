import googleapiclient.discovery
from google.cloud import storage

# https://cloud.google.com/storage/docs/reference/libraries
# https://googleapis.github.io/google-cloud-python/latest/storage/client.html

storage_client = storage.Client()

bucket = storage_client.get_bucket('hail-cseed')
print(bucket)
blob = bucket.blob('cs-hack/test.json')
print(blob)
print(blob.public_url)
s = blob.upload_from_string('{"foo":"bar"}')
print(s)
