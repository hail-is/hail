import os
import hailjwt as hj

with open(os.environ['HAIL_TOKEN_FILE']) as f:
    userdata = hj.JWTClient.unsafe_decode(f.read())
    BUCKET = f'gs://{userdata["bucket_name"]}'

GCP_PROJECT = os.environ['HAIL_GCP_PROJECT']
DOMAIN = os.environ['HAIL_DOMAIN']
IP = os.environ['HAIL_IP']
BUILDER_UTILS_IMAGE = os.environ['HAIL_BUILDER_UTILS_IMAGE']
