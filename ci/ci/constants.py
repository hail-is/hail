import os
import hailtop.gear.auth as hj

GITHUB_CLONE_URL = 'https://github.com/'

with open(os.environ['HAIL_TOKEN_FILE']) as f:
    userdata = hj.JWTClient.unsafe_decode(f.read())
    BUCKET = f'gs://{userdata["bucket_name"]}'

AUTHORIZED_USERS = {
    'danking',
    'cseed',
    'konradjk',
    'jigold',
    'jbloom22',
    'patrick-schultz',
    'lfrancioli',
    'akotlar',
    'tpoterba',
    'chrisvittal',
    'catoverdrive',
    'daniel-goldstein',
    'ahiduchick',
    'GreatBrando',
    'johnc1231'
}

profiles = {
    "batch_test": [
        'default_ns',
        'deploy_batch_sa',
        'batch_pods_ns',
        'base_image',
        'create_accounts',
        'batch_image',
        'test_batch_image',
        'batch_database',
        'create_batch_tables_image',
        'create_batch_tables',
        'create_batch_tables2',
        'deploy_batch',
        'deploy_batch_pods',
        'deploy_batch_test'
    ]
}