import googleapiclient.discovery

# https://developers.google.com/resources/api-libraries/documentation/compute/v1/python/latest/compute_v1.instances.html
# https://cloud.google.com/compute/docs/reference/rest/v1/instances/insert
# https://github.com/GoogleCloudPlatform/python-docs-samples/blob/master/compute/api/create_instance.py

compute = googleapiclient.discovery.build('compute', 'v1')

config = {
    'name': 'instances-insert-test',
    'machineType': 'projects/broad-ctsa/zones/us-central1-a/machineTypes/n1-standard-1',

    'labels': {
        'app': 'hail_pipeline',
        'role': 'pipeline_worker',
        'pipeline_token': 'foobar'
    },

    # Specify the boot disk and the image to use as a source.
    'disks': [{
        'boot': True,
        'autoDelete': True,
        'diskSizeGb': "20",
        'initializeParams': {
            'sourceImage': 'projects/broad-ctsa/global/images/cs-hack',
        }
    }],

    'networkInterfaces': [{
        'network': 'global/networks/default',
        'networkTier': 'PREMIUM',
        'accessConfigs': [{
            'type': 'ONE_TO_ONE_NAT',
            'name': 'external-nat'
        }]
    }],

    'scheduling': {
        'automaticRestart': False,
        'onHostMaintenance': "TERMINATE",
        'preemptible': True
    },

    'serviceAccounts': [{
        'email': '842871226259-compute@developer.gserviceaccount.com',
        'scopes': [
            'https://www.googleapis.com/auth/cloud-platform'
        ]
    }],
    
    # Metadata is readable from the instance and allows you to
    # pass configuration from deployment scripts to instances.
    'metadata': {
        'items': [{
            'key': 'master',
            'value': 'cs-hack-master'
        }, {
            'key': 'token',
            'value': 'foobar'
        }, {
            'key': 'startup-script-url',
            'value': 'gs://hail-cseed/cs-hack/task-startup.sh'
        }]
    }
}

inst = compute.instances().insert(
    project='broad-ctsa',
    zone='us-central1-a',
    body=config).execute()
print(inst)

# {'kind': 'compute#operation', 'id': '5148396906448616754', 'name': 'operation-1564165084479-58e99903aff84-0542e048-00b936f7', 'zone': 'https://www.googleapis.com/compute/v1/projects/broad-ctsa/zones/us-central1-a', 'operationType': 'insert', 'targetLink': 'https://www.googleapis.com/compute/v1/projects/broad-ctsa/zones/us-central1-a/instances/instances-insert-test', 'targetId': '8739634858971036979', 'status': 'RUNNING', 'user': 'cseed@broadinstitute.org', 'progress': 0, 'insertTime': '2019-07-26T11:18:06.105-07:00', 'startTime': '2019-07-26T11:18:06.108-07:00', 'selfLink': 'https://www.googleapis.com/compute/v1/projects/broad-ctsa/zones/us-central1-a/operations/operation-1564165084479-58e99903aff84-0542e048-00b936f7'}
