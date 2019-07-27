from google.cloud import logging
logging_client = logging.Client()

PROJECT = 'broad-ctsa'
filter = f'logName:projects/{PROJECT}/logs/compute.googleapis.com%2Factivity_log'

i = 0
for entry in logging_client.list_entries(filter_=filter, order_by=logging.DESCENDING):
    print('---')
    if entry.payload:
        payload = entry.payload
        version = payload["version"]
        event_type = payload['event_type']
        event_subtype = payload['event_subtype']
        resource = payload['resource']
        name = resource['name']
        
        print(f'event {version} {event_type} {event_subtype} {name}')
        
    print(f'when {entry.timestamp}')
    i += 1
    if i > 3:
        break
