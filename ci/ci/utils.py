import datetime
import secrets
import string
import urllib.parse
from typing import List, Optional

from gear import Database


def generate_token(size=12):
    assert size > 0
    alpha = string.ascii_lowercase
    alnum = string.ascii_lowercase + string.digits
    return secrets.choice(alpha) + ''.join([secrets.choice(alnum) for _ in range(size - 1)])


async def add_deployed_services(
    db: Database,
    namespace: str,
    services: List[str],
    expiration_time: Optional[datetime.datetime],
):
    expiration = expiration_time.strftime('%Y-%m-%d %H:%M:%S') if expiration_time else None
    await db.execute_insertone(
        '''
INSERT INTO active_namespaces (`namespace`, `expiration_time`)
VALUES (%s, %s) as new_ns
ON DUPLICATE KEY UPDATE expiration_time = new_ns.expiration_time
        ''',
        (namespace, expiration),
    )
    await db.execute_many(
        '''
INSERT INTO deployed_services (`namespace`, `service`)
VALUES (%s, %s)
ON DUPLICATE KEY UPDATE namespace = namespace;
''',
        [(namespace, service) for service in services],
    )


def generate_gcp_service_logging_url(
    project: str,
    services: List[str],
    namespace: str,
    start_time: str,
    end_time: Optional[str],
    severity: Optional[List[str]],
):
    service_queries = []
    for service in services:
        service_queries.append(
            f'''
(
resource.type="k8s_container"
resource.labels.namespace_name="{namespace}"
resource.labels.container_name="{service}"
)
'''
        )

    query = ' OR '.join(service_queries)

    if severity is not None:
        severity_queries = []
        for level in severity:
            severity_queries.append(f'severity={level}')
        query += ' OR '.join(severity_queries)

    timestamp_query = f';startTime={start_time}'
    if end_time is not None:
        timestamp_query += f';endTime = {end_time}'

    return f'https://console.cloud.google.com/logs/query;query={urllib.parse.quote_plus(query)};{urllib.parse.quote_plus(timestamp_query)}?project={project}'


def generate_gcp_worker_logging_url(
    project: str, namespace: str, start_time: str, end_time: Optional[str], severity: Optional[List[str]]
):
    query = f'''
(
resource.type="gce_instance"
logName:"worker"
labels.namespace="{namespace}"
)
'''

    if severity is not None:
        severity_queries = []
        for level in severity:
            severity_queries.append(f'severity={level}')
        query += ' OR '.join(severity_queries)

    timestamp_query = f';startTime={start_time}'
    if end_time is not None:
        timestamp_query += f';endTime = {end_time}'

    return f'https://console.cloud.google.com/logs/query;query={urllib.parse.quote_plus(query)};{urllib.parse.quote_plus(timestamp_query)}?project={project}'
