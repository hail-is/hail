import datetime
import secrets
import string
import urllib.parse
from typing import List, Optional

from gear import Database
from gear.cloud_config import get_gcp_config


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
        """
INSERT INTO active_namespaces (`namespace`, `expiration_time`)
VALUES (%s, %s) as new_ns
ON DUPLICATE KEY UPDATE expiration_time = new_ns.expiration_time
        """,
        (namespace, expiration),
    )
    await db.execute_many(
        """
INSERT INTO deployed_services (`namespace`, `service`)
VALUES (%s, %s)
ON DUPLICATE KEY UPDATE namespace = namespace;
""",
        [(namespace, service) for service in services],
    )


def severity_query_str(severity: List[str]) -> str:
    severity_queries = []
    for level in severity:
        severity_queries.append(f'severity={level}')
    return ' OR '.join(severity_queries)


def timestamp_query_str(start_time: str, end_time: Optional[str]) -> str:
    timestamp_query = f';startTime={start_time}'
    if end_time is not None:
        timestamp_query += f';endTime={end_time}'
    return timestamp_query


def gcp_service_logging_url(
    project: str,
    services: List[str],
    namespace: str,
    start_time: str,
    end_time: Optional[str],
    severity: Optional[List[str]],
) -> str:
    service_queries = []
    for service in services:
        service_queries.append(
            f"""
(
resource.type="k8s_container"
resource.labels.namespace_name="{namespace}"
resource.labels.container_name="{service}"
)
"""
        )

    query = ' OR '.join(service_queries)

    if severity is not None:
        query += severity_query_str(severity)

    timestamp_query = timestamp_query_str(start_time, end_time)

    return f'https://console.cloud.google.com/logs/query;query={urllib.parse.quote(query)};{urllib.parse.quote(timestamp_query)}?project={project}'


def gcp_worker_logging_url(
    project: str, namespace: str, start_time: str, end_time: Optional[str], severity: Optional[List[str]]
) -> str:
    query = f"""
(
resource.type="gce_instance"
logName:"worker"
labels.namespace="{namespace}"
)
"""

    if severity is not None:
        query += severity_query_str(severity)

    timestamp_query = timestamp_query_str(start_time, end_time)

    return f'https://console.cloud.google.com/logs/query;query={urllib.parse.quote(query)};{urllib.parse.quote(timestamp_query)}?project={project}'


def gcp_logging_queries(namespace: str, start_time: str, end_time: Optional[str]):
    project = get_gcp_config().project
    return {
        'batch-k8s-error-warning': gcp_service_logging_url(
            project,
            ['batch', 'batch-driver'],
            namespace,
            start_time,
            end_time,
            ['ERROR', 'WARNING'],
        ),
        'batch-workers-error-warning': gcp_worker_logging_url(
            project, namespace, start_time, end_time, ['ERROR', 'WARNING']
        ),
        'ci-k8s-error-warning': gcp_service_logging_url(
            project, ['ci'], namespace, start_time, end_time, ['ERROR', 'WARNING']
        ),
        'auth-k8s-error-warning': gcp_service_logging_url(
            project, ['auth', 'auth-driver'], namespace, start_time, end_time, ['ERROR', 'WARNING']
        ),
    }
