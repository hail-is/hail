from typing import Dict, List
import os
import sys
import yaml

DOMAIN = os.environ['DOMAIN']


def create_rds_response(services_per_namespace: Dict[str, List[str]]) -> dict:
    assert 'default' in services_per_namespace
    hosts = default_hosts(services_per_namespace['default'])
    if len(services_per_namespace) > 1:
        internal = internal_host({k: v for k, v in services_per_namespace.items() if k != 'default'})
        hosts.append(internal)
    return {
        'version_info': '2',
        'type_url': 'type.googleapis.com/envoy.config.route.v3.RouteConfiguration',
        'resources': [
            {
                '@type': 'type.googleapis.com/envoy.config.route.v3.RouteConfiguration',
                'name': 'https_routes',
                'virtual_hosts': hosts,
            }
        ],
        'control_plane': {
            'identifier': 'ci',
        },
    }


def create_cds_response(services_per_namespace: Dict[str, List[str]], requester: str) -> dict:
    return {
        'version_info': '2',
        'type_url': 'type.googleapis.com/envoy.config.cluster.v3.Cluster',
        'resources': clusters(services_per_namespace, requester),
        'control_plane': {
            'identifier': 'ci',
        },
    }


# TODO ukbb-rg needs some tinkering
def default_hosts(services: List[str]) -> List[dict]:
    hosts = []
    for service in services:
        domains = [f'{service}.{DOMAIN}']
        if service == 'www':
            domains.append(DOMAIN)
        routes = [
            {
                'match': {'prefix': '/'},
                'route': {'timeout': '0s', 'cluster': service},
            }
        ]
        hosts.append(
            {
                '@type': 'type.googleapis.com/envoy.config.route.v3.VirtualHost',
                'name': service,
                'domains': domains,
                'routes': routes,
            }
        )
    return hosts


def internal_host(services_per_namespace: Dict[str, List[str]]) -> dict:
    return {
        '@type': 'type.googleapis.com/envoy.config.route.v3.VirtualHost',
        'name': 'internal',
        'domains': [f'internal.{DOMAIN}'],
        'routes': [
            {
                'match': {'prefix': f'/{namespace}/{service}'},
                'route': {'timeout': '0s', 'cluster': f'{namespace}-{service}'},
            }
            for namespace, services in services_per_namespace
            for service in services
        ],
    }


def clusters(services_per_namespace: Dict[str, List[str]], requester: str) -> List[dict]:
    clusters = []
    for namespace, services in services_per_namespace.items():
        for service in services:
            name = service if namespace == 'default' else f'{namespace}-{service}'
            clusters.append(
                {
                    '@type': 'type.googleapis.com/envoy.config.cluster.v3.Cluster',
                    'name': name,
                    'type': 'STRICT_DNS',
                    'lb_policy': 'ROUND_ROBIN',
                    'load_assignment': {
                        'cluster_name': name,
                        'endpoints': [
                            {
                                'lb_endpoints': [
                                    {
                                        'endpoint': {
                                            'address': {
                                                'socket_address': {
                                                    'address': f'{service}.{namespace}',
                                                    'port_value': 443,
                                                }
                                            }
                                        }
                                    }
                                ]
                            }
                        ],
                    },
                    'transport_socket': {
                        'name': 'envoy.transport_sockets.tls',
                        'typed_config': {
                            '@type': 'type.googleapis.com/envoy.extensions.transport_sockets.tls.v3.UpstreamTlsContext',
                            'common_tls_context': {
                                'tls_certificates': [
                                    {
                                        'certificate_chain': {'filename': f'/ssl-config/{requester}-cert.pem'},
                                        'private_key': {'filename': f'/ssl-config/{requester}-key.pem'},
                                    }
                                ]
                            },
                            'validation_context': {
                                'trusted_ca': {'filename': f'/ssl-config/{requester}-outgoing.pem'},
                            },
                        },
                    },
                }
            )
    return clusters


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as services_file:
        services = [service.rstrip() for service in services_file.readlines()]

    services_per_namespace = {'default': services}
    with open(sys.argv[2], 'w') as cds_file:
        cds_file.write(yaml.dump(create_cds_response(services_per_namespace, 'gateway')))
    with open(sys.argv[3], 'w') as rds_file:
        rds_file.write(yaml.dump(create_rds_response(services_per_namespace)))
