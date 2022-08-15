from typing import Dict, List
import os
import sys
import yaml

DOMAIN = os.environ['HAIL_DOMAIN']


def create_rds_response(services_per_namespace: Dict[str, List[str]], requester: str) -> dict:
    assert 'default' in services_per_namespace
    default_services = services_per_namespace['default']
    internal_services_per_namespace = {k: v for k, v in services_per_namespace.items() if k != 'default'}
    if requester == 'gateway':
        default_host = gateway_default_host
        internal_host = gateway_internal_host
    else:
        assert requester == 'internal-gateway'
        default_host = internal_gateway_default_host
        internal_host = internal_gateway_internal_host

    hosts = [default_host(service) for service in default_services]
    if len(internal_services_per_namespace) > 0:
        hosts.append(internal_host(internal_services_per_namespace))
    return {
        'version_info': 'dummy',
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
        'version_info': 'dummy',
        'type_url': 'type.googleapis.com/envoy.config.cluster.v3.Cluster',
        'resources': clusters(services_per_namespace, requester),
        'control_plane': {
            'identifier': 'ci',
        },
    }


# TODO ukbb-rg needs some tinkering
def gateway_default_host(service: str) -> dict:
    domains = [f'{service}.{DOMAIN}']
    if service == 'www':
        domains.append(DOMAIN)
    return {
        '@type': 'type.googleapis.com/envoy.config.route.v3.VirtualHost',
        'name': service,
        'domains': domains,
        'routes': [
            {
                'match': {'prefix': '/'},
                'route': {'timeout': '0s', 'cluster': service},
                'typed_per_filter_config': {
                    'envoy.filters.http.ext_authz': auth_check_exemption(),
                },
            }
        ],
    }


def gateway_internal_host(services_per_namespace: Dict[str, List[str]]) -> dict:
    return {
        '@type': 'type.googleapis.com/envoy.config.route.v3.VirtualHost',
        'name': 'internal',
        'domains': [f'internal.{DOMAIN}'],
        'routes': [
            {
                'match': {'prefix': f'/{namespace}/{service}'},
                'route': {'timeout': '0s', 'cluster': f'{namespace}-{service}'},
                'typed_per_filter_config': {
                    'envoy.filters.http.local_ratelimit': rate_limit_config(),
                    'envoy.filters.http.ext_authz': auth_check_exemption(),
                },
            }
            for namespace, services in services_per_namespace.items()
            for service in services
        ],
    }


def internal_gateway_default_host(service: str) -> dict:
    return {
        '@type': 'type.googleapis.com/envoy.config.route.v3.VirtualHost',
        'name': service,
        'domains': [f'{service}.hail'],
        'routes': [
            {
                'match': {'prefix': '/'},
                'route': {'timeout': '0s', 'cluster': service},
                'typed_per_filter_config': {
                    'envoy.filters.http.local_ratelimit': rate_limit_config(),
                },
            }
        ],
    }


def internal_gateway_internal_host(services_per_namespace: Dict[str, List[str]]) -> dict:
    return {
        '@type': 'type.googleapis.com/envoy.config.route.v3.VirtualHost',
        'name': 'internal',
        'domains': ['internal.hail'],
        'routes': [
            {
                'match': {'prefix': f'/{namespace}/{service}'},
                'route': {'timeout': '0s', 'cluster': f'{namespace}-{service}'},
                'typed_per_filter_config': {
                    'envoy.filters.http.local_ratelimit': rate_limit_config(),
                },
            }
            for namespace, services in services_per_namespace.items()
            for service in services
        ],
    }


def auth_check_exemption() -> dict:
    return {
        '@type': 'type.googleapis.com/envoy.extensions.filters.http.ext_authz.v3.ExtAuthzPerRoute',
        'disabled': True,
    }


def rate_limit_config() -> dict:
    return {
        '@type': 'type.googleapis.com/envoy.extensions.filters.http.local_ratelimit.v3.LocalRateLimit',
        'stat_prefix': 'http_local_rate_limiter',
        'token_bucket': {
            'max_tokens': 150,
            'tokens_per_fill': 150,
            'fill_interval': '1s',
        },
        'filter_enabled': {
            'runtime_key': 'local_rate_limit_enabled',
            'default_value': {
                'numerator': 100,
                'denominator': 'HUNDRED',
            },
        },
        'filter_enforced': {
            'runtime_key': 'local_rate_limit_enabled',
            'default_value': {
                'numerator': 100,
                'denominator': 'HUNDRED',
            },
        },
    }


def clusters(services_per_namespace: Dict[str, List[str]], requester: str) -> List[dict]:
    clusters = []
    for namespace, services in services_per_namespace.items():
        for service in services:
            name = service if namespace == 'default' else f'{namespace}-{service}'
            cluster = {
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
            # We don't verify namespacd services
            if namespace != 'default':
                del cluster['transport_socket']['typed_config']['validation_context']
            clusters.append(cluster)

    return clusters


if __name__ == '__main__':
    requester = sys.argv[1]
    with open(sys.argv[2], 'r', encoding='utf-8') as services_file:
        services = [service.rstrip() for service in services_file.readlines()]

    services_per_namespace = {'default': services}

    with open(sys.argv[3], 'w', encoding='utf-8') as cds_file:
        cds_file.write(yaml.dump(create_cds_response(services_per_namespace, requester)))
    with open(sys.argv[4], 'w', encoding='utf-8') as rds_file:
        rds_file.write(yaml.dump(create_rds_response(services_per_namespace, requester)))
