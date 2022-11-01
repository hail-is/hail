import os
import sys
from typing import Dict, List

import yaml

DOMAIN = os.environ['HAIL_DOMAIN']


def create_rds_response(
    default_services: List[str], internal_services_per_namespace: Dict[str, List[str]], proxy: str
) -> dict:
    if proxy == 'gateway':
        default_host = gateway_default_host
        internal_host = gateway_internal_host
    else:
        assert proxy == 'internal-gateway'
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
                'request_headers_to_add': [
                    {
                        'header': {'key': 'X-Real-IP', 'value': '%DOWNSTREAM_REMOTE_ADDRESS%'},
                        'append_action': 'OVERWRITE_IF_EXISTS_OR_ADD',
                    }
                ],
                'virtual_hosts': hosts,
            }
        ],
        'control_plane': {
            'identifier': 'ci',
        },
    }


def create_cds_response(
    default_services: List[str], internal_services_per_namespace: Dict[str, List[str]], proxy: str
) -> dict:
    return {
        'version_info': 'dummy',
        'type_url': 'type.googleapis.com/envoy.config.cluster.v3.Cluster',
        'resources': clusters(default_services, internal_services_per_namespace, proxy),
        'control_plane': {
            'identifier': 'ci',
        },
    }


def gateway_default_host(service: str) -> dict:
    domains = [f'{service}.{DOMAIN}']
    if service == 'www':
        domains.append(DOMAIN)

    if service == 'ukbb-rg':
        return {
            '@type': 'type.googleapis.com/envoy.config.route.v3.VirtualHost',
            'name': service,
            'domains': domains,
            'routes': [
                {
                    'match': {'prefix': '/rg_browser'},
                    'route': route_to_cluster('ukbb-rg-browser'),
                    'typed_per_filter_config': {
                        'envoy.filters.http.ext_authz': auth_check_exemption(),
                    },
                },
                {
                    'match': {'prefix': '/'},
                    'route': route_to_cluster('ukbb-rg-static'),
                    'typed_per_filter_config': {
                        'envoy.filters.http.ext_authz': auth_check_exemption(),
                    },
                },
            ],
        }

    return {
        '@type': 'type.googleapis.com/envoy.config.route.v3.VirtualHost',
        'name': service,
        'domains': domains,
        'routes': [
            {
                'match': {'prefix': '/'},
                'route': route_to_cluster(service),
                'typed_per_filter_config': {
                    'envoy.filters.http.local_ratelimit': rate_limit_config(service),
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
                'match': {'path_separated_prefix': f'/{namespace}/{service}'},
                'route': route_to_cluster(f'{namespace}-{service}'),
                'typed_per_filter_config': {
                    'envoy.filters.http.local_ratelimit': rate_limit_config(service),
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
                'route': route_to_cluster(service),
                'typed_per_filter_config': {
                    'envoy.filters.http.local_ratelimit': rate_limit_config(service),
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
                'match': {'path_separated_prefix': f'/{namespace}/{service}'},
                'route': route_to_cluster(f'{namespace}-{service}'),
                'typed_per_filter_config': {
                    'envoy.filters.http.local_ratelimit': rate_limit_config(service),
                },
            }
            for namespace, services in services_per_namespace.items()
            for service in services
        ],
    }


def route_to_cluster(cluster_name: str) -> dict:
    return {
        'timeout': '0s',
        'cluster': cluster_name,
        'auto_host_rewrite': True,
        'append_x_forwarded_host': True,
    }


def auth_check_exemption() -> dict:
    return {
        '@type': 'type.googleapis.com/envoy.extensions.filters.http.ext_authz.v3.ExtAuthzPerRoute',
        'disabled': True,
    }


def rate_limit_config(service: str) -> dict:
    max_rps = 60 if service == 'batch-driver' else 200

    return {
        '@type': 'type.googleapis.com/envoy.extensions.filters.http.local_ratelimit.v3.LocalRateLimit',
        'stat_prefix': 'http_local_rate_limiter',
        'token_bucket': {
            'max_tokens': max_rps,
            'tokens_per_fill': max_rps,
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


def clusters(
    default_services: List[str], internal_services_per_namespace: Dict[str, List[str]], proxy: str
) -> List[dict]:
    clusters = []
    for service in default_services:
        if service == 'ukbb-rg':
            browser_cluster = make_cluster('ukbb-rg-browser', 'ukbb-rg-browser.ukbb-rg', proxy, verify_ca=True)
            static_cluster = make_cluster('ukbb-rg-static', 'ukbb-rg-static.ukbb-rg', proxy, verify_ca=True)
            clusters.append(browser_cluster)
            clusters.append(static_cluster)
        else:
            clusters.append(make_cluster(service, f'{service}.default.svc.cluster.local', proxy, verify_ca=True))

    for namespace, services in internal_services_per_namespace.items():
        for service in services:
            clusters.append(
                make_cluster(
                    f'{namespace}-{service}', f'{service}.{namespace}.svc.cluster.local', proxy, verify_ca=False
                )
            )

    return clusters


def make_cluster(name: str, address: str, proxy: str, *, verify_ca: bool) -> dict:
    return {
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
                                        'address': address,
                                        'port_value': 443,
                                    }
                                }
                            }
                        }
                    ]
                }
            ],
        },
        'transport_socket': upstream_transport_socket(proxy, verify_ca),
    }


def upstream_transport_socket(proxy: str, verify_ca: bool) -> dict:
    if verify_ca:
        return {
            'name': 'envoy.transport_sockets.tls',
            'typed_config': {
                '@type': 'type.googleapis.com/envoy.extensions.transport_sockets.tls.v3.UpstreamTlsContext',
                'common_tls_context': {
                    'tls_certificates': [
                        {
                            'certificate_chain': {'filename': f'/ssl-config/{proxy}-cert.pem'},
                            'private_key': {'filename': f'/ssl-config/{proxy}-key.pem'},
                        },
                    ],
                },
                'validation_context': {'filename': f'/ssl-config/{proxy}-outgoing.pem'},
            },
        }

    return {
        'name': 'envoy.transport_sockets.tls',
        'typed_config': {
            '@type': 'type.googleapis.com/envoy.extensions.transport_sockets.tls.v3.UpstreamTlsContext',
            'common_tls_context': {
                'tls_certificates': [],
            },
            'validation_context': {
                'trust_chain_verification': 'ACCEPT_UNTRUSTED',
            },
        },
    }


if __name__ == '__main__':
    proxy = sys.argv[1]
    with open(sys.argv[2], 'r', encoding='utf-8') as services_file:
        services = [service.rstrip() for service in services_file.readlines()]

    with open(sys.argv[3], 'w', encoding='utf-8') as cds_file:
        cds_file.write(yaml.dump(create_cds_response(services, {}, proxy)))
    with open(sys.argv[4], 'w', encoding='utf-8') as rds_file:
        rds_file.write(yaml.dump(create_rds_response(services, {}, proxy)))
