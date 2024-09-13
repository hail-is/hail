import sys
from typing import Dict, List, Optional

import yaml

LOAD_BALANCER_REPLICAS = 3


class Service:
    def __init__(self, name: str, rate_limit_rps: Optional[int] = None):
        self.name = name
        if rate_limit_rps is not None:
            self.rate_limit_rps = rate_limit_rps
        else:
            self.rate_limit_rps = 180 if name == 'batch-driver' else 600


def create_rds_response(
    default_services: List[Service],
    internal_services_per_namespace: Dict[str, List[Service]],
    proxy: str,
    *,
    domain: str,
) -> dict:
    if proxy == 'gateway':
        default_host = lambda service: gateway_default_host(service, domain)
        internal_host = lambda services_per_namespace: gateway_internal_host(services_per_namespace, domain)
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
    default_services: List[Service], internal_services_per_namespace: Dict[str, List[Service]], proxy: str
) -> dict:
    return {
        'version_info': 'dummy',
        'type_url': 'type.googleapis.com/envoy.config.cluster.v3.Cluster',
        'resources': clusters(default_services, internal_services_per_namespace, proxy),
        'control_plane': {
            'identifier': 'ci',
        },
    }


def gateway_default_host(service: Service, domain: str) -> dict:
    domains = [f'{service.name}.{domain}']
    if service.name == 'www':
        domains.append(domain)

    if service.name == 'ukbb-rg':
        return {
            '@type': 'type.googleapis.com/envoy.config.route.v3.VirtualHost',
            'name': service.name,
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
        'name': service.name,
        'domains': domains,
        'routes': [
            {
                'match': {'prefix': '/'},
                'route': route_to_cluster(service.name),
                'typed_per_filter_config': {
                    'envoy.filters.http.local_ratelimit': rate_limit_config(service),
                    'envoy.filters.http.ext_authz': auth_check_exemption(),
                },
            }
        ],
    }


def gateway_internal_host(services_per_namespace: Dict[str, List[Service]], domain: str) -> dict:
    return {
        '@type': 'type.googleapis.com/envoy.config.route.v3.VirtualHost',
        'name': 'internal',
        'domains': [f'internal.{domain}'],
        'routes': [
            {
                'match': {'path_separated_prefix': f'/{namespace}/{service.name}'},
                'route': route_to_cluster(f'{namespace}-{service.name}'),
                'typed_per_filter_config': {
                    'envoy.filters.http.local_ratelimit': rate_limit_config(service),
                },
            }
            for namespace, services in services_per_namespace.items()
            for service in services
        ],
    }


def internal_gateway_default_host(service: Service) -> dict:
    return {
        '@type': 'type.googleapis.com/envoy.config.route.v3.VirtualHost',
        'name': service.name,
        'domains': [f'{service.name}.hail'],
        'routes': [
            {
                'match': {'prefix': '/'},
                'route': route_to_cluster(service.name),
                'typed_per_filter_config': {
                    'envoy.filters.http.local_ratelimit': rate_limit_config(service),
                },
            }
        ],
    }


def internal_gateway_internal_host(services_per_namespace: Dict[str, List[Service]]) -> dict:
    return {
        '@type': 'type.googleapis.com/envoy.config.route.v3.VirtualHost',
        'name': 'internal',
        'domains': ['internal.hail'],
        'routes': [
            {
                'match': {'path_separated_prefix': f'/{namespace}/{service.name}'},
                'route': route_to_cluster(f'{namespace}-{service.name}'),
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


def rate_limit_config(service: Service) -> dict:
    # The config is set per load balancer pod, so we must account for
    # multiple replicas of the load balancer
    max_rps = service.rate_limit_rps // LOAD_BALANCER_REPLICAS

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
    default_services: List[Service], internal_services_per_namespace: Dict[str, List[Service]], proxy: str
) -> List[dict]:
    clusters = []
    for service in default_services:
        if service.name == 'ukbb-rg':
            browser_cluster = make_cluster('ukbb-rg-browser', 'ukbb-rg-browser.ukbb-rg', proxy, verify_ca=True)
            static_cluster = make_cluster('ukbb-rg-static', 'ukbb-rg-static.ukbb-rg', proxy, verify_ca=True)
            clusters.append(browser_cluster)
            clusters.append(static_cluster)
        else:
            clusters.append(
                make_cluster(service.name, f'{service.name}.default.svc.cluster.local', proxy, verify_ca=True)
            )

    for namespace, services in internal_services_per_namespace.items():
        for service in services:
            clusters.append(
                make_cluster(
                    f'{namespace}-{service.name}',
                    f'{service.name}.{namespace}.svc.cluster.local',
                    proxy,
                    verify_ca=False,
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
    domain = sys.argv[2]
    with open(sys.argv[3], 'r', encoding='utf-8') as services_file:
        services = [Service(service.rstrip()) for service in services_file.readlines()]

    with open(sys.argv[4], 'w', encoding='utf-8') as cds_file:
        cds_file.write(yaml.dump(create_cds_response(services, {}, proxy)))
    with open(sys.argv[5], 'w', encoding='utf-8') as rds_file:
        rds_file.write(yaml.dump(create_rds_response(services, {}, proxy, domain=domain)))
