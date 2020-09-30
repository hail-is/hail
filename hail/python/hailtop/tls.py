from typing import Dict, Any, List, Union
import aiohttp
import logging
import json
import os
import ssl
import socket
from ssl import Purpose
import requests
from requests.adapters import HTTPAdapter
import urllib3
from urllib3.poolmanager import PoolManager
from requests.compat import urlparse, urlunparse
from hailtop.config import get_deploy_config

log = logging.getLogger('hailtop.ssl')
server_ssl_context = None
client_ssl_context = None
no_hostname_checks_client_ssl_context = None


class NoSSLConfigFound(Exception):
    pass


def _get_ssl_config() -> Dict[str, str]:
    config_file = os.environ.get('HAIL_SSL_CONFIG_FILE', '/ssl-config/ssl-config.json')
    if os.path.isfile(config_file):
        log.info(f'ssl config file found at {config_file}')
        with open(config_file, 'r') as f:
            ssl_config = json.load(f)
        check_ssl_config(ssl_config)
        return ssl_config
    raise NoSSLConfigFound(f'no ssl config found at {config_file}')


def get_in_cluster_server_ssl_context() -> ssl.SSLContext:
    global server_ssl_context
    if server_ssl_context is None:
        ssl_config = _get_ssl_config()
        server_ssl_context = ssl.create_default_context(
            purpose=Purpose.CLIENT_AUTH,
            cafile=ssl_config['incoming_trust'])
        server_ssl_context.load_cert_chain(ssl_config['cert'],
                                           keyfile=ssl_config['key'],
                                           password=None)
        server_ssl_context.verify_mode = ssl.CERT_OPTIONAL
        # FIXME: mTLS
        # server_ssl_context.verify_mode = ssl.CERT_REQURIED
        server_ssl_context.check_hostname = False  # clients have no hostnames
    return server_ssl_context


def get_in_cluster_client_ssl_context() -> ssl.SSLContext:
    global client_ssl_context
    if client_ssl_context is None:
        ssl_config = _get_ssl_config()
        client_ssl_context = ssl.create_default_context(
            purpose=Purpose.SERVER_AUTH,
            cafile=ssl_config['outgoing_trust'])
        client_ssl_context.load_cert_chain(ssl_config['cert'],
                                           keyfile=ssl_config['key'],
                                           password=None)
        client_ssl_context.verify_mode = ssl.CERT_REQUIRED
        client_ssl_context.check_hostname = True
    return client_ssl_context


def get_context_specific_client_ssl_context() -> ssl.SSLContext:
    try:
        return get_in_cluster_client_ssl_context()
    except NoSSLConfigFound:
        log.info('no ssl config file found, using external configuration. This '
                 'context cannot connect directly to services inside the cluster.')
        return ssl.create_default_context(purpose=Purpose.SERVER_AUTH)


def in_cluster_ssl_client_session(*args, **kwargs) -> aiohttp.ClientSession:
    assert 'connector' not in kwargs
    kwargs['connector'] = aiohttp.TCPConnector(
        ssl=get_in_cluster_client_ssl_context(),
        resolver=HailResolver())
    return aiohttp.ClientSession(*args, **kwargs)


def get_context_specific_ssl_client_session(*args, **kwargs) -> aiohttp.ClientSession:
    assert 'connector' not in kwargs
    kwargs['connector'] = aiohttp.TCPConnector(
        ssl=get_context_specific_client_ssl_context(),
        resolver=HailResolver())
    return aiohttp.ClientSession(*args, **kwargs)


class HailResolver(aiohttp.abc.AbstractResolver):
    """Use Hail in-cluster DNS with fallback."""
    def __init__(self):
        self.dns = aiohttp.AsyncResolver()
        self.deploy_config = get_deploy_config()

    async def resolve(self, host: str, port: int, family: int) -> List[Dict[str, Any]]:
        if family == socket.AF_INET or family == socket.AF_INET6:
            maybe_address_and_port = self.deploy_config.maybe_address(host)
            if maybe_address_and_port is not None:
                address, resolved_port = maybe_address_and_port
                return [{'hostname': host,
                         'host': address,
                         'port': resolved_port,
                         'family': family,
                         'proto': 0,
                         'flags': 0}]
        return self.dns.resolve(host, port, family)

    async def close(self) -> None:
        self.dns.close()


def in_cluster_ssl_requests_client_session() -> requests.Session:
    session = requests.Session()
    ssl_config = _get_ssl_config()
    session.mount('https://', TLSAdapter(ssl_config['cert'],
                                         ssl_config['key'],
                                         ssl_config['outgoing_trust'],
                                         max_retries=1,
                                         timeout=5))
    return session


def check_ssl_config(ssl_config: Dict[str, str]):
    for key in ('cert', 'key', 'outgoing_trust', 'incoming_trust'):
        assert ssl_config.get(key) is not None, key
    for key in ('cert', 'key', 'outgoing_trust', 'incoming_trust'):
        if not os.path.isfile(ssl_config[key]):
            raise ValueError(f'specified {key}, {ssl_config[key]} does not exist')
    log.info('using tls and verifying client and server certificates')


class TLSAdapter(HTTPAdapter):
    def __init__(self, ssl_cert: str, ssl_key: str, ssl_ca: str, max_retries: int, timeout: Union[int, float]):
        super().__init__()
        self.ssl_cert = ssl_cert
        self.ssl_key = ssl_key
        self.ssl_ca = ssl_ca
        self.max_retries = max_retries
        self.timeout = timeout

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            key_file=self.ssl_key,
            cert_file=self.ssl_cert,
            ca_certs=self.ssl_ca,
            assert_hostname=True,
            retries=self.max_retries,
            timeout=self.timeout)
