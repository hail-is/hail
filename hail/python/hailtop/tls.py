from typing import Dict
import aiohttp
import logging
import json
import os
import ssl
from ssl import Purpose
import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager  # type: ignore

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
            ssl_config = json.loads(f.read())
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


def get_in_cluster_no_hostname_checks_client_ssl_context() -> ssl.SSLContext:
    global no_hostname_checks_client_ssl_context
    if no_hostname_checks_client_ssl_context is None:
        ssl_config = _get_ssl_config()
        no_hostname_checks_client_ssl_context = ssl.create_default_context(
            purpose=Purpose.SERVER_AUTH,
            cafile=ssl_config['outgoing_trust'])
        no_hostname_checks_client_ssl_context.load_cert_chain(
            ssl_config['cert'],
            keyfile=ssl_config['key'],
            password=None)
        no_hostname_checks_client_ssl_context.verify_mode = ssl.CERT_REQUIRED
        no_hostname_checks_client_ssl_context.check_hostname = False
    return no_hostname_checks_client_ssl_context


def get_context_specific_client_ssl_context() -> ssl.SSLContext:
    try:
        return get_in_cluster_client_ssl_context()
    except NoSSLConfigFound:
        log.info('no ssl config file found, using external configuration. This '
                 'context cannot connect directly to services inside the cluster.')
        return ssl.create_default_context(purpose=Purpose.SERVER_AUTH)


def in_cluster_ssl_client_session(*args, **kwargs) -> aiohttp.ClientSession:
    assert 'connector' not in kwargs
    kwargs['connector'] = aiohttp.TCPConnector(ssl=get_in_cluster_client_ssl_context())
    return aiohttp.ClientSession(*args, **kwargs)


def in_cluster_no_hostname_checks_ssl_client_connection(*args, **kwargs) -> aiohttp.ClientSession:
    assert 'connector' not in kwargs
    kwargs['connector'] = aiohttp.TCPConnector(ssl=get_in_cluster_no_hostname_checks_client_ssl_context())
    return aiohttp.ClientSession(*args, **kwargs)


def get_context_specific_ssl_client_session(*args, **kwargs) -> aiohttp.ClientSession:
    assert 'connector' not in kwargs
    kwargs['connector'] = aiohttp.TCPConnector(ssl=get_context_specific_client_ssl_context())
    return aiohttp.ClientSession(*args, **kwargs)


def in_cluster_ssl_requests_client_session() -> requests.Session:
    session = requests.Session()
    ssl_config = _get_ssl_config()
    session.mount('https://', TLSAdapter(ssl_config['cert'],
                                         ssl_config['key'],
                                         ssl_config['outgoing_trust']))
    return session


def check_ssl_config(ssl_config: Dict[str, str]):
    for key in ('cert', 'key', 'outgoing_trust', 'incoming_trust'):
        assert ssl_config.get(key) is not None, key
    for key in ('cert', 'key', 'outgoing_trust', 'incoming_trust'):
        if not os.path.isfile(ssl_config[key]):
            raise ValueError(f'specified {key}, {ssl_config[key]} does not exist')
    log.info('using tls and verifying client and server certificates')


class TLSAdapter(HTTPAdapter):
    def __init__(self, ssl_cert, ssl_key, ssl_ca):
        super(TLSAdapter, self).__init__()
        self.ssl_cert = ssl_cert
        self.ssl_key = ssl_key
        self.ssl_ca = ssl_ca

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = PoolManager(
            key_file=self.ssl_key,
            cert_file=self.ssl_cert,
            ca_certs=self.ssl_ca,
            assert_hostname=True)
