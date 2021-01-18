import requests
import aiohttp
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

from .tls import internal_client_ssl_context, external_client_ssl_context, _get_ssl_config
from .config.deploy_config import get_deploy_config


def client_session(*args, **kwargs) -> aiohttp.ClientSession:
    location = get_deploy_config().location()
    if location == 'external':
        tls = external_client_ssl_context()
    elif location == 'k8s':
        tls = internal_client_ssl_context()
    else:
        assert location == 'gce'
        # no encryption on the internal gateway
        tls = external_client_ssl_context()

    assert 'connector' not in kwargs
    kwargs['connector'] = aiohttp.TCPConnector(ssl=tls)

    return aiohttp.ClientSession(*args, **kwargs)


class TLSAdapter(HTTPAdapter):
    def __init__(self, ssl_cert, ssl_key, ssl_ca, max_retries, timeout):
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


def blocking_client_session() -> requests.Session:
    session = requests.Session()
    ssl_config = _get_ssl_config()
    session.mount('https://', TLSAdapter(ssl_config['cert'],
                                         ssl_config['key'],
                                         ssl_config['outgoing_trust'],
                                         max_retries=1,
                                         timeout=5))
    return session
