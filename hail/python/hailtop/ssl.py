import aiohttp
import logging
import json
import os
import ssl
import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

log = logging.getLogger('hailtop.ssl')
ssl_context = None


def _get_ssl_config():
    config_file = os.environ.get('HAIL_SSL_CONFIG_FILE', '/ssl-config/ssl-config.json')
    if os.path.isfile(config_file):
        log.info(f'ssl config file found at {config_file}')
        with open(config_file, 'r') as f:
            ssl_config = json.loads(f.read())
        check_ssl_config(ssl_config)
        return ssl_config
    else:
        raise ValueError(f'no ssl config found at {config_file}')


def get_ssl_context():
    global ssl_context
    if ssl_context is None:
        ssl_config = _get_ssl_config()
        ssl_context = ssl.create_default_context(cafile=ssl_config['ssl-ca'])
        ssl_context.load_cert_chain(ssl_config['ssl-cert'],
                                    keyfile=ssl_config['ssl-key'],
                                    password=None)
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        ssl_context.check_hostname = True
    return ssl_context


def ssl_client_session(*args, **kwargs):
    return TLSAIOHTTPClientSession(
        get_ssl_context(),
        aiohttp.ClientSession(*args, **kwargs))


def ssl_requests_client_session(*args, **kwargs):
    session = requests.Session(*args, **kwargs)
    ssl_config = _get_ssl_config()
    session.mount('https://', TLSAdapter(ssl_config['ssl-cert'],
                                         ssl_config['ssl-key'],
                                         ssl_config['ssl-ca']))
    return session


def check_ssl_config(ssl_config):
    for key in ('ssl-cert', 'ssl-key', 'ssl-ca'):
        assert ssl_config.get(key) is not None, key
    for key in ('ssl-cert', 'ssl-key', 'ssl-ca'):
        if not os.path.isfile(ssl_config[key]):
            raise ValueError(f'specified {key}, {ssl_config[key]} does not exist')
    log.info(f'using tls and verifying client and server certificates')


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


class TLSAIOHTTPClientSession:
    def __init__(self, ssl_context, session):
        self.ssl_context = ssl_context
        self.session = session

    def __del__(self):
        self.session.__del__()

    def request(self, *args, **kwargs):
        if 'ssl' not in kwargs:
            kwargs['ssl'] = self.ssl_context
        return self.session.request(*args, **kwargs)

    def ws_connect(self, *args, **kwargs):
        if 'ssl' not in kwargs:
            kwargs['ssl'] = self.ssl_context
        return self.ws_connect(*args, **kwargs)

    def get(self, *args, **kwargs):
        if 'ssl' not in kwargs:
            kwargs['ssl'] = self.ssl_context
        return self.session.get(*args, **kwargs)

    def options(self, *args, **kwargs):
        if 'ssl' not in kwargs:
            kwargs['ssl'] = self.ssl_context
        return self.session.options(*args, **kwargs)

    def head(self, *args, **kwargs):
        if 'ssl' not in kwargs:
            kwargs['ssl'] = self.ssl_context
        return self.session.head(*args, **kwargs)

    def post(self, *args, **kwargs):
        if 'ssl' not in kwargs:
            kwargs['ssl'] = self.ssl_context
        return self.session.post(*args, **kwargs)

    def put(self, *args, **kwargs):
        if 'ssl' not in kwargs:
            kwargs['ssl'] = self.ssl_context
        return self.session.put(*args, **kwargs)

    def patch(self, *args, **kwargs):
        if 'ssl' not in kwargs:
            kwargs['ssl'] = self.ssl_context
        return self.session.patch(*args, **kwargs)

    def delete(self, *args, **kwargs):
        if 'ssl' not in kwargs:
            kwargs['ssl'] = self.ssl_context
        return self.session.delete(*args, **kwargs)

    async def close(self):
        return await self.session.close()

    @property
    def closed(self):
        return self.session.closed()

    @property
    def connector(self):
        return self.session.connector()

    @property
    def cookie_jar(self):
        return self.session.cookie_jar()

    @property
    def version(self):
        return self.session.version()

    @property
    def requote_redirect_url(self):
        return self.session.requote_redirect_url()

    @property
    def timeout(self):
        return self.session.timeout()

    @property
    def headers(self):
        return self.session.headers()

    @property
    def skip_auto_headers(self):
        return self.session.skip_auto_headers()

    @property
    def auth(self):
        return self.session.auth()

    @property
    def json_serialize(self):
        return self.session.json_serialize()

    @property
    def connector_owner(self):
        return self.session.connector_owner()

    @property
    def raise_for_status(self):
        return self.session.raise_for_status()

    @property
    def auto_decompress(self):
        return self.session.auto_decompress()

    @property
    def trust_env(self):
        return self.session.trust_env()

    @property
    def trace_configs(self):
        return self.session.trace_configs()

    def detach(self) -> None:
        self.session.detach()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.__aexit__(exc_type, exc_val, exc_tb)
