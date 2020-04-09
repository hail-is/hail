import aiohttp
import logging
import json
import os
import ssl
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.poolmanager import PoolManager

log = logging.getLogger('hailtop.ssl')

# ssl-config schema, inspired by MySQL's parameters
# {
#     "ssl-mode": one-of("DISABLED", "REQUIRED", "VERIFY_CA"),
#     "ssl-ca": filepath,
#     "ssl-cert": filepath,
#     "ssl-key": filepath,
# }
#
# The ssl-cert should refer to a PEM-encoded certificate. The ssl-key should
# refer to a private key. The ssl-ca should refer to PEM-encoded server
# certificates which we trust.


class SSLParameters:
    def __init__(self, disabled, ssl_cert, ssl_key, ssl_ca, check_hostname):
        self.disabled = disabled
        assert disabled or None not in (ssl_cert, ssl_key, ssl_ca, check_hostname)
        self.ssl_cert = ssl_cert
        self.ssl_key = ssl_key
        self.ssl_ca = ssl_ca
        self.check_hostname = check_hostname


def parameters_from_ssl_config(ssl_config, encryption_required=False, verification_required=False):
    ssl_mode = ssl_config.get('ssl-mode') or 'DISABLED'
    if ssl_mode == 'DISABLED':
        if encryption_required or verification_required:
            raise ValueError(f'cleartext connections are not permitted. '
                             f'{json.dumps(ssl_config)}')
        log.warning(f'!!! not using tls !!!')
        return SSLParameters(True, None, None, None, None)

    assert ssl_config.get('ssl-cert') is not None
    assert ssl_config.get('ssl-key') is not None
    assert ssl_config.get('ssl-ca') is not None

    if not os.path.isfile(ssl_config['ssl-cert']):
        raise ValueError(f'specified ssl-cert, {ssl_config["ssl-cert"]} does not exist')
    if not os.path.isfile(ssl_config['ssl-key']):
        raise ValueError(f'specified ssl-key, {ssl_config["ssl-key"]} does not exist')
    if not os.path.isfile(ssl_config['ssl-ca']):
        raise ValueError(f'specified ssl-ca, {ssl_config["ssl-ca"]} does not exist')

    if ssl_mode == 'REQUIRED':
        if verification_required:
            raise ValueError(f'unverified connections are not permitted. '
                             f'{json.dumps(ssl_config)}')
        log.warning(f'using tls and not verifying certificates')
        return SSLParameters(False, ssl_config['ssl-cert'], ssl_config['ssl-key'], ssl_config['ssl-ca'], False)
    if ssl_mode == 'VERIFY_CA':
        log.info(f'using tls and verifying certificates')
        return SSLParameters(False, ssl_config['ssl-cert'], ssl_config['ssl-key'], ssl_config['ssl-ca'], True)
    raise ValueError(f'Only DISABLED, REQURIED, and VERIFY_CA are '
                     f'supported for ssl-mode. ssl-mode was set to '
                     f'{json.dumps(ssl_config)}.')


def ssl_context_from_config(ssl_config, encryption_required=False, verification_required=False):
    params = parameters_from_ssl_config(ssl_config, encryption_required, verification_required)
    if params.disabled:
        return False
    context = ssl.create_default_context(cafile=ssl_config['ssl-ca'])
    context.load_cert_chain(params.ssl_cert, keyfile=params.ssl_key, password=None)
    context.check_hostname = params.check_hostname
    return context


class TLSAdapter(HTTPAdapter):
    def __init__(self, ssl_cert, ssl_key, ssl_ca, check_hostname):
        self.ssl_cert = ssl_cert
        self.ssl_key = ssl_key
        self.ssl_ca = ssl_ca
        self.check_hostname = check_hostname

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = PoolManager(
            key_file=self.ssl_key,
            cert_file=self.ssl_cert,
            ca_certs=self.ssl_ca,
            assert_hostname=self.check_hostname)


def requests_session_from_config(ssl_config,
                                 encryption_required=False,
                                 verification_required=False,
                                 *args,
                                 **kwargs):
    session = requests.Session(*args, **kwargs)
    if ssl_config is None:
        return session
    params = parameters_from_ssl_config(ssl_config, encryption_required, verification_required)
    if params.disabled:
        return session
    session.mount('https://', TLSAdapter(params.ssl_cert,
                                         params.ssl_key,
                                         params.ssl_ca,
                                         params.check_hostname))
    return session


ssl_context = None


def _get_ssl_config():
    config_file = os.environ.get('HAIL_SSL_CONFIG_FILE', '/ssl-config/ssl-config.json')
    if os.path.isfile(config_file):
        log.info(f'ssl config file found at {config_file}')
        with open(config_file, 'r') as f:
            return json.loads(f.read())
    else:
        log.warning(f'no ssl config found at {config_file}')
        return None


def get_ssl_context():
    global ssl_context
    if ssl_context is None:
        ssl_config = _get_ssl_config()
        if ssl_config is None:
            ssl_context = ssl_context_from_config(ssl_config)
        else:
            log.warning(f'no config file found, using default ssl context')
            ssl_context = ssl.create_default_context()
    return ssl_context


def ssl_client_session(*args, **kwargs):
    return TLSAIOHTTPClientSession(
        get_ssl_context(),
        aiohttp.ClientSession(*args, **kwargs))


def ssl_requests_client_session(*args, **kwargs):
    return requests_session_from_config(_get_ssl_config(), *args, **kwargs)

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
