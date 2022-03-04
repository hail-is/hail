from typing import Dict
import logging
import json
import os
import ssl
from ssl import Purpose

log = logging.getLogger('hailtop.tls')
_server_ssl_context = None
_client_ssl_context = None


class NoSSLConfigFound(Exception):
    pass


def _get_ssl_config() -> Dict[str, str]:
    config_dir = os.environ.get('HAIL_SSL_CONFIG_DIR', '/ssl-config')
    config_file = f'{config_dir}/ssl-config.json'
    if os.path.isfile(config_file):
        log.info(f'ssl config file found at {config_file}')
        with open(config_file, 'r', encoding='utf-8') as f:
            ssl_config = json.load(f)
            for config_name, rel_path in ssl_config.items():
                ssl_config[config_name] = f'{config_dir}/{rel_path}'
        check_ssl_config(ssl_config)
        return ssl_config
    raise NoSSLConfigFound(f'no ssl config found at {config_file}')


def check_ssl_config(ssl_config: Dict[str, str]):
    for key in ('cert', 'key', 'outgoing_trust', 'incoming_trust'):
        assert ssl_config.get(key) is not None, key
    for key in ('cert', 'key', 'outgoing_trust', 'incoming_trust'):
        if not os.path.isfile(ssl_config[key]):
            raise ValueError(f'specified {key}, {ssl_config[key]} does not exist')
    log.info('using tls and verifying client and server certificates')


def internal_server_ssl_context() -> ssl.SSLContext:
    global _server_ssl_context
    if _server_ssl_context is None:
        ssl_config = _get_ssl_config()
        _server_ssl_context = ssl.create_default_context(
            purpose=Purpose.CLIENT_AUTH,
            cafile=ssl_config['incoming_trust'])
        _server_ssl_context.load_cert_chain(ssl_config['cert'],
                                            keyfile=ssl_config['key'],
                                            password=None)
        _server_ssl_context.verify_mode = ssl.CERT_OPTIONAL
        # FIXME: mTLS
        # _server_ssl_context.verify_mode = ssl.CERT_REQURIED
        _server_ssl_context.check_hostname = False  # clients have no hostnames
    return _server_ssl_context


def internal_client_ssl_context() -> ssl.SSLContext:
    global _client_ssl_context
    if _client_ssl_context is None:
        ssl_config = _get_ssl_config()
        _client_ssl_context = ssl.create_default_context(
            purpose=Purpose.SERVER_AUTH,
            cafile=ssl_config['outgoing_trust'])
        # setting cafile in `create_default_context` ignores the system default
        # certificates. We must explicitly request them again with
        # load_default_certs.
        _client_ssl_context.load_default_certs()
        _client_ssl_context.load_cert_chain(ssl_config['cert'],
                                            keyfile=ssl_config['key'],
                                            password=None)
        _client_ssl_context.verify_mode = ssl.CERT_REQUIRED
        _client_ssl_context.check_hostname = True
    return _client_ssl_context


def external_client_ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(purpose=Purpose.SERVER_AUTH)
