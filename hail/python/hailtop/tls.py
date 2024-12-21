import json
import logging
import os
import ssl
from ssl import Purpose
from typing import Dict, Optional

log = logging.getLogger('hailtop.tls')


class NoSSLConfigFound(Exception):
    pass


def _get_ssl_config(config_dir: Optional[str] = None) -> Dict[str, str]:
    if config_dir is None:
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


def internal_server_ssl_context(config_dir: Optional[str] = None) -> ssl.SSLContext:
    ssl_config = _get_ssl_config(config_dir)
    server_ssl_context = ssl.create_default_context(purpose=Purpose.CLIENT_AUTH, cafile=ssl_config['incoming_trust'])
    server_ssl_context.load_cert_chain(ssl_config['cert'], keyfile=ssl_config['key'], password=None)
    server_ssl_context.verify_mode = ssl.CERT_OPTIONAL
    # FIXME: mTLS
    # server_ssl_context.verify_mode = ssl.CERT_REQURIED
    server_ssl_context.check_hostname = False  # clients have no hostnames
    return server_ssl_context


def internal_client_ssl_context(*, check_hostname: bool = True) -> ssl.SSLContext:
    ssl_config = _get_ssl_config()
    client_ssl_context = ssl.create_default_context(purpose=Purpose.SERVER_AUTH, cafile=ssl_config['outgoing_trust'])
    # setting cafile in `create_default_context` ignores the system default
    # certificates. We must explicitly request them again with
    # load_default_certs.
    client_ssl_context.load_default_certs()
    client_ssl_context.load_cert_chain(ssl_config['cert'], keyfile=ssl_config['key'], password=None)
    client_ssl_context.verify_mode = ssl.CERT_REQUIRED
    client_ssl_context.check_hostname = check_hostname
    return client_ssl_context


def external_client_ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(purpose=Purpose.SERVER_AUTH)
