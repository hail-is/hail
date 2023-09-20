import os

import aiohttp_session
import aiohttp_session.cookie_storage

from hailtop.config import get_deploy_config

from .cloud_config import get_global_config


def setup_aiohttp_session(app):
    deploy_config = get_deploy_config()
    cloud = get_global_config()['cloud']
    cookie_name = cloud + '_' + deploy_config.auth_session_cookie_name()
    with open('/session-secret-key/session-secret-key', 'rb') as f:
        aiohttp_session.setup(
            app,
            aiohttp_session.cookie_storage.EncryptedCookieStorage(
                f.read(),
                cookie_name=cookie_name,
                secure=True,
                httponly=True,
                samesite='Lax',
                domain=os.environ['HAIL_DOMAIN'],
                # 2592000s = 30d
                max_age=2592000,
            ),
        )
