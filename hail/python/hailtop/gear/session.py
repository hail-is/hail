import aiohttp_session
import aiohttp_session.cookie_storage

def setup_aiohttp_session(app):
    with open('/session-secret-keys/aiohttp-session-secret-key', 'rb') as f:
        aiohttp_session.setup(app, aiohttp_session.cookie_storage.EncryptedCookieStorage(
            f.read(),
            cookie_name='session',
            # 2592000s = 30d
            max_age=2592000))
