import aiohttp_session
import aiohttp_session.cookie_storage

def setup_aiohttp_session(app):
    with open('/aiohttp-session-secret-key/aiohttp-session-secret-key', 'rb') as f:
        aiohttp_session.setup(app, aiohttp_session.cookie_storage.EncryptedCookieStorage(
            f.read(),
            cookie_name='session',
            # 30d
            max_age=30 * 24 * 60 * 60))
