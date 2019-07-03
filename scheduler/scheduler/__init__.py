from aiohttp import web
from .scheduler import app

if __name__ == '__main__':
    web.run_app(app, host='0.0.0.0', port=5000)
