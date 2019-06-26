from aiohttp import web
from .scheduler import app

web.run_app(app, host='0.0.0.0', port=5000)
