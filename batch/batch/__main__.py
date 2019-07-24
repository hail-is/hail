from aiohttp import web
from .batch import main_app

web.run_app(main_app, host='0.0.0.0', port=5000)
