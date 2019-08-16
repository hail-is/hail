from aiohttp import web

from hailtop import gear

from .batch import app

gear.configure_logging()
web.run_app(app, host='0.0.0.0', port=5000)
