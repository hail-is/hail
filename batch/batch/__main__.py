from aiohttp import web

from hailtop.gear import configure_logging, get_deploy_config

from .batch import app

configure_logging()

deploy_config = get_deploy_config()
base_path = deploy_config('batch')
if base_path:
    root_app = web.Application()
    root_app.add_subapp(base_path, app)
else:
    root_app = app

web.run_app(root_app, host='0.0.0.0', port=5000)
