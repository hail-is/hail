from aiohttp import web

from hailtop.gear import configure_logging, get_deploy_config

from .batch import app

configure_logging()

deploy_config = get_deploy_config()

web.run_app(deploy_config.prefix_application(app, 'batch'), host='0.0.0.0', port=5000)
