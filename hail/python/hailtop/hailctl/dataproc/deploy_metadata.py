import importlib.resources as r

import yaml


def get_deploy_metadata():
    content = r.files('hailtop.hailctl').joinpath('deploy.yaml').read_text('utf-8')
    return yaml.safe_load(content)["dataproc"]
