import importlib.resources as r

import yaml


def get_deploy_metadata():
    with r.files('hailtop.hailctl').joinpath("deploy.yaml").open('r') as fp:
        deploy_metadata = yaml.safe_load(fp.read())

    return deploy_metadata["dataproc"]
