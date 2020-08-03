import pkg_resources
import yaml


def get_deploy_metadata():
    if not pkg_resources.resource_exists("hailtop.hailctl", "deploy.yaml"):
        raise RuntimeError("package has no 'deploy.yaml' file")

    deploy_metadata = yaml.safe_load(pkg_resources.resource_stream("hailtop.hailctl", "deploy.yaml"))

    return deploy_metadata["dataproc"]
