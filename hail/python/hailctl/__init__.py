import pkg_resources
import yaml

from . import dataproc


def version() -> str:
    import pkg_resources
    return pkg_resources.resource_string(__name__, 'hail_version').decode().strip()


if not pkg_resources.resource_exists(__name__, "deploy.yaml"):
    raise RuntimeError(f"package has no 'deploy.yaml' file")
_deploy_metadata = yaml.safe_load(
    pkg_resources.resource_stream(__name__, "deploy.yaml"))

__all__ = [
    'dataproc',
    'version',
    '_deploy_metadata'
]
