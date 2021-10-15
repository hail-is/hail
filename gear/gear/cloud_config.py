from typing import Dict
import os


class GCPConfig:
    project: str
    region: str
    zone: str

    @staticmethod
    def from_global_config(global_config):
        conf = GCPConfig()
        conf.project = global_config['gcp_project']
        conf.region = global_config['gcp_region']
        conf.zone = global_config['gcp_zone']
        return conf


gcp_config = None
global_config = None


def get_global_config() -> Dict[str, str]:
    global global_config
    if global_config is None:
        global_config = read_config_secret('/global-config')
    return global_config


def get_gcp_config() -> GCPConfig:
    global gcp_config
    if gcp_config is None:
        gcp_config = GCPConfig.from_global_config(get_global_config())
    return gcp_config


def read_config_secret(path: str) -> Dict[str, str]:
    config = {}
    for field in os.listdir(path):
        # Kubernetes inserts some hidden files that we don't care about
        if not field.startswith('.'):
            with open(f'{path}/{field}') as value:
                config[field] = value.read()

    return config
