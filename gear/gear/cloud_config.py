import json
import os
from typing import Dict, Set


class AzureConfig:
    @staticmethod
    def from_global_config(global_config):
        return AzureConfig(
            global_config['azure_subscription_id'],
            global_config['azure_resource_group'],
            global_config['azure_location'],
        )

    def __init__(self, subscription_id: str, resource_group: str, region: str):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.region = region

    def __str__(self):
        data = {
            'subscription_id': self.subscription_id,
            'resource_group': self.resource_group,
            'region': self.region,
        }
        return str(data)


class GCPConfig:
    @staticmethod
    def from_global_config(global_config):
        regions: Set[str] = set(json.loads(global_config['batch_gcp_regions']))
        region = global_config['gcp_region']
        regions.add(region)
        return GCPConfig(
            global_config['gcp_project'],
            region,
            global_config['gcp_zone'],
            regions,
        )

    def __init__(self, project, region, zone, regions):
        self.project = project
        self.region = region
        self.zone = zone
        self.regions = regions

    def __str__(self):
        data = {
            'project': self.project,
            'region': self.region,
            'zone': self.zone,
            'regions': self.regions,
        }
        return str(data)


azure_config = None
gcp_config = None
global_config = None


def get_global_config() -> Dict[str, str]:
    global global_config
    if global_config is None:
        global_config = read_config_secret('/global-config')
    return global_config


def get_azure_config() -> AzureConfig:
    global azure_config
    if azure_config is None:
        azure_config = AzureConfig.from_global_config(get_global_config())
    return azure_config


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
            with open(f'{path}/{field}', encoding='utf-8') as value:
                config[field] = value.read()

    return config
