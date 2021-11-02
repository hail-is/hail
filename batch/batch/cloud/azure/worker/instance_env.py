import os

from ....worker.instance_env import CloudInstanceEnvironment


class AzureInstanceEnvironment(CloudInstanceEnvironment):
    @staticmethod
    def from_env():
        subscription_id = os.environ['SUBSCRIPTION_ID']
        resource_group = os.environ['RESOURCE_GROUP']
        return AzureInstanceEnvironment(subscription_id, resource_group)

    def __init__(self, subscription_id: str, resource_group: str):
        self.subscription_id = subscription_id
        self.resource_group = resource_group

    def __str__(self):
        return f'subscription_id={self.subscription_id} resource_group={self.resource_group}'
