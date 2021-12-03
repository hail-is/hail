from typing import List, Dict
from collections import namedtuple
import logging

from hailtop.aiocloud import aioazure
from gear import Database, transaction

from ....driver.billing_manager import CloudBillingManager, ProductVersions, product_version_to_resource

from .pricing import AzureVMPrice, fetch_prices

log = logging.getLogger('billing_manager')


AzureVMIdentifier = namedtuple('AzureVMIdentifier', ['machine_type', 'preemptible', 'region'])


class AzureBillingManager(CloudBillingManager):
    @staticmethod
    async def create(db: Database,
                     pricing_client: aioazure.AzurePricingClient,  # BORROWED
                     regions: List[str],
                     ):
        rm = AzureBillingManager(db, pricing_client, regions)
        await rm.refresh_resources()
        await rm.refresh_resources_from_retail_prices()
        return rm

    def __init__(self,
                 db: Database,
                 pricing_client: aioazure.AzurePricingClient,
                 regions: List[str],
                 ):
        self.db = db
        self.product_versions = ProductVersions()
        self.resource_rates: Dict[str, float] = {}
        self.pricing_client = pricing_client
        self.regions = regions
        self.vm_price_cache: Dict[AzureVMIdentifier, AzureVMPrice] = {}

    async def get_spot_billing_price(self, machine_type: str, location: str) -> float:
        vm_identifier = AzureVMIdentifier(machine_type=machine_type, preemptible=True, region=location)
        return self.vm_price_cache[vm_identifier].cost_per_hour

    async def refresh_resources_from_retail_prices(self):
        log.info('refreshing resources from retail prices')
        resource_updates = []
        resource_version_updates = []

        for price in await fetch_prices(self.pricing_client, self.regions):
            product = price.product
            latest_product_version = price.version
            latest_resource_rate = price.rate

            resource_name = product_version_to_resource(product, latest_product_version)

            current_product_version = self.product_versions.latest_version(product)
            current_resource_rate = self.resource_rates.get(resource_name)

            if current_resource_rate is None:
                resource_updates.append((resource_name, latest_resource_rate))
            elif current_resource_rate != latest_resource_rate:
                log.exception(f'resource {resource_name} does not have the latest rate in the database for '
                              f'version {current_product_version}: {current_resource_rate} vs {latest_resource_rate}; '
                              f'did the vm price change without a version change?')
                continue

            if price.is_current_price() and (
                    current_product_version is None or current_product_version != latest_product_version):
                resource_version_updates.append((product, latest_product_version))

            if isinstance(price, AzureVMPrice):
                vm_identifier = AzureVMIdentifier(price.machine_type, price.preemptible, price.region)
                self.vm_price_cache[vm_identifier] = price

        @transaction(self.db)
        async def insert_or_update(tx):
            if resource_updates:
                await tx.execute_many('''
INSERT INTO `resources` (resource, rate)
VALUES (%s, %s)
''',
                                      resource_updates)

            if resource_version_updates:
                await tx.execute_many('''
INSERT INTO `latest_product_versions` (product, version)
VALUES (%s, %s)
ON DUPLICATE KEY UPDATE version = VALUES(version)
''',
                                      resource_version_updates)

        await insert_or_update()  # pylint: disable=no-value-for-parameter

        if resource_updates or resource_version_updates:
            await self.refresh_resources()
