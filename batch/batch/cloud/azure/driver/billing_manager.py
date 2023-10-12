import logging
from collections import namedtuple
from typing import Dict, List

from gear import Database
from hailtop.aiocloud import aioazure

from ....driver.billing_manager import (
    CloudBillingManager,
    ProductVersionInfo,
    ProductVersions,
    refresh_product_versions_from_db,
)
from .pricing import AzureVMPrice, fetch_prices

log = logging.getLogger('billing_manager')


AzureVMIdentifier = namedtuple('AzureVMIdentifier', ['machine_type', 'preemptible', 'region'])


class AzureBillingManager(CloudBillingManager):
    @staticmethod
    async def create(
        db: Database,
        pricing_client: aioazure.AzurePricingClient,  # BORROWED
        regions: List[str],
    ):
        product_versions_dict = await refresh_product_versions_from_db(db)
        rm = AzureBillingManager(db, pricing_client, regions, product_versions_dict)
        await rm.refresh_resources()
        await rm.refresh_resources_from_retail_prices()
        return rm

    def __init__(
        self,
        db: Database,
        pricing_client: aioazure.AzurePricingClient,
        regions: List[str],
        product_versions_dict: Dict[str, ProductVersionInfo],
    ):
        self.db = db
        self.product_versions = ProductVersions(product_versions_dict)
        self.resource_rates: Dict[str, float] = {}
        self.pricing_client = pricing_client
        self.regions = regions
        self.vm_price_cache: Dict[AzureVMIdentifier, AzureVMPrice] = {}

    async def get_spot_billing_price(self, machine_type: str, location: str) -> float:
        vm_identifier = AzureVMIdentifier(machine_type=machine_type, preemptible=True, region=location)
        return self.vm_price_cache[vm_identifier].cost_per_hour

    async def refresh_resources_from_retail_prices(self):
        prices = await fetch_prices(self.pricing_client, self.regions)

        await self._refresh_resources_from_retail_prices(prices)

        for price in prices:
            if isinstance(price, AzureVMPrice):
                vm_identifier = AzureVMIdentifier(price.machine_type, price.preemptible, price.region)
                self.vm_price_cache[vm_identifier] = price
