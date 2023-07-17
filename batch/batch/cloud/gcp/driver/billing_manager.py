import logging
from typing import Dict, List

from gear import Database
from hailtop.aiocloud import aiogoogle

from ....driver.billing_manager import CloudBillingManager, ProductVersions, refresh_product_versions_from_db
from .pricing import fetch_prices

log = logging.getLogger('billing_manager')


class GCPBillingManager(CloudBillingManager):
    @staticmethod
    async def create(db: Database, billing_client: aiogoogle.GoogleBillingClient, regions: List[str]):
        product_versions_dict = await refresh_product_versions_from_db(db)
        bm = GCPBillingManager(db, product_versions_dict, billing_client, regions)
        await bm.refresh_resources()
        await bm.refresh_resources_from_retail_prices()
        return bm

    def __init__(
        self,
        db: Database,
        product_versions_dict: Dict[str, str],
        billing_client: aiogoogle.GoogleBillingClient,
        regions: List[str],
    ):
        self.db = db
        self.resource_rates: Dict[str, float] = {}
        self.product_versions = ProductVersions(product_versions_dict)
        self.billing_client = billing_client
        self.regions = regions
        self.currency_code = 'USD'
        self.spot_percent_increase = None

    async def refresh_resources_from_retail_prices(self):
        prices = [price async for price in fetch_prices(self.billing_client, self.regions, self.currency_code)]
        await self._refresh_resources_from_retail_prices(prices)
