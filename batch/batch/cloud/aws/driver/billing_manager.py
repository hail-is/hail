import logging
from typing import Dict

from gear import Database

from ....driver.billing_manager import CloudBillingManager, ProductVersions

log = logging.getLogger('billing_manager')


class AWSBillingManager(CloudBillingManager):
    @staticmethod
    async def create(db: Database):
        return AWSBillingManager(db)

    def __init__(self, db: Database):
        self.db = db
        self.resource_rates: Dict[str, float] = {}
        self.product_versions = ProductVersions({})

    async def refresh_resources_from_retail_prices(self):
        pass

    async def close(self):
        pass
