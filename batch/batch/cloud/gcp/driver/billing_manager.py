from typing import Dict
import logging

from gear import Database

from ....driver.billing_manager import CloudBillingManager, ProductVersions

log = logging.getLogger('resource_manager')


class GCPBillingManager(CloudBillingManager):
    @staticmethod
    async def create(db: Database):
        rm = GCPBillingManager(db)
        await rm.refresh_resources()
        return rm

    def __init__(self, db: Database):
        self.db = db
        self.resource_rates: Dict[str, float] = {}
        self.product_versions = ProductVersions()
