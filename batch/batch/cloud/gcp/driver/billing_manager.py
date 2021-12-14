from typing import Dict
import logging

from gear import Database

from ....driver.billing_manager import CloudBillingManager, ProductVersions, refresh_product_versions_from_db

log = logging.getLogger('resource_manager')


class GCPBillingManager(CloudBillingManager):
    @staticmethod
    async def create(db: Database):
        product_versions_dict = await refresh_product_versions_from_db(db)
        rm = GCPBillingManager(db, product_versions_dict)
        await rm.refresh_resources()
        return rm

    def __init__(self, db: Database, product_versions_dict: dict):
        self.db = db
        self.resource_rates: Dict[str, float] = {}
        self.product_versions = ProductVersions(product_versions_dict)
