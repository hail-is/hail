import abc
import logging
from typing import Dict, Optional

from gear import Database, transaction

log = logging.getLogger('billing_manager')


def product_version_to_resource(product: str, version: str) -> str:
    return f'{product}/{version}'


class ProductVersions:
    def __init__(self, data: Dict[str, str]):
        self._product_versions = data

    def latest_version(self, product: str) -> Optional[str]:
        return self._product_versions.get(product)

    def resource_name(self, product: str) -> Optional[str]:
        version = self.latest_version(product)
        assert version is not None
        return product_version_to_resource(product, version)

    def update(self, data: Dict[str, str]):
        for product, old_version in self._product_versions.items():
            new_version = data.get(product)
            if new_version is None:
                log.error(f'product {product} does not appear in new data; keeping in existing latest_product_versions')
                continue

            if new_version != old_version:
                log.info(f'updated product version for {product} from {old_version} to {new_version}')

        for product, new_version in data.items():
            if product not in self._product_versions:
                log.info(f'added product {product} with version {new_version}')

        self._product_versions.update(data)

    def to_dict(self) -> Dict[str, str]:
        return self._product_versions


class CloudBillingManager(abc.ABC):
    db: Database
    product_versions: ProductVersions
    resource_rates: Dict[str, float]

    async def refresh_resources(self):
        @transaction(self.db)
        async def refresh(tx):
            self.resource_rates = await refresh_resource_rates_from_db(tx)
            log.info('refreshed resource rates')

            latest_product_versions = await refresh_product_versions_from_db(tx)
            self.product_versions.update(latest_product_versions)
            log.info('refreshed product versions')

        await refresh()  # pylint: disable=no-value-for-parameter


async def refresh_product_versions_from_db(db: Database) -> Dict[str, str]:
    records = db.execute_and_fetchall('SELECT product, version FROM latest_product_versions')
    return {record['product']: record['version'] async for record in records}


async def refresh_resource_rates_from_db(db: Database) -> Dict[str, str]:
    records = db.execute_and_fetchall('SELECT resource, rate FROM resources')
    return {record['resource']: record['rate'] async for record in records}
