import abc
import logging
from typing import Dict, List, Optional

from gear import Database, transaction

from .pricing import Price

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
        assert version is not None, (product, str(self._product_versions))
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

    async def _refresh_resources_from_retail_prices(self, prices: List[Price]):
        log.info('refreshing resources from retail prices')
        resource_updates = []
        product_version_updates = []

        for price in prices:
            product = price.product
            latest_product_version = price.version
            latest_resource_rate = price.rate

            resource_name = product_version_to_resource(product, latest_product_version)

            current_product_version = self.product_versions.latest_version(product)
            current_resource_rate = self.resource_rates.get(resource_name)

            if current_resource_rate is None:
                resource_updates.append((resource_name, latest_resource_rate))
            elif abs(current_resource_rate - latest_resource_rate) > 1e-20:
                log.error(
                    f'resource {resource_name} does not have the latest rate in the database for '
                    f'version {current_product_version}: {current_resource_rate} vs {latest_resource_rate}; '
                    f'did the vm price change without a version change?'
                )
                continue

            if price.is_current_price() and (
                current_product_version is None or current_product_version != latest_product_version
            ):
                product_version_updates.append((product, latest_product_version))

        @transaction(self.db)
        async def insert_or_update(tx):
            if resource_updates:
                await tx.execute_many(
                    '''
INSERT INTO `resources` (resource, rate)
VALUES (%s, %s)
''',
                    resource_updates,
                )

            if product_version_updates:
                await tx.execute_many(
                    '''
INSERT INTO `latest_product_versions` (product, version)
VALUES (%s, %s)
ON DUPLICATE KEY UPDATE version = VALUES(version)
''',
                    product_version_updates,
                )

        await insert_or_update()  # pylint: disable=no-value-for-parameter

        if resource_updates or product_version_updates:
            await self.refresh_resources()

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


async def refresh_resource_rates_from_db(db: Database) -> Dict[str, float]:
    records = db.execute_and_fetchall('SELECT resource, rate FROM resources')
    return {record['resource']: float(record['rate']) async for record in records}
