import abc
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from gear import Database, transaction

from .pricing import Price

log = logging.getLogger('billing_manager')


@dataclass
class ProductVersionInfo:
    latest_version: str
    sku: Optional[str]


def product_version_to_resource(product: str, version: str) -> str:
    return f'{product}/{version}'


class ProductVersions:
    def __init__(self, data: Dict[str, ProductVersionInfo]):
        self._product_versions = data

    def latest_version(self, product: str) -> Optional[str]:
        info = self._product_versions.get(product)
        if info:
            return info.latest_version
        return None

    def sku(self, product: str) -> Optional[str]:
        info = self._product_versions.get(product)
        if info:
            return info.sku
        return None

    def resource_name(self, product: str) -> Optional[str]:
        version = self.latest_version(product)
        assert version is not None, (product, str(self._product_versions))
        return product_version_to_resource(product, version)

    def update(self, new_data: Dict[str, ProductVersionInfo]):
        valid_updates: Dict[str, ProductVersionInfo] = {}
        for product, old_product_info in self._product_versions.items():
            old_version = old_product_info.latest_version
            old_sku = old_product_info.sku

            new_product_info = new_data.get(product)

            if new_product_info is None:
                log.error(f'product {product} does not appear in new data; keeping in existing latest_product_versions')
            else:
                new_version = new_product_info.latest_version
                new_sku = new_product_info.sku

                if old_sku is not None and old_sku != new_sku:
                    log.error(
                        f'product {product} does not have the same SKU as is currently in the database ({old_sku}, {new_sku})'
                    )
                elif new_version != old_version:
                    valid_updates[product] = new_product_info
                    log.info(f'updated product version for {product} from {old_version} to {new_version}')

        for product, new_product_info in new_data.items():
            if product not in self._product_versions:
                valid_updates[product] = new_product_info
                log.info(
                    f'added product {product} with version {new_product_info.latest_version} and sku {new_product_info.sku}'
                )

        self._product_versions.update(valid_updates)


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
            latest_sku = price.sku

            current_product_version = self.product_versions.latest_version(product)
            current_sku = self.product_versions.sku(product)

            is_new_product = current_product_version is None

            if is_new_product:
                resource_name = product_version_to_resource(product, latest_product_version)
                resource_updates.append((resource_name, latest_resource_rate))
                product_version_updates.append((product, latest_product_version, latest_sku))
                log.info(
                    f'adding new resource {resource_name} {latest_product_version} with rate change of {latest_resource_rate} and sku {latest_sku}'
                )
            else:
                assert current_product_version
                current_resource_name = product_version_to_resource(product, current_product_version)
                current_resource_rate = self.resource_rates.get(current_resource_name)

                have_latest_version = current_product_version == latest_product_version
                have_latest_rate = current_resource_rate == latest_resource_rate
                have_latest_sku = current_sku is None or current_sku == latest_sku

                if not have_latest_sku:
                    log.error(
                        f'the new sku for product {product} does not match the current sku in the database for '
                        f'version {current_product_version}: {current_sku} vs {latest_sku}; '
                        f'did the sku change without a product change?'
                    )

                if have_latest_version and not have_latest_rate:
                    log.error(
                        f'product {product} does not have the latest rate in the database for '
                        f'version {current_product_version}: {current_resource_rate} vs {latest_resource_rate}; '
                        f'did the vm price change without a version change?'
                    )
                elif not have_latest_version and have_latest_rate:
                    # this prevents having too many resources in the database with redundant information
                    log.info(
                        f'ignoring price update for product {product} -- the latest rate is equal to the previous rate '
                        f'({current_product_version}) => ({latest_product_version}) with rate {latest_resource_rate}'
                    )
                elif not have_latest_version and not have_latest_rate:
                    if price.is_current_price():
                        latest_resource_name = product_version_to_resource(product, latest_product_version)
                        resource_updates.append((latest_resource_name, latest_resource_rate))
                        product_version_updates.append((product, latest_product_version, latest_sku))
                        log.info(
                            f'product {product} changed from {current_product_version} to {latest_product_version} with rate change of '
                            f'({current_resource_rate}) => ({latest_resource_rate}) and sku {latest_sku}'
                        )
                    else:
                        log.error(
                            f'price changed but the price is not current {product} ({current_product_version}) => ({latest_product_version}) ({current_resource_rate}) => ({latest_resource_rate}) '
                            f'{price.effective_start_date} {price.effective_end_date}'
                        )
                else:
                    assert (
                        have_latest_version and have_latest_rate and have_latest_sku
                    ), f'{current_product_version} {latest_product_version} {current_resource_rate} {latest_resource_rate} {latest_sku}'

        @transaction(self.db)
        async def insert_or_update(tx):
            if resource_updates:
                last_resource_id = await tx.execute_and_fetchone(
                    """
SELECT COALESCE(MAX(resource_id), 0) AS last_resource_id
FROM resources
FOR UPDATE
"""
                )
                last_resource_id = last_resource_id['last_resource_id']

                await tx.execute_many(
                    """
INSERT INTO `resources` (resource, rate)
VALUES (%s, %s)
""",
                    resource_updates,
                )

                await tx.execute_update(
                    """
UPDATE resources
SET deduped_resource_id = resource_id
WHERE resource_id > %s AND deduped_resource_id IS NULL
""",
                    (last_resource_id,),
                )

            if product_version_updates:
                await tx.execute_many(
                    """
INSERT INTO `latest_product_versions` (product, version, sku)
VALUES (%s, %s, %s)
ON DUPLICATE KEY UPDATE version = VALUES(version)
""",
                    product_version_updates,
                )

        await insert_or_update()

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

        await refresh()


async def refresh_product_versions_from_db(db: Database) -> Dict[str, ProductVersionInfo]:
    records = db.execute_and_fetchall('SELECT product, version, sku FROM latest_product_versions')
    return {
        record['product']: ProductVersionInfo(latest_version=record['version'], sku=record['sku'])
        async for record in records
    }


async def refresh_resource_rates_from_db(db: Database) -> Dict[str, float]:
    records = db.execute_and_fetchall('SELECT resource, rate FROM resources')
    return {record['resource']: float(record['rate']) async for record in records}
