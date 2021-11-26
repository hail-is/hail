import abc
from typing import Dict, Optional
import logging

from gear import Database, transaction


log = logging.getLogger('resource_manager')


def resource_prefix_version_to_name(prefix: str, version: str) -> str:
    return f'{prefix}/{version}'


class ResourceVersions:
    def __init__(self, data: Optional[Dict[str, str]] = None):
        if data is None:
            data = {}
        self._resource_versions = data

    def latest_version(self, prefix: str) -> Optional[str]:
        return self._resource_versions.get(prefix)

    def latest_resource_name(self, prefix: str) -> Optional[str]:
        version = self.latest_version(prefix)
        assert version, prefix
        return resource_prefix_version_to_name(prefix, version)

    def update(self, data: Dict[str, str]):
        for prefix, old_version in self._resource_versions.items():
            new_version = data.get(prefix)
            if new_version is None:
                log.exception(f'resource {prefix} does not appear in new data; keeping in resource versions')
                continue

            if new_version != old_version:
                log.info(f'updated resource version for {prefix} from {old_version} to {new_version}')

        if self._resource_versions:
            for prefix, new_version in data.items():
                if prefix not in self._resource_versions:
                    log.info(f'added resource {prefix} with version {new_version}')

        self._resource_versions.update(data)

    def to_dict(self) -> Dict[str, str]:
        return self._resource_versions


class CloudBillingManager(abc.ABC):
    db: Database
    resource_versions: ResourceVersions
    resource_rates: Dict[str, float]

    async def refresh_resources(self):
        @transaction(self.db)
        async def refresh(tx):
            self.resource_rates = await refresh_resource_rates_from_db(tx)
            log.info('refreshed resource rates')

            latest_resource_versions = await refresh_resource_versions_from_db(tx)
            self.resource_versions.update(latest_resource_versions)
            log.info('refreshed resource versions')

        await refresh()  # pylint: disable=no-value-for-parameter


async def refresh_resource_versions_from_db(db: Database) -> Dict[str, str]:
    records = db.execute_and_fetchall('SELECT prefix, version FROM latest_resource_versions')
    return {record['prefix']: record['version'] async for record in records}


async def refresh_resource_rates_from_db(db: Database) -> Dict[str, str]:
    records = db.execute_and_fetchall('SELECT resource, rate FROM resources')
    return {record['resource']: record['rate'] async for record in records}
