import abc
from typing import List, Optional, Dict

import datetime
import dateutil.parser
import asyncio
import logging

from hailtop.aiocloud import aioazure
from hailtop.utils import flatten, grouped, time_msecs, parse_timestamp_msecs

from ..resource_utils import azure_valid_machine_types, azure_disk_name_to_storage_gib


log = logging.getLogger('pricing')


class AzurePrice(abc.ABC):
    region: str
    effective_start_date: int
    effective_end_date: Optional[int]
    time_updated: int

    def is_current_price(self):
        now = time_msecs()
        return (now >= self.effective_start_date
                and (self.effective_end_date is None or now <= self.effective_end_date))

    @property
    def version(self) -> str:
        return datetime.datetime.utcfromtimestamp(self.effective_start_date / 1000).strftime('%Y-%m-%d')


class AzureVMPrice(AzurePrice):
    def __init__(self,
                 machine_type: str,
                 preemptible: bool,
                 region: str,
                 cost_per_hour: float,
                 effective_start_date: int,
                 effective_end_date: Optional[int] = None,
                 ):
        self.machine_type = machine_type
        self.preemptible = preemptible
        self.region = region
        self.cost_per_hour = cost_per_hour
        self.effective_start_date = effective_start_date
        self.effective_end_date = effective_end_date


class AzureDiskPrice(AzurePrice):
    def __init__(self,
                 disk_name: str,
                 redundancy_type: str,
                 size_gib: int,
                 region: str,
                 cost_per_month: float,
                 effective_start_date: int,
                 effective_end_date: Optional[int] = None,
                 ):
        self.disk_name = disk_name
        self.redundancy_type = redundancy_type
        self.size_gib = size_gib
        self.region = region
        self.cost_per_month = cost_per_month
        self.effective_start_date = effective_start_date
        self.effective_end_date = effective_end_date

    @property
    def disk_product(self):
        return f'{self.disk_product}_{self.redundancy_type}'

    @property
    def disk_type(self):
        return self.disk_product[0]

    @property
    def cost_per_gib_month(self):
        return self.cost_per_month / self.size_gib


async def get_vm_prices_by_region(pricing_client: aioazure.AzurePricingClient,
                                  region: str,
                                  machine_types: List[str],
                                  ) -> List[AzureVMPrice]:
    prices: List[AzureVMPrice] = []
    seen_vm_names: Dict[str, str] = {}

    filter_args = ['(serviceName eq \'Virtual Machines\')',
                   f'(armRegionName eq \'{region}\')']

    machine_type_filter_args = []
    for machine_type in machine_types:
        machine_type_filter_args.append(f'(armSkuName eq \'{machine_type}\')')

    machine_type_filter = '(' + ' or '.join(machine_type_filter_args) + ')'
    filter_args.append(machine_type_filter)

    filter = ' and '.join(filter_args)
    async for data in pricing_client.list_prices(filter=filter):
        if 'Windows' in data['productName'] or data['type'] != 'Consumption' or 'Low Priority' in data['skuName']:
            continue
        assert data['unitOfMeasure'] == '1 Hour' and data['currencyCode'] == 'USD', data

        sku_name = data['skuName']
        machine_type = data['armSkuName']
        preemptible = ('Spot' in sku_name)
        vm_cost_per_hour = float(data['retailPrice'])

        start_date = parse_timestamp_msecs(data['effectiveStartDate'])
        end_date = parse_timestamp_msecs(data.get('effectiveEndDate'))

        if sku_name in seen_vm_names:
            seen_data = seen_vm_names[sku_name]
            log.exception(f'already seen pricing for vm {sku_name}; {seen_data} vs {data}; aborting')
            raise Exception()
        seen_vm_names[sku_name] = data

        vm_price = AzureVMPrice(machine_type, preemptible, region, vm_cost_per_hour, start_date, end_date)
        prices.append(vm_price)

    return prices


async def get_managed_disk_prices_by_region(pricing_client: aioazure.AzurePricingClient, region: str) -> List[AzureDiskPrice]:
    prices: List[AzureDiskPrice] = []
    seen_disk_names: Dict[str, str] = {}

    filter = f'serviceName eq \'Storage\' and armRegionName eq \'{region}\' and endswith(meterName,\'Disks\')'
    async for data in pricing_client.list_prices(filter=filter):
        if data['type'] != 'Consumption' or data['productName'] == 'Premium Page Blob':
            continue

        assert data['unitOfMeasure'] == '1/Month' and data['currencyCode'] == 'USD', data

        sku_name = data['skuName']
        disk_name, redundancy_type = sku_name.split()
        assert redundancy_type in ('LRS', 'ZRS'), redundancy_type
        size_gib = azure_disk_name_to_storage_gib[disk_name]

        if sku_name in seen_disk_names:
            seen_data = seen_disk_names[sku_name]
            log.exception(f'already seen pricing for disk {sku_name}; {seen_data} vs {data}; aborting')
            raise Exception()
        seen_disk_names[sku_name] = data

        start_date = int(dateutil.parser.isoparse(data['effectiveStartDate']).timestamp() * 1000 + 0.5)
        cost_per_month = data['retailPrice']

        disk_price = AzureDiskPrice(disk_name, redundancy_type, size_gib, region, cost_per_month, start_date)
        prices.append(disk_price)

    return prices


async def fetch_prices(pricing_client: aioazure.AzurePricingClient, regions: List[str]) -> List[AzurePrice]:
    # Azure seems to have a limit on how long the OData filter request can be so we split the query into smaller groups
    vm_coros = [get_vm_prices_by_region(pricing_client, region, machine_types)
                for region in regions
                for machine_types in grouped(8, azure_valid_machine_types)]

    disk_coros = [get_managed_disk_prices_by_region(pricing_client, region) for region in regions]

    prices = await asyncio.gather(*vm_coros, *disk_coros)
    return flatten(prices)
