import asyncio
import logging
from typing import Dict, List, Optional

import dateutil.parser

from hailtop.aiocloud import aioazure
from hailtop.utils import flatten, grouped
from hailtop.utils.rates import rate_gib_month_to_mib_msec, rate_instance_hour_to_fraction_msec

from ....driver.pricing import Price
from ..resource_utils import azure_disk_name_to_storage_gib, azure_valid_machine_types
from ..resources import AzureStaticSizedDiskResource, AzureVMResource

log = logging.getLogger('pricing')


class AzureVMPrice(Price):
    def __init__(
        self,
        machine_type: str,
        preemptible: bool,
        region: str,
        cost_per_hour: float,
        sku: str,
        effective_start_date: int,
        effective_end_date: Optional[int] = None,
    ):
        super().__init__(
            region=region, effective_start_date=effective_start_date, effective_end_date=effective_end_date, sku=sku
        )
        self.machine_type = machine_type
        self.preemptible = preemptible
        self.cost_per_hour = cost_per_hour

    @property
    def product(self):
        return AzureVMResource.product_name(self.machine_type, self.preemptible, self.region)

    @property
    def rate(self):
        return rate_instance_hour_to_fraction_msec(self.cost_per_hour, 1024)


class AzureDiskPrice(Price):
    def __init__(
        self,
        disk_name: str,
        redundancy_type: str,
        size_gib: int,
        region: str,
        cost_per_month: float,
        sku: str,
        effective_start_date: int,
        effective_end_date: Optional[int] = None,
    ):
        super().__init__(
            region=region, effective_start_date=effective_start_date, effective_end_date=effective_end_date, sku=sku
        )
        self.disk_name = disk_name
        self.redundancy_type = redundancy_type
        self.size_gib = size_gib
        self.cost_per_month = cost_per_month

    @property
    def cost_per_gib_month(self):
        return self.cost_per_month / self.size_gib

    @property
    def product(self):
        return AzureStaticSizedDiskResource.product_name(self.disk_name, self.redundancy_type, self.region)

    @property
    def rate(self):
        return rate_gib_month_to_mib_msec(self.cost_per_gib_month)


async def vm_prices_by_region(
    pricing_client: aioazure.AzurePricingClient,
    region: str,
    machine_types: List[str],
) -> List[AzureVMPrice]:
    prices: List[AzureVMPrice] = []
    seen_vm_names: Dict[str, str] = {}

    filter_args = ['(serviceName eq \'Virtual Machines\')', f'(armRegionName eq \'{region}\')']

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

        sku_id = data['skuId']
        sku_name = data['skuName']
        machine_type = data['armSkuName']
        preemptible = 'Spot' in sku_name
        vm_cost_per_hour = float(data['retailPrice'])

        start_date = int(dateutil.parser.isoparse(data['effectiveStartDate']).timestamp() * 1000 + 0.5)
        end_date = data.get('effectiveEndDate')
        if end_date is not None:
            end_date = int(dateutil.parser.isoparse(data['effectiveEndDate']).timestamp() * 1000 + 0.5)

        if sku_name in seen_vm_names:
            seen_data = seen_vm_names[sku_name]
            raise ValueError(f'already seen pricing for vm {sku_name}; {seen_data} vs {data}; aborting')
        seen_vm_names[sku_name] = data

        vm_price = AzureVMPrice(machine_type, preemptible, region, vm_cost_per_hour, sku_id, start_date, end_date)
        prices.append(vm_price)

    return prices


async def managed_disk_prices_by_region(
    pricing_client: aioazure.AzurePricingClient, region: str
) -> List[AzureDiskPrice]:
    prices: List[AzureDiskPrice] = []
    seen_disk_names: Dict[str, str] = {}

    filter = f'serviceName eq \'Storage\' and armRegionName eq \'{region}\' and ( endswith(meterName,\'Disk\') or endswith(meterName,\'Disks\') )'
    async for data in pricing_client.list_prices(filter=filter):
        if data['type'] != 'Consumption' or data['productName'] == 'Premium Page Blob':
            continue

        assert data['unitOfMeasure'] == '1/Month' and data['currencyCode'] == 'USD', data

        sku_id = data['skuId']
        sku_name = data['skuName']
        disk_name, redundancy_type = sku_name.split()
        assert redundancy_type in ('LRS', 'ZRS'), redundancy_type
        size_gib = azure_disk_name_to_storage_gib[disk_name]

        if sku_name in seen_disk_names:
            seen_data = seen_disk_names[sku_name]
            raise ValueError(f'already seen pricing for disk {sku_name}; {seen_data} vs {data}; aborting')
        seen_disk_names[sku_name] = data

        start_date = int(dateutil.parser.isoparse(data['effectiveStartDate']).timestamp() * 1000 + 0.5)
        cost_per_month = data['retailPrice']

        disk_price = AzureDiskPrice(disk_name, redundancy_type, size_gib, region, cost_per_month, sku_id, start_date)
        prices.append(disk_price)

    return prices


async def fetch_prices(pricing_client: aioazure.AzurePricingClient, regions: List[str]) -> List[Price]:
    # Azure seems to have a limit on how long the OData filter request can be so we split the query into smaller groups
    vm_coros = [
        vm_prices_by_region(pricing_client, region, machine_types)
        for region in regions
        for machine_types in grouped(8, azure_valid_machine_types)
    ]

    disk_coros = [managed_disk_prices_by_region(pricing_client, region) for region in regions]

    prices = await asyncio.gather(*vm_coros, *disk_coros)
    return flatten(prices)
