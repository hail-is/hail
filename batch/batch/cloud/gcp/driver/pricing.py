import logging
from typing import AsyncGenerator, List, Optional

import dateutil.parser

from hailtop.aiocloud import aiogoogle
from hailtop.utils import (
    rate_cpu_hour_to_mcpu_msec,
    rate_gib_hour_to_mib_msec,
    rate_gib_month_to_mib_msec,
    rate_instance_hour_to_fraction_msec,
)

from ....driver.pricing import Price
from ..resources import (
    GCPAcceleratorResource,
    GCPComputeResource,
    GCPLocalSSDStaticSizedDiskResource,
    GCPMemoryResource,
    GCPStaticSizedDiskResource,
)

log = logging.getLogger('pricing')


G2_CPU_SKUS = [
    # G2 Instance Core running in Iowa
    'D18E-3563-415F',
    # G2 Instance Core running in Columbus
    'A312-C1EC-7F1A',
    # Spot Preemptible G2 Instance Core running in Iowa
    '59A6-9C49-BDDB',
    # Spot Preemptible G2 Instance Core running in Columbus
    'B56F-7169-FA2D',
    # Spot Preemptible G2 Instance Core running in Las Vegas
    'E7D9-CF86-B013',
    # G2 Instance Core running in Virginia
    'CF07-A027-C390',
    # G2 Instance Core running in Los Angeles
    'F81A-966F-28CD',
    # Spot Preemptible G2 Instance Core running in Dallas
    '36AA-92D4-D377',
    # Spot Preemptible G2 Instance Core running in Los Angeles
    'BA15-120B-D484',
    # G2 Instance Core running in Salt Lake City
    '321F-5134-70E7',
    # Spot Preemptible G2 Instance Core running in Salt Lake City
    '38F3-C0B3-4807',
    # Spot Preemptible G2 Instance Core running in Virginia
    '55E2-9EB8-778F',
    # G2 Instance Core running in Dallas
    'B911-29A9-DC06',
    # G2 Instance Core running in Las Vegas
    '29FF-9F02-F5CE',
]

G2_RAM_SKUS = [
    # G2 Instance Ram running in Virginia
    'A15F-B56F-97FE',
    # G2 Instance Ram running in Iowa
    'F0A4-E3D3-33BD',
    # Spot Preemptible G2 Instance Ram running in Iowa
    'BF80-650A-78E6',
    # Spot Preemptible G2 Instance Ram running in Los Angeles
    '5076-3195-EAF2',
    # G2 Instance Ram running in Columbus
    'AD5C-91E0-93B5',
    # Spot Preemptible G2 Instance Ram running in Virginia
    'CE56-B589-1E68',
    # Spot Preemptible G2 Instance Ram running in Dallas
    '92B9-A092-A5E8',
    # G2 Instance Ram running in Los Angeles
    'E560-4373-83C8',
    # Spot Preemptible G2 Instance Ram running in Las Vegas
    '22F3-4DB2-9B5C',
    # Spot Preemptible G2 Instance Ram running in Salt Lake City
    'BE68-B175-373C',
    # Spot Preemptible G2 Instance Ram running in Columbus
    '3E11-DA70-84B2',
    # G2 Instance Ram running in Dallas
    'BAC8-4077-EA4B',
    # G2 Instance Ram running in Salt Lake City
    'D79F-7937-D293',
    # G2 Instance Ram running in Las Vegas
    'BB7E-DDF7-ACB4',
]

NVIDIA_L4_GPUS = [
    # Nvidia L4 GPU running in Dallas
    '4E35-7276-6341',
    # Nvidia L4 GPU running in Virginia
    '1D4C-3419-0297',
    # Nvidia L4 GPU attached to Spot Preemptible VMs running in Virginia
    'E548-B46D-71CD',
    # Nvidia L4 GPU attached to Spot Preemptible VMs running in Las Vegas
    '25E8-AAEC-A70B',
    # Nvidia L4 GPU running in Las Vegas
    '6E7A-3CB4-0441',
    # Nvidia L4 GPU attached to Spot Preemptible VMs running in Los Angeles
    'A73C-2F6F-F66F',
    'F20C-EDFB-A03A',
    # Nvidia L4 GPU running in Iowa
    'A88A-5A60-E821',
    # Nvidia L4 GPU running in Columbus
    '1BBE-D828-4747',
    # Nvidia L4 GPU running in Los Angeles
    '455E-02FE-F1F8',
    # Nvidia L4 GPU attached to Spot Preemptible VMs running in Iowa
    'AEC2-3D5C-61BF',
    # Nvidia L4 GPU attached to Spot Preemptible VMs running in Salt Lake City
    '229F-226E-AB3E',
    # Nvidia L4 GPU attached to Spot Preemptible VMs running in Dallas
    'A7DD-B2F1-3095',
    # Nvidia L4 GPU running in Salt Lake City
    '18DB-5EFF-CABE',
]


class GCPComputePrice(Price):
    def __init__(
        self,
        instance_family: str,
        preemptible: bool,
        region: str,
        cost_per_hour: float,
        effective_start_date: int,
        effective_end_date: Optional[int] = None,
    ):
        self.instance_family = instance_family
        self.preemptible = preemptible
        self.region = region
        self.cost_per_hour = cost_per_hour
        self.effective_start_date = effective_start_date
        self.effective_end_date = effective_end_date

    @property
    def product(self):
        return GCPComputeResource.product_name(self.instance_family, self.preemptible, self.region)

    @property
    def rate(self):
        return rate_cpu_hour_to_mcpu_msec(self.cost_per_hour)


class GCPAcceleratorPrice(Price):
    def __init__(
        self,
        accelerator_family: str,
        preemptible: bool,
        region: str,
        cost_per_hour: float,
        effective_start_date: int,
        effective_end_date: Optional[int] = None,
    ):
        self.accelerator_family = accelerator_family
        self.preemptible = preemptible
        self.region = region
        self.cost_per_hour = cost_per_hour
        self.effective_start_date = effective_start_date
        self.effective_end_date = effective_end_date

    @property
    def product(self):
        return GCPAcceleratorResource.product_name(self.accelerator_family, self.preemptible, self.region)

    @property
    def rate(self):
        # return rate_cpu_hour_to_mcpu_msec(self.cost_per_hour)
        return rate_instance_hour_to_fraction_msec(self.cost_per_hour, 1024)


class GCPMemoryPrice(Price):
    def __init__(
        self,
        instance_family: str,
        preemptible: bool,
        region: str,
        cost_per_hour: float,
        effective_start_date: int,
        effective_end_date: Optional[int] = None,
    ):
        self.instance_family = instance_family
        self.preemptible = preemptible
        self.region = region
        self.cost_per_hour = cost_per_hour
        self.effective_start_date = effective_start_date
        self.effective_end_date = effective_end_date

    @property
    def product(self):
        return GCPMemoryResource.product_name(self.instance_family, self.preemptible, self.region)

    @property
    def rate(self):
        return rate_gib_hour_to_mib_msec(self.cost_per_hour)


class GCPLocalSSDDiskPrice(Price):
    def __init__(
        self,
        preemptible: bool,
        region: str,
        cost_per_month: float,
        effective_start_date: int,
        effective_end_date: Optional[int] = None,
    ):
        self.preemptible = preemptible
        self.region = region
        self.cost_per_month = cost_per_month
        self.effective_start_date = effective_start_date
        self.effective_end_date = effective_end_date

    @property
    def cost_per_gib_month(self):
        return self.cost_per_month

    @property
    def product(self):
        return GCPLocalSSDStaticSizedDiskResource.product_name(self.preemptible, self.region)

    @property
    def rate(self):
        return rate_gib_month_to_mib_msec(self.cost_per_gib_month)


class GCPDiskPrice(Price):
    def __init__(
        self,
        disk_type: str,
        region: str,
        cost_per_month: float,
        effective_start_date: int,
        effective_end_date: Optional[int] = None,
    ):
        self.disk_type = disk_type
        self.region = region
        self.cost_per_month = cost_per_month
        self.effective_start_date = effective_start_date
        self.effective_end_date = effective_end_date

    @property
    def cost_per_gib_month(self):
        return self.cost_per_month

    @property
    def product(self):
        return GCPStaticSizedDiskResource.product_name(self.disk_type, self.region)

    @property
    def rate(self):
        return rate_gib_month_to_mib_msec(self.cost_per_gib_month)


def parse_effective_start_date(sku: dict) -> int:
    return int(dateutil.parser.isoparse(sku['pricingInfo'][0]['effectiveTime']).timestamp() * 1000 + 0.5)


def pricing_expression_to_price_per_unit(pricing_expression: dict) -> float:
    # https://cloud.google.com/billing/docs/reference/rest/v1/services.skus/list
    # https://cloud.google.com/billing/docs/reference/rest/v1/services.skus/list#Money
    assert len(pricing_expression['tieredRates']) == 1, str(pricing_expression)
    unit_price = pricing_expression['tieredRates'][0]['unitPrice']
    assert unit_price['currencyCode'] == 'USD', str(pricing_expression)
    return int(unit_price['units']) + (unit_price['nanos'] / 1000 / 1000 / 1000)


def instance_family_from_sku(sku: dict) -> Optional[str]:
    category = sku['category']
    if category['resourceGroup'] == 'N1Standard':
        return 'n1'
    if sku['description'].startswith("G2 Instance") or sku['description'].startswith("Spot Preemptible G2 Instance"):
        return 'g2'
    return None


def accelerator_from_sku(sku) -> Optional[str]:
    description = sku['description']
    if description.startswith("Nvidia L4 GPU"):
        return 'l4'
    return None


def preemptible_from_sku(sku: dict) -> Optional[bool]:
    category = sku['category']
    usageType = category['usageType']

    if usageType == 'OnDemand':
        return False
    if usageType == 'Preemptible':
        return True
    return None


def process_compute_sku(sku: dict, regions: List[str]) -> List[GCPComputePrice]:
    category = sku['category']
    assert category['resourceFamily'] == 'Compute', sku
    assert 'Core' in sku['description']

    instance_family = instance_family_from_sku(sku)
    preemptible = preemptible_from_sku(sku)

    if instance_family is None or preemptible is None:
        return []

    effective_start_date = parse_effective_start_date(sku)

    # https://cloud.google.com/billing/docs/reference/rest/v1/services.skus/list#sku
    pricing_info = sku['pricingInfo'][-1]  # A timeline of pricing info for this SKU in chronological order.
    pricing_expression = pricing_info['pricingExpression']
    assert (
        pricing_expression['usageUnitDescription'] == 'hour'
        and pricing_expression['baseUnit'] == 's'
        and pricing_expression['baseUnitConversionFactor'] == 3600
    ), sku
    cost_per_hour = pricing_expression_to_price_per_unit(pricing_expression)

    compute_prices = []
    service_regions = sku['serviceRegions']
    for service_region in service_regions:
        if service_region in regions:
            if instance_family == 'g2':
                log.info(f'SKUID {sku["skuId"]} with description {sku["description"]}')
                assert sku["skuId"] in G2_CPU_SKUS, sku
            compute_prices.append(
                GCPComputePrice(instance_family, preemptible, service_region, cost_per_hour, effective_start_date)
            )
    return compute_prices


def process_accelerator_sku(sku: dict, regions: List[str]) -> List[GCPAcceleratorPrice]:
    category = sku['category']
    assert category['resourceFamily'] == 'Compute', sku
    assert 'GPU' in category['resourceGroup']

    accelerator_family = accelerator_from_sku(sku)
    preemptible = preemptible_from_sku(sku)

    if accelerator_family is None or preemptible is None:
        return []

    effective_start_date = parse_effective_start_date(sku)

    # https://cloud.google.com/billing/docs/reference/rest/v1/services.skus/list#sku
    pricing_info = sku['pricingInfo'][-1]  # A timeline of pricing info for this SKU in chronological order.
    pricing_expression = pricing_info['pricingExpression']
    assert (
        pricing_expression['usageUnitDescription'] == 'hour'
        and pricing_expression['baseUnit'] == 's'
        and pricing_expression['baseUnitConversionFactor'] == 3600
    ), sku
    cost_per_hour = pricing_expression_to_price_per_unit(pricing_expression)

    compute_prices = []
    service_regions = sku['serviceRegions']
    for service_region in service_regions:
        if service_region in regions:
            if accelerator_family == 'l4':
                log.info(f'SKUID {sku["skuId"]} with description {sku["description"]}')
                assert sku["skuId"] in NVIDIA_L4_GPUS, sku
            compute_prices.append(
                GCPAcceleratorPrice(
                    accelerator_family, preemptible, service_region, cost_per_hour, effective_start_date
                )
            )
    return compute_prices


def process_memory_sku(sku: dict, regions: List[str]) -> List[GCPMemoryPrice]:
    category = sku['category']
    assert category['resourceFamily'] == 'Compute', sku
    assert 'Ram' in sku['description']

    instance_family = instance_family_from_sku(sku)
    preemptible = preemptible_from_sku(sku)

    if instance_family is None or preemptible is None:
        return []

    effective_start_date = parse_effective_start_date(sku)

    # https://cloud.google.com/billing/docs/reference/rest/v1/services.skus/list#sku
    pricing_info = sku['pricingInfo'][-1]  # A timeline of pricing info for this SKU in chronological order.
    pricing_expression = pricing_info['pricingExpression']
    assert pricing_expression['usageUnit'] == 'GiBy.h', sku
    cost_per_hour = pricing_expression_to_price_per_unit(pricing_expression)

    memory_prices = []
    service_regions = sku['serviceRegions']
    for service_region in service_regions:
        if service_region in regions:
            if instance_family == 'g2':
                log.info(f'SKUID {sku["skuId"]} with description {sku["description"]}')
                assert sku["skuId"] in G2_RAM_SKUS, sku
            memory_prices.append(
                GCPMemoryPrice(instance_family, preemptible, service_region, cost_per_hour, effective_start_date)
            )
    return memory_prices


def process_local_ssd_sku(sku: dict, regions: List[str]) -> List[GCPLocalSSDDiskPrice]:
    category = sku['category']
    assert category['resourceFamily'] == 'Storage', sku
    assert category['resourceGroup'] == 'LocalSSD', sku

    preemptible = preemptible_from_sku(sku)
    if preemptible is None:
        return []

    effective_start_date = parse_effective_start_date(sku)

    # https://cloud.google.com/billing/docs/reference/rest/v1/services.skus/list#sku
    pricing_info = sku['pricingInfo'][-1]  # A timeline of pricing info for this SKU in chronological order.
    pricing_expression = pricing_info['pricingExpression']
    assert pricing_expression['usageUnit'] == 'GiBy.mo', sku
    cost_per_month = pricing_expression_to_price_per_unit(pricing_expression)

    local_ssd_prices = []
    service_regions = sku['serviceRegions']
    for service_region in service_regions:
        if service_region in regions:
            local_ssd_prices.append(
                GCPLocalSSDDiskPrice(preemptible, service_region, cost_per_month, effective_start_date)
            )
    return local_ssd_prices


def process_disk_sku(sku: dict, regions: List[str]) -> List[GCPDiskPrice]:
    category = sku['category']
    assert category['resourceFamily'] == 'Storage', sku
    assert category['resourceGroup'] == 'SSD', sku
    assert 'SSD backed PD Capacity' in sku['description'], sku
    assert 'Regional' not in sku['description'], sku

    effective_start_date = parse_effective_start_date(sku)

    # https://cloud.google.com/billing/docs/reference/rest/v1/services.skus/list#sku
    pricing_info = sku['pricingInfo'][-1]  # A timeline of pricing info for this SKU in chronological order.
    pricing_expression = pricing_info['pricingExpression']
    assert pricing_expression['usageUnit'] == 'GiBy.mo', sku
    cost_per_month = pricing_expression_to_price_per_unit(pricing_expression)

    disk_prices = []
    service_regions = sku['serviceRegions']
    for service_region in service_regions:
        if service_region in regions:
            disk_prices.append(GCPDiskPrice('pd-ssd', service_region, cost_per_month, effective_start_date))
    return disk_prices


# Not called but this logic is how I got the gpu SKUs
async def get_gpu_skus(billing_client: aiogoogle.GoogleBillingClient, currency_code: str) -> List[set]:
    params = {'currencyCode': currency_code}
    g2_ram_total = set()
    g2_cpu_total = set()
    gpu_total = set()
    async for sku in billing_client.list_skus('/6F81-5844-456A/skus', params=params):
        category = sku['category']
        if (
            sku['description'].startswith("G2 Instance")
            or sku['description'].startswith("Spot Preemptible G2 Instance")
        ) and "us-" in sku['serviceRegions'][0]:
            if 'Core' in sku['description']:
                g2_cpu_total.add(sku["skuId"])
            elif 'Ram' in sku['description']:
                g2_ram_total.add(sku["skuId"])
        if (
            sku['description'].startswith("Nvidia L4 GPU")
            and 'GPU' in category['resourceGroup']
            and "us-" in sku['serviceRegions'][0]
        ):
            gpu_total.add(sku["skuId"])
    log.info(f'g2_cpu_total {g2_cpu_total}')
    log.info(f'g2_ram_total {g2_ram_total}')
    log.info(f'gpu_total {gpu_total}')
    return [g2_cpu_total, g2_ram_total, gpu_total]


async def fetch_prices(
    billing_client: aiogoogle.GoogleBillingClient, regions: List[str], currency_code: str
) -> AsyncGenerator[Price, None]:
    params = {'currencyCode': currency_code}
    log.info("before-gpu-func")
    await get_gpu_skus(billing_client, currency_code)
    log.info("after-gpu-func")
    async for sku in billing_client.list_skus('/6F81-5844-456A/skus', params=params):
        category = sku['category']
        if 'GPU' in category['resourceGroup']:
            for accelerator_price in process_accelerator_sku(sku, regions):
                yield accelerator_price
        if category['resourceFamily'] == 'Compute':
            if 'Core' in sku['description']:
                for compute_price in process_compute_sku(sku, regions):
                    yield compute_price
            elif 'Ram' in sku['description']:
                for memory_price in process_memory_sku(sku, regions):
                    yield memory_price
        elif category['resourceFamily'] == 'Storage':
            if category['resourceGroup'] == 'LocalSSD':
                for local_ssd_price in process_local_ssd_sku(sku, regions):
                    yield local_ssd_price
            elif (
                category['resourceGroup'] == 'SSD'
                and 'SSD backed PD Capacity' in sku['description']
                and 'Regional' not in sku['description']
            ):
                for disk_price in process_disk_sku(sku, regions):
                    yield disk_price
