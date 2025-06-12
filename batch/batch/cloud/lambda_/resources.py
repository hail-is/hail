from typing import Any, Dict, Optional

from ...driver.billing_manager import ProductVersions
from ...resources import (
    ComputeResourceMixin,
    DynamicSizedDiskResourceMixin,
    IPFeeResourceMixin,
    MemoryResourceMixin,
    QuantifiedResource,
    Resource,
    ServiceFeeResourceMixin,
    StaticSizedDiskResourceMixin,
    VMResourceMixin,
)


class LambdaResource(Resource):
    pass


def lambda_resource_from_dict(data: dict) -> LambdaResource:
    raise NotImplementedError
