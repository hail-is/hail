import random
from typing import Optional

from ....driver.instance_collection.machine_type_rollout import MachineTypeRollout
from ..resource_utils import gcp_pool_rollout_machine_type


class GCPN4Rollout(MachineTypeRollout):
    """Rolls a GCP_POOL_N4_FRACTION share of a pool's new autoscaled instances out onto n4."""

    def machine_type(self, worker_type: str, cores: int) -> Optional[str]:
        return gcp_pool_rollout_machine_type(worker_type, cores, random.random())
