import abc
import os
from dataclasses import dataclass
from typing import Optional, Union

from hailtop import pip_version


@dataclass
class CheckpointConfigMixin(abc.ABC):
    """Mixin class for specifying how to generate and use checkpointed files."""

    use_checkpoints: bool = False
    """Use checkpointed files if they exist rather than rerunning the step."""

    checkpoint_output: bool = False
    """Checkpoint any output files at the end of running the step."""


@dataclass
class JobConfigMixin(abc.ABC):
    """Mixin class for specifying job resources for SAIGE jobs."""

    image: Optional[str] = os.environ.get('HAIL_GENETICS_SAIGE_IMAGE', 'us-docker.pkg.dev/hail-vdc/hail/hailgenetics/saige:dev-c6k0x2y4mrkc')  # f'hailgenetics/vep-grch37-85:{pip_version()}'  # FIXME: before releasing make sure Wei makes a tag for the github branch
    """Docker image that has both Hail and SAIGE installed. The version of SAIGE installed is v1.3.3."""

    cpu: Optional[Union[str, int]] = 1
    """The amount of CPU to give the job. This value will be used for the number of threads option "--nThreads"."""

    memory: Optional[str] = 'standard'
    """The amount of memory to give the job. This value can be one of
    "lowmem", "highmem", "standard", or a numeric value. "lowmem" is approximately
    1 Gi per core, "highmem" is approximately 8 Gi per core, and "standard" is approximately
    4 Gi per core."""

    storage: Optional[str] = '10Gi'
    """The amount of storage space to give a job."""

    spot: Optional[bool] = True
    """Whether this job can be run on spot instances."""
