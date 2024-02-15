import abc
from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class CheckpointConfigMixin(abc.ABC):
    use_checkpoints: bool = False
    checkpoint_output: bool = False
    overwrite: bool = True


@dataclass
class JobConfigMixin(abc.ABC):
    image: Optional[str] = 'us-docker.pkg.dev/hail-vdc/hail/hailgenetics/saige:dev-hajvnjte0fv5'  # 'wzhou88/saige:1.3.0'  # FIXME: before releasing make sure Wei makes a tag for the github branch
    cpu: Optional[Union[str, int]] = 1
    memory: Optional[str] = 'standard'
    storage: Optional[str] = '10Gi'
    spot: Optional[bool] = True
