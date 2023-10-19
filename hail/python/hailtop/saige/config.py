import abc
from dataclasses import dataclass
import jinja2
from typing import Dict, Optional, Union


@dataclass
class CheckpointConfigMixin(abc.ABC):
    use_checkpoints: bool = False
    checkpoint_output: bool = False
    overwrite: bool = True


@dataclass
class JobConfigMixin(abc.ABC):
    image: Optional[str] = 'wzhou88/saige:1.3.0'
    cpu: Optional[Union[str, int]] = None
    memory: Optional[str] = None
    storage: Optional[str] = None
    spot: Optional[bool] = None
