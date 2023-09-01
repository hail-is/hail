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
    base_name: Optional[str] = None
    base_attrs: Optional[Dict[str, str]] = None

    def name(self, *, name=None, **kwargs):
        if name is not None:
            return name
        if self.base_name is not None:
            template = jinja2.Environment(loader=jinja2.BaseLoader()).from_string(self.base_name)
            return template.render(**kwargs)
        return None

    def attributes(self, **kwargs):
        attrs = self.base_attrs or {}
        attrs.update(kwargs)
        return attrs
