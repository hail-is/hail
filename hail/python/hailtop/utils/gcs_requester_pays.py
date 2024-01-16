from typing import Any, Dict, FrozenSet, Generic, Optional, Tuple, Type, TypeVar, Union

from hailtop.aiocloud.aiogoogle import GCSRequesterPaysConfiguration

FS = TypeVar("FS")
MaybeGCSRequesterPaysConfiguration = Optional[GCSRequesterPaysConfiguration]
FrozenKey = Optional[Union[str, Tuple[str, FrozenSet[str]]]]


class GCSRequesterPaysFSCache(Generic[FS]):
    def __init__(self, fs_constructor: Type[FS], default_kwargs: Optional[Dict[str, Any]] = None) -> None:
        self._fs_constructor = fs_constructor
        self._default_kwargs = default_kwargs if default_kwargs is not None else {}
        self._dict: Dict[FrozenKey, FS] = {}

    def __getitem__(self, gcs_requester_pays_configuration: MaybeGCSRequesterPaysConfiguration) -> FS:
        frozen_key = self._freeze_key(gcs_requester_pays_configuration)
        fs = self._dict.get(frozen_key)
        if fs is None:
            if gcs_requester_pays_configuration is None:
                kwargs = self._default_kwargs
            else:
                kwargs = {"gcs_kwargs": {"gcs_requester_pays_configuration": gcs_requester_pays_configuration}}
            fs = self._fs_constructor(**kwargs)
            self._dict[frozen_key] = fs
        return fs

    def _freeze_key(self, gcs_requester_pays_configuration: MaybeGCSRequesterPaysConfiguration) -> FrozenKey:
        if isinstance(gcs_requester_pays_configuration, tuple):
            project, buckets = gcs_requester_pays_configuration
            return (project, frozenset(buckets))
        return gcs_requester_pays_configuration
