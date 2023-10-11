from typing import Any, DefaultDict, Dict, FrozenSet, Optional, Tuple, Type, TypeVar, Union

from hailtop.aiocloud.aiogoogle import GCSRequesterPaysConfiguration


FS = TypeVar("FS")
MaybeGCSRequesterPaysConfiguration = Optional[GCSRequesterPaysConfiguration]
GCSRequesterPaysKey = Optional[Union[str, Tuple[str, FrozenSet[str]]]]


class GCSRequesterPaysFSCache(DefaultDict[MaybeGCSRequesterPaysConfiguration, FS]):
    def __init__(self, fs_constructor: Type[FS], default_kwargs: Dict[str, Any] = {}) -> None:
        self._fs_constructor = fs_constructor
        self._default_kwargs = default_kwargs
        super().__init__()

    def __getitem__(self, key: MaybeGCSRequesterPaysConfiguration) -> FS:
        return super().__getitem__((key[0], frozenset(key[1])) if isinstance(key, tuple) else key)

    def __missing__(self, key: GCSRequesterPaysKey) -> FS:
        self[key] = value = self._fs_constructor(**(self._default_kwargs if key is None else {"gcs_kwargs": {"gcs_requester_pays_configuration": (key[0], list(key[1])) if isinstance(key, tuple) else key}}))
        return value


def gcs_requester_pays_fs_cache(fs_constructor: Type[FS], default_kwargs: Dict[str, Any] = {}) -> GCSRequesterPaysFSCache:
    return GCSRequesterPaysFSCache(fs_constructor=fs_constructor, default_kwargs=default_kwargs)
