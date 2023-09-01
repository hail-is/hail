from typing import Dict

import hailtop.batch as hb


class AliasedResourceGroup:
    def __init__(self, rg: hb.ResourceGroup, aliases: Dict[str, str]):
        self._rg = rg
        self.aliases = aliases

    def __getitem__(self, item: str) -> hb.ResourceFile:
        alias = self.aliases.get(item)
        name = alias or item
        return self._rg.__getitem__(name)

    def __getattr__(self, item: str) -> hb.ResourceFile:
        alias = self.aliases.get(item)
        name = alias or item
        return self._rg.__getattr__(name)

    def __str__(self):
        return str(self._rg)
