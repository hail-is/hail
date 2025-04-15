import warnings

from ..aiocloud.aiogoogle import *

warnings.warn(
    "importing hailtop.aiogoogle is deprecated, please use hailtop.aiocloud.aiogoogle", DeprecationWarning, stacklevel=2
)
