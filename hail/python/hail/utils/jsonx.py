import json
from typing import Any

import numpy as np
import pandas as pd

from ..genetics.locus import Locus
from ..genetics.reference_genome import ReferenceGenome
from .frozendict import frozendict
from .interval import Interval
from .misc import escape_str
from .struct import Struct


class JSONEncoder(json.JSONEncoder):
    """JSONEncoder that supports some Hail types."""

    def default(self, o: Any):
        if isinstance(o, (frozendict, Struct)):
            return dict(o)
        if isinstance(o, Interval):
            return {
                "start": o.start,
                "end": o.end,
                "includes_start": o.includes_start,
                "includes_end": o.includes_end,
            }
        if isinstance(o, Locus):
            return {
                "contig": o.contig,
                "position": o.position,
                "reference_genome": o.reference_genome,
            }
        if isinstance(o, ReferenceGenome):
            return o.name
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if o is pd.NA:
            return None
        return super().default(o)


def dump_json(obj: Any) -> str:
    return f'"{escape_str(json.dumps(obj, cls=JSONEncoder))}"'
