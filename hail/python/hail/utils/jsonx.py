import json

import hail as hl


class JSONEncoder(json.JSONEncoder):
    """JSONEncoder that supports some Hail types."""

    def default(self, o):
        if isinstance(o, (hl.utils.frozendict, hl.utils.Struct)):
            return dict(o)

        if isinstance(o, hl.utils.Interval):
            return {
                "start": o.start,
                "end": o.end,
                "includes_start": o.includes_start,
                "includes_end": o.includes_end,
            }

        if isinstance(o, hl.genetics.Locus):
            return {
                "contig": o.contig,
                "position": o.position,
                "reference_genome": o.reference_genome,
            }

        if isinstance(o, hl.genetics.ReferenceGenome):
            return o.name

        return json.JSONEncoder.default(self, o)
