from typing import List, Optional

import hail as hl


class VariantChunk:
    idx: int = 1

    def __init__(self, interval: hl.Interval, groups: Optional[List[str]] = None):
        idx = VariantChunk.idx
        VariantChunk.idx += 1
        self.idx = idx
        self.interval = interval
        self.groups = groups

    @property
    def name(self) -> str:
        return (
            self.interval.start.contig + '_' + str(self.interval.start.position) + '_' + str(self.interval.end.position)
        )

    def to_interval_str(self) -> str:
        return str(self.interval)

    def to_dict(self):
        return {'idx': self.idx, 'interval': self.to_interval_str(), 'groups': self.groups}
