from collections import Counter

from .globals import next_id, batch_id_batch


class Batch:
    def __init__(self, attributes):
        self.attributes = attributes
        self.id = next_id()
        batch_id_batch[self.id] = self
        self.jobs = []

    def delete(self):
        del batch_id_batch[self.id]
        for j in self.jobs:
            assert j.batch_id == self.id
            j.batch_id = None

    def to_json(self):
        state_count = Counter([j._state for j in self.jobs])
        return {
            'id': self.id,
            'jobs': {
                'Created': state_count.get('Created', 0),
                'Complete': state_count.get('Complete', 0),
                'Cancelled': state_count.get('Cancelled', 0)
            },
            'attributes': self.attributes
        }
