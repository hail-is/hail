from collections.abc import Mapping

from hail.expr import Expression, literal


class Aesthetic(Mapping):
    def __init__(self, properties):
        self.properties = properties

    def __getitem__(self, item):
        return self.properties[item]

    def __len__(self):
        return len(self.properties)

    def __contains__(self, item):
        return item in self.properties

    def __iter__(self):
        return iter(self.properties)

    def __repr__(self):
        return self.properties.__repr__()

    def merge(self, other):
        return Aesthetic({**self.properties, **other.properties})


def aes(**kwargs):
    """Create an aesthetic mapping

    Parameters
    ----------
    kwargs:
        Map aesthetic names to hail expressions based on table's plot.

    Returns
    -------
    :class:`.Aesthetic`
        The aesthetic mapping to be applied.

    """
    hail_field_properties = {}

    for k, v in kwargs.items():
        _v = v
        if not isinstance(v, Expression):
            _v = literal(v)
        hail_field_properties[k] = _v
    return Aesthetic(hail_field_properties)
