import json
from hail.utils.java import escape_str

class Value(object):
    def __init__(self, typ, value):
        self.typ = typ
        self.value = value

    def __str__(self):
        return '{} "{}"'.format(
            self.typ._jtype.parsableString(),
            escape_str(json.dumps(self.value)))
