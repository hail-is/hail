import json
import hail as hl


class Value(object):
    def __init__(self, typ, value):
        self.typ = typ
        self.value = value

    def __str__(self):
        return '{} "{}"'.format(
            self.typ._jtype.parsableString(),
            hl.utils.java.escape_str(
                json.dumps(self.value, default=self.typ._convert_to_json)))
