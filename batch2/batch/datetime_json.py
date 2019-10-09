import json
from datetime import datetime


class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):  # pylint: disable=E0202
        if isinstance(o, datetime):
            return o.isoformat()

        return json.JSONEncoder.default(self, o)


JSON_ENCODER = DateTimeEncoder()
