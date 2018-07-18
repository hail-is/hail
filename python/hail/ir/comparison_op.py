from hail.utils.java import escape_id

class ComparisonOp(object):
    def __init__(self, op, typ=None):
        self.op = op
        self.typ = typ

    def __str__(self):
        if self.typ is None:
            return '({})'.format(escape_id(self.op))
        ts = self.typ._jtype.parsableString()
        return '({} {} {})'.format(escape_id(self.op), ts, ts)
