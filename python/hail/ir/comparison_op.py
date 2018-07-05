from hail.utils.java import escape_id

class ComparisonOp(object):
    def __init__(self, op, typ):
        self.op = op
        self.typ = typ

    def __str__(self):
        ts = self.typ._jtype.parsableString()
        return '({} {} {})'.format(escape_id(self.op), ts, ts)
