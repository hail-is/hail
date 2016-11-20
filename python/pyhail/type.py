
class Type(object):
    def __init__(self, jtype):
        self.jtype = jtype

    def __repr__(self):
        return self.jtype.toString()

    def __str__(self):
        return self.jtype.toPrettyString(False, False)
