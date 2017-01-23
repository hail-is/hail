from context import scala_object

class Type(object):
    """Type of values."""
    
    def __init__(self, jtype):
        self.jtype = jtype

    def __repr__(self):
        return self.jtype.toString()

    def __str__(self):
        return self.jtype.toPrettyString(False, False)

class TInt(Type):

    def __init__(self, jvm):
        super(scala_object(jvm))

