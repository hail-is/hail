from hail.utils.java import escape_id

class AggSignature(object):
    def __init__(self, op, ctor_arg_types, initop_arg_types, seqop_arg_types):
        self.op = op
        self.ctor_arg_types = ctor_arg_types
        self.initop_arg_types = initop_arg_types
        self.seqop_arg_types = seqop_arg_types

    def __str__(self):
        return '({} ({}) {} ({}))'.format(
            self.op,
            ' '.join([x._jtype.parsableString() for x in self.ctor_arg_types]),
            ('(' + ' '.join([x._jtype.parsableString() for x in self.initop_arg_types]) + ')'
             if self.initop_arg_types else 'None'),
            ' '.join([x._jtype.parsableString() for x in self.seqop_arg_types]))

    def __eq__(self, other):
        return isinstance(other, AggSignature) and \
               self.op == other.op and \
               self.ctor_arg_types == other.ctor_arg_types and \
               self.initop_arg_types == other.initop_arg_types and \
               self.seqop_arg_types == other.seqop_arg_types
