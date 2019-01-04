from hail.utils.java import _escape_id, _unescape_id

class Symbol(object):
    @staticmethod
    def parse(s):
        if s[0] == ':':
            s = s[1:]
            t = _internal_symbol_table.get(s)
            if t:
                return t
            
            [base, count] = s.split('-')
            count = int(count)
            return Generated(base, count)
        else:
            return _unescape_id(s)
    
    @staticmethod
    def from_jsymbol(s):
        return Symbol.parse(s.toString())

class Identifier(Symbol):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return _escape_id(self.name)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Identifier) and self.name == other.name

class Generated(Symbol):
    def __init__(base, count):
        self.base = base
        self.count = count

    def __str__(self):
        return f':{base}-{count}'

    def __hash__(self):
        return hash(self.name) ^ hash(self.count)

    def __eq__(self, other):
        return (isinstance(other, Generated)
                and self.base == other.base
                and self.count == self.count)

class InternalSymbol(Symbol):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f':{self.name}'

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, InternalSymbol) and self.name == other.name

global_sym = InternalSymbol('global')
row_sym = InternalSymbol('row')
col_sym = InternalSymbol('col')
rows_sym = InternalSymbol('rows')
entry_sym = InternalSymbol('entry')

_internal_symbols = [global_sym, row_sym, col_sym, rows_sym, entry_sym]
_internal_symbol_table = {s.name: s for s in _internal_symbols}
