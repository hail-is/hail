
class Renderer(object):
    def __init__(self, stop_at_jir):
        self.stop_at_jir = stop_at_jir
        self.count = 0
        self.jirs = {}

    def add_jir(self, jir):
        jir_id = f'm{self.count}'
        self.count = self.count + 1
        self.jirs[jir_id] = jir
        return jir_id

    def __call__(self, x):
        if self.stop_at_jir and hasattr(x, '_jir'):
            jir_id = self.add_jir(self._jir)
            if isinstance(x, MatrixIR):
                return f'(CachedMatrixTable {jir_id})'
            elif isinstance(x, TableIR):
                return f'(CachedTable {jir_id})'
            else:
                assert isinstance(x, IR)
                return f'(CachedValue {jir_id})'
        else:
            return x.render(self)
