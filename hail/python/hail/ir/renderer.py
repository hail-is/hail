from hail import ir

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
            jir_id = self.add_jir(x._jir)
            if isinstance(x, ir.MatrixIR):
                return f'(JavaMatrix {jir_id})'
            elif isinstance(x, ir.TableIR):
                return f'(JavaTable {jir_id})'
            else:
                assert isinstance(x, ir.IR)
                return f'(JavaIR {jir_id})'
        else:
            return x.render(self)
